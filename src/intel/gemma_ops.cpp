// Gemma formulas follow src/gemma4/{gemma4.cpp,kernels.cu} (MIT; NOTICE).
#include "strata/intel/gemma_ops.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include <oneapi/mkl/blas.hpp>
#endif
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::intel {
namespace {
void ordered(const sycl::queue& q) {
    if(!q.has_property<sycl::property::queue::in_order>())throw std::invalid_argument("Gemma operations require an in-order queue");
}
void cache_shape(GemmaKv c) {
    if(!c.key||!c.value||!c.key_scale||!c.value_scale||c.capacity<1||c.heads<1||c.dim<32||c.dim>1024||c.dim%32||c.window<0)
        throw std::invalid_argument("Invalid Gemma INT8 cache");
}
float gelu(float x) { return .5f*x*(1.f+sycl::tanh(.79788456080286535588f*x*(1.f+.044715f*x*x))); }
}
sycl::event gemma_head_norm_rope(sycl::queue& q,const float* x,const float* w,float* y,
    int heads,int dim,int rotation,int position,int tokens,float epsilon,float base,const float* freq) {
    ordered(q);
    if(!x||!y||heads<1||dim<1||dim>1024||rotation<0||rotation>dim||rotation%2||position<0||tokens<1||!(epsilon>0)||!(base>0))
        throw std::invalid_argument("Invalid Gemma head normalization/rotation");
    const float theta_scale=std::pow(base,-2.f/float(std::max(rotation,1)));
    return q.submit([&](sycl::handler& h){
        sycl::local_accessor<float,1> normalized(size_t(dim),h);
        h.parallel_for(sycl::nd_range<1>(size_t(tokens)*heads*256,256),[=](sycl::nd_item<1> it){
            const size_t row=it.get_group_linear_id(),lane=it.get_local_linear_id();
            float squares=0;
            for(size_t col=lane;col<size_t(dim);col+=256){const float v=x[row*dim+col];squares+=v*v;}
            const float scale=sycl::rsqrt(sycl::reduce_over_group(it.get_group(),squares,sycl::plus<float>())/float(dim)+epsilon);
            for(size_t col=lane;col<size_t(dim);col+=256)normalized[col]=x[row*dim+col]*scale*(w?w[col]:1.f);
            sycl::group_barrier(it.get_group());
            for(size_t col=lane;col<size_t(dim);col+=256){
                float result=normalized[col];const int half=rotation/2;
                if(col<size_t(rotation)){
                    const size_t j=col<size_t(half)?col:col-half;
                    const float angle=float(position+int(row/heads))*sycl::pow(theta_scale,float(j))/(freq?freq[j]:1.f);
                    const float a=normalized[j],b=normalized[j+half],s=sycl::sin(angle),c=sycl::cos(angle);
                    result=col<size_t(half)?a*c-b*s:a*s+b*c;
                }
                y[row*dim+col]=result;
            }
        });
    });
}
sycl::event gemma_store_q8(sycl::queue& q,const float* key,const float* value,GemmaKv c,int position,int tokens,int32_t* error) {
    ordered(q);cache_shape(c);
    if(!key||!value||!error||position<0||tokens<1||tokens>c.capacity||(!c.window&&position+tokens>c.capacity))
        throw std::invalid_argument("Invalid Gemma cache store extent");
    const size_t blocks=size_t(tokens)*c.heads*(c.dim/32);
    return q.parallel_for(sycl::range<1>(blocks),[=](sycl::id<1> id){
        const size_t block=id[0],col=(block%(c.dim/32))*32,head=(block/(c.dim/32))%c.heads,token=block/(c.dim/32)/c.heads;
        const size_t input=(token*c.heads+head)*c.dim+col;
        const size_t output=(head*c.capacity+(position+int(token))%c.capacity)*c.dim+col;
        for(int which=0;which<2;++which){
            const float* x=(which?value:key)+input;float maximum=0;bool finite=true;
            for(int j=0;j<32;++j){finite=finite&&sycl::isfinite(x[j]);maximum=sycl::fmax(maximum,sycl::fabs(x[j]));}
            const float d=maximum/127.f;const sycl::half stored=sycl::half(d);
            if(!finite||!sycl::isfinite(float(stored))){
                sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device> flag(*error);flag.fetch_or(finite?2:1);
            }
            auto* codes=which?c.value:c.key;auto* scales=which?c.value_scale:c.key_scale;
            scales[output/32]=finite&&sycl::isfinite(float(stored))?stored:sycl::half(0);
            // Match Gemma's CUDA quantizer: codes use unrounded FP32 d,
            // while persistent scales are rounded to FP16.
            const float inverse=d>0?1.f/d:0.f;
            for(int j=0;j<32;++j)codes[output+j]=finite?int8_t(sycl::clamp(sycl::rint(x[j]*inverse),-127.f,127.f)):0;
        }
    });
}
sycl::event gemma_geglu(sycl::queue& q,const float* gate,const float* up,float* output,size_t count) {
    ordered(q);if(!count)throw std::invalid_argument("Empty Gemma GELU");
    return q.parallel_for(sycl::range<1>(count),[=](sycl::id<1> i){output[i]=gelu(gate[i])*up[i];});
}
sycl::event gemma_geglu_packed(sycl::queue& q,const float* gu,float* output,int width,int rows) {
    ordered(q);if(width<1||rows<1)throw std::invalid_argument("Invalid packed Gemma GELU");
    return q.parallel_for(sycl::range<1>(size_t(width)*rows),[=](sycl::id<1> i){
        const size_t row=i[0]/width,col=i[0]%width;output[i]=gelu(gu[row*2*width+col])*gu[row*2*width+width+col];
    });
}
sycl::event gemma_route(sycl::queue& q,const float* logits,const float* escale,int32_t* ids,float* weights,
    int experts,int picks,int tokens,int width) {
    ordered(q);if(experts<1||experts>512||picks<1||picks>16||picks>experts||tokens<1||width<1)throw std::invalid_argument("Invalid Gemma routing geometry");
    const float scale=1.f/std::sqrt(float(width));
    return q.parallel_for(sycl::nd_range<1>(size_t(tokens)*128,128),[=](sycl::nd_item<1> it){
        const size_t token=it.get_group_linear_id(),lane=it.get_local_linear_id();const float* row=logits+token*experts;
        float mx=-INFINITY;for(int e=int(lane);e<experts;e+=128)mx=sycl::fmax(mx,row[e]*scale);
        const float maximum=sycl::reduce_over_group(it.get_group(),mx,sycl::maximum<float>());
        float subtotal=0;for(int e=int(lane);e<experts;e+=128)subtotal+=sycl::exp(row[e]*scale-maximum);
        const float total=sycl::reduce_over_group(it.get_group(),subtotal,sycl::plus<float>());
        int selected[16];
        for(int pick=0;pick<picks;++pick){
            float best=-INFINITY;int best_id=experts;
            for(int e=int(lane);e<experts;e+=128){
                bool used=false;for(int k=0;k<pick;++k)used=used||selected[k]==e;
                if(!used&&(row[e]>best||(row[e]==best&&e<best_id))){best=row[e];best_id=e;}
            }
            const float picked_max=sycl::reduce_over_group(it.get_group(),best,sycl::maximum<float>());
            selected[pick]=sycl::reduce_over_group(it.get_group(),best==picked_max?best_id:experts,sycl::minimum<int>());
        }
        if(lane==0){
            float sum=0;for(int k=0;k<picks;++k){const float v=sycl::exp(row[selected[k]]*scale-maximum)/total;weights[token*picks+k]=v;sum+=v;}
            for(int k=0;k<picks;++k){ids[token*picks+k]=selected[k];weights[token*picks+k]=weights[token*picks+k]/sycl::fmax(sum,6.103515625e-5f)*(escale?escale[selected[k]]:1.f);}
        }
    });
}

struct GemmaAttention::Impl {
    sycl::queue& q;int context,tokens,heads,dim;size_t cache_elements;bool mkl;
    float *scores=nullptr,*keys=nullptr,*values=nullptr,*temporary=nullptr;
    std::vector<void*> allocations;
    template<class T>T* allocate(size_t count){auto* p=sycl::malloc_device<T>(count,q);if(!p)throw std::runtime_error("Gemma attention allocation failed");allocations.push_back(p);return p;}
    Impl(sycl::queue& queue,int c,int t,int h,int d,size_t elements,bool use_mkl):q(queue),context(c),tokens(t),heads(h),dim(d),cache_elements(elements),mkl(use_mkl){
        ordered(q);if(c<1||t<1||t>1024||h<1||d<1||elements<1)throw std::invalid_argument("Invalid Gemma attention scratch");
#ifndef TINYSTRATA_INTEL_MKL
        if(mkl)throw std::runtime_error("Gemma prefill oneMKL requires TINYSTRATA_INTEL_MKL");
#endif
        try{
            scores=allocate<float>(size_t(c)*std::max(t,8*h));
            if(mkl){keys=allocate<float>(elements);values=allocate<float>(elements);temporary=allocate<float>(size_t(t)*d);}
        }catch(...){for(auto* p:allocations)sycl::free(p,q);throw;}
    }
    ~Impl(){try{q.wait_and_throw();}catch(...){}for(auto* p:allocations)try{sycl::free(p,q);}catch(...){} }
    sycl::event normalize(float* score,int rows,int stride,int position,int first,int window){
        return q.parallel_for(sycl::nd_range<1>(size_t(rows)*128,128),[=](sycl::nd_item<1> it){
            const size_t row=it.get_group_linear_id(),lane=it.get_local_linear_id();const int end=position+int(row),low=window?sycl::max(0,end-window+1):0;
            float maximum=-INFINITY;for(int col=int(lane);col<stride;col+=128)if(first+col>=low&&first+col<=end)maximum=sycl::fmax(maximum,score[row*stride+col]);
            maximum=sycl::reduce_over_group(it.get_group(),maximum,sycl::maximum<float>());
            float total=0;for(int col=int(lane);col<stride;col+=128)if(first+col>=low&&first+col<=end)total+=sycl::exp(score[row*stride+col]-maximum);
            total=sycl::reduce_over_group(it.get_group(),total,sycl::plus<float>());
            for(int col=int(lane);col<stride;col+=128)score[row*stride+col]=first+col>=low&&first+col<=end?sycl::exp(score[row*stride+col]-maximum)/total:0.f;
        });
    }
};
GemmaAttention::GemmaAttention(sycl::queue& q,int c,int t,int h,int d,size_t e,bool m):impl_(std::make_unique<Impl>(q,c,t,h,d,e,m)){}
GemmaAttention::~GemmaAttention()=default;
sycl::event GemmaAttention::run(const float* query,GemmaKv c,float* output,int heads,int position,int tokens,float scale){
    auto& p=*impl_;auto& q=p.q;cache_shape(c);
    if(heads<1||heads>p.heads||heads%c.heads||c.dim>p.dim||tokens<1||tokens>p.tokens||position<0||position+tokens>p.context||!std::isfinite(scale)||!(scale>0))
        throw std::invalid_argument("Invalid Gemma attention extent");
    const int first=c.window?std::max(0,position-c.window+1):0,span=position+tokens-first;
    if(span>c.capacity)throw std::invalid_argument("Gemma ring no longer holds this causal chunk");
    float* scores=p.scores;
#ifdef TINYSTRATA_INTEL_MKL
    if(p.mkl&&tokens>=16){
        const size_t count=size_t(c.heads)*span*c.dim;if(count>p.cache_elements)throw std::runtime_error("Gemma KV expansion exceeds its bound");
        float *keys=p.keys,*values=p.values,*temporary=p.temporary;
        q.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id){
            const size_t col=id[0]%c.dim,cell=(id[0]/c.dim)%span,head=id[0]/c.dim/span;
            const size_t source=(head*c.capacity+(first+cell)%c.capacity)*c.dim+col;
            keys[id]=float(c.key[source])*float(c.key_scale[source/32]);values[id]=float(c.value[source])*float(c.value_scale[source/32]);
        });
        sycl::event last;
        for(int head=0;head<heads;++head){
            const int kv=head/(heads/c.heads);
            oneapi::mkl::blas::row_major::gemm(q,oneapi::mkl::transpose::nontrans,oneapi::mkl::transpose::trans,
                tokens,span,c.dim,scale,query+head*c.dim,heads*c.dim,keys+size_t(kv)*span*c.dim,c.dim,0.f,scores,span);
            p.normalize(scores,tokens,span,position,first,c.window);
            auto pv=oneapi::mkl::blas::row_major::gemm(q,oneapi::mkl::transpose::nontrans,oneapi::mkl::transpose::nontrans,
                tokens,c.dim,span,1.f,scores,span,values+size_t(kv)*span*c.dim,c.dim,0.f,temporary,c.dim);
            last=q.submit([&](sycl::handler& h){h.depends_on(pv);h.parallel_for(sycl::range<1>(size_t(tokens)*c.dim),[=](sycl::id<1> i){output[(i[0]/c.dim*heads+head)*c.dim+i[0]%c.dim]=temporary[i];});});
        }
        return last;
    }
#endif
    sycl::event last;
    for(int begin=0;begin<tokens;begin+=8){
        const int n=std::min(8,tokens-begin),base=position+begin;
        const float* queries=query+size_t(begin)*heads*c.dim;float* out=output+size_t(begin)*heads*c.dim;
        const size_t entries=size_t(n)*heads*span;
        q.parallel_for(sycl::nd_range<1>(((entries+7)/8)*128,128),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg=it.get_sub_group();const size_t entry=it.get_group_linear_id()*8+sg.get_group_linear_id();if(entry>=entries)return;
            const size_t cell=entry%span,head=(entry/span)%heads,token=entry/span/heads,lane=sg.get_local_linear_id(),kv=head/(heads/c.heads);
            const size_t source=(kv*c.capacity+(first+cell)%c.capacity)*c.dim;float sum=0;
            for(size_t col=lane;col<size_t(c.dim);col+=16)sum=sycl::fma(queries[(token*heads+head)*c.dim+col],float(c.key[source+col])*float(c.key_scale[(source+col)/32]),sum);
            sum=sycl::reduce_over_group(sg,sum,sycl::plus<float>());if(lane==0)scores[entry]=sum*scale;
        });
        q.parallel_for(sycl::nd_range<1>(size_t(n)*heads*128,128),[=](sycl::nd_item<1> it){
            const size_t row=it.get_group_linear_id(),token=row/heads,lane=it.get_local_linear_id();const int end=base+int(token),low=c.window?sycl::max(0,end-c.window+1):0;
            float maximum=-INFINITY;for(int col=int(lane);col<span;col+=128)if(first+col>=low&&first+col<=end)maximum=sycl::fmax(maximum,scores[row*span+col]);
            maximum=sycl::reduce_over_group(it.get_group(),maximum,sycl::maximum<float>());
            float sum=0;for(int col=int(lane);col<span;col+=128)if(first+col>=low&&first+col<=end)sum+=sycl::exp(scores[row*span+col]-maximum);
            sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
            for(int col=int(lane);col<span;col+=128)scores[row*span+col]=first+col>=low&&first+col<=end?sycl::exp(scores[row*span+col]-maximum)/sum:0.f;
        });
        last=q.parallel_for(sycl::range<1>(size_t(n)*heads*c.dim),[=](sycl::id<1> id){
            const size_t col=id[0]%c.dim,head=(id[0]/c.dim)%heads,token=id[0]/c.dim/heads,kv=head/(heads/c.heads);float sum=0;
            for(int cell=0;cell<span;++cell){const size_t src=(kv*c.capacity+(first+cell)%c.capacity)*c.dim+col;sum=sycl::fma(scores[(token*heads+head)*span+cell],float(c.value[src])*float(c.value_scale[src/32]),sum);}
            out[id]=sum;
        });
    }
    return last;
}
}
