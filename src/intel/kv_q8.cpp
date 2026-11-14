#include "strata/intel/kv_q8.hpp"
#include <cstring>
#include <stdexcept>
namespace strata::intel {
namespace {
void validate(const sycl::queue& q, Q8Kv c) {
    if (!q.has_property<sycl::property::queue::in_order>() || c.context < 1 || c.heads < 1 ||
        !c.key || !c.value || !c.key_scale || !c.value_scale)
        throw std::invalid_argument("Invalid Intel Q8 KV geometry/queue");
}
// Preserve each query's original FMA/reduction order while sharing decoded
// KV words across three/four queries. Admission uses the CPU-known visible end.
template<int Tokens> sycl::event token_reuse_attention(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* scores,float* partial,float* output,int heads,const int32_t* position,int extent,sycl::event* stages) {
    static_assert(Tokens>=3 && Tokens<=4);
    constexpr size_t group=128,subgroup=16,per_group=group/subgroup,block=256;
    const size_t entries=size_t(heads)*extent,splits=(c.context+block-1)/block;
    auto qk=q.parallel_for(sycl::nd_range<1>(((entries+per_group-1)/per_group)*group,group),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg=it.get_sub_group();const size_t entry=it.get_group_linear_id()*per_group+sg.get_group_linear_id();
        const size_t head=entry%heads,cell=entry/heads,lane=sg.get_local_linear_id();
        if(cell>=size_t(extent))return;
        bool active[Tokens];bool any=false;float sum[Tokens]={};
#pragma unroll
        for(int token=0;token<Tokens;++token){active[token]=cell<=size_t(position[token]);any=any||active[token];}
        if(!any)return;
        const size_t kv=head/(heads/c.heads);
        for(size_t word=lane;word<64;word+=subgroup){
            const size_t col=word*4,i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.key+i,sizeof(codes));
            const float scale=float(c.key_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j){
                const float value=float(int8_t(codes>>(8*j)))*scale;
#pragma unroll
                for(int token=0;token<Tokens;++token)if(active[token])
                    sum[token]=sycl::fma(query[(size_t(token)*heads+head)*256+col+j],value,sum[token]);
            }
        }
#pragma unroll
        for(int token=0;token<Tokens;++token){
            const float reduced=sycl::reduce_over_group(sg,sum[token],sycl::plus<float>());
            if(lane==0 && active[token])scores[(size_t(token)*heads+head)*c.context+cell]=reduced/16.f;
        }
    });
    auto softmax=q.parallel_for(sycl::nd_range<2>({size_t(Tokens),size_t(heads)*group},{1,group}),[=](sycl::nd_item<2> it){
        const size_t head=it.get_group(0)*heads+it.get_group(1),lane=it.get_local_linear_id(),end=position[it.get_group(0)];float maximum=-INFINITY;
        for(size_t cell=lane;cell<=end;cell+=group)maximum=sycl::fmax(maximum,scores[head*c.context+cell]);
        maximum=sycl::reduce_over_group(it.get_group(),maximum,sycl::maximum<float>());float sum=0;
        for(size_t cell=lane;cell<=end;cell+=group)sum+=sycl::exp(scores[head*c.context+cell]-maximum);
        sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
        for(size_t cell=lane;cell<=end;cell+=group)scores[head*c.context+cell]=sycl::exp(scores[head*c.context+cell]-maximum)/sum;
    });
    constexpr size_t pv_group=64;
    auto pv=q.parallel_for(sycl::nd_range<1>(size_t((extent+block-1)/block)*heads*pv_group,pv_group),[=](sycl::nd_item<1> it){
        const size_t head=it.get_group_linear_id()%heads,split=it.get_group_linear_id()/heads,lane=it.get_local_linear_id(),kv=head/(heads/c.heads);
        const size_t col=lane*4,begin=split*block;float a[Tokens][4]={};
        int latest=position[0];
#pragma unroll
        for(int token=1;token<Tokens;++token)latest=sycl::max(latest,position[token]);
        const size_t end=sycl::min(begin+block,size_t(latest+1));
        for(size_t cell=begin;cell<end;++cell){
            const size_t i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.value+i,sizeof(codes));
            const float scale=float(c.value_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j){
                const float value=float(int8_t(codes>>(8*j)))*scale;
#pragma unroll
                for(int token=0;token<Tokens;++token)if(cell<=size_t(position[token]))
                    a[token][j]=sycl::fma(scores[(size_t(token)*heads+head)*c.context+cell],value,a[token][j]);
            }
        }
#pragma unroll
        for(int token=0;token<Tokens;++token){
            const size_t target=((size_t(token)*heads+head)*splits+split)*256+col;
#pragma unroll
            for(int j=0;j<4;++j)partial[target+j]=a[token][j];
        }
    });
    if(stages){stages[0]=qk;stages[1]=softmax;stages[2]=pv;}
    return q.parallel_for(sycl::range<2>(size_t(Tokens),size_t(heads)*256),[=](sycl::id<2> id){
        const size_t head=id[0]*heads+id[1]/256,col=id[1]%256,used=(size_t(position[id[0]]+1)+block-1)/block;float sum=0;
        for(size_t split=0;split<used;++split)sum+=partial[(head*splits+split)*256+col];
        const size_t index=head*256+col;
        output[index]=sum/(1.f+sycl::exp(-gate[index]));
    });
}
}
sycl::event store_q8_kv(sycl::queue& q,const float* key,const float* value,Q8Kv c,
    const int32_t* position,int tokens,int32_t* error) {
    validate(q,c);
    if(tokens<1 || tokens>1024 || !error)throw std::invalid_argument("Invalid Q8 KV store");
    return q.parallel_for(sycl::nd_range<1>(size_t(tokens)*c.heads*256,32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t i=it.get_global_linear_id(), row=i/256, col=i%256, token=row/c.heads, head=row%c.heads;
        const int cell=position[0]+int(token);
        const auto sg=it.get_sub_group();
        if(cell<0 || cell>=c.context){
            if(sg.get_local_linear_id()==0)sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(error[0]).store(1);
            return;
        }
        const float k=key[i],v=value[i];
        const bool nonfinite=sycl::any_of_group(sg,!sycl::isfinite(k)||!sycl::isfinite(v));
        const float km=sycl::reduce_over_group(sg,sycl::fabs(k),sycl::maximum<float>());
        const float vm=sycl::reduce_over_group(sg,sycl::fabs(v),sycl::maximum<float>());
        const sycl::half kd=sycl::half(km/127.f),vd=sycl::half(vm/127.f);
        const bool bad=nonfinite || !sycl::isfinite(float(kd)) || !sycl::isfinite(float(vd));
        const size_t target=(head*c.context+size_t(cell))*256+col;
        const float kc=!bad && km>0 ? sycl::rint(k/km*127.f) : 0.f;
        const float vc=!bad && vm>0 ? sycl::rint(v/vm*127.f) : 0.f;
        c.key[target]=int8_t(sycl::clamp(kc,-127.f,127.f));
        c.value[target]=int8_t(sycl::clamp(vc,-127.f,127.f));
        if(sg.get_local_linear_id()==0){
            c.key_scale[target/32]=bad ? sycl::half(0.f) : kd;
            c.value_scale[target/32]=bad ? sycl::half(0.f) : vd;
            if(bad)sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(error[0]).store(1);
        }
    });
}
sycl::event dequantize_q8_kv(sycl::queue& q,Q8Kv c,float* key,float* value,int visible){
    validate(q,c);
    if(visible<1||visible>c.context)throw std::invalid_argument("Invalid Q8 KV visibility");
    return q.parallel_for(sycl::range<1>(size_t(c.heads)*visible*256),[=](sycl::id<1> id){
        const size_t head=id[0]/(size_t(visible)*256),within=id[0]%(size_t(visible)*256),i=head*c.context*256+within;
        key[i]=float(c.key[i])*float(c.key_scale[i/32]);
        value[i]=float(c.value[i])*float(c.value_scale[i/32]);
    });
}
sycl::event decode_attention_q8(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* scores,float* partial,float* output,int heads,const int32_t* position,int extent,sycl::event* stages,int tokens,int actual_visible){
    validate(q,c);
    if(heads<1||heads%c.heads||extent<1||extent>c.context||tokens<1||tokens>8)throw std::invalid_argument("Invalid Q8 decode attention geometry");
    if(actual_visible<0 || actual_visible>extent)throw std::invalid_argument("Invalid Q8 actual visibility");
    if(actual_visible>=16384 && tokens==3)return token_reuse_attention<3>(q,query,gate,c,scores,partial,output,heads,position,extent,stages);
    if(actual_visible>=16384 && tokens==4)return token_reuse_attention<4>(q,query,gate,c,scores,partial,output,heads,position,extent,stages);
    constexpr size_t group=128,subgroup=16,per_group=group/subgroup,block=256;
    const size_t entries=size_t(heads)*extent,splits=(c.context+block-1)/block;
    auto qk=q.parallel_for(sycl::nd_range<2>({size_t(tokens),((entries+per_group-1)/per_group)*group},{1,group}),[=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg=it.get_sub_group();const size_t entry=it.get_group(1)*per_group+sg.get_group_linear_id();
        const size_t head=entry%heads,cell=entry/heads,token=it.get_group(0),lane=sg.get_local_linear_id();
        if(cell>=size_t(extent)||cell>size_t(position[token]))return;
        const size_t row=token*heads+head;
        const size_t kv=head/(heads/c.heads);float sum=0;
        for(size_t word=lane;word<64;word+=subgroup){
            const size_t col=word*4,i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.key+i,sizeof(codes));
            const float scale=float(c.key_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j)
                sum=sycl::fma(query[row*256+col+j],float(int8_t(codes>>(8*j)))*scale,sum);
        }
        sum=sycl::reduce_over_group(sg,sum,sycl::plus<float>());
        if(lane==0)scores[row*c.context+cell]=sum/16.f;
    });
    auto softmax=q.parallel_for(sycl::nd_range<2>({size_t(tokens),size_t(heads)*group},{1,group}),[=](sycl::nd_item<2> it){
        const size_t head=it.get_group(0)*heads+it.get_group(1),lane=it.get_local_linear_id(),end=position[it.get_group(0)];float maximum=-INFINITY;
        for(size_t cell=lane;cell<=end;cell+=group)maximum=sycl::fmax(maximum,scores[head*c.context+cell]);
        maximum=sycl::reduce_over_group(it.get_group(),maximum,sycl::maximum<float>());float sum=0;
        for(size_t cell=lane;cell<=end;cell+=group)sum+=sycl::exp(scores[head*c.context+cell]-maximum);
        sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
        for(size_t cell=lane;cell<=end;cell+=group)scores[head*c.context+cell]=sycl::exp(scores[head*c.context+cell]-maximum)/sum;
    });
    constexpr size_t pv_group=64;
    auto pv=q.parallel_for(sycl::nd_range<2>({size_t(tokens),size_t((extent+block-1)/block)*heads*pv_group},{1,pv_group}),[=](sycl::nd_item<2> it){
        const size_t local_head=it.get_group(1)%heads,head=it.get_group(0)*heads+local_head,split=it.get_group(1)/heads,lane=it.get_local_linear_id(),kv=local_head/(heads/c.heads);
        const size_t col=lane*4,begin=split*block,end=sycl::min(begin+block,size_t(position[it.get_group(0)]+1));float a[4]={};
        for(size_t cell=begin;cell<end;++cell){
            const float weight=scores[head*c.context+cell];const size_t i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.value+i,sizeof(codes));
            const float scale=float(c.value_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j)a[j]=sycl::fma(weight,float(int8_t(codes>>(8*j)))*scale,a[j]);
        }
        const size_t target=(head*splits+split)*256+col;
#pragma unroll
        for(int j=0;j<4;++j)partial[target+j]=a[j];
    });
    if(stages){stages[0]=qk;stages[1]=softmax;stages[2]=pv;}
    return q.parallel_for(sycl::range<2>(size_t(tokens),size_t(heads)*256),[=](sycl::id<2> id){
        const size_t head=id[0]*heads+id[1]/256,col=id[1]%256,used=(size_t(position[id[0]]+1)+block-1)/block;float sum=0;
        for(size_t split=0;split<used;++split)sum+=partial[(head*splits+split)*256+col];
        const size_t index=head*256+col;
        output[index]=sum/(1.f+sycl::exp(-gate[index]));
    });
}
}
