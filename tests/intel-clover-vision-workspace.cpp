// Model-free GPU alias gate. References reproduce the prior inline engine
// arithmetic, independently of the new alias-capable operation entry points.
#include "strata/intel/device.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/context_policy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace si=strata::intel;
namespace {
constexpr float guard=-12345.25f;
int cases=0;size_t checked_floats=0;
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
template<class T>struct Buffer{
    sycl::queue& q;T* owner;T* data;size_t size;
    Buffer(sycl::queue& queue,size_t n):q(queue),owner(sycl::malloc_shared<T>(n+32,q)),data(owner?owner+16:nullptr),size(n){
        if(!owner)throw std::bad_alloc();std::fill_n(owner,n+32,T(guard));}
    ~Buffer(){try{q.wait_and_throw();sycl::free(owner,q);}catch(...){}}
    void guards()const{for(int i=0;i<16;++i)require(owner[i]==T(guard)&&data[size+i]==T(guard),"Alias operation wrote a canary");}
};
void identical(const float* a,const float* b,size_t n,const char* why){
    if(n&&std::memcmp(a,b,n*sizeof(float)))throw std::runtime_error(why);checked_floats+=n;
}
float sample(size_t i,int salt){return float(int((i*size_t(salt)+17)%137)-68)/53.f;}
void silu(sycl::queue& q,int tokens,int width){
    const size_t n=size_t(tokens)*width;
    Buffer<float> gate(q,n),up(q,n),out(q,n),alias(q,n);
    for(size_t i=0;i<n;++i){gate.data[i]=sample(i,13);up.data[i]=sample(i,29);alias.data[i]=gate.data[i];}
    const std::vector<float> raw(gate.data,gate.data+n),raw_up(up.data,up.data+n);
    const float* x=gate.data;const float* u=up.data;float* y=out.data;
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){y[i]=x[i]/(1.f+sycl::exp(-x[i]))*u[i];});
    si::silu_mul(q,alias.data,up.data,alias.data,n).wait_and_throw();
    identical(out.data,alias.data,n,"In-place SiLU product differs bitwise from prior engine");
    identical(gate.data,raw.data(),n,"Reference SiLU changed its input");identical(up.data,raw_up.data(),n,"SiLU changed up projection");
    gate.guards();up.guards();out.guards();alias.guards();++cases;
}
void convolution_reference(sycl::queue& q,const float* input,float* history,const float* weight,float* output,int tokens,
                           int hk=16,int hv=32,float epsilon=1e-6f){
    const int channels=(2*hk+hv)*128;
    q.parallel_for(sycl::nd_range<1>(channels,128),[=](sycl::nd_item<1> it){
        const size_t channel=it.get_global_linear_id();
        for(int token=0;token<tokens;++token){
            const float current=input[size_t(token)*channels+channel];
            float value=history[channel*3]*weight[channel*4]+history[channel*3+1]*weight[channel*4+1]
                +history[channel*3+2]*weight[channel*4+2]+current*weight[channel*4+3];
            history[channel*3]=history[channel*3+1];history[channel*3+1]=history[channel*3+2];history[channel*3+2]=current;
            value=value/(1.f+sycl::exp(-value));
            if(it.get_group_linear_id()<size_t(2*hk)){
                const float sum=sycl::reduce_over_group(it.get_group(),value*value,sycl::plus<float>());
                value*=sycl::rsqrt(sum+epsilon);
            }
            output[size_t(token)*channels+channel]=value;
        }
    });
}
void conv(sycl::queue& q,int tokens){
    constexpr size_t channels=8192,hsize=channels*3;const size_t n=size_t(tokens)*channels;
    Buffer<float> input(q,n),reference(q,n),alias(q,n),separate(q,n),weights(q,channels*4);
    Buffer<float> rh(q,hsize),ah(q,hsize),sh(q,hsize),next(q,channels),next_ref(q,channels);
    for(size_t i=0;i<n;++i)input.data[i]=alias.data[i]=sample(i,17);
    for(size_t i=0;i<weights.size;++i)weights.data[i]=sample(i,19)/4;
    for(size_t i=0;i<hsize;++i)rh.data[i]=ah.data[i]=sh.data[i]=sample(i,31);
    const std::vector<float> raw(input.data,input.data+n),w(weights.data,weights.data+weights.size);
    std::vector<float> history(rh.data,rh.data+hsize);
    for(int t=0;t<tokens;++t)for(size_t c=0;c<channels;++c){history[c*3]=history[c*3+1];history[c*3+1]=history[c*3+2];history[c*3+2]=raw[size_t(t)*channels+c];}
    convolution_reference(q,input.data,rh.data,weights.data,reference.data,tokens);
    si::conv_step_many(q,input.data,sh.data,weights.data,separate.data,16,32,tokens,1e-6f);
    si::conv_step_many(q,alias.data,ah.data,weights.data,alias.data,16,32,tokens,1e-6f).wait_and_throw();
    identical(reference.data,separate.data,n,"Separate convolution differs bitwise from prior engine");
    identical(reference.data,alias.data,n,"Aliased convolution differs bitwise from prior engine");
    identical(ah.data,history.data(),hsize,"Aliased convolution history retained normalized values instead of raw input");
    identical(rh.data,ah.data,hsize,"Convolution history differs with alias");identical(sh.data,ah.data,hsize,"Separate convolution history differs");
    identical(input.data,raw.data(),n,"Reference convolution modified raw QKV");identical(weights.data,w.data(),weights.size,"Convolution changed weights");
    // A following decode token must consume exactly the post-prefill history.
    const std::vector<float> prior_tail(alias.data+channels,alias.data+n);
    for(size_t c=0;c<channels;++c)next.data[c]=alias.data[c]=sample(c,43);
    convolution_reference(q,next.data,rh.data,weights.data,next_ref.data,1);
    si::conv_step_many(q,alias.data,ah.data,weights.data,alias.data,16,32,1,1e-6f).wait_and_throw();
    identical(next_ref.data,alias.data,channels,"Post-prefill decode convolution differs");
    identical(rh.data,ah.data,hsize,"Post-prefill decode history differs");
    identical(alias.data+channels,prior_tail.data(),prior_tail.size(),"Decode convolution overwrote a later scratch row");
    for(auto* b:{&input,&reference,&alias,&separate,&weights,&rh,&ah,&sh,&next,&next_ref})b->guards();++cases;
}
void norm_gate(sycl::queue& q,int tokens){
    constexpr int width=128;const int rows=tokens*32;const size_t n=size_t(rows)*width;
    Buffer<float> input(q,n),reference(q,n),alias(q,n),weight(q,width),gate(q,n);
    for(size_t i=0;i<n;++i){input.data[i]=alias.data[i]=sample(i,47);gate.data[i]=sample(i,59);}
    for(int c=0;c<width;++c)weight.data[c]=1.f+sample(c,61)/4;
    const std::vector<float> raw(input.data,input.data+n),g(gate.data,gate.data+n),w(weight.data,weight.data+width);
    si::rms_norm(q,input.data,weight.data,reference.data,rows,width,1e-6f);
    si::rms_norm(q,alias.data,weight.data,alias.data,rows,width,1e-6f).wait_and_throw();
    identical(reference.data,alias.data,n,"RMS reduction/output overlap differs bitwise");
    float* a=alias.data;float* r=reference.data;const float* z=gate.data;
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){r[i]*=z[i]/(1.f+sycl::exp(-z[i]));});
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){a[i]*=z[i]/(1.f+sycl::exp(-z[i]));}).wait_and_throw();
    identical(reference.data,alias.data,n,"Aliased gated normalization differs bitwise");
    identical(input.data,raw.data(),n,"RMS reference modified recurrent input");identical(gate.data,g.data(),n,"RMS gate changed");
    identical(weight.data,w.data(),width,"RMS weights changed");
    for(auto* b:{&input,&reference,&alias,&weight,&gate})b->guards();++cases;
}
// Exercise the actual full-attention and GDN producers/consumers across the
// MoE phase boundary. The reference has separate storage for every tensor;
// only the candidate uses the production phase layout and prior in-place ops.
void phase_workspace(sycl::queue& q,int tokens){
    constexpr int capacity=512,raw_width=8192,query_width=4096,routed_width=16384;
    const auto layout=si::context_policy::clover_phase_layout(capacity,raw_width,query_width,4096,routed_width,4096,4096);
    Buffer<float> arena(q,layout.routed_count),gate_owner(q,layout.gate_count);
    Buffer<float> raw(q,capacity*raw_width),conv(q,capacity*raw_width),query(q,capacity*query_width);
    Buffer<float> gate(q,capacity*query_width),attention(q,capacity*query_width),gdn(q,capacity*query_width),norm(q,capacity*query_width);
    Buffer<float> routed(q,capacity*routed_width),rh(q,raw_width*3),ah(q,raw_width*3),weight(q,raw_width*4);
    Buffer<float> rs(q,32*128*128),as(q,32*128*128),alpha(q,capacity*32),beta(q,capacity*32),nw(q,256);
    Buffer<float> rk(q,capacity*512),ak(q,capacity*512),values(q,capacity*512);
    Buffer<float> rck(q,capacity*512),ack(q,capacity*512),rcv(q,capacity*512),acv(q,capacity*512);
    Buffer<int32_t> positions(q,capacity),mrope(q,capacity*3);
    float *ap=arena.data,*aq=arena.data+layout.query_offset,*ao=arena.data+layout.attention_offset,*az=gate_owner.data;
    require(ap+capacity*raw_width==aq&&aq+capacity*query_width==ao,"Full attention views overlap");
    require(ao+capacity*query_width==arena.data+arena.size,"Phase arena exceeds its sole owner");
    for(size_t i=0;i<rh.size;++i)rh.data[i]=ah.data[i]=sample(i,31);
    for(size_t i=0;i<weight.size;++i)weight.data[i]=sample(i,19)/4;
    std::fill_n(rs.data,rs.size,0.f);std::fill_n(as.data,as.size,0.f);
    for(int i=0;i<256;++i)nw.data[i]=1.f+sample(i,61)/4;
    for(int i=0;i<capacity;++i){positions.data[i]=i;for(int axis=0;axis<3;++axis)mrope.data[i*3+axis]=i;}
    auto produce=[&](float* dst,size_t n,int salt){q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){
        dst[i]=float(int((i[0]*size_t(salt)+17)%137)-68)/53.f;});};
    auto gdn_phase=[&](int count,int salt){
        const size_t raw_n=size_t(count)*raw_width,n=size_t(count)*query_width;
        produce(raw.data,raw_n,salt);produce(ap,raw_n,salt);produce(gate.data,n,salt+2);produce(az,n,salt+2);
        float *a=alpha.data,*b=beta.data;
        q.parallel_for(sycl::range<1>(size_t(count)*32),[=](sycl::id<1> i){a[i]=-.05f;b[i]=.25f;});
        convolution_reference(q,raw.data,rh.data,weight.data,conv.data,count);
        si::conv_step_many(q,ap,ah.data,weight.data,ap,16,32,count,1e-6f);
        si::gdn_recurrence_tiled(q,rs.data,conv.data,alpha.data,beta.data,gdn.data,16,32,count);
        si::gdn_recurrence_tiled(q,as.data,ap,alpha.data,beta.data,aq,16,32,count);
        si::rms_norm(q,gdn.data,nw.data,norm.data,count*32,128,1e-6f);
        si::rms_norm(q,aq,nw.data,aq,count*32,128,1e-6f);
        auto apply_gate=[&](float* dst,const float* z){q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){dst[i]*=z[i]/(1.f+sycl::exp(-z[i]));});};
        apply_gate(norm.data,gate.data);apply_gate(aq,az);q.wait_and_throw();
        identical(conv.data,ap,raw_n,"Phase arena changed convolution");
        identical(norm.data,aq,n,"Phase arena changed GDN gated output");
        identical(rh.data,ah.data,rh.size,"Phase arena changed next-token raw history");
        identical(rs.data,as.data,rs.size,"Phase arena changed recurrent state");
    };
    auto moe_phase=[&](int count,int salt){
        const size_t n=size_t(count)*query_width,down_n=size_t(count)*routed_width;
        // Routed gate projection replaces the expired GDN/attention gate.
        produce(gate.data,n,salt);produce(az,n,salt);
        float *r=routed.data;const float* g=gate.data;
        q.parallel_for(sycl::range<1>(down_n),[=](sycl::id<1> i){r[i]=g[i[0]%n]*.5f+float(i[0]%8);});
        q.parallel_for(sycl::range<1>(down_n),[=](sycl::id<1> i){ap[i]=az[i[0]%n]*.5f+float(i[0]%8);}).wait_and_throw();
        identical(routed.data,ap,down_n,"Routed output changed after phase reuse");
        identical(gate.data,az,n,"Expert gate changed after phase reuse");
    };
    auto attention_phase=[&](int count,int salt){
        const size_t raw_n=size_t(count)*raw_width,n=size_t(count)*query_width,kv_n=size_t(count)*512;
        produce(raw.data,raw_n,salt);produce(ap,raw_n,salt);
        produce(rk.data,kv_n,salt+2);produce(ak.data,kv_n,salt+2);produce(values.data,kv_n,salt+4);
        si::prepare_qwen_attention(q,raw.data,rk.data,values.data,nw.data,nw.data,query.data,gate.data,rk.data,rck.data,rcv.data,
            16,2,capacity,64,count,positions.data,mrope.data,1000000.f,1e-6f,false);
        si::prepare_qwen_attention(q,ap,ak.data,values.data,nw.data,nw.data,aq,az,ak.data,ack.data,acv.data,
            16,2,capacity,64,count,positions.data,mrope.data,1000000.f,1e-6f,false);
        si::causal_attention(q,query.data,gate.data,rck.data,rcv.data,attention.data,16,2,capacity,0,count);
        si::causal_attention(q,aq,az,ack.data,acv.data,ao,16,2,capacity,0,count).wait_and_throw();
        identical(query.data,aq,n,"Phase arena changed prepared attention queries");
        identical(gate.data,az,n,"Phase arena changed prepared attention gates");
        identical(attention.data,ao,n,"Phase arena changed full attention output");
        identical(raw.data,ap,raw_n,"Full attention overwrote its raw producer");
    };
    gdn_phase(tokens,17);moe_phase(tokens,71);attention_phase(tokens,83);moe_phase(tokens,97);
    // Reuse both owner arrays once more. The following decode consumes the
    // exact history/state left before full-attention and MoE overwrote scratch.
    gdn_phase(1,101);moe_phase(1,107);
    for(auto* b:{&arena,&gate_owner,&raw,&conv,&query,&gate,&attention,&gdn,&norm,&routed,&rh,&ah,&weight,&rs,&as,
        &alpha,&beta,&nw,&rk,&ak,&values,&rck,&ack,&rcv,&acv})b->guards();
    positions.guards();mrope.guards();++cases;
}
}
int main()try{
    sycl::queue q(si::select_device(0),sycl::property::queue::in_order{});
    for(int tokens:{1,7,512}){silu(q,tokens,8*512);silu(q,tokens,512);conv(q,tokens);norm_gate(q,tokens);}
    for(int tokens:{1,7,256,512})phase_workspace(q,tokens);
    std::printf("Clover image workspace alias GPU PASS cases=%d bitwise_floats=%zu token_counts=1,7,256,512 phase_reuse=passed next_history=passed canaries=passed\n",cases,checked_floats);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
