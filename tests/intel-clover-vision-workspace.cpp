// Model-free GPU alias gate. References reproduce the prior inline engine
// arithmetic, independently of the new alias-capable operation entry points.
#include "strata/intel/device.hpp"
#include "strata/intel/qwen_ops.hpp"
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
    if(std::memcmp(a,b,n*sizeof(float)))throw std::runtime_error(why);checked_floats+=n;
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
}
int main()try{
    sycl::queue q(si::select_device(0),sycl::property::queue::in_order{});
    for(int tokens:{1,7,512}){silu(q,tokens,8*512);silu(q,tokens,512);conv(q,tokens);norm_gate(q,tokens);}
    std::printf("Clover image workspace alias GPU PASS cases=%d bitwise_floats=%zu token_counts=1,7,512 canaries=passed\n",cases,checked_floats);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
