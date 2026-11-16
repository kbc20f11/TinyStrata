#include "strata/intel/device.hpp"
#include "strata/intel/qwen_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace strata::intel;
constexpr float canary=-12345.25f;
template<class T> struct Buffer {
    sycl::queue& q;T* owner;T* data;size_t count;
    Buffer(sycl::queue& queue,size_t n):q(queue),count(n){owner=sycl::malloc_shared<T>(n+32,q);if(!owner)throw std::bad_alloc();data=owner+16;}
    ~Buffer(){try{q.wait_and_throw();}catch(...){}sycl::free(owner,q);}
    void fill(T x){std::fill(owner,owner+count+32,x);}
    void guards(T x)const{for(size_t i=0;i<16;++i)if(owner[i]!=x||data[count+i]!=x)throw std::runtime_error("Qwen vision operation guard changed");}
};
static void need(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
int main() try {
    auto device=select_device(0);sycl::queue q(device,sycl::property_list{sycl::property::queue::in_order{}});
    constexpr int heads=16,kh=2,ctx=19,dim=256,max_t=8;constexpr double epsilon=1e-6,base=1e7;
    Buffer<float> packed(q,max_t*heads*512),keys(q,max_t*kh*dim),values(q,max_t*kh*dim),qw(q,dim),kw(q,dim);
    Buffer<float> queries(q,max_t*heads*dim),gates(q,max_t*heads*dim),rotated(q,max_t*kh*dim),ck(q,kh*ctx*dim),cv(q,kh*ctx*dim);
    Buffer<int32_t> position(q,1),positions(q,ctx*3);
    for(auto* b:{&packed,&keys,&values,&qw,&kw,&queries,&gates,&rotated,&ck,&cv})b->fill(canary);
    position.fill(-777);positions.fill(-777);
    for(size_t i=0;i<packed.count;++i)packed.data[i]=float(int(i*17%119)-59)/71;
    for(size_t i=0;i<keys.count;++i)keys.data[i]=float(int(i*13%101)-50)/67;
    for(size_t i=0;i<values.count;++i)values.data[i]=float(int(i*19%89)-44)/53;
    for(int i=0;i<dim;++i){qw.data[i]=0.8f+float(i%19)/73;kw.data[i]=1.1f-float(i%17)/81;}
    const std::vector<float> original_q(packed.data,packed.data+packed.count),original_k(keys.data,keys.data+keys.count),original_v(values.data,values.data+values.count);
    const std::vector<float> original_qw(qw.data,qw.data+dim),original_kw(kw.data,kw.data+dim);
    size_t cases=0;double worst=0;
    for(int pos:{5,13})for(int tokens:{1,3})for(bool quantized:{false,true})for(bool multimodal:{false,true}) {
        position.data[0]=pos;
        for(int c=0;c<ctx;++c)for(int a=0;a<3;++a)positions.data[c*3+a]=c;
        if(multimodal){for(int c=5;c<13;++c){positions.data[c*3]=3;positions.data[c*3+1]=3+(c-5)/4;positions.data[c*3+2]=3+(c-5)%4;}
            for(int c=13;c<ctx;++c)for(int a=0;a<3;++a)positions.data[c*3+a]=c-6;}
        for(auto* b:{&queries,&gates,&rotated,&ck,&cv})b->fill(canary);
        prepare_qwen_attention(q,packed.data,keys.data,values.data,qw.data,kw.data,queries.data,gates.data,
            rotated.data,ck.data,cv.data,heads,kh,ctx,64,tokens,position.data,multimodal?positions.data:nullptr,float(base),float(epsilon),quantized).wait_and_throw();
        for(int token=0;token<tokens;++token)for(int h=0;h<heads+kh;++h){
            const bool query=h<heads;const auto& data=query?original_q:original_k;
            const size_t at=query?(size_t(token)*heads+h)*512:(size_t(token)*kh+h-heads)*dim;
            double sum=0;for(int c=0;c<dim;++c)sum+=double(data[at+c])*data[at+c];
            const auto& gamma=query?original_qw:original_kw;double norm[dim];
            for(int c=0;c<dim;++c)norm[c]=double(data[at+c])/std::sqrt(sum/dim+epsilon)*gamma[c];
            for(int c=0;c<dim;++c){double expected=norm[c];
                if(c<64){const int pair=c<32?c:c-32;const int rp=multimodal?positions.data[(pos+token)*3+pair%3]:pos+token;
                    const double theta=rp*std::pow(base,-2.0*pair/64);
                    expected=c<32?norm[c]*std::cos(theta)-norm[c+32]*std::sin(theta):norm[c-32]*std::sin(theta)+norm[c]*std::cos(theta);}
                const size_t out=query?(size_t(token)*heads+h)*dim+c:(size_t(token)*kh+h-heads)*dim+c;
                const size_t cache=(size_t(h-heads)*ctx+pos+token)*dim+c;
                const float got=query?queries.data[out]:quantized?rotated.data[out]:ck.data[cache];
                const double error=std::abs(double(got)-expected);worst=std::max(worst,error);need(error<2e-5,"Qwen M-RoPE differs from independent FP64 reference");
                if(query)need(gates.data[out]==original_q[at+256+c],"Query gate changed");
                else if(!quantized)need(cv.data[cache]==original_v[(size_t(token)*kh+h-heads)*dim+c],"Value used rotary coordinates as KV address");
            }
        }
        for(int h=0;h<kh;++h)for(int cell=0;cell<ctx;++cell)for(int c=0;c<dim;++c)
            if(quantized||cell<pos||cell>=pos+tokens){const auto i=(size_t(h)*ctx+cell)*dim+c;need(ck.data[i]==canary&&cv.data[i]==canary,"Cache modified outside physical token range");}
        if(!multimodal){const std::vector<float> one(queries.data,queries.data+size_t(tokens)*heads*dim);
            prepare_qwen_attention(q,packed.data,keys.data,values.data,qw.data,kw.data,queries.data,gates.data,rotated.data,ck.data,cv.data,
                heads,kh,ctx,64,tokens,position.data,positions.data,float(base),float(epsilon),quantized).wait_and_throw();
            need(std::memcmp(one.data(),queries.data,one.size()*sizeof(float))==0,"Identity image positions changed text arithmetic");}
        for(size_t i=size_t(tokens)*heads*dim;i<queries.count;++i)need(queries.data[i]==canary&&gates.data[i]==canary,"Unused query rows changed");
        ++cases;
    }
    // Production INT8 KV aliases its rotated key rows with the normalized input.
    position.data[0]=5;std::copy(original_k.begin(),original_k.end(),keys.data);
    prepare_qwen_attention(q,packed.data,keys.data,values.data,qw.data,kw.data,queries.data,gates.data,rotated.data,nullptr,nullptr,
        heads,kh,ctx,64,8,position.data,positions.data,float(base),float(epsilon),true).wait_and_throw();
    const std::vector<float> separate(rotated.data,rotated.data+8*kh*dim);
    prepare_qwen_attention(q,packed.data,keys.data,values.data,qw.data,kw.data,queries.data,gates.data,keys.data,nullptr,nullptr,
        heads,kh,ctx,64,8,position.data,positions.data,float(base),float(epsilon),true).wait_and_throw();
    need(std::memcmp(separate.data(),keys.data,separate.size()*sizeof(float))==0,"Aliased rotated keys differ");
    need(std::memcmp(original_q.data(),packed.data,original_q.size()*sizeof(float))==0,"Packed query input mutated");
    need(std::memcmp(original_v.data(),values.data,original_v.size()*sizeof(float))==0,"Value input mutated");
    for(auto* b:{&packed,&keys,&values,&qw,&kw,&queries,&gates,&rotated,&ck,&cv})b->guards(canary);
    position.guards(-777);positions.guards(-777);
    std::printf("{\"passed\":true,\"cases\":%zu,\"FP64_max_abs\":%.9g,\"physical_KV_addresses\":true,\"identity_bit_exact\":true,\"aliased_rotated_keys\":true,\"canaries\":true}\n",cases+1,worst);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
