// Explicit model-free GPU gate; never part of the automatic CPU CTest group.
#include "strata/intel/device.hpp"
#include "strata/intel/kv_q8.hpp"
#include "strata/intel/qwen_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace si=strata::intel;
namespace {
constexpr int width=256,capacity=8;
constexpr float guard=-77.f;
int cases=0;
void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class T>struct Buffer{
    sycl::queue& q;T *owner,*data;size_t size;
    Buffer(sycl::queue& queue,size_t n):q(queue),owner(sycl::malloc_shared<T>(n+64,q)),data(owner?owner+32:nullptr),size(n){
        if(!owner)throw std::bad_alloc();std::fill_n(owner,n+64,T(guard));}
    ~Buffer(){try{q.wait_and_throw();sycl::free(owner,q);}catch(...){}}
    void guards(){for(int i=0;i<32;++i)check(owner[i]==T(guard)&&data[size+i]==T(guard),"KV allocation canary changed");}
};
// Numeric FP8 reference is independent of the production bit codec.
double reference_decode(uint8_t code){
    const int magnitude=code&127,e=magnitude/8,m=magnitude%8;
    const double a=magnitude==127?std::numeric_limits<double>::quiet_NaN():e?std::ldexp(1.+double(m)/8,e-7):std::ldexp(double(m),-9);
    return code&128?-a:a;
}
uint8_t reference_encode(float x){
    const bool sign=std::signbit(x);const double a=std::abs(double(x));
    unsigned selected=0;double error=a;
    for(unsigned c=1;c<=126;++c){const double d=std::abs(reference_decode(uint8_t(c))-a);
        if(d<error||(d==error&&!(c&1))){selected=c;error=d;}}
    return uint8_t((sign?128:0)|selected);
}
float decoded(int8_t code,sycl::half scale,bool fp8){return float(fp8?reference_decode(uint8_t(code)):double(code))*float(scale);}
void storage(sycl::queue& q,bool fp8){
    constexpr int context=17,heads=2,tokens=7;const size_t cells=context*heads*width,n=tokens*heads*width;
    Buffer<int8_t> k(q,cells),v(q,cells);Buffer<sycl::half> ks(q,cells/32),vs(q,cells/32);
    Buffer<float> input_k(q,n),input_v(q,n),dk(q,cells),dv(q,cells);Buffer<int32_t> position(q,1),error(q,1);
    si::Q8Kv cache{k.data,v.data,ks.data,vs.data,context,heads,fp8};
    std::fill_n(k.data,cells,int8_t(0));std::fill_n(v.data,cells,int8_t(0));std::fill_n(ks.data,cells/32,sycl::half(0));std::fill_n(vs.data,cells/32,sycl::half(0));
    for(size_t i=0;i<n;++i){input_k.data[i]=float(int((i*17+3)%255)-127)/31.f;input_v.data[i]=float(int((i*29+11)%255)-127)/53.f;}
    std::fill_n(input_k.data,32,0.f);for(int i=32;i<64;++i)input_k.data[i]=float(i-48)*0x1p-28f;
    position.data[0]=3;error.data[0]=0;
    si::store_q8_kv(q,input_k.data,input_v.data,cache,position.data,tokens,error.data).wait_and_throw();
    check(error.data[0]==0,"Valid KV store failed");
    for(int t=0;t<tokens;++t)for(int h=0;h<heads;++h)for(int block=0;block<8;++block){
        const size_t source=(size_t(t)*heads+h)*256+block*32,target=(size_t(h)*context+3+t)*256+block*32;
        float km=0,vm=0;for(int j=0;j<32;++j){km=std::max(km,std::abs(input_k.data[source+j]));vm=std::max(vm,std::abs(input_v.data[source+j]));}
        sycl::half kd(km/(fp8?448.f:127.f)),vd(vm/(fp8?448.f:127.f));
        if(fp8&&km>0&&float(kd)==0)kd=sycl::half(0x1p-24f);if(fp8&&vm>0&&float(vd)==0)vd=sycl::half(0x1p-24f);
        check(ks.data[target/32]==kd&&vs.data[target/32]==vd,"Stored scale differs from independent block maximum");
        for(int j=0;j<32;++j){
            const int8_t wanted_k=fp8?int8_t(km?reference_encode(input_k.data[source+j]/float(kd)):0):int8_t(km?std::nearbyint(input_k.data[source+j]/km*127.f):0);
            const int8_t wanted_v=fp8?int8_t(vm?reference_encode(input_v.data[source+j]/float(vd)):0):int8_t(vm?std::nearbyint(input_v.data[source+j]/vm*127.f):0);
            check(k.data[target+j]==wanted_k&&v.data[target+j]==wanted_v,"KV codes differ from independent reference/INT8 bitwise control");
        }
    }
    si::dequantize_q8_kv(q,cache,dk.data,dv.data,11).wait_and_throw();
    for(int h=0;h<heads;++h)for(int cell=0;cell<context;++cell)for(int col=0;col<256;++col){
        const size_t i=(size_t(h)*context+cell)*256+col;
        check(cell<11?dk.data[i]==decoded(k.data[i],ks.data[i/32],fp8)&&dv.data[i]==decoded(v.data[i],vs.data[i/32],fp8)
            :dk.data[i]==guard&&dv.data[i]==guard,"Dequantization value/visibility changed");
    }
    for(float bad:{INFINITY,std::numeric_limits<float>::quiet_NaN(),1e30f}){
        std::fill_n(input_k.data,n,0.f);std::fill_n(input_v.data,n,0.f);input_k.data[35]=bad;position.data[0]=0;error.data[0]=0;
        si::store_q8_kv(q,input_k.data,input_v.data,cache,position.data,1,error.data).wait_and_throw();
        check(error.data[0]==1,"Nonfinite/scale-overflow KV input was accepted");
        for(int c=32;c<64;++c)check(k.data[c]==0&&v.data[c]==0&&float(ks.data[c/32])==0&&float(vs.data[c/32])==0,"Rejected KV block was not zeroed");
    }
    for(int invalid:{-1,context}){position.data[0]=invalid;error.data[0]=0;
        si::store_q8_kv(q,input_k.data,input_v.data,cache,position.data,1,error.data).wait_and_throw();check(error.data[0]==1,"Out-of-range KV position accepted");}
    k.guards();v.guards();ks.guards();vs.guards();input_k.guards();input_v.guards();dk.guards();dv.guards();position.guards();error.guards();++cases;
}
void attention(sycl::queue& q,int context,int heads,bool fp8){
    constexpr int kv_heads=2;const int visible=context==32768?16512:context;const size_t cells=size_t(kv_heads)*context*width;
    Buffer<int8_t> k(q,cells),v(q,cells);Buffer<sycl::half> ks(q,cells/32),vs(q,cells/32);
    Buffer<float> fk(q,cells),fv(q,cells),query(q,capacity*heads*width),gate(q,capacity*heads*width);
    Buffer<float> scores(q,size_t(capacity)*heads*context),partial(q,size_t(capacity)*heads*((context+255)/256)*256);
    Buffer<float> output(q,capacity*heads*width),reference(q,capacity*heads*width);Buffer<int32_t> positions(q,capacity);
    for(size_t i=0;i<cells/32;++i){ks.data[i]=sycl::half(float(i%7+1)/2048);vs.data[i]=sycl::half(float(i%11+1)/1024);}
    for(size_t i=0;i<cells;++i){k.data[i]=fp8?int8_t(((i*17+3)%127)|((i&1)?128:0)):int8_t(int((i*17+3)%255)-127);
        v.data[i]=fp8?int8_t(((i*29+11)%127)|((i&2)?128:0)):int8_t(int((i*29+11)%255)-127);
        fk.data[i]=decoded(k.data[i],ks.data[i/32],fp8);fv.data[i]=decoded(v.data[i],vs.data[i/32],fp8);}
    for(size_t i=0;i<query.size;++i){query.data[i]=float(std::sin(i*.13));gate.data[i]=float(std::cos(i*.07));}
    const si::Q8Kv cache{k.data,v.data,ks.data,vs.data,context,kv_heads,fp8};
    for(int tokens=1;tokens<=capacity;++tokens){
        for(int t=0;t<tokens;++t)positions.data[t]=visible-tokens+t;
        si::causal_attention(q,query.data,gate.data,fk.data,fv.data,reference.data,heads,kv_heads,context,visible-tokens,tokens).wait_and_throw();
        const auto verify=[&](bool fused,bool hint){
            std::fill_n(output.data,output.size,guard);std::fill_n(scores.data,scores.size,guard);std::fill_n(partial.data,partial.size,guard);
            if(fused)si::fused_decode_attention_q8(q,query.data,gate.data,cache,scores.data,partial.data,output.data,heads,positions.data,context,nullptr,tokens).wait_and_throw();
            else si::decode_attention_q8(q,query.data,gate.data,cache,scores.data,partial.data,output.data,heads,positions.data,context,nullptr,tokens,hint?visible:0).wait_and_throw();
            float error=0;for(size_t i=0;i<size_t(tokens)*heads*256;++i){const float d=std::abs(output.data[i]-reference.data[i]);error=std::max(error,d);
                check(std::isfinite(output.data[i])&&d<=5e-5f*(1+std::abs(reference.data[i])),"Quantized attention differs from independent decoded FP32 reference");}
            for(size_t i=size_t(tokens)*heads*256;i<output.size;++i)check(output.data[i]==guard,"Inactive MTP output row changed");
            output.guards();scores.guards();partial.guards();++cases;
            std::printf("FP8_KV attention fp8=%d ctx=%d heads=%d tokens=%d fused=%d hint=%d max_abs=%.8g PASS\n",int(fp8),context,heads,tokens,int(fused),int(hint),error);
        };
        verify(false,false);if(context==32768)verify(false,true);if(heads==16)verify(true,false);
        // Independent causal positions, including split boundaries and an
        // early query which must not see the later rows' already stored KV.
        if(context!=32768){
            for(int t=0;t<tokens;++t)positions.data[t]=std::min(context-1,t==0?0:t==1?255:t==2?256:context-1);
            si::decode_attention_q8(q,query.data,gate.data,cache,scores.data,partial.data,output.data,heads,positions.data,context,nullptr,tokens,0).wait_and_throw();
            for(int t=0;t<tokens;++t)for(int h=0;h<heads;++h){const int kh=h/(heads/kv_heads),end=positions.data[t];std::vector<double> weights(size_t(end)+1);double total=0,maximum=-INFINITY;
                for(int cell=0;cell<=end;++cell){double sum=0;for(int col=0;col<256;++col)sum+=double(query.data[(size_t(t)*heads+h)*256+col])*fk.data[(size_t(kh)*context+cell)*256+col];weights[cell]=sum/16;maximum=std::max(maximum,weights[cell]);}
                for(double& w:weights)total+=w=std::exp(w-maximum);
                for(int col=0;col<256;++col){double sum=0;for(int cell=0;cell<=end;++cell)sum+=weights[cell]/total*fv.data[(size_t(kh)*context+cell)*256+col];sum/=1+std::exp(-double(gate.data[(size_t(t)*heads+h)*256+col]));
                    const float actual=output.data[(size_t(t)*heads+h)*256+col];check(std::isfinite(actual)&&std::abs(actual-sum)<=5e-5*(1+std::abs(sum)),"Irregular MTP causal/GQA FP64 reference differs");}
            }++cases;
        }
    }
    k.guards();v.guards();ks.guards();vs.guards();fk.guards();fv.guards();query.guards();gate.guards();positions.guards();
}
}
int main()try{
    sycl::queue q(si::select_device(0),sycl::property::queue::in_order{});
    for(bool fp8:{false,true}){storage(q,fp8);attention(q,257,4,fp8);attention(q,1031,16,fp8);attention(q,32768,16,fp8);}
    std::printf("FP8 KV GPU PASS cases=%d formats=INT8,E4M3FN rows=1..8 GQA causal_tails scale_errors canaries independent_FP32_FP64 INT8_store_bitwise_control\n",cases);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FP8 KV GPU FAIL: %s\n",e.what());return 1;}
