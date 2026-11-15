// Opt-in model-free GPU regression for D256/GQA8 fused INT8 attention.
// Build: cmake --build BUILD --target intel_fused_q8_attention_test
// Run explicitly on a compatible Intel GPU; optional --device N.
// Not registered in the automatic CPU CTest group. No performance claim.
#include "strata/intel/kv_q8.hpp"
#include <stdexcept>
#include <cstdlib>
#include "strata/intel/device.hpp"
#include <algorithm>
#include <cstring>
#include <optional>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace si=strata::intel;
namespace {
constexpr int heads=16,kv_heads=2,width=256,capacity=8;
constexpr float sentinel=123456.f;
template<class T> struct Buffer {
    sycl::queue& q;T* data;size_t size;
    Buffer(sycl::queue& queue,size_t count):q(queue),data(sycl::malloc_shared<T>(count,queue)),size(count){
        if(!data)throw std::runtime_error("Attention fixture allocation failed");
    }
    ~Buffer(){try{q.wait_and_throw();sycl::free(data,q);}catch(...){}}
    Buffer(const Buffer&)=delete;
};
struct Scratch {
    Buffer<float> scores,partial,output;
    Scratch(sycl::queue& q,int context):scores(q,capacity*heads*context+64),
        partial(q,capacity*heads*((context+255)/256)*width+64),output(q,capacity*heads*width+64){}
    void clear(){for(auto* b:{&scores,&partial,&output})std::fill_n(b->data,b->size,sentinel);}
};
constexpr const char* fused_environment="TINYSTRATA_INTEL_FUSED_Q8_ATTN";
struct EnvironmentGuard {
    std::optional<std::string> original;
    EnvironmentGuard(){if(const char* value=std::getenv(fused_environment))original=value;}
    ~EnvironmentGuard(){_putenv_s(fused_environment,original?original->c_str():"");}
};
void environment(const char* value){
    if(_putenv_s(fused_environment,value))throw std::runtime_error("Could not select attention regression mode");
}
void run(sycl::queue& q,bool direct,const float* query,const float* gate,si::Q8Kv cache,
         Scratch& scratch,const int32_t* position,int extent,int tokens,int actual_visible){
    auto done=direct?si::fused_decode_attention_q8(q,query,gate,cache,scratch.scores.data,scratch.partial.data,
        scratch.output.data,heads,position,extent,nullptr,tokens):si::decode_attention_q8(q,query,gate,cache,
        scratch.scores.data,scratch.partial.data,scratch.output.data,heads,position,extent,nullptr,tokens,actual_visible);
    done.wait_and_throw();
}
void identical(const Scratch& a,const Scratch& b,const char* label){
    for(const auto pair:{std::pair{&a.output,&b.output},std::pair{&a.scores,&b.scores},std::pair{&a.partial,&b.partial}})
        if(pair.first->size!=pair.second->size || std::memcmp(pair.first->data,pair.second->data,pair.first->size*sizeof(float)))
            throw std::runtime_error(std::string("Attention dispatcher selected an unexpected path: ")+label);
}
void same(const Buffer<float>& a,const Buffer<float>& b,const char* label){
    if(a.size!=b.size)throw std::runtime_error("Output allocation mismatch");
    for(size_t i=0;i<a.size;++i)
        if(!std::isfinite(a.data[i]) || !std::isfinite(b.data[i]) || std::abs(a.data[i]-b.data[i])>2e-5f*(1.f+std::abs(a.data[i])))
            throw std::runtime_error(std::string("Current/candidate numeric mismatch: ")+label);
}
void guards(const Scratch& scratch,const int32_t* position,int context,int extent,int tokens,bool candidate=false){
    const size_t rows=size_t(tokens)*heads,splits=(context+255)/256,visible=(extent+255)/256;
    for(size_t i=rows*width;i<scratch.output.size;++i)if(scratch.output.data[i]!=sentinel)
        throw std::runtime_error("KV output inactive row/tail changed");
    for(size_t row=0;row<size_t(capacity)*heads;++row)for(int cell=0;cell<context;++cell)
        if((row>=rows || (candidate?size_t(cell)>=2*((size_t(position[row/heads]+1)+255)/256):cell>position[row/heads]))
            && scratch.scores.data[row*context+cell]!=sentinel)
            throw std::runtime_error("KV scores future/inactive row changed");
    for(size_t row=0;row<size_t(capacity)*heads;++row)for(size_t split=0;split<splits;++split)
        if(row>=rows || split>=(candidate?(size_t(position[row/heads]+1)+255)/256:visible))for(int col=0;col<width;++col)
            if(scratch.partial.data[(row*splits+split)*width+col]!=sentinel)
                throw std::runtime_error("KV partial inactive row/split changed");
    for(size_t i=size_t(capacity)*heads*context;i<scratch.scores.size;++i)if(scratch.scores.data[i]!=sentinel)
        throw std::runtime_error("KV scores tail changed");
    for(size_t i=size_t(capacity)*heads*splits*width;i<scratch.partial.size;++i)if(scratch.partial.data[i]!=sentinel)
        throw std::runtime_error("KV partial tail changed");
}
double fp64(const Scratch& result,const float* query,const float* gate,si::Q8Kv cache,
            const int32_t* position,int tokens){
    double maximum_error=0;
    for(int token=0;token<tokens;++token)for(int head=0;head<heads;++head){
        const size_t row=size_t(token)*heads+head;const int kh=head/(heads/kv_heads),end=position[token];
        std::vector<double> probabilities(size_t(end)+1,0.);
        for(int cell=0;cell<=end;++cell)for(int col=0;col<width;++col){
            const size_t i=(size_t(kh)*cache.context+cell)*width+col;
            probabilities[cell]+=double(query[row*width+col])*double(cache.key[i])*float(cache.key_scale[i/32])/16.;
        }
        const double maximum=*std::max_element(probabilities.begin(),probabilities.end());double total=0;
        for(auto& p:probabilities)total+=p=std::exp(p-maximum);
        for(int col=0;col<width;++col){
            double wanted=0;
            for(int cell=0;cell<=end;++cell){
                const size_t i=(size_t(kh)*cache.context+cell)*width+col;
                wanted+=probabilities[cell]/total*double(cache.value[i])*float(cache.value_scale[i/32]);
            }
            wanted/=1.+std::exp(-double(gate[row*width+col]));
            const float actual=result.output.data[row*width+col];const double error=std::abs(double(actual)-wanted);
            if(!std::isfinite(actual)||error>2e-5*(1+std::abs(wanted)))throw std::runtime_error("KV independent FP64 mismatch");
            maximum_error=std::max(maximum_error,error);
        }
    }
    return maximum_error;
}
void store_error_test(sycl::queue& q){
    Buffer<int8_t> k(q,kv_heads*width),v(q,kv_heads*width);
    Buffer<sycl::half> ks(q,kv_heads*width/32),vs(q,kv_heads*width/32);
    Buffer<float> key(q,kv_heads*width),value(q,kv_heads*width);
    Buffer<int32_t> position(q,1),error(q,1);position.data[0]=0;
    const si::Q8Kv cache{k.data,v.data,ks.data,vs.data,1,kv_heads};
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),1e30f}){
        std::fill_n(key.data,key.size,0.f);std::fill_n(value.data,value.size,0.f);key.data[35]=bad;error.data[0]=0;
        si::store_q8_kv(q,key.data,value.data,cache,position.data,1,error.data).wait_and_throw();
        if(error.data[0]!=1)throw std::runtime_error("Existing Q8 store lost nonfinite/scale-overflow error flag");
        for(int col=32;col<64;++col)if(k.data[col]!=0 || float(ks.data[col/32])!=0.f)
            throw std::runtime_error("Invalid Q8 store block was not zeroed");
    }
    std::puts("{\"store_nonfinite_overflow_error_policy\":true}");
}
}
int main(int argc,char** argv){
    try{
        unsigned selected=0;
        if(argc==3 && std::string(argv[1])=="--device") {
            const std::string text(argv[2]);
            if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos || std::stoul(text)>1024)
                throw std::invalid_argument("Device must be an integer from0 through1024");
            selected=unsigned(std::stoul(text));
        } else if(argc!=1)throw std::invalid_argument("Usage: intel_fused_q8_attention_test [--device N]");
        const auto device=si::select_device(selected);const auto info=si::inspect_device(device);
        if(const char* limitation=si::baseline_limitation(info.capabilities))throw std::runtime_error(limitation);
        sycl::queue q(device,sycl::property_list{sycl::property::queue::in_order{}});
        std::printf("Device vendor=%04x device=%04x name=%s architecture=%s\n",info.vendor_id,info.device_id,info.name.c_str(),info.architecture.c_str());
        EnvironmentGuard restore_environment;
        store_error_test(q);
        const std::vector<int> contexts={255,256,511,512,1031,32768};
        int numerical_cases=0;
        for(int context:contexts){
            const size_t cache_size=size_t(kv_heads)*context*width;
            Buffer<int8_t> keys(q,cache_size),values(q,cache_size);
            Buffer<sycl::half> key_scales(q,cache_size/32),value_scales(q,cache_size/32);
            Buffer<float> query(q,capacity*heads*width),gate(q,capacity*heads*width);
            // Tight row-count allocations in every case also catch candidate
            // reads of nonexistent position rows for the T3 specialization.
            for(size_t i=0;i<cache_size;++i){keys.data[i]=int8_t(int((i*17+3)%255)-127);values.data[i]=int8_t(int((i*29+11)%255)-127);}
            for(size_t i=0;i<cache_size/32;++i){key_scales.data[i]=sycl::half(float(i%11+1)/1024);value_scales.data[i]=sycl::half(float(i%7+1)/256);}
            for(size_t i=0;i<query.size;++i){query.data[i]=float(std::sin(i*.13));gate.data[i]=float(std::cos(i*.07));}
            const si::Q8Kv cache{keys.data,values.data,key_scales.data,value_scales.data,context,kv_heads};
            const std::vector<int> extents=context==32768?std::vector<int>{16384,16512}:std::vector<int>{context};
            for(int visible:extents){
            Scratch current(q,context),candidate(q,context),dispatched(q,context);
            const std::vector<int> numerical_tokens={1,3,4,8};
            for(int tokens:numerical_tokens)for(int pattern=0;pattern<3;++pattern){
                Buffer<int32_t> positions(q,tokens);
                const int extent=visible,active=pattern==2?std::min(visible,513):visible;
                for(int t=0;t<tokens;++t)positions.data[t]=pattern==1?std::min(active-1,t==0?0:t==1?255:t==2?256:active-1):active-tokens+t;
                current.clear();candidate.clear();
                const int actual_visible=*std::max_element(positions.data,positions.data+tokens)+1;
                // Explicit0 retains the prior four-stage implementation even
                // after fusion becomes a product default. It includes the
                // existing16K T3/T4 query-token reuse optimization.
                environment("0");
                run(q,false,query.data,gate.data,cache,current,positions.data,extent,tokens,actual_visible);
                run(q,true,query.data,gate.data,cache,candidate,positions.data,extent,tokens,actual_visible);
                same(current.output,candidate.output,"output");
                guards(current,positions.data,context,extent,tokens);guards(candidate,positions.data,context,extent,tokens,true);
                const double error=std::max(fp64(current,query.data,gate.data,cache,positions.data,tokens),fp64(candidate,query.data,gate.data,cache,positions.data,tokens));
                environment("1");
                dispatched.clear();
                run(q,false,query.data,gate.data,cache,dispatched,positions.data,extent,tokens,actual_visible);
                const bool eligible=actual_visible>=16384;
                identical(dispatched,eligible?candidate:current,"explicit1 admission");
                guards(dispatched,positions.data,context,extent,tokens,eligible);
                environment("");
                dispatched.clear();
                run(q,false,query.data,gate.data,cache,dispatched,positions.data,extent,tokens,actual_visible);
                const bool default_fused=eligible && info.ptl_h_aot && info.capabilities.esimd &&
                    info.capabilities.subgroup32 && info.capabilities.max_work_group>=128;
                identical(dispatched,default_fused?candidate:current,"actual-device default admission");
                guards(dispatched,positions.data,context,extent,tokens,default_fused);
                // Rounded graph extent must not be mistaken for actual
                // visibility: zero hints retain the original implementation.
                dispatched.clear();
                run(q,false,query.data,gate.data,cache,dispatched,positions.data,extent,tokens,0);
                identical(dispatched,current,"zero-hint graph fallback");
                guards(dispatched,positions.data,context,extent,tokens);
                ++numerical_cases;
                std::printf("{\"numeric\":true,\"context\":%d,\"extent\":%d,\"tokens\":%d,\"pattern\":%d,\"numeric_tolerance\":true,\"canary\":true,\"max_abs_fp64\":%.12g}\n",context,extent,tokens,pattern,error);std::fflush(stdout);
            }
            }
        }
        if(numerical_cases!=84)throw std::runtime_error("Incomplete fused attention regression matrix");
        std::puts("Fused INT8 attention84 numerical/FP64/causal/tail/dispatcher checks passed; store error policy retained");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"Fused INT8 attention regression failed: %s\n",error.what());return 1;}
}
