// Explicit opt-in GPU test. Both implementations are compared independently
// with an FP64 formula over the original Q8 codes; no model or API is started.
#include "strata/intel/gemma_ops.hpp"
#include "strata/intel/device.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace si = strata::intel;
namespace {
size_t checks=0;
double maximum_error=0;
void require(bool condition,const char* text){++checks;if(!condition)throw std::runtime_error(text);}
void close(float value,double expected){
    ++checks;const double error=std::abs(double(value)-expected);
    maximum_error=std::max(maximum_error,error);
    if(!std::isfinite(value)||error>1e-5*(1+std::abs(expected)))
        throw std::runtime_error("PV split independent FP64 reference mismatch");
}
template<class T>struct DeviceBuffer{
    sycl::queue& q;size_t count;std::vector<T> host;T* allocation;T* data;
    DeviceBuffer(sycl::queue& queue,size_t size):q(queue),count(size),host(size+32,T(7)),
        allocation(sycl::malloc_device<T>(size+32,q)),data(allocation?allocation+16:nullptr){
        if(!allocation)throw std::runtime_error("PV split fixture USM allocation failed");
    }
    ~DeviceBuffer(){try{q.wait_and_throw();sycl::free(allocation,q);}catch(...){}}
    DeviceBuffer(const DeviceBuffer&)=delete;
    T& at(size_t i){return host[i+16];}
    const T& at(size_t i)const{return host[i+16];}
    void upload(){q.memcpy(allocation,host.data(),host.size()*sizeof(T)).wait_and_throw();}
    void download(){q.memcpy(host.data(),allocation,host.size()*sizeof(T)).wait_and_throw();}
    void guards()const{for(size_t i=0;i<16;++i)require(host[i]==T(7)&&host[count+16+i]==T(7),"PV split canary overwritten");}
    void unchanged(){
        std::vector<T> copy(host.size());q.memcpy(copy.data(),allocation,copy.size()*sizeof(T)).wait_and_throw();
        require(std::memcmp(copy.data(),host.data(),copy.size()*sizeof(T))==0,"PV split changed a read-only input");
    }
};
struct Shape{int dim,heads,kv,window,position,tokens;float scale;int distribution;};
struct Measurement{Shape shape;double control_ms=0,candidate_ms=0;size_t elements=0;bool active=false;};

Measurement test(sycl::queue& q,Shape s,bool timing){
    const int context=std::max(32768,s.position+s.tokens),capacity=s.window?std::min(context,s.window+512):context;
    const int first=s.window?std::max(0,s.position-s.window+1):0,span=s.position+s.tokens-first;
    const size_t cells=size_t(s.kv)*capacity*s.dim,n=size_t(s.tokens)*s.heads*s.dim;
    DeviceBuffer<int8_t> k(q,cells),v(q,cells);
    DeviceBuffer<sycl::half> ks(q,cells/32),vs(q,cells/32);
    DeviceBuffer<float> query(q,n),control(q,n),candidate(q,n);
    for(int logical=first;logical<s.position+s.tokens;++logical)for(int h=0;h<s.kv;++h)for(int col=0;col<s.dim;++col){
        const size_t at=(size_t(h)*capacity+logical%capacity)*s.dim+col;
        k.at(at)=int8_t((logical*3+h*5+col*7)%255-127);
        v.at(at)=int8_t((logical*11+h*13+col*3)%255-127);
        ks.at(at/32)=sycl::half(.002f*float(1+(logical+h+col/32)%3));
        vs.at(at/32)=sycl::half(.003f*float(1+(logical+h+col/32)%5));
    }
    for(size_t i=0;i<n;++i)query.at(i)=s.distribution==1?0.f:float(std::sin(i*.037)*(.04*(s.distribution==2?80:1)));
    k.upload();v.upload();ks.upload();vs.upload();query.upload();control.upload();candidate.upload();
    si::GemmaKv cache{k.data,v.data,ks.data,vs.data,capacity,s.kv,s.dim,s.window};
    // Both scratch sets are bounded by the same context. Only the candidate
    // has an additional 4 MiB at 32K or 32 MiB at 256K for real Gemma geometry.
    si::GemmaAttention base(q,context,s.tokens,s.heads,s.dim,cells,false,false);
    si::GemmaAttention split(q,context,s.tokens,s.heads,s.dim,cells,false,true);
    base.run(query.data,cache,control.data,s.heads,s.position,s.tokens,s.scale).wait_and_throw();
    split.run(query.data,cache,candidate.data,s.heads,s.position,s.tokens,s.scale).wait_and_throw();
    control.download();candidate.download();
    // Every output, each causal tail, all GQA heads, and all value columns are
    // independently evaluated. Using the original host codes also catches
    // wrong ring indexing and accidental future-token visibility.
    for(int token=0;token<s.tokens;++token)for(int head=0;head<s.heads;++head){
        const int end=s.position+token,low=s.window?std::max(0,end-s.window+1):0,kv=head/(s.heads/s.kv);
        std::vector<double> probabilities(size_t(end-low+1));double maximum=-INFINITY;
        for(int logical=low;logical<=end;++logical){
            double dot=0;
            for(int col=0;col<s.dim;++col){
                const size_t at=(size_t(kv)*capacity+logical%capacity)*s.dim+col;
                dot+=double(query.at((size_t(token)*s.heads+head)*s.dim+col))*double(k.at(at))*float(ks.at(at/32));
            }
            const double score=dot*s.scale;probabilities[size_t(logical-low)]=score;maximum=std::max(maximum,score);
        }
        double total=0;for(double& score:probabilities){score=std::exp(score-maximum);total+=score;}
        for(int col=0;col<s.dim;++col){
            double expected=0;
            for(int logical=low;logical<=end;++logical){
                const size_t at=(size_t(kv)*capacity+logical%capacity)*s.dim+col;
                expected+=probabilities[size_t(logical-low)]/total*double(v.at(at))*float(vs.at(at/32));
            }
            const size_t out=(size_t(token)*s.heads+head)*s.dim+col;
            close(control.at(out),expected);close(candidate.at(out),expected);close(candidate.at(out),control.at(out));
        }
    }
    k.unchanged();v.unchanged();ks.unchanged();vs.unchanged();query.unchanged();
    control.guards();candidate.guards();
    Measurement result{s,0,0,n,s.tokens==1&&span>=1024};
    if(timing&&s.tokens==1&&s.distribution==0){
        auto run=[&](bool alternate){auto& a=alternate?split:base;float* out=alternate?candidate.data:control.data;
            const auto start=std::chrono::steady_clock::now();
            a.run(query.data,cache,out,s.heads,s.position,s.tokens,s.scale).wait_and_throw();
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
        for(int i=0;i<3;++i){run(false);run(true);}
        // Alternating ABBA prevents all control work preceding all treatment
        // work. Host-submit-to-completion timing is separate from correctness.
        constexpr int repeats=8;
        for(int i=0;i<repeats;++i){
            if(i%2){result.candidate_ms+=run(true);result.control_ms+=run(false);}
            else{result.control_ms+=run(false);result.candidate_ms+=run(true);}
        }
        result.control_ms/=repeats;result.candidate_ms/=repeats;
    }
    std::printf("PASS D%d H%d KV%d W%d P%d T%d scale%.3f dist%d split_active%d all%zu control%.6fms candidate%.6fms\n",
        s.dim,s.heads,s.kv,s.window,s.position,s.tokens,s.scale,s.distribution,int(result.active),n,result.control_ms,result.candidate_ms);
    std::fflush(stdout);return result;
}
}
int main(int argc,char** argv){try{
    bool timing=true;std::string json;
    for(int i=1;i<argc;++i){const std::string a=argv[i];if(a=="--no-timing")timing=false;else if(a=="--json"&&i+1<argc)json=argv[++i];else throw std::invalid_argument("Use [--no-timing] [--json FILE]");}
    auto device=si::select_device(0);sycl::queue q(device,sycl::property::queue::in_order{});
    std::printf("device=%s; no model loaded; split reduction default remains off\n",device.get_info<sycl::info::device::name>().c_str());
    std::vector<Measurement> results;
    const Shape shapes[]={
        {512,16,2,0,1022,1,1.f,0},{512,16,2,0,1023,1,1.f,0},
        {512,16,2,0,1024,1,1.f,0},{512,16,2,0,4095,1,1.f,0},
        {512,16,2,0,16383,1,1.f,0},{512,16,2,0,32767,1,1.f,0},
        {256,16,8,1024,1023,1,1.f,0},{256,16,8,1024,1024,1,1.f,0},
        {256,16,8,1024,4097,1,1.f,0},{256,16,8,1024,32767,1,1.f,0},
        {256,16,8,1024,1025,3,1.f,0},{256,16,8,1024,4097,8,1.f,0},
        {512,16,2,0,1024,3,1.f,0},{256,16,8,1024,4097,1,1.f,1},
        {512,16,2,0,1024,1,1.f,2},{64,32,8,0,1024,1,.125f,0},
        // Each test owns its buffers only until its return: full 128K/256K
        // cases do not remain resident together. The largest case stays
        // within the actual Gemma 256K scratch/cache allocation geometry.
        {512,16,2,0,131071,1,1.f,0},
        {512,16,2,0,262143,1,1.f,0},
        {512,16,2,0,131069,3,1.f,0},
        // Late logical positions wrap the 1536-entry SWA ring repeatedly;
        // multi-token tails stop exactly at the trained allocation ceiling.
        {256,16,8,1024,131071,1,1.f,0},
        {256,16,8,1024,262143,1,1.f,0},
        {256,16,8,1024,131069,3,1.f,0},
        {256,16,8,1024,262136,8,1.f,0}};
    for(const auto& s:shapes)results.push_back(test(q,s,timing));q.wait_and_throw();
    if(!json.empty()){
        std::ofstream out(json);if(!out)throw std::runtime_error("Cannot save split fixture JSON");
        out.precision(12);out<<"{\"passed\":true,\"model_loaded\":false,\"checks\":"<<checks<<",\"maximum_absolute_error\":"<<maximum_error<<",\"timing\":\"alternating host submit-to-completion, eight samples per path after three warmups\",\"shapes\":[";
        for(size_t i=0;i<results.size();++i){const auto& r=results[i];const auto& s=r.shape;if(i)out<<',';
            out<<"{\"dim\":"<<s.dim<<",\"heads\":"<<s.heads<<",\"kv_heads\":"<<s.kv<<",\"window\":"<<s.window<<",\"position\":"<<s.position<<",\"tokens\":"<<s.tokens<<",\"scale\":"<<s.scale<<",\"distribution\":"<<s.distribution<<",\"candidate_active\":"<<(r.active?"true":"false")<<",\"output_elements\":"<<r.elements<<",\"control_mean_ms\":"<<r.control_ms<<",\"candidate_mean_ms\":"<<r.candidate_ms<<'}';
        }
        out<<"]}\n";
    }
    std::printf("PASS %zu independent FP64/comparison/canary checks; max_abs_error=%.9g\n",checks,maximum_error);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}}
