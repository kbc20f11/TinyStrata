// Exact eager CPU-row compact combine against the unchanged scatter+combine.
// Independent poisoned inputs/outputs, CPU FP64 weighted oracle, Graph replay.
// Compile with the engine's arithmetic flags (CUDA fast math, normal HIP math).
// Optional --bench reports device-only warm Graph ABBA, excluding host uploads.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: compact CPU combine requires CUDA or HIP"); return 77; }
#else
#include "q36_kernels.cuh"
#include "strata/qwen35/hybrid_mapped_output.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { const auto e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2); } } while(0)
constexpr int N=2048, TOP=8, MAXT=8, ENTRIES=MAXT*TOP, GUARD=16;
constexpr float EPS=1e-6f;
#if defined(__HIPCC__)
constexpr auto CAPTURE_MODE=hipStreamCaptureModeGlobal;
#else
constexpr auto CAPTURE_MODE=cudaStreamCaptureModeGlobal;
#endif
struct Device {
    void* p=nullptr; size_t bytes;
    explicit Device(size_t n):bytes(n) { CK(cudaMalloc(&p,n)); }
    ~Device() { cudaFree(p); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void poison(cudaStream_t s) const { CK(cudaMemsetAsync(p,0xff,bytes,s)); }
};
struct Arm {
    Device x{size_t(MAXT*N+GUARD)*4}, h{size_t(MAXT*N+GUARD)*4};
    Device q{size_t(MAXT*(N/32))*sizeof(q36k::QB)+GUARD*4};
    Device parts{size_t(ENTRIES*N+GUARD)*4}, cpu{size_t(ENTRIES*N+GUARD)*4};
    Device map{size_t(ENTRIES+GUARD)*4};
};
struct Inputs {
    Device weights{size_t(ENTRIES)*4}, shared{size_t(MAXT*N)*4};
    Device sg{size_t(MAXT*257)*4}, norm{size_t(N)*4};
};
// Frozen reference from q36_engine.cu: the CPU rows were indexed by entry.
__global__ void old_merge(float* parts,const float* cpu,const int32_t* miss,int n) {
    const int e=blockIdx.x;
    if(!miss[e])return;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)parts[size_t(e)*n+i]=cpu[size_t(e)*n+i];
}
void launch(bool compact,Arm& a,const Inputs& in,int T,int mode,int sg_stride,cudaStream_t s) {
    const float* wn=mode ? in.norm.as<float>() : nullptr;
    auto* q=mode==2 ? a.q.as<q36k::QB>() : nullptr;
    if(compact)
        q36k::k_moe_out_cpu_small<<<T,1024,0,s>>>(a.x.as<float>(),a.parts.as<float>(),a.cpu.as<float>(),a.map.as<int32_t>(),
            nullptr,in.weights.as<float>(),TOP,in.shared.as<float>(),in.sg.as<float>(),sg_stride,wn,a.h.as<float>(),N,EPS,q);
    else {
        old_merge<<<T*TOP,256,0,s>>>(a.parts.as<float>(),a.cpu.as<float>(),a.map.as<int32_t>(),N);
        q36k::k_moe_out_small<<<T,1024,0,s>>>(a.x.as<float>(),a.parts.as<float>(),nullptr,in.weights.as<float>(),TOP,
            in.shared.as<float>(),in.sg.as<float>(),sg_stride,wn,a.h.as<float>(),N,EPS,q);
    }
    CK(cudaGetLastError());
}
[[noreturn]] void fail(const char* what,int T,int count,int pattern,int mode,size_t word=0) {
    std::fprintf(stderr,"FAIL %s T=%d cpu_per_token=%d pattern=%d mode=%d word=%zu\n",what,T,count,pattern,mode,word);
    std::exit(1);
}
std::vector<uint32_t> read_words(const Device& d,cudaStream_t s) {
    std::vector<uint32_t> out(d.bytes/4);
    CK(cudaMemcpyAsync(out.data(),d.p,d.bytes,cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));return out;
}
void check_equal(const Device& a,const Device& b,int T,int count,int pattern,int mode,const char* name,cudaStream_t s) {
    auto x=read_words(a,s),y=read_words(b,s);
    for(size_t i=0;i<x.size();++i)if(x[i]!=y[i])fail(name,T,count,pattern,mode,i);
}
void check_poison_after(const Device& d,size_t used,int T,int count,int pattern,int mode,const char* name,cudaStream_t s) {
    auto words=read_words(d,s);
    for(size_t i=used/4;i<words.size();++i)if(words[i]!=0xffffffffu)fail(name,T,count,pattern,mode,i);
}
void compare(Arm& ref,Arm& cand,int T,int count,int pattern,int mode,const std::vector<float>& parts,cudaStream_t s) {
    check_equal(ref.x,cand.x,T,count,pattern,mode,"residual",s);
    check_equal(ref.h,cand.h,T,count,pattern,mode,"normalized",s);
    check_equal(ref.q,cand.q,T,count,pattern,mode,"q8_1",s);
    for(auto* a:{&ref,&cand}) {
        check_poison_after(a->x,size_t(T)*N*4,T,count,pattern,mode,"residual guard",s);
        check_poison_after(a->h,mode?size_t(T)*N*4:0,T,count,pattern,mode,"norm guard",s);
        check_poison_after(a->q,mode==2?size_t(T)*(N/32)*sizeof(q36k::QB):0,T,count,pattern,mode,"q8 guard",s);
    }
    auto unchanged=read_words(cand.parts,s);
    if(std::memcmp(unchanged.data(),parts.data(),size_t(T)*TOP*N*4))fail("candidate modified parts",T,count,pattern,mode);
    for(size_t i=size_t(T)*TOP*N;i<unchanged.size();++i)
        if(unchanged[i]!=0xffffffffu)fail("parts guard",T,count,pattern,mode,i);
}
void graph_bench(cudaGraphExec_t old,cudaGraphExec_t direct,int T,int cpu_count,cudaStream_t s) {
    constexpr int CALLS=256;
    cudaEvent_t begin{},end{};CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    double totals[2]{};
    for(int arm:{0,1,1,0}) {
        const auto graph=arm?direct:old;
        for(int i=0;i<16;++i)CK(cudaGraphLaunch(graph,s));CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(begin,s));for(int i=0;i<CALLS;++i)CK(cudaGraphLaunch(graph,s));
        CK(cudaEventRecord(end,s));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,begin,end));totals[arm]+=ms;
    }
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
    std::printf("WARM_DEVICE_GRAPH_ABBA T=%d cpu_per_token=%d old_us=%.3f direct_us=%.3f calls=2x%d excludes_H2D=1\n",
        T,cpu_count,totals[0]*1000/(2*CALLS),totals[1]*1000/(2*CALLS),CALLS);
}

// Production keeps its existing per-layer route event as the lifetime bound.
// This fixture uses an event after both combines before rewriting the same
// mapped rows/map. Repeated changing payloads catch stale host-memory reads.
struct Mapped {
    void *host=nullptr,*device=nullptr;
    explicit Mapped(size_t bytes) {
        const char* reason="not attempted";
        if(!strata::qwen35::hybrid_mapped_output::allocate(&host,&device,bytes,reason))
            std::printf("MAPPED_UNAVAILABLE %s\n",reason);
    }
    ~Mapped() {if(host)cudaFreeHost(host);}
};
void mapped_launch(Arm& a,const Inputs& in,int T,int mode,const float* cpu,const int32_t* map,cudaStream_t s) {
    q36k::k_moe_out_cpu_small<<<T,1024,0,s>>>(a.x.as<float>(),a.parts.as<float>(),cpu,map,nullptr,
        in.weights.as<float>(),TOP,in.shared.as<float>(),in.sg.as<float>(),1,
        mode?in.norm.as<float>():nullptr,a.h.as<float>(),N,EPS,mode==2?a.q.as<q36k::QB>():nullptr);
    CK(cudaGetLastError());
}
int mapped_test(cudaStream_t s,bool bench) {
    Mapped host_cpu(size_t(ENTRIES*N+GUARD)*4),host_map(size_t(ENTRIES+GUARD)*4);
    if(!host_cpu.device||!host_map.device) {
        std::puts("SKIP: coherent mapped CPU output unavailable; production retains DMA");
        return 77;
    }
    Arm ref,cand;Inputs in;
    std::vector<float> initial(MAXT*N),gpu(ENTRIES*N),packed(ENTRIES*N),weights(ENTRIES);
    std::vector<float> shared(MAXT*N),norm(N),gates(MAXT*257);
    for(size_t i=0;i<initial.size();++i)initial[i]=float(int(i%97)-48)/1024.f;
    for(size_t i=0;i<gpu.size();++i)gpu[i]=float(int((i*17)%251)-125)/512.f;
    for(size_t i=0;i<shared.size();++i)shared[i]=float(int(i%31)-15)/512.f;
    for(size_t i=0;i<norm.size();++i)norm[i]=1.f+float(int(i%13)-6)/32.f;
    for(size_t i=0;i<weights.size();++i)weights[i]=float(i%TOP+1)/36.f;
    for(size_t i=0;i<gates.size();++i)gates[i]=float(int(i%19)-9)/32.f;
    CK(cudaMemcpy(in.weights.p,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(in.shared.p,shared.data(),shared.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(in.norm.p,norm.data(),norm.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(in.sg.p,gates.data(),gates.size()*4,cudaMemcpyHostToDevice));
    cudaEvent_t consumed{};CK(cudaEventCreateWithFlags(&consumed,cudaEventDisableTiming));
    int checks=0;
    for(int T:{1,2,4,8})for(int count:{0,1,4,8})for(int mode=0;mode<3;++mode) {
        for(int repeat=0;repeat<4;++repeat) {
            const int entries=T*TOP;
            std::vector<int32_t> map(entries,-1);
            std::vector<float> parts(gpu.begin(),gpu.begin()+size_t(entries)*N);
            int missed=0;
            for(int e=0;e<entries;++e) {
                const int per_token=repeat==3?(e/TOP)%9:count;
                if((e*3+repeat)%TOP>=per_token)continue;
                map[e]=missed;
                for(int i=0;i<N;++i)
                    packed[size_t(missed)*N+i]=float(int((size_t(e)*N+i+repeat*337)%263)-131)/256.f;
                std::memset(parts.data()+size_t(e)*N,0xff,N*4);
                ++missed;
            }
            std::memset(host_cpu.host,0xff,size_t(ENTRIES*N+GUARD)*4);
            std::memset(host_map.host,0xff,size_t(ENTRIES+GUARD)*4);
            std::memcpy(host_cpu.host,packed.data(),size_t(missed)*N*4);
            std::memcpy(host_map.host,map.data(),size_t(entries)*4);
            strata::qwen35::hybrid_mapped_output::publish_host_writes();
            for(auto* a:{&ref,&cand}) {
                for(auto* d:{&a->x,&a->h,&a->q,&a->parts,&a->cpu,&a->map})d->poison(s);
                CK(cudaMemcpyAsync(a->x.p,initial.data(),size_t(T)*N*4,cudaMemcpyHostToDevice,s));
                CK(cudaMemcpyAsync(a->parts.p,parts.data(),size_t(entries)*N*4,cudaMemcpyHostToDevice,s));
            }
            if(missed)CK(cudaMemcpyAsync(ref.cpu.p,host_cpu.host,size_t(missed)*N*4,cudaMemcpyHostToDevice,s));
            CK(cudaMemcpyAsync(ref.map.p,host_map.host,size_t(entries)*4,cudaMemcpyHostToDevice,s));
            mapped_launch(ref,in,T,mode,ref.cpu.as<float>(),ref.map.as<int32_t>(),s);
            mapped_launch(cand,in,T,mode,static_cast<float*>(host_cpu.device),static_cast<int32_t*>(host_map.device),s);
            CK(cudaEventRecord(consumed,s));CK(cudaEventSynchronize(consumed));
            compare(ref,cand,T,count,repeat,mode,parts,s);
            if(std::memcmp(host_cpu.host,packed.data(),size_t(missed)*N*4)||
               std::memcmp(host_map.host,map.data(),size_t(entries)*4))fail("mapped host input changed",T,count,repeat,mode);
            const auto* cpu_words=static_cast<const uint32_t*>(host_cpu.host);
            const auto* map_words=static_cast<const uint32_t*>(host_map.host);
            for(size_t i=size_t(missed)*N;i<ENTRIES*N+GUARD;++i)
                if(cpu_words[i]!=0xffffffffu)fail("mapped CPU guard",T,count,repeat,mode,i);
            for(size_t i=entries;i<ENTRIES+GUARD;++i)
                if(map_words[i]!=0xffffffffu)fail("mapped map guard",T,count,repeat,mode,i);
            ++checks;
        }
    }
    if(bench)for(int T:{1,4}) {
        constexpr int CALLS=32;
        const int entries=T*TOP,missed=T*4;
        auto* map=static_cast<int32_t*>(host_map.host);
        for(int e=0;e<entries;++e)map[e]=e%TOP<4?(e/TOP)*4+e%TOP:-1;
        for(int i=0;i<missed*N;++i)static_cast<float*>(host_cpu.host)[i]=float(i%37-18)/256.f;
        strata::qwen35::hybrid_mapped_output::publish_host_writes();
        for(auto* a:{&ref,&cand}) {
            CK(cudaMemcpyAsync(a->x.p,initial.data(),size_t(T)*N*4,cudaMemcpyHostToDevice,s));
            CK(cudaMemcpyAsync(a->parts.p,gpu.data(),size_t(entries)*N*4,cudaMemcpyHostToDevice,s));
        }
        CK(cudaStreamSynchronize(s));
        double totals[2]{};
        for(int arm:{0,1,1,0}) {
            auto one=[&] {
                if(!arm) {
                    CK(cudaMemcpyAsync(ref.cpu.p,host_cpu.host,size_t(missed)*N*4,cudaMemcpyHostToDevice,s));
                    CK(cudaMemcpyAsync(ref.map.p,host_map.host,size_t(entries)*4,cudaMemcpyHostToDevice,s));
                }
                mapped_launch(arm?cand:ref,in,T,2,arm?static_cast<float*>(host_cpu.device):ref.cpu.as<float>(),
                    arm?static_cast<int32_t*>(host_map.device):ref.map.as<int32_t>(),s);
                CK(cudaEventRecord(consumed,s));CK(cudaEventSynchronize(consumed));
            };
            for(int i=0;i<4;++i)one();
            const auto begin=std::chrono::steady_clock::now();
            for(int i=0;i<CALLS;++i)one();
            totals[arm]+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
        }
        std::printf("MAPPED_EAGER_ROUNDTRIP_ABBA T=%d cpu_per_token=4 dma_us=%.3f mapped_us=%.3f calls=2x%d includes_H2D=1\n",
            T,totals[0]/(2*CALLS),totals[1]/(2*CALLS),CALLS);
    }
    CK(cudaEventDestroy(consumed));
    std::printf("PASS %d coherent mapped residual/norm/Q8 checks; repeated payload/map rewrites and guards\n",checks);
    return 0;
}
int main(int argc,char** argv) {
    const bool bench=argc==2&&std::strcmp(argv[1],"--bench")==0;
    const bool mapped=argc==2&&(std::strcmp(argv[1],"--mapped")==0||std::strcmp(argv[1],"--mapped-bench")==0);
    if(argc>1&&!bench&&!mapped){std::fputs("usage: q36-cpu-combine [--bench|--mapped|--mapped-bench]\n",stderr);return 2;}
    cudaStream_t s{};CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    if(mapped) {
        const int result=mapped_test(s,std::strcmp(argv[1],"--mapped-bench")==0);
        CK(cudaStreamDestroy(s));return result;
    }
    Arm ref,cand;Inputs in;
    uint32_t random=0x31415926u;
    const auto next=[&]() {random=random*1664525u+1013904223u;return random;};
    const auto value=[&]() {return float(int(next()%2001)-1000)/2048.f;};
    std::vector<float> initial(MAXT*N),shared(MAXT*N),norm(N),weights(ENTRIES),gates(MAXT*257);
    std::vector<float> gpu(ENTRIES*N),cpu(ENTRIES*N),packed(ENTRIES*N);
    for(auto* v:{&initial,&shared,&norm,&gates,&gpu,&cpu})for(auto& x:*v)x=value();
    for(auto& x:norm)x+=1.f;
    for(int t=0;t<MAXT;++t) {
        float sum=0;for(int k=0;k<TOP;++k)sum+=(weights[t*TOP+k]=float(1+next()%32));
        for(int k=0;k<TOP;++k)weights[t*TOP+k]/=sum;
    }
    CK(cudaMemcpy(in.shared.p,shared.data(),shared.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(in.norm.p,norm.data(),norm.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(in.weights.p,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(in.sg.p,gates.data(),gates.size()*4,cudaMemcpyHostToDevice));
    int checks=0;double max_cpu_abs=0;
    for(int T=1;T<=MAXT;++T)for(int count=0;count<=TOP;++count)for(int pattern=0;pattern<4;++pattern) {
        // One extra case mixes empty, full and partial CPU rows across tokens,
        // exercising packed offsets which uniform per-token miss counts miss.
        // It does not need repeating for every uniform-count case.
        if(pattern==3&&count!=0)continue;
        const int entries=T*TOP,sg_stride=pattern==1?257:1;
        std::vector<int32_t> flags(entries),map(entries,-1);std::vector<float> parts(gpu.begin(),gpu.begin()+size_t(entries)*N);
        int missed=0;
        for(int e=0;e<entries;++e) {
            const int t=e/TOP,k=e%TOP;
            const int perm=pattern==0?k:pattern==1?(k+t)%TOP:(3*k+5*t)%TOP;
            constexpr int irregular_counts[MAXT]={0,TOP,1,TOP-1,2,TOP-2,3,TOP-3};
            const int token_count=pattern==3?irregular_counts[t]:count;
            flags[e]=perm<token_count;
            if(flags[e]) {
                map[e]=missed;std::memcpy(packed.data()+size_t(missed)*N,cpu.data()+size_t(e)*N,N*4);
                std::memset(parts.data()+size_t(e)*N,0xff,N*4);++missed;
            }
        }
        for(int mode=0;mode<3;++mode) {
            const auto reset=[&] {
                for(auto* a:{&ref,&cand}) {
                    for(auto* d:{&a->x,&a->h,&a->q,&a->parts,&a->cpu,&a->map})d->poison(s);
                    CK(cudaMemcpyAsync(a->x.p,initial.data(),size_t(T)*N*4,cudaMemcpyHostToDevice,s));
                    CK(cudaMemcpyAsync(a->parts.p,parts.data(),size_t(entries)*N*4,cudaMemcpyHostToDevice,s));
                }
                CK(cudaMemcpyAsync(ref.cpu.p,cpu.data(),size_t(entries)*N*4,cudaMemcpyHostToDevice,s));
                if(missed)CK(cudaMemcpyAsync(cand.cpu.p,packed.data(),size_t(missed)*N*4,cudaMemcpyHostToDevice,s));
                CK(cudaMemcpyAsync(ref.map.p,flags.data(),size_t(entries)*4,cudaMemcpyHostToDevice,s));
                CK(cudaMemcpyAsync(cand.map.p,map.data(),size_t(entries)*4,cudaMemcpyHostToDevice,s));
            };
            reset();launch(false,ref,in,T,mode,sg_stride,s);launch(true,cand,in,T,mode,sg_stride,s);CK(cudaStreamSynchronize(s));
            compare(ref,cand,T,count,pattern,mode,parts,s);++checks;
            auto actual=read_words(cand.x,s);std::vector<float> xf(actual.size());
            std::memcpy(xf.data(),actual.data(),actual.size()*4);
            for(int t=0;t<T;++t)for(int i=0;i<N;i+=97) {
                double oracle=0;
                for(int k=0;k<TOP;++k) {
                    const int e=t*TOP+k;const float v=flags[e]?cpu[size_t(e)*N+i]:gpu[size_t(e)*N+i];
                    oracle=std::fma(double(weights[e]),double(v),oracle);
                }
                oracle+=double(shared[size_t(t)*N+i])/(1.+std::exp(-double(gates[t*sg_stride])));
                oracle+=initial[size_t(t)*N+i];const double error=std::fabs(double(xf[size_t(t)*N+i])-oracle);
                max_cpu_abs=std::max(max_cpu_abs,error);
                if(!std::isfinite(xf[size_t(t)*N+i])||error>2e-6*(1.+std::fabs(oracle)))fail("CPU weighted oracle",T,count,pattern,mode,i);
            }
            cudaGraph_t graphs[2]{};cudaGraphExec_t exec[2]{};
            for(int arm=0;arm<2;++arm) {
                CK(cudaStreamBeginCapture(s,CAPTURE_MODE));launch(arm!=0,arm?cand:ref,in,T,mode,sg_stride,s);
                CK(cudaStreamEndCapture(s,&graphs[arm]));CK(cudaGraphInstantiate(&exec[arm],graphs[arm],nullptr,nullptr,0));
            }
            for(int replay=0;replay<3;++replay) {
                reset();CK(cudaGraphLaunch(exec[0],s));CK(cudaGraphLaunch(exec[1],s));CK(cudaStreamSynchronize(s));
                compare(ref,cand,T,count,pattern,mode,parts,s);++checks;
            }
            if(bench&&mode==2&&pattern==0&&count==4&&(T==1||T==4||T==8))graph_bench(exec[0],exec[1],T,count,s);
            for(int arm=0;arm<2;++arm){CK(cudaGraphExecDestroy(exec[arm]));CK(cudaGraphDestroy(graphs[arm]));}
        }
    }
    CK(cudaStreamDestroy(s));
    std::printf("PASS %d exact residual/norm/Q8 eager+Graph checks; CPU max_weighted_abs_error=%.9g\n",checks,max_cpu_abs);
    return 0;
}
#endif
