// Exact stored-byte parity of GPT-OSS activation+native/MMQ quantization.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/gemma4/activation_quant.hpp"
#include "strata/gemma4/kernels.hpp"
#include "strata/gemma4/matvec.hpp"
#include "mmq.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace G=strata::gemma4;
namespace K=strata::kernels;
namespace A=G::activation_quant;
#define CK(x) do { const auto e_=(x); if(e_!=cudaSuccess) { std::fprintf(stderr,"line%d: %s\n",__LINE__,cudaGetErrorString(e_));std::exit(2); } } while(0)
#define CHECK(x) do { if(!(x)) { std::fprintf(stderr,"FAIL line%d: %s\n",__LINE__,#x);std::exit(1); } } while(0)
constexpr int EXPERTS=32;
constexpr float ALPHA=1.702f,LIMIT=7.f;
static unsigned cases=0;
static uint64_t compared=0;
static uint32_t rng=0x7F14BA21u;
static uint32_t word() { rng^=rng<<13;rng^=rng>>17;return rng^=rng<<5; }
static float real() { return float(double(word())/4294967296.0*2.0-1.0); }

struct Device {
    uint8_t* p=nullptr;size_t n;cudaStream_t s;
    Device(size_t bytes,cudaStream_t stream):n(bytes),s(stream) {
        CK(cudaMalloc(reinterpret_cast<void**>(&p),n+128));
        CK(cudaMemsetAsync(p,0xA7,n+128,s));
    }
    ~Device() { if(p) (void)cudaFree(p); }
    template<class T>T* data()const { return reinterpret_cast<T*>(p+64); }
    template<class T>void put(const std::vector<T>& v) {
        CHECK(v.size()*sizeof(T)==n);CK(cudaMemcpyAsync(p+64,v.data(),n,cudaMemcpyHostToDevice,s));
    }
    void poison() { CK(cudaMemsetAsync(p+64,0xCD,n,s)); }
    std::vector<uint8_t> get()const {
        CK(cudaStreamSynchronize(s));std::vector<uint8_t> v(n+128);
        CK(cudaMemcpy(v.data(),p,v.size(),cudaMemcpyDeviceToHost));
        for(size_t j=0;j<64;++j) CHECK(v[j]==0xA7 && v[n+64+j]==0xA7);
        return {v.begin()+64,v.end()-64};
    }
    template<class T>void unchanged(const std::vector<T>& v)const {
        const auto b=get();CHECK(b.size()==v.size()*sizeof(T));CHECK(std::memcmp(b.data(),v.data(),n)==0);
    }
};

static void equal(const std::vector<uint8_t>& ref,const std::vector<uint8_t>& got,const char* stage,int n,int rows) {
    CHECK(ref.size()==got.size());
    for(size_t i=0;i<ref.size();++i) if(ref[i]!=got[i]) {
        std::fprintf(stderr,"FAIL %s width=%d rows=%d byte=%zu reference=%02x fused=%02x\n",stage,n,rows,i,unsigned(ref[i]),unsigned(got[i]));std::exit(1);
    }
    compared+=ref.size();++cases;
}
static void graph_run(cudaStream_t s,auto&& work,auto&& replay) {
    CK(cudaStreamSynchronize(s));cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));work();
    CK(cudaStreamEndCapture(s,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int i=0;i<2;++i) { replay(i);CK(cudaGraphLaunch(exec,s));CK(cudaStreamSynchronize(s)); }
    CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
}

struct Fixture {
    int n,rows;cudaStream_t s;size_t plain_bytes,mmq_bytes,written_mmq;
    std::vector<float> input,bg,bu;
    std::vector<int32_t> ids;
    Device dx,dbg,dbu,di,h,plain,plain_new,slots_old,slots_new,mmq,mmq_new;
    Fixture(int width,int count,cudaStream_t stream):n(width),rows(count),s(stream),
        plain_bytes(size_t(width)*count/32*36),mmq_bytes(G::Mmq::q8_bytes(count,width)),
        written_mmq(size_t(count)*((width+511)/512*512)/128*144),
        input(size_t(count)*width*2),bg(size_t(EXPERTS)*width),bu(bg.size()),ids(count),
        dx(input.size()*4,s),dbg(bg.size()*4,s),dbu(bu.size()*4,s),di(ids.size()*4,s),h(size_t(count)*width*4,s),
        plain(plain_bytes,s),plain_new(plain_bytes,s),slots_old(plain_bytes,s),slots_new(plain_bytes,s),
        mmq(mmq_bytes,s),mmq_new(mmq_bytes,s) {}
    void fill(int pattern,int mapping=0) {
        for(int r=0;r<rows;++r) ids[r]=(r%5==0)?7:(r*13+mapping*9+3)%EXPERTS;
        for(auto& x:bg) x=(pattern==2 || pattern==4)?0.f:real()*0.73f;
        for(auto& x:bu) x=(pattern==2 || pattern==4)?0.f:real()*0.81f;
        for(int r=0;r<rows;++r) for(int i=0;i<n;++i) {
            float g=real()*11.f,u=real()*11.f;
            if(pattern==1) { g=(i%3==0)?-1e20f:(i%3==1?1e20f:7.f);u=(i&1)?1e20f:-1e20f; }
            if(pattern==2) { g=(i&1)?-0.f:0.f;u=-1.f; }
            if(pattern==3) { g=1.25f;u=float((i%32)-16)/32.f; }
            if(pattern==4) { g=std::ldexp(float((i%31)-15),-125);u=0.f; }
            input[(size_t(r)*2)*n+i]=g;input[(size_t(r)*2+1)*n+i]=u;
        }
        upload();
    }
    void upload() { dx.put(input);dbg.put(bg);dbu.put(bu);di.put(ids); }
    G::ExpertBias slot_bias()const {
        CHECK(rows<=32);G::ExpertBias eb{};
        for(int r=0;r<rows;++r) { eb.g[r]=dbg.data<float>()+size_t(ids[r])*n;eb.u[r]=dbu.data<float>()+size_t(ids[r])*n; }
        return eb;
    }
    void baseline(bool mmq_mode) {
        G::swiglu_oai_ids(dx.data<float>(),h.data<float>(),n,rows,di.data<int32_t>(),dbg.data<float>(),dbu.data<float>(),ALPHA,LIMIT,s);
        if(mmq_mode) G::Mmq::quantize(h.data<float>(),nullptr,mmq.data<void>(),39,n,n,rows,s);
        else K::gemma_native_quantize_q8_1(h.data<float>(),plain.data<void>(),n*rows,1,s);
    }
    void fused(bool mmq_mode) {
        if(mmq_mode) CHECK(A::mmq_mxfp4_ids(dx.data<float>(),mmq_new.data<void>(),n,rows,di.data<int32_t>(),dbg.data<float>(),dbu.data<float>(),ALPHA,LIMIT,s));
        else CHECK(A::q8_1_ids(dx.data<float>(),plain_new.data<void>(),n,rows,di.data<int32_t>(),dbg.data<float>(),dbu.data<float>(),ALPHA,LIMIT,s));
    }
    void guards() {
        dx.unchanged(input);dbg.unchanged(bg);dbu.unchanged(bu);di.unchanged(ids);
        const auto activations=h.get();
        for(size_t j=0;j<activations.size();j+=4) { float x;std::memcpy(&x,activations.data()+j,4);CHECK(std::isfinite(x)); }
    }
    void check_bytes(bool native=true) {
        if(native) equal(plain.get(),plain_new.get(),"native IDs",n,rows);
        const auto a=mmq.get(),b=mmq_new.get();equal(a,b,"MMQ MXFP4 IDs/pad/tail",n,rows);
        CHECK(written_mmq<=a.size());
        for(size_t j=written_mmq;j<a.size();++j) CHECK(a[j]==0xCD && b[j]==0xCD);
        for(size_t block=0;block<written_mmq/144;++block) for(int j=0;j<4;++j) {
            float d;std::memcpy(&d,b.data()+block*144+j*4,4);CHECK(std::isfinite(d) && d>=0.f);
        }
        guards();
    }
    void all(bool graph=false) {
        const bool native=n%32==0;
        auto work=[&] { if(native) { baseline(false);fused(false); }baseline(true);fused(true); };
        auto poison=[&] { plain.poison();plain_new.poison();mmq.poison();mmq_new.poison(); };
        if(graph) {
            graph_run(s,work,[&](int replay) {
                if(replay) { for(auto& e:ids)e=(e+11)%EXPERTS;for(auto& x:bg)x*=0.973f;for(auto& x:input)x*=0.817f;upload(); }
                poison();
            });
        } else { poison();work(); }
        check_bytes(native);
    }
    void slots(bool graph) {
        CHECK(n%32==0 && rows<=32);
        // Host pointer tables are captured by value. Recapture for a different
        // table; stable-pointer bias data and activations may change on replay.
        for(int mapping=0;mapping<(graph?2:1);++mapping) {
            if(mapping) { for(auto& e:ids)e=(e+5)%EXPERTS;di.put(ids); }
            const auto eb=slot_bias();
            auto work=[&] {
                G::swiglu_oai_slots(dx.data<float>(),h.data<float>(),n,rows,eb,ALPHA,LIMIT,s);
                K::gemma_native_quantize_q8_1(h.data<float>(),slots_old.data<void>(),n*rows,1,s);
                CHECK(A::q8_1_slots(dx.data<float>(),slots_new.data<void>(),n,rows,eb,ALPHA,LIMIT,s));
            };
            if(graph) graph_run(s,work,[&](int replay) {
                if(replay) { for(auto& x:bu)x*=1.031f;for(auto& x:input)x*=0.921f;upload(); }
                slots_old.poison();slots_new.poison();
            });
            else { slots_old.poison();slots_new.poison();work(); }
            equal(slots_old.get(),slots_new.get(),"native slots/captured host pointers",n,rows);
            baseline(false);equal(plain.get(),slots_new.get(),"slots versus expert IDs",n,rows);guards();
        }
    }
    double time(bool merged,bool mmq_mode,bool host) {
        for(int i=0;i<3;++i) { if(merged) fused(mmq_mode);else baseline(mmq_mode); }
        CK(cudaStreamSynchronize(s));
        constexpr int REPEAT=20;cudaEvent_t a=nullptr,b=nullptr;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
        const auto start=std::chrono::steady_clock::now();CK(cudaEventRecord(a,s));
        for(int i=0;i<REPEAT;++i) { if(merged) fused(mmq_mode);else baseline(mmq_mode);if(host) CK(cudaStreamSynchronize(s)); }
        CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms=0;CK(cudaEventElapsedTime(&ms,a,b));
        const double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/REPEAT;
        CK(cudaEventDestroy(b));CK(cudaEventDestroy(a));return host?wall:ms*1000./REPEAT;
    }
};

int main(int argc,char** argv) {
    const bool bench=argc==2 && std::strcmp(argv[1],"--bench")==0;
    CHECK(argc==1 || bench);
    if(!A::available()) { std::puts("SKIP: fused activation quant is HIP-only");return 77; }
    cudaStream_t s=nullptr;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    { Fixture f(32,1,s);f.fill(0);f.plain_new.poison();f.mmq_new.poison();
      const auto before=f.plain_new.get();
      CHECK(!A::q8_1_ids(nullptr,f.plain_new.data<void>(),32,1,f.di.data<int32_t>(),f.dbg.data<float>(),f.dbu.data<float>(),ALPHA,LIMIT,s));
      CHECK(!A::q8_1_ids(f.dx.data<float>(),f.plain_new.data<void>(),31,1,f.di.data<int32_t>(),f.dbg.data<float>(),f.dbu.data<float>(),ALPHA,LIMIT,s));
      CHECK(!A::q8_1_ids(f.dx.data<float>(),f.plain_new.data<void>(),32,0,f.di.data<int32_t>(),f.dbg.data<float>(),f.dbu.data<float>(),ALPHA,LIMIT,s));
      CHECK(!A::mmq_mxfp4_ids(f.dx.data<float>(),f.mmq_new.data<void>(),31,1,f.di.data<int32_t>(),f.dbg.data<float>(),f.dbu.data<float>(),ALPHA,LIMIT,s));
      CHECK(!A::mmq_mxfp4_ids(f.dx.data<float>(),f.mmq_new.data<void>(),32,1,f.di.data<int32_t>(),f.dbg.data<float>(),f.dbu.data<float>(),ALPHA,INFINITY,s));
      G::ExpertBias empty{};CHECK(!A::q8_1_slots(f.dx.data<float>(),f.plain_new.data<void>(),32,1,empty,ALPHA,LIMIT,s));
      CHECK(!A::q8_1_slots(f.dx.data<float>(),f.plain_new.data<void>(),32,33,empty,ALPHA,LIMIT,s));
      equal(before,f.plain_new.get(),"unsupported launches nothing",32,1);
    }
    for(int rows=1;rows<=32;++rows) { Fixture f(2880,rows,s);f.fill(0);f.all();f.slots(false); }
    for(int pattern:{1,2,3,4}) { Fixture f(2880,17,s);f.fill(pattern);f.all();f.slots(false); }
    for(int width:{32,96,36,2876}) { Fixture f(width,7,s);f.fill(0);f.all(); }
    for(int rows:{7,32}) { Fixture f(2880,rows,s);f.fill(0);f.all(true);f.slots(true); }
    { Fixture f(2880,1112*4,s);f.fill(0);f.all(true); }
    // The default1120-token/top4 batch and a large odd compact/staged prefix
    // exercise the MMQ row-interleave stride independently of the4448 case.
    for(int rows:{1120*4,1837}) { Fixture f(2880,rows,s);f.fill(0);f.all(true); }
    if(bench) for(int rows:{4,32,1112*4}) {
        Fixture f(2880,rows,s);f.fill(0);f.all();
        for(bool mmq_mode:{false,true}) for(bool host:{false,true}) {
            const double a=f.time(false,mmq_mode,host),b=f.time(true,mmq_mode,host),c=f.time(true,mmq_mode,host),d=f.time(false,mmq_mode,host);
            std::printf("MICRO width=2880 rows=%d layout=%s completion_sync=%d old_us=%.6f fused_us=%.6f A0=%.6f B0=%.6f B1=%.6f A1=%.6f timer=%s\n",rows,mmq_mode?"MMQ-D4":"native-Q81",int(host),(a+d)/2,(b+c)/2,a,b,c,d,host?"host-wall":"GPU-events");
        }
    }
    CK(cudaStreamDestroy(s));std::printf("PASS activation quant cases=%u compared_bytes=%llu (exact native scale/sum/codes; MMQ D4/padding/unchanged tail; guards; graph IDs/slots)\n",cases,(unsigned long long)compared);return 0;
}
#endif
