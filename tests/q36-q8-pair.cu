// Bitwise paired GDN projections against the existing separate Q8_0 kernels,
// with a sampled CPU FP64 oracle. No model files or test framework are needed.
// CUDA: nvcc -std=c++20 -O2 --use_fast_math -Iinclude -Isrc/qwen35 tests/q36-q8-pair.cu -o q36-q8-pair
// Add the GPU architecture flag for the device being tested. Optional --bench
// prints eager timings and Graph ABBA timings; run outside other GPU work.
#include <cstdio>
#if !defined(__CUDACC__) || defined(__HIPCC__)
int main() { std::puts("SKIP: paired projection regression requires CUDA"); return 0; }
#else
#include "q36_pair.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { const auto error=(call); if(error!=cudaSuccess) { std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(error));std::exit(2); } } while(0)
struct Q80 { half d; int8_t q[32]; };
static_assert(sizeof(Q80)==34,"GGUF Q8_0 block");
struct Device {
    void* p=nullptr;
    explicit Device(size_t bytes) { CK(cudaMalloc(&p,bytes)); }
    ~Device() { cudaFree(p); }
};
template<int NC,int NJ> void launch(bool pair,void* a,void* b,void* x,float* ya,float* yb,int nb,int na,int nbout,cudaStream_t s) {
    if(pair) q36k::launch_q8_rows_pair<NC,NJ>(a,b,x,ya,yb,nb,na,nbout,s);
    else { q36k::launch_q8_rows<NC,NJ,false>(a,nullptr,x,ya,nb,na,s);q36k::launch_q8_rows<NC,NJ,false>(b,nullptr,x,yb,nb,nbout,s); }
}
void dispatch(bool pair,int nc,void* a,void* b,void* x,float* ya,float* yb,int nb,int na,int nbout,cudaStream_t s) {
#define CASE(N) case N: if(nb<=32) launch<N,1>(pair,a,b,x,ya,yb,nb,na,nbout,s);else launch<N,2>(pair,a,b,x,ya,yb,nb,na,nbout,s);break;
    switch(nc) { CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) }
#undef CASE
}
int main(int argc,char** argv) {
    const bool bench=argc==2 && std::strcmp(argv[1],"--bench")==0;
    if(argc>1 && !bench) { std::fputs("usage: q36-q8-pair [--bench]\n",stderr);return 2; }
    cudaStream_t s{};CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    uint32_t random=0x46348b23;
    auto next=[&] { random=random*1664525u+1013904223u;return random; };
    struct Shape { int n,a,b; };
    const Shape shapes[]={{256,19,7},{1024,255,129},{2048,8192,4096},{2048,8191,4097}};
    int checks=0;double max_abs=0,max_rel=0;bool cpu_pass=true;
    for(auto shape:shapes) {
        const int nb=shape.n/32;
        std::vector<Q80> ha(size_t(nb)*shape.a),hb(size_t(nb)*shape.b);
        std::vector<q36k::QB> hx(size_t(nb)*8);
        for(auto* weights:{&ha,&hb}) for(auto& block:*weights) {
            block.d=__float2half(float(1+next()%4096)/16384.f);
            for(auto& q:block.q) q=int8_t(int(next()%255)-127);
        }
        for(auto& block:hx) {
            block.ds=__halves2half2(__float2half(float(17+next()%2048)/8192.f),__float2half(0));
            for(auto& q:block.qs) q=int8_t(int(next()%255)-127);
        }
        Device a(ha.size()*sizeof(Q80)),b(hb.size()*sizeof(Q80)),x(hx.size()*sizeof(q36k::QB));
        Device ra(size_t(shape.a)*8*sizeof(float)),rb(size_t(shape.b)*8*sizeof(float));
        Device ya(size_t(shape.a)*8*sizeof(float)),yb(size_t(shape.b)*8*sizeof(float));
        const auto poison = [&] {
            CK(cudaMemsetAsync(ya.p,0xff,size_t(shape.a)*8*sizeof(float),s));
            CK(cudaMemsetAsync(yb.p,0xff,size_t(shape.b)*8*sizeof(float),s));
        };
        const auto unused_columns_intact = [&](int nc) {
            for(int which=0;which<2;++which) {
                const int rows=which ? shape.b : shape.a;
                const auto* output=static_cast<const uint32_t*>(which ? yb.p : ya.p);
                std::vector<uint32_t> unused(size_t(8-nc)*rows);
                if(unused.empty()) continue;
                CK(cudaMemcpyAsync(unused.data(),output+size_t(nc)*rows,unused.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,s));
                CK(cudaStreamSynchronize(s));
                if(!std::all_of(unused.begin(),unused.end(),[](uint32_t value) { return value==0xffffffffu; })) return false;
            }
            return true;
        };
        CK(cudaMemcpy(a.p,ha.data(),ha.size()*sizeof(Q80),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.p,hb.data(),hb.size()*sizeof(Q80),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x.p,hx.data(),hx.size()*sizeof(q36k::QB),cudaMemcpyHostToDevice));
        for(int nc=1;nc<=8;++nc) {
            std::vector<float> oa(size_t(shape.a)*nc),ob(size_t(shape.b)*nc),pa(oa.size()),pb(ob.size());
            dispatch(false,nc,a.p,b.p,x.p,(float*)ra.p,(float*)rb.p,nb,shape.a,shape.b,s);
            CK(cudaMemcpyAsync(oa.data(),ra.p,oa.size()*sizeof(float),cudaMemcpyDeviceToHost,s));
            CK(cudaMemcpyAsync(ob.data(),rb.p,ob.size()*sizeof(float),cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));
            poison();
            dispatch(true,nc,a.p,b.p,x.p,(float*)ya.p,(float*)yb.p,nb,shape.a,shape.b,s);
            CK(cudaMemcpyAsync(pa.data(),ya.p,pa.size()*sizeof(float),cudaMemcpyDeviceToHost,s));
            CK(cudaMemcpyAsync(pb.data(),yb.p,pb.size()*sizeof(float),cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));
            if(std::memcmp(oa.data(),pa.data(),oa.size()*sizeof(float)) || std::memcmp(ob.data(),pb.data(),ob.size()*sizeof(float)) || !unused_columns_intact(nc)) {
                std::fprintf(stderr,"FAIL bitwise n=%d a=%d b=%d nc=%d\n",shape.n,shape.a,shape.b,nc);return 1;
            }
            for(int which=0;which<2;++which) {
                const auto& hw=which ? hb : ha;const auto& actual=which ? pb : pa;const int rows=which ? shape.b : shape.a;
                for(int col=0;col<nc;++col) for(int sample=0;sample<std::min(rows,32);++sample) {
                    const int row=(int64_t(sample)*(rows-1))/std::max(1,std::min(rows,32)-1);
                    double exact=0,sum_abs=0;
                    for(int k=0;k<shape.n;++k) {
                        const auto& w=hw[size_t(row)*nb+k/32];const auto& input=hx[size_t(col)*nb+k/32];
                        const double term=double(__half2float(w.d))*w.q[k%32]*double(__half2float(__low2half(input.ds)))*input.qs[k%32];
                        exact+=term;sum_abs+=std::abs(term);
                    }
                    const double error=std::abs(double(actual[size_t(col)*rows+row])-exact);
                    max_abs=std::max(max_abs,error);max_rel=std::max(max_rel,error/std::max(1.,std::abs(exact)));
                    cpu_pass=cpu_pass && std::isfinite(error) && error<=sum_abs*2e-6+1e-6;
                }
            }
            ++checks;
            if(shape.a==8192 && (nc==1||nc==4||nc==8)) {
                cudaGraph_t graphs[2]{};cudaGraphExec_t executables[2]{};
                for(int mode=0;mode<2;++mode) {
                    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
                    dispatch(mode!=0,nc,a.p,b.p,x.p,(float*)(mode ? ya.p : ra.p),(float*)(mode ? yb.p : rb.p),nb,shape.a,shape.b,s);
                    CK(cudaStreamEndCapture(s,&graphs[mode]));CK(cudaGraphInstantiate(&executables[mode],graphs[mode],nullptr,nullptr,0));
                }
                for(int replay=0;replay<3;++replay) {
                    CK(cudaGraphLaunch(executables[0],s));
                    CK(cudaMemcpyAsync(oa.data(),ra.p,oa.size()*sizeof(float),cudaMemcpyDeviceToHost,s));
                    CK(cudaMemcpyAsync(ob.data(),rb.p,ob.size()*sizeof(float),cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));
                    poison();
                    CK(cudaGraphLaunch(executables[1],s));
                    CK(cudaMemcpyAsync(pa.data(),ya.p,pa.size()*sizeof(float),cudaMemcpyDeviceToHost,s));
                    CK(cudaMemcpyAsync(pb.data(),yb.p,pb.size()*sizeof(float),cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));
                    if(std::memcmp(oa.data(),pa.data(),oa.size()*sizeof(float)) || std::memcmp(ob.data(),pb.data(),ob.size()*sizeof(float)) || !unused_columns_intact(nc)) {
                        std::fprintf(stderr,"FAIL graph replay n=%d a=%d b=%d nc=%d replay=%d\n",shape.n,shape.a,shape.b,nc,replay);return 1;
                    }
                    ++checks;
                }
                if(bench) {
                cudaEvent_t begin{},end{};CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
                float ms[2]{};
                for(int mode=0;mode<2;++mode) {
                    for(int i=0;i<16;++i) dispatch(mode!=0,nc,a.p,b.p,x.p,(float*)(mode ? ya.p : ra.p),(float*)(mode ? yb.p : rb.p),nb,shape.a,shape.b,s);
                    CK(cudaEventRecord(begin,s));
                    for(int i=0;i<256;++i) dispatch(mode!=0,nc,a.p,b.p,x.p,(float*)(mode ? ya.p : ra.p),(float*)(mode ? yb.p : rb.p),nb,shape.a,shape.b,s);
                    CK(cudaEventRecord(end,s));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms[mode],begin,end));
                }
                std::printf("EAGER_MICRO n=%d qkv=%d z=%d T=%d old_us=%.3f pair_us=%.3f\n",shape.n,shape.a,shape.b,nc,ms[0]*1000/256,ms[1]*1000/256);
                for(int mode=0;mode<2;++mode) for(int i=0;i<16;++i) CK(cudaGraphLaunch(executables[mode],s));
                CK(cudaStreamSynchronize(s));
                double total[2]{};int count[2]{};
                for(int mode : {0,1,1,0}) {
                    CK(cudaEventRecord(begin,s));
                    for(int i=0;i<256;++i) CK(cudaGraphLaunch(executables[mode],s));
                    CK(cudaEventRecord(end,s));CK(cudaEventSynchronize(end));float elapsed=0;CK(cudaEventElapsedTime(&elapsed,begin,end));
                    total[mode]+=elapsed;++count[mode];
                }
                std::printf("GRAPH_MICRO n=%d qkv=%d z=%d T=%d old_us=%.3f pair_us=%.3f ABBA=2+2x256replays\n",shape.n,shape.a,shape.b,nc,total[0]*1000/(256*count[0]),total[1]*1000/(256*count[1]));
                CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
                }
                for(int mode=0;mode<2;++mode) { CK(cudaGraphExecDestroy(executables[mode]));CK(cudaGraphDestroy(graphs[mode])); }
            }
        }
    }
    CK(cudaStreamDestroy(s));
    std::printf("%s %d exact paired-vs-separate shapes/T checks; CPU sampled max_abs=%.9g max_rel=%.9g\n",cpu_pass ? "PASS" : "FAIL",checks,max_abs,max_rel);
    return cpu_pass ? 0 : 1;
}

#endif
