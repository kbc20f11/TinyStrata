#include "strata/gemma4/mxfp4_down.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace K=strata::kernels;
namespace P=strata::kernels::mxfp4_down;
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)
constexpr int N=2880,NB=90,EX=4,XCOUNT=32,OUTCOUNT=K::kMmvqSlotsMax,YS=N+8;
constexpr size_t WB=size_t(N)*1530,QB=NB*36;
static unsigned checks=0,cases=0;
static void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);++checks;}
static uint32_t seed=0x6d374abeu;
static uint32_t random32(){seed^=seed<<13;seed^=seed>>17;return seed^=seed<<5;}
static uint16_t halfbits(float v){const __half h=__float2half(v);uint16_t b;std::memcpy(&b,&h,2);return b;}
static double halfvalue(uint16_t h){const int e=(h>>10)&31,f=h&1023;return std::ldexp(e?1.0+f/1024.0:double(f),e?e-15:-24)*(h&0x8000?-1:1);}
static float scale(unsigned e){uint32_t b=e==0?0x00200000u:e==1?0x00400000u:e==255?0x7f800000u:(e-1)<<23;float f;std::memcpy(&f,&b,4);return f;}
struct Q81{uint16_t d,sum;int8_t q[32];};static_assert(sizeof(Q81)==36);
struct Device{
    uint8_t*base=nullptr;size_t bytes;cudaStream_t s;
    Device(size_t n,cudaStream_t stream):bytes(n),s(stream){CK(cudaMalloc((void**)&base,n+128));CK(cudaMemsetAsync(base,0xA7,n+128,s));}
    ~Device(){if(base)(void)cudaFree(base);}
    void* data()const{return base+64;}
    template<class T>T*as()const{return static_cast<T*>(data());}
    void put(const void*p,size_t n=size_t(-1)){if(n==size_t(-1))n=bytes;need(n<=bytes,"input copy exceeds buffer");CK(cudaMemcpyAsync(data(),p,n,cudaMemcpyHostToDevice,s));}
    void poison(){CK(cudaMemsetAsync(data(),0xFF,bytes,s));}
    std::vector<uint8_t>get()const{std::vector<uint8_t>v(bytes);CK(cudaMemcpyAsync(v.data(),data(),bytes,cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));return v;}
    void guards()const{uint8_t a[64],b[64];CK(cudaMemcpyAsync(a,base,64,cudaMemcpyDeviceToHost,s));CK(cudaMemcpyAsync(b,base+64+bytes,64,cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));for(int i=0;i<64;++i)need(a[i]==0xA7&&b[i]==0xA7,"redzone modified");}
};
struct Oracle{std::vector<double>value,magnitude;};
struct Fixture{
    cudaStream_t s;int variant;
    std::vector<uint8_t>w;
    std::vector<Q81>q;
    Device dw,dq,baseline,actual,table,indices;
    std::map<std::pair<int,int>,Oracle>oracles;
    Fixture(cudaStream_t stream,int v):s(stream),variant(v),w(EX*WB),q(XCOUNT*NB),dw(w.size(),s),dq(XCOUNT*QB,s),
        baseline(OUTCOUNT*YS*4,s),actual(OUTCOUNT*YS*4,s),table(EX*sizeof(void*),s),indices(OUTCOUNT*4,s){
        for(int e=0;e<EX;++e)for(int r=0;r<N;++r)for(int b=0;b<NB;++b){
            auto*p=w.data()+size_t(e)*WB+(size_t(r)*NB+b)*17;
            p[0]=uint8_t(v==3?(b%4==3?254:b%4):v==4?0:119+random32()%14);
            for(int k=0;k<16;++k)p[1+k]=uint8_t(random32());
        }
        dw.put(w.data());
        if(v==0){
            std::vector<float>x(XCOUNT*N);for(float&a:x)a=float(int(random32()%2049)-1024)/257.f;
            Device dx(x.size()*4,s);dx.put(x.data());K::gemma_native_quantize_q8_1(dx.as<float>(),dq.data(),N,XCOUNT,s);
            const auto bytes=dq.get();std::memcpy(q.data(),bytes.data(),bytes.size());dx.guards();
        }else{
            for(int a=0;a<XCOUNT;++a)for(int b=0;b<NB;++b){auto&z=q[a*NB+b];
                z.d=v==3?(b%4==3?uint16_t(1):halfbits(65504.f)):v==4?halfbits(65504.f):halfbits(1.f/128);
                int sum=0;for(int i=0;i<32;++i){z.q[i]=v==2?0:int8_t(v==1?(i&1?127:-128):int(random32()%256)-128);sum+=z.q[i];}
                z.sum=halfbits(float(std::clamp(halfvalue(z.d)*sum,-65504.0,65504.0)));
            }
            dq.put(q.data());
        }
        const void* ptrs[EX];for(int e=0;e<EX;++e)ptrs[e]=dw.as<uint8_t>()+size_t(e)*WB;table.put(ptrs);
    }
    const Oracle&oracle(int expert,int input){
        const auto key=std::make_pair(expert,input);auto found=oracles.find(key);if(found!=oracles.end())return found->second;
        static constexpr int values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        Oracle out;out.value.resize(N);out.magnitude.resize(N);
        for(int r=0;r<N;++r)for(int b=0;b<NB;++b){
            const auto*p=w.data()+size_t(expert)*WB+(size_t(r)*NB+b)*17;const auto&x=q[input*NB+b];
            int dot=0,absolute=0;
            for(int i=0;i<32;++i){const int code=(p[1+i%16]>>(4*(i/16)))&15;const int term=values[code]*int(x.q[i]);dot+=term;absolute+=std::abs(term);}
            const double d=double(scale(p[0]))*halfvalue(x.d);out.value[r]+=d*dot;out.magnitude[r]+=std::abs(d)*absolute;
        }
        return oracles.emplace(key,std::move(out)).first->second;
    }
    K::MmvqSlots bundle(const int*experts,const int*inputs,int count,bool candidate){
        K::MmvqSlots p{};for(int j=0;j<count;++j){p.w[j]=dw.as<uint8_t>()+size_t(experts[j])*WB;p.x[j]=dq.as<uint8_t>()+size_t(inputs[j])*QB;p.y[j]=(candidate?actual:baseline).as<float>()+j*YS;}return p;
    }
    void compare(const int*experts,const int*inputs,int count,const char*name){
        const auto ba=baseline.get(),ac=actual.get();double maxa=0,maxb=0,rmsa=0,rmsb=0,maxnorm=0,diff=0;size_t n=0;
        for(int j=0;j<OUTCOUNT;++j)for(int r=0;r<YS;++r){const size_t at=(size_t(j)*YS+r)*4;
            if(j>=count||r>=N){for(int k=0;k<4;++k)need(ba[at+k]==0xFF&&ac[at+k]==0xFF,"inactive output/row gap overwritten");continue;}
            float a,b;std::memcpy(&a,ac.data()+at,4);std::memcpy(&b,ba.data()+at,4);
            const auto&o=oracle(experts[j],inputs[j]);const double ea=std::abs(double(a)-o.value[r]),eb=std::abs(double(b)-o.value[r]);
            const double bound=2e-6*o.magnitude[r]+1e-40;
            if(!std::isfinite(a)||!std::isfinite(b)||ea>bound||eb>bound){std::fprintf(stderr,"FAIL %s variant%d j%d row%d ref%.17g new%.9g old%.9g bound%.9g\n",name,variant,j,r,o.value[r],a,b,bound);throw std::runtime_error("numerical bound");}
            if(variant==2)need(a==0&&b==0,"zero activation changed");
            maxa=std::max(maxa,ea);maxb=std::max(maxb,eb);rmsa+=ea*ea;rmsb+=eb*eb;diff=std::max(diff,std::abs(double(a)-b));
            maxnorm=std::max(maxnorm,ea/std::max(o.magnitude[r],1e-100));++n;
        }
        std::printf("PARITY variant=%d path=%s count=%d maxabs_new=%.9g maxabs_old=%.9g rms_new=%.9g rms_old=%.9g max_normalized=%.9g max_vs_old=%.9g\n",variant,name,count,maxa,maxb,std::sqrt(rmsa/n),std::sqrt(rmsb/n),maxnorm,diff);++cases;
        actual.guards();baseline.guards();
    }
    void check_inputs(){need(dw.get()==w,"MXFP4 bytes modified");const auto b=dq.get();need(std::memcmp(b.data(),q.data(),b.size())==0,"Q8_1 bytes modified");dw.guards();dq.guards();table.guards();indices.guards();}
};
static void probe(cudaStream_t s){
    Device scales(256*4,s),codes(16*4,s);P::probe(scales.as<uint32_t>(),codes.as<uint32_t>(),s);
    const auto a=scales.get(),b=codes.get();
    const int values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    for(unsigned i=0;i<256;++i){uint32_t got,want;const float f=scale(i);std::memcpy(&want,&f,4);std::memcpy(&got,a.data()+i*4,4);need(got==want,"scale bits differ including e0/e1/e255");}
    for(int i=0;i<16;++i){const uint8_t c=uint8_t(int8_t(values[i]));const uint32_t want=uint32_t(c)*0x01010101u;uint32_t got;std::memcpy(&got,b.data()+i*4,4);need(got==want,"E2M1 nibble decode differs");}
    scales.guards();codes.guards();
    for(int r=0;r<N;++r)for(int block=0;block<NB;++block){const size_t address=size_t(r)*1530+17*block,p=address&~size_t(3);need(p+20<=WB,"aligned load exceeds exact matrix end");}
    need(((size_t(N-1)*1530+17*89)&3)==3,"final block must end on five-word boundary");
}
static void parity(cudaStream_t s,int variant){
    Fixture f(s,variant);const int unique[4]={0,1,2,3},input[4]={0,1,2,3};
    for(int count:{1,2,4}){f.baseline.poison();f.actual.poison();const auto a=f.bundle(unique,input,count,true),b=f.bundle(unique,input,count,false);
        if(count==1){K::gemma_native_mmvq(39,b.w[0],b.x[0],b.y[0],N,N,1,s);need(P::direct(39,a.w[0],a.x[0],a.y[0],N,N,1,s),"direct dispatch refused");}
        else{need(K::gemma_native_mmvq_slots(39,b,count,N,N,s),"baseline slots refused");need(P::slots(39,a,count,N,N,s),"prototype slots refused");}
        f.compare(unique,input,count,count==1?"direct":"slots");
    }
    const int reordered[4]={3,0,3,1},repeated[4]={0,0,1,1};f.indices.put(reordered,sizeof(reordered));
    for(int count:{1,3,4}){f.baseline.poison();f.actual.poison();
        K::MmvqIds a{f.table.as<const void*>(),f.indices.as<int32_t>(),0,f.dq.data(),QB,2,f.actual.as<float>(),YS},b=a;b.y=f.baseline.as<float>();
        need(K::gemma_native_mmvq_ids(39,b,count,N,N,s),"baseline IDs refused");need(P::ids(39,a,count,N,N,s),"prototype IDs refused");f.compare(reordered,repeated,count,"ids-repeat-reorder");
    }
    if(variant==0){
        // A target eval and any lookup window must retain identical per-product bits.
        // Count is only grid.y: compare every prefix as the batch grows through all1..32.
        int many_experts[OUTCOUNT],many_inputs[OUTCOUNT];
        for(int j=0;j<OUTCOUNT;++j){many_experts[j]=j%EX;many_inputs[j]=j;}
        std::vector<uint8_t> previous;
        for(int count=1;count<=OUTCOUNT;++count){
            f.baseline.poison();f.actual.poison();const auto a=f.bundle(many_experts,many_inputs,count,true),b=f.bundle(many_experts,many_inputs,count,false);
            need(K::gemma_native_mmvq_slots(39,b,count,N,N,s),"all-count baseline refused");need(P::slots(39,a,count,N,N,s),"all-count prototype refused");
            f.compare(many_experts,many_inputs,count,"all-count-bitwise-prefix");
            const auto now=f.actual.get();need(previous.empty()||std::memcmp(previous.data(),now.data(),previous.size())==0,"count changed a product's bits");
            previous.assign(now.begin(),now.begin()+size_t(count)*YS*4);
        }
        for(int every:{1,2}){
            for(int j=0;j<OUTCOUNT;++j){many_experts[j]=(3*j+1)%EX;many_inputs[j]=j/every;}
            f.indices.put(many_experts,sizeof(many_experts));f.baseline.poison();f.actual.poison();
            K::MmvqIds a{f.table.as<const void*>(),f.indices.as<int32_t>(),0,f.dq.data(),QB,every,f.actual.as<float>(),YS},b=a;b.y=f.baseline.as<float>();
            need(K::gemma_native_mmvq_ids(39,b,OUTCOUNT,N,N,s),"full-window IDs baseline refused");need(P::ids(39,a,OUTCOUNT,N,N,s),"full-window IDs prototype refused");f.compare(many_experts,many_inputs,OUTCOUNT,"full-window-IDs");
        }
        f.indices.put(reordered,sizeof(reordered));
        f.baseline.poison();f.actual.poison();const auto a=f.bundle(unique,input,4,true),b=f.bundle(unique,input,4,false);
        need(K::gemma_native_mmvq_slots(39,b,4,N,N,s),"graph baseline refused");CK(cudaStreamSynchronize(s));
        cudaGraph_t g=nullptr;cudaGraphExec_t exec=nullptr;CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));need(P::slots(39,a,4,N,N,s),"graph prototype refused");CK(cudaStreamEndCapture(s,&g));CK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
        for(int repeat=0;repeat<2;++repeat){f.actual.poison();CK(cudaGraphLaunch(exec,s));f.compare(unique,input,4,"graph-replay");}
        CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(g));
        // The all-hit route changes device IDs/table entries while the captured argument pointers stay fixed.
        K::MmvqIds id_a{f.table.as<const void*>(),f.indices.as<int32_t>(),0,f.dq.data(),QB,2,f.actual.as<float>(),YS},id_b=id_a;id_b.y=f.baseline.as<float>();
        CK(cudaStreamSynchronize(s));CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));need(P::ids(39,id_a,4,N,N,s),"IDs graph refused");CK(cudaStreamEndCapture(s,&g));CK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
        for(int repeat=0;repeat<2;++repeat){
            const int selected0[4]={3,0,3,1},selected1[4]={1,2,0,2},table0[4]={0,1,2,3},table1[4]={1,0,3,2};
            const int* selected=repeat?selected1:selected0;const int* map=repeat?table1:table0;
            const void* pointers[EX];int effective[EX];for(int j=0;j<EX;++j){pointers[j]=f.dw.as<uint8_t>()+size_t(map[j])*WB;effective[j]=map[selected[j]];}
            f.indices.put(selected,4*sizeof(int));f.table.put(pointers);f.baseline.poison();f.actual.poison();
            need(K::gemma_native_mmvq_ids(39,id_b,4,N,N,s),"dynamic IDs graph baseline refused");CK(cudaGraphLaunch(exec,s));f.compare(effective,repeated,4,"graph-ids-and-table-update");
        }
        CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(g));
        f.actual.poison();need(!P::slots(2,a,4,N,N,s)&&!P::slots(39,a,0,N,N,s)&&!P::slots(39,a,OUTCOUNT+1,N,N,s)&&!P::slots(39,a,4,N-32,N,s)&&!P::slots(39,a,4,N,N-1,s),"unsupported slots did not fall back");
        need(!P::direct(39,a.w[0],a.x[0],a.y[0],N,N,2,s)&&!P::direct(39,(const char*)a.w[0]+2,a.x[0],a.y[0],N,N,1,s),"unsupported direct did not fall back");
        auto invalid=id_a;invalid.table=nullptr;need(!P::ids(39,invalid,4,N,N,s),"null IDs table accepted");
        invalid=id_a;invalid.ids=nullptr;need(!P::ids(39,invalid,4,N,N,s),"null device IDs accepted");
        invalid=id_a;invalid.x_every=0;need(!P::ids(39,invalid,4,N,N,s),"zero x_every accepted");
        invalid=id_a;invalid.offset=2;need(!P::ids(39,invalid,4,N,N,s),"unaligned IDs matrix offset accepted");
        invalid=id_a;invalid.x_stride=QB-4;need(!P::ids(39,invalid,4,N,N,s),"short IDs input stride accepted");
        invalid=id_a;invalid.y_stride=N-1;need(!P::ids(39,invalid,4,N,N,s),"short IDs output stride accepted");
        need(!P::ids(39,id_a,OUTCOUNT+1,N,N,s)&&!P::ids(2,id_a,4,N,N,s)&&!P::ids(39,id_a,4,N,N-1,s),"unsupported IDs shape/count accepted");
        const auto untouched=f.actual.get();need(std::all_of(untouched.begin(),untouched.end(),[](uint8_t c){return c==0xFF;}),"fallback launched or wrote");
    }
    f.check_inputs();
}
static void bench(cudaStream_t s){
    Fixture f(s,0);const int ids[4]={0,1,2,3},xs[4]={0,1,2,3};
    for(int count:{1,4})for(int arm:{0,1,1,0}){
        const auto a=f.bundle(ids,xs,count,true),b=f.bundle(ids,xs,count,false);
        const auto launch=[&]{if(arm){need(P::slots(39,a,count,N,N,s),"bench prototype refused");}else if(count==1)K::gemma_native_mmvq(39,b.w[0],b.x[0],b.y[0],N,N,1,s);else need(K::gemma_native_mmvq_slots(39,b,count,N,N,s),"bench baseline refused");};
        for(int i=0;i<8;++i)launch();CK(cudaStreamSynchronize(s));cudaEvent_t a0,a1;CK(cudaEventCreate(&a0));CK(cudaEventCreate(&a1));
        const auto start=std::chrono::steady_clock::now();CK(cudaEventRecord(a0,s));for(int i=0;i<100;++i)launch();CK(cudaEventRecord(a1,s));CK(cudaEventSynchronize(a1));
        float ms=0;CK(cudaEventElapsedTime(&ms,a0,a1));const double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100;
        std::printf("BENCH resident_hot_weights=1 count=%d candidate=%d gpu_us=%.6f wall_us=%.6f iterations=100\n",count,arm,ms*10,wall);std::fflush(stdout);CK(cudaEventDestroy(a0));CK(cudaEventDestroy(a1));
    }
}
int main(int argc,char**argv){try{
    const bool timing=argc==2&&std::string(argv[1])=="--bench";need(argc==1||timing,"only --bench is accepted");
    cudaDeviceProp props{};CK(cudaGetDeviceProperties(&props,0));
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
    const std::string architecture=props.gcnArchName;
    need(architecture.starts_with("gfx1200")||architecture.starts_with("gfx1201"),"MXFP4 down unit requires gfx1200 or gfx1201");
#else
    throw std::runtime_error("HIP prototype only");
#endif
    cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));probe(s);for(int v=0;v<5;++v)parity(s,v);if(timing)bench(s);CK(cudaStreamDestroy(s));
    std::printf("PASS MXFP4 down prototype cases=%u checks=%u exact_shape=2880 row_bytes=1530 no_repack=1\n",cases,checks);return 0;
}catch(const std::exception&ex){std::fprintf(stderr,"mxfp4_down_test: %s\n",ex.what());return 1;}}
