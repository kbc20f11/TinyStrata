// Clover INT8 query-sharing regression; no model weights required.
// From the repository root, then run the resulting executable on a GPU:
// CUDA: nvcc -std=c++20 -O3 -Iinclude tests/clover-attention-gpu.cu -o clover-attention-gpu
// HIP: hipcc -std=c++20 -O3 -Iinclude/strata/hip_compat -Iinclude tests/clover-attention-gpu.cu -o clover-attention-gpu
// Set the compiler GPU architecture option. Do not enable fast-math on just one route.
// --bench [--reps 32|64] adds host-fenced ABBA timings; GPU events are not primary.
// Complete partial buffers and outputs are checked bitwise against the original
// independent-query kernel, with a separate FP64 oracle for small contexts.
#include "strata/qwen35/clover_attention.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
namespace A=strata::qwen35::attention;
namespace Q=strata::qwen35::clover_attention;
#define GPU(call) do {auto e=(call);if(e!=cudaSuccess)throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e));} while(0)
constexpr float SENTINEL=12345.25f;
constexpr size_t GUARD=16;
static int checks=0;
static double max_reference_error=0;
static void require(bool okay,const std::string& message) {if(!okay)throw std::runtime_error(message);++checks;}
__global__ void fill_values(float* p,size_t count,float value) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)p[i]=value;
}
template<class T>struct Device {
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){GPU(cudaMalloc((void**)&p,n*sizeof(T)));}
    ~Device(){if(p)(void)cudaFree(p);}
    void put(const std::vector<T>& values,cudaStream_t stream){require(values.size()==n,"input shape");GPU(cudaMemcpyAsync(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice,stream));}
    std::vector<T> get(cudaStream_t stream){std::vector<T> out(n);GPU(cudaMemcpyAsync(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,stream));GPU(cudaStreamSynchronize(stream));return out;}
};
struct Partials {
    Device<float> a,m,l,out;
    Partials(int H,int chunks,int total_queries):a((size_t)8*H*chunks*256+2*GUARD),
        m((size_t)8*H*chunks+2*GUARD),l(m.n),out((size_t)total_queries*H*256+2*GUARD){}
    float* acc(){return a.p+GUARD;}float* maximum(){return m.p+GUARD;}float* sum(){return l.p+GUARD;}float* result(){return out.p+GUARD;}
    void clear(cudaStream_t stream){for(auto* b:{&a,&m,&l,&out})fill_values<<<(b->n+255)/256,256,0,stream>>>(b->p,b->n,SENTINEL);GPU(cudaGetLastError());}
};
static float half_value(uint16_t bits) {
    const int exp=(bits>>10)&31,mantissa=bits&1023;
    const float sign=(bits&32768)?-1.f:1.f;
    if(exp==31)return mantissa?NAN:sign*INFINITY;
    return sign*(exp?std::ldexp(1.f+mantissa/1024.f,exp-15):std::ldexp((float)mantissa,-24));
}
static void equal_arrays(const std::vector<float>& a,const std::vector<float>& b,const std::string& label) {
    require(a.size()==b.size(),label+": shape");
    if(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))!=0) {
        size_t first=0;while(first<a.size() && std::memcmp(&a[first],&b[first],sizeof(float))==0)++first;
        char text[220];std::snprintf(text,sizeof(text),"%s differs at %zu: old %.9g new %.9g",label.c_str(),first,a[first],b[first]);
        throw std::runtime_error(text);
    }
    ++checks;
}
static void guards(const std::vector<float>& a,const std::string& label) {
    for(size_t i=0;i<GUARD;++i)require(a[i]==SENTINEL && a[a.size()-1-i]==SENTINEL,label+": outer sentinel");
}
struct Case {int rows,position,t0,ctx;bool preview,zero=false,full_grid=false;};
static void one(Case c,int H,int HKV,cudaStream_t stream,bool benchmark,int repetitions) {
    const int total_queries=c.t0+c.rows+2,chunk_stride=(c.ctx+255)/256;
    const int end=c.position+c.t0+c.rows,visible=c.preview?end:0;
    const int grid=c.full_grid?chunk_stride:(end+255)/256;
    require(c.rows>=2 && c.rows<=4 && H==8*HKV && end<=c.ctx,"caller shape/context preflight");
    const std::string label="T"+std::to_string(c.rows)+" P"+std::to_string(c.position)+" t0="+std::to_string(c.t0)+(c.preview?" preview":" causal")+(c.zero?" zero":" random");
    const size_t codes=(size_t)HKV*c.ctx*256;
    std::vector<int8_t> key(codes),value(codes);
    uint32_t rng=0xa53f2c17u;
    auto random=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
    for(size_t i=0;i<codes;++i){key[i]=c.zero?0:(int8_t)((int)(random()%255)-127);value[i]=c.zero?0:(int8_t)((int)(random()%255)-127);}
    std::vector<uint16_t> ks(codes/64),vs(codes/64);
    const uint16_t powers[]={0x2400,0x2800,0x2c00,0x3000};
    for(int kh=0;kh<HKV;++kh)for(int p=0;p<c.ctx;++p)for(int g=0;g<4;++g) {
        const size_t i=((size_t)kh*c.ctx+p)*4+g;
        // Poison unneeded logical-capacity cells. They must not contaminate
        // a live query, and the source kernel's VMM read predicates stay intact.
        ks[i]=p<end?powers[(p+kh+g)%4]:0x7e00;
        vs[i]=p<end?powers[(p*3+kh+g)%3]:0x7e00;
    }
    const size_t nq=(size_t)total_queries*H*256;
    std::vector<float> q(nq),gate(nq);
    for(size_t i=0;i<nq;++i){q[i]=c.zero?0:((int)(random()%201)-100)*.001f;gate[i]=((int)(random()%101)-50)*.025f;}
    Device<int8_t> dk(codes),dv(codes);Device<uint16_t> dks(ks.size()),dvs(vs.size());
    Device<float> dq(nq),dg(nq);Device<int32_t> pos(1);
    dk.put(key,stream);dv.put(value,stream);dks.put(ks,stream);dvs.put(vs,stream);dq.put(q,stream);dg.put(gate,stream);pos.put({c.position},stream);
    A::KvCache cache;cache.key=dk.p;cache.value=dv.p;cache.key_scale16=dks.p;cache.value_scale16=dvs.p;
    Partials before(H,chunk_stride,total_queries),after(H,chunk_stride,total_queries);
    before.clear(stream);after.clear(stream);
    auto launch=[&](int queries) {
        auto& b=queries?after:before;
        if(queries)GPU(Q::launch(dq.p,cache,pos.p,c.t0,c.rows,H,HKV,c.ctx,1.f/16,chunk_stride,grid,b.acc(),b.maximum(),b.sum(),stream,visible,queries));
        else {
            A::k_attn_part<2><<<dim3(grid,c.rows*H),256,0,stream>>>(dq.p,cache,pos.p,c.t0,H,HKV,c.ctx,1.f/16,chunk_stride,b.acc(),b.maximum(),b.sum(),visible);
            GPU(cudaGetLastError());
        }
        A::k_attn_merge<<<c.rows*H,256,0,stream>>>(b.acc(),b.maximum(),b.sum(),dg.p,pos.p,c.t0,H,chunk_stride,b.result(),visible);
        GPU(cudaGetLastError());
    };
    launch(0);GPU(cudaStreamSynchronize(stream));
    const auto old_acc=before.a.get(stream),old_m=before.m.get(stream),old_l=before.l.get(stream),old_out=before.out.get(stream);
    for(int queries:{2,4}) {
        if(c.rows==2 && queries==4)continue;
        after.clear(stream);launch(queries);GPU(cudaStreamSynchronize(stream));
        const auto new_acc=after.a.get(stream),new_m=after.m.get(stream),new_l=after.l.get(stream),new_out=after.out.get(stream);
    equal_arrays(old_acc,new_acc,label+" pacc");equal_arrays(old_m,new_m,label+" pm");
    equal_arrays(old_l,new_l,label+" pl");equal_arrays(old_out,new_out,label+" merge");
    for(const auto* v:{&new_acc,&new_m,&new_l,&new_out})guards(*v,label);
    // Existing scratch reserves eight queries; all unused rows/chunks must
    // remain byte-for-byte sentinel, not just the final allocation guards.
    bool inactive_acc=true,inactive_scalar=true,inactive_out=true;
    for(int row=0;row<8*H;++row)for(int chunk=0;chunk<chunk_stride;++chunk) {
        const int np=row<c.rows*H?(c.preview?end:c.position+c.t0+row/H+1):0;
        if(row<c.rows*H && chunk<grid && chunk*256<np)continue;
        const size_t at=GUARD+(size_t)row*chunk_stride+chunk;
        inactive_scalar &= new_m[at]==SENTINEL && new_l[at]==SENTINEL;
        const size_t begin=GUARD+((size_t)row*chunk_stride+chunk)*256;
        inactive_acc &= std::all_of(new_acc.begin()+begin,new_acc.begin()+begin+256,[](float v){return v==SENTINEL;});
    }
    for(int t=0;t<total_queries;++t)if(t<c.t0 || t>=c.t0+c.rows)
        for(int h=0;h<H;++h)for(int d=0;d<256;++d)
            inactive_out &= new_out[GUARD+((size_t)t*H+h)*256+d]==SENTINEL;
    require(inactive_scalar,label+": inactive max/sum sentinel");
    require(inactive_acc,label+": inactive pacc sentinel");
    require(inactive_out,label+": inactive output sentinel");
    // Independent FP64 attention on two heads, first/last queries and four
    // output dimensions. Both old and new are checked against the same oracle;
    // the full 64K partial buffers above are compared bitwise in every case.
    if(end<=1025)for(int r:{0,c.rows-1})for(int h:{0,H-1}) {
        const int t=c.t0+r,np=c.preview?end:c.position+t+1,kh=h/(H/HKV);
        std::vector<double> scores(np);double maximum=-INFINITY;
        for(int p=0;p<np;++p) {
            double sum=0;const size_t base=((size_t)kh*c.ctx+p)*256;
            for(int d=0;d<256;++d) {
                const float k=(float)key[base+d]*half_value(ks[(base+d)/64]);
                sum+=(double)q[((size_t)t*H+h)*256+d]*k;
            }
            scores[p]=sum/16.;maximum=std::max(maximum,scores[p]);
        }
        double denominator=0;for(double& s:scores){s=std::exp(s-maximum);denominator+=s;}
        for(int d:{0,17,127,255}) {
            double sum=0;for(int p=0;p<np;++p) {
                const size_t i=((size_t)kh*c.ctx+p)*256+d;
                const float v=(float)value[i]*half_value(vs[i/64]);sum+=scores[p]*v;
            }
            const size_t at=((size_t)t*H+h)*256+d;
            const double expected=sum/denominator/(1+std::exp(-(double)gate[at]));
            const double error=std::abs(new_out[GUARD+at]-expected);
            max_reference_error=std::max(max_reference_error,error);
            require(std::isfinite(old_out[GUARD+at]) && std::abs(old_out[GUARD+at]-expected)<3e-5,label+": old FP64 oracle");
            require(std::isfinite(new_out[GUARD+at]) && error<3e-5,label+": new FP64 oracle");
        }
    }
    if(benchmark && c.position>=65535 && !c.zero && !c.full_grid) {
        for(int i=0;i<8;++i){launch(0);launch(queries);}GPU(cudaStreamSynchronize(stream));
        std::vector<double> baseline,shared;
        for(int round=0;round<3;++round)for(bool which:{false,true,true,false}) {
            GPU(cudaStreamSynchronize(stream));const auto begin=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<repetitions;++repeat)launch(which?queries:0);
            GPU(cudaStreamSynchronize(stream));const auto end_time=std::chrono::steady_clock::now();
            const double us=std::chrono::duration<double,std::micro>(end_time-begin).count()/repetitions;
            (which?shared:baseline).push_back(us);
        }
        const double b=std::accumulate(baseline.begin(),baseline.end(),0.)/baseline.size();
        const double n=std::accumulate(shared.begin(),shared.end(),0.)/shared.size();
        std::printf("{\"rows\":%d,\"queries\":%d,\"mode\":\"%s\",\"heads\":%d,\"kv_heads\":%d,\"position\":%d,\"t0\":%d,\"ctx\":%d,\"grid_chunks\":%d,\"repetitions\":%d,\"ABBA_rounds\":3,\"baseline_host_us\":[",c.rows,queries,c.preview?"preview":"causal",H,HKV,c.position,c.t0,c.ctx,grid,repetitions);
        for(size_t i=0;i<baseline.size();++i)std::printf("%s%.6f",i?",":"",baseline[i]);
        std::printf("],\"shared_host_us\":[");for(size_t i=0;i<shared.size();++i)std::printf("%s%.6f",i?",":"",shared[i]);
        std::printf("],\"baseline_mean_us\":%.6f,\"shared_mean_us\":%.6f,\"change_percent\":%.6f}\n",b,n,(n/b-1)*100);
    }
    } // Query-tile variants; both checked against the same old buffers.
}
int main(int argc,char** argv) {
    try {
        bool benchmark=false;int repetitions=32,H=16,HKV=2;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--bench")benchmark=true;
            else if(a=="--reps" && i+1<argc)repetitions=std::stoi(argv[++i]);
            else if(a=="--heads" && i+1<argc)H=std::stoi(argv[++i]);
            else if(a=="--kv-heads" && i+1<argc)HKV=std::stoi(argv[++i]);
            else throw std::invalid_argument("Use --bench [--reps 32|64] [--heads 16 --kv-heads 2]");
        }
        require(repetitions==32 || repetitions==64,"Timing repetitions must be32 or64");
        require(H>0 && HKV>0 && H==8*HKV,"This probe requires GQA8; Clover/Qwen default H16/HKV2");
        cudaDeviceProp device{};GPU(cudaGetDeviceProperties(&device,0));
        int runtime_version=0;
#if defined(__HIPCC__)
        GPU(hipRuntimeGetVersion(&runtime_version));
        std::printf("{\"device\":\"%s\",\"arch\":\"%s\",\"runtime_version\":%d}\n",device.name,device.gcnArchName,runtime_version);
#else
        GPU(cudaRuntimeGetVersion(&runtime_version));
        std::printf("{\"device\":\"%s\",\"cuda_cc\":%d,\"runtime_version\":%d}\n",device.name,device.major*10+device.minor,runtime_version);
#endif
        cudaStream_t stream;GPU(cudaStreamCreate(&stream));int cases=0;
        for(int rows:{2,3,4}) {
            for(int position:{0,253,254,255,256,257,1023,1024})for(bool preview:{false,true}) {
                const int t0=(position==257 || position==1024)?3:0;
                one({rows,position,t0,4099,preview},H,HKV,stream,false,repetitions);++cases;
            }
            for(bool preview:{false,true}) {
                one({rows,65535,3,131081,preview},H,HKV,stream,benchmark,repetitions);++cases;
            }
            one({rows,255,1,4099,false,true},H,HKV,stream,false,repetitions);++cases;
        }
        one({4,65536,0,131081,true,true,true},H,HKV,stream,false,repetitions);++cases;
        for(bool preview:{false,true}) {
            one({4,65532,0,65536,preview},H,HKV,stream,false,repetitions);++cases;
        }
        cudaFuncAttributes old{},q2{},q4{};GPU(cudaFuncGetAttributes(&old,A::k_attn_part<2>));
        GPU(cudaFuncGetAttributes(&q2,Q::part<2>));GPU(cudaFuncGetAttributes(&q4,Q::part<4>));
        GPU(cudaStreamDestroy(stream));
        std::printf("{\"passed\":true,\"cases\":%d,\"checks\":%d,\"max_FP64_error\":%.9g,\"old_registers\":%d,\"old_shared_bytes\":%zu,\"Q2_registers\":%d,\"Q2_shared_bytes\":%zu,\"Q4_registers\":%d,\"Q4_shared_bytes\":%zu,\"timing\":\"host steady_clock, fenced, ABBA32or64; no HIP event primary\"}\n",cases,checks,max_reference_error,old.numRegs,old.sharedSizeBytes,q2.numRegs,q2.sharedSizeBytes,q4.numRegs,q4.sharedSizeBytes);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
