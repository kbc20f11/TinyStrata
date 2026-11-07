// Production K-quant MMVQ/grouped kernels: bitwise old-path parity and an
// independent scalar decode of the stored weight/activation values.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/iq_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace K = strata::kernels;
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
static unsigned cases = 0;
static uint32_t rng = 0xA710384Du;
static uint32_t random_word() { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static double maximum_error = 0;
static uint64_t hash = 1469598103934665603ull;
static uint16_t half_bits(float f) { __half h = __float2half(f); uint16_t b; std::memcpy(&b,&h,2); return b; }
static double half_value(uint16_t h) {
    const int e = (h >> 10) & 31, f = h & 1023;
    const double v = e ? std::ldexp(1.0 + f/1024.0, e-15) : std::ldexp(double(f), -24);
    return h & 0x8000 ? -v : v;
}
static uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v,p,2); return v; }
static uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v,p,4); return v; }
struct Q8 { uint16_t d, sum; int8_t q[32]; };
static_assert(sizeof(Q8) == 36, "Q8_1 layout");
struct Device {
    uint8_t* base = nullptr; size_t bytes;
    Device(size_t n, cudaStream_t s) : bytes(n) { CK(cudaMalloc((void**)&base,n+128)); CK(cudaMemsetAsync(base,0xA7,n+128,s)); }
    ~Device() { if (base) cudaFree(base); }
    void* data() const { return base+64; }
    template<class T> T* as() const { return static_cast<T*>(data()); }
    void put(const void* p, size_t n, cudaStream_t s) { CHECK(n<=bytes); CK(cudaMemcpyAsync(data(),p,n,cudaMemcpyHostToDevice,s)); }
    void guards() const {
        uint8_t a[64], b[64]; CK(cudaMemcpy(a,base,64,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b,base+64+bytes,64,cudaMemcpyDeviceToHost));
        for(int i=0;i<64;++i) CHECK(a[i]==0xA7 && b[i]==0xA7);
    }
};
static int block_values(int type) { return type==6 ? 32 : 256; }
static size_t block_bytes(int type) { return type==6 ? 22 : type==10 ? 84 : 110; }
static std::vector<uint8_t> weights(int type, int rows, int n, int salt) {
    const int values=block_values(type), blocks=n/values;
    std::vector<uint8_t> w(size_t(rows)*blocks*block_bytes(type));
    for(auto& b:w)b=uint8_t(random_word());
    for(int r=0;r<rows;++r)for(int b=0;b<blocks;++b) {
        uint8_t* p=w.data()+(size_t(r)*blocks+b)*block_bytes(type);
        const uint16_t d=half_bits(std::ldexp(((r+b+salt)&1) ? -1.0f : 1.0f,-10+(r+b+salt)%4));
        const size_t off=type==6 ? 0 : type==10 ? 80 : 108;
        std::memcpy(p+off,&d,2);
        if(type==10) { const uint16_t m=half_bits(1.0f/2048); std::memcpy(p+82,&m,2); }
    }
    return w;
}
static double weight_value(int type, const uint8_t* p, int i) {
    if(type==6) {
        const int code=((p[6+(i&15)]>>(4*(i/16)))&15)|int(((u32(p+2)>>i)&1u)<<4);
        return half_value(u16(p))*(code-16);
    }
    const int bank=i/128, shift=(i%128)/32, lane=i%32, scale=8*bank+2*shift+lane/16;
    if(type==10) {
        const int sc=p[scale], code=(p[16+32*bank+lane]>>(2*shift))&3;
        return half_value(u16(p+80))*(sc&15)*code-half_value(u16(p+82))*(sc>>4);
    }
    const int sc=((p[96+scale%8]>>(4*(scale/8)))&15)|(((p[104+scale%4]>>(2*(scale/4)))&3)<<4);
    const int code=((p[32+32*bank+lane]>>(2*shift))&3)-((p[lane]&(1u<<(4*bank+shift))) ? 0 : 4);
    return half_value(u16(p+108))*(sc-32)*code;
}
static std::vector<Q8> activations(int columns, int n) {
    std::vector<Q8> q(size_t(columns)*(n/32));
    for(size_t b=0;b<q.size();++b) {
        q[b].d=half_bits(std::ldexp(1.0f,-7+int(b%3)));
        int sum=0;
        for(int j=0;j<32;++j) { q[b].q[j]=int8_t(int(random_word()%255)-127); sum+=q[b].q[j]; }
        // Deliberately distinguish Q8_1's original-input sum from d*sum(q).
        q[b].sum=half_bits(float(half_value(q[b].d)*sum)+0.25f);
    }
    return q;
}
static void identical(const std::vector<float>& a, const std::vector<float>& b, const char* phase, int g, int d, int n, int columns) {
    for(size_t i=0;i<a.size();++i) {
        uint32_t x,y; std::memcpy(&x,&a[i],4);std::memcpy(&y,&b[i],4);
        if(x!=y || !std::isfinite(a[i])) {
            std::fprintf(stderr,"FAIL %s gu=%d down=%d n=%d columns=%d index=%zu old=%08x new=%08x\n",phase,g,d,n,columns,i,x,y);std::exit(1);
        }
        hash=(hash^x)*1099511628211ull;
    }
}
static void matrix_case(int type, int n, int columns, cudaStream_t s) {
    constexpr int rows=13;
    auto w=weights(type,rows,n,columns);auto q=activations(columns,n);
    Device dw(w.size(),s),dq(q.size()*sizeof(Q8),s),out(size_t(rows)*columns*4,s);
    dw.put(w.data(),w.size(),s);dq.put(q.data(),dq.bytes,s);
    std::vector<float> previous,got(size_t(rows)*columns);
    for(bool old:{true,false}) {
        K::iq_set_old_kernels(old);CK(cudaMemsetAsync(out.data(),0xFF,out.bytes,s));
        K::iq_mmvq(type,dw.data(),dq.data(),out.as<float>(),n,rows,columns,s);CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(got.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost));
        for(int c=0;c<columns;++c)for(int r=0;r<rows;++r) {
            double expected=0,bound=0;
            for(int i=0;i<n;++i) {
                const auto& a=q[size_t(c)*(n/32)+i/32];
                const uint8_t* wb=w.data()+size_t(r)*K::iq_row_bytes(type,n)+(i/block_values(type))*block_bytes(type);
                const double term=weight_value(type,wb,i%block_values(type))*half_value(a.d)*a.q[i%32];
                expected+=term;bound+=std::fabs(term);
            }
            const double error=std::fabs(got[size_t(c)*rows+r]-expected),limit=2e-6*std::max(1.0,bound);
            if(!std::isfinite(got[size_t(c)*rows+r]) || error>limit) {
                std::fprintf(stderr,"FAIL scalar type=%d n=%d cols=%d old=%d row=%d col=%d got=%.9g expected=%.12g error=%.9g limit=%.9g\n",type,n,columns,int(old),r,c,got[size_t(c)*rows+r],expected,error,limit);std::exit(1);
            }
            maximum_error=std::max(maximum_error,error);
        }
        if(old)previous=got;else identical(previous,got,"MMVQ",type,type,n,columns);
    }
    dw.guards();dq.guards();out.guards();++cases;
}
static void grouped_case(int gu, int down, int n, int ff, int entries, int grid_groups, bool legacy, cudaStream_t s) {
    constexpr int groups=3,cap_groups=5;
    const auto L=K::native_expert_layout(gu,down,n,ff);CHECK(K::native_expert_supported(gu,down,n,ff));
    std::vector<uint8_t> w(L.bytes*groups);
    for(int g=0;g<groups;++g) {
        const auto up=weights(gu,2*ff,n,g),d=weights(down,n,ff,g+7);
        std::memcpy(w.data()+g*L.bytes,up.data(),up.size());std::memcpy(w.data()+g*L.bytes+L.down_off,d.data(),d.size());
    }
    auto q=activations(entries,n);
    std::vector<unsigned long long> ptr(cap_groups);
    const std::vector<int32_t> starts{0,1,1,entries};const int32_t count=groups;
    std::vector<int32_t> dst(entries),tok(entries);
    for(int e=0;e<entries;++e) { dst[e]=entries-1-e;tok[e]=(e+2)%entries; }
    Device dw(w.size(),s),dq(q.size()*sizeof(Q8),s),dp(ptr.size()*8,s),ds(starts.size()*4,s),dc(4,s),dd(entries*4,s),dt(entries*4,s);
    Device scratch(K::native_expert_scratch_bytes(entries,ff),s),out(size_t(entries)*n*4,s);
    for(int g=0;g<groups;++g)ptr[g]=(unsigned long long)(uintptr_t)(dw.as<uint8_t>()+g*L.bytes);
    dw.put(w.data(),w.size(),s);dq.put(q.data(),dq.bytes,s);dp.put(ptr.data(),dp.bytes,s);ds.put(starts.data(),ds.bytes,s);
    dc.put(&count,4,s);dd.put(dst.data(),dd.bytes,s);dt.put(tok.data(),dt.bytes,s);
    std::vector<float> previous,got(size_t(entries)*n);
    K::native_grouped_set_v1(legacy);K::native_expert_set_mode(0,0);
    for(bool old:{true,false}) {
        K::iq_set_old_kernels(old);CK(cudaMemsetAsync(out.data(),0xFF,out.bytes,s));CK(cudaMemsetAsync(scratch.data(),0,scratch.bytes,s));
        K::native_expert_grouped(L,dp.as<unsigned long long>(),ds.as<int32_t>(),dc.as<int32_t>(),dd.as<int32_t>(),dt.as<int32_t>(),cap_groups,entries,dq.data(),scratch.data(),out.as<float>(),s,grid_groups);
        CK(cudaStreamSynchronize(s));CK(cudaMemcpy(got.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost));
        if(old)previous=got;else identical(previous,got,"grouped",gu,down,n,entries);
    }
    for(auto* p:{&dw,&dq,&dp,&ds,&dc,&dd,&dt,&scratch,&out})p->guards();++cases;
}
struct BenchTimes { double old_ms=0,new_ms=0; };
template<class Launch>
static BenchTimes timed_pair(Launch launch, cudaStream_t s) {
    constexpr int repeats=16;
    for(bool old:{true,false}) {
        K::iq_set_old_kernels(old);
        for(int i=0;i<3;++i)launch();
    }
    CK(cudaStreamSynchronize(s));
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    BenchTimes result;
    // Equal counts on the same hot buffers; report each API call's GPU time.
    for(bool old:{true,false,false,true}) {
        K::iq_set_old_kernels(old);CK(cudaEventRecord(begin,s));
        for(int i=0;i<repeats;++i)launch();
        CK(cudaEventRecord(end,s));CK(cudaEventSynchronize(end));CK(cudaGetLastError());
        float elapsed=0;CK(cudaEventElapsedTime(&elapsed,begin,end));
        (old ? result.old_ms : result.new_ms)+=double(elapsed)/(2*repeats);
    }
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
    return result;
}
static void bench_mmvq(int type, int input, int output, int columns, cudaStream_t s) {
    auto w=weights(type,output,input,columns);auto q=activations(columns,input);
    Device dw(w.size(),s),dq(q.size()*sizeof(Q8),s),out(size_t(output)*columns*4,s);
    dw.put(w.data(),w.size(),s);dq.put(q.data(),dq.bytes,s);CK(cudaStreamSynchronize(s));
    const auto t=timed_pair([&]{K::iq_mmvq(type,dw.data(),dq.data(),out.as<float>(),input,output,columns,s);},s);
    std::printf("BENCH api=mmvq type=%d input=%d output=%d columns=%d old_ms=%.6f new_ms=%.6f change_pct=%+.3f\n",
                type,input,output,columns,t.old_ms,t.new_ms,100*(t.new_ms/t.old_ms-1));
    std::fflush(stdout);
}
static void bench_grouped(int gu, int down, int columns, cudaStream_t s) {
    constexpr int n=2048,ff=512;
    const auto L=K::native_expert_layout(gu,down,n,ff);CHECK(K::native_expert_supported(gu,down,n,ff));
    auto up=weights(gu,2*ff,n,1),d=weights(down,n,ff,7);auto q=activations(columns,n);
    std::vector<uint8_t> w(L.bytes);
    std::memcpy(w.data(),up.data(),up.size());std::memcpy(w.data()+L.down_off,d.data(),d.size());
    std::vector<int32_t> dst(columns),tok(columns);
    for(int c=0;c<columns;++c)dst[c]=tok[c]=c;
    const int32_t starts[2]={0,columns},count=1;
    Device dw(w.size(),s),dq(q.size()*sizeof(Q8),s),dp(8,s),ds(8,s),dc(4,s),dd(columns*4,s),dt(columns*4,s);
    Device scratch(K::native_expert_scratch_bytes(columns,ff),s),out(size_t(columns)*n*4,s);
    const auto pointer=(unsigned long long)(uintptr_t)dw.data();
    dw.put(w.data(),w.size(),s);dq.put(q.data(),dq.bytes,s);dp.put(&pointer,8,s);ds.put(starts,8,s);
    dc.put(&count,4,s);dd.put(dst.data(),dd.bytes,s);dt.put(tok.data(),dt.bytes,s);CK(cudaStreamSynchronize(s));
    K::native_grouped_set_v1(false);K::native_expert_set_mode(0,0);
    const auto t=timed_pair([&]{
        K::native_expert_grouped(L,dp.as<unsigned long long>(),ds.as<int32_t>(),dc.as<int32_t>(),dd.as<int32_t>(),dt.as<int32_t>(),1,columns,dq.data(),scratch.data(),out.as<float>(),s,1);
    },s);
    std::printf("BENCH api=grouped gu=%d down=%d input=%d ff=%d columns=%d old_ms=%.6f new_ms=%.6f change_pct=%+.3f\n",
                gu,down,n,ff,columns,t.old_ms,t.new_ms,100*(t.new_ms/t.old_ms-1));
    std::fflush(stdout);
}
static void benchmark(cudaStream_t s) {
    std::puts("BENCH protocol=hot_buffers_abba repeats=16 warmup=3 GPU_event_ms_per_API_call; microbenchmark only");
    for(int type:{6,10,11})for(int columns:{1,2,4,8}) {
        bench_mmvq(type,2048,512,columns,s);
        bench_mmvq(type,512,2048,columns,s);
    }
    for(int columns:{1,2,4,8}) {
        bench_grouped(10,11,columns,s);bench_grouped(11,10,columns,s);bench_grouped(6,6,columns,s);
    }
}
int main(int argc, char** argv) {
    const bool bench=argc==2 && std::strcmp(argv[1],"--bench")==0;
    if(argc>1 && !bench) { std::fputs("usage: native-kq-grouped-parity [--bench]\n",stderr);return 2; }
    int count=0;CK(cudaGetDeviceCount(&count));if(!count)return 77;
    cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    if(bench)benchmark(s);
    else {
        for(int type:{6,10,11})for(int n:{256,512,768,2048,2560})for(int columns:{1,2,3,4,7,8})matrix_case(type,n,columns,s);
        for(int entries:{1,3,9})for(bool legacy:{false,true}) {
            grouped_case(10,11,256,256,entries,1,legacy,s);
            grouped_case(11,10,256,256,entries,5,legacy,s);
            grouped_case(6,6,256,256,entries,1,legacy,s);
            grouped_case(10,11,2048,512,entries,5,legacy,s);
            grouped_case(10,11,2560,512,entries,5,legacy,s);
        }
    }
    K::iq_set_old_kernels(false);K::native_grouped_set_v1(false);K::native_expert_set_mode(-1,0);CK(cudaStreamDestroy(s));
    if(!bench)std::printf("PASS K-quant MMVQ/grouped cases=%u bitwise parity, max scalar error=%.9g hash=%016llx\n",cases,maximum_error,(unsigned long long)hash);
}
#endif
