// Model-free regression of the activation-layout contract of GPT-OSS's MMQ.
// MXFP4 weights stay in GGUF form. First isolate layout/routing with exactly
// Q8-representable inputs, then independently validate ordinary Q8 scales/codes
// and compare each kernel with its own decoded-Q8 FP64 dot-product oracle.
// The latter does not demand identical float32-MMQ and float16-MMVQ scales.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "mmq.hpp"
#include "strata/gemma4/matvec.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace G = strata::gemma4;
namespace K = strata::kernels;
constexpr int kType = 39, kIn = 2880, kBlock = 32;
static unsigned cases = 0, checks = 0, real_cases = 0, large_cases = 0, rounding_boundary_codes = 0;
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)
static void need(bool b, const char* what) { ++checks; if (!b) { std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1); } }

struct Device {
    uint8_t* p = nullptr;
    size_t size;
    explicit Device(size_t n) : size(n) { CK(cudaMalloc(reinterpret_cast<void**>(&p), n + 256));CK(cudaMemset(p,0,n));CK(cudaMemset(p+n,0xA7,256)); }
    ~Device() { if (p) cudaFree(p); }
    template<class T> T* as() const { return reinterpret_cast<T*>(p); }
    void put(const void* data, size_t n) { need(n <= size,"host upload fits");CK(cudaMemcpy(p,data,n,cudaMemcpyHostToDevice)); }
    void guard() const { uint8_t h[256];CK(cudaMemcpy(h,p+size,256,cudaMemcpyDeviceToHost));for(auto c:h)need(c==0xA7,"device red zone"); }
};
static uint32_t mix(uint32_t x) { x ^= x >> 16;x *= 0x7feb352du;x ^= x >> 15;x *= 0x846ca68bu;return x ^ (x >> 16); }
static std::vector<uint8_t> weights(int outputs, int experts,size_t stride,size_t offset) {
    std::vector<uint8_t> w(stride*experts + 256,0);
    for(int e=0;e<experts;++e) for(int r=0;r<outputs;++r) for(int b=0;b<kIn/kBlock;++b) {
        auto* block = w.data()+size_t(e)*stride+offset+(size_t(r)*(kIn/kBlock)+b)*17;
        const uint32_t seed=mix(uint32_t(e*397+r*1031+b*53+19));
        block[0]=uint8_t(119+seed%5);
        for(int j=0;j<16;++j)block[j+1]=uint8_t(mix(seed+uint32_t(j)*0x9e3779b9u));
    }
    return w;
}
static std::vector<float> inputs(int rows, bool real) {
    std::vector<float> x(size_t(rows)*kIn);
    for(int r=0;r<rows;++r)for(int i=0;i<kIn;++i) {
        // Every group of 32 contains both extrema, so both quantizers' scales
        // are exactly 1/8 and every quantized code reproduces x exactly.
        const int j=i%32;
        const int v=j==0?127:j==1?-127:int(mix(uint32_t(r*991+i*17+7))%255)-127;
        if (!real) x[size_t(r)*kIn+i]=float(v)*0.125f;
        else {
            const uint32_t seed=mix(uint32_t(r*991+(i/32)*17+13));
            const float amplitude=0.01713f+float(seed%1000)*0.004317f;
            // Non-half-exact scales, awkward rounding near (but away from)
            // half-integers, both signs, and whole all-zero blocks.
            const float f=j==0?1.f:j==1?-0.99813f:
                j==2?(17.5f+0.0005f)/127.f:j==3?(-55.5f-0.0005f)/127.f:
                float(int(mix(seed+uint32_t(j)*1973u)&0xffffffu)-0x800000)/8388608.f;
            x[size_t(r)*kIn+i]=(seed%19==0)?0.f:amplitude*f;
        }
    }
    return x;
}
template<class T> static T load(const std::vector<uint8_t>& bytes,size_t at) {
    need(at+sizeof(T)<=bytes.size(),"packed Q8 read fits");T value;std::memcpy(&value,bytes.data()+at,sizeof(T));return value;
}
static void quant_code(int got,float x,float amax) {
    if(amax==0.f)return; // A zero scale makes any stored integer code exactly zero.
    const double exact=double(x)*127./double(amax);
    const int expected=int(std::round(exact));
    if(got==expected)return;
    // CUDA float reciprocal/multiply and the CPU double expression can round
    // opposite ways only at a Q8 half-integer. Never allow a distant code error.
    const double half_distance=std::abs(std::abs(exact)-std::floor(std::abs(exact))-0.5);
    need(std::abs(got-expected)==1 && half_distance<2e-5,"Q8 code matches independent nearest integer away from rounding boundary");
    ++rounding_boundary_codes;
}
static void decoded_quantizations(const Device& mmq,const Device& mmvq,const std::vector<float>& x,
                                  const std::vector<int32_t>& gather,int rows,
                                  std::vector<float>& x_mmq,std::vector<float>& x_mmvq) {
    std::vector<uint8_t> a(mmq.size),b(mmvq.size);
    CK(cudaMemcpy(a.data(),mmq.p,a.size(),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(),mmvq.p,b.size(),cudaMemcpyDeviceToHost));
    x_mmq.resize(x.size());x_mmvq.resize(x.size());
    for(int r=0;r<rows;++r)for(int block=0;block<kIn/32;++block) {
        float max_a=0.f,max_b=0.f;
        for(int j=0;j<32;++j) {
            max_a=std::max(max_a,std::abs(x[size_t(gather[r])*kIn+block*32+j]));
            max_b=std::max(max_b,std::abs(x[size_t(r)*kIn+block*32+j]));
        }
        // GGML D4 is transposed by 128-value block then compact input row:
        // float scales[4], signed byte codes[128], total144bytes. Native Q81
        // is a row-major36byte block: half scale, half sum, signed codes[32].
        const size_t ai=(size_t(block/4)*rows+r)*144;
        const size_t bi=(size_t(r)*(kIn/32)+block)*36;
        const float da=load<float>(a,ai+size_t(block%4)*4);
        const uint16_t bits=load<uint16_t>(b,bi);__half half;std::memcpy(&half,&bits,2);
        const float db=__half2float(half),ea=max_a/127.f,eb=max_b/127.f;
        need(std::isfinite(da) && std::abs(double(da)-ea)<=1e-12+1e-6*std::abs(double(ea)),"MMQ D4 scale matches independent amax/127 within float32 rounding");
        need(std::isfinite(db) && std::abs(double(db)-eb)<=2e-8+0.0006*std::abs(double(eb)),"MMVQ scale matches independent amax/127 within float16 rounding");
        if(max_a==0.f)need(da==0.f,"MMQ all-zero block scale");
        if(max_b==0.f)need(db==0.f,"MMVQ all-zero block scale");
        for(int j=0;j<32;++j) {
            const int qa=load<int8_t>(a,ai+16+size_t(block%4)*32+j),qb=load<int8_t>(b,bi+4+j);
            quant_code(qa,x[size_t(gather[r])*kIn+block*32+j],max_a);
            quant_code(qb,x[size_t(r)*kIn+block*32+j],max_b);
            x_mmq[size_t(r)*kIn+block*32+j]=da*float(qa);
            x_mmvq[size_t(r)*kIn+block*32+j]=db*float(qb);
        }
    }
}
static double reference(const std::vector<uint8_t>& w,const std::vector<float>& x,
                        int outputs,int expert,int output,int input,size_t stride,size_t offset) {
    static constexpr int values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    const uint8_t* row=w.data()+size_t(expert)*stride+offset+size_t(output)*(kIn/32)*17;
    double sum=0;
    for(int i=0;i<kIn;++i) {
        const uint8_t* b=row+size_t(i/32)*17;
        const int j=i%32,code=(b[1+j%16]>>(j<16?0:4))&15;
        const double weight=std::ldexp(double(values[code]),int(b[0])-128);
        sum+=weight*double(x[size_t(input)*kIn+i]);
    }
    return sum;
}
static void compare(float got,double expected,const char* path,int outputs,int rows,int r,int c) {
    const double error=std::abs(double(got)-expected),tol=0.01+5e-5*std::abs(expected);
    ++checks;
    if(!std::isfinite(got)||error>tol) {
        std::fprintf(stderr,"FAIL %s outputs=%d rows=%d row=%d column=%d got=%.9g expected=%.12g error=%.9g tolerance=%.9g\n",path,outputs,rows,r,c,got,expected,error,tol);
        std::exit(1);
    }
}
static void run_case(int outputs,int rows,bool routed,cudaStream_t stream,bool real=false,bool repeat_gather=false) {
    const int experts=routed?3:1;
    const size_t payload=size_t(outputs)*(kIn/32)*17;
    const size_t gu_bytes=size_t(2*kIn)*(kIn/32)*17,down_bytes=size_t(kIn)*(kIn/32)*17;
    // The real engine stores gate/up + down +256tail in each cached expert
    // slot, rounded to lcm(17,8)=136bytes. Exercise its actual strided channels
    // and the nonzero down-matrix offset in the repeated-routing large case.
    const size_t stride=repeat_gather?((gu_bytes+down_bytes+256+135)/136)*136:payload;
    const size_t offset=repeat_gather&&outputs==kIn?gu_bytes:0;
    need(offset+payload<=stride,"weight matrix fits expert slot");
    const auto w=weights(outputs,experts,stride,offset);const auto x=inputs(rows,real);
    std::vector<int32_t> gather(rows+128,0),dest(rows+128,0),bounds(experts+1);
    // A routed prompt repeats each input token for its selected experts. The
    // large case uses repeated gather IDs as well as a separate output map;
    // the original cases retain their full input/output permutations.
    // Each source token appears at most once within an expert, and can recur
    // across experts. This mirrors top-k routing rather than duplicating one
    // token multiple times for the same expert.
    const int input_rows=repeat_gather?(rows+experts-1)/experts:rows;
    for(int r=0;r<rows;++r) {gather[r]=routed?(r*5+7)%input_rows:r;dest[r]=routed?(r*17+3)%rows:r;}
    for(int e=0;e<=experts;++e)bounds[e]=rows*e/experts;
    const int max_rows=(rows+experts-1)/experts;
    Device dw(w.size()),dx(x.size()*sizeof(float)),dq(G::Mmq::q8_bytes(rows,kIn)),
           dg(gather.size()*sizeof(int32_t)),dd(dest.size()*sizeof(int32_t)),db(bounds.size()*sizeof(int32_t)),
           dy(size_t(rows)*outputs*sizeof(float)),q81(size_t(rows)*K::gemma_native_q8_1_bytes(kIn)),mv(size_t(outputs)*sizeof(float));
    dw.put(w.data(),w.size());dx.put(x.data(),x.size()*sizeof(float));dg.put(gather.data(),gather.size()*sizeof(int32_t));dd.put(dest.data(),dest.size()*sizeof(int32_t));db.put(bounds.data(),bounds.size()*sizeof(int32_t));
    need(G::Mmq::fits(kType,outputs,kIn),"Q8 MMQ shape eligible");
    G::Mmq mmq;
    G::Mmq::quantize(dx.as<float>(),routed?dg.as<int32_t>():nullptr,dq.p,kType,kIn,kIn,rows,stream);
    mmq.mul(dw.p+offset,kType,outputs,kIn,stride,experts,dq.p,routed?db.as<int32_t>():nullptr,routed?dd.as<int32_t>():nullptr,rows,max_rows,dy.as<float>(),outputs,stream);
    K::gemma_native_quantize_q8_1(dx.as<float>(),q81.p,kIn*rows,1,stream);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> x_mmq,x_mmvq;
    if(real)decoded_quantizations(dq,q81,x,gather,rows,x_mmq,x_mmvq);
    std::vector<float> y(size_t(rows)*outputs);CK(cudaMemcpy(y.data(),dy.p,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float v:y)need(std::isfinite(v),"all MMQ output finite");
    const int columns[]={0,1,31,32,63,64,127,128,255,256,1023,outputs/2,outputs-2,outputs-1};
    for(int r=0;r<rows;++r) {
        const int e=routed?int(std::upper_bound(bounds.begin(),bounds.end(),r)-bounds.begin())-1:0;
        for(int c:columns)compare(y[size_t(dest[r])*outputs+c],reference(w,real?x_mmq:x,outputs,e,c,real?r:gather[r],stride,offset),real?"MMQ/own Q8 CPU":"MMQ/CPU",outputs,rows,r,c);
    }
    const int sampled_rows[]={0,1,rows/2,rows-1};
    for(int r:sampled_rows) {
        const int e=routed?int(std::upper_bound(bounds.begin(),bounds.end(),r)-bounds.begin())-1:0;
        const size_t qoff=size_t(gather[r])*K::gemma_native_q8_1_bytes(kIn);
        K::gemma_native_mmvq(kType,dw.p+size_t(e)*stride+offset,q81.p+qoff,mv.as<float>(),kIn,outputs,1,stream);
        CK(cudaStreamSynchronize(stream));std::vector<float> v(outputs);CK(cudaMemcpy(v.data(),mv.p,v.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int c:columns)compare(v[c],reference(w,real?x_mmvq:x,outputs,e,c,gather[r],stride,offset),real?"MMVQ/own Q8 CPU":"MMVQ/CPU",outputs,rows,r,c);
    }
    // Kernels may fill their allocated padding, never the buffers after it.
    for(const Device* p:{&dw,&dx,&dq,&dg,&dd,&db,&dy,&q81,&mv})p->guard();
    std::vector<float> unchanged(x.size());CK(cudaMemcpy(unchanged.data(),dx.p,unchanged.size()*sizeof(float),cudaMemcpyDeviceToHost));need(unchanged==x,"input vectors unchanged");
    ++cases;
    if(real)++real_cases;
    if(rows>=257)++large_cases;
}
int main() {
    cudaStream_t stream=nullptr;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int outputs:{2880,5760})for(int rows:{32,69})for(bool routed:{false,true})run_case(outputs,rows,routed,stream);
    for(int outputs:{2880,5760})for(bool routed:{false,true})run_case(outputs,69,routed,stream,true);
    for(int outputs:{2880,5760}) {
        run_case(outputs,257,false,stream,true);
        run_case(outputs,769,true,stream,true,true); // expert counts256/256/257; repeated input token IDs
    }
    // Fallback2880-output configurations selectJ128 for257rows. Nonfallback
    //5760-output can selectJ96 there;511rows uniquely makesJ128 fewer tiles.
    run_case(5760,511,false,stream,true);
    CK(cudaStreamDestroy(stream));std::printf("PASS GPT-OSS MXFP4/Q8 MMQ cases=%u real_input_cases=%u large_row_cases=%u checks=%u rounding_boundary_codes=%u actual_shapes=2880x2880,5760x2880 prompt_rows=32,69,257,511,769 independent_cpu=1 routed_ids=1 repeated_input_ids=1\n",cases,real_cases,large_cases,checks,rounding_boundary_codes);
    return 0;
}
