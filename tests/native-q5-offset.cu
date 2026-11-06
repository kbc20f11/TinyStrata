// Q5_0 offset regression: compare production MMVQ and grouped Expert down
// products with an independent scalar reference of the stored quantized values.
// Zero-valued Q5 rows make the old original-input-sum correction observable.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: Q5 offset test requires CUDA or HIP"); return 77; }
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
#define CK(x) do { const auto error=(x); if(error!=cudaSuccess) { \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(error));std::exit(2); } } while(0)
#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1); } } while(0)

#pragma pack(push,1)
struct Q5 { uint16_t d; uint32_t high; uint8_t low[16]; };
struct Q8 { uint16_t d, sum; int8_t q[32]; };
#pragma pack(pop)
static_assert(sizeof(Q5)==22 && sizeof(Q8)==36,"GGUF block layout");
constexpr size_t GUARD=64;
constexpr uint8_t POISON=0xA7;
unsigned cases=0, old_rejected=0;
double maximum_error=0;

double half_value(uint16_t h) {
    const unsigned exponent=(h>>10)&31, fraction=h&1023;
    const double value=exponent ? std::ldexp(1.0+fraction/1024.0,int(exponent)-15) : std::ldexp(double(fraction),-24);
    return h&0x8000 ? -value : value;
}
uint16_t half_bits(float value) {
    const __half h=__float2half(value); uint16_t bits;std::memcpy(&bits,&h,2);return bits;
}
int weight_code(const Q5& block,int i) {
    return ((block.low[i&15]>>((i>>4)*4))&15)|int(((block.high>>i)&1u)<<4);
}
Q5 make_weight(int row,int block,int salt) {
    Q5 result{};
    result.d=half_bits(std::ldexp((row+salt)&1 ? -1.0f:1.0f,-6+((row+block+salt)%4)*2));
    for(int i=0;i<32;++i) {
        // The first row is exactly zero, regardless of activation quantization.
        const unsigned code=row==0 ? 16u : unsigned((i*11+row*7+block*3+salt*5)&31);
        result.low[i&15]|=uint8_t((code&15u)<<((i>>4)*4));
        result.high|=((code>>4)&1u)<<i;
    }
    return result;
}
float activation(int index,int pattern) {
    const float jitter=float((index*37+13)%101)/100.0f;
    switch(pattern) {
    case 0:return 0.13f+0.73f*jitter;           // positive, like SwiGLU products
    case 1:return -0.17f-0.91f*jitter;          // negative nonzero mean
    case 2:return (index&1 ? -1.0f:1.0f)*(0.09f+0.31f*jitter);
    case 3:return index%32==0 ? 3.7f : 0.019f; // a large quantization residual in the sum
    default:return 0.0f;
    }
}
struct Device {
    uint8_t* allocation=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n) { CK(cudaMalloc((void**)&allocation,n+2*GUARD));CK(cudaMemset(allocation,POISON,n+2*GUARD)); }
    ~Device() { if(allocation)cudaFree(allocation); }
    void* data() const { return allocation+GUARD; }
    template<class T>T* as() const { return static_cast<T*>(data()); }
    void put(const void* source,size_t n) const { REQUIRE(n<=bytes);CK(cudaMemcpy(data(),source,n,cudaMemcpyHostToDevice)); }
    std::vector<uint8_t> snapshot() const {
        std::vector<uint8_t> result(bytes+2*GUARD);CK(cudaMemcpy(result.data(),allocation,result.size(),cudaMemcpyDeviceToHost));return result;
    }
    void guards() const {
        auto values=snapshot();
        REQUIRE(std::all_of(values.begin(),values.begin()+GUARD,[](uint8_t v){return v==POISON;}));
        REQUIRE(std::all_of(values.end()-GUARD,values.end(),[](uint8_t v){return v==POISON;}));
    }
};
struct Reference { double value=0, bound=0, old=0; };
Reference reference(const Q5* weights,const Q8* activations,int blocks) {
    Reference result;
    for(int block=0;block<blocks;++block) {
        const auto& w=weights[block];const auto& x=activations[block];
        int64_t centered=0,unshifted=0;
        for(int i=0;i<32;++i) {
            const int q=weight_code(w,i), a=x.q[i];
            centered+=int64_t(q-16)*a;unshifted+=int64_t(q)*a;
            result.bound+=std::fabs(half_value(w.d)*half_value(x.d)*double((q-16)*a));
        }
        result.value+=half_value(w.d)*half_value(x.d)*double(centered);
        // Independent whole-block prediction of the rejected formula.
        result.old+=half_value(w.d)*(half_value(x.d)*double(unshifted)-16.0*half_value(x.sum));
    }
    return result;
}
void compare(float actual,const Reference& expected,const char* path,int row,int column) {
    const double tolerance=2e-6*std::max(1.0,expected.bound);
    const double error=std::fabs(double(actual)-expected.value);
    if(!std::isfinite(actual)||error>tolerance) {
        std::fprintf(stderr,"%s row=%d column=%d actual=%.9g expected=%.12g old=%.12g error=%.9g limit=%.9g\n",
            path,row,column,actual,expected.value,expected.old,error,tolerance);std::exit(1);
    }
    maximum_error=std::max(maximum_error,error);
    if(std::fabs(expected.old-expected.value)>tolerance)++old_rejected;
    if(row==0)REQUIRE(actual==0.0f);
}

void matrix_case(int input,int columns,int pattern,cudaStream_t stream) {
    constexpr int ROWS=7;const int blocks=input/32;
    std::vector<Q5> weights(ROWS*blocks);
    for(int r=0;r<ROWS;++r)for(int b=0;b<blocks;++b)weights[r*blocks+b]=make_weight(r,b,pattern);
    std::vector<float> source(columns*input);
    for(size_t i=0;i<source.size();++i)source[i]=activation(int(i),pattern)*std::ldexp(1.0f,int(i/input)%3-1);
    Device w(weights.size()*sizeof(Q5)),x(source.size()*4),q(size_t(columns*blocks)*sizeof(Q8)),out(size_t(columns*ROWS)*4);
    CK(cudaMemset(out.data(),0xFF,out.bytes)); // an unwritten output must fail, including zero fixtures
    w.put(weights.data(),w.bytes);x.put(source.data(),x.bytes);
    const auto w_before=w.snapshot(),x_before=x.snapshot();
    K::quantize_q8_1_rows(x.as<float>(),columns,input,q.data(),stream);CK(cudaStreamSynchronize(stream));
    const auto q_before=q.snapshot();
    std::vector<Q8> stored(columns*blocks);CK(cudaMemcpy(stored.data(),q.data(),q.bytes,cudaMemcpyDeviceToHost));
    K::iq_mmvq(6,w.data(),q.data(),out.as<float>(),input,ROWS,columns,stream);CK(cudaStreamSynchronize(stream));
    std::vector<float> result(columns*ROWS);CK(cudaMemcpy(result.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost));
    for(int c=0;c<columns;++c)for(int r=0;r<ROWS;++r)
        compare(result[c*ROWS+r],reference(weights.data()+r*blocks,stored.data()+c*blocks,blocks),"MMVQ",r,c);
    REQUIRE(w.snapshot()==w_before && x.snapshot()==x_before && q.snapshot()==q_before);
    w.guards();x.guards();q.guards();out.guards();++cases;
}

template<int N=256,int FF=256>
void grouped_case(int entries,int pattern,int grid_groups,bool legacy,cudaStream_t stream) {
    constexpr int GROUPS=3,CAP_GROUPS=5;
    const auto layout=K::native_expert_layout(8,6,N,FF);
    REQUIRE(K::native_expert_supported(8,6,N,FF));
    const size_t f=size_t(entries)*FF*4,aligned=(f+255)&~size_t(255);
    const size_t scratch_bytes=K::native_expert_scratch_bytes(entries,FF);
    Device blobs(layout.bytes*GROUPS),scratch(scratch_bytes),out(size_t(entries)*N*4);
    CK(cudaMemset(out.data(),0xFF,out.bytes));
    std::vector<uint8_t> host(layout.bytes*GROUPS,0);
    // Q8_0 gate/up weights; independent down reference reads only the stored
    // intermediate Q8_1 image produced by the ordinary SwiGLU path.
    for(int g=0;g<GROUPS;++g) {
        for(int r=0;r<2*FF;++r)for(int b=0;b<N/32;++b) {
            uint8_t* block=host.data()+g*layout.bytes+size_t(r)*layout.gu_row+b*34;
            const auto d=half_bits(1.0f/128);std::memcpy(block,&d,2);
            for(int i=0;i<32;++i)block[2+i]=uint8_t(int8_t(1+((r+i+b+g)%3)));
        }
        for(int r=0;r<N;++r)for(int b=0;b<FF/32;++b) {
            const Q5 w=make_weight(r,b,pattern+g);
            std::memcpy(host.data()+g*layout.bytes+layout.down_off+size_t(r)*layout.d_row+b*sizeof(Q5),&w,sizeof(w));
        }
    }
    blobs.put(host.data(),host.size());const auto weights_before=blobs.snapshot();
    std::vector<float> input(size_t(entries)*N);
    for(size_t i=0;i<input.size();++i)input[i]=activation(int(i),pattern);
    Device x(input.size()*4),q(size_t(entries)*(N/32)*sizeof(Q8));x.put(input.data(),x.bytes);
    K::quantize_q8_1_rows(x.as<float>(),entries,N,q.data(),stream);CK(cudaStreamSynchronize(stream));
    const auto input_before=x.snapshot(),q_before=q.snapshot();
    std::vector<unsigned long long> ptrs(CAP_GROUPS,0);
    for(int g=0;g<GROUPS;++g)ptrs[g]=(unsigned long long)(uintptr_t)(blobs.as<uint8_t>()+g*layout.bytes);
    const int32_t count=GROUPS;const std::vector<int32_t> starts{0,1,1,entries};
    std::vector<int32_t> destination(entries),tokens(entries);
    for(int e=0;e<entries;++e) { destination[e]=(e*2+1)%entries;tokens[e]=entries-e-1; }
    Device pointers(ptrs.size()*8),boundaries(starts.size()*4),ng(4),dst(entries*4),tok(entries*4);
    pointers.put(ptrs.data(),pointers.bytes);boundaries.put(starts.data(),boundaries.bytes);ng.put(&count,4);
    dst.put(destination.data(),dst.bytes);tok.put(tokens.data(),tok.bytes);
    const auto pointers_before=pointers.snapshot(),boundaries_before=boundaries.snapshot(),ng_before=ng.snapshot();
    const auto dst_before=dst.snapshot(),tok_before=tok.snapshot();
    // The public benchmark phase selector provides the ordinary intermediate
    // image. Its offset is the production scratch contract (three aligned rows).
    K::native_grouped_set_v1(legacy);K::native_expert_set_mode(0,1);
    K::native_expert_grouped(layout,pointers.as<unsigned long long>(),boundaries.as<int32_t>(),ng.as<int32_t>(),dst.as<int32_t>(),
        tok.as<int32_t>(),CAP_GROUPS,entries,q.data(),scratch.data(),out.as<float>(),stream,grid_groups);
    CK(cudaStreamSynchronize(stream));
    std::vector<Q8> intermediate(size_t(entries)*(FF/32));
    CK(cudaMemcpy(intermediate.data(),scratch.as<uint8_t>()+3*aligned,intermediate.size()*sizeof(Q8),cudaMemcpyDeviceToHost));
    const auto scratch_before=scratch.snapshot();
    K::native_expert_set_mode(0,2);
    K::native_expert_grouped(layout,pointers.as<unsigned long long>(),boundaries.as<int32_t>(),ng.as<int32_t>(),dst.as<int32_t>(),
        tok.as<int32_t>(),CAP_GROUPS,entries,q.data(),scratch.data(),out.as<float>(),stream,grid_groups);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> result(size_t(entries)*N);CK(cudaMemcpy(result.data(),out.data(),out.bytes,cudaMemcpyDeviceToHost));
    for(int e=0;e<entries;++e)for(int r=0;r<N;++r) {
        const int group=e<1 ? 0:2;
        const auto* weights=reinterpret_cast<const Q5*>(host.data()+group*layout.bytes+layout.down_off+size_t(r)*layout.d_row);
        compare(result[size_t(destination[e])*N+r],reference(weights,intermediate.data()+size_t(e)*(FF/32),FF/32),"grouped",r,e);
    }
    REQUIRE(blobs.snapshot()==weights_before && x.snapshot()==input_before && q.snapshot()==q_before && scratch.snapshot()==scratch_before);
    REQUIRE(pointers.snapshot()==pointers_before && boundaries.snapshot()==boundaries_before && ng.snapshot()==ng_before);
    REQUIRE(dst.snapshot()==dst_before && tok.snapshot()==tok_before);
    for(auto* device:{&blobs,&scratch,&out,&x,&q,&pointers,&boundaries,&ng,&dst,&tok})device->guards();
    K::native_expert_set_mode(-1,0);K::native_grouped_set_v1(false);++cases;
}

int main() {
    cudaStream_t stream;CK(cudaStreamCreate(&stream));
    for(int pattern=0;pattern<5;++pattern)for(int input:{32,96,256,512})for(int columns:{1,3,8})matrix_case(input,columns,pattern,stream);
    for(int entries:{1,3,5,9})for(int pattern:{0,1,3}) {
        for(int groups:{1,5})grouped_case(entries,pattern,groups,false,stream);
        grouped_case(entries,pattern,5,true,stream);
    }
    grouped_case<2048,512>(3,0,1,false,stream); // managed Qwen/Clover/Tiel geometry
    REQUIRE(old_rejected>=20);CK(cudaStreamDestroy(stream));
    std::printf("PASS Q5_0 offset: %u matrix/grouped cases, %u old-formula rejections, max abs error %.9g; guards and inputs unchanged\n",
        cases,old_rejected,maximum_error);
}
#endif
