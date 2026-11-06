// Register-byte extraction and real IQ/K-quant MMVQ parity against the stored
// weights' dequantized values and a scalar double-precision dot product.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: packed-byte GPU test requires CUDA or HIP"); return 77; }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/packed_byte.hpp"
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

struct Device {
    void* data=nullptr;
    explicit Device(size_t bytes) { CK(cudaMalloc(&data,bytes)); }
    ~Device() { if(data)cudaFree(data); }
    template<class T>T* as()const { return static_cast<T*>(data); }
};
uint16_t half_bits(float f) { const half h=__float2half(f);uint16_t v;std::memcpy(&v,&h,2);return v; }
float half_value(uint16_t v) { half h;std::memcpy(&h,&v,2);return __half2float(h); }
uint32_t random_word(uint32_t& state) { state^=state<<13;state^=state>>17;state^=state<<5;return state; }
uint64_t result_hash=14695981039346656037ull;
unsigned matrix_cases=0;
double max_error=0;
void hash_floats(const std::vector<float>& values) {
    const auto* bytes=reinterpret_cast<const unsigned char*>(values.data());
    for(size_t i=0;i<values.size()*sizeof(float);++i)result_hash=(result_hash^bytes[i])*1099511628211ull;
}

static_assert(sizeof(uint2)==8 && sizeof(int2)==8,"packed vector layout");
__global__ void extract_bytes(const uint2* input,uint32_t* output,uint32_t* mapping,int count) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count)return;
    const uint2 word=input[i];
    const int2 signed_word=make_int2(int(word.x),int(word.y));
    mapping[size_t(i)*4+0]=blockIdx.x;
    mapping[size_t(i)*4+1]=threadIdx.x;
    mapping[size_t(i)*4+2]=blockDim.x;
    mapping[size_t(i)*4+3]=gridDim.x;
    uint32_t* out=output+size_t(i)*24;
    for(int j=0;j<4;++j)out[j]=K::packed_byte(word.x,(j+int(threadIdx.x))&3);
    for(int j=0;j<8;++j)out[4+j]=K::packed_byte(word,(j+int(threadIdx.x))&7);
    for(int j=0;j<8;++j)out[12+j]=K::packed_byte(signed_word,(j+int(threadIdx.x))&7);
    const uint16_t aux[2]={uint16_t(word.x),uint16_t(word.y)};
    const uint32_t scales=K::pack_scale_min(aux);
    for(int j=0;j<4;++j)out[20+j]=K::packed_byte(scales,j);
}
void check_bytes(cudaStream_t stream) {
    constexpr int COUNT=65536;
    std::vector<uint2> input(COUNT);
    uint32_t state=0x3875AD91;
    for(int i=0;i<COUNT;++i) {
        uint64_t bits;
        if(i<2048)bits=uint64_t(i&255)<<(8*(i/256));
        else { const uint32_t low=random_word(state),high=random_word(state);bits=uint64_t(low)|(uint64_t(high)<<32); }
        std::memcpy(&input[i],&bits,8);
    }
    Device source(input.size()*sizeof(uint2)),destination(size_t(COUNT)*24*sizeof(uint32_t)),mapping(size_t(COUNT)*4*sizeof(uint32_t));
    CK(cudaMemcpyAsync(source.data,input.data(),input.size()*sizeof(uint2),cudaMemcpyHostToDevice,stream));
    CK(cudaMemsetAsync(destination.data,0xff,size_t(COUNT)*24*sizeof(uint32_t),stream));
    CK(cudaMemsetAsync(mapping.data,0xff,size_t(COUNT)*4*sizeof(uint32_t),stream));
    extract_bytes<<<(COUNT+255)/256,256,0,stream>>>(source.as<uint2>(),destination.as<uint32_t>(),mapping.as<uint32_t>(),COUNT);
    CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));
    std::vector<uint32_t> got(size_t(COUNT)*24);
    std::vector<uint32_t> map(size_t(COUNT)*4);
    CK(cudaMemcpy(got.data(),destination.data,got.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(map.data(),mapping.data,map.size()*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<COUNT;++i) {
        if(map[size_t(i)*4]!=unsigned(i/256) || map[size_t(i)*4+1]!=unsigned(i%256) ||
           map[size_t(i)*4+2]!=256 || map[size_t(i)*4+3]!=unsigned((COUNT+255)/256)) {
            std::fprintf(stderr,"thread mapping i=%d block=%u thread=%u blockDim=%u gridDim=%u expected_block=%d expected_thread=%d\n",
                i,map[size_t(i)*4],map[size_t(i)*4+1],map[size_t(i)*4+2],map[size_t(i)*4+3],i/256,i%256);std::exit(1);
        }
        uint8_t bytes[8],scales[4];std::memcpy(bytes,&input[i],8);
        const uint16_t aux[2]={uint16_t(input[i].x),uint16_t(input[i].y)};
        std::memcpy(scales,aux,4);
        const auto check=[&](const char* path,int j,int slot,int index,uint32_t expected) {
            const uint32_t actual=got[size_t(i)*24+slot];
            if(actual!=expected) {
                std::fprintf(stderr,"byte mismatch path=%s i=%d j=%d index=%d words=%08x/%08x got=%08x expected=%08x block=%u thread=%u blockDim=%u gridDim=%u\n",
                    path,i,j,index,input[i].x,input[i].y,actual,expected,map[size_t(i)*4],map[size_t(i)*4+1],map[size_t(i)*4+2],map[size_t(i)*4+3]);std::exit(1);
            }
        };
        for(int j=0;j<4;++j)check("u32",j,j,(j+(i&255))&3,bytes[(j+(i&255))&3]);
        for(int j=0;j<8;++j) {
            check("uint2",j,4+j,(j+(i&255))&7,bytes[(j+(i&255))&7]);
            check("int2",j,12+j,(j+(i&255))&7,bytes[(j+(i&255))&7]);
        }
        for(int j=0;j<4;++j)check("scales",j,20+j,j,scales[j]);
    }
    std::puts("PASS thread_mapping blocks=256 block_size=256 words=65536");
}

#pragma pack(push,1)
struct Q8 { uint16_t d,sum;int8_t q[32]; };
#pragma pack(pop)
static_assert(sizeof(Q8)==36,"stored Q8_1 layout");
void matrix_case(int type,int width,int columns,cudaStream_t stream) {
    constexpr int ROWS=7;
    const size_t row_bytes=K::iq_row_bytes(type,width),block_bytes=K::iq_row_bytes(type,256);
    REQUIRE(row_bytes && width%256==0);
    std::vector<uint8_t> weights(ROWS*row_bytes);
    uint32_t seed=0xFA4C135u+unsigned(type*17+width+columns);
    for(auto& byte:weights)byte=uint8_t(random_word(seed));
    for(int row=0;row<ROWS;++row)for(int b=0;b<width/256;++b) {
        uint8_t* block=weights.data()+row*row_bytes+b*block_bytes;
        const uint16_t d=half_bits(std::ldexp(1.0f,-8+(row+b)%3));
        if(type==29) {
            // IQ1_M stores the block scale as four high nibbles among its
            // sub-block scales, rather than a separate half at offset zero.
            for(int p=0;p<4;++p) {
                uint16_t sc;std::memcpy(&sc,block+48+2*p,2);
                sc=uint16_t((sc&0xfff)|(((d>>(4*p))&15)<<12));
                std::memcpy(block+48+2*p,&sc,2);
            }
        }else {
            std::memcpy(block,&d,2);
            if(type==12 || type==13) { const uint16_t m=half_bits(1.0f/512);std::memcpy(block+2,&m,2); }
        }
    }
    std::vector<Q8> q(size_t(columns)*width/32);
    for(size_t b=0;b<q.size();++b) {
        q[b].d=half_bits(std::ldexp(1.0f,-3+int(b/(width/32))%3));
        int sum=0;
        for(int j=0;j<32;++j) { q[b].q[j]=int8_t(8*(int(random_word(seed)%31)-15));sum+=q[b].q[j]; }
        q[b].sum=half_bits(float(sum)*half_value(q[b].d));
    }
    Device dw(weights.size()),dq(q.size()*sizeof(Q8)),dequant(size_t(ROWS)*width*4),out(size_t(ROWS)*columns*4);
    CK(cudaMemcpyAsync(dw.data,weights.data(),weights.size(),cudaMemcpyHostToDevice,stream));
    CK(cudaMemcpyAsync(dq.data,q.data(),q.size()*sizeof(Q8),cudaMemcpyHostToDevice,stream));
    K::iq_dequant_f32(type,dw.data,size_t(ROWS)*width,dequant.as<float>(),stream);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> decoded(size_t(ROWS)*width);
    CK(cudaMemcpy(decoded.data(),dequant.data,decoded.size()*4,cudaMemcpyDeviceToHost));
    std::vector<float> previous;
    for(bool old:{true,false}) {
        K::iq_set_old_kernels(old);
        CK(cudaMemsetAsync(out.data,0xff,size_t(ROWS)*columns*4,stream));
        K::iq_mmvq(type,dw.data,dq.data,out.as<float>(),width,ROWS,columns,stream);
        CK(cudaStreamSynchronize(stream));
        std::vector<float> got(size_t(ROWS)*columns);
        CK(cudaMemcpy(got.data(),out.data,got.size()*4,cudaMemcpyDeviceToHost));
        for(int col=0;col<columns;++col)for(int row=0;row<ROWS;++row) {
            double expected=0,bound=0;
            for(int k=0;k<width;++k) {
                const auto& block=q[size_t(col)*(width/32)+k/32];
                const double product=double(decoded[size_t(row)*width+k])*half_value(block.d)*block.q[k&31];
                expected+=product;bound+=std::fabs(product);
            }
            const float actual=got[size_t(col)*ROWS+row];
            const double error=std::fabs(double(actual)-expected),limit=2e-6*std::max(1.0,bound);
            if(!std::isfinite(actual)||error>limit) {
                std::fprintf(stderr,"type=%d width=%d cols=%d old=%d row=%d col=%d actual=%.9g expected=%.12g error=%.9g limit=%.9g\n",
                    type,width,columns,int(old),row,col,actual,expected,error,limit);std::exit(1);
            }
            max_error=std::max(max_error,error);
        }
        if(old)previous=got;
        else REQUIRE(std::memcmp(previous.data(),got.data(),got.size()*sizeof(float))==0);
        hash_floats(got);++matrix_cases;
    }
}
int main() {
    int count=0;CK(cudaGetDeviceCount(&count));if(!count)return 77;
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check_bytes(stream);
    for(int type:{16,18,21,22,29,12,13})for(int width:{256,512})for(int columns:{1,2,4,8})matrix_case(type,width,columns,stream);
    K::iq_set_old_kernels(false);CK(cudaStreamDestroy(stream));
    std::printf("PASS byte_words=65536 matrix_cases=%u max_abs_error=%.9g output_hash=%016llx\n",matrix_cases,max_error,(unsigned long long)result_hash);
}
#endif
