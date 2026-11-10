// GPT-OSS activation expression from TinyStrata kernels.cu (MIT; Niko1221 and
// Strata contributors,2026). Native Q8_1 reduction from matvec.cu; MMQ D4 layout
// and float4 quantization from ggml-cuda/quantize.cu at llama.cpp revision
//3cf03257f219afbe7334045ff7c6a06ac68c627d (third_party/ggml/VERSION.txt).
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "strata/gemma4/activation_quant.hpp"
#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(__HIPCC__)
#include <cuda_runtime.h>
#include "strata/kernels/q8_1_finite.hpp"

namespace strata::gemma4::activation_quant {
namespace {
constexpr int WARP=32, THREADS=256, MMQ_THREADS=128;
struct Q81 { half2 ds; int8_t qs[32]; };
struct MmqD4 { float d[4]; int8_t qs[128]; };
static_assert(sizeof(Q81)==36 && offsetof(Q81,qs)==4);
static_assert(sizeof(MmqD4)==144 && offsetof(MmqD4,qs)==16);

struct IdBias {
    const int32_t* ids;
    const float* gate;
    const float* up;
    __device__ __forceinline__ float g(int r,int i,int n) const { return gate[size_t(ids[r])*n+i]; }
    __device__ __forceinline__ float u(int r,int i,int n) const { return up[size_t(ids[r])*n+i]; }
};
struct SlotBias {
    const float* gate[32];
    const float* up[32];
    __device__ __forceinline__ float g(int r,int i,int) const { return gate[r][i]; }
    __device__ __forceinline__ float u(int r,int i,int) const { return up[r][i]; }
};

template<class Bias>
__device__ __forceinline__ float activation(const float* input,int row,int i,int n,Bias bias,float alpha,float limit) {
#pragma clang fp contract(off)
    const float* p=input+size_t(row)*2*n;
    float g=p[i]+bias.g(row,i,n),u=p[n+i]+bias.u(row,i,n);
    // Keep kernels.cu's expression and operation order. In particular, the
    // unbounded negative gate, alpha and up+1 differ from ordinary SwiGLU.
    g=fminf(g,limit);
    u=fminf(fmaxf(u,-limit),limit);
    // The separate producer stores this product as FP32 before quantization.
    // HIP's __fmul_rn is itself a plain multiply in the current SDK, so use
    // contract(off), as iq_kernels.cu's fused SwiGLU/Q8 producer does. This
    // prevents its final product contracting into the first warp-sum add.
    return g/(1.0f+expf(-alpha*g))*(u+1.0f);
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for(int o=WARP/2;o>0;o>>=1) x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,o,WARP));
    return x;
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for(int o=WARP/2;o>0;o>>=1) x+=__shfl_xor_sync(0xffffffff,x,o,WARP);
    return x;
}

template<class Bias>
__global__ __launch_bounds__(THREADS,1)
void native_kernel(const float* input,Q81* out,int n,int total,Bias bias,float alpha,float limit) {
    const int i=int(blockIdx.x)*THREADS+int(threadIdx.x);
    if(i>=total) return; // only complete32-lane quantization groups return
    const float x=activation(input,i/n,i%n,n,bias,alpha,limit);
    const float amax=warp_max(fabsf(x)),sum=warp_sum(x);
    const float d=kernels::q8_1_finite(amax/127.0f);
    out[i/32].qs[i%32]=kernels::q8_1_quant(x,d,amax);
    if(i%32==0) out[i/32].ds=kernels::q8_1_ds(d,sum);
}

__global__ __launch_bounds__(MMQ_THREADS)
void mmq_kernel(const float* input,MmqD4* out,int n,int rows,int padded,IdBias bias,float alpha,float limit) {
    const int i=(int(blockIdx.y)*MMQ_THREADS+int(threadIdx.x))*4;
    if(i>=padded) return;
    const int row=int(blockIdx.x);
    // Preserve ggml's4-values/thread reduction, reciprocal association and
    // rounding, including the all-zero padded blocks. This is NOT native Q8_1.
    const float4 x=i<n ? make_float4(activation(input,row,i,n,bias,alpha,limit),
                                   activation(input,row,i+1,n,bias,alpha,limit),
                                   activation(input,row,i+2,n,bias,alpha,limit),
                                   activation(input,row,i+3,n,bias,alpha,limit)) : make_float4(0.f,0.f,0.f,0.f);
    float amax=fabsf(x.x);
    amax=fmaxf(amax,fabsf(x.y)); amax=fmaxf(amax,fabsf(x.z)); amax=fmaxf(amax,fabsf(x.w));
#pragma unroll
    for(int offset=4;offset>0;offset>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,offset,WARP));
    const float inv=127.0f/amax;
    char4 q;
    q.x=roundf(x.x*inv); q.y=roundf(x.y*inv); q.z=roundf(x.z*inv); q.w=roundf(x.w*inv);
    const float d=1.0f/inv;
    const size_t block=size_t(i/128)*rows+row;
    reinterpret_cast<char4*>(out[block].qs)[(i%128)/4]=q;
    if(i%32==0) out[block].d[(i%128)/32]=d;
}

bool geometry(const float* x,void* y,int n,int rows,float alpha,float limit) {
    return x && y && n>0 && rows>0 && alpha>0.f && limit>0.f && std::isfinite(alpha) && std::isfinite(limit);
}
void checked_launch() {
    if(const auto e=cudaGetLastError();e!=cudaSuccess)
        throw std::runtime_error(std::string("activation quant launch: ")+cudaGetErrorString(e));
}
} // namespace

bool available() { return true; }
bool q8_1_ids(const float* x,void* y,int n,int rows,const int32_t* ids,const float* bg,const float* bu,
              float alpha,float limit,void* stream) {
    if(!geometry(x,y,n,rows,alpha,limit) || !ids || !bg || !bu || n%32 || int64_t(n)*rows>INT_MAX) return false;
    const int total=n*rows;
    native_kernel<<<(total-1)/THREADS+1,THREADS,0,(cudaStream_t)stream>>>(x,(Q81*)y,n,total,IdBias{ids,bg,bu},alpha,limit);
    checked_launch();return true;
}
bool q8_1_slots(const float* x,void* y,int n,int rows,const ExpertBias& eb,float alpha,float limit,void* stream) {
    if(!geometry(x,y,n,rows,alpha,limit) || n%32 || rows>32 || int64_t(n)*rows>INT_MAX) return false;
    SlotBias bias{};
    for(int r=0;r<rows;++r) { if(!eb.g[r] || !eb.u[r]) return false; bias.gate[r]=eb.g[r];bias.up[r]=eb.u[r]; }
    const int total=n*rows;
    native_kernel<<<(total-1)/THREADS+1,THREADS,0,(cudaStream_t)stream>>>(x,(Q81*)y,n,total,bias,alpha,limit);
    checked_launch();return true;
}
bool mmq_mxfp4_ids(const float* x,void* y,int n,int rows,const int32_t* ids,const float* bg,const float* bu,
                   float alpha,float limit,void* stream) {
    if(!geometry(x,y,n,rows,alpha,limit) || !ids || !bg || !bu || n%4 || n>INT_MAX-511) return false;
    const int padded=(n+511)/512*512;
    if((padded-1)/(4*MMQ_THREADS)+1>65535) return false;
    mmq_kernel<<<dim3(rows,(padded-1)/(4*MMQ_THREADS)+1),MMQ_THREADS,0,(cudaStream_t)stream>>>(
        x,(MmqD4*)y,n,rows,padded,IdBias{ids,bg,bu},alpha,limit);
    checked_launch();return true;
}
} // namespace strata::gemma4::activation_quant
#else
namespace strata::gemma4::activation_quant {
bool available() { return false; }
bool q8_1_ids(const float*,void*,int,int,const int32_t*,const float*,const float*,float,float,void*) { return false; }
bool q8_1_slots(const float*,void*,int,int,const ExpertBias&,float,float,void*) { return false; }
bool mmq_mxfp4_ids(const float*,void*,int,int,const int32_t*,const float*,const float*,float,float,void*) { return false; }
} // namespace strata::gemma4::activation_quant
#endif
