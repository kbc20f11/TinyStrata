// Staged activation and four-rows-per-warp down layout adapted from Basalt
// src/models/deepseek41/dspark_kernels.cu, commit 30ad59a67a9447449f0c2afa5251e356fb07837b.
// Copyright (c) 2026 Jesús Gulfo.
// E2M1 lookup follows ggml (Copyright (c) 2023-2026 The ggml authors, MIT).
// TinyStrata uses absolute 17-byte block alignment, its original scale conversion
// including e255, and its original scale multiplication association. No VRAM repack.
//
// MIT License
//
// Copyright (c) 2026 Jesús Gulfo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/gemma4/mxfp4_down.hpp"
#include "slots.cuh"
#include "strata/kernels/dp4a.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace strata::kernels::mxfp4_down {
namespace {
constexpr int kWidth=2880,kBlocks=90,kWarps=8,kIters=3;
constexpr size_t kRowBytes=1530,kMatrixBytes=size_t(kWidth)*kRowBytes;
struct Q81 { __half2 ds; int8_t q[32]; };
struct XBlock { int4 lo,hi; float d,pad[3]; };
static_assert(sizeof(Q81)==36&&sizeof(XBlock)==48);

__device__ __forceinline__ float original_scale(uint32_t e) {
    return __uint_as_float(e==0?0x00400000u:e<<23)*0.5f;
}
__device__ __forceinline__ int e2m1x4(uint32_t q) {
    const uint32_t pos=__byte_perm(0x03020100u,0x0c080604u,q);
    const uint32_t neg=__byte_perm(0xfdfeff00u,0xf4f8fafcu,q);
    return (int)__byte_perm(pos,neg,0x3210u|((q&0x8888u)>>1));
}
__device__ __forceinline__ void stage(const Q81* x,XBlock* s) {
    for(int b=int(threadIdx.x);b<kBlocks;b+=int(blockDim.x)) {
        const int* q=reinterpret_cast<const int*>(x[b].q);
        int w[8];
#pragma unroll
        for(int i=0;i<4;++i) {
            w[2*i]=__byte_perm(q[i],q[i+4],0x5140);
            w[2*i+1]=__byte_perm(q[i],q[i+4],0x7362);
        }
        s[b].lo=make_int4(w[0],w[1],w[2],w[3]);
        s[b].hi=make_int4(w[4],w[5],w[6],w[7]);
        s[b].d=__low2float(x[b].ds);
    }
}
struct Loaded { uint32_t w[kIters][5]; unsigned sh[kIters]; };
__device__ __forceinline__ void row_load(const uint8_t* row,int lane,Loaded& r) {
#pragma unroll
    for(int it=0;it<kIters;++it) {
        const int b=lane+32*it;
        if(b<kBlocks) {
            // A 1530-byte row alternates its starting alignment. Align the absolute block address,
            // never its offset within a row. In the final matrix block sh=24 and p+20==matrix_end.
            const uintptr_t address=reinterpret_cast<uintptr_t>(row)+size_t(b)*17;
            const uint32_t* p=reinterpret_cast<const uint32_t*>(address&~uintptr_t(3));
            r.sh[it]=unsigned(address&3)*8;
#pragma unroll
            for(int k=0;k<5;++k)r.w[it][k]=__ldg(p+k);
        }
    }
}
__device__ __forceinline__ float row_dot(const Loaded& r,const XBlock* s,int lane) {
    float acc=0;
#pragma unroll
    for(int it=0;it<kIters;++it) {
        const int b=lane+32*it;
        if(b>=kBlocks)continue;
        const unsigned sh=r.sh[it];
        const float scale=original_scale((r.w[it][0]>>sh)&255);
        const int4 lo=s[b].lo,hi=s[b].hi;
        const int q[8]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        int dot=0;
#pragma unroll
        for(int k=0;k<4;++k) {
            const unsigned shift=sh+8;
            const uint32_t bits=shift==32?r.w[it][k+1]:
                (r.w[it][k]>>shift)|(r.w[it][k+1]<<(32-shift));
            dot=STRATA_DP4A(e2m1x4(bits),q[2*k],dot);
            dot=STRATA_DP4A(e2m1x4(bits>>16),q[2*k+1],dot);
        }
        const float d=scale*s[b].d; // same association as current SmallTraits<MXFP4Block,4>
        acc+=d*float(dot);
    }
#pragma unroll
    for(int mask=16;mask;mask>>=1)acc+=__shfl_xor_sync(0xffffffffu,acc,mask);
    return acc;
}
template<class S,int Groups>
__global__ __launch_bounds__(256) void down_kernel(S bundle) {
    extern __shared__ __align__(16) uint8_t shared[];
    XBlock* xs=reinterpret_cast<XBlock*>(shared);
    const auto p=strata::kernels::slot_ptrs(bundle,int(blockIdx.y));
    const int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    const int row0=int(blockIdx.x)*(kWarps*Groups)+warp;
    const uint8_t* row=static_cast<const uint8_t*>(p.w)+size_t(row0)*kRowBytes;
    Loaded w[2];
    row_load(row,lane,w[0]);
    stage(static_cast<const Q81*>(p.x),xs);
    __syncthreads();
#pragma unroll
    for(int group=0;group<Groups;++group) {
        if(group+1<Groups)row_load(row+size_t(group+1)*kWarps*kRowBytes,lane,w[(group+1)&1]);
        const float value=row_dot(w[group&1],xs,lane);
        if(lane==0)p.y[row0+group*kWarps]=value;
    }
}
__global__ void probe_kernel(uint32_t* scales,uint32_t* codes) {
    const unsigned i=threadIdx.x;
    if(i<256)scales[i]=__float_as_uint(original_scale(i));
    if(i<16)codes[i]=uint32_t(e2m1x4(i*0x1111u));
}
bool shape(int type,int count,int n_in,int n_out) {
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
    return type==39&&count>=1&&count<=kMmvqSlotsMax&&n_in==kWidth&&n_out==kWidth;
#else
    return false;
#endif
}
bool aligned(const void* p){return p&&(reinterpret_cast<uintptr_t>(p)&3)==0;}
void checked(){if(auto e=cudaGetLastError();e!=cudaSuccess)throw std::runtime_error(std::string("mxfp4 prototype: ")+cudaGetErrorString(e));}
template<class S>void launch(const S& s,int count,void* stream) {
    // More CTAs hide memory latency for the few cached experts of a hybrid token.
    // Per-row load/dot/reduction order stays identical across both groupings.
    if(count<=3)
        down_kernel<S,2><<<dim3(kWidth/(kWarps*2),count),256,kBlocks*sizeof(XBlock),(cudaStream_t)stream>>>(s);
    else
        down_kernel<S,4><<<dim3(kWidth/(kWarps*4),count),256,kBlocks*sizeof(XBlock),(cudaStream_t)stream>>>(s);
    checked();
}
}
bool slots(int type,const strata::kernels::MmvqSlots& s,int count,int ni,int no,void* stream) {
    if(!shape(type,count,ni,no))return false;
    for(int i=0;i<count;++i)if(!aligned(s.w[i])||!aligned(s.x[i])||!aligned(s.y[i]))return false;
    launch(s,count,stream);return true;
}
bool direct(int type,const void* w,const void* x,float* y,int ni,int no,int nc,void* stream) {
    if(nc!=1)return false;
    strata::kernels::MmvqSlots s{};s.w[0]=w;s.x[0]=x;s.y[0]=y;
    return slots(type,s,1,ni,no,stream);
}
bool ids(int type,const strata::kernels::MmvqIds& s,int count,int ni,int no,void* stream) {
    if(!shape(type,count,ni,no)||!s.table||!s.ids||!aligned(s.x)||!aligned(s.y)||s.x_every<1||
       (s.offset&3)||(s.x_stride&3)||s.x_stride<size_t(kBlocks)*sizeof(Q81)||s.y_stride<size_t(kWidth))return false;
    launch(s,count,stream);return true;
}
void probe(uint32_t* a,uint32_t* b,void* stream) {
    probe_kernel<<<1,256,0,(cudaStream_t)stream>>>(a,b);checked();
}
}
