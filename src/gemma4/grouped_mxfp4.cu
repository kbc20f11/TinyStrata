// The native MXFP4 dot/reduction is adapted from TinyStrata matvec.cu (MIT,
// Niko1221 and Strata contributors; ggml authors 2023-2026). The optimized
// down row_load/row_dot follows TinyStrata mxfp4_down.cu, adapted from Basalt
// src/models/deepseek41/dspark_kernels.cu at 30ad59a67a9447449f0c2afa5251e356fb07837b.
// Copyright (c) 2026 Jesus Gulfo. Grouping changes neither dot expression.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Jesus Gulfo
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/gemma4/grouped_mxfp4.hpp"
#include "strata/kernels/dp4a.hpp"
#include "slots.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels::grouped_mxfp4 {
namespace {
constexpr int kWidth = 2880, kBlocks = 90, kExperts = 32, kRows = 32;
constexpr size_t kRowBytes = 1530;
struct Groups { int counts[kExperts], rows[kExperts][kRows]; uint32_t invalid; };
struct Q81 { __half2 ds; int8_t q[32]; };
struct Mxfp4 { uint8_t e, qs[16]; };
struct XBlock { int4 lo, hi; float d, pad[3]; };
static_assert(sizeof(Q81) == 36 && sizeof(Mxfp4) == 17 && sizeof(XBlock) == 48);

__global__ void prepare_ids_kernel(const int32_t* ids, int count, Groups* groups) {
    const int leader = int(threadIdx.x);
    if (leader == 0) {
        uint32_t invalid = 0;
        for (int row = 0; row < count; ++row)
            if (ids[row] < 0 || ids[row] >= kExperts) invalid = 1;
        groups->invalid = invalid;
    }
    int n = 0;
    if (leader < count) {
        bool first = true;
        for (int row = 0; row < leader; ++row) if (ids[row] == ids[leader]) first = false;
        if (first)
            for (int row = leader; row < count; ++row)
                if (ids[row] == ids[leader]) groups->rows[leader][n++] = row;
    }
    groups->counts[leader] = n;
}

__global__ void prepare_slots_kernel(MmvqSlots slots, int count, Groups* groups) {
    const int leader = int(threadIdx.x);
    if (leader == 0) groups->invalid = 0;
    int n = 0;
    if (leader < count) {
        bool first = true;
        for (int row = 0; row < leader; ++row) if (slots.w[row] == slots.w[leader]) first = false;
        if (first)
            for (int row = leader; row < count; ++row)
                if (slots.w[row] == slots.w[leader]) groups->rows[leader][n++] = row;
    }
    groups->counts[leader] = n;
}

__device__ __forceinline__ const uint8_t* weight(const MmvqIds& b, int leader) {
    return static_cast<const uint8_t*>(b.table[b.ids[leader]]) + b.offset;
}
__device__ __forceinline__ const uint8_t* weight(const MmvqSlots& b, int leader) {
    return static_cast<const uint8_t*>(b.w[leader]);
}
__device__ __forceinline__ const Q81* input(const MmvqIds& b, int row) {
    return reinterpret_cast<const Q81*>(static_cast<const uint8_t*>(b.x) + size_t(row / b.x_every) * b.x_stride);
}
__device__ __forceinline__ const Q81* input(const MmvqSlots& b, int row) { return static_cast<const Q81*>(b.x[row]); }
__device__ __forceinline__ float* output(const MmvqIds& b, int row) { return b.y + size_t(row) * b.y_stride; }
__device__ __forceinline__ float* output(const MmvqSlots& b, int row) { return b.y[row]; }

__device__ __forceinline__ float scale(uint32_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : e << 23) * 0.5f;
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask);
    return value;
}
__device__ __forceinline__ int byte_word(const void* p, int i) {
    const auto* x = static_cast<const uint8_t*>(p) + 4 * i;
    return x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24;
}
__device__ __forceinline__ int2 lookup(int q4) {
    // Exact table16_lookup(mxfp4_values) from the existing native dot.
    constexpr uint32_t table[4] = {0x03020100u, 0x0c080604u, 0xfdfeff00u, 0xf4f8fafcu};
    uint32_t tmp[2];
    const uint32_t selection = 0x32103210u | ((uint32_t(q4) & 0x88888888u) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table[0], table[1], uint32_t(q4) >> shift);
        const uint32_t high = __byte_perm(table[2], table[3], uint32_t(q4) >> shift);
        tmp[i] = __byte_perm(low, high, selection >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
struct NativeWeight { int2 v[2]; float d; };
__device__ __forceinline__ NativeWeight native_load(const Mxfp4* w, int iqs) {
    NativeWeight r;
#pragma unroll
    for (int i = 0; i < 2; ++i) r.v[i] = lookup(byte_word(w->qs, iqs + i));
    r.d = scale(w->e);
    return r;
}
__device__ __forceinline__ float native_dot(const NativeWeight& w, const Q81* x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->q) + iqs;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        sum = STRATA_DP4A(w.v[i].x, q8[i], sum);
        sum = STRATA_DP4A(w.v[i].y, q8[i + 4], sum);
    }
    const float d = w.d * __low2float(x->ds);
    return d * sum;
}

template<class Bundle, int Columns>
__global__ __launch_bounds__(128, 4) void native_kernel(Bundle bundle, const Groups* groups, int count, int n_out) {
    const int leader = int(blockIdx.y), row = int(blockIdx.x);
    if (groups->invalid) {
        if (leader < count && threadIdx.x == 0 && threadIdx.y == 0) output(bundle, leader)[row] = NAN;
        return;
    }
    const int columns = groups->counts[leader];
    if (!columns) return;
    const auto* w = reinterpret_cast<const Mxfp4*>(weight(bundle, leader));
    const int tid = 32 * int(threadIdx.y) + int(threadIdx.x), iqs = 2 * (tid % 2);
    __shared__ float partial[3][Columns][32];
    for (int first = 0; first < columns; first += Columns) {
        const int n = min(Columns, columns - first);
        float tmp[Columns] = {};
        for (int block = tid / 2; block < kBlocks; block += 64) {
            const NativeWeight loaded = native_load(w + size_t(row) * kBlocks + block, iqs);
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                if (col < n) tmp[col] += native_dot(loaded, input(bundle, groups->rows[leader][first + col]) + block, iqs);
        }
        if (threadIdx.y > 0)
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                if (col < n) partial[threadIdx.y - 1][col][threadIdx.x] = tmp[col];
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
            for (int col = 0; col < Columns; ++col) if (col < n) {
#pragma unroll
                for (int warp = 0; warp < 3; ++warp) tmp[col] += partial[warp][col][threadIdx.x];
                tmp[col] = warp_sum(tmp[col]);
                if (threadIdx.x == 0) output(bundle, groups->rows[leader][first + col])[row] = tmp[col];
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int e2m1x4(uint32_t q) {
    const uint32_t positive = __byte_perm(0x03020100u, 0x0c080604u, q);
    const uint32_t negative = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, q);
    return int(__byte_perm(positive, negative, 0x3210u | ((q & 0x8888u) >> 1)));
}
struct DownWeight { uint32_t w[3][5]; unsigned shift[3]; };
__device__ __forceinline__ DownWeight down_load(const uint8_t* row, int lane) {
    DownWeight r;
#pragma unroll
    for (int it = 0; it < 3; ++it) {
        const int b = lane + 32 * it;
        if (b < kBlocks) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(row) + size_t(b) * 17;
            const auto* p = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
            r.shift[it] = unsigned(address & 3) * 8;
#pragma unroll
            for (int k = 0; k < 5; ++k) r.w[it][k] = __ldg(p + k);
        }
    }
    return r;
}
__device__ __forceinline__ float down_dot(const DownWeight& w, const XBlock* x, int lane) {
    float acc = 0;
#pragma unroll
    for (int it = 0; it < 3; ++it) {
        const int b = lane + 32 * it;
        if (b >= kBlocks) continue;
        const unsigned shift = w.shift[it];
        const float d = scale((w.w[it][0] >> shift) & 255);
        const int4 lo = x[b].lo, hi = x[b].hi;
        const int q[8] = {lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        int dot = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned s = shift + 8;
            const uint32_t bits = s == 32 ? w.w[it][k + 1] :
                (w.w[it][k] >> s) | (w.w[it][k + 1] << (32 - s));
            dot = STRATA_DP4A(e2m1x4(bits), q[2*k], dot);
            dot = STRATA_DP4A(e2m1x4(bits >> 16), q[2*k+1], dot);
        }
        const float s = d * x[b].d;
        acc += s * float(dot);
    }
    return warp_sum(acc);
}
template<class Bundle>
__global__ __launch_bounds__(256) void down_kernel(Bundle bundle, const Groups* groups, int count, int tile) {
    const int leader = int(blockIdx.y), warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    if (groups->invalid) {
        if (leader < count)
            for (int row = int(blockIdx.x) * 32 + int(threadIdx.x); row < min(kWidth,int(blockIdx.x)*32+32); row += 256)
                output(bundle, leader)[row] = NAN;
        return;
    }
    const int columns = groups->counts[leader];
    if (!columns) return;
    const uint8_t* matrix = weight(bundle, leader);
    extern __shared__ __align__(16) uint8_t shared[];
    XBlock* xs = reinterpret_cast<XBlock*>(shared);
    for (int first = 0; first < columns; first += tile) {
        const int n = min(tile, columns - first);
        for (int at = int(threadIdx.x); at < n * kBlocks; at += 256) {
            const int col = at / kBlocks, b = at % kBlocks;
            const Q81* x = input(bundle, groups->rows[leader][first + col]) + b;
            const int* q = reinterpret_cast<const int*>(x->q);
            int words[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                words[2*i] = __byte_perm(q[i], q[i+4], 0x5140);
                words[2*i+1] = __byte_perm(q[i], q[i+4], 0x7362);
            }
            xs[col*kBlocks+b].lo = make_int4(words[0],words[1],words[2],words[3]);
            xs[col*kBlocks+b].hi = make_int4(words[4],words[5],words[6],words[7]);
            xs[col*kBlocks+b].d = __low2float(x->ds);
        }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int row = int(blockIdx.x) * 32 + warp + 8 * group;
            const DownWeight loaded = down_load(matrix + size_t(row) * kRowBytes, lane);
            for (int col = 0; col < n; ++col) {
                const float value = down_dot(loaded, xs + col*kBlocks, lane);
                if (lane == 0) output(bundle, groups->rows[leader][first + col])[row] = value;
            }
        }
        __syncthreads();
    }
}

bool aligned(const void* pointer, size_t alignment) { return pointer && uintptr_t(pointer) % alignment == 0; }
bool supported(int type, int count, int n_in, int n_out, bool down) {
#if defined(STRATA_HIP_GFX906)
    return false;
#else
    return type == 39 && count >= 2 && count <= kRows && n_in == kWidth &&
        (n_out == kWidth || n_out == 2*kWidth) && (!down || n_out == kWidth);
#endif
}
void checked() {
    if (const auto e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("grouped MXFP4: ") + cudaGetErrorString(e));
}
int down_columns() {
    static const int columns = [] {
        const char* value = std::getenv("TINYSTRATA_GPTOSS_GROUPED_COLUMNS");
        if (!value || !std::strcmp(value,"4")) return 4;
        if (!std::strcmp(value,"2")) return 2;
        if (!std::strcmp(value,"8")) return 8;
        throw std::invalid_argument("TINYSTRATA_GPTOSS_GROUPED_COLUMNS must be 2, 4 or 8");
    }();
    return columns;
}
template<class Bundle>
void launch(const Bundle& bundle, int count, int n_out, const void* workspace, bool down, void* stream) {
    const auto* groups = static_cast<const Groups*>(workspace);
    const auto s = static_cast<cudaStream_t>(stream);
    const int tile = down_columns();
    if (down) {
        down_kernel<<<dim3(kWidth/32,count),256,size_t(tile)*kBlocks*sizeof(XBlock),s>>>(bundle,groups,count,tile);
    }
    else if (tile == 2)
        native_kernel<Bundle,2><<<dim3(n_out,count),dim3(32,4),0,s>>>(bundle,groups,count,n_out);
    else if (tile == 4)
        native_kernel<Bundle,4><<<dim3(n_out,count),dim3(32,4),0,s>>>(bundle,groups,count,n_out);
    else
        native_kernel<Bundle,8><<<dim3(n_out,count),dim3(32,4),0,s>>>(bundle,groups,count,n_out);
    checked();
}
} // namespace

size_t workspace_bytes() { return (sizeof(Groups) + 255) & ~size_t(255); }
bool prepare_ids(const int32_t* ids, int count, int experts, void* workspace, size_t bytes, void* stream) {
    if (count < 2 || count > kRows || experts != kExperts) return false;
    if (!aligned(ids,alignof(int32_t)) || !aligned(workspace,16) || bytes < workspace_bytes())
        throw std::invalid_argument("invalid grouped MXFP4 IDs/workspace");
    prepare_ids_kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(ids,count,static_cast<Groups*>(workspace));
    checked(); return true;
}
bool prepare_slots(const MmvqSlots& bundle, int count, void* workspace, size_t bytes, void* stream) {
    if (count < 2 || count > kMmvqSlotsMax) return false;
    if (!aligned(workspace,16) || bytes < workspace_bytes()) throw std::invalid_argument("invalid grouped MXFP4 slots workspace");
    for (int i=0;i<count;++i)
        if (!aligned(bundle.w[i],4) || !aligned(bundle.x[i],4) || !aligned(bundle.y[i],4))
            throw std::invalid_argument("invalid grouped MXFP4 slot pointers");
    prepare_slots_kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(bundle,count,static_cast<Groups*>(workspace));
    checked(); return true;
}
bool ids(int type,const MmvqIds& bundle,int count,int n_in,int n_out,const void* workspace,bool down,void* stream) {
    if (!supported(type,count,n_in,n_out,down)) return false;
    if (!aligned(workspace,16) || !aligned(bundle.table,alignof(void*)) || !aligned(bundle.ids,4) ||
        !aligned(bundle.x,4) || !aligned(bundle.y,4) || bundle.x_every<1 || bundle.offset%4 ||
        bundle.x_stride<size_t(kBlocks)*sizeof(Q81) || bundle.x_stride%4 || bundle.y_stride<size_t(n_out))
        throw std::invalid_argument("invalid grouped MXFP4 IDs bundle");
    launch(bundle,count,n_out,workspace,down,stream); return true;
}
bool slots(int type,const MmvqSlots& bundle,int count,int n_in,int n_out,const void* workspace,bool down,void* stream) {
    if (!supported(type,count,n_in,n_out,down) || count>kMmvqSlotsMax) return false;
    if (!aligned(workspace,16)) throw std::invalid_argument("invalid grouped MXFP4 slots workspace");
    for(int i=0;i<count;++i)
        if(!aligned(bundle.w[i],4)||!aligned(bundle.x[i],4)||!aligned(bundle.y[i],4))
            throw std::invalid_argument("invalid grouped MXFP4 slot pointers");
    launch(bundle,count,n_out,workspace,down,stream);return true;
}
} // namespace strata::kernels::grouped_mxfp4
