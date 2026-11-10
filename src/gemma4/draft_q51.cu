// Q5_1 dot expression adapted from TinyStrata iq_kernels.cu and llama.cpp
// ggml-cuda/vecdotq.cuh at revision 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Niko1221 and the Strata contributors
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
#include "strata/gemma4/draft_q51.hpp"
#include "strata/gemma4/matvec.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::gemma4 {
namespace {

struct Q51 { half2 dm; uint8_t qh[4], qs[16]; };
struct Q81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q51) == 24 && sizeof(Q81) == 36);

__device__ __forceinline__ int word(const void* src, int index) {
    return static_cast<const int*>(src)[index];
}

struct Weight {
    int lo[2], hi[2];
    float2 dm;
};

__device__ __forceinline__ Weight load(const Q51* block, int iqs) {
    Weight w;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = word(block->qs, iqs + i);
        const int vh = word(block->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        w.lo[i] = vi0; w.hi[i] = vi1;
    }
    w.dm = __half22float2(block->dm);
    return w;
}

__device__ __forceinline__ float dot(const Weight& w, const Q81* x, int iqs) {
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int u0 = word(x->qs, iqs + i), u1 = word(x->qs, iqs + i + 4);
        sumi = STRATA_DP4A(w.lo[i], u0, sumi);
        sumi = STRATA_DP4A(w.hi[i], u1, sumi);
        sumu = STRATA_DP4A(0x01010101, u1, STRATA_DP4A(0x01010101, u0, sumu));
    }
    const float d8 = __low2float(x->ds);
    // The reference computes its offset from the integer codes, not Q8_1's
    // stored original-float sum. Keep both float products and the add in order.
    return sumi * (w.dm.x * d8) + sumu * (w.dm.y * d8);
}

template<int NC>
__global__ __launch_bounds__(128) void read_once(const Q51* __restrict__ weights,
                                                const Q81* __restrict__ x, float* __restrict__ y,
                                                int n_in, int n_out) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    if (row >= n_out) return;
    const int lane = int(threadIdx.x), nb = n_in / 32;
    const Q51* wr = weights + size_t(row) * nb;
    float sum[NC] = {};
    // Two reference dots cover one32-value Q5_1 block. Each output retains
    // lane-strided calls k=lane,lane+32,... and the same xor reduction tree.
    for (int step = lane; step < nb * 2; step += 32) {
        const int block = step / 2, iqs = 2 * (step % 2);
        const Weight w = load(wr + block, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) sum[c] += dot(w, x + size_t(c) * nb + block, iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum[c] += __shfl_xor_sync(0xffffffffu, sum[c], offset);
        if (lane == 0) y[size_t(c) * n_out + row] = sum[c];
    }
}

}  // namespace

void draft_q51_mmvq(const void* weights, const void* q8_1, float* output,
                     int n_in, int n_out, int columns, void* stream) {
    if (n_in <= 0 || n_in % 32 || n_out <= 0 || columns < 1 || columns > 8 ||
        !weights || !q8_1 || !output || !stream ||
        uintptr_t(weights) % 4 || uintptr_t(q8_1) % 4 || uintptr_t(output) % 4)
        throw std::invalid_argument("Draft Q5_1: invalid shape, alignment or stream");
    static const bool read_once_enabled = [] {
        const char* value = std::getenv("TINYSTRATA_DRAFT_Q51_READONCE");
        if (!value || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        throw std::invalid_argument("TINYSTRATA_DRAFT_Q51_READONCE must be 0 or 1");
    }();
    if (!read_once_enabled) {
        kernels::gemma_native_mmvq(7, weights, q8_1, output, n_in, n_out, columns, stream);
        return;
    }
    const dim3 grid(unsigned((n_out + 3) / 4)), threads(32, 4);
    const auto* w = static_cast<const Q51*>(weights);
    const auto* x = static_cast<const Q81*>(q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
#define LAUNCH(N) case N: read_once<N><<<grid, threads, 0, s>>>(w, x, output, n_in, n_out); break
    switch (columns) { LAUNCH(1); LAUNCH(2); LAUNCH(3); LAUNCH(4); LAUNCH(5); LAUNCH(6); LAUNCH(7); LAUNCH(8); }
#undef LAUNCH
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Draft Q5_1: ") + cudaGetErrorString(error));
}

}  // namespace strata::gemma4
