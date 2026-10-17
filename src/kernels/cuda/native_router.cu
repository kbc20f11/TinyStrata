// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; Qwen3.6 finite F32 top-k routing.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
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
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
template <int NE, int K>
__launch_bounds__(256, 1)
__global__ void route_topk(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                           int n_tok) {
    static_assert(NE % 32 == 0 && K <= 32, "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE; ids += (size_t) tk * K; weights += (size_t) tk * K;
    const int lane = threadIdx.x;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
}
bool native_router_topk_supported(int n_expert, int k) { return n_expert == 256 && k == 8; }
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !native_router_topk_supported(n_expert, k) ||
        !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("native_router_topk: a stream, a supported (experts, k) and aligned buffers");
    route_topk<256, 8><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}
