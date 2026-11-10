// Flash-decoding strategy and grouped value reuse adapted from PolyStrata (MIT):
// src/poly/hip_split_attend.cuh, 221be3e61a87d20e6c70c86907e0504ec429134b,
// value mapping b7e169988f13902b74acba2a6bf732315ac29d5d.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// Copyright (c) 2026 VecSzn and the PolyStrata contributors (MIT).
// This D64/GQA8 specialization keeps FP32 queries and adds the learned sink exactly once in the join.
#include "strata/gemma4/sink_attention.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>

namespace strata::gemma4 {
namespace {
constexpr int kDim = 64, kGroup = 8, kKeys = 512, kThreads = 256, kTile = 64, kStride = kDim + 2;

__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d, 32);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d, 32));
    return x;
}
__device__ __forceinline__ int actual_keys(long long pos, const long long* at, int capacity, int row) {
    const long long p = at ? at[2 * row] : pos + row;
    return p >= 0 && p < capacity ? int(p) + 1 : 0;
}

union TileStorage {
    struct { float scores[kGroup][kTile], weights[kTile][kGroup]; } tile;
    float reduction[3][kGroup][kDim];
};

__global__ __launch_bounds__(kThreads)
void split_kernel(const float* __restrict__ q, const int8_t* __restrict__ kc, const uint16_t* __restrict__ ks,
                  const int8_t* __restrict__ vc, const uint16_t* __restrict__ vs, float* __restrict__ part,
                  int kv_heads, long long pos, const long long* __restrict__ at, int capacity,
                  float scale, int blocks) {
    const int query_row = int(blockIdx.z);
    const int keys = actual_keys(pos, at, capacity, query_row);
    const int k0 = int(blockIdx.x) * kKeys;
    if (k0 >= keys) return; // uniform CTA exit; the join never reads these untouched partials
    const int count = min(kKeys, keys - k0), group = int(blockIdx.y), tid = int(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    __shared__ float query[kGroup][kDim], alpha[kGroup];
    __shared__ TileStorage shared;
    for (int i = tid; i < kGroup * kDim; i += kThreads)
        query[i / kDim][i % kDim] = q[((size_t(query_row) * kv_heads + group) * kGroup + i / kDim) * kDim + i % kDim];
    float output[kGroup] = {};
    float maximum = -INFINITY, denominator = 0.f; // warp w owns head w's online softmax
    __syncthreads();

    const int dimension = tid & 63, key_subset = tid >> 6; // 64 coalesced dimensions, four key subsets
    for (int t0 = 0; t0 < count; t0 += kTile) {
        const int tile = min(kTile, count - t0);
        // A half warp owns one key, four dimensions per lane. Reuse its dequantized key for all eight queries.
        for (int first = tid / 16; first < kTile; first += kThreads / 16) {
            const int d0 = (tid & 15) * 4, key = k0 + t0 + first;
            float scores[kGroup] = {};
            if (first < tile) {
                const size_t row = size_t(key) * kv_heads + group;
                const float sc = __half2float(__ushort_as_half(ks[row * 2 + d0 / 32]));
                for (int d = 0; d < 4; ++d) {
                    const float value = float(kc[row * kDim + d0 + d]) * sc;
                    for (int h = 0; h < kGroup; ++h) scores[h] = fmaf(query[h][d0 + d], value, scores[h]);
                }
            }
            for (int h = 0; h < kGroup; ++h) {
                for (int d = 8; d; d >>= 1) scores[h] += __shfl_xor_sync(0xffffffff, scores[h], d, 32);
                if ((tid & 15) == 0) shared.tile.scores[h][first] = first < tile ? scores[h] * scale : -INFINITY;
            }
        }
        __syncthreads();
        // Each warp handles one head. Padded keys are -inf and receive zero weight.
        const float s0 = shared.tile.scores[warp][lane], s1 = shared.tile.scores[warp][lane + 32];
        const float next_maximum = fmaxf(maximum, warp_max(fmaxf(s0, s1)));
        const float shrink = next_maximum == -INFINITY ? 1.f : expf(maximum - next_maximum);
        const float p0 = next_maximum == -INFINITY ? 0.f : expf(s0 - next_maximum);
        const float p1 = next_maximum == -INFINITY ? 0.f : expf(s1 - next_maximum);
        shared.tile.weights[lane][warp] = p0;
        shared.tile.weights[lane + 32][warp] = p1;
        denominator = denominator * shrink + warp_sum(p0 + p1);
        maximum = next_maximum;
        if (lane == 0) alpha[warp] = shrink;
        __syncthreads();

        for (int h = 0; h < kGroup; ++h) output[h] *= alpha[h];
        for (int j = key_subset; j < tile; j += 4) {
            const size_t row = size_t(k0 + t0 + j) * kv_heads + group;
            const float sc = __half2float(__ushort_as_half(vs[row * 2 + dimension / 32]));
            const float value = float(vc[row * kDim + dimension]) * sc;
            for (int h = 0; h < kGroup; ++h) output[h] = fmaf(shared.tile.weights[j][h], value, output[h]);
        }
        __syncthreads();
    }
    // Reuse the tile's LDS to combine the four key subsets, with the same max scale for each head.
    if (key_subset > 0)
        for (int h = 0; h < kGroup; ++h) shared.reduction[key_subset - 1][h][dimension] = output[h];
    __syncthreads();
    if (key_subset == 0) {
        for (int h = 0; h < kGroup; ++h) {
            float sum = output[h];
            for (int k = 0; k < 3; ++k) sum += shared.reduction[k][h][dimension];
            part[(((size_t(query_row) * kv_heads + group) * kGroup + h) * blocks + blockIdx.x) * kStride + dimension] = sum;
        }
    }
    if (lane == 0) {
        float* p = part + (((size_t(query_row) * kv_heads + group) * kGroup + warp) * blocks + blockIdx.x) * kStride;
        p[kDim] = maximum;
        p[kDim + 1] = denominator;
    }
}

__global__ void join_kernel(const float* __restrict__ part, const float* __restrict__ sinks,
                             float* __restrict__ out, long long pos, const long long* __restrict__ at,
                             int capacity, int blocks) {
    const int h = int(blockIdx.x), d = int(threadIdx.x), query_row = int(blockIdx.y);
    const int used = (actual_keys(pos, at, capacity, query_row) + kKeys - 1) / kKeys;
    const float* base = part + (size_t(query_row) * gridDim.x + h) * blocks * kStride;
    // The sink is one extra softmax logit with value zero, once per query head (not once per split).
    float maximum = sinks[h];
    for (int b = 0; b < used; ++b) maximum = fmaxf(maximum, base[size_t(b) * kStride + kDim]);
    float numerator = 0.f, denominator = expf(sinks[h] - maximum);
    for (int b = 0; b < used; ++b) {
        const float m = base[size_t(b) * kStride + kDim];
        if (m == -INFINITY) continue;
        const float weight = expf(m - maximum);
        numerator = fmaf(weight, base[size_t(b) * kStride + d], numerator);
        denominator += weight * base[size_t(b) * kStride + kDim + 1];
    }
    out[(size_t(query_row) * gridDim.x + h) * kDim + d] = numerator / denominator;
}
} // namespace

size_t sink_attention_workspace_floats(int heads, int max_keys, int rows) {
    if (heads <= 0 || max_keys <= 0 || rows < 1 || rows > 8) return 0;
    return size_t(rows) * heads * ((size_t(max_keys) + kKeys - 1) / kKeys) * kStride;
}

bool sink_attention_q8(const float* q, const int8_t* kcodes, const uint16_t* kscales,
                        const int8_t* vcodes, const uint16_t* vscales, const float* sinks,
                        float* out, float* part, size_t part_floats, int heads, int kv_heads,
                        int64_t pos, const long long* at, int room, float scale, void* stream, int rows) {
    if (rows < 1 || rows > 8 || kv_heads <= 0 || kv_heads > INT_MAX / kGroup || heads != kv_heads * kGroup ||
        !q || !kcodes || !kscales || !vcodes || !vscales || !sinks || !out || !part ||
        !(scale > 0.f) || !std::isfinite(scale) || (!at && (pos < 0 || pos > INT_MAX - rows))) return false;
    const int capacity = at ? room : int(pos) + rows;
    const size_t need = sink_attention_workspace_floats(heads, capacity, rows);
    if (!need || part_floats < need) return false;
    const int blocks = int((size_t(capacity) + kKeys - 1) / kKeys);
    const auto s = static_cast<cudaStream_t>(stream);
    split_kernel<<<dim3(blocks, kv_heads, rows), kThreads, 0, s>>>(q, kcodes, kscales, vcodes, vscales, part,
                                                           kv_heads, pos, at, capacity, scale, blocks);
    join_kernel<<<dim3(heads, rows), kDim, 0, s>>>(part, sinks, out, pos, at, capacity, blocks);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        throw std::runtime_error(std::string("sink attention launch: ") + cudaGetErrorString(error));
    return true;
}
} // namespace strata::gemma4
