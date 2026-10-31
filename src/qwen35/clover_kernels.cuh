// Clover's read-only block preview, implemented for TinyStrata's FP32 state layout.
// Equations: Diffuspace/Clover-35B-A3B-Preview/modeling_clover_moe.py,
// bd_gdn_conv1d, bd_gdn_local_memory and coreset_torch (Apache-2.0 model implementation).
// GGUF channel/head ordering follows llama.cpp PR #31306, conversion/clover.py.
#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstddef>

namespace strata::qwen35::clover {

constexpr int kHead = 128;
constexpr int kMaxPreview = 32;
constexpr int kMaxExperts = 512;

// Launch (C / 128, T), 128 threads. The committed convolution history is
// [channel, oldest..newest], qkv/hc are [token, channel]. Clover's right weights
// are independent learned taps at future distances 1,2,3. Nothing is committed.
__global__ void __launch_bounds__(128) conv_preview(
    const float* hist, const float* qkv, const float* left, const float* right,
    float* hc, int C, int HK, float eps, int T) {
    __shared__ float sums[4];
    const int c = (int) blockIdx.x * kHead + (int) threadIdx.x;
    const int t = (int) blockIdx.y;
    if (T < 1 || T > kMaxPreview || t >= T || C % kHead || c >= C) return;
    float sum = 0.f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        const int at = t + tap - 3;
        const float value = at < 0 ? hist[(size_t) c * 3 + at + 3] : qkv[(size_t) at * C + c];
        sum += value * left[(size_t) c * 4 + tap];
    }
#pragma unroll
    for (int distance = 1; distance <= 3; ++distance)
        if (t + distance < T)
            sum += qkv[(size_t) (t + distance) * C + c] * right[(size_t) c * 3 + distance - 1];
    float value = sum / (1.f + expf(-sum));
    if ((int) blockIdx.x < 2 * HK) {
        float square = value * value;
        for (int step = 16; step; step >>= 1) square += __shfl_xor_sync(0xffffffffu, square, step);
        if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = square;
        __syncthreads();
        const float norm = sums[0] + sums[1] + sums[2] + sums[3];
        value *= rsqrtf(norm + eps);
    }
    hc[(size_t) t * C + c] = value;
}

// Launch (HV, T), 128 threads. hc already has normalized Q/K. State uses the
// existing k_gdn_seq layout [key component, value head, value component], not
// HF's [head,value,key] storage. GGUF reorders V heads; head % HK is deliberate.
// The softmax contains all T block keys plus one history bucket. Its history
// value is exp(g) * (q @ S) / sqrt(128); beta is not part of preview.
// Output is raw [T, HV*128], consumed by the existing gated GDN RMS norm.
__global__ void __launch_bounds__(128) gdn_preview(
    const float* state, const float* hc, const float* g,
    const float* logit_scale, const float* history_bias, const float* history_c,
    float* output, int C, int HK, int HV, int T) {
    __shared__ float query[kHead];
    __shared__ float scores[kMaxPreview + 1];
    __shared__ float partial[4];
    const int head = (int) blockIdx.x, t = (int) blockIdx.y, col = (int) threadIdx.x;
    if (T < 1 || T > kMaxPreview || t >= T || head >= HV || HK <= 0) return;
    const int qhead = head % HK;
    query[col] = hc[(size_t) t * C + qhead * kHead + col];
    __syncthreads();
    for (int key = 0; key < T; ++key) {
        float dot = query[col] * hc[(size_t) key * C + HK * kHead + qhead * kHead + col];
        for (int step = 16; step; step >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, step);
        if ((col & 31) == 0) partial[col >> 5] = dot;
        __syncthreads();
        if (col == 0)
            scores[key] = (partial[0] + partial[1] + partial[2] + partial[3]) * logit_scale[head];
        __syncthreads();
    }
    if (col == 0) {
        scores[T] = history_bias[head] + history_c[head] * logf((float) T / 32.f);
        float maximum = scores[0];
        for (int key = 1; key <= T; ++key) maximum = fmaxf(maximum, scores[key]);
        float denominator = 0.f;
        for (int key = 0; key <= T; ++key) {
            scores[key] = expf(scores[key] - maximum);
            denominator += scores[key];
        }
        for (int key = 0; key <= T; ++key) scores[key] /= denominator;
    }
    __syncthreads();
    float remembered = 0.f;
    for (int row = 0; row < kHead; ++row)
        remembered = fmaf(query[row], state[((size_t) row * HV + head) * kHead + col], remembered);
    remembered *= expf(g[(size_t) t * HV + head]) * rsqrtf((float) kHead);
    float local = 0.f;
    for (int key = 0; key < T; ++key)
        local = fmaf(scores[key], hc[(size_t) key * C + 2 * HK * kHead + head * kHead + col], local);
    output[((size_t) t * HV + head) * kHead + col] = fmaf(scores[T], remembered, local);
}

// Optional alternative: one CTA handles a small tile of queries for one value
// head. The recurrence matrix and each block K/V element are loaded once and
// consumed by all queries in the tile. Each query retains the old dot, softmax
// and ordered FMA accumulation; committed state remains read-only.
// Launch (HV, ceil(T / Queries)), 128 threads; no additional global scratch.
template <int Queries = 4>
__global__ void __launch_bounds__(128) gdn_preview_tiled(
    const float* state, const float* hc, const float* g,
    const float* logit_scale, const float* history_bias, const float* history_c,
    float* output, int C, int HK, int HV, int T) {
    static_assert(Queries >= 1 && Queries <= 8);
    __shared__ float query[Queries][kHead];
    __shared__ float scores[Queries][kMaxPreview + 1];
    __shared__ float partial[Queries][4];
    const int head = (int) blockIdx.x, first = (int) blockIdx.y * Queries;
    const int col = (int) threadIdx.x;
    if (T < 1 || T > kMaxPreview || first >= T || head >= HV || HK <= 0) return;
    const int qhead = head % HK;
#pragma unroll
    for (int q = 0; q < Queries; ++q)
        query[q][col] = first + q < T ? hc[(size_t) (first + q) * C + qhead * kHead + col] : 0.f;
    __syncthreads();
    for (int key = 0; key < T; ++key) {
        const float key_value = hc[(size_t) key * C + HK * kHead + qhead * kHead + col];
        float dot[Queries];
#pragma unroll
        for (int q = 0; q < Queries; ++q) dot[q] = query[q][col] * key_value;
        for (int step = 16; step; step >>= 1) {
#pragma unroll
            for (int q = 0; q < Queries; ++q)
                dot[q] += __shfl_xor_sync(0xffffffffu, dot[q], step);
        }
        if ((col & 31) == 0) {
#pragma unroll
            for (int q = 0; q < Queries; ++q) partial[q][col >> 5] = dot[q];
        }
        __syncthreads();
        if (col == 0) {
#pragma unroll
            for (int q = 0; q < Queries; ++q)
                if (first + q < T)
                    scores[q][key] = (partial[q][0] + partial[q][1] + partial[q][2] + partial[q][3]) * logit_scale[head];
        }
        __syncthreads();
    }
    if (col == 0) {
#pragma unroll
        for (int q = 0; q < Queries; ++q) {
            if (first + q >= T) continue;
            scores[q][T] = history_bias[head] + history_c[head] * logf((float) T / 32.f);
            float maximum = scores[q][0];
            for (int key = 1; key <= T; ++key) maximum = fmaxf(maximum, scores[q][key]);
            float denominator = 0.f;
            for (int key = 0; key <= T; ++key) {
                scores[q][key] = expf(scores[q][key] - maximum);
                denominator += scores[q][key];
            }
            for (int key = 0; key <= T; ++key) scores[q][key] /= denominator;
        }
    }
    __syncthreads();
    float remembered[Queries] = {}, local[Queries] = {};
    for (int row = 0; row < kHead; ++row) {
        const float state_value = state[((size_t) row * HV + head) * kHead + col];
#pragma unroll
        for (int q = 0; q < Queries; ++q)
            remembered[q] = fmaf(query[q][row], state_value, remembered[q]);
    }
#pragma unroll
    for (int q = 0; q < Queries; ++q)
        if (first + q < T)
            remembered[q] *= expf(g[(size_t) (first + q) * HV + head]) * rsqrtf((float) kHead);
    for (int key = 0; key < T; ++key) {
        const float value = hc[(size_t) key * C + 2 * HK * kHead + head * kHead + col];
#pragma unroll
        for (int q = 0; q < Queries; ++q)
            if (first + q < T) local[q] = fmaf(scores[q][key], value, local[q]);
    }
#pragma unroll
    for (int q = 0; q < Queries; ++q)
        if (first + q < T)
            output[((size_t) (first + q) * HV + head) * kHead + col] = fmaf(scores[q][T], remembered[q], local[q]);
}

// Launch one 256-thread block. Aggregate the *full softmax* probability of each
// token's stable top-K experts, then select the block's cumulative-mass coreset.
// `stride` may include the appended shared-expert gate; that column is untouched.
// The reference may keep fewer than K experts; subsequent top-K gives zero
// weight to -infinity entries. At least min_experts (normally 1) remain finite.
__global__ void __launch_bounds__(256) router_coreset(
    float* logits, int T, int NE, int K, int stride, float top_p, int min_experts = 1) {
    __shared__ float row[kMaxExperts], mass[kMaxExperts], reduce[256];
    __shared__ float maximum, denominator, total;
    __shared__ unsigned char keep[kMaxExperts];
    const int lane = (int) threadIdx.x;
    if (T < 1 || T > kMaxPreview || NE < 1 || NE > kMaxExperts || K < 1 || K > NE || stride < NE) return;
    for (int expert = lane; expert < NE; expert += 256) mass[expert] = 0.f;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        float local_max = -FLT_MAX;
        for (int expert = lane; expert < NE; expert += 256) {
            row[expert] = logits[(size_t) t * stride + expert];
            local_max = fmaxf(local_max, row[expert]);
        }
        reduce[lane] = local_max;
        __syncthreads();
        for (int step = 128; step; step >>= 1) {
            if (lane < step) reduce[lane] = fmaxf(reduce[lane], reduce[lane + step]);
            __syncthreads();
        }
        if (lane == 0) maximum = reduce[0];
        __syncthreads();
        float local_sum = 0.f;
        for (int expert = lane; expert < NE; expert += 256) local_sum += expf(row[expert] - maximum);
        reduce[lane] = local_sum;
        __syncthreads();
        for (int step = 128; step; step >>= 1) {
            if (lane < step) reduce[lane] += reduce[lane + step];
            __syncthreads();
        }
        if (lane == 0) denominator = reduce[0];
        __syncthreads();
        for (int expert = lane; expert < NE; expert += 256) {
            int rank = 0;
            for (int other = 0; other < NE; ++other)
                rank += row[other] > row[expert] || (row[other] == row[expert] && other < expert);
            if (rank < K) mass[expert] += expf(row[expert] - maximum) / denominator;
        }
        __syncthreads();
    }
    if (lane == 0) {
        total = 0.f;
        for (int expert = 0; expert < NE; ++expert) total += mass[expert];
    }
    __syncthreads();
    for (int expert = lane; expert < NE; expert += 256) {
        int rank = 0;
        float before = 0.f;
        for (int other = 0; other < NE; ++other) {
            if (mass[other] > mass[expert] || (mass[other] == mass[expert] && other < expert)) {
                ++rank;
                before += mass[other];
            }
        }
        keep[expert] = rank < min_experts || before < top_p * total;
    }
    __syncthreads();
    for (int expert = lane; expert < NE; expert += 256)
        if (!keep[expert])
            for (int t = 0; t < T; ++t) logits[(size_t) t * stride + expert] = -INFINITY;
}

// Alternative first phase: independent rows run concurrently. `scores` is
// [T, NE] scratch. Softmax reduction shape and stable top-K tie order are the
// same as router_coreset; no block mass is accumulated with atomics.
__global__ void __launch_bounds__(256) router_coreset_rows(
    const float* logits, float* scores, int T, int NE, int K, int stride) {
    __shared__ float row[kMaxExperts], reduce[256];
    __shared__ float maximum, denominator;
    const int lane = (int) threadIdx.x, t = (int) blockIdx.x;
    if (T < 1 || T > kMaxPreview || t >= T || NE < 1 || NE > kMaxExperts ||
        K < 1 || K > NE || stride < NE) return;
    float local_max = -FLT_MAX;
    for (int expert = lane; expert < NE; expert += 256) {
        row[expert] = logits[(size_t) t * stride + expert];
        local_max = fmaxf(local_max, row[expert]);
    }
    reduce[lane] = local_max;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) reduce[lane] = fmaxf(reduce[lane], reduce[lane + step]);
        __syncthreads();
    }
    if (lane == 0) maximum = reduce[0];
    __syncthreads();
    float local_sum = 0.f;
    for (int expert = lane; expert < NE; expert += 256)
        local_sum += expf(row[expert] - maximum);
    reduce[lane] = local_sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) reduce[lane] += reduce[lane + step];
        __syncthreads();
    }
    if (lane == 0) denominator = reduce[0];
    __syncthreads();
    for (int expert = lane; expert < NE; expert += 256) {
        int rank = 0;
        for (int other = 0; other < NE; ++other)
            rank += row[other] > row[expert] || (row[other] == row[expert] && other < expert);
        scores[(size_t) t * NE + expert] = rank < K
            ? expf(row[expert] - maximum) / denominator : 0.f;
    }
}

// Alternative second phase: sum each expert in token order, then preserve the
// original expert-order total and stable mass-rank/prefix calculations.
// Launch after router_coreset_rows on the same stream: one 256-thread block.
__global__ void __launch_bounds__(256) router_coreset_finish(
    float* logits, const float* scores, int T, int NE, int stride,
    float top_p, int min_experts = 1) {
    __shared__ float mass[kMaxExperts], total;
    __shared__ unsigned char keep[kMaxExperts];
    const int lane = (int) threadIdx.x;
    if (T < 1 || T > kMaxPreview || NE < 1 || NE > kMaxExperts || stride < NE) return;
    for (int expert = lane; expert < NE; expert += 256) {
        float sum = 0.f;
        for (int t = 0; t < T; ++t) sum += scores[(size_t) t * NE + expert];
        mass[expert] = sum;
    }
    __syncthreads();
    if (lane == 0) {
        total = 0.f;
        for (int expert = 0; expert < NE; ++expert) total += mass[expert];
    }
    __syncthreads();
    for (int expert = lane; expert < NE; expert += 256) {
        int rank = 0;
        float before = 0.f;
        for (int other = 0; other < NE; ++other) {
            if (mass[other] > mass[expert] || (mass[other] == mass[expert] && other < expert)) {
                ++rank;
                before += mass[other];
            }
        }
        keep[expert] = rank < min_experts || before < top_p * total;
    }
    __syncthreads();
    for (int expert = lane; expert < NE; expert += 256)
        if (!keep[expert])
            for (int t = 0; t < T; ++t) logits[(size_t) t * stride + expert] = -INFINITY;
}

}  // namespace strata::qwen35::clover
