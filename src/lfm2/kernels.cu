// LFM-only adaptation from VecSzn/PolyStrata, src/poly/kernels.cu
// Revision 26e4cb5df7996edfb5ff549a63b82d8581e1e658 (MIT).
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// src/poly/kernels.cu - see the header.  The reference for every formula is llama.cpp's graph of the model and the
// ggml ops it builds.  Where a kernel takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/lfm2/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace strata::lfm2 {
namespace {

constexpr int WARP = 32;
// attn_score_kernel: a block's warps, a row and a group each; the values of a key a thread reads at a step; the
// queries a key's values are kept for; attn_score_reg_kernel's rows a warp
constexpr int SCORE_WARPS = 8, SCORE_VALUES = 4, SCORE_QUERIES = 4, SCORE_KEYS = 8;
// attend_batch(): the bytes of a chunk's results, which every block of rows reads and writes once more: in the
// card's L2 cache (32 MiB on the 4070 Laptop's AD106) that costs little, from its memory a lot
constexpr size_t kAttendResults = size_t(8) << 20;
// attn_value_kernel: a block's value dimensions, a sum's shares, a block's queries
constexpr int VALUE_DIMS = 32, VALUE_SHARES = 32, VALUE_QUERIES = 8;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// One value per thread summed over the block.  blockDim.x is a multiple of 32, and `part` has a float a warp: 8
// where a block is 256 threads at most, 32 where it can be 1024.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float ldw(const float* w, size_t i) { return w[i]; }
__device__ __forceinline__ float ldw(const uint16_t* w, size_t i) { return __uint_as_float(uint32_t(w[i]) << 16); }

template<typename W>
__global__ void gemv_kernel(const W* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in,
                            int n_out, int rows_per_x) {
    __shared__ float part[32];
    const int o = blockIdx.x;
    const size_t tok = blockIdx.y;
    const W* wr = w + size_t(o) * n_in;
    const float* xr = x + (tok * size_t(n_out / rows_per_x) + size_t(o / rows_per_x)) * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) y[tok * n_out + o] = acc;
}

}  // namespace

// The threads of a block that sums over n values.  A thread reads its values one after the other, and a kernel
// this small takes as long as those reads in a row: four a thread at most, while a block has the threads.  With
// 256 threads for 4096 values, GLM 5.3 Flash's layer norms, small products and stream mixing were 2.7 of the 11
// ms a token its card worked before a layer's router had its scores.
int sum_threads(int n) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < n) threads *= 2;
    return threads;
}

namespace {

template<typename W>
void gemv(const W* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    const int threads = n_in >= 128 ? sum_threads(n_in) : 32;
    gemv_kernel<W><<<dim3(n_out, rows), threads, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, rows_per_x);
}

// A prompt batch's product with a matrix of few rows (a gate a head, the indexer's weights).  cuBLAS cuts such a
// product into a few blocks that each walk all of n_in: 0.44 ms for 1,024 tokens of 2,048 values and 32 outputs on
// an RTX 5090, a quarter of what Qwen3.6's recurrent layers took to read a prompt.  A block here has SMALL_TOK tokens
// and SMALL_OUT outputs, a warp a token and a lane an output, and takes the inputs SMALL_K values at a time.
constexpr int SMALL_TOK = 8, SMALL_OUT = 32, SMALL_K = 64;

__global__ void __launch_bounds__(SMALL_TOK * WARP)
gemm_small_kernel(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
                  int rows) {
    constexpr int THREADS = SMALL_TOK * WARP, XS = SMALL_TOK * SMALL_K / THREADS, WS = SMALL_OUT * SMALL_K / THREADS;
    __shared__ float sx[SMALL_TOK][SMALL_K + 1], sw[SMALL_OUT][SMALL_K + 1];
    const int t0 = blockIdx.x * SMALL_TOK, o0 = blockIdx.y * SMALL_OUT;
    const int tt = threadIdx.x / WARP, oo = threadIdx.x % WARP;
    // the next step's inputs and weights, read while this one is worked on
    float xr[XS], wr[WS];
    const auto fetch = [&](int k0) {
        for (int j = 0; j < XS; ++j) {
            const int i = threadIdx.x + j * THREADS, t = t0 + i / SMALL_K, k = k0 + i % SMALL_K;
            xr[j] = t < rows && k < n_in ? x[size_t(t) * n_in + k] : 0.0f;
        }
        for (int j = 0; j < WS; ++j) {
            const int i = threadIdx.x + j * THREADS, o = o0 + i / SMALL_K, k = k0 + i % SMALL_K;
            wr[j] = o < n_out && k < n_in ? w[size_t(o) * n_in + k] : 0.0f;
        }
    };
    fetch(0);
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_in; k0 += SMALL_K) {
        for (int j = 0; j < XS; ++j) {
            const int i = threadIdx.x + j * THREADS;
            sx[i / SMALL_K][i % SMALL_K] = xr[j];
        }
        for (int j = 0; j < WS; ++j) {
            const int i = threadIdx.x + j * THREADS;
            sw[i / SMALL_K][i % SMALL_K] = wr[j];
        }
        __syncthreads();
        if (k0 + SMALL_K < n_in) fetch(k0 + SMALL_K);
#pragma unroll 16
        for (int c = 0; c < SMALL_K; ++c) acc = fmaf(sx[tt][c], sw[oo][c], acc);
        __syncthreads();
    }
    const int t = t0 + tt, o = o0 + oo;
    if (t < rows && o < n_out) y[size_t(t) * n_out + o] = acc;
}

__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ w, float* __restrict__ out, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ldw(w, i);
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* y, int n,
                                float eps, int y_stride) {
    __shared__ float part[32];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * y_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
}

__global__ void store_f16_kernel(const float* __restrict__ x, half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(x[i]);
}

// Rows from the position the card holds when the launch runs.
__global__ void store_f16_at_kernel(const float* __restrict__ x, half* __restrict__ cache, int n, int rows,
                                    const long long* __restrict__ at, int ring) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rows) return;
    const long long row = ring > 0 ? (*at + i / n) % ring : *at + i / n;
    cache[size_t(row) * n + i % n] = __float2half(x[i]);
}

__global__ void swiglu_clamp_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                    float* __restrict__ out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    out[i] = g / (1.0f + expf(-g)) * u;
}

__global__ void axpy_kernel(const float* __restrict__ x, float a, float* __restrict__ acc, int n, bool add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = add ? acc[i] + x[i] * a : x[i] * a;
}

__global__ void copy2_kernel(const uint32_t* __restrict__ x0, uint32_t* __restrict__ y0, int n0,
                             const uint32_t* __restrict__ x1, uint32_t* __restrict__ y1, int n1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n0) y0[i] = x0[i];
    else if (i < n0 + n1) y1[i - n0] = x1[i - n0];
}

// The sums may be one and the same, so a thread adds its value of each vector in turn.
__global__ void add_scaled_kernel(ScaledRows rows, int count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < count; ++k) rows.acc[k][i] = rows.acc[k][i] + rows.x[k][i] * rows.w[k];
}

__global__ void swiglu_clamp_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n,
                                         float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float g = fminf(row[i], limit);
    const float u = fminf(fmaxf(row[n + i], -limit), limit);
    out[size_t(blockIdx.y) * n + i] = g / (1.0f + expf(-g)) * u;
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

__global__ void geglu_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    out[size_t(blockIdx.y) * n + i] = gelu(row[i]) * row[n + i];
}

// A warp a token, a lane every 32nd expert.  The pick is the host's: the n_used largest of the probabilities plus
// the bias, the lower expert first among equals, one after the other.
constexpr int ROUTE_PER = 16;   // experts a lane: up to 512
// The kernel takes PER experts a lane, the fewest that hold the model's: a slot past them is a -inf that changes no
// pick and no sum, and walking all sixteen made a token's routing 3.5 us on an RTX 5090.
template<int PER>

__global__ void route_kernel(const float* __restrict__ logits, int n_expert, int n_used, int rows, int gate,
                             float in_scale, const float* __restrict__ bias, const float* __restrict__ escale,
                             bool norm, float w_scale, int32_t* __restrict__ ids, float* __restrict__ w) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP, lane = threadIdx.x % WARP;
    if (row >= rows) return;
    const float* lg = logits + size_t(row) * n_expert;
    float p[PER], pick[PER];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int e = lane + k * WARP;
        p[k] = e < n_expert ? lg[e] * in_scale : -INFINITY;
        mx = fmaxf(mx, p[k]);
    }
    if (gate == 0) {   // softmax
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o, WARP));
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            p[k] = lane + k * WARP < n_expert ? expf(p[k] - mx) : 0.0f;
            sum += p[k];
        }
        const float inv = 1.0f / warp_sum(sum);
#pragma unroll
        for (int k = 0; k < PER; ++k) p[k] *= inv;
    } else {
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const float x = p[k];
            p[k] = gate == 1 ? 1.0f / (1.0f + expf(-x)) : sqrtf(x > 20.0f ? x : logf(1.0f + expf(x)));
        }
    }
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int e = lane + k * WARP;
        pick[k] = e < n_expert ? p[k] + (bias ? bias[e] : 0.0f) : -INFINITY;
    }
    float sum = 0.0f, mine = 0.0f;
    int my_id = 0;
    for (int j = 0; j < n_used; ++j) {
        float best = -INFINITY;
        int at = n_expert;
#pragma unroll
        for (int k = 0; k < PER; ++k)
            if (pick[k] > best) best = pick[k], at = lane + k * WARP;
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            const float b = __shfl_xor_sync(0xffffffff, best, o, WARP);
            const int a = __shfl_xor_sync(0xffffffff, at, o, WARP);
            if (b > best || (b == best && a < at)) best = b, at = a;
        }
        float pw = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k)
            if (lane + k * WARP == at) pw = p[k], pick[k] = -INFINITY;
        pw = __shfl_sync(0xffffffff, pw, at % WARP, WARP);
        sum += pw;
        if (lane == j) mine = pw, my_id = at;
    }
    if (lane >= n_used) return;
    float v = mine;
    if (norm) v /= fmaxf(sum, 6.103515625e-5f);   // the floor ggml's MoE graph puts under the gate sum
    if (escale) v *= escale[my_id];
    if (w_scale != 0.0f) v *= w_scale;
    ids[size_t(row) * n_used + lane] = my_id;
    w[size_t(row) * n_used + lane] = v;
}

__global__ void moe_sum_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ acc,
                               int n, int n_used) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t first = size_t(blockIdx.y) * n_used;
    float s = 0.0f;
    for (int k = 0; k < n_used; ++k) s += w[first + k] * x[(first + k) * n + i];
    acc[size_t(blockIdx.y) * n + i] += s;
}

// One block per head and token.  The pairs of the rotation are i and i + n_rot / 2 for i below n_rot / 2.  With
// axes, a token has a position on each of three axes, and pair j turns by the one on axis (j % sections) % 3.
// With `at`, the first token's position is what the card holds there when the launch runs.
__global__ void head_norm_rope_kernel(const float* __restrict__ x, int x_stride, const float* __restrict__ w,
                                      float* __restrict__ y, int dim, int heads, int n_rot, float theta_scale,
                                      const float* __restrict__ freq, long long pos, float eps,
                                      const int* __restrict__ axes, int sections,
                                      const long long* __restrict__ at) {
    __shared__ float part[8];
    const float* xr = x + size_t(blockIdx.x) * x_stride;
    float* yr = y + size_t(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(dim) + eps);
    const int tok = blockIdx.x / heads;
    const float own = float((at ? *at : pos) + tok);
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i >= n_rot) {
            yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
            continue;
        }
        const int j = i < half ? i : i - half;
        const float p = axes ? float(axes[size_t(tok) * 3 + (j % sections) % 3]) : own;
        const float turn = p * powf(theta_scale, float(j));
        const float theta = freq ? turn / freq[j] : turn;
        const float x0 = w ? xr[j] * scale * w[j] : xr[j] * scale;
        const float x1 = w ? xr[j + half] * scale * w[j + half] : xr[j + half] * scale;
        yr[i] = i < half ? x0 * cosf(theta) - x1 * sinf(theta) : x0 * sinf(theta) + x1 * cosf(theta);
    }
}

// The rows the token at position p attends: how many, and the first of them.  They end at its own row, or at row
// `last` when that is given: the tokens of the call then see each other.
__device__ __forceinline__ int attn_rows(long long p, int window, long long last, long long* lo) {
    *lo = window > 0 && p >= window ? p - window + 1 : 0;
    return int((last < 0 ? p : last) - *lo) + 1;
}

// The rows the `tokens` tokens from position p attend between them: how many, and the first of them.  A launch's
// scores lie in that order, so that its tokens read a row of the caches once between them.
__device__ __forceinline__ int attn_span(long long p, int tokens, int window, long long last, long long* lo) {
    long long lo_last;
    const int n = attn_rows(p + tokens - 1, window, last, &lo_last);
    attn_rows(p, window, last, lo);
    return int(lo_last - *lo) + n;
}

// The scores of the key k with N queries that lie one after the other from q, for the warp of attn_score_kernel.
// score: where the first of them has its score of the key's row; the others lie n_keys floats apart.
template <int N, bool Q8 = false>
__device__ __forceinline__ void attn_scores(const float* __restrict__ q, const half* __restrict__ k,
                                            const int8_t* __restrict__ kcodes, const uint16_t* __restrict__ kscales,
                                            float* __restrict__ score, int n_keys, int dim, float scale) {
    static_assert(SCORE_VALUES == 4, "a step is four halves of the key in one read and a float4 of a query");
    const int lane = threadIdx.x % WARP;
    const float* qi[N];
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        qi[i] = q + size_t(i) * dim;
        acc[i] = 0.0f;
    }
    for (int d = SCORE_VALUES * lane; d < dim; d += SCORE_VALUES * WARP) {
        float x[SCORE_VALUES];
        if constexpr (Q8) {
            // kcodes/kscales point at the row's (group's) first code and scale: a 32-value block holds one scale,
            // so the step's block is (SCORE_VALUES * lane) / 32, one scale per 8 lanes of the warp.
            const float sc = __half2float(__ushort_as_half(kscales[d / 32]));
#pragma unroll
            for (int j = 0; j < SCORE_VALUES; ++j) x[j] = (float) kcodes[d + j] * sc;
        } else {
            const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + d);
#pragma unroll
            for (int j = 0; j < SCORE_VALUES; ++j)
                x[j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float4 y = *reinterpret_cast<const float4*>(qi[i] + d);
            acc[i] += y.x * x[0];
            acc[i] += y.y * x[1];
            acc[i] += y.z * x[2];
            acc[i] += y.w * x[3];
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

// attn_scores() for n queries: N at a time, then what is left by halves of N.
template <int N, bool Q8 = false>
__device__ __forceinline__ void attn_scores_by(int n, const float* __restrict__ q, const half* __restrict__ k,
                                               const int8_t* __restrict__ kcodes, const uint16_t* __restrict__ kscales,
                                               float* __restrict__ score, int n_keys, int dim, float scale) {
    for (; n >= N; n -= N, q += size_t(N) * dim, score += size_t(N) * n_keys)
        attn_scores<N, Q8>(q, k, kcodes, kscales, score, n_keys, dim, scale);
    if constexpr (N > 1) attn_scores_by<N / 2, Q8>(n, q, k, kcodes, kscales, score, n_keys, dim, scale);
}

// score[token][head][i] = scale * (q[token][head] . the key of the head's group in the i-th row of the launch's
// span).  A warp has a row and one of the row's groups of heads.  Its threads read the key side by side,
// SCORE_VALUES values each a step, so that a step reads one place of the cache, and the launch's tokens read it
// once between them: the cache of a long context is what the kernel waits for.  A thread sums its products with
// a query, for up to SCORE_QUERIES heads of a token at a time, and a query's 32 sums are added in pairs.  A
// token gets scores for the rows of the span it does not attend as well, which the softmax leaves out.
template <bool Q8 = false>
__global__ void attn_score_kernel(const float* __restrict__ q, const half* __restrict__ kcache,
                                  float* __restrict__ score, int n_keys, int dim, int n_head, int kv_heads,
                                  long long pos, int first, int tokens, float scale, int window, int ring,
                                  long long last, const long long* __restrict__ at,
                                  const int8_t* __restrict__ kcodes = nullptr,
                                  const uint16_t* __restrict__ kscales = nullptr) {
    const int unit = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;   // a row and a group
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    // the row's key of this group: the codes use the fp16 cache's index, the scales block a 32-value block
    const size_t base = (size_t(row) * kv_heads + g) * dim;
    const half* k = kcache + base;
    const int8_t* kc = kcodes + base;
    const uint16_t* ks = kscales + base / 32;
    q += (size_t(first) * n_head + g * per) * dim;   // the group's heads of the launch's first token
    score += size_t(g) * per * n_keys + key;         // and their scores of the row
    for (int t = 0; t < tokens; ++t, q += size_t(n_head) * dim, score += size_t(n_head) * n_keys)
        attn_scores_by<SCORE_QUERIES, Q8>(per, q, k, kc, ks, score, n_keys, dim, scale);
}

// The sums of N scores a lane holds parts of, added over the warp in the tree attn_scores() adds one score in: at
// each of its steps a lane adds the part a lane `off` away has.  While a lane holds more than one score it keeps
// half of them and gives the other half away, so one exchange serves two scores; then the four lanes of a score
// finish it.  A sum is the other tree's to the bit, since a + b is b + a.  Returns the lane's score, `which` of the N.
template <int N, int FROM = WARP / 2>
__device__ __forceinline__ float scores_sum(float* v, int lane, int* which) {
    int idx = 0;
#pragma unroll
    for (int off = FROM, m = N; off > 0; off >>= 1) {
        if (m > 1) {
            const int half = m / 2;
            const bool upper = lane & off;
#pragma unroll
            for (int i = 0; i < half; ++i) {
                const float send = upper ? v[i] : v[i + half], keep = upper ? v[i + half] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
            }
            idx = 2 * idx + (upper ? 1 : 0);
            m = half;
        } else {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], off);
        }
    }
    *which = idx;
    return v[0];
}

// attn_score_kernel for heads of 128 * STEPS values, N of a group's heads at a time: the same products in the same
// order and the same sums, but a warp has SCORE_KEYS rows of a group, and a token's queries stay in its registers
// while it goes through them.  The other kernel read them again for every row, and passed every score's parts from lane to
// lane on its own: those reads and moves were what it waited for.
template <int N, int STEPS, bool Q8 = false>
__global__ void __launch_bounds__(SCORE_WARPS * WARP)
attn_score_reg_kernel(const float* __restrict__ q, const half* __restrict__ kcache, float* __restrict__ score,
                      int n_keys, int n_head, int kv_heads, long long pos, int first, int tokens, float scale,
                      int window, int ring, long long last, const long long* __restrict__ at,
                      const int8_t* __restrict__ kcodes = nullptr, const uint16_t* __restrict__ kscales = nullptr) {
    static_assert(SCORE_VALUES == 4, "a step is four halves of the key in one read and a float4 of a query");
    constexpr int dim = SCORE_VALUES * WARP * STEPS;
    const int unit = blockIdx.x * SCORE_WARPS + threadIdx.x / WARP;   // a group's block of rows
    const int g = unit % kv_heads, k0 = unit / kv_heads * SCORE_KEYS, lane = threadIdx.x % WARP;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, window, last, &lo);
    if (k0 >= span) return;
    const int rows = min(SCORE_KEYS, span - k0), per = n_head / kv_heads;
    for (int th = 0; th < tokens * per; th += N) {   // N heads of a token
        const int t = th / per, h = g * per + th % per;
        const float* qt = q + ((size_t(first) + t) * n_head + h) * dim + SCORE_VALUES * lane;
        float4 y[N][STEPS];
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
            for (int s = 0; s < STEPS; ++s) y[i][s] = *reinterpret_cast<const float4*>(qt + i * dim + 128 * s);
        float* st = score + (size_t(t) * n_head + h) * n_keys + k0;
        long long row = ring > 0 ? (lo + k0) % ring : lo + k0;
        for (int r = 0; r < rows; ++r) {
            // the row's key of this group: the codes use the fp16 cache's index, the scales block a 32-value block
            const size_t base = (size_t(row) * kv_heads + g) * dim;
            const half* k = kcache + base + SCORE_VALUES * lane;
            float acc[N];
#pragma unroll
            for (int i = 0; i < N; ++i) acc[i] = 0.0f;
#pragma unroll
            for (int s = 0; s < STEPS; ++s) {
                float x[SCORE_VALUES];
                if constexpr (Q8) {
                    const int off = SCORE_VALUES * lane + 128 * s;
                    const float d = __half2float(__ushort_as_half(kscales[(base + off) / 32]));
#pragma unroll
                    for (int j = 0; j < SCORE_VALUES; ++j) x[j] = (float) kcodes[base + off + j] * d;
                } else {
                    const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + 128 * s);
#pragma unroll
                    for (int j = 0; j < SCORE_VALUES; ++j)
                        x[j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
                }
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    acc[i] += y[i][s].x * x[0];
                    acc[i] += y[i][s].y * x[1];
                    acc[i] += y[i][s].z * x[2];
                    acc[i] += y[i][s].w * x[3];
                }
            }
            int which;
            const float sum = scores_sum<N>(acc, lane, &which);
            if ((lane & (WARP / N - 1)) == 0) st[size_t(which) * n_keys + r] = sum * scale;
            if (++row == ring) row = 0;
        }
    }
}

// Softmax in place over a head's scores of the rows its token attends; the other rows of the launch's span get
// zero, which adds nothing to the token's values.  One block per head and token.  With `wt` the weights go there
// instead, in the order attn_value_t_kernel reads them: for each of the span's rows, the VALUE_QUERIES queries of
// a block of the group's queries side by side (per: the group's heads).
__global__ void attn_softmax_kernel(float* score, int n_keys, long long pos, int first, int tokens, int window,
                                    long long last, const long long* __restrict__ at, float* __restrict__ wt,
                                    int per) {
    __shared__ float part[8];
    const long long p = (at ? *at : pos) + first;
    long long lo, lo0;
    const int span = attn_span(p, tokens, window, last, &lo0);
    const int n = attn_rows(p + blockIdx.y, window, last, &lo), own = int(lo - lo0);
    float* all = score + (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * n_keys;
    float* row = all + own;   // the token's rows of the span
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
    if (!wt) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] *= inv;
        for (int j = threadIdx.x; j < own; j += blockDim.x) all[j] = 0.0f;
        for (int j = own + n + threadIdx.x; j < span; j += blockDim.x) all[j] = 0.0f;
        return;
    }
    const int h = blockIdx.x, query = blockIdx.y * per + h % per;   // the query among its group's
    const int blocks = (tokens * per + VALUE_QUERIES - 1) / VALUE_QUERIES;
    float* o = wt + (size_t(h / per) * blocks + query / VALUE_QUERIES) * n_keys * VALUE_QUERIES + query % VALUE_QUERIES;
    for (int j = threadIdx.x; j < own; j += blockDim.x) o[size_t(j) * VALUE_QUERIES] = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) o[size_t(own + j) * VALUE_QUERIES] = row[j] * inv;
    for (int j = own + n + threadIdx.x; j < span; j += blockDim.x) o[size_t(j) * VALUE_QUERIES] = 0.0f;
}

// A thread's sums of attn_value_kernel for N queries: over its share of the span's rows from `row` of the cache
// on, of a row's value with each query's weight there.  w: the weights of the lane's query; the warp's first N
// lanes have the N queries between them.
template <int N, bool Q8>
__device__ __forceinline__ void attn_values(const float* __restrict__ w, const half* __restrict__ v,
                                            const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                            size_t width, size_t width_sc, long long row, int ring, int span,
                                            float* sums) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    v += size_t(row) * width;
    if constexpr (Q8) {
        codes += size_t(row) * width;
        scales += size_t(row) * width_sc;
    }
    for (int j = threadIdx.x / VALUE_DIMS; j < span; j += VALUE_SHARES) {
        float x;
        if constexpr (Q8) x = (float) (*codes) * __half2float(__ushort_as_half(*scales));
        else x = __half2float(*v);
        const float weight = w[j];
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i] += __shfl_sync(0xffffffffu, weight, i) * x;
        v += VALUE_SHARES * width;
        if constexpr (Q8) {
            codes += VALUE_SHARES * width;
            scales += VALUE_SHARES * width_sc;
        }
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) {
            row -= ring;
            v -= size_t(ring) * width;
            if constexpr (Q8) {
                codes -= size_t(ring) * width;
                scales -= size_t(ring) * width_sc;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

// attn_values() for n queries, with N if no smaller power of two holds them.
template <int N, bool Q8>
__device__ __forceinline__ void attn_values_for(int n, const float* __restrict__ w, const half* __restrict__ v,
                                                const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                                size_t width, size_t width_sc, long long row, int ring, int span,
                                                float* sums) {
    if constexpr (N > 1)
        if (n <= N / 2)
            return attn_values_for<N / 2, Q8>(n, w, v, codes, scales, width, width_sc, row, ring, span, sums);
    attn_values<N, Q8>(w, v, codes, scales, width, width_sc, row, ring, span, sums);
}

// out[token][head][d] = sum over the rows of score[token][head][row] * the value of the head's group in that
// row.  A block has a group of heads, VALUE_DIMS value dimensions (blockIdx.x) and up to VALUE_QUERIES of the
// group's queries (blockIdx.y): the group's heads of the launch's first token, then of its second, and so on.
// A thread has one of the dimensions and one of VALUE_SHARES shares of the span's rows, every VALUE_SHARES-th
// row from its own, so that many reads of the cache are on their way at a time.  It reads a row's value once
// for the block's queries; their weights of the row come in one read as well, a lane a query, and the warp
// passes them around.  A token's shares are counted from its own first row and added in their order, so that it
// comes out the same whatever tokens the launch has beside it.
template <bool Q8 = false>
__global__ void attn_value_kernel(const float* __restrict__ score, const half* __restrict__ vcache,
                                  float* __restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
                                  long long pos, int first, int tokens, int window, int ring, long long last,
                                  const long long* __restrict__ at, const int8_t* __restrict__ vcodes = nullptr,
                                  const uint16_t* __restrict__ vscales = nullptr) {
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES <= WARP, "a warp is a share and reads a query's weight a lane");
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = blockIdx.x / dims, q0 = blockIdx.y * VALUE_QUERIES;
    const int live = min(VALUE_QUERIES, tokens * per - q0);   // the block's queries
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    // the lane's query, and for a lane past the head's last value a value to read all the same
    const int mine = q0 + min(lane % VALUE_QUERIES, live - 1);
    const int dd = min(d, dim - 1);   // a lane past the last value reads one all the same
    float sums[VALUE_QUERIES];
    attn_values_for<VALUE_QUERIES, Q8>(
        live, score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
        vcache + size_t(g) * dim + dd, vcodes + size_t(g) * dim + dd,
        vscales + size_t(g) * (dim / 32) + dd / 32, size_t(kv_heads) * dim, size_t(kv_heads) * (dim / 32),
        ring > 0 ? (lo + s) % ring : lo + s, ring, span, sums);
    for (int i = 0; i < live; ++i) {
        long long own;   // the first row of the query's token: the thread's rows are this share of the token's
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    __syncthreads();
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

// attend() for a few tokens over the whole conversation, for heads of 64 values in groups of 4 (LFM2's): the keys
// in blocks of SPLIT_KEYS, a launch block each with a group and a token, which scores them (half a warp a row),
// takes the softmax's exponentials of its block for each head (a warp a head) and their values; attn_join_kernel
// adds the blocks up.  For a token of this shape the score, softmax and value kernels launch 16 and 32 blocks, each
// walking the whole conversation.  A block's rows are counted from the cache's first, so a token comes out the
// same whatever tokens the call has beside it.
constexpr int SPLIT_KEYS = 64, SPLIT_THREADS = 128;

__global__ void __launch_bounds__(SPLIT_THREADS)
attn_split_kernel(const float* __restrict__ q, const half* __restrict__ kcache, const half* __restrict__ vcache,
                  float* __restrict__ part, int n_head, int kv_heads, long long pos, int first, int tokens, float scale,
                  int blocks, const long long* __restrict__ at) {
    constexpr int DIM = 64, G = 4, HALF = WARP / 2, WARPS = SPLIT_THREADS / WARP;
    static_assert(SCORE_VALUES * HALF == DIM && WARPS == G, "half a warp a row, a warp a head");
    __shared__ float e[G][SPLIT_KEYS];
    const int g = blockIdx.y, b = blockIdx.x, t = blockIdx.z, lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int side = lane / HALF, hl = lane % HALF;
    const long long p = (at ? *at : pos) + first;
    const int k0 = b * SPLIT_KEYS, span = int(p) + tokens;   // the call's tokens attend rows [0, span)
    if (k0 >= span) return;
    const int nk = min(SPLIT_KEYS, span - k0);
    const int own = int(p) + t + 1;   // the token attends rows [0, own)
    const float* qt = q + ((size_t(first) + t) * n_head + g * G) * DIM + SCORE_VALUES * hl;
    float4 y[G];
#pragma unroll
    for (int i = 0; i < G; ++i) y[i] = *reinterpret_cast<const float4*>(qt + i * DIM);
    for (int pair = w; 2 * pair < nk; pair += WARPS) {   // every lane of the warp goes round alike
        const int r = 2 * pair + side, key = k0 + r;
        float acc[G];
#pragma unroll
        for (int i = 0; i < G; ++i) acc[i] = 0.0f;
        if (r < nk && key < own) {
            const half* k = kcache + (size_t(key) * kv_heads + g) * DIM + SCORE_VALUES * hl;
            const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k);
            float x[SCORE_VALUES];
#pragma unroll
            for (int j = 0; j < SCORE_VALUES; ++j)
                x[j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
#pragma unroll
            for (int i = 0; i < G; ++i) {
                acc[i] += y[i].x * x[0];
                acc[i] += y[i].y * x[1];
                acc[i] += y[i].z * x[2];
                acc[i] += y[i].w * x[3];
            }
        }
        int which;
        const float sum = scores_sum<G, HALF / 2>(acc, lane, &which);
        if (r < nk && (hl & (HALF / G - 1)) == 0) e[which][r] = key < own ? sum * scale : -INFINITY;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = lane; j < nk; j += WARP) m = fmaxf(m, e[w][j]);
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.0f;
    for (int j = lane; j < nk; j += WARP) {
        const float x = m == -INFINITY ? 0.0f : expf(e[w][j] - m);   // a block the token sees none of: 0
        e[w][j] = x;
        l += x;
    }
    l = warp_sum(l);
    __syncthreads();
    // the values: warp w takes every WARPS-th row for all the heads, a lane two values of a row, so a row of
    // the cache is read once; then warp w adds the warps' sums of head w
    float o[G][2];
#pragma unroll
    for (int i = 0; i < G; ++i) o[i][0] = o[i][1] = 0.0f;
    const half* v = vcache + (size_t(k0) * kv_heads + g) * DIM + 2 * lane;
    for (int j = w; j < nk; j += WARPS) {
        const float2 x = __half22float2(*reinterpret_cast<const half2*>(v + size_t(j) * kv_heads * DIM));
#pragma unroll
        for (int i = 0; i < G; ++i) {
            o[i][0] += e[i][j] * x.x;
            o[i][1] += e[i][j] * x.y;
        }
    }
    __shared__ float2 sums[WARPS][G][WARP];
#pragma unroll
    for (int i = 0; i < G; ++i) sums[w][i][lane] = make_float2(o[i][0], o[i][1]);
    __syncthreads();
    float o0 = 0.0f, o1 = 0.0f;
#pragma unroll
    for (int k = 0; k < WARPS; ++k) {
        o0 += sums[k][w][lane].x;
        o1 += sums[k][w][lane].y;
    }
    float* out = part + ((size_t(t) * n_head + g * G + w) * blocks + b) * (DIM + 2);
    out[2 * lane] = o0;
    out[2 * lane + 1] = o1;
    if (lane == 0) {
        out[DIM] = m;
        out[DIM + 1] = l;
    }
}

// The blocks of attn_split_kernel added up for a head of a token: each block's sums scaled to the largest score.  A
// warp takes every JOIN_WARPS-th block, a lane two values, and the warps' sums are added in their order; a block past
// the token's own rows has none it attends and is skipped, so the sums are the same whatever tokens the call has.
constexpr int JOIN_WARPS = 16;

__global__ void __launch_bounds__(JOIN_WARPS * WARP)
attn_join_kernel(const float* __restrict__ part, float* __restrict__ out, int n_head, long long pos, int first,
                 int tokens, int blocks, const long long* __restrict__ at) {
    constexpr int DIM = 64;
    __shared__ float top[JOIN_WARPS], sl[JOIN_WARPS], so[JOIN_WARPS][DIM];
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const long long p = (at ? *at : pos) + first;
    const int used = int((p + tokens + SPLIT_KEYS - 1) / SPLIT_KEYS);   // the blocks the split kernel wrote
    const float* pb = part + (size_t(t) * n_head + h) * blocks * (DIM + 2);
    float mx = -INFINITY;
    for (int b = threadIdx.x; b < used; b += JOIN_WARPS * WARP) mx = fmaxf(mx, pb[size_t(b) * (DIM + 2) + DIM]);
    mx = block_max(mx, top);
    float l = 0.0f, o0 = 0.0f, o1 = 0.0f;
#pragma unroll 4
    for (int b = w; b < used; b += JOIN_WARPS) {
        const float* x = pb + size_t(b) * (DIM + 2);
        if (x[DIM] == -INFINITY) continue;
        const float f = expf(x[DIM] - mx);
        l += f * x[DIM + 1];
        o0 += f * x[2 * lane];
        o1 += f * x[2 * lane + 1];
    }
    so[w][2 * lane] = o0;
    so[w][2 * lane + 1] = o1;
    if (lane == 0) sl[w] = l;
    __syncthreads();
    if (w != 0) return;
    float s0 = 0.0f, s1 = 0.0f, sum = 0.0f;
    for (int k = 0; k < JOIN_WARPS; ++k) {
        s0 += so[k][2 * lane];
        s1 += so[k][2 * lane + 1];
        sum += sl[k];
    }
    float* y = out + ((size_t(first) + t) * n_head + h) * DIM;
    y[2 * lane] = s0 / sum;
    y[2 * lane + 1] = s1 / sum;
}

// attn_value_kernel with the weights as attn_softmax_kernel leaves them with `wt`: a row's queries side by side,
// so a warp reads a row's weights in two loads that every lane makes alike, where the other passes them from lane
// to lane, and so waited on the passing (Nsight Compute).  A block has TILES of the blocks of VALUE_QUERIES
// queries, which read the cache's values once between them.  The products go in the rows' order: the sums are
// the other kernel's to the bit.
template <int TILES, bool Q8 = false>
__global__ void __launch_bounds__(VALUE_DIMS * VALUE_SHARES)
attn_value_t_kernel(const float* __restrict__ wt, const half* __restrict__ vcache, float* __restrict__ out, int n_keys,
                    int dim, int n_head, int kv_heads, long long pos, int first, int tokens, int window, int ring,
                    long long last, const long long* __restrict__ at,
                    const int8_t* __restrict__ vcodes = nullptr, const uint16_t* __restrict__ vscales = nullptr) {
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES == 8, "a warp is a share, and a row's weights are two float4");
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int blocks = (tokens * per + VALUE_QUERIES - 1) / VALUE_QUERIES;   // the group's blocks of queries
    const int g = blockIdx.x / dims, b0 = blockIdx.y * TILES;
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    const size_t width = size_t(kv_heads) * dim;
    const half* v = vcache + size_t(g) * dim + min(d, dim - 1);   // a lane past the last value reads one all the same
    // the q8 read: the codes use the fp16 cache's index, the scales a 32-value block (width / 32, one a row's block)
    const int8_t* vc = vcodes + size_t(g) * dim + min(d, dim - 1);
    const uint16_t* vs = vscales + size_t(g) * (dim / 32) + min(d, dim - 1) / 32;
    const float* w[TILES];   // a tile past the group's last reads the block's first again, for nothing
#pragma unroll
    for (int t = 0; t < TILES; ++t)
        w[t] = wt + (size_t(g) * blocks + (b0 + t < blocks ? b0 + t : b0)) * n_keys * VALUE_QUERIES;
    float acc[TILES][VALUE_QUERIES];
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int i = 0; i < VALUE_QUERIES; ++i) acc[t][i] = 0.0f;
    long long row = ring > 0 ? (lo + s) % ring : lo + s;
    for (int j = s; j < span; j += VALUE_SHARES) {
        float x;
        if constexpr (Q8)
            x = (float) vc[size_t(row) * width] * __half2float(__ushort_as_half(vs[size_t(row) * (width / 32)]));
        else
            x = __half2float(v[size_t(row) * width]);
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const float4 a = *reinterpret_cast<const float4*>(w[t] + size_t(j) * VALUE_QUERIES);
            const float4 b = *reinterpret_cast<const float4*>(w[t] + size_t(j) * VALUE_QUERIES + 4);
            acc[t][0] += a.x * x;
            acc[t][1] += a.y * x;
            acc[t][2] += a.z * x;
            acc[t][3] += a.w * x;
            acc[t][4] += b.x * x;
            acc[t][5] += b.y * x;
            acc[t][6] += b.z * x;
            acc[t][7] += b.w * x;
        }
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) row -= ring;
    }
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int q0 = (b0 + t) * VALUE_QUERIES, live = min(VALUE_QUERIES, tokens * per - q0);
        if (live <= 0) break;
        if (t > 0) __syncthreads();   // the tile before has its sums out of share
        for (int i = 0; i < live; ++i) {
            long long own;   // the first row of the query's token: the thread's rows are this share of the token's
            attn_rows(p + (q0 + i) / per, window, last, &own);
            share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = acc[t][i];
        }
        __syncthreads();
        if (s == 0 && d < dim)
            for (int i = 0; i < live; ++i) {
                float sum = share[i][0][lane];
                for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
                const int query = q0 + i;
                out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
            }
    }
}

// A chunk's queries and its results lie in two orders: the engine's (token, head) and the one a product per
// group of heads takes (group, token, head of the group).  This is where the engine's `row` lies in the second.
__device__ __forceinline__ size_t attn_by_group(int row, int n_head, int kv_heads, int tokens) {
    const int per = n_head / kv_heads, t = row / n_head, h = row % n_head;
    return (size_t(h / per) * tokens + t) * per + h % per;
}

// The queries to the order of the groups, as the halves the products take.  blockIdx.y is the token and the head.
__global__ void attn_group_kernel(const float* __restrict__ src, half* __restrict__ dst, int dim, int n_head,
                                  int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float x = src[size_t(blockIdx.y) * dim + i];
    dst[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i] = __float2half(x);
}

// The results back to the engine's order.
__global__ void attn_ungroup_kernel(const float* __restrict__ src, float* __restrict__ dst, int dim, int n_head,
                                    int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    dst[size_t(blockIdx.y) * dim + i] = src[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i];
}

// A softmax over a chunk's rows a block of rows at a time, for scores in the order of the groups: blockIdx.x
// counts (group, token, head of the group), and the block's rows start at row0 of the sequence.  A score's weight
// is its exp against the largest its token and head have had (top), zero for a row the token does not attend, as
// the half the values' product takes.  sum is the sum of the weights against that top, and shrink what the
// weights of the blocks before are worth against it.  With `fresh` there was no block before.
__global__ void attn_softmax_block_kernel(const float* __restrict__ score, half* __restrict__ weight, int n_keys,
                                          long long row0, long long pos, int first, int window, int tokens, int per,
                                          long long last, float* top, float* sum, float* shrink, bool fresh) {
    __shared__ float part[8];
    const int tok = first + (blockIdx.x % (tokens * per)) / per;
    long long lo;
    const int n = attn_rows(pos + tok, window, last, &lo);
    const long long end = lo + n;   // the token attends the sequence's rows from lo to before end
    const int j0 = lo > row0 ? int(lo - row0) : 0, j1 = end - row0 < n_keys ? int(end - row0) : n_keys;
    const float* row = score + size_t(blockIdx.x) * n_keys;
    half* w = weight + size_t(blockIdx.x) * n_keys;
    float mx = -INFINITY;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    const float before = fresh ? -INFINITY : top[blockIdx.x], now = fmaxf(before, mx);
    float s = 0.0f;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) {
        w[j] = __float2half(j >= j0 && j < j1 ? expf(row[j] - now) : 0.0f);
        s += __half2float(w[j]);   // the sum is of the weights as the values get them
    }
    s = block_sum(s, part);
    if (threadIdx.x != 0) return;
    const float f = mx == -INFINITY ? 1.0f : expf(before - now);   // none of the block's rows: nothing changes
    top[blockIdx.x] = now;
    sum[blockIdx.x] = (fresh ? 0.0f : sum[blockIdx.x] * f) + s;
    shrink[blockIdx.x] = f;
}

// A chunk's results in the order of the groups, `dim` values a token and head: times its factor, or divided by it.
__global__ void attn_scale_kernel(float* out, const float* __restrict__ factor, int dim, bool divide) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float f = factor[blockIdx.y];
    // a row whose largest score this block of keys did not raise is times 1: it is not read or written
    if (!divide && f == 1.0f) return;
    out[size_t(blockIdx.y) * dim + i] *= divide ? 1.0f / f : f;
}

// The tensor cores' m16n8k16 step and ldmatrix need Turing; AMD builds have neither, and the host keeps the cuBLAS
// products there (tensor_cores()).  Turing runs the step as two of k = 8 on the same registers.
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750)
#define STRATA_POLY_MMA 0
#else
#define STRATA_POLY_MMA 1
#endif

#if STRATA_POLY_MMA
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t half2_bits(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// 16 bytes from global to shared memory, or zeros where `whole` is false; asynchronous from Ampere on (cp.async: a
// thread's copies are a group from flash_commit() on, waited for by flash_wait()), a plain load and store on Turing.
__device__ __forceinline__ void flash_copy(half* dst, const half* src, bool whole) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(whole ? 16 : 0));
#else
    *reinterpret_cast<uint4*>(dst) = whole ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void flash_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most `ahead` of the thread's groups are still on their way.
template <int ahead> __device__ __forceinline__ void flash_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(ahead));
#endif
}
#endif

// attend_batch() for heads of 64 values in groups of 4 (LFM2's), the way FlashAttention-2 goes: a launch block takes
// FLASH_TOKENS tokens of a group, 64 rows of queries with a token's 4 heads side by side, a warp 16 of them, and walks
// the keys FLASH_KEYS at a time on the tensor cores, keeping each row's softmax as it goes, so no score leaves the
// registers.  The queries and the weights are rounded to halves and the products sum floats, as with the cuBLAS
// products.  Fragment layouts are PTX's for mma.m16n8k16: a thread holds rows gid and gid + 8, columns 2 tig and
// 2 tig + 1 of each block of 8.  The next block of keys and values is copied in while one is worked on, two buffers
// in turn.
constexpr int FLASH_TOKENS = 16;
__global__ void __launch_bounds__(4 * WARP)
attn_flash64_kernel(const float* __restrict__ q, const half* __restrict__ kcache, const half* __restrict__ vcache,
                    float* __restrict__ out, int rows, int n_head, int kv_heads, long long pos, float scale_log2) {
#if STRATA_POLY_MMA
    // FLASH_ROW: a cache row in shared memory, in halves; 16-byte aligned, and ldmatrix's eight rows on different
    // banks.
    constexpr int DIM = 64, G = 4, FLASH_KEYS = 64, FLASH_ROW = 64 + 8;
    __shared__ __align__(16) half ks[2][FLASH_KEYS][FLASH_ROW], vs[2][FLASH_KEYS][FLASH_ROW];
    const int g = blockIdx.y, t0 = blockIdx.x * FLASH_TOKENS, lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int gid = lane / 4, tig = lane % 4;
    int tok[2], head[2];   // the thread's rows gid and gid + 8 of the warp's
    for (int i = 0; i < 2; ++i) {
        const int r = w * 16 + gid + 8 * i;
        tok[i] = t0 + r / G;
        head[i] = g * G + r % G;
    }
    uint32_t qa[4][4];   // a block of 16 values a step, in log2 units
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < rows)
                    x = *reinterpret_cast<const float2*>(q + (size_t(tok[i]) * n_head + head[i]) * DIM + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = half2_bits(x.x * scale_log2, x.y * scale_log2);
            }
    float o[8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    const int n_keys = int(pos) + min(t0 + FLASH_TOKENS, rows);   // the block's last token attends rows [0, n_keys)
    const auto fetch = [&](int buf, int k0) {
        for (int i = threadIdx.x; i < FLASH_KEYS * DIM / 8; i += 4 * WARP) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const bool whole = k0 + r < n_keys;
            const size_t at = whole ? (size_t(k0 + r) * kv_heads + g) * DIM + c : 0;
            flash_copy(&ks[buf][r][c], kcache + at, whole);
            flash_copy(&vs[buf][r][c], vcache + at, whole);
        }
        flash_commit();
    };
    fetch(0, 0);
    for (int k0 = 0, buf = 0; k0 < n_keys; k0 += FLASH_KEYS, buf ^= 1) {
        __syncthreads();   // the buffer the last block was worked on in is free
        if (k0 + FLASH_KEYS < n_keys) {
            fetch(buf ^ 1, k0 + FLASH_KEYS);
            flash_wait<1>();
        } else {
            flash_wait<0>();
        }
        __syncthreads();
        float s[8][4];
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 4; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        for (int i = 0; i < 2; ++i) {
            const int own = int(pos) + tok[i];   // the row's token attends rows [0, own]
            float mx = -INFINITY;
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    if (k0 + 8 * n + 2 * tig + j % 2 > own) s[n][j] = -INFINITY;
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
            for (int n = 0; n < 8; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = exp2f(s[n][j] - base);
                    l[i] += s[n][j];
                }
            }
        }
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 8; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= rows) continue;
        float* y = out + (size_t(tok[i]) * n_head + head[i]) * DIM + 2 * tig;
        for (int n = 0; n < 8; ++n)
            *reinterpret_cast<float2*>(y + 8 * n) = make_float2(o[n][2 * i] / l[i], o[n][2 * i + 1] / l[i]);
    }
#else
    (void) q, (void) kcache, (void) vcache, (void) out, (void) rows, (void) n_head, (void) kv_heads, (void) pos,
        (void) scale_log2;   // a card without the tensor cores' steps never launches it (tensor_cores())
#endif
}

// One block of 1024; a thread has every 1024th value.
__global__ void likeliest_kernel(const float* __restrict__ x, int n, Likeliest* __restrict__ out) {
    __shared__ float part[32];
    __shared__ int first[32];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    m = block_max(m, part);
    int at = n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] == m) at = min(at, i);
        if (x[i] > m - 20.0f) sum += expf(x[i] - m);
    }
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) at = min(at, __shfl_xor_sync(0xffffffff, at, o, WARP));
    if ((threadIdx.x & 31) == 0) first[threadIdx.x >> 5] = at;
    sum = block_sum(sum, part);   // and first is written
    if (threadIdx.x != 0) return;
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) at = min(at, first[k]);
    out->token = at < n ? at : 0;
    out->prob = 1.0f / sum;
}

// Where a token leaves the recurrent state as it is after it: `snaps` has a bit for each of a call's tokens that
// does, and their states lie in the tokens' order, `stride` floats apart.  Null for a token that leaves none.
__device__ __forceinline__ float* snap_of(float* snap, size_t stride, unsigned snaps, int row) {
    return row < 32 && (snaps >> row & 1u) ? snap + size_t(__popc(snaps & ((1u << row) - 1u))) * stride : nullptr;
}

// From this many tokens a causal convolution takes them all at once: a walk is one launch, and the two launches of
// the parallel way cost a token alone more than its walk.
constexpr int CONV_ROWS = 16;

// LFM2's short convolution.  A token's in_proj row is B, C and x, `channels` values each; the convolution runs over
// B * x with three taps, and C scales its output.  The state is the two B * x before the call, oldest first.  A
// thread is a channel and walks the tokens; the state as it is after a token of `snaps` goes to snap.
__global__ void shortconv_kernel(const float* __restrict__ bcx, float* __restrict__ state,
                                 const float* __restrict__ w, float* __restrict__ out, int channels, int rows,
                                 float* __restrict__ snap, size_t stride, unsigned snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + 3 * size_t(c);
    float* st = state + size_t(c) * 2;
    float s0 = st[0], s1 = st[1];
    for (int row = 0; row < rows; ++row) {
        const float* in = bcx + size_t(row) * 3 * channels;
        const float bx = in[c] * in[2 * channels + c];
        const float y = s0 * wc[0] + s1 * wc[1] + bx * wc[2];
        s0 = s1;
        s1 = bx;
        out[size_t(row) * channels + c] = in[channels + c] * y;
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m[size_t(c) * 2] = s0;
            m[size_t(c) * 2 + 1] = s1;
        }
    }
    st[0] = s0;
    st[1] = s1;
}

// shortconv_kernel for a call that leaves no snapshots: a thread a channel of a token, as conv4_silu_rows_kernel.
__global__ void shortconv_rows_kernel(const float* __restrict__ bcx, const float* __restrict__ state,
                                      const float* __restrict__ w, float* __restrict__ out, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int row = blockIdx.y;
    const float* wc = w + 3 * size_t(c);
    auto bx = [&](int r) {
        if (r < 0) return state[size_t(c) * 2 + 2 + r];
        const float* in = bcx + size_t(r) * 3 * channels;
        return in[c] * in[2 * channels + c];
    };
    const float y = bx(row - 2) * wc[0] + bx(row - 1) * wc[1] + bx(row) * wc[2];
    out[size_t(row) * channels + c] = bcx[size_t(row) * 3 * channels + channels + c] * y;
}

// The short convolution's state after the call: its last two B * x, the state's own where the call has fewer.
__global__ void shortconv_state_kernel(const float* __restrict__ bcx, float* __restrict__ state, int channels,
                                       int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* st = state + size_t(c) * 2;
    auto bx = [&](int r) {
        if (r < 0) return st[2 + r];
        const float* in = bcx + size_t(r) * 3 * channels;
        return in[c] * in[2 * channels + c];
    };
    const float a = bx(rows - 2), b = bx(rows - 1);
    st[0] = a;
    st[1] = b;
}

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

// Whether the tensor-core kernels run as such here: built for Turing or newer (STRATA_POLY_MMA), and not from a PTX
// of an older card compiled for this one.
bool tensor_cores() {
#if defined(__HIPCC__)
    return false;
#else
    cudaFuncAttributes a{};
    return cudaFuncGetAttributes(&a, attn_flash64_kernel) == cudaSuccess && a.ptxVersion >= 75;
#endif
}

void blas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
}

}  // namespace

void* blas_create(void* stream) {
    cublasHandle_t h = nullptr;
    blas_ck(cublasCreate(&h), "cublasCreate");
    if (const cublasStatus_t s = cublasSetStream(h, cs(stream)); s != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(h);
        blas_ck(s, "cublasSetStream");
    }
    // cuBLAS takes 75 ms over a process's first product of halves, whatever its size: a prompt batch's attention
    // would wait for it.  A product that fails here fails again where its status is read.
    const int n = 16;
    void* p = nullptr;
    if (cudaMalloc(&p, n * n * (2 * sizeof(half) + sizeof(float))) != cudaSuccess) {
        cudaGetLastError();   // or the next launch's check reports it
        return h;
    }
    cudaMemsetAsync(p, 0, n * n * (2 * sizeof(half) + sizeof(float)), cs(stream));
    const float one = 1.0f, zero = 0.0f;
    const half* a = (const half*) p;
    cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, a, CUDA_R_16F, n, n * n, a + n * n,
                               CUDA_R_16F, n, n * n, &zero, (float*) (a + 2 * n * n), CUDA_R_32F, n, n * n, 1,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(cs(stream));
    cudaFree(p);
    return h;
}

void blas_destroy(void* blas) {
    if (blas) cublasDestroy((cublasHandle_t) blas);
}

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

// A buffer of rows is its transpose to cuBLAS, so y[row] = w x[row] is Y = W^T X.  A group's rows of w, its part
// of every token's x and of every token's y are each a matrix with a stride.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int groups = n_out / rows_per_x;
    if (groups == 1 && n_out <= 8 * SMALL_OUT) {
        cudaStream_t stream = nullptr;
        blas_ck(cublasGetStream(h, &stream), "cublasGetStream");
        gemm_small_kernel<<<dim3((rows + SMALL_TOK - 1) / SMALL_TOK, (n_out + SMALL_OUT - 1) / SMALL_OUT),
                            SMALL_TOK * WARP, 0, stream>>>(w, x, y, n_in, n_out, rows);
        return;
    }
    if (groups == 1) {
        blas_ck(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, rows, n_in, &one, w, n_in, x, n_in, &zero, y, n_out),
                "a batch's projection");
        return;
    }
    blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, rows_per_x, rows, n_in, &one, w, n_in,
                                      (long long) rows_per_x * n_in, x, groups * n_in, n_in, &zero, y, n_out,
                                      rows_per_x, groups),
            "a batch's projections");
}

void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream) {
    bf16_to_f32_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(w, out, n);
}

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride) {
    rms_norm_kernel<<<rows, sum_threads(n), 0, cs(stream)>>>(x, w, y, n, eps, y_stride ? y_stride : n);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("rms_norm: ") + cudaGetErrorString(e));
}

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream) {
    const size_t all = size_t(n) * rows;
    store_f16_kernel<<<grid(all, 256), 256, 0, cs(stream)>>>(x, (half*) cache + size_t(pos) * n, int(all));
}

void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream) {
    store_f16_at_kernel<<<grid(size_t(n) * rows, 256), 256, 0, cs(stream)>>>(x, (half*) cache, n, rows, at, ring);
}

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n, limit);
}

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    axpy_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, a, acc, n, add);
}

void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream) {
    copy2_kernel<<<grid(n0 + n1, 256), 256, 0, cs(stream)>>>((const uint32_t*) x0, (uint32_t*) y0, n0,
                                                             (const uint32_t*) x1, (uint32_t*) y1, n1);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("copy2: ") + cudaGetErrorString(e));
}

void add_scaled(const ScaledRows& rows, int count, int n, void* stream) {
    if (count < 1 || count > 32) throw std::invalid_argument("add_scaled: 1 to 32 vectors");
    add_scaled_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(rows, count, n);
}

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream) {
    swiglu_clamp_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, limit);
}

void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream) {
    geglu_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n);
}

void route_rows(const float* logits, int n_expert, int n_used, int rows, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int32_t* ids, float* w, void* stream) {
    if (rows <= 0) return;
    if (n_expert > ROUTE_PER * WARP || n_used > WARP)
        throw std::invalid_argument("route_rows: more than 512 experts, or more than 32 a token");
    const auto launch = [&](auto kernel) {
        kernel<<<grid(rows * WARP, 256), 256, 0, cs(stream)>>>(logits, n_expert, n_used, rows, gate, in_scale, bias,
                                                               escale, norm, w_scale, ids, w);
    };
    if (n_expert <= 2 * WARP) launch(route_kernel<2>);
    else if (n_expert <= 4 * WARP) launch(route_kernel<4>);
    else if (n_expert <= 8 * WARP) launch(route_kernel<8>);
    else launch(route_kernel<ROUTE_PER>);
}

void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream) {
    moe_sum_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(x, w, acc, n, n_used);
}

void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq,
                    const int32_t* axes, int sections, const long long* at) {
    head_norm_rope_kernel<<<rows * heads, dim >= 256 ? 256 : 128, 0, cs(stream)>>>(
        x, x_stride, w, y, dim, heads, n_rot, theta_scale, freq, (long long) pos, eps, axes, sections, at);
}

void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
            int ring, bool whole, const long long* at, int room, size_t score_floats, Kvq8 kq8, Kvq8 vq8) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend: a position on the card needs its room");
    if (dim % SCORE_VALUES || uintptr_t(q) % (SCORE_VALUES * sizeof(float)) || uintptr_t(kcache) % (SCORE_VALUES * 2))
        throw std::invalid_argument("attend: queries and keys are read four values at a time");
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    // attn_split_kernel, where score has room for every block's sums at the longest the call can attend
    const int most = grid(size_t(at ? room : pos + rows), SPLIT_KEYS);
    if (dim == 64 && per == 4 && window == 0 && ring == 0 && !whole && !kq && !vq &&
        size_t(chunk) * n_head * most * (dim + 2) <= score_floats) {
        for (int first = 0; first < rows; first += chunk) {
            const int c = std::min(chunk, rows - first);
            const int blocks = grid(size_t(at ? room : pos + first + c), SPLIT_KEYS);
            attn_split_kernel<<<dim3(blocks, kv_heads, c), SPLIT_THREADS, 0, cs(stream)>>>(
                q, (const half*) kcache, (const half*) vcache, score, n_head, kv_heads, (long long) pos, first, c,
                scale, blocks, at);
            attn_join_kernel<<<dim3(n_head, c), JOIN_WARPS * WARP, 0, cs(stream)>>>(score, out, n_head, (long long) pos,
                                                                                first, c, blocks, at);
        }
        return;
    }
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // the launch's span: from the first row its first token attends to the last its last token does
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        const unsigned units = grid(size_t(keys), SCORE_KEYS) * kv_heads, threads = SCORE_WARPS * WARP;
        const auto reg = [&](auto kernel) {
            kernel<<<grid(units, SCORE_WARPS), threads, 0, cs(stream)>>>(q, (const half*) kcache, score, keys, n_head,
                                                                        kv_heads, (long long) pos, first, c, scale,
                                                                        window, ring, last, at, kq8.codes,
                                                                        kq8.scales);
        };
        // The queries in registers where a head has 128, 256 or 512 values and a group 4 or 8 heads.  With fewer
        // heads a group the cache is the most of what is read and the other kernel is as fast.  A q8 key cache
        // takes the same kernel the fp16 cache does, which reads it as Q8_0.
        if (per % 4 == 0 && dim == 128) {
            if (kq) reg(attn_score_reg_kernel<4, 1, true>);
            else reg(attn_score_reg_kernel<4, 1, false>);
        } else if (per % 8 == 0 && dim == 256) {
            if (kq) reg(attn_score_reg_kernel<8, 2, true>);
            else reg(attn_score_reg_kernel<8, 2, false>);
        } else if (per % 4 == 0 && dim == 256) {
            if (kq) reg(attn_score_reg_kernel<4, 2, true>);
            else reg(attn_score_reg_kernel<4, 2, false>);
        } else if (per % 4 == 0 && dim == 512) {
            if (kq) reg(attn_score_reg_kernel<4, 4, true>);
            else reg(attn_score_reg_kernel<4, 4, false>);
        } else if (kq)
            attn_score_kernel<true><<<grid(size_t(keys) * kv_heads, SCORE_WARPS), threads, 0, cs(stream)>>>(
                q, (const half*) kcache, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale,
                window, ring, last, at, kq8.codes, kq8.scales);
        else
            attn_score_kernel<false><<<grid(size_t(keys) * kv_heads, SCORE_WARPS), threads, 0, cs(stream)>>>(
                q, (const half*) kcache, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale,
                window, ring, last, at);
        // the weights as attn_value_t_kernel reads them, behind the scores where the buffer has room for them.
        // A q8 value cache takes the same value kernel the fp16 cache does, which reads it as Q8_0.
        const int blocks = grid(c * per, VALUE_QUERIES);
        float* wt = per >= 4 && size_t(kv_heads) * blocks * VALUE_QUERIES <= size_t(chunk - c) * n_head
                        ? score + size_t(c) * n_head * keys : nullptr;
        attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c, window,
                                                                    last, at, wt, per);
        // Two blocks of queries a launch block read the values half as often, where the launch still has
        // blocks for most of the card's 36 multiprocessors: for Qwen3.6's two tokens they are half as many and
        // slower, for three or four tokens and for Gemma 4's full layers faster.
        const int dims = grid(dim, VALUE_DIMS);
        if (wt && blocks > 1 && kv_heads * dims * grid(blocks, 2) >= 32) {
            if (vq)
                attn_value_t_kernel<2, true><<<dim3(kv_heads * dims, grid(blocks, 2)), VALUE_DIMS * VALUE_SHARES,
                                               0, cs(stream)>>>(wt, (const half*) vcache, out, keys, dim, n_head,
                                                                kv_heads, (long long) pos, first, c, window, ring, last,
                                                                at, vq8.codes, vq8.scales);
            else
                attn_value_t_kernel<2, false><<<dim3(kv_heads * dims, grid(blocks, 2)), VALUE_DIMS * VALUE_SHARES,
                                                0, cs(stream)>>>(wt, (const half*) vcache, out, keys, dim, n_head,
                                                                 kv_heads, (long long) pos, first, c, window, ring, last,
                                                                 at);
        } else if (wt) {
            if (vq)
                attn_value_t_kernel<1, true><<<dim3(kv_heads * dims, blocks), VALUE_DIMS * VALUE_SHARES, 0,
                                               cs(stream)>>>(wt, (const half*) vcache, out, keys, dim, n_head,
                                                             kv_heads, (long long) pos, first, c, window, ring, last,
                                                             at, vq8.codes, vq8.scales);
            else
                attn_value_t_kernel<1, false><<<dim3(kv_heads * dims, blocks), VALUE_DIMS * VALUE_SHARES, 0,
                                                cs(stream)>>>(wt, (const half*) vcache, out, keys, dim, n_head,
                                                              kv_heads, (long long) pos, first, c, window, ring,
                                                              last, at);
        } else if (vq)
            attn_value_kernel<true><<<dim3(kv_heads * dims, blocks), VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(
                score, (const half*) vcache, out, keys, dim, n_head, kv_heads, (long long) pos, first, c, window, ring,
                last, at, vq8.codes, vq8.scales);
        else
            attn_value_kernel<false><<<dim3(kv_heads * dims, blocks), VALUE_DIMS * VALUE_SHARES, 0,
                                      cs(stream)>>>(score, (const half*) vcache, out, keys, dim, n_head, kv_heads,
                                                    (long long) pos, first, c, window, ring, last, at);
    }
}

// The rows of a q8_0 cache a batch product is about to read, dequantized into the fp16 workspace it reads
// instead: out[r * width + i] = (float)codes[row + r][i] * __half2float(scales[row + r][i / 32]).  blockIdx.y is
// the row of the block (blockIdx.x the columns).  A block never wraps a ring: it ends where a ring starts over.
__global__ void dequant_q8_kernel(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                  half* __restrict__ out, int width, long long row) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t r = (size_t) (row + blockIdx.y), at = r * width;
    const float d = __half2float(__ushort_as_half(scales[r * (width >> 5) + (i >> 5)]));
    out[(size_t) blockIdx.y * width + i] = __float2half((float) codes[at + i] * d);
}

size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim) {
    return size_t(chunk) * n_head * grid(size_t(max_rows), SPLIT_KEYS) * (dim + 2);
}

void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window, int ring, bool whole, Kvq8 kq8, Kvq8 vq8, uint16_t* kwork,
                  uint16_t* vwork, size_t work_halves) {
    const int per = n_head / kv_heads, width = kv_heads * dim;
    // The fp16-only dim==64 fast path reads the fp16 caches, which are null under q8 (a q8 cache is load-restricted
    // to qwen35moe's head_dim 128, so it never reaches here); guard explicitly so it is provably not taken for q8.
    if (dim == 64 && per == 4 && window == 0 && ring == 0 && !whole && kq8.codes == nullptr && vq8.codes == nullptr &&
        tensor_cores()) {
        attn_flash64_kernel<<<dim3(grid(rows, FLASH_TOKENS), kv_heads), 4 * WARP, 0, cs(stream)>>>(
            q, (const half*) kcache, (const half*) vcache, out, rows, n_head, kv_heads, (long long) pos,
            scale * 1.44269504f);
        return;
    }
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const long long last = whole ? pos + rows - 1 : -1;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    // the rows a q8 workspace holds: a block of keys larger than this is split into more products
    const int work_rows = width > 0 ? (int) std::min<size_t>(work_halves / (size_t) width, (size_t) INT_MAX) : 0;
    if (kq && (kwork == nullptr || work_rows < 1))
        throw std::invalid_argument("attend_batch: a q8 key cache needs a row of its workspace");
    if (vq && (vwork == nullptr || work_rows < 1))
        throw std::invalid_argument("attend_batch: a q8 value cache needs a row of its workspace");
    // The products take halves and sum floats, on the tensor cores: the cache's rows as they lie there, the
    // queries and the softmax's weights rounded.  Two thirds of the scratch are for a block's scores, the rest
    // for their weights.
    const size_t cells = score_floats / 3 * 2;
    half* weight = (half*) (score + cells);
    for (int first = 0, c = 0; first < rows; first += c) {
        // every token left, unless a grid's 65535 rows or the scratch for one row's scores is less, or the results
        // would outgrow kAttendResults: every block of rows reads and writes them all again (og)
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head,
                            std::max<size_t>(16, kAttendResults / (size_t(n_head) * dim * sizeof(float)))});
        const long long p0 = pos + first;   // the chunk's first position, and the first row its tokens read
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;                      // the queries by group
        float* og = work + (heads * dim + 1) / 2;     // the results by group
        float* top = og + heads * dim;       // a token and head's largest score so far, the sum of its weights
        float* sum = top + heads;            // against it, and what a block of rows makes the sums before it worth
        float* shrink = sum + heads;
        attn_group_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(q + size_t(first) * n_head * dim,
                                                                                   qg, dim, n_head, kv_heads, c);
        // The rows go a block at a time, as many as have their scores in the scratch, so that a product has the
        // chunk's every token for its columns however long the context is: cuBLAS is several times slower on a
        // few columns.  A token's softmax is kept from block to block (top, sum), and its values so far shrink
        // to the new top.
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = fit > 16 ? fit & ~15 : fit;   // the tensor cores are half as fast on a ragged block
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            // the block's rows where the cache has them: a block ends where a ring starts over
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            if ((kq || vq) && m > work_rows) m = work_rows;   // the workspace's rows cap the block
            // a q8 cache's rows are dequantized into the workspace the products read as an fp16 cache
            if (kq)
                dequant_q8_kernel<<<dim3(grid(width, 256), m), 256, 0, cs(stream)>>>(kq8.codes, kq8.scales,
                                                                                     (half*) kwork, width, row);
            if (vq)
                dequant_q8_kernel<<<dim3(grid(width, 256), m), 256, 0, cs(stream)>>>(vq8.codes, vq8.scales,
                                                                                     (half*) vwork, width, row);
            const half* krow = kq ? (const half*) kwork : (const half*) kcache + size_t(row) * width;
            const half* vrow = vq ? (const half*) vwork : (const half*) vcache + size_t(row) * width;
            // A product a group: its heads' queries of every token of the chunk against the group's keys.  cuBLAS
            // reads the cache's rows as the transpose, and the groups' keys lie `dim` apart in a row.
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                               krow, CUDA_R_16F, width, dim, qg,
                                               CUDA_R_16F, dim, cols * dim, &zero, score, CUDA_R_32F, m, cols * m,
                                               kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention scores");
            attn_softmax_block_kernel<<<c * n_head, 128, 0, cs(stream)>>>(score, weight, m, lo0 + k0, (long long) pos,
                                                                         first, window, c, per, last, top, sum,
                                                                         shrink, k0 == 0);
            if (k0 > 0)
                attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                               vrow, CUDA_R_16F, width, dim,
                                               weight, CUDA_R_16F, m, cols * m, k0 > 0 ? &one : &zero, og, CUDA_R_32F,
                                               dim, cols * dim, kv_heads, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention values");
        }
        attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, sum, dim, true);
        attn_ungroup_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            og, out + size_t(first) * n_head * dim, dim, n_head, kv_heads, c);
    }
}

void likeliest(const float* logits, int n, Likeliest* out, void* stream) {
    likeliest_kernel<<<1, 1024, 0, cs(stream)>>>(logits, n, out);
}

void shortconv(const float* bcx, float* state, const float* w, float* out, int channels, int rows, void* stream,
               float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    if (rows >= CONV_ROWS && (!snap || !snaps)) {
        shortconv_rows_kernel<<<dim3(grid(channels, 256), rows), 256, 0, cs(stream)>>>(bcx, state, w, out, channels);
        shortconv_state_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(bcx, state, channels, rows);
    } else {
        shortconv_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(bcx, state, w, out, channels, rows, snap,
                                                                     stride, snaps);
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("shortconv: ") + cudaGetErrorString(e));
}

}  // namespace strata::lfm2
