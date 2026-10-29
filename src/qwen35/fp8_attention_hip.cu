// Native RDNA4 E4M3 attention for Qwen3.6, using FP8 WMMA and FP32 sums.
// The dimension-group layout follows TinyStrata's HIP INT8 attention, adapted
// from Niko1221/Strata qsa_prompt_attn.cu (MIT), revision
// 5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d. See LICENSE and NOTICE.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Q retains four and masked P*V_scale three compensated components. KV codes are
// OCP E4M3, never FNUZ; split-K publishes the existing engine merge contract.
#include "strata/qwen35/fp8_attention.hpp"
#include <hip/hip_fp16.h>
// HIP's BF16 overload declarations must precede the CUDA compatibility macros.
// This source also builds with the runtime shim force-included by CMake.
#pragma push_macro("__shfl_sync")
#pragma push_macro("__shfl_down_sync")
#pragma push_macro("__shfl_up_sync")
#pragma push_macro("__shfl_xor_sync")
#undef __shfl_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_xor_sync
#include <hip/hip_fp8.h>
#pragma pop_macro("__shfl_xor_sync")
#pragma pop_macro("__shfl_up_sync")
#pragma pop_macro("__shfl_down_sync")
#pragma pop_macro("__shfl_sync")
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::qwen35::fp8_attention {
namespace {
constexpr int HD = 256, GQA = 8, QUERIES = 2, ROWS = QUERIES * GQA;
constexpr int THREADS = 128, GROUP = 64, CHUNK = 32, SPLIT_CELLS = 256, V_STRIDE = 80;
constexpr float RESIDUAL_UP = 32.f, RESIDUAL_DOWN = 1.f / RESIDUAL_UP;
constexpr float TAIL_DOWN = RESIDUAL_DOWN * RESIDUAL_DOWN;
constexpr float Q_LAST_DOWN = TAIL_DOWN * RESIDUAL_DOWN;

#if defined(__gfx1200__) || defined(__gfx1201__)
#define TINY_QWEN_FP8_DEVICE 1
#else
#define TINY_QWEN_FP8_DEVICE 0
#endif
using ByteFragment = int __attribute__((ext_vector_type(2)));
using FloatFragment = float __attribute__((ext_vector_type(8)));

struct alignas(16) Shared {
    float dot[4][ROWS][CHUNK + 1];
    float probability[ROWS][CHUNK + 1];
    float k_scale[4][CHUNK], v_scale[4][CHUNK];
    std::uint8_t value[4][CHUNK][V_STRIDE];
    float q_max[4][ROWS];
    int q_up[ROWS];
    float q_down[ROWS];
    float alpha[ROWS], denominator[ROWS], maximum[ROWS];
};

__device__ __forceinline__ FloatFragment zero_fragment() {
    return FloatFragment{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}
__device__ __forceinline__ std::uint8_t encode(float x) {
    return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3);
}
__device__ __forceinline__ float decode(std::uint8_t x) {
    __hip_fp8_e4m3 v;
    v.__x = x;
    return float(v);
}
__device__ __forceinline__ void split3(float x, std::uint8_t& hi,
                                     std::uint8_t& lo, std::uint8_t& tail) {
#pragma clang fp contract(off)
    hi = encode(x);
    const float residual = x - decode(hi);
    lo = encode(residual * RESIDUAL_UP);
    tail = encode((residual - decode(lo) * RESIDUAL_DOWN) * (RESIDUAL_UP * RESIDUAL_UP));
}
__device__ __forceinline__ void split_q4(float x, std::uint8_t& hi,
                                       std::uint8_t& lo, std::uint8_t& tail, std::uint8_t& last) {
#pragma clang fp contract(off)
    hi = encode(x);
    const float r1 = x - decode(hi);
    lo = encode(r1 * RESIDUAL_UP);
    const float r2 = r1 - decode(lo) * RESIDUAL_DOWN;
    tail = encode(r2 * (RESIDUAL_UP * RESIDUAL_UP));
    const float r3 = r2 - decode(tail) * TAIL_DOWN;
    last = encode(r3 * (RESIDUAL_UP * RESIDUAL_UP * RESIDUAL_UP));
}
__device__ __forceinline__ void put(ByteFragment& v, int i, std::uint8_t code) {
    // Unsigned shifts preserve packed high-byte bits without signed overflow.
    v[i / 4] = int(std::uint32_t(v[i / 4]) | (std::uint32_t(code) << (8 * (i % 4))));
}
__device__ __forceinline__ FloatFragment wmma(ByteFragment a, ByteFragment b, FloatFragment c) {
#if TINY_QWEN_FP8_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    (void) a; (void) b;
    return c;
#endif
}
__device__ __forceinline__ float combine(float hi, float lo, float tail) {
    return fmaf(tail, TAIL_DOWN, fmaf(lo, RESIDUAL_DOWN, hi));
}
__device__ __forceinline__ float combine_q4(float hi, float lo, float tail, float last) {
    return fmaf(last, Q_LAST_DOWN, combine(hi, lo, tail));
}
__device__ __forceinline__ float scale_at(const std::uint16_t* scale, std::size_t row, int group) {
    return __half2float(__ushort_as_half(__ldg(scale + row * (HD / GROUP) + group)));
}
__device__ __forceinline__ ByteFragment load_key(const std::uint8_t* source, bool valid) {
    if (!valid) return ByteFragment{0, 0};
    const uint2 words = __ldg(reinterpret_cast<const uint2*>(source));
    return ByteFragment{int(words.x), int(words.y)};
}

template<bool Split>
__global__ void __launch_bounds__(THREADS) dense_fp8_wmma_kernel(
    const float* __restrict__ q, Cache cache, const std::int32_t* __restrict__ pos0,
    int queries, int kv_heads, int capacity, float scale_log2, float* __restrict__ output,
    int row_offset, int chunk_stride, float* __restrict__ partial_max, float* __restrict__ partial_sum) {
#if TINY_QWEN_FP8_DEVICE
    __shared__ Shared s;
    const int q0 = int(blockIdx.x) * QUERIES, kv_head = int(blockIdx.y);
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int col = lane & 15, half = lane >> 4, dim0 = wave * GROUP;
    const int heads = kv_heads * GQA, base = __ldg(pos0) + row_offset;
    const int last_query = min(q0 + QUERIES, queries);
    const int maximum_key = max(0, min(capacity, base + last_query));
    const int first_cell = Split ? int(blockIdx.z) * SPLIT_CELLS : 0;
    const int end_cell = Split ? min(maximum_key, first_cell + SPLIT_CELLS) : maximum_key;
    const int query = q0 + col / GQA, query_head = kv_head * GQA + col % GQA;

    float qm = 0.f;
    if (query < queries) {
#pragma unroll
        for (int kk = 0; kk < GROUP / 16; ++kk) {
            const float* src = q + ((std::size_t(query + row_offset) * heads + query_head) * HD)
                                + dim0 + kk * 16 + half * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) qm = fmaxf(qm, fabsf(src[i]));
        }
    }
    qm = fmaxf(qm, __shfl_xor(qm, 16));
    if (half == 0) s.q_max[wave][col] = qm;
    if (t < ROWS) { s.maximum[t] = -INFINITY; s.denominator[t] = 0.f; }
    __syncthreads();
    if (t < ROWS) {
        const float m = fmaxf(fmaxf(s.q_max[0][t], s.q_max[1][t]), fmaxf(s.q_max[2][t], s.q_max[3][t]));
        int exponent = 0;
        if (m > 0.f) frexpf(m, &exponent);
        s.q_up[t] = m > 0.f ? 8 - exponent : 0;
        s.q_down[t] = ldexpf(scale_log2, -s.q_up[t]);
    }
    __syncthreads();

    ByteFragment qh[GROUP / 16], ql[GROUP / 16], qt[GROUP / 16], qlast[GROUP / 16];
#pragma unroll
    for (int kk = 0; kk < GROUP / 16; ++kk) {
        qh[kk] = ql[kk] = qt[kk] = qlast[kk] = ByteFragment{0, 0};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float value = 0.f;
            if (query < queries)
                value = q[(std::size_t(query + row_offset) * heads + query_head) * HD
                          + dim0 + kk * 16 + half * 8 + i];
            std::uint8_t hi, lo, tail, last;
            split_q4(ldexpf(value, s.q_up[col]), hi, lo, tail, last);
            put(qh[kk], i, hi); put(ql[kk], i, lo); put(qt[kk], i, tail); put(qlast[kk], i, last);
        }
    }
    FloatFragment accumulator[GROUP / 16];
#pragma unroll
    for (int j = 0; j < GROUP / 16; ++j) accumulator[j] = zero_fragment();

#pragma unroll 1
    for (int c0 = first_cell; c0 < end_cell; c0 += CHUNK) {
        const int cell = c0 + lane;
        const bool valid = cell < end_cell;
        const std::size_t row = valid ? std::size_t(kv_head) * capacity + cell : 0;
        uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
        float vs = 0.f;
        // Never read the reserved-but-unmapped VMM tail, even in empty splits.
        if (valid) {
            const uint4* src = reinterpret_cast<const uint4*>(cache.v_q + row * HD + dim0);
            v0 = __ldg(src); v1 = __ldg(src + 1); v2 = __ldg(src + 2); v3 = __ldg(src + 3);
            vs = scale_at(cache.v_scale, row, wave);
        }
        uint4* dst = reinterpret_cast<uint4*>(&s.value[wave][lane][0]);
        dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
        s.v_scale[wave][lane] = vs;

#pragma unroll
        for (int nt = 0; nt < CHUNK / 16; ++nt) {
            const int key = c0 + nt * 16 + col;
            const bool key_valid = key < end_cell;
            const std::size_t key_row = key_valid ? std::size_t(kv_head) * capacity + key : 0;
            float ks = 0.f;
            if (key_valid) ks = scale_at(cache.k_scale, key_row, wave);
            if (half == 0) s.k_scale[wave][nt * 16 + col] = ks;
            FloatFragment high = zero_fragment(), low = zero_fragment(), tail = zero_fragment(), last = zero_fragment();
#pragma unroll
            for (int kk = 0; kk < GROUP / 16; ++kk) {
                const ByteFragment k = load_key(cache.k_q + key_row * HD + dim0 + kk * 16 + half * 8, key_valid);
                high = wmma(qh[kk], k, high);
                low = wmma(ql[kk], k, low);
                tail = wmma(qt[kk], k, tail);
                last = wmma(qlast[kk], k, last);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                s.dot[wave][half * 8 + i][nt * 16 + col] = combine_q4(high[i], low[i], tail[i], last[i]);
        }
        __syncthreads();

        {
            const int r = t >> 3, sub = t & 7, qi = q0 + r / GQA;
            const int upper = qi < queries ? max(0, min(capacity, base + qi + 1)) : 0;
            float x[4], tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                float dot = 0.f;
#pragma unroll
                for (int g = 0; g < HD / GROUP; ++g) dot = fmaf(s.dot[g][r][c], s.k_scale[g][c], dot);
                x[j] = c0 + c < upper && c0 + c < end_cell ? dot * s.q_down[r] : -INFINITY;
                tile_max = fmaxf(tile_max, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) tile_max = fmaxf(tile_max, __shfl_xor(tile_max, o));
            const float old_max = s.maximum[r], next_max = fmaxf(old_max, tile_max);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float probability = x[j] == -INFINITY ? 0.f : exp2f(x[j] - next_max);
                s.probability[r][sub * 4 + j] = probability;
                sum += probability;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor(sum, o);
            if (sub == 0) {
                const float alpha = old_max == -INFINITY ? 0.f : exp2f(old_max - next_max);
                s.alpha[r] = alpha;
                s.denominator[r] = fmaf(s.denominator[r], alpha, sum);
                s.maximum[r] = next_max;
            }
        }
        __syncthreads();

        float weighted_max = 0.f;
#pragma unroll
        for (int ks = 0; ks < CHUNK / 16; ++ks) {
            const int cb = ks * 16 + half * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float probability = s.probability[col][cb + i];
                const float weighted = probability == 0.f ? 0.f : probability * s.v_scale[wave][cb + i];
                weighted_max = fmaxf(weighted_max, fabsf(weighted));
            }
        }
        weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 16));
        int exponent = 0;
        if (weighted_max > 0.f) frexpf(weighted_max, &exponent);
        const int up = weighted_max > 0.f ? 8 - exponent : 0;
        ByteFragment ph[CHUNK / 16], pl[CHUNK / 16], pt[CHUNK / 16];
#pragma unroll
        for (int ks = 0; ks < CHUNK / 16; ++ks) {
            ph[ks] = pl[ks] = pt[ks] = ByteFragment{0, 0};
            const int cb = ks * 16 + half * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float probability = s.probability[col][cb + i];
                const float weighted = probability == 0.f ? 0.f : probability * s.v_scale[wave][cb + i];
                std::uint8_t hi, lo, tail;
                split3(ldexpf(weighted, up), hi, lo, tail);
                put(ph[ks], i, hi); put(pl[ks], i, lo); put(pt[ks], i, tail);
            }
        }

#pragma unroll
        for (int j = 0; j < GROUP / 16; ++j) {
            FloatFragment high = zero_fragment(), low = zero_fragment(), tail = zero_fragment();
#pragma unroll
            for (int ks = 0; ks < CHUNK / 16; ++ks) {
                const int cb = ks * 16 + half * 8;
                ByteFragment v{0, 0};
#pragma unroll
                for (int i = 0; i < 8; ++i) put(v, i, s.value[wave][cb + i][j * 16 + col]);
                high = wmma(ph[ks], v, high);
                low = wmma(pl[ks], v, low);
                tail = wmma(pt[ks], v, tail);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = half * 8 + i;
                const int row_up = __shfl(up, r);
                const float tile = ldexpf(combine(high[i], low[i], tail[i]), -row_up);
                accumulator[j][i] = fmaf(accumulator[j][i], s.alpha[r], tile);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = half * 8 + i, qi = q0 + r / GQA, head = kv_head * GQA + r % GQA;
        if (qi < queries) {
            const std::size_t out_row = std::size_t(qi) * heads + head;
            const std::size_t index = Split ? out_row * chunk_stride + blockIdx.z : out_row;
            if constexpr (Split) {
                if (wave == 0 && col == 0) {
                    partial_max[index] = s.maximum[r] * 0.6931471805599453f;
                    partial_sum[index] = s.denominator[r];
                }
            }
            const float inverse = s.denominator[r] > 0.f ? (Split ? 1.f : 1.f / s.denominator[r]) : 0.f;
#pragma unroll
            for (int j = 0; j < GROUP / 16; ++j)
                output[index * HD + dim0 + j * 16 + col] = accumulator[j][i] * inverse;
        }
    }
#else
    (void) q; (void) cache; (void) pos0; (void) queries; (void) kv_heads;
    (void) capacity; (void) scale_log2; (void) output; (void) row_offset;
    (void) chunk_stride; (void) partial_max; (void) partial_sum;
#endif
}

bool compiled_arch(const char* arch) {
#if defined(STRATA_HIP_ARCHS)
    const std::size_t size = std::strlen(arch);
    const char* item = STRATA_HIP_ARCHS;
    while (*item) {
        const char* end = std::strchr(item, ',');
        if (!end) end = item + std::strlen(item);
        if (std::size_t(end - item) == size && !std::strncmp(item, arch, size)) return true;
        item = *end ? end + 1 : end;
    }
#else
    (void) arch;
#endif
    return false;
}
bool usable_device() {
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) { (void) hipGetLastError(); return false; }
    thread_local int checked_device = -1;
    thread_local bool checked_result = false;
    if (dev == checked_device) return checked_result;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
    char arch[64]{};
    const std::size_t length = std::strcspn(prop.gcnArchName, ":");
    if (length >= sizeof(arch)) return false;
    std::memcpy(arch, prop.gcnArchName, length);
    checked_device = dev;
    checked_result = (!std::strcmp(arch, "gfx1200") || !std::strcmp(arch, "gfx1201")) &&
                     prop.warpSize == 32 && compiled_arch(arch) && sizeof(Shared) <= prop.sharedMemPerBlock;
    return checked_result;
}
} // namespace

bool available() { return usable_device(); }

void run(const float* q, Cache cache, const std::int32_t* pos, int rows, int heads,
         int context, float* output, void* stream) {
    if (rows <= 0) return;
    if (heads <= 0 || context <= 0 || !usable_device())
        throw std::runtime_error("native HIP FP8 attention requires a compiled RDNA4 gfx1200/gfx1201 image");
    dense_fp8_wmma_kernel<false><<<dim3((rows + QUERIES - 1) / QUERIES, heads), THREADS, 0, (hipStream_t) stream>>>(
        q, cache, pos, rows, heads, context, 1.4426950408889634f / 16.f, output, 0, 0, nullptr, nullptr);
}

void run_split(const float* q, Cache cache, const std::int32_t* pos, int row_offset, int rows,
               int heads, int context, int grid_chunks, int chunk_stride,
               float* partial, float* maxima, float* sums, void* stream) {
    if (rows <= 0 || grid_chunks <= 0) return;
    if (heads <= 0 || context <= 0 || row_offset < 0 || chunk_stride < grid_chunks || !usable_device())
        throw std::runtime_error("invalid native HIP FP8 split-K attention configuration");
    dense_fp8_wmma_kernel<true><<<dim3((rows + QUERIES - 1) / QUERIES, heads, grid_chunks), THREADS, 0,
                                        (hipStream_t) stream>>>(
        q, cache, pos, rows, heads, context, 1.4426950408889634f / 16.f, partial,
        row_offset, chunk_stride, maxima, sums);
}
} // namespace strata::qwen35::fp8_attention
