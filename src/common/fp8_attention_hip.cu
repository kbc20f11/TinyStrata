// Native RDNA4 E4M3 attention for token-major Gemma / LFM KV caches.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// The wave32 fragment mapping follows TinyStrata's HIP INT8 attention, adapted
// from Strata's MIT WMMA implementation. QK and PV use FP8 WMMA, four-component
// Q / three-component P compensation, and FP32 online-softmax outer accumulation.
#include "strata/common/fp8_attention.hpp"
#include <hip/hip_fp16.h>
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::fp8_attention {
namespace {
constexpr int CELLS = 32, SPLITS = 32, TARGET_CTAS = 64;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float RESIDUAL_UP = 32.f, RESIDUAL_DOWN = 1.f / 32.f;
constexpr float TAIL_DOWN = 1.f / 1024.f;
constexpr float Q_FINAL_DOWN = 1.f / 32768.f;
#if defined(__gfx1200__) || defined(__gfx1201__)
#define TINY_FP8_WMMA_DEVICE 1
#else
#define TINY_FP8_WMMA_DEVICE 0
#endif
using Packed = int __attribute__((ext_vector_type(2)));
using Fragment = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ Fragment zero_fragment() {
    return Fragment{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}
__device__ __forceinline__ std::uint8_t encode(float x) {
    return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3);
}
__device__ __forceinline__ float decode(std::uint8_t x) {
    __hip_fp8_e4m3 value;
    value.__x = x;
    return static_cast<float>(value);
}
__device__ __forceinline__ void encode3(float x, std::uint8_t& high,
                                       std::uint8_t& low, std::uint8_t& tail) {
#pragma clang fp contract(off)
    high = encode(x);
    const float residual = x - decode(high);
    low = encode(residual * RESIDUAL_UP);
    tail = encode((residual - decode(low) * RESIDUAL_DOWN) * 1024.f);
}
__device__ __forceinline__ void encode_q4(float x, std::uint8_t& high,
                                         std::uint8_t& low, std::uint8_t& tail,
                                         std::uint8_t& final) {
#pragma clang fp contract(off)
    high = encode(x);
    const float residual = x - decode(high);
    low = encode(residual * RESIDUAL_UP);
    const float remainder = residual - decode(low) * RESIDUAL_DOWN;
    tail = encode(remainder * 1024.f);
    final = encode((remainder - decode(tail) * TAIL_DOWN) * 32768.f);
}
__device__ __forceinline__ Fragment mma(Packed a, Packed b, Fragment c) {
#if TINY_FP8_WMMA_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    // Non-RDNA4 code images remain legal. available() rejects them before use.
    (void) a; (void) b;
    return c;
#endif
}
__device__ __forceinline__ void pack_byte(Packed& p, int i, std::uint8_t x) {
    p[i >> 2] |= static_cast<int>(std::uint32_t(x) << ((i & 3) * 8));
}
__device__ __forceinline__ Packed load_codes(const std::uint8_t* p, bool valid) {
    if (!valid) return Packed{0, 0};
    const uint2 x = __ldg(reinterpret_cast<const uint2*>(p));
    return Packed{static_cast<int>(x.x), static_cast<int>(x.y)};
}
__device__ __forceinline__ float multiplier(const std::uint16_t* p) {
    return __half2float(__ushort_as_half(__ldg(p)));
}

__global__ void store_kernel(const float* __restrict__ src, std::uint8_t* __restrict__ codes,
                             std::uint16_t* __restrict__ scales, int width,
                             std::int64_t pos, const std::int64_t* at, int ring) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (group >= width / 32) return;
    std::int64_t row = (at ? at[0] : pos) + blockIdx.y;
    if (ring) row %= ring;
    const int d = group * 32 + lane;
    const float x = src[std::size_t(blockIdx.y) * width + d];
    float m = fabsf(x);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
    // Divide by the rounded multiplier which is actually kept in the cache.
    const __half h = __float2half_rn(fminf(65504.f, fmaxf(m / 448.f, 0x1p-24f)));
    codes[std::size_t(row) * width + d] = encode(x / __half2float(h));
    if (!lane) scales[std::size_t(row) * (width / 32) + group] = __half_as_ushort(h);
}

template<int HD> struct Shared {
    static constexpr int WAVES = HD == 64 ? 2 : 4;
    // Each wave accumulates complete per-32 K scale groups. Only its final
    // scaled contribution is shared, keeping every group's scale distinct.
    float dot[WAVES][16][CELLS + 1];
    float probability[16][CELLS + 1];
    alignas(16) std::uint8_t value[CELLS][HD + 16];
    std::uint16_t vs[CELLS][HD / 32];
    float qmax[WAVES][16], qdown[16];
    int qup[16];
    float maximum[16], denominator[16], alpha[16];
};

__device__ __forceinline__ int split_count(int span, int kv_heads) {
    const int target = min(SPLITS, (TARGET_CTAS + kv_heads - 1) / kv_heads);
    return min(target, max(1, (span + CELLS - 1) / CELLS));
}
__device__ __forceinline__ std::int64_t lower_bound(std::int64_t position, int window) {
    return window > 0 && position >= window ? position - window + 1 : 0;
}

template<int HD, bool SPLIT>
__global__ void __launch_bounds__(Shared<HD>::WAVES * 32)
attention_kernel(const float* __restrict__ q, Cache cache, std::int64_t pos,
                 const std::int64_t* at, int rows, int heads, int kv_heads,
                 int gqa, int ring, int window, bool whole, float scale_log2,
                 float* __restrict__ out, float* __restrict__ partial) {
#if TINY_FP8_WMMA_DEVICE
    constexpr int WAVES = Shared<HD>::WAVES, THREADS = WAVES * 32;
    constexpr int LOCAL = HD / WAVES, ROW_THREADS = THREADS / 16;
    __shared__ Shared<HD> s;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4, dim0 = wave * LOCAL;
    const int tile_rows = 16 / gqa, q0 = int(blockIdx.x) * tile_rows, kvh = blockIdx.y;
    const int split = SPLIT ? int(blockIdx.z) : 0;
    const std::int64_t base = at ? at[0] : pos;
    const std::int64_t span_lo = lower_bound(base + q0, window);
    const std::int64_t span_hi = base + (whole ? rows : min(rows, q0 + tile_rows));
    const int span = int(span_hi - span_lo);
    const int active = SPLIT ? split_count(span, kv_heads) : 1;
    if (split >= active) return;
    const int chunks = (span + CELLS - 1) / CELLS;
    const std::int64_t begin = span_lo + (SPLIT ? (split * chunks / active) * CELLS : 0);
    const std::int64_t end = SPLIT ? min(span_hi, span_lo + ((split + 1) * chunks / active) * CELLS) : span_hi;
    const int qi = q0 + col / gqa, head = kvh * gqa + col % gqa;

    // Normalize each query/head independently below 2^8, then keep high,
    // residual x32, tail x1024, and final x32768 in independent MMA chains.
    // The final component retains small score differences when large Q/K
    // terms cancel; PV retains its three compensated components.
    float qm = 0.f;
    if (qi < rows) {
#pragma unroll
        for (int k = 0; k < LOCAL / 16; ++k) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                qm = fmaxf(qm, fabsf(q[(std::size_t(qi) * heads + head) * HD + dim0 + k * 16 + half * 8 + i]));
        }
    }
    qm = fmaxf(qm, __shfl_xor(qm, 16));
    if (!half) s.qmax[wave][col] = qm;
    if (tid < 16) {
        s.maximum[tid] = -INFINITY;
        s.denominator[tid] = 0.f;
    }
    __syncthreads();
    if (tid < 16) {
        float maximum = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) maximum = fmaxf(maximum, s.qmax[w][tid]);
        int exponent = 0;
        if (maximum > 0.f) frexpf(maximum, &exponent);
        s.qup[tid] = maximum > 0.f ? 8 - exponent : 0;
        s.qdown[tid] = ldexpf(scale_log2, -s.qup[tid]);
    }
    __syncthreads();
    Packed qh[LOCAL / 16], ql[LOCAL / 16], qt[LOCAL / 16], qf[LOCAL / 16];
#pragma unroll
    for (int k = 0; k < LOCAL / 16; ++k) {
        qh[k] = ql[k] = qt[k] = qf[k] = Packed{0, 0};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float x = qi < rows
                ? q[(std::size_t(qi) * heads + head) * HD + dim0 + k * 16 + half * 8 + i] : 0.f;
            std::uint8_t h, l, t, f;
            encode_q4(ldexpf(x, s.qup[col]), h, l, t, f);
            pack_byte(qh[k], i, h); pack_byte(ql[k], i, l); pack_byte(qt[k], i, t);
            pack_byte(qf[k], i, f);
        }
    }
    Fragment accumulator[LOCAL / 16];
#pragma unroll
    for (int j = 0; j < LOCAL / 16; ++j) accumulator[j] = zero_fragment();

    for (std::int64_t c0 = begin; c0 < end; c0 += CELLS) {
        // Predicate before every KV load. Reserved but unmapped VMM pages past
        // the current logical history are never touched, even inside a graph.
        for (int index = tid; index < CELLS * (HD / 16); index += THREADS) {
            const int cell = index / (HD / 16), piece = index % (HD / 16);
            const std::int64_t absolute = c0 + cell;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (absolute < end) {
                const std::size_t row = std::size_t(ring ? absolute % ring : absolute) * kv_heads + kvh;
                x = __ldg(reinterpret_cast<const uint4*>(cache.v + row * HD + piece * 16));
            }
            *reinterpret_cast<uint4*>(s.value[cell] + piece * 16) = x;
        }
        for (int index = tid; index < CELLS * (HD / 32); index += THREADS) {
            const int cell = index / (HD / 32), group = index % (HD / 32);
            const std::int64_t absolute = c0 + cell;
            std::uint16_t value = 0;
            if (absolute < end) {
                const std::size_t row = std::size_t(ring ? absolute % ring : absolute) * kv_heads + kvh;
                value = __ldg(cache.vs + row * (HD / 32) + group);
            }
            s.vs[cell][group] = value;
        }
#pragma unroll
        for (int nt = 0; nt < CELLS / 16; ++nt) {
            const std::int64_t absolute = c0 + nt * 16 + col;
            const bool valid = absolute < end;
            const std::size_t row = valid ? std::size_t(ring ? absolute % ring : absolute) * kv_heads + kvh : 0;
            Fragment dot = zero_fragment();
#pragma unroll
            for (int g = 0; g < LOCAL / 32; ++g) {
                Fragment high = zero_fragment(), low = zero_fragment(), tail = zero_fragment(), final = zero_fragment();
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int dimension = dim0 + g * 32 + k * 16 + half * 8;
                    const Packed key = load_codes(cache.k + row * HD + dimension, valid);
                    high = mma(qh[g * 2 + k], key, high);
                    low = mma(ql[g * 2 + k], key, low);
                    tail = mma(qt[g * 2 + k], key, tail);
                    final = mma(qf[g * 2 + k], key, final);
                }
                const float ks = valid ? multiplier(cache.ks + row * (HD / 32) + dim0 / 32 + g) : 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    dot[i] = fmaf(fmaf(final[i], Q_FINAL_DOWN,
                                      fmaf(tail[i], TAIL_DOWN, fmaf(low[i], RESIDUAL_DOWN, high[i]))), ks, dot[i]);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                s.dot[wave][half * 8 + i][nt * 16 + col] = dot[i];
        }
        __syncthreads();

        // The row's adjacent lanes reduce softmax together; never reduce over
        // query/head boundaries. Whole image blocks only change each upper bound.
        const int r = tid / ROW_THREADS, sub = tid % ROW_THREADS, query = q0 + r / gqa;
        const std::int64_t lo = lower_bound(base + query, window);
        const std::int64_t hi = base + (whole ? rows : query + 1);
        float scores[CELLS / ROW_THREADS], maximum = -INFINITY;
#pragma unroll
        for (int j = 0; j < CELLS / ROW_THREADS; ++j) {
            const int c = sub + j * ROW_THREADS;
            float dot = 0.f;
#pragma unroll
            for (int w = 0; w < WAVES; ++w) dot += s.dot[w][r][c];
            const std::int64_t cell = c0 + c;
            scores[j] = query < rows && cell < end && cell >= lo && cell < hi ? dot * s.qdown[r] : -INFINITY;
            maximum = fmaxf(maximum, scores[j]);
        }
#pragma unroll
        for (int o = 1; o < ROW_THREADS; o <<= 1) maximum = fmaxf(maximum, __shfl_xor(maximum, o));
        const float old_max = s.maximum[r], next_max = fmaxf(old_max, maximum);
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < CELLS / ROW_THREADS; ++j) {
            const float p = scores[j] == -INFINITY ? 0.f : exp2f(scores[j] - next_max);
            s.probability[r][sub + j * ROW_THREADS] = p;
            sum += p;
        }
#pragma unroll
        for (int o = 1; o < ROW_THREADS; o <<= 1) sum += __shfl_xor(sum, o);
        if (!sub) {
            const float alpha = old_max == -INFINITY ? 0.f : exp2f(old_max - next_max);
            s.alpha[r] = alpha;
            s.denominator[r] = fmaf(s.denominator[r], alpha, sum);
            s.maximum[r] = next_max;
        }
        __syncthreads();

        // Per-32 V scales belong inside P's MMA operand. Scale only a row's
        // actual masked P*V_scale values before encoding the three components.
#pragma unroll
        for (int g = 0; g < LOCAL / 32; ++g) {
            float maximum = 0.f;
#pragma unroll
            for (int ks = 0; ks < CELLS / 16; ++ks) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cell = ks * 16 + half * 8 + i;
                    const float p = s.probability[col][cell];
                    const float weighted = p == 0.f ? 0.f : p * __half2float(__ushort_as_half(s.vs[cell][dim0 / 32 + g]));
                    maximum = fmaxf(maximum, fabsf(weighted));
                }
            }
            maximum = fmaxf(maximum, __shfl_xor(maximum, 16));
            int exponent = 0;
            if (maximum > 0.f) frexpf(maximum, &exponent);
            const int up = maximum > 0.f ? 8 - exponent : 0;
            Packed ph[CELLS / 16], pl[CELLS / 16], pt[CELLS / 16];
#pragma unroll
            for (int ks = 0; ks < CELLS / 16; ++ks) {
                ph[ks] = pl[ks] = pt[ks] = Packed{0, 0};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cell = ks * 16 + half * 8 + i;
                    const float p = s.probability[col][cell];
                    const float weighted = p == 0.f ? 0.f : p * __half2float(__ushort_as_half(s.vs[cell][dim0 / 32 + g]));
                    std::uint8_t h, l, t;
                    encode3(ldexpf(weighted, up), h, l, t);
                    pack_byte(ph[ks], i, h); pack_byte(pl[ks], i, l); pack_byte(pt[ks], i, t);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                Fragment high = zero_fragment(), low = zero_fragment(), tail = zero_fragment();
#pragma unroll
                for (int ks = 0; ks < CELLS / 16; ++ks) {
                    Packed value{0, 0};
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        pack_byte(value, i, s.value[ks * 16 + half * 8 + i][dim0 + g * 32 + j * 16 + col]);
                    high = mma(ph[ks], value, high);
                    low = mma(pl[ks], value, low);
                    tail = mma(pt[ks], value, tail);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = half * 8 + i;
                    const int row_up = __shfl(up, row);
                    const float tile = ldexpf(fmaf(tail[i], TAIL_DOWN, fmaf(low[i], RESIDUAL_DOWN, high[i])), -row_up);
                    accumulator[g * 2 + j][i] = fmaf(accumulator[g * 2 + j][i], s.alpha[row], tile);
                }
            }
        }
        // The cooperative V loader in the next tile can write another wave's
        // output columns, so all waves must finish consuming this tile first.
        __syncthreads();
    }

    if constexpr (SPLIT) {
        if (tid < 16) {
            const int query = q0 + tid / gqa, h = kvh * gqa + tid % gqa;
            if (query < rows) {
                float* p = partial + ((std::size_t(query) * heads + h) * SPLITS + split) * (HD + 2);
                p[HD] = s.maximum[tid]; p[HD + 1] = s.denominator[tid];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < LOCAL / 16; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = half * 8 + i, query = q0 + row / gqa, h = kvh * gqa + row % gqa;
            if (query < rows) {
                const std::size_t output_row = std::size_t(query) * heads + h;
                const float inverse = SPLIT ? 1.f : (s.denominator[row] > 0.f ? 1.f / s.denominator[row] : 0.f);
                float* destination = SPLIT ? partial + (output_row * SPLITS + split) * (HD + 2) : out + output_row * HD;
                destination[dim0 + j * 16 + col] = accumulator[j][i] * inverse;
            }
        }
    }
#else
    (void) q; (void) cache; (void) pos; (void) at; (void) rows; (void) heads; (void) kv_heads;
    (void) gqa; (void) ring; (void) window; (void) whole; (void) scale_log2; (void) out; (void) partial;
#endif
}

template<int HD> __global__ void merge_kernel(const float* __restrict__ partial,
                                             float* __restrict__ out, std::int64_t pos,
                                             const std::int64_t* at, int rows, int heads,
                                             int kv_heads, int gqa, int window, bool whole) {
    const int lane = threadIdx.x & 31, query = blockIdx.x / heads;
    const int tile_rows = 16 / gqa, q0 = (query / tile_rows) * tile_rows;
    const std::int64_t base = at ? at[0] : pos;
    const std::int64_t lo = lower_bound(base + q0, window);
    const int span = int(base + (whole ? rows : min(rows, q0 + tile_rows)) - lo);
    const int active = split_count(span, kv_heads);
    const float* p = partial + std::size_t(blockIdx.x) * SPLITS * (HD + 2);
    float maximum = lane < active ? p[lane * (HD + 2) + HD] : -INFINITY;
#pragma unroll
    for (int o = 16; o; o >>= 1) maximum = fmaxf(maximum, __shfl_xor(maximum, o));
    const float l = lane < active ? p[lane * (HD + 2) + HD + 1] : 0.f;
    const float factor = l > 0.f ? exp2f(p[lane * (HD + 2) + HD] - maximum) : 0.f;
    float denominator = factor * l;
#pragma unroll
    for (int o = 16; o; o >>= 1) denominator += __shfl_xor(denominator, o);
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float sum = 0.f;
#pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float f = __shfl(factor, s);
            if (s < active) sum = fmaf(p[s * (HD + 2) + d], f, sum);
        }
        out[std::size_t(blockIdx.x) * HD + d] = denominator > 0.f ? sum / denominator : 0.f;
    }
}

bool compiled_arch(const char* arch) {
#if defined(STRATA_HIP_ARCHS)
    const std::size_t length = std::strlen(arch);
    const char* item = STRATA_HIP_ARCHS;
    while (*item) {
        const char* end = std::strchr(item, ',');
        if (!end) end = item + std::strlen(item);
        if (std::size_t(end - item) == length && !std::strncmp(item, arch, length)) return true;
        item = *end ? end + 1 : end;
    }
#else
    (void) arch;
#endif
    return false;
}
template<int HD, bool SPLIT> bool kernel_available() {
    hipFuncAttributes a{};
    return hipFuncGetAttributes(&a, reinterpret_cast<const void*>(attention_kernel<HD, SPLIT>)) == hipSuccess;
}
bool usable_device() {
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) { (void) hipGetLastError(); return false; }
    thread_local int checked_device = -1;
    thread_local bool checked = false;
    if (checked_device == dev) return checked;
    checked_device = dev;
    checked = false;
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
    char arch[64]{};
    const std::size_t length = std::strcspn(p.gcnArchName, ":");
    if (length >= sizeof(arch)) return false;
    std::memcpy(arch, p.gcnArchName, length);
    if ((std::strcmp(arch, "gfx1200") && std::strcmp(arch, "gfx1201")) || p.warpSize != 32 ||
        !compiled_arch(arch) || sizeof(Shared<512>) > p.sharedMemPerBlock) return false;
    checked = kernel_available<64, false>() && kernel_available<64, true>() &&
              kernel_available<256, false>() && kernel_available<256, true>() &&
              kernel_available<512, false>() && kernel_available<512, true>();
    if (!checked) (void) hipGetLastError();
    return checked;
}
template<int HD>
void launch(const float* q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads,
            float scale, float* out, const std::int64_t* at, int ring, int window,
            bool whole, float* scratch, bool split, hipStream_t stream) {
    const int gqa = heads / kv_heads, tile_rows = 16 / gqa;
    const dim3 grid((rows + tile_rows - 1) / tile_rows, kv_heads, split ? SPLITS : 1);
    if (split) {
        attention_kernel<HD, true><<<grid, Shared<HD>::WAVES * 32, 0, stream>>>(
            q, cache, pos, at, rows, heads, kv_heads, gqa, ring, window, whole, scale * LOG2E, out, scratch);
        merge_kernel<HD><<<rows * heads, 128, 0, stream>>>(
            scratch, out, pos, at, rows, heads, kv_heads, gqa, window, whole);
    } else {
        attention_kernel<HD, false><<<grid, Shared<HD>::WAVES * 32, 0, stream>>>(
            q, cache, pos, at, rows, heads, kv_heads, gqa, ring, window, whole, scale * LOG2E, out, nullptr);
    }
}
} // namespace

bool available(int head_dim) {
    return (!head_dim || head_dim == 64 || head_dim == 256 || head_dim == 512) && usable_device();
}
bool available_gemma() { return usable_device(); }
void store(const float* src, std::uint8_t* codes, std::uint16_t* scales, int width,
           std::int64_t pos, int rows, const std::int64_t* at, int ring, void* stream) {
    if (width < 32 || width % 32 || rows < 0 || ring < 0) throw std::invalid_argument("FP8 KV store: invalid shape");
    if (!rows) return;
    store_kernel<<<dim3((width / 32 + 3) / 4, rows), 128, 0, static_cast<hipStream_t>(stream)>>>(
        src, codes, scales, width, pos, at, ring);
}
std::size_t scratch_bytes(int rows, int q_heads, int head_dim) {
    if (rows <= 0 || q_heads <= 0 || head_dim <= 0) return 0;
    return std::size_t(std::min(rows, 8)) * q_heads * (head_dim + 2) * SPLITS * sizeof(float);
}
void run(const float* q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads,
         int hd, int context, float scale, float* out, const std::int64_t* at,
         int ring, int window, bool whole, void* scratch, std::size_t size, void* stream) {
    // available() is a load-time precondition. No probing/allocation occurs
    // during graph capture or on the separate hybrid inference worker.
    if (rows < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || context <= 0 || ring < 0 || window < 0)
        throw std::invalid_argument("HIP native FP8 attention: invalid shape");
    const int gqa = heads / kv_heads;
    if (gqa > 16 || 16 % gqa || (hd != 64 && hd != 256 && hd != 512))
        throw std::invalid_argument("HIP native FP8 attention: unsupported head dimension or GQA ratio");
    if (!rows) return;
    const bool split = rows <= 8 && scratch && size >= scratch_bytes(rows, heads, hd);
    const auto call = [&]<int HD>() {
        launch<HD>(q, cache, pos, rows, heads, kv_heads, scale, out, at, ring, window, whole,
                   static_cast<float*>(scratch), split, static_cast<hipStream_t>(stream));
    };
    if (hd == 64) call.template operator()<64>();
    else if (hd == 256) call.template operator()<256>();
    else call.template operator()<512>();
}
void run_gemma(const float* q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads,
               int hd, int context, float scale, float* out, const std::int64_t* at,
               int ring, int window, bool whole, void* scratch, std::size_t size, void* stream) {
    if (hd != 256 && hd != 512)
        throw std::invalid_argument("Gemma HIP FP8 attention: unsupported head dimension");
    run(q, cache, pos, rows, heads, kv_heads, hd, context, scale, out, at, ring, window,
        whole, scratch, size, stream);
}
} // namespace strata::fp8_attention
