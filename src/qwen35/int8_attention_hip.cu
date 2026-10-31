// Dense resident INT8 prompt attention for Qwen3.6 on HIP wave32 GPUs.
// WMMA fragment mapping and the dimension-group tile design are adapted from
// Niko1221/Strata src/kernels/cuda/qsa_prompt_attn.cu (MIT), revision
// 5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d. See LICENSE and NOTICE.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Tiny adaptations: two causal queries/GQA8 per block, dense head-major KV,
// row-wise power-of-two scaling, amplified FP16 residuals with independent
// high/low MMA chains, and unscaled FP32 online-softmax outer accumulation.
#include "strata/qwen35/int8_attention.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::qwen35::int8_attention {
namespace {
constexpr int HD = 256, GQA = 8, QUERIES = 2, ROWS = QUERIES * GQA;
constexpr int THREADS = 128, GROUP = 64, CHUNK = 32, V_STRIDE = 80;
constexpr float RESIDUAL_UP = 2048.0f, RESIDUAL_DOWN = 1.0f / RESIDUAL_UP;

#if defined(__gfx1200__) || defined(__gfx1201__)
#define TINY_WMMA_DEVICE 1
#define TINY_WMMA_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define TINY_WMMA_DEVICE 1
#define TINY_WMMA_GFX11 1
#else
#define TINY_WMMA_DEVICE 0
#define TINY_WMMA_GFX11 0
#endif

#if TINY_WMMA_GFX11
constexpr int KL = 16;
__device__ __forceinline__ int k_offset(int) { return 0; }
__device__ __forceinline__ int mma_row(int i, int half) { return 2 * i + half; }
#else
constexpr int KL = 8;
__device__ __forceinline__ int k_offset(int half) { return half * 8; }
__device__ __forceinline__ int mma_row(int i, int half) { return half * 8 + i; }
#endif
using HalfFragment = _Float16 __attribute__((ext_vector_type(KL)));
using FloatFragment = float __attribute__((ext_vector_type(8)));

struct alignas(16) Shared {
    float dot[4][ROWS][CHUNK + 1];
    float probability[ROWS][CHUNK + 1];
    float k_scale[4][CHUNK];
    float v_scale[4][CHUNK];
    int8_t value[4][CHUNK][V_STRIDE];
    float q_max[4][ROWS];
    int q_up[ROWS];
    float q_down[ROWS];
    float alpha[ROWS], denominator[ROWS], maximum[ROWS];
};

__device__ __forceinline__ FloatFragment zero_fragment() {
    return FloatFragment{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__device__ __forceinline__ FloatFragment wmma(HalfFragment a, HalfFragment b, FloatFragment c) {
#if TINY_WMMA_DEVICE && TINY_WMMA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif TINY_WMMA_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    // The host rejects this target. Keep a legal, non-trapping device body for
    // portable/fat builds which also contain targets without matrix cores.
    (void) a;
    (void) b;
    return c;
#endif
}

__device__ __forceinline__ void split(float value, _Float16& high, _Float16& low) {
#pragma clang fp contract(off)
    high = (_Float16) value;
    low = (_Float16) ((value - (float) high) * RESIDUAL_UP);
}

__device__ __forceinline__ float scale_at(const uint16_t* scale, size_t row, int group) {
    return __half2float(__ushort_as_half(__ldg(scale + row * (HD / GROUP) + group)));
}

__device__ __forceinline__ HalfFragment load_key_fragment(const int8_t* source, bool valid) {
    uint32_t words[KL / 4] = {};
    if (valid) {
#if TINY_WMMA_GFX11
        const uint4 packed = __ldg(reinterpret_cast<const uint4*>(source));
        words[0] = packed.x; words[1] = packed.y; words[2] = packed.z; words[3] = packed.w;
#else
        const uint2 packed = __ldg(reinterpret_cast<const uint2*>(source));
        words[0] = packed.x; words[1] = packed.y;
#endif
    }
    HalfFragment result;
#pragma unroll
    for (int i = 0; i < KL; ++i)
        result[i] = (_Float16) (int) (int8_t) (words[i / 4] >> (8 * (i % 4)));
    return result;
}

__global__ void __launch_bounds__(THREADS) dense_int8_wmma_kernel(
    const float* __restrict__ q, Cache cache, const int32_t* __restrict__ pos0,
    int queries, int kv_heads, int capacity, float scale_log2, float* __restrict__ output, int visible_end) {
#if TINY_WMMA_DEVICE
    __shared__ Shared s;
    const int q0 = (int) blockIdx.x * QUERIES, kv_head = (int) blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int col = lane & 15, half = lane >> 4, dim0 = wave * GROUP;
    const int heads = kv_heads * GQA, base = __ldg(pos0);
    const int last_query = min(q0 + QUERIES, queries);
    const int maximum_key = max(0, min(capacity, visible_end > 0 ? visible_end : base + last_query));
    const int query = q0 + col / GQA, query_head = kv_head * GQA + col % GQA;

    // One row is one (query, head). The two half-waves own the two halves
    // of its 64-dimension scale group; the four waves cover all 256 dims.
    float qm = 0.0f;
    if (query < queries) {
#pragma unroll
        for (int kk = 0; kk < GROUP / 16; ++kk) {
            const float* src = q + ((size_t) query * heads + query_head) * HD + dim0 + kk * 16 + k_offset(half);
#pragma unroll
            for (int i = 0; i < KL; ++i) qm = fmaxf(qm, fabsf(src[i]));
        }
    }
    qm = fmaxf(qm, __shfl_xor(qm, 16));
    if (half == 0) s.q_max[wave][col] = qm;
    if (t < ROWS) {
        s.maximum[t] = -INFINITY;
        s.denominator[t] = 0.0f;
    }
    __syncthreads();
    if (t < ROWS) {
        float m = fmaxf(fmaxf(s.q_max[0][t], s.q_max[1][t]), fmaxf(s.q_max[2][t], s.q_max[3][t]));
        int exponent = 0;
        if (m > 0.0f) frexpf(m, &exponent);
        s.q_up[t] = 14 - exponent;
        s.q_down[t] = ldexpf(scale_log2, exponent - 14);
    }
    __syncthreads();

    HalfFragment qh[GROUP / 16], ql[GROUP / 16];
#pragma unroll
    for (int kk = 0; kk < GROUP / 16; ++kk) {
#pragma unroll
        for (int i = 0; i < KL; ++i) {
            float value = 0.0f;
            if (query < queries)
                value = q[((size_t) query * heads + query_head) * HD + dim0 + kk * 16 + k_offset(half) + i];
            _Float16 high, low;
            split(ldexpf(value, s.q_up[col]), high, low);
            qh[kk][i] = high;
            ql[kk][i] = low;
        }
    }

    // The online numerator always stays in real V units. Normalization
    // exponents are tile-local and removed before updating these sums.
    FloatFragment accumulator[GROUP / 16];
#pragma unroll
    for (int j = 0; j < GROUP / 16; ++j) accumulator[j] = zero_fragment();

#pragma unroll 1
    for (int c0 = 0; c0 < maximum_key; c0 += CHUNK) {
        const int cell = c0 + lane;
        const bool valid = cell < maximum_key;
        const size_t row = valid ? (size_t) kv_head * capacity + cell : 0;
        uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
        float vs = 0.0f;
        // Predicate before every global KV load: capacity may include
        // reserved virtual pages which have never been physically mapped.
        if (valid) {
            const uint4* src = reinterpret_cast<const uint4*>(cache.v_q + row * HD + dim0);
            v0 = __ldg(src); v1 = __ldg(src + 1); v2 = __ldg(src + 2); v3 = __ldg(src + 3);
            vs = scale_at(cache.v_scale, row, wave);
        }
        uint4* dst = reinterpret_cast<uint4*>(&s.value[wave][lane][0]);
        dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
        s.v_scale[wave][lane] = vs;

        // Independent high/residual chains across each complete 64-dim
        // scale group; combine only after its four K=16 steps are done.
#pragma unroll
        for (int nt = 0; nt < CHUNK / 16; ++nt) {
            const int key = c0 + nt * 16 + col;
            const bool key_valid = key < maximum_key;
            const size_t key_row = key_valid ? (size_t) kv_head * capacity + key : 0;
            float ks = 0.0f;
            if (key_valid) ks = scale_at(cache.k_scale, key_row, wave);
            if (half == 0) s.k_scale[wave][nt * 16 + col] = ks;
            FloatFragment high = zero_fragment(), low = zero_fragment();
#pragma unroll
            for (int kk = 0; kk < GROUP / 16; ++kk) {
                // Packed global reads, exact INT8 -> FP16 conversion. The
                // validity predicate stays inside the helper, before load.
                const HalfFragment k = load_key_fragment(
                    cache.k_q + key_row * HD + dim0 + kk * 16 + k_offset(half), key_valid);
                high = wmma(qh[kk], k, high);
                low = wmma(ql[kk], k, low);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                s.dot[wave][mma_row(i, half)][nt * 16 + col] = fmaf(low[i], RESIDUAL_DOWN, high[i]);
        }
        __syncthreads();

        // Four adjacent keys per thread, eight threads per query/head row.
        // Causal rows have distinct upper bounds. Clover preview explicitly
        // gives every active row the same complete noisy-block upper bound.
        {
            const int r = t >> 3, sub = t & 7, qi = q0 + r / GQA;
            const int upper = qi < queries ? max(0, min(capacity, visible_end > 0 ? visible_end : base + qi + 1)) : 0;
            float x[4], tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                float dot = 0.0f;
#pragma unroll
                for (int g = 0; g < HD / GROUP; ++g) dot = fmaf(s.dot[g][r][c], s.k_scale[g][c], dot);
                x[j] = c0 + c < upper ? dot * s.q_down[r] : -INFINITY;
                tile_max = fmaxf(tile_max, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) tile_max = fmaxf(tile_max, __shfl_xor(tile_max, o));
            const float old_max = s.maximum[r], next_max = fmaxf(old_max, tile_max);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float probability = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - next_max);
                s.probability[r][sub * 4 + j] = probability;
                sum += probability;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor(sum, o);
            if (sub == 0) {
                const float alpha = old_max == -INFINITY ? 0.0f : exp2f(old_max - next_max);
                s.alpha[r] = alpha;
                s.denominator[r] = fmaf(s.denominator[r], alpha, sum);
                s.maximum[r] = next_max;
            }
        }
        __syncthreads();

        // Normalize the actual masked P*V_scale per query/head AND group.
        // In particular a masked scale cannot affect the maximum or yield
        // 0*Inf. Never reduce maxima across different query/head rows.
        float weighted_max = 0.0f;
#pragma unroll
        for (int ks = 0; ks < CHUNK / 16; ++ks) {
            const int cb = ks * 16 + k_offset(half);
#pragma unroll
            for (int i = 0; i < KL; ++i) {
                const float probability = s.probability[col][cb + i];
                const float weighted = probability == 0.0f ? 0.0f : probability * s.v_scale[wave][cb + i];
                weighted_max = fmaxf(weighted_max, fabsf(weighted));
            }
        }
        weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 16));
        int weighted_exponent = 0;
        if (weighted_max > 0.0f) frexpf(weighted_max, &weighted_exponent);
        const int up = weighted_max > 0.0f ? 14 - weighted_exponent : 0;
        HalfFragment ph[CHUNK / 16], pl[CHUNK / 16];
#pragma unroll
        for (int ks = 0; ks < CHUNK / 16; ++ks) {
            const int cb = ks * 16 + k_offset(half);
#pragma unroll
            for (int i = 0; i < KL; ++i) {
                const float probability = s.probability[col][cb + i];
                const float weighted = probability == 0.0f ? 0.0f : probability * s.v_scale[wave][cb + i];
                _Float16 high, low;
                split(ldexpf(weighted, up), high, low);
                ph[ks][i] = high;
                pl[ks][i] = low;
            }
        }

        // Only one 16-output-column high/low pair stays live at a time.
#pragma unroll
        for (int j = 0; j < GROUP / 16; ++j) {
            FloatFragment high = zero_fragment(), low = zero_fragment();
#pragma unroll
            for (int ks = 0; ks < CHUNK / 16; ++ks) {
                const int cb = ks * 16 + k_offset(half);
                HalfFragment v;
#pragma unroll
                for (int i = 0; i < KL; ++i) v[i] = (_Float16) (int) s.value[wave][cb + i][j * 16 + col];
                high = wmma(ph[ks], v, high);
                low = wmma(pl[ks], v, low);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = mma_row(i, half);
                const int row_up = __shfl(up, r);
                const float tile = ldexpf(fmaf(low[i], RESIDUAL_DOWN, high[i]), -row_up);
                accumulator[j][i] = fmaf(accumulator[j][i], s.alpha[r], tile);
            }
        }
        // V/scale are wave-private. The next tile's first block barrier
        // precedes overwriting shared probabilities and online row state.
    }

    FloatFragment inverse;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float denominator = s.denominator[mma_row(i, half)];
        inverse[i] = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < GROUP / 16; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = mma_row(i, half), qi = q0 + r / GQA, head = kv_head * GQA + r % GQA;
            if (qi < queries) {
                output[((size_t) qi * heads + head) * HD + dim0 + j * 16 + col] = accumulator[j][i] * inverse[i];
            }
        }
    }
#else
    (void) q; (void) cache; (void) pos0; (void) queries; (void) kv_heads;
    (void) capacity; (void) scale_log2; (void) output; (void) visible_end;
#endif
}

bool compiled_arch(const char* arch) {
#if defined(STRATA_HIP_ARCHS)
    const size_t size = std::strlen(arch);
    const char* item = STRATA_HIP_ARCHS;
    while (*item) {
        const char* end = std::strchr(item, ',');
        if (!end) end = item + std::strlen(item);
        if ((size_t) (end - item) == size && !std::strncmp(item, arch, size)) return true;
        item = *end ? end + 1 : end;
    }
#else
    (void) arch;
#endif
    return false;
}

bool usable_device(bool rdna4_only = false) {
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) {
        (void) hipGetLastError(); // Optional backend query must not poison fallback.
        return false;
    }
    thread_local int checked_device = -1;
    thread_local bool checked_result = false;
    thread_local bool checked_rdna4 = false;
    if (dev == checked_device) return checked_result && (!rdna4_only || checked_rdna4);
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        (void) hipGetLastError();
        return false;
    }
    char arch[64]{};
    size_t length = std::strcspn(prop.gcnArchName, ":");
    if (length >= sizeof(arch)) return false;
    std::memcpy(arch, prop.gcnArchName, length);
    const bool supported = !std::strcmp(arch, "gfx1200") || !std::strcmp(arch, "gfx1201") ||
                           !std::strcmp(arch, "gfx1100") || !std::strcmp(arch, "gfx1101") ||
                           !std::strcmp(arch, "gfx1102") || !std::strcmp(arch, "gfx1103") ||
                           !std::strcmp(arch, "gfx1150") || !std::strcmp(arch, "gfx1151");
    checked_device = dev;
    checked_rdna4 = !std::strcmp(arch, "gfx1200") || !std::strcmp(arch, "gfx1201");
    checked_result = supported && prop.warpSize == 32 && compiled_arch(arch) &&
                     sizeof(Shared) <= prop.sharedMemPerBlock;
    return checked_result && (!rdna4_only || checked_rdna4);
}
} // namespace

bool available() {
    const char* requested = std::getenv("TINYSTRATA_HIP_WMMA");
    if (requested && *requested && std::strcmp(requested, "auto"))
        return !std::strcmp(requested, "1") && usable_device();
    // The measured RDNA4 path is automatic. RDNA3 has a different fragment
    // layout and remains opt-in until it has an actual-device comparison.
    return usable_device(true);
}

void run(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
         float* output, void* stream) {
    if (rows <= 0) return;
    if (kv_heads <= 0 || context <= 0 || !usable_device())
        throw std::runtime_error("HIP dense INT8 WMMA attention is unavailable for this device/build");
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    dense_int8_wmma_kernel<<<dim3((rows + QUERIES - 1) / QUERIES, kv_heads), THREADS, 0, (hipStream_t) stream>>>(
        q, cache, pos0, rows, kv_heads, context, scale_log2, output, 0);
}

void run_preview(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
                 int visible_end, float* output, void* stream) {
    if (rows <= 0) return;
    if (kv_heads <= 0 || context <= 0 || visible_end < 1 || visible_end > context || !usable_device())
        throw std::runtime_error("Invalid or unavailable HIP INT8 WMMA preview attention");
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    dense_int8_wmma_kernel<<<dim3((rows + QUERIES - 1) / QUERIES, kv_heads), THREADS, 0, (hipStream_t) stream>>>(
        q, cache, pos0, rows, kv_heads, context, scale_log2, output, visible_end);
}
} // namespace strata::qwen35::int8_attention
