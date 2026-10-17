// FP8 Tensor Core adaptation of aflin/momoe's dense attention (MIT).
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// MoMoE source: qsa_prompt_attn.cu, 47e5b2b2ab52982e105315bba2ac48ec75de01bf.
// Changes: FP8 E4M3 QK/PV, per-64 FP16 KV scales, FP8 residual Q/P,
// and resident Qwen3.6 head-major layout. All causal positions are retained.
// Reference: SageAttention's separate FP32 outer accumulation; its kernel is
// not copied here. The native MMA wrapper is adapted from FlashInfer (BSD-3).
// Experimental opt-in: see docs/fp8-native-attention.md.
// REVIEW CANDIDATE: row-wise P*V-scale quantization and real-unit FP32
// outer accumulation. Not CUDA-compiled or GPU-validated in this review.
#include "strata/qwen35/fp8_attention.hpp"
#include "strata/qwen35/fp8_mma.cuh"
#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
namespace strata::qwen35::fp8_attention {
namespace {
constexpr int HD = 256, DCH = 32, KV_Q8_GROUP = 64;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
#define FP8_MMA_SUPPORTED 1
#else
#define FP8_MMA_SUPPORTED 0
#endif
__device__ uint8_t encode(float x) { return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3); }
__device__ float decode(uint8_t x) {
    __nv_fp8_e4m3 v;
    v.__x = x;
    return (float)v;
}
__device__ __forceinline__ void mma(float *c, const uint32_t *a, const uint32_t *b) {
    const auto r = detail::mma_fp8_m16n8k32(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
    c[0] = r.d0;
    c[1] = r.d1;
    c[2] = r.d2;
    c[3] = r.d3;
}
template <int DQ> struct SmemD {
    uint32_t qh[DQ * 8][64], ql[DQ * 8][64];
    uint4 k[32][16], v[32][16];
    uint2 ksc[32], vsc[32];
    float qdn[DQ * 8];
};
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, bool valid) {
#if !FP8_MMA_SUPPORTED
    __trap();
#else
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if FP8_MMA_SUPPORTED
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if FP8_MMA_SUPPORTED
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async8(void *smem, const void *gmem, bool valid) {
#if !FP8_MMA_SUPPORTED
    (void)smem;
    (void)gmem;
    (void)valid;
    __trap();
#else
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 8 : 0));
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if FP8_MMA_SUPPORTED
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

template <bool Residual, int DQ, bool Split>
__global__ void __launch_bounds__(DQ * 16, 1)
    prompt_attn_dense_kernel(const float *__restrict__ q, Cache p, const int32_t *__restrict__ pos0, int n_q,
                             int n_kv_heads, int page_size, float scale_log2, float *__restrict__ attn, int row_offset,
                             int chunk_stride, float *partial_max, float *partial_sum) {
    constexpr int GQ = 8, PIECES = 16, ROWB = PIECES * 16, THREADS = DQ * 16;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    SmemD<DQ> &S = *reinterpret_cast<SmemD<DQ> *>(smem_raw);
    const int q0 = blockIdx.x * DQ, kvh = blockIdx.y, n_head = n_kv_heads * GQ;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int rbase = warp * 16; // this warp's MMA rows: query q0 + 2w (rows 0-7), q0 + 2w + 1 (rows 8-15), by head
    auto width = [&](int i) { return q0 + i < n_q ? (__ldg(pos0) + row_offset + q0 + i + 1) : 0; };
    const int na = width(2 * warp), nb = width(2 * warp + 1);
    int n_max = 0;
#pragma unroll
    for (int i = 0; i < DQ; ++i)
        n_max = max(n_max, width(i));

    const int cell_begin = Split ? int(blockIdx.z) * 256 : 0;
    if constexpr (Split)
        n_max = min(n_max, cell_begin + 256);
    if (n_max <= cell_begin)
        return; // graph grid can cover unused future chunks

    auto row_of = [&](int cell) -> long long {
        const long long page = 0; // TinyStrata: one head-major resident span, page_size == context capacity
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    // the K (kv 0) or V (kv 1) rows of the chunk at cell c0, cells past the widest query zero-filled
    auto issue = [&](int kv, int c0) {
        const unsigned char *base = reinterpret_cast<const unsigned char *>(kv ? p.v_q : p.k_q);
        uint4(*dst)[PIECES] = kv ? S.v : S.k;
#pragma unroll
        for (int i = t; i < DCH * PIECES; i += THREADS) {
            const int c = i / PIECES, pc = i % PIECES, cell = c0 + c;
            const bool ok = cell < n_max;
            const size_t off = ok ? (size_t)row_of(cell) * ROWB + pc * 16 : 0;
            cp_async16(&dst[c][pc ^ (c & 7)], base + off, ok);
        }
        if (t < DCH) {
            const int cell = c0 + t;
            const bool ok = cell < n_max;
            const size_t off = ok ? (size_t)row_of(cell) * (HD / KV_Q8_GROUP) : 0;
            cp_async8(kv ? &S.vsc[t] : &S.ksc[t], (kv ? p.v_scale : p.k_scale) + off, ok);
        }
    };
    if (n_max > 0) {
        issue(0, cell_begin);
        cp_async_commit();
        issue(1, cell_begin);
        cp_async_commit();
    }

    // A row is scaled into the FP8 finite range. The second FP8 component
    // carries the first component's residual, amplified by 16.
    for (int r = 0; r < 16; ++r) {
        const int qi = q0 + 2 * warp + (r >> 3), row = rbase + r;
        float x[8] = {};
        if (qi < n_q) {
            const float4 *src =
                reinterpret_cast<const float4 *>(q + ((size_t)(qi + row_offset) * n_head + kvh * GQ + (r & 7)) * HD) +
                lane * 2;
            const float4 a = src[0], b = src[1];
            x[0] = a.x;
            x[1] = a.y;
            x[2] = a.z;
            x[3] = a.w;
            x[4] = b.x;
            x[5] = b.y;
            x[6] = b.z;
            x[7] = b.w;
        }
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            m = fmaxf(m, fabsf(x[j]));
        for (int o = 16; o > 0; o >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        int e = 0;
        if (m > 0)
            frexpf(m, &e);
        const float up = ldexpf(1.f, 8 - e);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t hi = 0, lo = 0;
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float value = x[j * 4 + d] * up;
                const uint8_t code = encode(value);
                hi |= uint32_t(code) << (8 * d);
                if constexpr (Residual)
                    lo |= uint32_t(encode((value - decode(code)) * 16.f)) << (8 * d);
            }
            S.qh[row][(lane * 2 + j) ^ ((row & 7) * 4)] = hi;
            if constexpr (Residual)
                S.ql[row][(lane * 2 + j) ^ ((row & 7) * 4)] = lo;
        }
        if (lane == 0)
            S.qdn[row] = ldexpf(scale_log2, e - 8);
    }
    __syncwarp();
    const float da = S.qdn[rbase + gid], db = S.qdn[rbase + 8 + gid];

    float acc[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; ++j)
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.0f, l_b = 0.0f; // l: this thread's columns only (summed at the end)
    // acc is kept in real V units; each tile's FP8 scale is removed
    // before updating this FP32 outer accumulator.

#pragma unroll 1
    for (int c0 = cell_begin; c0 < n_max; c0 += DCH) {
        const bool more = c0 + DCH < n_max;
        cp_async_wait1();
        __syncthreads(); // K in

        float s[4][4] = {};
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            float th[4][4] = {}, tl[4][4] = {};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int word = (g * 64 + ks * 32) / 4 + tig;
                uint32_t ah[4], al[4];
                const int ra = rbase + gid, rb = ra + 8;
                ah[0] = S.qh[ra][word ^ ((ra & 7) * 4)];
                ah[1] = S.qh[rb][word ^ ((rb & 7) * 4)];
                ah[2] = S.qh[ra][(word + 4) ^ ((ra & 7) * 4)];
                ah[3] = S.qh[rb][(word + 4) ^ ((rb & 7) * 4)];
                if constexpr (Residual) {
                    al[0] = S.ql[ra][word ^ ((ra & 7) * 4)];
                    al[1] = S.ql[rb][word ^ ((rb & 7) * 4)];
                    al[2] = S.ql[ra][(word + 4) ^ ((ra & 7) * 4)];
                    al[3] = S.ql[rb][(word + 4) ^ ((rb & 7) * 4)];
                }
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int cell = nt * 8 + gid, dim = g * 64 + ks * 32 + 4 * tig;
                    auto ptr = [&](int d) {
                        return reinterpret_cast<const uint8_t *>(&S.k[0][0]) + cell * 256 +
                               ((d / 16) ^ (cell & 7)) * 16 + (d & 15);
                    };
                    uint32_t bb[2] = {*reinterpret_cast<const uint32_t *>(ptr(dim)),
                                      *reinterpret_cast<const uint32_t *>(ptr(dim + 16))};
                    mma(th[nt], ah, bb);
                    if constexpr (Residual)
                        mma(tl[nt], al, bb);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = nt * 8 + 2 * tig;
                const float k0 = __half2float(reinterpret_cast<const __half *>(&S.ksc[c])[g]);
                const float k1 = __half2float(reinterpret_cast<const __half *>(&S.ksc[c + 1])[g]);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    s[nt][j] = fmaf(th[nt][j] + tl[nt][j] * (1.f / 16), j % 2 ? k1 : k0, s[nt][j]);
            }
        }
        __syncthreads(); // every warp is done with K
        if (more) {
            issue(0, c0 + DCH);
            cp_async_commit();
        }
        // online softmax in registers: rows a (gid) and b (gid + 8), this thread's 8 columns, the row's 4 threads
        float alpha_a, alpha_b;
        {
            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < DCH / 8; ++nt) {
                const int cell = c0 + nt * 8 + 2 * tig;
                s[nt][0] = cell < na ? s[nt][0] * da : -INFINITY;
                s[nt][1] = cell + 1 < na ? s[nt][1] * da : -INFINITY;
                s[nt][2] = cell < nb ? s[nt][2] * db : -INFINITY;
                s[nt][3] = cell + 1 < nb ? s[nt][3] * db : -INFINITY;
                mxa = fmaxf(mxa, fmaxf(s[nt][0], s[nt][1]));
                mxb = fmaxf(mxb, fmaxf(s[nt][2], s[nt][3]));
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
            }
            const float ma = fmaxf(m_a, mxa), mb = fmaxf(m_b, mxb);
            alpha_a = m_a == -INFINITY ? 0.0f : exp2f(m_a - ma);
            alpha_b = m_b == -INFINITY ? 0.0f : exp2f(m_b - mb);
            float sa = 0.0f, sb = 0.0f;
#pragma unroll
            for (int nt = 0; nt < DCH / 8; ++nt) {
                s[nt][0] = s[nt][0] == -INFINITY ? 0.0f : exp2f(s[nt][0] - ma);
                s[nt][1] = s[nt][1] == -INFINITY ? 0.0f : exp2f(s[nt][1] - ma);
                s[nt][2] = s[nt][2] == -INFINITY ? 0.0f : exp2f(s[nt][2] - mb);
                s[nt][3] = s[nt][3] == -INFINITY ? 0.0f : exp2f(s[nt][3] - mb);
                sa += s[nt][0] + s[nt][1];
                sb += s[nt][2] + s[nt][3];
            }
            l_a = fmaf(l_a, alpha_a, sa);
            l_b = fmaf(l_b, alpha_b, sb);
            m_a = ma;
            m_b = mb;
        }
        if (more)
            cp_async_wait1();
        else
            cp_async_wait0();
        __syncthreads(); // V in

#pragma unroll

        for (int g = 0; g < 4; ++g) {
            // Choose a scale from the actual MMA operand, P * V_scale,
            // independently for rows a and b. P is already causally masked.
            // A masked/future V outlier must not set another row's scale.
            float wmax_a = 0.0f, wmax_b = 0.0f;
#pragma unroll
            for (int nt = 0; nt < DCH / 8; ++nt) {
                const int cell = nt * 8 + 2 * tig;
                const float v0 = fabsf(__half2float(reinterpret_cast<const __half *>(&S.vsc[cell])[g]));
                const float v1 = fabsf(__half2float(reinterpret_cast<const __half *>(&S.vsc[cell + 1])[g]));
                wmax_a = fmaxf(wmax_a, fmaxf(s[nt][0] * v0, s[nt][1] * v1));
                wmax_b = fmaxf(wmax_b, fmaxf(s[nt][2] * v0, s[nt][3] * v1));
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                wmax_a = fmaxf(wmax_a, __shfl_xor_sync(0xffffffffu, wmax_a, o));
                wmax_b = fmaxf(wmax_b, __shfl_xor_sync(0xffffffffu, wmax_b, o));
            }
            int exp_a = 0, exp_b = 0;
            if (wmax_a > 0.0f)
                frexpf(wmax_a, &exp_a);
            if (wmax_b > 0.0f)
                frexpf(wmax_b, &exp_b);
            const int up_a = wmax_a > 0.0f ? 8 - exp_a : 0;
            const int up_b = wmax_b > 0.0f ? 8 - exp_b : 0;
            uint32_t ah[4] = {}, al[4] = {};
#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cell = (reg / 2) * 16 + 4 * tig + j;
                    const int nt = cell / 8, component = (reg % 2) * 2 + (cell & 1),
                              source_lane = gid * 4 + (cell % 8) / 2;
                    float prob = 0.f;
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const float v = __shfl_sync(0xffffffffu, s[n][component], source_lane);
                        if (nt == n)
                            prob = v;
                    }
                    const float weighted_prob =
                        prob * __half2float(reinterpret_cast<const __half *>(&S.vsc[cell])[g]);
                    // Scale the product, not V_scale first: a small P can
                    // otherwise require an overflowing intermediate scale.
                    const float value = ldexpf(weighted_prob, (reg & 1) ? up_b : up_a);
                    const uint8_t code = encode(value);
                    ah[reg] |= uint32_t(code) << (j * 8);
                    if constexpr (Residual)
                        al[reg] |= uint32_t(encode((value - decode(code)) * 16.f)) << (j * 8);
                }
            }
            const unsigned char *V = reinterpret_cast<const unsigned char *>(&S.v[0][0]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int dim = g * 64 + j * 8 + gid;
                auto at = [&](int cell) {
                    return uint32_t(V[cell * 256 + ((dim / 16) ^ (cell & 7)) * 16 + (dim & 15)]);
                };
                uint32_t bb[2] = {};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    bb[0] |= at(4 * tig + k) << (8 * k);
                    bb[1] |= at(16 + 4 * tig + k) << (8 * k);
                }
                float th[4] = {}, tl[4] = {};
                mma(th, ah, bb);
                if constexpr (Residual)
                    mma(tl, al, bb);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float tile_value = ldexpf(th[k] + tl[k] * (1.f / 16), k < 2 ? -up_a : -up_b);
                    acc[g * 8 + j][k] = fmaf(acc[g * 8 + j][k], k < 2 ? alpha_a : alpha_b, tile_value);
                }
            }
        }
        __syncthreads(); // every warp is done with V
        if (more) {
            issue(1, c0 + DCH);
            cp_async_commit();
        }
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        l_a += __shfl_xor_sync(0xffffffffu, l_a, o);
        l_b += __shfl_xor_sync(0xffffffffu, l_b, o);
    }
    const int qa = q0 + 2 * warp, qb = qa + 1;
    const size_t oa_index = (size_t)qa * n_head + kvh * GQ + gid;
    const size_t ob_index = (size_t)qb * n_head + kvh * GQ + gid;
    const size_t pa = oa_index * chunk_stride + blockIdx.z, pb = ob_index * chunk_stride + blockIdx.z;
    float *oa = attn + (Split ? pa : oa_index) * HD;
    float *ob = attn + (Split ? pb : ob_index) * HD;
    if constexpr (Split) {
        // The shared engine merge consumes natural-log maxima and unnormalized
        // numerators. Convert the base-2 softmax maximum, not the denominator.
        if (tig == 0 && qa < n_q) {
            partial_max[pa] = m_a * 0.6931471805599453f;
            partial_sum[pa] = l_a;
        }
        if (tig == 0 && qb < n_q) {
            partial_max[pb] = m_b * 0.6931471805599453f;
            partial_sum[pb] = l_b;
        }
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const float ia = l_a > 0.0f ? (Split ? 1.0f : 1.0f / l_a) : 0.0f,
                    ib = l_b > 0.0f ? (Split ? 1.0f : 1.0f / l_b) : 0.0f;
#pragma unroll
        for (int j = 8 * g; j < 8 * g + 8; ++j) {
            const int d = j * 8 + 2 * tig;
            if (qa < n_q)
                *reinterpret_cast<float2 *>(oa + d) = make_float2(acc[j][0] * ia, acc[j][1] * ia);
            if (qb < n_q)
                *reinterpret_cast<float2 *>(ob + d) = make_float2(acc[j][2] * ib, acc[j][3] * ib);
        }
    }
}

} // namespace
bool available() {
    int dev;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess ||
        p.major * 10 + p.minor < 89)
        return false;
    cudaFuncAttributes attr{};
    // A compute_86-only build can JIT on a newer GPU, but its compiled kernel
    // contains the unsupported-instruction branch. Check the selected image.
    if (cudaFuncGetAttributes(&attr, prompt_attn_dense_kernel<true, 8, false>) != cudaSuccess || attr.ptxVersion < 89)
        return false;
    return cudaFuncSetAttribute(prompt_attn_dense_kernel<true, 8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(SmemD<8>)) == cudaSuccess &&
           cudaFuncSetAttribute(prompt_attn_dense_kernel<true, 2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(SmemD<2>)) == cudaSuccess;
}
void run(const float *q, Cache cache, const int32_t *pos, int rows, int heads, int ctx, float *out, void *stream) {
    const dim3 grid((rows + 7) / 8, heads);
    prompt_attn_dense_kernel<true, 8, false><<<grid, 128, sizeof(SmemD<8>), (cudaStream_t)stream>>>(
        q, cache, pos, rows, heads, ctx, 1.4426950408889634f / 16, out, 0, 0, nullptr, nullptr);
}
void run_split(const float *q, Cache cache, const int32_t *pos, int row_offset, int rows, int heads, int ctx,
               int grid_chunks, int chunk_stride, float *partial, float *maxima, float *sums, void *stream) {
    const dim3 grid((rows + 1) / 2, heads, grid_chunks);
    prompt_attn_dense_kernel<true, 2, true><<<grid, 32, sizeof(SmemD<2>), (cudaStream_t)stream>>>(
        q, cache, pos, rows, heads, ctx, 1.4426950408889634f / 16, partial, row_offset, chunk_stride, maxima, sums);
}
} // namespace strata::qwen35::fp8_attention
