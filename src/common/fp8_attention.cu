// Native E4M3 attention, adapted from TinyStrata's corrected Qwen FP8 kernel.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// The Qwen kernel derives from Aaron Flin's MIT dense attention and the
// BSD-3-Clause FlashInfer MMA wrapper (see include/strata/qwen35/fp8_mma.cuh).
// Changes: token-major/per-32 KV, head dimensions 64/256/512, arbitrary GQA
// divisors of 16, sliding rings, bidirectional image blocks, and bounded
// output tiles with a 32-way split-K path. QK and PV both use E4M3 MMA directly;
// KV is never expanded to an FP16 attention workspace.
#include "strata/common/fp8_attention.hpp"
#include "strata/qwen35/fp8_mma.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::fp8_attention {
namespace {
constexpr int CELLS = 32, SPLITS = 32;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float RESIDUAL_UP = 16.f, RESIDUAL_DOWN = 1.f / RESIDUAL_UP;
thread_local int prepared_device = -1;
thread_local bool prepared = false;
thread_local int gemma_prepared_device = -1;
thread_local bool gemma_prepared = false;

// PTX ISA version is not the target SM. A compute_86 image JIT-compiled for
// Blackwell still contains the preprocessor's sm_86 branches. Query the actual
// source compilation target before making this path available.
__global__ void compiled_target_kernel(int *target) {
#if defined(__CUDA_ARCH__)
    *target = __CUDA_ARCH__;
#else
    *target = 0;
#endif
}

__device__ __forceinline__ std::uint8_t encode(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float decode(std::uint8_t x) {
    __nv_fp8_e4m3 v;
    v.__x = x;
    return float(v);
}
__device__ __forceinline__ void mma(float *c, const std::uint32_t *a, const std::uint32_t *b) {
    const auto r = qwen35::fp8_attention::detail::mma_fp8_m16n8k32(
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
    c[0] = r.d0; c[1] = r.d1; c[2] = r.d2; c[3] = r.d3;
}
template<int BYTES>
__device__ __forceinline__ void copy_async(void *smem, const void *gmem, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const unsigned sa = unsigned(__cvta_generic_to_shared(smem));
    if constexpr (BYTES == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
    else if constexpr (BYTES == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(sa), "l"(gmem), "r"(valid ? 8 : 0));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(gmem), "r"(valid ? 4 : 0));
#endif
}
__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template<int N> __device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__global__ void store_kernel(const float *__restrict__ src, std::uint8_t *__restrict__ codes,
                             std::uint16_t *__restrict__ scales, int width, std::int64_t pos,
                             const std::int64_t *__restrict__ at, int ring) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (group >= width / 32) return;
    std::int64_t row = (at ? at[0] : pos) + blockIdx.y;
    if (ring) row %= ring;
    const int d = group * 32 + lane;
    const float x = src[std::size_t(blockIdx.y) * width + d];
    float m = fabsf(x);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    // Quantize against the multiplier actually stored, not its unrounded FP32
    // precursor. Keep a positive representable multiplier for tiny/zero groups.
    const __half hs = __float2half_rn(fminf(65504.f, fmaxf(m / 448.f, 0x1p-24f)));
    const float scale = __half2float(hs);
    codes[std::size_t(row) * width + d] = encode(x / scale);
    if (!lane) scales[std::size_t(row) * (width / 32) + group] = __half_as_ushort(hs);
}

template<int HD, int WARPS> struct Shared {
    static constexpr int OUT = HD == 64 ? 64 : 128;
    static constexpr int QROWS = WARPS * 16;
    static constexpr int KWORDS = HD / 16, VWORDS = OUT / 16;
    std::uint32_t qh[QROWS][HD / 4], ql[QROWS][HD / 4];
    uint4 k[CELLS][KWORDS], v[CELLS][VWORDS];
    __align__(16) std::uint16_t ks[CELLS][HD / 32];
    __align__(16) std::uint16_t vs[CELLS][OUT / 32];
    float qdown[QROWS];
};

template<int HD, int WARPS, bool SPLIT>
__global__ void __launch_bounds__(WARPS * 32, 1)
attention_kernel(const float *__restrict__ q, Cache p, std::int64_t pos,
                 const std::int64_t *__restrict__ at, int rows, int heads, int kv_heads,
                 int gqa, int ring, int window, bool whole, float scale_log2,
                 float *__restrict__ out, float *__restrict__ partial) {
    using SType = Shared<HD, WARPS>;
    constexpr int OUT = SType::OUT, THREADS = WARPS * 32;
    constexpr int QSWIZZLE = (HD / 16 < 8 ? HD / 16 : 8) - 1;
    constexpr int KSWIZZLE = (SType::KWORDS < 8 ? SType::KWORDS : 8) - 1;
    constexpr int VSWIZZLE = SType::VWORDS - 1;
    extern __shared__ __align__(16) unsigned char raw[];
    SType &s = *reinterpret_cast<SType *>(raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, rbase = warp * 16;
    const int tile_rows = SType::QROWS / gqa, q0 = blockIdx.x * tile_rows, kvh = blockIdx.y;
    const int output_slice = int(blockIdx.z) % (HD / OUT);
    const int split = SPLIT ? int(blockIdx.z) / (HD / OUT) : 0;
    const int out_dim = output_slice * OUT;
    const std::int64_t base = at ? at[0] : pos;
    const std::int64_t first_pos = base + q0;
    const std::int64_t span_lo = window > 0 && first_pos >= window ? first_pos - window + 1 : 0;
    const std::int64_t span_hi = base + (whole ? rows : min(rows, q0 + tile_rows));
    const int span = int(span_hi - span_lo);
    const int segment = SPLIT ? ((span + SPLITS * CELLS - 1) / (SPLITS * CELLS)) * CELLS : span;
    const std::int64_t begin = span_lo + (SPLIT ? split * segment : 0);
    const std::int64_t end = SPLIT ? min(span_hi, begin + segment) : span_hi;
    const int ra = rbase + gid, rb = ra + 8;
    const int qa = q0 + ra / gqa, qb = q0 + rb / gqa;
    const int ha = kvh * gqa + ra % gqa, hb = kvh * gqa + rb % gqa;
    const auto lower = [&](int qi) -> std::int64_t {
        const std::int64_t p0 = base + qi;
        return window > 0 && p0 >= window ? p0 - window + 1 : 0;
    };
    const std::int64_t loa = lower(qa), lob = lower(qb);
    const std::int64_t hia = base + (whole ? rows : qa + 1), hib = base + (whole ? rows : qb + 1);

    const auto cache_row = [&](std::int64_t cell) -> std::size_t {
        return std::size_t(ring ? cell % ring : cell) * kv_heads + kvh;
    };
    const auto issue = [&](bool value, std::int64_t c0) {
        constexpr int KP = SType::KWORDS, VP = SType::VWORDS;
        if (!value) {
            for (int i = tid; i < CELLS * KP; i += THREADS) {
                const int c = i / KP, pc = i % KP;
                const bool ok = c0 + c < end;
                const std::size_t off = ok ? cache_row(c0 + c) * HD + pc * 16 : 0;
                copy_async<16>(&s.k[c][pc ^ (c & KSWIZZLE)], p.k + off, ok);
            }
            for (int c = tid; c < CELLS; c += THREADS) {
                const bool ok = c0 + c < end;
                const std::size_t off = ok ? cache_row(c0 + c) * (HD / 32) : 0;
                if constexpr (HD == 64) copy_async<4>(s.ks[c], p.ks + off, ok);
                else if constexpr (HD == 256) copy_async<16>(s.ks[c], p.ks + off, ok);
                else {
                    copy_async<16>(s.ks[c], p.ks + off, ok);
                    copy_async<16>(s.ks[c] + 8, p.ks + off + 8, ok);
                }
            }
        } else {
            for (int i = tid; i < CELLS * VP; i += THREADS) {
                const int c = i / VP, pc = i % VP;
                const bool ok = c0 + c < end;
                const std::size_t off = ok ? cache_row(c0 + c) * HD + out_dim + pc * 16 : 0;
                copy_async<16>(&s.v[c][pc ^ (c & VSWIZZLE)], p.v + off, ok);
            }
            for (int c = tid; c < CELLS; c += THREADS) {
                const bool ok = c0 + c < end;
                const std::size_t off = ok ? cache_row(c0 + c) * (HD / 32) + out_dim / 32 : 0;
                if constexpr (OUT == 64) copy_async<4>(s.vs[c], p.vs + off, ok);
                else copy_async<8>(s.vs[c], p.vs + off, ok);
            }
        }
    };
    if (begin < end) {
        issue(false, begin); commit();
        issue(true, begin); commit();
    }

    // A warp builds its sixteen query/head rows. The second E4M3 component
    // carries the first component's residual amplified by sixteen.
#pragma unroll 1
    for (int r = 0; r < 16; ++r) {
        const int row = rbase + r, qi = q0 + row / gqa, h = kvh * gqa + row % gqa;
        const float *src = q + (std::size_t(qi) * heads + h) * HD;
        float m = 0.f;
#pragma unroll
        for (int d = lane; d < HD; d += 32) if (qi < rows) m = fmaxf(m, fabsf(src[d]));
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        int e = 0;
        if (m > 0.f) frexpf(m, &e);
        const int up = 8 - e;
        for (int word = lane; word < HD / 4; word += 32) {
            std::uint32_t hi = 0, lo = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float x = qi < rows ? ldexpf(src[word * 4 + j], up) : 0.f;
                const std::uint8_t b = encode(x);
                hi |= std::uint32_t(b) << (8 * j);
                lo |= std::uint32_t(encode((x - decode(b)) * RESIDUAL_UP)) << (8 * j);
            }
            const int sw = word ^ ((row & QSWIZZLE) * 4);
            s.qh[row][sw] = hi; s.ql[row][sw] = lo;
        }
        if (!lane) s.qdown[row] = ldexpf(scale_log2, e - 8);
    }
    __syncwarp();
    const float da = s.qdown[ra], db = s.qdown[rb];
    float acc[OUT / 8][4] = {};
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;

#pragma unroll 1
    for (std::int64_t c0 = begin; c0 < end; c0 += CELLS) {
        const bool more = c0 + CELLS < end;
        wait<1>(); __syncthreads();
        float score[4][4] = {};
#pragma unroll
        for (int g = 0; g < HD / 32; ++g) {
            const int word = g * 8 + tig;
            const int swa = (ra & QSWIZZLE) * 4, swb = (rb & QSWIZZLE) * 4;
            std::uint32_t ah[4] = {s.qh[ra][word ^ swa], s.qh[rb][word ^ swb],
                                   s.qh[ra][(word + 4) ^ swa], s.qh[rb][(word + 4) ^ swb]};
            std::uint32_t al[4] = {s.ql[ra][word ^ swa], s.ql[rb][word ^ swb],
                                   s.ql[ra][(word + 4) ^ swa], s.ql[rb][(word + 4) ^ swb]};
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = nt * 8 + gid, d = g * 32 + 4 * tig;
                const auto ptr = [&](int dim) {
                    return reinterpret_cast<const std::uint8_t *>(&s.k[0][0]) + c * HD +
                           ((dim / 16) ^ (c & KSWIZZLE)) * 16 + (dim & 15);
                };
                std::uint32_t b[2] = {*reinterpret_cast<const std::uint32_t *>(ptr(d)),
                                      *reinterpret_cast<const std::uint32_t *>(ptr(d + 16))};
                float hi[4] = {}, lo[4] = {};
                mma(hi, ah, b); mma(lo, al, b);
                const int sc = nt * 8 + 2 * tig;
                const float k0 = __half2float(__ushort_as_half(s.ks[sc][g]));
                const float k1 = __half2float(__ushort_as_half(s.ks[sc + 1][g]));
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    score[nt][j] = fmaf(fmaf(lo[j], RESIDUAL_DOWN, hi[j]), (j & 1) ? k1 : k0, score[nt][j]);
            }
        }
        __syncthreads();
        if (more) { issue(false, c0 + CELLS); commit(); }
        float xa = -INFINITY, xb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const std::int64_t cell = c0 + nt * 8 + 2 * tig;
            score[nt][0] = qa < rows && cell >= loa && cell < hia ? score[nt][0] * da : -INFINITY;
            score[nt][1] = qa < rows && cell + 1 >= loa && cell + 1 < hia ? score[nt][1] * da : -INFINITY;
            score[nt][2] = qb < rows && cell >= lob && cell < hib ? score[nt][2] * db : -INFINITY;
            score[nt][3] = qb < rows && cell + 1 >= lob && cell + 1 < hib ? score[nt][3] * db : -INFINITY;
            xa = fmaxf(xa, fmaxf(score[nt][0], score[nt][1]));
            xb = fmaxf(xb, fmaxf(score[nt][2], score[nt][3]));
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o));
        }
        const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        const float aa = ma == -INFINITY ? 0.f : exp2f(ma - na);
        const float ab = mb == -INFINITY ? 0.f : exp2f(mb - nb);
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                score[nt][j] = score[nt][j] == -INFINITY ? 0.f : exp2f(score[nt][j] - (j < 2 ? na : nb));
            sa += score[nt][0] + score[nt][1]; sb += score[nt][2] + score[nt][3];
        }
        la = fmaf(la, aa, sa); lb = fmaf(lb, ab, sb); ma = na; mb = nb;
        if (more) wait<1>(); else wait<0>();
        __syncthreads();
#pragma unroll
        for (int g = 0; g < OUT / 32; ++g) {
            float wma = 0.f, wmb = 0.f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = nt * 8 + 2 * tig;
                const float v0 = __half2float(__ushort_as_half(s.vs[c][g]));
                const float v1 = __half2float(__ushort_as_half(s.vs[c + 1][g]));
                wma = fmaxf(wma, fmaxf(score[nt][0] * v0, score[nt][1] * v1));
                wmb = fmaxf(wmb, fmaxf(score[nt][2] * v0, score[nt][3] * v1));
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                wma = fmaxf(wma, __shfl_xor_sync(0xffffffffu, wma, o));
                wmb = fmaxf(wmb, __shfl_xor_sync(0xffffffffu, wmb, o));
            }
            int ea = 0, eb = 0;
            if (wma > 0.f) frexpf(wma, &ea);
            if (wmb > 0.f) frexpf(wmb, &eb);
            const int ua = wma > 0.f ? 8 - ea : 0, ub = wmb > 0.f ? 8 - eb : 0;
            std::uint32_t ph[4] = {}, pl[4] = {};
#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = (reg / 2) * 16 + 4 * tig + j;
                    const int nt = c / 8, component = (reg % 2) * 2 + (c & 1);
                    const int source_lane = gid * 4 + (c % 8) / 2;
                    float prob = 0.f;
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const float v = __shfl_sync(0xffffffffu, score[n][component], source_lane);
                        if (nt == n) prob = v;
                    }
                    const float vs = __half2float(__ushort_as_half(s.vs[c][g]));
                    const float weighted = prob == 0.f ? 0.f : prob * vs;
                    const float x = ldexpf(weighted, (reg & 1) ? ub : ua);
                    const std::uint8_t b = encode(x);
                    ph[reg] |= std::uint32_t(b) << (j * 8);
                    pl[reg] |= std::uint32_t(encode((x - decode(b)) * RESIDUAL_UP)) << (j * 8);
                }
            }
            const auto *v = reinterpret_cast<const std::uint8_t *>(&s.v[0][0]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dim = g * 32 + j * 8 + gid;
                const auto value = [&](int c) {
                    return std::uint32_t(v[c * OUT + ((dim / 16) ^ (c & VSWIZZLE)) * 16 + (dim & 15)]);
                };
                std::uint32_t b[2] = {};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    b[0] |= value(4 * tig + k) << (k * 8);
                    b[1] |= value(16 + 4 * tig + k) << (k * 8);
                }
                float hi[4] = {}, lo[4] = {};
                mma(hi, ph, b); mma(lo, pl, b);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float tile = ldexpf(fmaf(lo[k], RESIDUAL_DOWN, hi[k]), k < 2 ? -ua : -ub);
                    acc[g * 4 + j][k] = fmaf(acc[g * 4 + j][k], k < 2 ? aa : ab, tile);
                }
            }
        }
        __syncthreads();
        if (more) { issue(true, c0 + CELLS); commit(); }
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, o); lb += __shfl_xor_sync(0xffffffffu, lb, o);
    }
    const std::size_t ia = std::size_t(qa) * heads + ha, ib = std::size_t(qb) * heads + hb;
    float *oa = SPLIT ? partial + (ia * SPLITS + split) * (HD + 2) : out + ia * HD;
    float *ob = SPLIT ? partial + (ib * SPLITS + split) * (HD + 2) : out + ib * HD;
    if constexpr (SPLIT) {
        if (!output_slice && !tig) {
            if (qa < rows) { oa[HD] = ma; oa[HD + 1] = la; }
            if (qb < rows) { ob[HD] = mb; ob[HD + 1] = lb; }
        }
    }
    const float inva = SPLIT ? 1.f : (la > 0.f ? 1.f / la : 0.f);
    const float invb = SPLIT ? 1.f : (lb > 0.f ? 1.f / lb : 0.f);
#pragma unroll
    for (int j = 0; j < OUT / 8; ++j) {
        const int d = out_dim + j * 8 + 2 * tig;
        if (qa < rows) *reinterpret_cast<float2 *>(oa + d) = make_float2(acc[j][0] * inva, acc[j][1] * inva);
        if (qb < rows) *reinterpret_cast<float2 *>(ob + d) = make_float2(acc[j][2] * invb, acc[j][3] * invb);
    }
}

template<int HD> __global__ void merge_kernel(const float *__restrict__ part, float *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    const float *p = part + std::size_t(blockIdx.x) * SPLITS * (HD + 2);
    float m = p[lane * (HD + 2) + HD];
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float factor = p[lane * (HD + 2) + HD + 1] > 0.f ? exp2f(p[lane * (HD + 2) + HD] - m) : 0.f;
    float denom = factor * p[lane * (HD + 2) + HD + 1];
#pragma unroll
    for (int o = 16; o; o >>= 1) denom += __shfl_xor_sync(0xffffffffu, denom, o);
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float sum = 0.f;
#pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float f = __shfl_sync(0xffffffffu, factor, s);
            sum = fmaf(p[s * (HD + 2) + d], f, sum);
        }
        out[std::size_t(blockIdx.x) * HD + d] = denom > 0.f ? sum / denom : 0.f;
    }
}

template<int HD, int WARPS, bool SPLIT> bool prepare_kernel() {
    cudaFuncAttributes a{};
    auto k = attention_kernel<HD, WARPS, SPLIT>;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess || a.ptxVersion < 89) return false;
    return cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared<HD, WARPS>)) == cudaSuccess;
}

// Gemma uses 256/512-dimensional heads. Repeating QK and softmax for each
// 128-value output slice wastes most of the wide-head path's work. Here one
// producer warp evaluates the sixteen query/head rows once. Every warp then
// owns a different 128-value output slice, with probabilities shared between
// them. Keeping the per-thread output accumulator bounded also avoids the
// register spills of a single-warp, full-width accumulator.
constexpr float GEMMA_RESIDUAL_UP = 32.f;
constexpr float GEMMA_RESIDUAL_DOWN = 1.f / GEMMA_RESIDUAL_UP;
constexpr float GEMMA_TAIL_DOWN = GEMMA_RESIDUAL_DOWN * GEMMA_RESIDUAL_DOWN;
constexpr int GEMMA_TARGET_CTAS = 64;

__device__ __forceinline__ void gemma_encode3(float x, std::uint8_t &hi,
                                             std::uint8_t &lo, std::uint8_t &tail) {
    hi = encode(x);
    const float residual = __fsub_rn(x, decode(hi));
    lo = encode(__fmul_rn(residual, GEMMA_RESIDUAL_UP));
    tail = encode(__fmul_rn(__fsub_rn(residual, __fmul_rn(decode(lo), GEMMA_RESIDUAL_DOWN)),
                           GEMMA_RESIDUAL_UP * GEMMA_RESIDUAL_UP));
}

template<int HD> struct SharedGemma {
    static constexpr int WORDS = HD / 16;
    std::uint32_t q[3][16][HD / 4];
    uint4 k[CELLS][WORDS], v[CELLS][WORDS];
    __align__(16) std::uint16_t ks[CELLS][HD / 32];
    __align__(16) std::uint16_t vs[CELLS][HD / 32];
    // Pad the row stride: PV's eight different query/head rows must not all
    // hit the same shared-memory bank when loading the same token column.
    float probability[16][CELLS + 1], alpha[16], qdown[16];
};

__device__ __forceinline__ int gemma_split_count(int span, int kv_heads) {
    // Gemma's full attention has only two KV heads. Cap the work per split
    // through the desired CTA count rather than using one long, under-filled
    // CTA per head. SWA's eight KV heads need fewer splits for the same fill.
    const int target = min(SPLITS, (GEMMA_TARGET_CTAS + kv_heads - 1) / kv_heads);
    return min(target, max(1, (span + CELLS - 1) / CELLS));
}

template<int HD, bool SPLIT>
__global__ void __launch_bounds__((HD / 128) * 32, 1)
gemma_attention_kernel(const float *__restrict__ q, Cache p, std::int64_t pos,
                       const std::int64_t *__restrict__ at, int rows, int heads, int kv_heads,
                       int gqa, int ring, int window, bool whole, float scale_log2,
                       float *__restrict__ out, float *__restrict__ partial) {
    constexpr int WARPS = HD / 128, THREADS = WARPS * 32, OUT = 128;
    constexpr int WORDS = HD / 16, SWIZZLE = 7;
    extern __shared__ __align__(16) unsigned char raw[];
    auto &s = *reinterpret_cast<SharedGemma<HD> *>(raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, ra = gid, rb = gid + 8;
    const int tile_rows = 16 / gqa, q0 = blockIdx.x * tile_rows, kvh = blockIdx.y;
    const int split = SPLIT ? int(blockIdx.z) : 0, out_dim = warp * OUT;
    const std::int64_t base = at ? at[0] : pos;
    const std::int64_t first_pos = base + q0;
    const std::int64_t span_lo = window > 0 && first_pos >= window ? first_pos - window + 1 : 0;
    const std::int64_t span_hi = base + (whole ? rows : min(rows, q0 + tile_rows));
    const int span = int(span_hi - span_lo);
    const int active = SPLIT ? gemma_split_count(span, kv_heads) : 1;
    if (split >= active) return; // device-side decision remains valid in a CUDA Graph
    const int chunks = (span + CELLS - 1) / CELLS;
    // Balance whole 32-cell tiles instead of rounding every segment upward:
    // e.g. 33 tiles over 32 splits must keep all 32 CTAs doing useful work.
    const std::int64_t begin = span_lo + (SPLIT ? (split * chunks / active) * CELLS : 0);
    const std::int64_t end = SPLIT ? min(span_hi, span_lo + ((split + 1) * chunks / active) * CELLS) : span_hi;
    const int qa = q0 + ra / gqa, qb = q0 + rb / gqa;
    const int ha = kvh * gqa + ra % gqa, hb = kvh * gqa + rb % gqa;
    const auto lower = [&](int qi) -> std::int64_t {
        const std::int64_t n = base + qi;
        return window > 0 && n >= window ? n - window + 1 : 0;
    };
    const std::int64_t loa = lower(qa), lob = lower(qb);
    const std::int64_t hia = base + (whole ? rows : qa + 1);
    const std::int64_t hib = base + (whole ? rows : qb + 1);
    const auto cache_row = [&](std::int64_t cell) -> std::size_t {
        return std::size_t(ring ? cell % ring : cell) * kv_heads + kvh;
    };
    const auto issue = [&](bool value, std::int64_t c0) {
        auto *dst = value ? s.v : s.k;
        const auto *src = value ? p.v : p.k;
        for (int i = tid; i < CELLS * WORDS; i += THREADS) {
            const int c = i / WORDS, pc = i % WORDS;
            const bool ok = c0 + c < end;
            const std::size_t off = ok ? cache_row(c0 + c) * HD + pc * 16 : 0;
            copy_async<16>(&dst[c][pc ^ (c & SWIZZLE)], src + off, ok);
        }
        for (int c = tid; c < CELLS; c += THREADS) {
            const bool ok = c0 + c < end;
            const std::size_t off = ok ? cache_row(c0 + c) * (HD / 32) : 0;
            auto *dest = value ? s.vs[c] : s.ks[c];
            const auto *sc = (value ? p.vs : p.ks) + off;
            copy_async<16>(dest, sc, ok);
            if constexpr (HD == 512) copy_async<16>(dest + 8, sc + 8, ok);
        }
    };
    if (begin < end) {
        issue(false, begin); commit();
        issue(true, begin); commit();
    }
    if (!warp) {
#pragma unroll 1
        for (int row = 0; row < 16; ++row) {
            const int qi = q0 + row / gqa, h = kvh * gqa + row % gqa;
            const float *src = q + (std::size_t(qi) * heads + h) * HD;
            float m = 0.f;
#pragma unroll
            for (int d = lane; d < HD; d += 32) if (qi < rows) m = fmaxf(m, fabsf(src[d]));
#pragma unroll
            for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            int e = 0;
            if (m > 0.f) frexpf(m, &e);
            const int up = 8 - e;
            for (int word = lane; word < HD / 4; word += 32) {
                std::uint32_t hi = 0, lo = 0, tail = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float x = qi < rows ? ldexpf(src[word * 4 + j], up) : 0.f;
                    std::uint8_t h, l, t;
                    gemma_encode3(x, h, l, t);
                    hi |= std::uint32_t(h) << (8 * j);
                    lo |= std::uint32_t(l) << (8 * j);
                    tail |= std::uint32_t(t) << (8 * j);
                }
                const int sw = word ^ ((row & SWIZZLE) * 4);
                s.q[0][row][sw] = hi; s.q[1][row][sw] = lo; s.q[2][row][sw] = tail;
            }
            if (!lane) s.qdown[row] = ldexpf(scale_log2, e - 8);
        }
    }
    __syncthreads();
    const float da = s.qdown[ra], db = s.qdown[rb];
    float acc[OUT / 8][4] = {};
    float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
#pragma unroll 1
    for (std::int64_t c0 = begin; c0 < end; c0 += CELLS) {
        const bool more = c0 + CELLS < end;
        wait<1>(); __syncthreads();
        if (!warp) {
            float score[4][4] = {};
#pragma unroll
            for (int g = 0; g < HD / 32; ++g) {
                const int word = g * 8 + tig, swa = (ra & SWIZZLE) * 4, swb = (rb & SWIZZLE) * 4;
                std::uint32_t a[3][4];
#pragma unroll
                for (int part = 0; part < 3; ++part) {
                    a[part][0] = s.q[part][ra][word ^ swa]; a[part][1] = s.q[part][rb][word ^ swb];
                    a[part][2] = s.q[part][ra][(word + 4) ^ swa]; a[part][3] = s.q[part][rb][(word + 4) ^ swb];
                }
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int c = nt * 8 + gid, d = g * 32 + 4 * tig;
                    const auto ptr = [&](int dim) {
                        return reinterpret_cast<const std::uint8_t *>(&s.k[0][0]) + c * HD +
                               ((dim / 16) ^ (c & SWIZZLE)) * 16 + (dim & 15);
                    };
                    std::uint32_t b[2] = {*reinterpret_cast<const std::uint32_t *>(ptr(d)),
                                          *reinterpret_cast<const std::uint32_t *>(ptr(d + 16))};
                    float h[4] = {}, l[4] = {}, t[4] = {};
                    mma(h, a[0], b); mma(l, a[1], b); mma(t, a[2], b);
                    const int sc = nt * 8 + 2 * tig;
                    const float k0 = __half2float(__ushort_as_half(s.ks[sc][g]));
                    const float k1 = __half2float(__ushort_as_half(s.ks[sc + 1][g]));
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float dot = fmaf(t[j], GEMMA_TAIL_DOWN, fmaf(l[j], GEMMA_RESIDUAL_DOWN, h[j]));
                        score[nt][j] = fmaf(dot, (j & 1) ? k1 : k0, score[nt][j]);
                    }
                }
            }
            float xa = -INFINITY, xb = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const std::int64_t cell = c0 + nt * 8 + 2 * tig;
                score[nt][0] = qa < rows && cell >= loa && cell < hia ? score[nt][0] * da : -INFINITY;
                score[nt][1] = qa < rows && cell + 1 >= loa && cell + 1 < hia ? score[nt][1] * da : -INFINITY;
                score[nt][2] = qb < rows && cell >= lob && cell < hib ? score[nt][2] * db : -INFINITY;
                score[nt][3] = qb < rows && cell + 1 >= lob && cell + 1 < hib ? score[nt][3] * db : -INFINITY;
                xa = fmaxf(xa, fmaxf(score[nt][0], score[nt][1]));
                xb = fmaxf(xb, fmaxf(score[nt][2], score[nt][3]));
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o));
                xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o));
            }
            const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
            const float aa = ma == -INFINITY ? 0.f : exp2f(ma - na);
            const float ab = mb == -INFINITY ? 0.f : exp2f(mb - nb);
            float sa = 0.f, sb = 0.f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    score[nt][j] = score[nt][j] == -INFINITY ? 0.f : exp2f(score[nt][j] - (j < 2 ? na : nb));
                const int c = nt * 8 + 2 * tig;
                s.probability[ra][c] = score[nt][0]; s.probability[ra][c + 1] = score[nt][1];
                s.probability[rb][c] = score[nt][2]; s.probability[rb][c + 1] = score[nt][3];
                sa += score[nt][0] + score[nt][1]; sb += score[nt][2] + score[nt][3];
            }
            if (!tig) { s.alpha[ra] = aa; s.alpha[rb] = ab; }
            la = fmaf(la, aa, sa); lb = fmaf(lb, ab, sb); ma = na; mb = nb;
        }
        __syncthreads(); // QK is consumed; probabilities/online rescale are published
        if (more) { issue(false, c0 + CELLS); commit(); }
        if (more) wait<1>(); else wait<0>();
        __syncthreads();
        const float aa = s.alpha[ra], ab = s.alpha[rb];
#pragma unroll
        for (int g = 0; g < OUT / 32; ++g) {
            const int vg = out_dim / 32 + g;
            float wma = 0.f, wmb = 0.f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = nt * 8 + 2 * tig;
                const float v0 = __half2float(__ushort_as_half(s.vs[c][vg]));
                const float v1 = __half2float(__ushort_as_half(s.vs[c + 1][vg]));
                wma = fmaxf(wma, fmaxf(s.probability[ra][c] * v0, s.probability[ra][c + 1] * v1));
                wmb = fmaxf(wmb, fmaxf(s.probability[rb][c] * v0, s.probability[rb][c + 1] * v1));
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                wma = fmaxf(wma, __shfl_xor_sync(0xffffffffu, wma, o));
                wmb = fmaxf(wmb, __shfl_xor_sync(0xffffffffu, wmb, o));
            }
            int ea = 0, eb = 0;
            if (wma > 0.f) frexpf(wma, &ea);
            if (wmb > 0.f) frexpf(wmb, &eb);
            const int ua = wma > 0.f ? 8 - ea : 0, ub = wmb > 0.f ? 8 - eb : 0;
            std::uint32_t ph[4] = {}, pl[4] = {}, pt[4] = {};
#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = (reg / 2) * 16 + 4 * tig + j;
                    const int row = gid + (reg & 1) * 8;
                    const float prob = s.probability[row][c];
                    const float vs = __half2float(__ushort_as_half(s.vs[c][vg]));
                    const float weighted = prob == 0.f ? 0.f : prob * vs;
                    const float x = ldexpf(weighted, (reg & 1) ? ub : ua);
                    std::uint8_t h, l, t;
                    gemma_encode3(x, h, l, t);
                    ph[reg] |= std::uint32_t(h) << (j * 8);
                    pl[reg] |= std::uint32_t(l) << (j * 8);
                    pt[reg] |= std::uint32_t(t) << (j * 8);
                }
            }
            const auto *v = reinterpret_cast<const std::uint8_t *>(&s.v[0][0]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dim = out_dim + g * 32 + j * 8 + gid;
                const auto value = [&](int c) {
                    return std::uint32_t(v[c * HD + ((dim / 16) ^ (c & SWIZZLE)) * 16 + (dim & 15)]);
                };
                std::uint32_t b[2] = {};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    b[0] |= value(4 * tig + k) << (k * 8);
                    b[1] |= value(16 + 4 * tig + k) << (k * 8);
                }
                float h[4] = {}, l[4] = {}, t[4] = {};
                mma(h, ph, b); mma(l, pl, b); mma(t, pt, b);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float dot = fmaf(t[k], GEMMA_TAIL_DOWN, fmaf(l[k], GEMMA_RESIDUAL_DOWN, h[k]));
                    const float tile = ldexpf(dot, k < 2 ? -ua : -ub);
                    acc[g * 4 + j][k] = fmaf(acc[g * 4 + j][k], k < 2 ? aa : ab, tile);
                }
            }
        }
        __syncthreads();
        if (more) { issue(true, c0 + CELLS); commit(); }
    }
    // Only the producer warp owns the denominator and split maximum. Publish
    // those to the output warps after the complete online reduction.
    if (!warp) {
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            la += __shfl_xor_sync(0xffffffffu, la, o); lb += __shfl_xor_sync(0xffffffffu, lb, o);
        }
        if (!tig) {
            s.alpha[ra] = la; s.alpha[rb] = lb;
            if constexpr (SPLIT) {
                if (qa < rows) {
                    float *p0 = partial + ((std::size_t(qa) * heads + ha) * SPLITS + split) * (HD + 2);
                    p0[HD] = ma; p0[HD + 1] = la;
                }
                if (qb < rows) {
                    float *p0 = partial + ((std::size_t(qb) * heads + hb) * SPLITS + split) * (HD + 2);
                    p0[HD] = mb; p0[HD + 1] = lb;
                }
            }
        }
    }
    __syncthreads();
    const std::size_t ia = std::size_t(qa) * heads + ha, ib = std::size_t(qb) * heads + hb;
    float *oa = SPLIT ? partial + (ia * SPLITS + split) * (HD + 2) : out + ia * HD;
    float *ob = SPLIT ? partial + (ib * SPLITS + split) * (HD + 2) : out + ib * HD;
    const float inva = SPLIT ? 1.f : (s.alpha[ra] > 0.f ? 1.f / s.alpha[ra] : 0.f);
    const float invb = SPLIT ? 1.f : (s.alpha[rb] > 0.f ? 1.f / s.alpha[rb] : 0.f);
#pragma unroll
    for (int j = 0; j < OUT / 8; ++j) {
        const int d = out_dim + j * 8 + 2 * tig;
        if (qa < rows) *reinterpret_cast<float2 *>(oa + d) = make_float2(acc[j][0] * inva, acc[j][1] * inva);
        if (qb < rows) *reinterpret_cast<float2 *>(ob + d) = make_float2(acc[j][2] * invb, acc[j][3] * invb);
    }
}

// Gemma SWA decode: one query with two heads per KV head. Four independent
// 64-value output tiles trade repeated QK for a smaller live accumulator and a
// single warp. All three compensated FP8 components and the split partition are
// identical to gemma_attention_kernel; only absent query/head rows are elided.
struct SharedGemmaSwaOne {
    static constexpr int HD = 256, OUT = 64, GQA = 2;
    std::uint32_t q[3][GQA][HD / 4];
    uint4 k[CELLS][HD / 16], v[CELLS][OUT / 16];
    __align__(16) std::uint16_t ks[CELLS][HD / 32];
    __align__(16) std::uint16_t vs[CELLS][OUT / 32];
    float probability[GQA][CELLS + 1], qdown[GQA];
};

__global__ void __launch_bounds__(32, 1)
gemma_swa_one_kernel(const float *__restrict__ q, Cache p, std::int64_t pos,
                     const std::int64_t *__restrict__ at, int heads, int kv_heads,
                     int ring, int window, float scale_log2, float *__restrict__ partial) {
    constexpr int HD = SharedGemmaSwaOne::HD, OUT = SharedGemmaSwaOne::OUT;
    constexpr int GQA = SharedGemmaSwaOne::GQA, WORDS = HD / 16, VWORDS = OUT / 16;
    constexpr int KSW = 7, VSW = VWORDS - 1;
    extern __shared__ __align__(16) unsigned char raw[];
    auto &s = *reinterpret_cast<SharedGemmaSwaOne *>(raw);
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    const int kvh = blockIdx.y, slice = blockIdx.z % (HD / OUT);
    const int split = blockIdx.z / (HD / OUT), out_dim = slice * OUT;
    const bool live = gid < GQA;
    const std::int64_t base = at ? at[0] : pos;
    const std::int64_t span_lo = base >= window ? base - window + 1 : 0;
    const std::int64_t span_hi = base + 1;
    const int span = int(span_hi - span_lo), active = gemma_split_count(span, kv_heads);
    if (split >= active) return;
    const int chunks = (span + CELLS - 1) / CELLS;
    const std::int64_t begin = span_lo + (split * chunks / active) * CELLS;
    const std::int64_t end = min(span_hi, span_lo + ((split + 1) * chunks / active) * CELLS);
    const auto cache_row = [&](std::int64_t cell) -> std::size_t {
        return std::size_t(ring ? cell % ring : cell) * kv_heads + kvh;
    };
    const auto issue = [&](bool value, std::int64_t c0) {
        if (!value) {
            for (int i = lane; i < CELLS * WORDS; i += 32) {
                const int c = i / WORDS, pc = i % WORDS;
                const bool ok = c0 + c < end;
                const std::size_t off = ok ? cache_row(c0 + c) * HD + pc * 16 : 0;
                copy_async<16>(&s.k[c][pc ^ (c & KSW)], p.k + off, ok);
            }
            const bool ok = c0 + lane < end;
            const std::size_t off = ok ? cache_row(c0 + lane) * (HD / 32) : 0;
            copy_async<16>(s.ks[lane], p.ks + off, ok);
        } else {
            for (int i = lane; i < CELLS * VWORDS; i += 32) {
                const int c = i / VWORDS, pc = i % VWORDS;
                const bool ok = c0 + c < end;
                const std::size_t off = ok ? cache_row(c0 + c) * HD + out_dim + pc * 16 : 0;
                copy_async<16>(&s.v[c][pc ^ (c & VSW)], p.v + off, ok);
            }
            const bool ok = c0 + lane < end;
            const std::size_t off = ok ? cache_row(c0 + lane) * (HD / 32) + out_dim / 32 : 0;
            copy_async<4>(s.vs[lane], p.vs + off, ok);
        }
    };
    issue(false, begin); commit();
    issue(true, begin); commit();
#pragma unroll 1
    for (int row = 0; row < GQA; ++row) {
        const float *src = q + std::size_t(kvh * GQA + row) * HD;
        float m = 0.f;
#pragma unroll
        for (int d = lane; d < HD; d += 32) m = fmaxf(m, fabsf(src[d]));
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        int e = 0;
        if (m > 0.f) frexpf(m, &e);
        const int up = 8 - e;
        for (int word = lane; word < HD / 4; word += 32) {
            std::uint32_t hi = 0, lo = 0, tail = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                std::uint8_t h, l, t;
                gemma_encode3(ldexpf(src[word * 4 + j], up), h, l, t);
                hi |= std::uint32_t(h) << (8 * j);
                lo |= std::uint32_t(l) << (8 * j);
                tail |= std::uint32_t(t) << (8 * j);
            }
            const int sw = word ^ ((row & KSW) * 4);
            s.q[0][row][sw] = hi; s.q[1][row][sw] = lo; s.q[2][row][sw] = tail;
        }
        if (!lane) s.qdown[row] = ldexpf(scale_log2, e - 8);
    }
    __syncwarp();
    const float down = live ? s.qdown[gid] : 0.f;
    float acc[OUT / 8][2] = {};
    float maximum = -INFINITY, denominator = 0.f;
#pragma unroll 1
    for (std::int64_t c0 = begin; c0 < end; c0 += CELLS) {
        const bool more = c0 + CELLS < end;
        wait<1>(); __syncwarp();
        float score[4][2] = {};
#pragma unroll 1
        for (int g = 0; g < HD / 32; ++g) {
            const int word = g * 8 + tig, sw = (gid & KSW) * 4;
            std::uint32_t a[3][4] = {};
#pragma unroll
            for (int part = 0; part < 3; ++part) {
                if (live) {
                    a[part][0] = s.q[part][gid][word ^ sw];
                    a[part][2] = s.q[part][gid][(word + 4) ^ sw];
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = nt * 8 + gid, d = g * 32 + 4 * tig;
                const auto ptr = [&](int dim) {
                    return reinterpret_cast<const std::uint8_t *>(&s.k[0][0]) + c * HD +
                           ((dim / 16) ^ (c & KSW)) * 16 + (dim & 15);
                };
                std::uint32_t b[2] = {*reinterpret_cast<const std::uint32_t *>(ptr(d)),
                                      *reinterpret_cast<const std::uint32_t *>(ptr(d + 16))};
                float h[4] = {}, l[4] = {}, t[4] = {};
                mma(h, a[0], b); mma(l, a[1], b); mma(t, a[2], b);
                const int sc = nt * 8 + 2 * tig;
                const float k0 = __half2float(__ushort_as_half(s.ks[sc][g]));
                const float k1 = __half2float(__ushort_as_half(s.ks[sc + 1][g]));
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float dot = fmaf(t[j], GEMMA_TAIL_DOWN, fmaf(l[j], GEMMA_RESIDUAL_DOWN, h[j]));
                    score[nt][j] = fmaf(dot, j ? k1 : k0, score[nt][j]);
                }
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const std::int64_t cell = c0 + nt * 8 + 2 * tig;
            score[nt][0] = live && cell >= span_lo && cell < span_hi ? score[nt][0] * down : -INFINITY;
            score[nt][1] = live && cell + 1 >= span_lo && cell + 1 < span_hi ? score[nt][1] * down : -INFINITY;
            mx = fmaxf(mx, fmaxf(score[nt][0], score[nt][1]));
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float next = fmaxf(maximum, mx);
        const float alpha = maximum == -INFINITY ? 0.f : exp2f(maximum - next);
        float total = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            score[nt][0] = score[nt][0] == -INFINITY ? 0.f : exp2f(score[nt][0] - next);
            score[nt][1] = score[nt][1] == -INFINITY ? 0.f : exp2f(score[nt][1] - next);
            if (live) {
                const int c = nt * 8 + 2 * tig;
                s.probability[gid][c] = score[nt][0]; s.probability[gid][c + 1] = score[nt][1];
            }
            total += score[nt][0] + score[nt][1];
        }
        denominator = fmaf(denominator, alpha, total); maximum = next;
        __syncwarp();
        if (more) { issue(false, c0 + CELLS); commit(); }
        if (more) wait<1>(); else wait<0>();
        __syncwarp();
#pragma unroll
        for (int g = 0; g < OUT / 32; ++g) {
            float wm = 0.f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = nt * 8 + 2 * tig;
                const float v0 = __half2float(__ushort_as_half(s.vs[c][g]));
                const float v1 = __half2float(__ushort_as_half(s.vs[c + 1][g]));
                const float p0 = live ? s.probability[gid][c] : 0.f;
                const float p1 = live ? s.probability[gid][c + 1] : 0.f;
                wm = fmaxf(wm, fmaxf(p0 * v0, p1 * v1));
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, o));
            int e = 0;
            if (wm > 0.f) frexpf(wm, &e);
            const int up = wm > 0.f ? 8 - e : 0;
            std::uint32_t ph[4] = {}, pl[4] = {}, pt[4] = {};
#pragma unroll
            for (int reg = 0; reg < 4; reg += 2) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = (reg / 2) * 16 + 4 * tig + j;
                    const float prob = live ? s.probability[gid][c] : 0.f;
                    const float vs = __half2float(__ushort_as_half(s.vs[c][g]));
                    const float weighted = prob == 0.f ? 0.f : prob * vs;
                    std::uint8_t h, l, t;
                    gemma_encode3(ldexpf(weighted, up), h, l, t);
                    ph[reg] |= std::uint32_t(h) << (j * 8);
                    pl[reg] |= std::uint32_t(l) << (j * 8);
                    pt[reg] |= std::uint32_t(t) << (j * 8);
                }
            }
            const auto *v = reinterpret_cast<const std::uint8_t *>(&s.v[0][0]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dim = g * 32 + j * 8 + gid;
                const auto value = [&](int c) {
                    return std::uint32_t(v[c * OUT + ((dim / 16) ^ (c & VSW)) * 16 + (dim & 15)]);
                };
                std::uint32_t b[2] = {};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    b[0] |= value(4 * tig + k) << (k * 8);
                    b[1] |= value(16 + 4 * tig + k) << (k * 8);
                }
                float h[4] = {}, l[4] = {}, t[4] = {};
                mma(h, ph, b); mma(l, pl, b); mma(t, pt, b);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float dot = fmaf(t[k], GEMMA_TAIL_DOWN, fmaf(l[k], GEMMA_RESIDUAL_DOWN, h[k]));
                    acc[g * 4 + j][k] = fmaf(acc[g * 4 + j][k], alpha, ldexpf(dot, -up));
                }
            }
        }
        __syncwarp();
        if (more) { issue(true, c0 + CELLS); commit(); }
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) denominator += __shfl_xor_sync(0xffffffffu, denominator, o);
    if (live) {
        float *dst = partial + ((std::size_t(kvh * GQA + gid) * SPLITS + split) * (HD + 2));
        if (!slice && !tig) { dst[HD] = maximum; dst[HD + 1] = denominator; }
#pragma unroll
        for (int j = 0; j < OUT / 8; ++j)
            *reinterpret_cast<float2 *>(dst + out_dim + j * 8 + 2 * tig) = make_float2(acc[j][0], acc[j][1]);
    }
}

template<int HD>
__global__ void gemma_merge_kernel(const float *__restrict__ part, float *__restrict__ out,
                                   std::int64_t pos, const std::int64_t *__restrict__ at,
                                   int rows, int heads, int kv_heads, int gqa, int window, bool whole) {
    const int lane = threadIdx.x & 31, qi = blockIdx.x / heads;
    const int q0 = (qi / (16 / gqa)) * (16 / gqa);
    const std::int64_t base = at ? at[0] : pos;
    const std::int64_t first_pos = base + q0;
    const std::int64_t lo = window > 0 && first_pos >= window ? first_pos - window + 1 : 0;
    const int span = int(base + (whole ? rows : min(rows, q0 + 16 / gqa)) - lo);
    const int active = gemma_split_count(span, kv_heads);
    const float *p = part + std::size_t(blockIdx.x) * SPLITS * (HD + 2);
    float m = lane < active ? p[lane * (HD + 2) + HD] : -INFINITY;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float l = lane < active ? p[lane * (HD + 2) + HD + 1] : 0.f;
    const float factor = l > 0.f ? exp2f(p[lane * (HD + 2) + HD] - m) : 0.f;
    float denom = factor * l;
#pragma unroll
    for (int o = 16; o; o >>= 1) denom += __shfl_xor_sync(0xffffffffu, denom, o);
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float sum = 0.f;
#pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float f = __shfl_sync(0xffffffffu, factor, s);
            if (s < active) sum = fmaf(p[s * (HD + 2) + d], f, sum);
        }
        out[std::size_t(blockIdx.x) * HD + d] = denom > 0.f ? sum / denom : 0.f;
    }
}

template<int HD, bool SPLIT> bool prepare_gemma_kernel() {
    cudaFuncAttributes a{};
    auto k = gemma_attention_kernel<HD, SPLIT>;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess || a.ptxVersion < 89) return false;
    return cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedGemma<HD>)) == cudaSuccess;
}

template<int HD>
void launch_gemma(const float *q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads,
                  float scale, float *out, const std::int64_t *at, int ring, int window,
                  bool whole, float *scratch, bool split, cudaStream_t stream) {
    const int gqa = heads / kv_heads, tile_rows = 16 / gqa;
    if constexpr (HD == SharedGemmaSwaOne::HD) {
        if (rows == 1 && gqa == SharedGemmaSwaOne::GQA && window > 0 && split) {
            const int limit = std::min(SPLITS, (GEMMA_TARGET_CTAS + kv_heads - 1) / kv_heads);
            gemma_swa_one_kernel<<<dim3(1, kv_heads, limit * (HD / SharedGemmaSwaOne::OUT)),
                                  32, sizeof(SharedGemmaSwaOne), stream>>>(
                q, cache, pos, at, heads, kv_heads, ring, window, scale * LOG2E, scratch);
            gemma_merge_kernel<HD><<<heads, 128, 0, stream>>>(scratch, out, pos, at, 1, heads, kv_heads,
                                                            gqa, window, whole);
            return;
        }
    }
    const dim3 grid((rows + tile_rows - 1) / tile_rows, kv_heads, split ? SPLITS : 1);
    if (split) {
        gemma_attention_kernel<HD, true><<<grid, (HD / 128) * 32, sizeof(SharedGemma<HD>), stream>>>(
            q, cache, pos, at, rows, heads, kv_heads, gqa, ring, window, whole, scale * LOG2E, out, scratch);
        gemma_merge_kernel<HD><<<rows * heads, 128, 0, stream>>>(scratch, out, pos, at, rows, heads, kv_heads, gqa, window, whole);
    } else {
        gemma_attention_kernel<HD, false><<<grid, (HD / 128) * 32, sizeof(SharedGemma<HD>), stream>>>(
            q, cache, pos, at, rows, heads, kv_heads, gqa, ring, window, whole, scale * LOG2E, out, nullptr);
    }
}
template<int HD>
void launch(const float *q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads, float scale,
            float *out, const std::int64_t *at, int ring, int window, bool whole, float *scratch,
            bool split, cudaStream_t stream) {
    const int gqa = heads / kv_heads;
    constexpr int OUT = HD == 64 ? 64 : 128;
    if (split) {
        const int qtile = 16 / gqa;
        attention_kernel<HD, 1, true><<<dim3((rows + qtile - 1) / qtile, kv_heads, SPLITS * (HD / OUT)), 32,
            sizeof(Shared<HD, 1>), stream>>>(q, cache, pos, at, rows, heads, kv_heads, gqa, ring, window,
                                            whole, scale * LOG2E, out, scratch);
        merge_kernel<HD><<<rows * heads, 128, 0, stream>>>(scratch, out);
    } else {
        const int qtile = 32 / gqa;
        attention_kernel<HD, 2, false><<<dim3((rows + qtile - 1) / qtile, kv_heads, HD / OUT), 64,
            sizeof(Shared<HD, 2>), stream>>>(q, cache, pos, at, rows, heads, kv_heads, gqa, ring, window,
                                            whole, scale * LOG2E, out, nullptr);
    }
}
} // namespace

bool available(int head_dim) {
    if (head_dim && head_dim != 64 && head_dim != 256 && head_dim != 512) return false;
    int device = -1;
    cudaDeviceProp p{};
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (prepared_device == device) return prepared;
    prepared_device = device;
    prepared = false;
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess || p.major * 10 + p.minor < 89) return false;
    int *probe = nullptr, target = 0;
    if (cudaMalloc(&probe, sizeof(int)) != cudaSuccess) { cudaGetLastError(); return false; }
    compiled_target_kernel<<<1, 1>>>(probe);
    const auto status = cudaMemcpy(&target, probe, sizeof(target), cudaMemcpyDeviceToHost);
    cudaFree(probe);
    if (status != cudaSuccess || target < 890) { cudaGetLastError(); return false; }
    prepared = prepare_kernel<64, 2, false>() && prepare_kernel<64, 1, true>() &&
               prepare_kernel<256, 2, false>() && prepare_kernel<256, 1, true>() &&
               prepare_kernel<512, 2, false>() && prepare_kernel<512, 1, true>();
    if (!prepared) cudaGetLastError();
    return prepared;
}
bool available_gemma() {
    if (!available()) return false;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (gemma_prepared_device == device) return gemma_prepared;
    gemma_prepared_device = device;
    gemma_prepared = prepare_gemma_kernel<256, false>() && prepare_gemma_kernel<256, true>() &&
                     prepare_gemma_kernel<512, false>() && prepare_gemma_kernel<512, true>();
    if (!gemma_prepared) cudaGetLastError();
    return gemma_prepared;
}
void store(const float *src, std::uint8_t *codes, std::uint16_t *scales, int width,
           std::int64_t pos, int rows, const std::int64_t *at, int ring, void *stream) {
    if (width < 32 || width % 32 || rows < 0 || ring < 0) throw std::invalid_argument("FP8 KV store: invalid shape");
    if (!rows) return;
    store_kernel<<<dim3((width / 32 + 3) / 4, rows), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        src, codes, scales, width, pos, at, ring);
}
std::size_t scratch_bytes(int rows, int q_heads, int head_dim) {
    if (rows <= 0 || q_heads <= 0 || head_dim <= 0) return 0;
    return std::size_t(std::min(rows, 8)) * q_heads * (head_dim + 2) * SPLITS * sizeof(float);
}
void run(const float *q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads,
         int hd, int context, float scale, float *out, const std::int64_t *at, int ring,
         int window, bool whole, void *scratch, std::size_t size, void *stream) {
    // available() is a load-time precondition, deliberately not queried here:
    // this function can run from a different hybrid worker or during capture.
    if (rows < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || context <= 0 || ring < 0 || window < 0)
        throw std::invalid_argument("Native FP8 attention: invalid shape");
    const int gqa = heads / kv_heads;
    if (gqa > 16 || 16 % gqa || (hd != 64 && hd != 256 && hd != 512))
        throw std::invalid_argument("Native FP8 attention: unsupported head dimension or GQA ratio");
    if (!rows) return;
    const bool split = rows <= 8 && scratch && size >= scratch_bytes(rows, heads, hd);
    const auto call = [&]<int HD>() {
        launch<HD>(q, cache, pos, rows, heads, kv_heads, scale, out, at, ring, window, whole,
                   static_cast<float *>(scratch), split, static_cast<cudaStream_t>(stream));
    };
    if (hd == 64) call.template operator()<64>();
    else if (hd == 256) call.template operator()<256>();
    else call.template operator()<512>();
}
void run_gemma(const float *q, Cache cache, std::int64_t pos, int rows, int heads, int kv_heads,
               int hd, int context, float scale, float *out, const std::int64_t *at, int ring,
               int window, bool whole, void *scratch, std::size_t size, void *stream) {
    if (rows < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || context <= 0 || ring < 0 || window < 0)
        throw std::invalid_argument("Gemma native FP8 attention: invalid shape");
    const int gqa = heads / kv_heads;
    if (gqa > 16 || 16 % gqa || (hd != 256 && hd != 512))
        throw std::invalid_argument("Gemma native FP8 attention: unsupported head dimension or GQA ratio");
    if (!rows) return;
    const bool split = rows <= 8 && scratch && size >= scratch_bytes(rows, heads, hd);
    const auto call = [&]<int HD>() {
        launch_gemma<HD>(q, cache, pos, rows, heads, kv_heads, scale, out, at, ring, window, whole,
                         static_cast<float *>(scratch), split, static_cast<cudaStream_t>(stream));
    };
    if (hd == 256) call.template operator()<256>();
    else call.template operator()<512>();
}
} // namespace strata::fp8_attention
