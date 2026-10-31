// Dense INT8 prompt attention adapted from aflin/momoe (MIT).
// Source: src/kernels/cuda/qsa_prompt_attn.cu at 47e5b2b2ab52982e105315bba2ac48ec75de01bf.
// Dense kernel introduced by Aaron Flin in e5fd6d3acbe99f26ed7f179b6b35b8ec5343fb38.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// Adaptations: resident head-major KV, device base position instead of a step table,
// INT8-only entry point, no sparse attention, Q4, Metal or other model engine.
// Precision changes: masked row-wise P*V-scale normalization, amplified
// FP16 residuals with separate MMA accumulators, and unscaled FP32 outer sums.
#include "strata/qwen35/int8_attention.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::qwen35::int8_attention {
namespace {
constexpr int HD = 256, THREADS = 128, KV_Q8_GROUP = 64;
// Q and weighted P are scaled below 2^14 before splitting. Amplifying their
// FP16 rounding residual by 2^11 keeps it representable without overflowing.
constexpr float FP16_RESIDUAL_UP = 2048.0f;
constexpr float FP16_RESIDUAL_DOWN = 1.0f / FP16_RESIDUAL_UP;
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif

// m16n8k16 with f16 inputs needs sm_80.  Turing (sm_75) has m16n8k8 with the SAME A/B/C register mapping, so the
// k=16 step is two k=8 steps on the fragments as they are already laid out: a[0]/a[1] are rows gid/gid+8 at k columns
// 2*tig..2*tig+1 (b[0]'s k rows), a[2]/a[3] the same rows at k columns 2*tig+8..2*tig+9 (b[1]'s k rows).  The
// products then add into the same FP32 C registers in the order hi-part-0, hi-part-1, which is the order the k16
// instruction accumulates in as well - but the sum now rounds twice, so the two paths do not agree bit for bit.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80 && (defined(__HIPCC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750)
    __trap();   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif !STRATA_PA_SM80
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

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    __trap();
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_PA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

constexpr int DQ = 8;     // queries per block
constexpr int DCH = 32;   // cells per chunk

template <int KV_MODE>   // 0: fp16 K and V; 1: int8 codes + an fp16 scale per 64 values
struct SmemD {
    static constexpr int PIECES = HD * (KV_MODE == 0 ? 2 : 1) / 16;   // 16-byte pieces per K/V row
    uint4 qh[DQ * 8][HD / 8];   // 64 rows; a row's pieces XOR-swizzled by row & 7 (conflict-free ldmatrix)
    uint4 ql[DQ * 8][HD / 8];
    uint4 k[DCH][PIECES];       // a cell's pieces XOR-swizzled by cell & 7
    uint4 v[DCH][PIECES];
    uint2 ksc[DCH];             // int8: the cell's 4 fp16 scales
    uint2 vsc[DCH];
    float qdn[DQ * 8];          // per row: the q prescale taken back out, times log2(e) / 16
};

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
#if STRATA_PA_SM80
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
    (void) r; (void) p;
    __trap();
#endif
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
#if STRATA_PA_SM80
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
    (void) r; (void) p;
    __trap();
#endif
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    (void) smem; (void) gmem; (void) valid;
    __trap();
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 8 : 0));
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// Reconstruct as hi + lo * FP16_RESIDUAL_DOWN. Keep the low component
// amplified through its own MMA chain; do not mix it into the high chain.
// Callers scale finite inputs to |x| < 2^14 before using this helper.
__device__ __forceinline__ void split_h2(float a, float b, uint32_t& hi, uint32_t& lo) {
    hi = pack_h2(a, b);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hi));
    lo = pack_h2(__fmul_rn(__fsub_rn(a, f.x), FP16_RESIDUAL_UP),
                 __fmul_rn(__fsub_rn(b, f.y), FP16_RESIDUAL_UP));
}

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS, 1) prompt_attn_dense_kernel(const float* __restrict__ q, Cache p,
                                                                       const int32_t* __restrict__ pos0, int n_q,
                                                                       int n_kv_heads, int page_size, float scale_log2,
                                                                       float* __restrict__ attn, int visible_end) {
    constexpr int GQ = 8, PIECES = SmemD<KV_MODE>::PIECES, ROWB = PIECES * 16;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    SmemD<KV_MODE>& S = *reinterpret_cast<SmemD<KV_MODE>*>(smem_raw);
    const int q0 = blockIdx.x * DQ, kvh = blockIdx.y, n_head = n_kv_heads * GQ;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int rbase = warp * 16;   // this warp's MMA rows: query q0 + 2w (rows 0-7), q0 + 2w + 1 (rows 8-15), by head
    auto width = [&](int i) {
        return q0 + i < n_q ? (visible_end > 0 ? visible_end : __ldg(pos0) + q0 + i + 1) : 0;
    };
    const int na = width(2 * warp), nb = width(2 * warp + 1);
    int n_max = 0;
#pragma unroll
    for (int i = 0; i < DQ; ++i) n_max = max(n_max, width(i));

    auto row_of = [&](int cell) -> long long {
        const long long page = 0; // TinyStrata: one head-major resident span, page_size == context capacity
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    // the K (kv 0) or V (kv 1) rows of the chunk at cell c0, cells past the widest query zero-filled
    auto issue = [&](int kv, int c0) {
        const unsigned char* base =
            KV_MODE == 0 ? reinterpret_cast<const unsigned char*>(kv ? p.v_pool : p.k_pool)
                         : reinterpret_cast<const unsigned char*>(kv ? p.v_q : p.k_q);
        uint4(*dst)[PIECES] = kv ? S.v : S.k;
#pragma unroll
        for (int i = t; i < DCH * PIECES; i += THREADS) {
            const int c = i / PIECES, pc = i % PIECES, cell = c0 + c;
            const bool ok = cell < n_max;
            const size_t off = ok ? (size_t) row_of(cell) * ROWB + pc * 16 : 0;
            cp_async16(&dst[c][pc ^ (c & 7)], base + off, ok);
        }
        if constexpr (KV_MODE == 1) {
            if (t < DCH) {
                const int cell = c0 + t;
                const bool ok = cell < n_max;
                const size_t off = ok ? (size_t) row_of(cell) * (HD / KV_Q8_GROUP) : 0;
                cp_async8(kv ? &S.vsc[t] : &S.ksc[t], (kv ? p.v_scale : p.k_scale) + off, ok);
            }
        }
    };
    if (n_max > 0) {
        issue(0, 0);
        cp_async_commit();
        issue(1, 0);
        cp_async_commit();
    }

    // Q: row-wise power-of-two scaling below 2^14, then FP16 high and
    // amplified residual components. Scale each value, not a standalone 2^e.
#pragma unroll 1
    for (int r = 0; r < 16; ++r) {
        const int qi = q0 + 2 * warp + (r >> 3), row = rbase + r;
        float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (qi < n_q) {
            const float4* src = reinterpret_cast<const float4*>(q + ((size_t) qi * n_head + kvh * GQ + (r & 7)) * HD) + lane * 2;
            const float4 a = src[0], b = src[1];
            x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        }
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(x[j]));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        int e = 0;
        if (m > 0.0f) frexpf(m, &e);   // m < 2^e
        const int q_up = 14 - e;
        uint4 hi, lo;
        split_h2(ldexpf(x[0], q_up), ldexpf(x[1], q_up), hi.x, lo.x);
        split_h2(ldexpf(x[2], q_up), ldexpf(x[3], q_up), hi.y, lo.y);
        split_h2(ldexpf(x[4], q_up), ldexpf(x[5], q_up), hi.z, lo.z);
        split_h2(ldexpf(x[6], q_up), ldexpf(x[7], q_up), hi.w, lo.w);
        S.qh[row][lane ^ (row & 7)] = hi;
        S.ql[row][lane ^ (row & 7)] = lo;
        if (lane == 0) S.qdn[row] = ldexpf(scale_log2, e - 14);
    }
    __syncwarp();
    const float da = S.qdn[rbase + gid], db = S.qdn[rbase + 8 + gid];

    float acc[HD / 8][4];
#pragma unroll
    for (int j = 0; j < HD / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.0f, l_b = 0.0f;   // l: this thread's columns only (summed at the end)
    // acc is the online-softmax numerator in real V units. Tile-local
    // power-of-two scales are removed before every outer FP32 update.

#pragma unroll 1
    for (int c0 = 0; c0 < n_max; c0 += DCH) {
        const bool more = c0 + DCH < n_max;
        cp_async_wait1();
        __syncthreads();   // K in
        // q.k: 4 n-tiles of 8 cells x 16 k-steps (hi and lo); int8 per 64-dim group, then times the cell's K scale
        float s[DCH / 8][4];
#pragma unroll
        for (int nt = 0; nt < DCH / 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            // Accumulate Q_hi*K and Q_lo*K independently; only combine
            // after the whole 64-dimension group has been evaluated.
            float th[DCH / 8][4] = {}, tl[DCH / 8][4] = {};
#pragma unroll
            for (int kq = 0; kq < 4; ++kq) {
                const int kk = g * 4 + kq;   // 16-dim step
                uint32_t ah[4], al[4];
                {
                    const int mi = lane >> 3, rr = rbase + (mi & 1) * 8 + (lane & 7), pc = kk * 2 + (mi >> 1);
                    ldsm_x4(ah, &S.qh[rr][pc ^ (rr & 7)]);
                    ldsm_x4(al, &S.ql[rr][pc ^ (rr & 7)]);
                }
                if constexpr (KV_MODE == 0) {
#pragma unroll
                    for (int np = 0; np < DCH / 16; ++np) {   // two n-tiles per ldmatrix
                        const int c = 16 * np + 8 * (lane >> 4) + (lane & 7), pc = kk * 2 + ((lane >> 3) & 1);
                        uint32_t b[4];
                        ldsm_x4(b, &S.k[c][pc ^ (c & 7)]);
                        mma16816(th[2 * np], ah, b);
                        mma16816(tl[2 * np], al, b);
                        mma16816(th[2 * np + 1], ah, b + 2);
                        mma16816(tl[2 * np + 1], al, b + 2);
                    }
                } else {
                    const unsigned char* K = reinterpret_cast<const unsigned char*>(&S.k[0][0]);
#pragma unroll
                    for (int nt = 0; nt < DCH / 8; ++nt) {
                        const int c = nt * 8 + gid;
                        const unsigned char* rowp = K + c * ROWB + ((kk ^ (c & 7)) << 4) + 2 * tig;
                        uint32_t b[2];
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(rowp));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(rowp + 8));
                        mma16816(th[nt], ah, b);
                        mma16816(tl[nt], al, b);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < DCH / 8; ++nt) {
                float k0 = 1.0f, k1 = 1.0f;
                if constexpr (KV_MODE == 1) {
                    const int c = nt * 8 + 2 * tig;
                    k0 = __half2float(reinterpret_cast<const __half*>(&S.ksc[c])[g]);
                    k1 = __half2float(reinterpret_cast<const __half*>(&S.ksc[c + 1])[g]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float dot = fmaf(tl[nt][j], FP16_RESIDUAL_DOWN, th[nt][j]);
                    s[nt][j] = fmaf(dot, (j & 1) ? k1 : k0, s[nt][j]);
                }
            }
        }
        __syncthreads();   // every warp is done with K
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
        if (more) cp_async_wait1(); else cp_async_wait0();
        __syncthreads();   // V in
        // P*V: form the actual P*V_scale operand first, after causal masking.
        // Normalize each query/head row independently, then return each tile
        // numerator to real V units before the online-softmax update.
#pragma unroll
        for (int g = 0; g < HD / KV_Q8_GROUP; ++g) {
            float weighted[DCH / 8][4];
            float wmax_a = 0.0f, wmax_b = 0.0f;
#pragma unroll
            for (int nt = 0; nt < DCH / 8; ++nt) {
                const int cell = nt * 8 + 2 * tig;
                float v0 = 1.0f, v1 = 1.0f;
                if constexpr (KV_MODE == 1) {
                    v0 = __half2float(reinterpret_cast<const __half*>(&S.vsc[cell])[g]);
                    v1 = __half2float(reinterpret_cast<const __half*>(&S.vsc[cell + 1])[g]);
                }
                // Explicit zeros prevent a masked cell's scale from entering
                // either the maximum or an MMA operand (including 0*Inf).
                weighted[nt][0] = s[nt][0] == 0.0f ? 0.0f : s[nt][0] * v0;
                weighted[nt][1] = s[nt][1] == 0.0f ? 0.0f : s[nt][1] * v1;
                weighted[nt][2] = s[nt][2] == 0.0f ? 0.0f : s[nt][2] * v0;
                weighted[nt][3] = s[nt][3] == 0.0f ? 0.0f : s[nt][3] * v1;
                wmax_a = fmaxf(wmax_a, fmaxf(fabsf(weighted[nt][0]), fabsf(weighted[nt][1])));
                wmax_b = fmaxf(wmax_b, fmaxf(fabsf(weighted[nt][2]), fabsf(weighted[nt][3])));
            }
            // Four adjacent lanes own one row. Do not reduce across heads.
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                wmax_a = fmaxf(wmax_a, __shfl_xor_sync(0xffffffffu, wmax_a, o));
                wmax_b = fmaxf(wmax_b, __shfl_xor_sync(0xffffffffu, wmax_b, o));
            }
            int exp_a = 0, exp_b = 0;
            if (wmax_a > 0.0f) frexpf(wmax_a, &exp_a);
            if (wmax_b > 0.0f) frexpf(wmax_b, &exp_b);
            const int up_a = wmax_a > 0.0f ? 14 - exp_a : 0;
            const int up_b = wmax_b > 0.0f ? 14 - exp_b : 0;

            // m16n8k16 A registers: [row a, low k], [row b, low k],
            // [row a, high k], [row b, high k]. Each packs two FP16 values.
            uint32_t ph[2][4], pl[2][4];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                split_h2(ldexpf(weighted[2 * ks][0], up_a),
                         ldexpf(weighted[2 * ks][1], up_a), ph[ks][0], pl[ks][0]);
                split_h2(ldexpf(weighted[2 * ks][2], up_b),
                         ldexpf(weighted[2 * ks][3], up_b), ph[ks][1], pl[ks][1]);
                split_h2(ldexpf(weighted[2 * ks + 1][0], up_a),
                         ldexpf(weighted[2 * ks + 1][1], up_a), ph[ks][2], pl[ks][2]);
                split_h2(ldexpf(weighted[2 * ks + 1][2], up_b),
                         ldexpf(weighted[2 * ks + 1][3], up_b), ph[ks][3], pl[ks][3]);
            }

            // Keep only two output n-tiles live at a time; separate high/low
            // accumulators must not multiply register usage for all 64 dims.
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                float th[2][4] = {}, tl[2][4] = {};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    if constexpr (KV_MODE == 0) {
                        const int jp = 4 * g + pair;
                        const int c = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        const int pc = 2 * jp + (lane >> 4);
                        uint32_t b[4];
                        ldsm_x4_t(b, &S.v[c][pc ^ (c & 7)]);
                        mma16816(th[0], ph[ks], b);
                        mma16816(tl[0], pl[ks], b);
                        mma16816(th[1], ph[ks], b + 2);
                        mma16816(tl[1], pl[ks], b + 2);
                    } else {
                        const unsigned char* V = reinterpret_cast<const unsigned char*>(&S.v[0][0]);
                        const int cA = ks * 16 + 2 * tig, cB = cA + 8;
#pragma unroll
                        for (int jj = 0; jj < 2; ++jj) {
                            const int j = 8 * g + 2 * pair + jj;
                            const int d = j * 8 + gid;
                            auto at = [&](int c) {
                                return (uint32_t) V[c * ROWB + (((d >> 4) ^ (c & 7)) << 4) + (d & 15)];
                            };
                            uint32_t b[2];
                            b[0] = i8x2_to_h2(at(cA) | (at(cA + 1) << 8));
                            b[1] = i8x2_to_h2(at(cB) | (at(cB + 1) << 8));
                            mma16816(th[jj], ph[ks], b);
                            mma16816(tl[jj], pl[ks], b);
                        }
                    }
                }
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const int j = 8 * g + 2 * pair + jj;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float scaled_tile = fmaf(tl[jj][k], FP16_RESIDUAL_DOWN, th[jj][k]);
                        const float tile = ldexpf(scaled_tile, k < 2 ? -up_a : -up_b);
                        acc[j][k] = fmaf(acc[j][k], k < 2 ? alpha_a : alpha_b, tile);
                    }
                }
            }
        }
        __syncthreads();   // every warp is done with V
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
    float* oa = attn + ((size_t) qa * n_head + kvh * GQ + gid) * HD;
    float* ob = attn + ((size_t) qb * n_head + kvh * GQ + gid) * HD;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
#pragma unroll
        for (int j = 8 * g; j < 8 * g + 8; ++j) {
            const int d = j * 8 + 2 * tig;
            if (qa < n_q) *reinterpret_cast<float2*>(oa + d) = make_float2(acc[j][0] * ia, acc[j][1] * ia);
            if (qb < n_q) *reinterpret_cast<float2*>(ob + d) = make_float2(acc[j][2] * ib, acc[j][3] * ib);
        }
    }
}


} // namespace
bool available() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess || prop.major < 8)
        return false;
    const auto status = cudaFuncSetAttribute(prompt_attn_dense_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             sizeof(SmemD<1>));
    if (status != cudaSuccess) cudaGetLastError(); // optional fast path: keep the fallback's error state clean
    return status == cudaSuccess;
}
void run(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
         float* output, void* stream) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    prompt_attn_dense_kernel<1><<<dim3((rows + DQ - 1) / DQ, kv_heads), THREADS, sizeof(SmemD<1>),
                                  (cudaStream_t) stream>>>(q, cache, pos0, rows, kv_heads, context,
                                                          scale_log2, output, 0);
}
void run_preview(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
                 int visible_end, float* output, void* stream) {
    if (rows <= 0) return;
    if (kv_heads <= 0 || context <= 0 || visible_end < 1 || visible_end > context)
        throw std::runtime_error("Invalid CUDA INT8 preview attention bounds");
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    prompt_attn_dense_kernel<1><<<dim3((rows + DQ - 1) / DQ, kv_heads), THREADS, sizeof(SmemD<1>),
                                  (cudaStream_t) stream>>>(q, cache, pos0, rows, kv_heads, context,
                                                          scale_log2, output, visible_end);
}
} // namespace strata::qwen35::int8_attention
