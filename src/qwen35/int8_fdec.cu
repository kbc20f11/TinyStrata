// Qwen INT8 FDEC adapted from PolyStrata src/poly/fattn.cu
// f472d85a30927301663401f7dc892e668578a90e, MIT.
// Copyright (c) 2026 Niko1221 and the Strata/PolyStrata contributors.
// Copyright (c) 2026 VecSzn and the PolyStrata contributors.
// Tiny: head-major/per64FP16scale/int32position, amplified residual on Q/P.
#include "strata/qwen35/int8_fdec.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
namespace strata::qwen35::int8_fdec {
namespace {
constexpr int kD = 256;
constexpr float kResidualUp = 2048.f, kResidualDown = 1.f / kResidualUp;
__device__ __forceinline__ void split_h2(float x, float y, uint32_t& hi, uint32_t& lo) {
    const half2 h = __floats2half2_rn(x, y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    const float2 f = __half22float2(h);
    const half2 l = __floats2half2_rn(__fmul_rn(__fsub_rn(x, f.x), kResidualUp),
                                    __fmul_rn(__fsub_rn(y, f.y), kResidualUp));
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
constexpr int kSplit = 1024;   // keys a CUDA block takes, counted from the cache's first row
constexpr int kTile = 64;      // keys a block takes at a time: 16 a warp
constexpr int kDecWarps = 4;
constexpr int kDecRows = 32;   // 4 tokens x 8 heads

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Two int8 of `w` (each with its top bit flipped) as a half2, exactly: 0x64XX is 1024 + XX.  sel 0x4140 takes
// bytes 0 and 1, 0x4342 bytes 2 and 3.
__device__ __forceinline__ uint32_t i8pair(uint32_t w, uint32_t sel) {
    uint32_t x = __byte_perm(w, 0x64646464u, sel);
    half2 h = __hsub2(*reinterpret_cast<half2*>(&x), __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint4 widen(uint2 v) { return make_uint4(v.x, v.y, 0u, 0u); }

__device__ __forceinline__ float scale_of(const uint4& v, int b) {   // half b of 8
    const uint32_t w = b < 4 ? (b < 2 ? v.x : v.y) : (b < 6 ? v.z : v.w);
    return __half2float(__ushort_as_half((unsigned short) (b & 1 ? w >> 16 : w & 0xffff)));
}

// One CUDA block: the keys [split * kSplit, +kSplit) of group blockIdx.y for the launch's tokens (MT * 16 / Group
// at most, row = token * Group + head of the group), as flash attention: per tile of 64 keys, a warp scores 16
// (S = Q K^T, mma m16n8k16, compensated queries, exact int8 codes as halves and each 64-value block's scale
// applied to its sum in floats), the softmax max and sum go on per row, and a warp adds 64 of the 256 value
// dimensions (O += P V, compensated P*Vscale and exact codes). It leaves the rows' unnormalized results with
// their max and sum; fdec_merge_kernel adds the splits. Masks and normalization are independent per row.
// Moving a query to another MMA fragment row can change floating-point rounding; tests check each against FP64.
// The dimensions are numbered as
// suits the reads (a thread's 8 values of K and 8 dimensions of V).
template <int MT>
__global__ void __launch_bounds__(kDecWarps * 32)
fdec_kernel(const float* __restrict__ q, const void* __restrict__ kc, const uint16_t* __restrict__ ks,
            const void* __restrict__ vc, const uint16_t* __restrict__ vs, long long pos,
            const int32_t* __restrict__ at, int first, int tokens, int n_head, int kv_heads, int context, float scale,
            float* __restrict__ part, float* __restrict__ ml) {
    constexpr int Group = 8, Q = MT * 16, QS = kD + 8, PS = kTile + 8;
    __shared__ __align__(16) half Qs[Q * QS], Ql[Q * QS];
    __shared__ __align__(16) float Ps[Q * PS];
    __shared__ int qup[Q];
    __shared__ float red[2][kDecWarps][Q];
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32, g8 = lane / 4, t = lane % 4;
    const int grp = blockIdx.y, split = blockIdx.x;
    const long long p0 = (at ? *at : pos) + first, last = p0 + tokens - 1, r0 = (long long) split * kSplit;
    if (p0 < 0 || last >= context || r0 > last) {
        // Empty graph-sized splits have neutral metadata. Their partials are
        // never read: merge uses this launch's device position, not capacity.
        for (int row = threadIdx.x; row < tokens * Group; row += blockDim.x) {
            const size_t idx = ((size_t) split * kv_heads + grp) * kDecRows + row;
            ml[2 * idx] = -INFINITY; ml[2 * idx + 1] = 0.f;
        }
        return;
    }
    const long long r1 = min(r0 + kSplit, last + 1);
    for (int row = w; row < Q; row += kDecWarps) {
        const int tok = row / Group;
        float maximum = 0.f;
        for (int d = lane; d < kD; d += 32) {
            const float x = tok < tokens ? q[((size_t) (first + tok) * n_head + grp * Group + row % Group) * kD + d] : 0.f;
            maximum = fmaxf(maximum, fabsf(x));
        }
        for (int d = 16; d; d >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d));
        int exponent = 0;
        if (maximum > 0.f) frexpf(maximum, &exponent);
        const int up = 14 - exponent;
        if (lane == 0) qup[row] = up;
        for (int d = 2 * lane; d < kD; d += 64) {
            const size_t idx = ((size_t) (first + tok) * n_head + grp * Group + row % Group) * kD + d;
            const float x0 = tok < tokens ? q[idx] : 0.f, x1 = tok < tokens ? q[idx + 1] : 0.f;
            uint32_t high, low;
            split_h2(ldexpf(x0, up), ldexpf(x1, up), high, low);
            *reinterpret_cast<uint32_t*>(&Qs[row * QS + d]) = high;
            *reinterpret_cast<uint32_t*>(&Ql[row * QS + d]) = low;
        }
    }
    constexpr uint32_t kZero = 0u; // raw int8 zero, before the i8pair sign flip
    const size_t width = kD, swidth = kD / 64;
    const char* kg = (const char*) kc + (size_t) grp * context * kD;
    const char* vg = (const char*) vc + (size_t) grp * context * kD + 64 * w + 8 * g8;
    const uint16_t* ksg = ks + (size_t) grp * context * swidth;
    const uint16_t* vsg = vs + (size_t) grp * context * swidth + w;
    float m[MT][2], l[MT][2], o[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        m[mt][0] = m[mt][1] = -INFINITY;
        l[mt][0] = l[mt][1] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.f;
    }
    __syncthreads();
    for (long long tb = r0; tb < r1; tb += kTile) {
        // the reads first, all of them: K of the warp's 16 keys (n-tile nt: key 8 nt + g8; a thread 8 values of
        // each 32-value block), their scales (keys 8 nt + 2t + j), V of all 64 keys for the warp's dimensions.
        // Each uint2 load contains eight signed INT8 codes.
        uint4 kx[2][8];
        uint4 ksc[2][2] = {};
        uint4 vx[4][4];
        uint16_t vsc[4][4] = {};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long key = tb + 16 * w + 8 * nt + g8;
#pragma unroll
            for (int b = 0; b < 8; ++b)
                kx[nt][b] = key > last ? make_uint4(kZero, kZero, kZero, kZero)
                          : widen(*reinterpret_cast<const uint2*>(kg + key * width + 32 * b + 8 * t));
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const long long k2 = tb + 16 * w + 8 * nt + 2 * t + j;
                if (k2 <= last) ksc[nt][j] = widen(*reinterpret_cast<const uint2*>(ksg + k2 * swidth));
            }
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const long long key = tb + 16 * kk + 2 * t + (i & 1) + 8 * (i >> 1);
                vx[kk][i] = key > last ? make_uint4(kZero, kZero, kZero, kZero)
                          : widen(*reinterpret_cast<const uint2*>(vg + key * width));
                if (key <= last) vsc[kk][i] = vsg[key * swidth];
            }
        // S = Q K^T, a 32-value block (two k-steps) at a time, then times the keys' scales of the block
        float s[MT][2][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) s[mt][nt][0] = s[mt][nt][1] = s[mt][nt][2] = s[mt][nt][3] = 0.f;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            float c[MT][2][4], cl[MT][2][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) c[mt][nt][0] = c[mt][nt][1] = c[mt][nt][2] = c[mt][nt][3] =
                    cl[mt][nt][0] = cl[mt][nt][1] = cl[mt][nt][2] = cl[mt][nt][3] = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = 32 * b + 8 * t + 4 * h;   // the thread's 4 dimensions of the step
                uint32_t bk[2][2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t word = (h ? kx[nt][b].y : kx[nt][b].x) ^ 0x80808080u;
                    bk[nt][0] = i8pair(word, 0x4140);
                    bk[nt][1] = i8pair(word, 0x4342);
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint2 lo = *reinterpret_cast<const uint2*>(&Qs[(mt * 16 + g8) * QS + d]);
                    const uint2 hi = *reinterpret_cast<const uint2*>(&Qs[(mt * 16 + g8 + 8) * QS + d]);
                    const uint32_t a[4] = {lo.x, hi.x, lo.y, hi.y};
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) mma16816(c[mt][nt], a, bk[nt][0], bk[nt][1]);
                    const uint2 ll = *reinterpret_cast<const uint2*>(&Ql[(mt * 16 + g8) * QS + d]);
                    const uint2 lh = *reinterpret_cast<const uint2*>(&Ql[(mt * 16 + g8 + 8) * QS + d]);
                    const uint32_t al[4] = {ll.x, lh.x, ll.y, lh.y};
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) mma16816(cl[mt][nt], al, bk[nt][0], bk[nt][1]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float sc = scale_of(ksc[nt][j], b / 2);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        s[mt][nt][j] += fmaf(cl[mt][nt][j], kResidualDown, c[mt][nt][j]) * sc;
                        s[mt][nt][2 + j] += fmaf(cl[mt][nt][2 + j], kResidualDown, c[mt][nt][2 + j]) * sc;
                    }
                }
        }
        // the causal mask, the tile's max per row over the four warps, the new max
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r, tok = row / Group;
                float mx = -INFINITY;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const long long key = tb + 16 * w + 8 * nt + 2 * t + j;
                        float& x = s[mt][nt][2 * r + j];
                        x = tok < tokens && key <= p0 + tok ? ldexpf(x, -qup[row]) * scale : -INFINITY;
                        mx = fmaxf(mx, x);
                    }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                if (t == 0) red[0][w][row] = mx;
            }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float top = m[mt][r];
#pragma unroll
                for (int k = 0; k < kDecWarps; ++k) top = fmaxf(top, red[0][k][row]);
                alpha[mt][r] = top == -INFINITY ? 1.f : expf(m[mt][r] - top);
                m[mt][r] = top;
                float sum = 0.f;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const float x0 = s[mt][nt][2 * r], x1 = s[mt][nt][2 * r + 1];
                    const float p0 = x0 == -INFINITY ? 0.f : expf(x0 - top);
                    const float p1 = x1 == -INFINITY ? 0.f : expf(x1 - top);
                    sum += p0 + p1;
                    Ps[row * PS + 16 * w + 8 * nt + 2 * t] = p0;
                    Ps[row * PS + 16 * w + 8 * nt + 2 * t + 1] = p1;
                }
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                if (t == 0) red[1][w][row] = sum;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o[mt][j][2 * r] *= alpha[mt][r];
                    o[mt][j][2 * r + 1] *= alpha[mt][r];
                }
            }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < kDecWarps; ++k) sum += red[1][k][row];
                l[mt][r] = l[mt][r] * alpha[mt][r] + sum;
            }
        // P*Vscale is split in FP32; V stays exact integer codes. Each query
        // and 64-value group chooses a power-of-two scale for this tile.
        int pup[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float maximum = 0.f;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int key = 16 * kk + 2 * t + (i & 1) + 8 * (i >> 1);
                        const float weighted = __fmul_rn(Ps[row * PS + key], __half2float(__ushort_as_half(vsc[kk][i])));
                        maximum = fmaxf(maximum, fabsf(weighted));
                    }
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 1));
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
                int exponent = 0;
                if (maximum > 0.f) frexpf(maximum, &exponent);
                pup[mt][r] = maximum > 0.f ? 14 - exponent : 0;
            }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t a[MT][4], al[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = mt * 16 + g8 + 8 * r;
                    float weighted[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int key = 16 * kk + 2 * t + (i & 1) + 8 * (i >> 1);
                        weighted[i] = ldexpf(__fmul_rn(Ps[row * PS + key], __half2float(__ushort_as_half(vsc[kk][i]))), pup[mt][r]);
                    }
                    split_h2(weighted[0], weighted[1], a[mt][r], al[mt][r]);
                    split_h2(weighted[2], weighted[3], a[mt][r + 2], al[mt][r + 2]);
                }
#pragma unroll
            for (int wd = 0; wd < 2; ++wd) {
                const uint32_t x0 = (wd ? vx[kk][0].y : vx[kk][0].x) ^ 0x80808080u;
                const uint32_t x1 = (wd ? vx[kk][1].y : vx[kk][1].x) ^ 0x80808080u;
                const uint32_t x2 = (wd ? vx[kk][2].y : vx[kk][2].x) ^ 0x80808080u;
                const uint32_t x3 = (wd ? vx[kk][3].y : vx[kk][3].x) ^ 0x80808080u;
                const uint32_t lo01 = __byte_perm(x0, x1, 0x5140), hi01 = __byte_perm(x0, x1, 0x7362);
                const uint32_t lo89 = __byte_perm(x2, x3, 0x5140), hi89 = __byte_perm(x2, x3, 0x7362);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t sel = jj & 1 ? 0x4342 : 0x4140;
                    const uint32_t b0 = i8pair(jj < 2 ? lo01 : hi01, sel);
                    const uint32_t b1 = i8pair(jj < 2 ? lo89 : hi89, sel);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        float high[4] = {}, low[4] = {};
                        mma16816(high, a[mt], b0, b1);
                        mma16816(low, al[mt], b0, b1);
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                            o[mt][4 * wd + jj][c] += ldexpf(fmaf(low[c], kResidualDown, high[c]), -pup[mt][c / 2]);
                    }
                }
            }
        }
        __syncthreads();   // Ps and red are the next tile's
    }
    // the rows' results: c0 of n-tile j is dimension 64 w + 16 t + j, c1 the one 8 on
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = mt * 16 + g8 + 8 * r;
            if (row / Group >= tokens) continue;
            const size_t idx = ((size_t) split * kv_heads + grp) * kDecRows + row;
            float4* dst = reinterpret_cast<float4*>(part + idx * kD + 64 * w + 16 * t);
            dst[0] = make_float4(o[mt][0][2 * r], o[mt][1][2 * r], o[mt][2][2 * r], o[mt][3][2 * r]);
            dst[1] = make_float4(o[mt][4][2 * r], o[mt][5][2 * r], o[mt][6][2 * r], o[mt][7][2 * r]);
            dst[2] = make_float4(o[mt][0][2 * r + 1], o[mt][1][2 * r + 1], o[mt][2][2 * r + 1], o[mt][3][2 * r + 1]);
            dst[3] = make_float4(o[mt][4][2 * r + 1], o[mt][5][2 * r + 1], o[mt][6][2 * r + 1], o[mt][7][2 * r + 1]);
            if (w == 0 && t == 0) {
                ml[2 * idx] = m[mt][r];
                ml[2 * idx + 1] = l[mt][r];
            }
        }
}

// The splits of a token and head added in their order, each by its sum against the largest max.  One CUDA block
// a head and token, a thread a dimension.  A token reads only the splits up to its own row.
__global__ void fdec_merge_kernel(const float* __restrict__ part, const float* __restrict__ ml, float* __restrict__ out,
                                  long long pos, const int32_t* __restrict__ at, int first, int n_head, int kv_heads, int context, const float* gate) {
    constexpr int Group = 8;
    const int h = blockIdx.x, tok = blockIdx.y, d = threadIdx.x, grp = h / Group;
    const long long p = (at ? *at : pos) + first + tok;
    if (p < 0 || p >= context) return;
    const int splits = int(p / kSplit) + 1, row = tok * Group + h % Group;
    float top = -INFINITY;
    for (int i = 0; i < splits; ++i) top = fmaxf(top, ml[2 * (((size_t) i * kv_heads + grp) * kDecRows + row)]);
    float sum = 0.f, acc = 0.f;
    for (int i = 0; i < splits; ++i) {
        const size_t idx = ((size_t) i * kv_heads + grp) * kDecRows + row;
        const float e = expf(ml[2 * idx] - top);
        sum += ml[2 * idx + 1] * e;
        acc += part[idx * kD + d] * e;
    }
    const size_t idx = ((size_t) (first + tok) * n_head + h) * kD + d;
    out[idx] = acc / sum * (gate ? 1.f / (1.f + expf(-gate[idx])) : 1.f);
}


__global__ void compiled_target(int* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    *out = 1;
#else
    *out = 0;
#endif
}
void check(cudaError_t status, const char* where) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("INT8 FDEC ") + where + ": " + cudaGetErrorString(status));
}
std::size_t floats(int context, int kv_heads, std::size_t width) {
    if (context < 4096 || kv_heads <= 0 || kv_heads > 65535) return 0;
    const std::size_t splits = (static_cast<std::size_t>(context) + kSplit - 1) / kSplit;
    const std::size_t rows = static_cast<std::size_t>(kv_heads) * kDecRows;
    if (splits > std::numeric_limits<std::size_t>::max() / rows / width) return 0;
    return splits * rows * width;
}
} // namespace

std::size_t part_floats(int context, int kv_heads) { return floats(context, kv_heads, kD); }
std::size_t meta_floats(int context, int kv_heads) { return floats(context, kv_heads, 2); }

bool available() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) {
        (void) cudaGetLastError(); return false;
    }
    static std::mutex mutex;
    static int known[64] = {};
    std::lock_guard<std::mutex> lock(mutex);
    if (known[device]) return known[device] == 2;
    known[device] = 1;
    int major = 0, shared = 0;
    cudaFuncAttributes attrs{};
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess || major < 8 ||
        cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlock, device) != cudaSuccess ||
        cudaFuncGetAttributes(&attrs, fdec_kernel<2>) != cudaSuccess || attrs.sharedSizeBytes > (std::size_t) shared) {
        (void) cudaGetLastError(); return false;
    }
    // A sm70-only PTX build may JIT on a newer device but still contains the
    // unsupported branch. Probe compiled code once, outside graph capture.
    int* probe = nullptr;
    if (cudaMalloc(&probe, sizeof(int)) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    compiled_target<<<1, 1>>>(probe);
    int target = 0;
    const auto launch = cudaGetLastError();
    const auto copied = launch == cudaSuccess ? cudaMemcpy(&target, probe, sizeof(int), cudaMemcpyDeviceToHost) : launch;
    const auto freed = cudaFree(probe);
    if (copied == cudaSuccess && freed == cudaSuccess && target == 1) known[device] = 2;
    else (void) cudaGetLastError();
    return known[device] == 2;
}

bool run(const float* query, Cache cache, const std::int32_t* position, std::int64_t host_pos,
         int n, int heads, int kv_heads, int context, float scale,
         float* part, std::size_t part_capacity, float* meta, std::size_t meta_capacity,
         const float* gate, float* output, void* stream, bool graph_grid) {
    const std::size_t need_part = part_floats(context, kv_heads), need_meta = meta_floats(context, kv_heads);
    if (!need_part || !need_meta || n < 1 || n > 4 || heads != 8 * kv_heads ||
        host_pos < 0 || host_pos > (std::int64_t) context - n || !std::isfinite(scale) ||
        !query || !cache.key || !cache.value || !cache.key_scale || !cache.value_scale || !output ||
        !part || !meta || part_capacity < need_part || meta_capacity < need_meta ||
        (reinterpret_cast<std::uintptr_t>(part) & 15u) ||
        (reinterpret_cast<std::uintptr_t>(cache.key) & 7u) || (reinterpret_cast<std::uintptr_t>(cache.value) & 7u) ||
        (reinterpret_cast<std::uintptr_t>(cache.key_scale) & 7u) || (reinterpret_cast<std::uintptr_t>(cache.value_scale) & 7u) ||
        !available()) return false;
    const int rows = graph_grid ? context : (int) (host_pos + n);
    const int splits = (int) ((static_cast<std::int64_t>(rows) + kSplit - 1) / kSplit);
    const auto st = static_cast<cudaStream_t>(stream);
    if (n <= 2)
        fdec_kernel<1><<<dim3(splits, kv_heads), 128, 0, st>>>(query, cache.key, cache.key_scale, cache.value,
            cache.value_scale, host_pos, position, 0, n, heads, kv_heads, context, scale, part, meta);
    else
        fdec_kernel<2><<<dim3(splits, kv_heads), 128, 0, st>>>(query, cache.key, cache.key_scale, cache.value,
            cache.value_scale, host_pos, position, 0, n, heads, kv_heads, context, scale, part, meta);
    check(cudaGetLastError(), "partial launch");
    fdec_merge_kernel<<<dim3(heads, n), kD, 0, st>>>(part, meta, output, host_pos, position, 0, heads, kv_heads, context, gate);
    check(cudaGetLastError(), "merge launch");
    return true;
}
} // namespace strata::qwen35::int8_fdec
