// Qwen3.6 attention kernels extracted from Strata (MIT; see LICENSE and NOTICE).
#pragma once
#include "strata/qwen35/fp8.cuh"
#include <cuda_fp16.h>
#include <cfloat>

namespace strata::qwen35::attention {
constexpr int kThreads = 256;
constexpr int kChunkCells = 256;
struct KvCache {
    void *key = nullptr, *value = nullptr;
    float *key_scale = nullptr, *value_scale = nullptr;
    uint16_t *key_scale16 = nullptr, *value_scale16 = nullptr;
};
// 0: FP32, 1: FP8 per head, 2: INT8 per 64, 3: FP8 per 64 (native prompt MMA).
template<int Format>
__device__ __forceinline__ float kv_read(const void* data, const float* scales, size_t index, int hd,
                                        const uint16_t* scales16 = nullptr) {
    if constexpr (Format == 1) return fp8_decode(((const uint8_t*) data)[index]) * scales[index / hd];
    else if constexpr (Format == 3)
        return fp8_decode(((const uint8_t*) data)[index]) * __half2float(reinterpret_cast<const __half*>(scales16)[index / 64]);
    else if constexpr (Format == 2)
        return (float) ((const int8_t*) data)[index] * __half2float(reinterpret_cast<const __half*>(scales16)[index / 64]);
    else return ((const float*) data)[index];
}
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// grid (H + HKV, T), block 256, HD == 256 (enforced by the model loader).
template<int Format>
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, KvCache cache, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* mrope = nullptr) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        // Qwen3.5/3.6 interleaved M-RoPE sections {11,11,10}: rotary pair p
        // reads t/h/w by p % 3. Cache addressing and causal masking still use
        // the token's cell index, never its compressed image rotary position.
        const int rotary_pos = mrope ? __ldg(mrope + (size_t) pos * 3 + p % 3) : pos;
        const float theta = (float) rotary_pos * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        const size_t slot = (size_t) kh * ctx + pos, index = slot * HD + i;
        const float vv = vx[((size_t) t * HKV + kh) * HD + i];
        if constexpr (Format == 1) {
            // Separate dynamic scales for every token, KV head and K/V vector.
            const float km = block_max(fabsf(v), sh), vm = block_max(fabsf(vv), sh);
            const float ks = km == 0 ? 1.0f : fmaxf(km / 448.0f, FLT_MIN);
            const float vs = vm == 0 ? 1.0f : fmaxf(vm / 448.0f, FLT_MIN);
            if (i == 0) { cache.key_scale[slot] = ks; cache.value_scale[slot] = vs; }
            ((uint8_t*) cache.key)[index] = fp8_encode(v / ks);
            ((uint8_t*) cache.value)[index] = fp8_encode(vv / vs);
        } else if constexpr (Format == 2 || Format == 3) {
            // Adapted from MoMoE kv_q8.cu: two warp maxima per group, then quantize against
            // the STORED (rounded FP16) scale, not the unrounded FP32 scale.
            float km = fabsf(v), vm = fabsf(vv);
            for (int off = 16; off > 0; off >>= 1) {
                km = fmaxf(km, __shfl_xor_sync(0xffffffffu, km, off));
                vm = fmaxf(vm, __shfl_xor_sync(0xffffffffu, vm, off));
            }
            if ((i & 31) == 0) { sh[i >> 5] = km; sh[8 + (i >> 5)] = vm; }
            __syncthreads();
            const int w = (i / 64) * 2;
            // The engine uses fast-math. Explicit rounded division keeps the
            // quantizer contract at half-scale/code rounding boundaries.
            constexpr float max_code = Format == 3 ? 448.0f : 127.0f;
            const __half ks = __float2half_rn(__fdiv_rn(fmaxf(sh[w], sh[w + 1]), max_code));
            const __half vs = __float2half_rn(__fdiv_rn(fmaxf(sh[8 + w], sh[9 + w]), max_code));
            const float ksf = __half2float(ks), vsf = __half2float(vs);
            if constexpr (Format == 3) {
                ((uint8_t*) cache.key)[index] = fp8_encode(ksf > 0 ? __fdiv_rn(v, ksf) : 0.f);
                ((uint8_t*) cache.value)[index] = fp8_encode(vsf > 0 ? __fdiv_rn(vv, vsf) : 0.f);
            } else {
                int ki = ksf > 0 ? __float2int_rn(__fdiv_rn(v, ksf)) : 0;
                int vi = vsf > 0 ? __float2int_rn(__fdiv_rn(vv, vsf)) : 0;
                ((int8_t*) cache.key)[index] = (int8_t) max(-127, min(127, ki));
                ((int8_t*) cache.value)[index] = (int8_t) max(-127, min(127, vi));
            }
            if ((i & 63) == 0) {
                reinterpret_cast<__half*>(cache.key_scale16)[index / 64] = ks;
                reinterpret_cast<__half*>(cache.value_scale16)[index / 64] = vs;
            }
        } else {
            ((float*) cache.key)[index] = v;
            ((float*) cache.value)[index] = vv;
        }
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.
template<int Format>
__global__ void k_attn_part(const float* q, KvCache cache, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const size_t offset = ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a += qs[i] * kv_read<Format>(cache.key, cache.key_scale, offset + i, HD, cache.key_scale16);
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    const size_t offset = ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k)
        acc += sc[k] * kv_read<Format>(cache.value, cache.value_scale, offset + (size_t) k * HD, HD, cache.value_scale16);
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * (1.0f / (1.0f + expf(-gate[gi])));
}

/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).
template<int Format>
__global__ void k_kv_f16(KvCache cache, __half* k16, __half* v16, int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    k16[i] = __float2half(kv_read<Format>(cache.key, cache.key_scale, src, HD, cache.key_scale16));
    v16[i] = __float2half(kv_read<Format>(cache.value, cache.value_scale, src, HD, cache.value_scale16));
}
} // namespace strata::qwen35::attention
