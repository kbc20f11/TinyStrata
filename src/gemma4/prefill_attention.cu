// FP16 WMMA fragment mapping follows TinyStrata src/qwen35/int8_attention_hip.cu,
// adapted there from Niko1221/Strata src/kernels/cuda/qsa_prompt_attn.cu (MIT),
// revision 5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d. See LICENSE and NOTICE.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// This D64 prefill specialization reads the existing token-major INT8 cache unchanged,
// using baseline FP16 Q/K/V/P precision, FP32 accumulation, and one sink contribution.
#include "strata/gemma4/prefill_attention.hpp"
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace strata::gemma4::prefill_attention {
namespace {
constexpr int D = 64, GQA = 8, WAVES = 8, THREADS = WAVES * 32;
constexpr int TOKENS = WAVES * 2, QUERY_ROWS = TOKENS * GQA, KEYS = kKeyTile, KV_STRIDE = D + 8;
using HalfFragment = _Float16 __attribute__((ext_vector_type(8)));
using FloatFragment = float __attribute__((ext_vector_type(8)));

struct alignas(16) Shared {
    _Float16 key[KEYS][KV_STRIDE], value[KEYS][KV_STRIDE];
    float score[QUERY_ROWS][KEYS + 1];
    _Float16 probability[QUERY_ROWS][KEYS + 1];
    float maximum[QUERY_ROWS], denominator[QUERY_ROWS], alpha[QUERY_ROWS];
};

__device__ __forceinline__ FloatFragment zero() {
    return FloatFragment{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
}

__device__ __forceinline__ FloatFragment wmma(HalfFragment a, HalfFragment b, FloatFragment c) {
#if defined(__gfx1200__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c);
#else
    (void)a; (void)b; return c; // host rejects images/devices without the specialization
#endif
}

__global__ void __launch_bounds__(THREADS)
prefill_kernel(const float* __restrict__ q, const int8_t* __restrict__ kc, const uint16_t* __restrict__ ks,
               const int8_t* __restrict__ vc, const uint16_t* __restrict__ vs, const float* __restrict__ sinks,
               float* __restrict__ out, int heads, int kv_heads, long long pos, int rows, float scale) {
#if defined(__gfx1200__)
    __shared__ Shared shared;
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4, group = int(blockIdx.y);
    const int q0 = int(blockIdx.x) * TOKENS;
    const int query = q0 + wave * 2 + col / GQA;
    const int head = group * GQA + col % GQA;
    const int last = int(pos) + min(q0 + TOKENS, rows);
    HalfFragment qa[D / 16];
#pragma unroll
    for (int k = 0; k < D / 16; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            qa[k][i] = query < rows ? (_Float16) q[(size_t(query) * heads + head) * D + k * 16 + half * 8 + i]
                                    : (_Float16) 0.f;
    FloatFragment output[D / 16];
#pragma unroll
    for (int j = 0; j < D / 16; ++j) output[j] = zero();
    if (tid < QUERY_ROWS) {
        shared.maximum[tid] = -INFINITY;
        shared.denominator[tid] = 0.f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < last; k0 += KEYS) {
        // The entire cache tile is loaded once per CTA and shared by all its causal query tokens.
        // Predicate before every global load: later reserved KV pages need not be physically mapped.
        for (int index = tid; index < KEYS * D; index += THREADS) {
            const int key = k0 + index / D, d = index % D;
            _Float16 key_value = (_Float16)0.f, value = (_Float16)0.f;
            if (key < last) {
                const size_t row = size_t(key) * kv_heads + group;
                key_value = (_Float16)(float(kc[row * D + d]) * __half2float(__ushort_as_half(ks[row * 2 + d / 32])));
                value = (_Float16)(float(vc[row * D + d]) * __half2float(__ushort_as_half(vs[row * 2 + d / 32])));
            }
            shared.key[index / D][d] = key_value;
            shared.value[index / D][d] = value;
        }
        __syncthreads();

#pragma unroll
        for (int nt = 0; nt < KEYS / 16; ++nt) {
            FloatFragment dot = zero();
#pragma unroll
            for (int k = 0; k < D / 16; ++k) {
                HalfFragment key;
#pragma unroll
                for (int i = 0; i < 8; ++i) key[i] = shared.key[nt * 16 + col][k * 16 + half * 8 + i];
                dot = wmma(qa[k],key,dot);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                shared.score[wave * 16 + half * 8 + i][nt * 16 + col] = dot[i] * scale;
        }
        __syncthreads();

        // A pair of lanes owns one query/head row; every wave participates. Each lane handles
        // alternating keys, then shares its max/sum with its partner. The denominator sums the SAME
        // half-rounded probabilities that PV consumes, as the existing BLAS path does.
        {
            const int r = tid >> 1, sub = tid & 1, token = q0 + r / GQA;
            const int upper = token < rows ? int(pos) + token + 1 : 0;
            float maximum = -INFINITY;
            for (int j = sub; j < KEYS; j += 2)
                if (k0 + j < upper) maximum = fmaxf(maximum,shared.score[r][j]);
            maximum = fmaxf(maximum,__shfl_xor(maximum,1,32));
            const float before = shared.maximum[r], now = fmaxf(before,maximum);
            const float alpha = before == -INFINITY ? 0.f : expf(before - now);
            float sum = 0.f;
            for (int j = sub; j < KEYS; j += 2) {
                const _Float16 p = k0 + j < upper && now != -INFINITY
                                      ? (_Float16)expf(shared.score[r][j] - now) : (_Float16)0.f;
                shared.probability[r][j] = p;
                sum += (float)p;
            }
            sum += __shfl_xor(sum,1,32);
            if (sub == 0) {
                shared.alpha[r] = alpha;
                shared.maximum[r] = now;
                shared.denominator[r] = fmaf(shared.denominator[r],alpha,sum);
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < D / 16; ++j) {
            FloatFragment tile = zero();
#pragma unroll
            for (int k = 0; k < KEYS / 16; ++k) {
                HalfFragment probability,value;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    probability[i] = shared.probability[wave * 16 + col][k * 16 + half * 8 + i];
                    value[i] = shared.value[k * 16 + half * 8 + i][j * 16 + col];
                }
                tile = wmma(probability,value,tile);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                output[j][i] = fmaf(output[j][i],shared.alpha[wave * 16 + half * 8 + i],tile[i]);
        }
        __syncthreads(); // all waves finish consuming KV/P before the next tile overwrites them
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = wave * 16 + half * 8 + i, token = q0 + r / GQA;
        if (token >= rows) continue;
        const int query_head = group * GQA + r % GQA;
        // Add the learned sink once, preserving QK-based half-P rounding. Rescale at the end
        // rather than exp(sink-QKmax), so a large finite sink cannot overflow the denominator.
        const float m = shared.maximum[r], sink = sinks[query_head], top = fmaxf(m,sink);
        const float factor = expf(m-top);
        const float denominator = fmaf(shared.denominator[r],factor,expf(sink-top));
#pragma unroll
        for (int j = 0; j < D / 16; ++j)
            out[(size_t(token) * heads + query_head) * D + j * 16 + col] = (output[j][i] * factor) / denominator;
    }
#else
    (void)q;(void)kc;(void)ks;(void)vc;(void)vs;(void)sinks;(void)out;
    (void)heads;(void)kv_heads;(void)pos;(void)rows;(void)scale;
#endif
}

bool compiled() {
#if defined(STRATA_HIP_ARCHS)
    const char* item = STRATA_HIP_ARCHS;
    while (*item) {
        const char* end = std::strchr(item,',');
        if (!end) end = item + std::strlen(item);
        if (end-item == 7 && !std::strncmp(item,"gfx1200",7)) return true;
        item = *end ? end+1 : end;
    }
#endif
    return false;
}
} // namespace

bool available() {
    if (!compiled()) return false;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess) return false;
    static thread_local int previous = -2;
    static thread_local bool supported = false;
    if (previous != device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties,device) != hipSuccess) return false;
        supported = !std::strncmp(properties.gcnArchName,"gfx1200",7) &&
                    (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':');
        previous = device;
    }
    return supported;
}

bool run(const float* q,const int8_t* kc,const uint16_t* ks,const int8_t* vc,const uint16_t* vs,
         const float* sinks,float* out,int heads,int kv_heads,int64_t pos,int rows,float scale,void* stream) {
    if (!q || !kc || !ks || !vc || !vs || !sinks || !out || kv_heads < 1 || kv_heads > 32 || heads < 1 || heads > 256 ||
        heads != GQA * kv_heads || rows < 1 || pos < 0 || pos > INT_MAX - int64_t(rows) ||
        !(scale > 0.f) || !std::isfinite(scale) || !available()) return false;
    prefill_kernel<<<dim3((rows-1)/TOKENS+1,kv_heads),THREADS,0,(hipStream_t)stream>>>(
        q,kc,ks,vc,vs,sinks,out,heads,kv_heads,pos,rows,scale);
    if (const auto error = hipGetLastError(); error != hipSuccess)
        throw std::runtime_error(std::string("prefill attention launch: ")+hipGetErrorString(error));
    return true;
}
} // namespace strata::gemma4::prefill_attention

#else
namespace strata::gemma4::prefill_attention {
bool available() { return false; }
bool run(const float*,const int8_t*,const uint16_t*,const int8_t*,const uint16_t*,const float*,float*,
         int,int,int64_t,int,float,void*) { return false; }
} // namespace strata::gemma4::prefill_attention
#endif
