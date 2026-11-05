// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Copyright (c) 2023-2026 The ggml authors.
// GDN/quantized-dot arithmetic extracted from TinyStrata's q36_engine.cu,
// q36_kernels.cuh and native_mmvq.cu. The native Q6_K dot is adapted from
// llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{vecdotq.cuh,mmvq.cu,common.cuh} and ggml/src/ggml-common.h.
// The cooperative full-mixer schedule is independently implemented after
// consulting Basalt's docs/unified-kernel.md at
// bbf2f5bb783adc5ffe5b49043f82c9399313bb46 (MIT; see NOTICE).
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#if defined(__CUDACC__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
#include "q36_kernels.cuh"
#include <cooperative_groups.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace q36g {

// One main-model causal/verify GDN layer. input_q is prep_in's existing Q8_1
// image; out_q is the ordinary mm(ssm_out)'s separate Q8_1 scratch. All stage
// outputs remain visible, including the inputs retained for a verify commit.
struct Args {
    int type = 8, T = 1, n = 2048, hk = 16, hv = 32;
    bool write_state = true;
    const void *wqkv = nullptr, *wz = nullptr, *wo = nullptr;
    const float *input = nullptr, *wa = nullptr, *wb = nullptr;
    const float *dt = nullptr, *ssm_a = nullptr, *conv_w = nullptr, *norm = nullptr;
    const q36k::QB* input_q = nullptr;
    float *hist = nullptr, *state = nullptr;
    float *qkv = nullptr, *z = nullptr, *alpha = nullptr, *beta = nullptr;
    float *hc = nullptr, *gdn_y = nullptr, *out = nullptr;
    q36k::QB* out_q = nullptr;
    float eps = 1e-6f;
};

// Host-only launch metadata: no owned allocation, counter or device lifetime.
// prepare caches the requested weight type at T1..4 before stream capture;
// supported/launch do not query the device or allocate, so the plan can be
// reused by CUDA Graphs. The unprepared type retains the existing path.
struct Plan {
    int device = -1, sms = 0, cooperative = 0;
    int blocks[2][4]{};
    int registers[2][4]{};
    size_t shared_bytes[2][4]{};
};

namespace detail {
constexpr int S = 128, N = 2048, HK = 16, HV = 32;
constexpr int C = (2 * HK + HV) * S, ZV = HV * S;

struct Q6Block {
    uint8_t ql[128], qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(Q6Block) == 210 && offsetof(Q6Block, d) == 208, "GGUF Q6_K");

__device__ __forceinline__ int load_b2(const uint8_t* p, int i) {
    const auto* v = reinterpret_cast<const uint16_t*>(p + 4 * i);
    return int(v[0]) | (int(v[1]) << 16);
}

// native_mmvq.cu:q6_q8_dot, including its signed sub-scales and exact DP4A
// order. Each lane's four original warp partials are combined in warp order
// before the same XOR tree, as q8_row_dot does for Q8_0.
__device__ __forceinline__ float q6_dot(const Q6Block* w, const q36k::QB* x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_b2(w->ql, iqs);
    const int vh = load_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = w->scales[scale_offset + 4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    const float d = w->d;
    return d * sumf;
}

template<int NC, int NI>
__device__ __forceinline__ void q6_row(const void* weights, const q36k::QB* x,
                                      int row, int lane, float (&result)[NC]) {
    constexpr int NB = NI / 256, XB = NI / 32;
    const auto* w = static_cast<const Q6Block*>(weights) + size_t(row) * NB;
    float part[NC][4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float v = 0.0f;
#pragma unroll
            for (int k = p; k < NB; k += 4) v += q6_dot(w + k, x + size_t(c) * XB + k * 8, lane);
            part[c][p] = v;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v = part[c][0];
        v += part[c][1]; v += part[c][2]; v += part[c][3];
        result[c] = q36k::wsum(v);
    }
}

template<int TYPE, int NC, int NI>
__device__ __forceinline__ void row_dot(const void* weights, const q36k::QB* x,
                                       int row, int lane, float (&result)[NC]) {
    if constexpr (TYPE == 8)
        q36k::q8_row_dot<NC, (NI / 32 + 31) / 32, false>(
            static_cast<const uint8_t*>(weights), nullptr, x, row, NI / 32, lane, result);
    else q6_row<NC, NI>(weights, x, row, lane, result);
}

struct Shared {
    float conv_part[4];
    float sk[S], sq[S];
    float red_kv[4][S], red_o[4][S];
    float wsum[S * 4 / 32];
};

// The existing k_gdn_ab expression, with a logical job replacing blockIdx.
__device__ __forceinline__ void ab(const Args& a, int job, int lane) {
    const int token = job / (2 * HV), o = job % (2 * HV);
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const auto* wr = reinterpret_cast<const float4*>((isb ? a.wb : a.wa) + size_t(r) * N);
    const auto* xr = reinterpret_cast<const float4*>(a.input + size_t(token) * N);
    float acc = 0.f;
    for (int i = lane; i < N / 4; i += 32) {
        const float4 av = wr[i], bv = xr[i];
        acc += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) a.beta[size_t(token) * HV + r] = 1.0f / (1.0f + expf(-acc));
    else {
        const float v = acc + a.dt[r];
        const float softplus = v > 20.f ? v : log1pf(expf(v));
        a.alpha[size_t(token) * HV + r] = softplus * a.ssm_a[r];
    }
}

// k_conv_seq's one-tile path. Only the first 128 threads do arithmetic, but
// all 512 participate in its CTA barriers before the next cooperative phase.
template<int NC>
__device__ __forceinline__ void conv(const Args& a, int head, Shared& sh) {
    const bool active = threadIdx.y == 0;
    const int col = threadIdx.x, c = head * S + col;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f;
    float w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
    if (active) {
        v0 = a.hist[c * 3]; v1 = a.hist[c * 3 + 1]; v2 = a.hist[c * 3 + 2];
        w0 = a.conv_w[c * 4]; w1 = a.conv_w[c * 4 + 1];
        w2 = a.conv_w[c * 4 + 2]; w3 = a.conv_w[c * 4 + 3];
    }
    const bool qk = head < 2 * HK;
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        float y = 0.f;
        if (active) {
            const float x = a.qkv[size_t(t) * C + c];
            // k_conv_seq's generated CUDA order starts with the second
            // product, then contracts the first. The unrolled fused context
            // otherwise contracts the opposite product and can differ by an
            // ULP before activation quantization. Keep the legacy order.
            float sum = __fmul_rn(w1, v1);
            sum = fmaf(w0, v0, sum);
            sum = fmaf(w2, v2, sum);
            sum = fmaf(w3, x, sum);
            v0 = v1; v1 = v2; v2 = x;
            y = sum / (1.0f + __expf(-sum));
        }
        if (qk) {
            if (active) {
                float sq = y * y;
                for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
                if ((col & 31) == 0) sh.conv_part[col >> 5] = sq;
            }
            __syncthreads();
            const float ss = sh.conv_part[0] + sh.conv_part[1] + sh.conv_part[2] + sh.conv_part[3];
            __syncthreads();
            y *= rsqrtf(ss + a.eps);
        }
        if (active) a.hc[size_t(t) * C + c] = y;
    }
    if (a.write_state && active) {
        a.hist[c * 3] = v0; a.hist[c * 3 + 1] = v1; a.hist[c * 3 + 2] = v2;
    }
}

// k_gdn_seq's body, with unchanged 128 x 4 thread geometry, state row
// ownership, four-way additions, FMAs and gated normalization expression.
template<int NC>
__device__ __forceinline__ void rec(const Args& a, int head, Shared& sh) {
    constexpr int RG = 4, RPG = 32;
    const int col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % HK;
    float s[RPG];
    float* base = a.state + (size_t(rg * RPG) * HV + head) * S + col;
    constexpr size_t row_stride = size_t(HV) * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        const float* ht = a.hc + size_t(t) * C;
        if (tid < S) { sh.sk[tid] = ht[HK * S + qh * S + tid]; sh.sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(a.alpha[size_t(t) * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sh.sk[rg * RPG + r], kv);
        sh.red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = sh.red_kv[0][col] + sh.red_kv[1][col] + sh.red_kv[2][col] + sh.red_kv[3][col];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * a.beta[size_t(t) * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sh.sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sh.sq[rg * RPG + r], o);
        }
        sh.red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (sh.red_o[0][col] + sh.red_o[1][col] + sh.red_o[2][col] + sh.red_o[3][col]) * rsqrtf(float(S));
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) sh.wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = sh.wsum[0] + sh.wsum[1] + sh.wsum[2] + sh.wsum[3];
            const float scale = rsqrtf(ss / float(S) + a.eps);
            const float zz = a.z[size_t(t) * ZV + head * S + col];
            const int i = t * ZV + head * S + col;
            a.gdn_y[i] = oc * scale * a.norm[col] * (zz / (1.0f + __expf(-zz)));
            // Preserve the original FP32 store/quantizer read boundary. The
            // volatile read prevents keeping a differently contracted value
            // across that boundary; these four whole warps each own one block.
            const float rounded = static_cast<volatile float*>(a.gdn_y)[i];
            q36k::q8_store(a.out_q, i, rounded, col & 31);
        }
        __syncthreads();
    }
    if (a.write_state) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

template<int TYPE, int NC>
__global__ void __launch_bounds__(512) chain(Args a) {
    const auto grid = cooperative_groups::this_grid();
    __shared__ Shared sh;
    const int tid = threadIdx.x + S * threadIdx.y;
    const int lane = tid & 31, warp = tid >> 5;
    const int global_warp = int(blockIdx.x) * 16 + warp;
    const int warps = int(gridDim.x) * 16;

    // All projection/conv head jobs are independent. Owning a complete
    // 128-channel qkv head lets this CTA convolve it after only a CTA barrier;
    // the 64 z rows per job cover ZV once with the same work on each CTA.
    // Beyond the 64 projection/conv jobs, otherwise idle CTAs can compute
    // the narrow F32 AB product while dense producers start immediately.
    // Smaller grids keep the ordinary spread so every AB row is covered.
    if (int(gridDim.x) > C / S) {
        if (int(blockIdx.x) >= C / S) {
            const int ab_warp = (int(blockIdx.x) - C / S) * 16 + warp;
            const int ab_warps = (int(gridDim.x) - C / S) * 16;
            for (int job = ab_warp; job < 2 * HV * NC; job += ab_warps) ab(a, job, lane);
        }
    } else {
        for (int job = global_warp; job < 2 * HV * NC; job += warps) ab(a, job, lane);
    }
    for (int head = int(blockIdx.x); head < C / S; head += int(gridDim.x)) {
        for (int r = warp; r < S; r += 16) {
            const int row = head * S + r;
            float value[NC];
            row_dot<TYPE, NC, N>(a.wqkv, a.input_q, row, lane, value);
            if (lane == 0) {
#pragma unroll
                for (int t = 0; t < NC; ++t) a.qkv[size_t(t) * C + row] = value[t];
            }
        }
        for (int r = warp; r < S / 2; r += 16) {
            const int row = head * (S / 2) + r;
            float value[NC];
            row_dot<TYPE, NC, N>(a.wz, a.input_q, row, lane, value);
            if (lane == 0) {
#pragma unroll
                for (int t = 0; t < NC; ++t) a.z[size_t(t) * ZV + row] = value[t];
            }
        }
        __syncthreads();
        conv<NC>(a, head, sh);
    }
    // Every recurrence reads its q/k heads and its own v/z/AB rows, all of
    // which are now complete. No per-head polling or publication counters.
    grid.sync();
    for (int head = int(blockIdx.x); head < HV; head += int(gridDim.x)) rec<NC>(a, head, sh);
    grid.sync();

    // Recurrence's rg==0 warps produced the exact Q8_1 image above. Its
    // cooperative barrier publishes both FP32 outputs and quantized blocks.
    for (int row = global_warp; row < N; row += warps) {
        float value[NC];
        row_dot<TYPE, NC, ZV>(a.wo, a.out_q, row, lane, value);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < NC; ++t) a.out[size_t(t) * N + row] = value[t];
        }
    }
}

inline bool shape(const Args& a) {
    return (a.type == 8 || a.type == 14) && a.T >= 1 && a.T <= 4 && a.n == N && a.hk == HK && a.hv == HV;
}

template<int TYPE, int NC>
inline cudaError_t prepare_one(Plan& p, int max_blocks) {
    int resident = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, chain<TYPE, NC>, 512, 0);
    if (e != cudaSuccess) return e;
    cudaFuncAttributes attributes{};
    e = cudaFuncGetAttributes(&attributes, chain<TYPE, NC>);
    if (e != cudaSuccess) return e;
    constexpr int ti = TYPE == 8 ? 0 : 1;
    p.blocks[ti][NC - 1] = std::min(max_blocks > 0 ? max_blocks : p.sms, resident * p.sms);
    p.registers[ti][NC - 1] = attributes.numRegs;
    p.shared_bytes[ti][NC - 1] = attributes.sharedSizeBytes;
    return cudaSuccess;
}
} // namespace detail

inline cudaError_t prepare(Plan& p, const Args& a, int max_blocks = 0) {
    p = {};
    if (!detail::shape(a) || max_blocks < 0) return cudaErrorNotSupported;
    cudaError_t e = cudaGetDevice(&p.device);
    if (e != cudaSuccess) return e;
    e = cudaDeviceGetAttribute(&p.cooperative, cudaDevAttrCooperativeLaunch, p.device);
    if (e != cudaSuccess) return e;
    if (!p.cooperative) return cudaErrorNotSupported;
    e = cudaDeviceGetAttribute(&p.sms, cudaDevAttrMultiProcessorCount, p.device);
    if (e != cudaSuccess) return e;
#define Q36G_PREP(TYPE,NC) e = detail::prepare_one<TYPE,NC>(p,max_blocks); if(e != cudaSuccess) return e;
    if (a.type == 8) {
        Q36G_PREP(8,1) Q36G_PREP(8,2) Q36G_PREP(8,3) Q36G_PREP(8,4)
    } else {
        Q36G_PREP(14,1) Q36G_PREP(14,2) Q36G_PREP(14,3) Q36G_PREP(14,4)
    }
#undef Q36G_PREP
    return p.blocks[a.type == 8 ? 0 : 1][a.T - 1] > 0 ? cudaSuccess : cudaErrorNotSupported;
}

inline bool supported(const Plan& p, const Args& a) {
    return detail::shape(a) && p.cooperative && p.blocks[a.type == 8 ? 0 : 1][a.T - 1] > 0 &&
           a.wqkv && a.wz && a.wo && a.input && a.input_q && a.wa && a.wb && a.dt && a.ssm_a && a.conv_w && a.norm &&
           a.hist && a.state && a.qkv && a.z && a.alpha && a.beta && a.hc && a.gdn_y && a.out && a.out_q;
}

template<int TYPE, int NC>
inline cudaError_t launch_t(const Plan& p, const Args& a, cudaStream_t stream) {
    Args copy = a;
    void* parameters[] = { &copy };
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(detail::chain<TYPE, NC>),
                                      dim3(p.blocks[TYPE == 8 ? 0 : 1][NC - 1]), dim3(128,4), parameters, 0, stream);
}

inline cudaError_t launch(const Plan& p, const Args& a, cudaStream_t stream) {
    if (!supported(p,a)) return cudaErrorNotSupported;
#define Q36G_CASE(NC) case NC: return a.type == 8 ? launch_t<8,NC>(p,a,stream) : launch_t<14,NC>(p,a,stream);
    switch(a.T) { Q36G_CASE(1) Q36G_CASE(2) Q36G_CASE(3) Q36G_CASE(4) }
#undef Q36G_CASE
    return cudaErrorNotSupported;
}
} // namespace q36g
#endif
