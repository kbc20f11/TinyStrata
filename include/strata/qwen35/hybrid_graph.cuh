// CPU/GPU graph handoff adapted from Niko1221/Strata (MIT).
// Source: src/core/session.cpp and src/kernels/cuda/elementwise.cu at
// 296b59d2f73c8f04a805df29c2fa97655adace78; see LICENSE and NOTICE.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// TinyStrata adaptations: fixed per-layer rings reset between replays,
// multi-token payloads, CPU-miss-only merging, and abortable device waits.
#pragma once

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::qwen35::hybrid_graph {

// Both the host pointer and the device alias must refer to cudaHostAllocMapped
// memory. Reset all three fields only after the preceding graph has finished.
// A captured ring is a layer number, starting at 1, NOT a replay counter.
struct Flags {
    uint32_t ready;
    uint32_t done;
    uint32_t abort;
};
static_assert(sizeof(Flags) == 3 * sizeof(uint32_t), "mapped flag ABI");

namespace detail {
__device__ __forceinline__ uint32_t load_flag(const uint32_t* flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // Explicit system acquire: the producer is a CPU, and a cached read must
    // never keep a spinning block from noticing host completion/cancellation.
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
    return value;
#else
    const uint32_t value = *reinterpret_cast<const volatile uint32_t*>(flag);
    __threadfence_system();
    return value;
#endif
}

__device__ __forceinline__ void spin_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
}

static __global__ void publish_kernel(const float* __restrict__ hidden,
                                     const int32_t* __restrict__ ids,
                                     float* mapped_hidden, int32_t* mapped_ids,
                                     Flags* flags, uint32_t ring, int rows,
                                     int N, int K) {
    __shared__ int cancelled;
    if (threadIdx.x == 0) cancelled = load_flag(&flags->abort) != 0;
    __syncthreads();
    if (cancelled) return;

    // Volatile stores target mapped host memory. Every publishing thread must
    // order its own payload writes before thread 0 publishes the ready flag.
    volatile float* out_hidden = mapped_hidden;
    volatile int32_t* out_ids = mapped_ids;
    for (size_t i = threadIdx.x; i < size_t(rows) * N; i += blockDim.x)
        out_hidden[i] = hidden[i];
    for (size_t i = threadIdx.x; i < size_t(rows) * K; i += blockDim.x)
        out_ids[i] = ids[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint32_t*>(&flags->ready) = ring;
        __threadfence_system();
    }
}

static __global__ void wait_and_merge_kernel(float* parts,
                                            const float* mapped_cpu,
                                            const int32_t* mapped_miss,
                                            Flags* flags, uint32_t ring,
                                            int N) {
    __shared__ int merge;
    if (threadIdx.x == 0) {
        merge = 0;
        for (;;) {
            if (load_flag(&flags->abort) != 0) break;
            if (load_flag(&flags->done) >= ring) {
                __threadfence_system();
                // Abort wins over a completion already visible to this block.
                // An aborted handoff never reads the mask or CPU result rows.
                if (load_flag(&flags->abort) == 0)
                    merge = reinterpret_cast<const volatile int32_t*>(mapped_miss)[blockIdx.x] != 0;
                break;
            }
            spin_pause();
        }
    }
    __syncthreads();
    if (!merge) return; // GPU-resident rows stay bit-for-bit untouched.

    __threadfence_system();
    const size_t base = size_t(blockIdx.x) * N;
    const volatile float* source = mapped_cpu;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        parts[base + i] = source[base + i];
}
} // namespace detail

// All mapped_* pointers and flags passed here must be DEVICE aliases. Their
// corresponding host allocations must stay at the same addresses until every
// captured graph referencing them has been destroyed. The caller checks launch
// errors using its usual CUDA error handling (also during graph capture).
inline void publish(const float* hidden, const int32_t* ids,
                    float* mapped_hidden, int32_t* mapped_ids, Flags* flags,
                    uint32_t ring, int rows, int N, int K, cudaStream_t stream) {
    detail::publish_kernel<<<1, 256, 0, stream>>>(hidden, ids, mapped_hidden,
                                                 mapped_ids, flags, ring,
                                                 rows, N, K);
}

// The host computes all CPU misses, writes the corresponding mask and rows,
// then release-publishes done=ring. For an all-hit layer it still publishes done
// with an all-zero mask. Cancellation publishes abort!=0 and drains the stream;
// no host buffers may be reset or freed while blocks can still reference them.
inline void wait_and_merge(float* parts, const float* mapped_cpu,
                           const int32_t* mapped_miss, Flags* flags,
                           uint32_t ring, int entries, int N,
                           cudaStream_t stream) {
    if (entries <= 0) return;
    detail::wait_and_merge_kernel<<<entries, 256, 0, stream>>>(parts, mapped_cpu,
                                                             mapped_miss, flags,
                                                             ring, N);
}

} // namespace strata::qwen35::hybrid_graph
#endif // NVIDIA CUDA only
