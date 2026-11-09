// Conditional routing payload publication adapted from Niko1221/Strata (MIT),
// a36be1db2973b3e110fa2c7582c6393e2971d107 and
// 0f1390bd11119ae2875913efb4275842d36e142f. See LICENSE and NOTICE.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35::hybrid_publish {

constexpr int kMaxRows = 32;

// Host outputs must have explicit mapped device aliases. The caller records
// an event on this stream and waits for its successful completion before any
// CPU reads. No host flag or GPU/CPU polling protocol is introduced here.
static __global__ void publish_kernel(const float* hidden, const int32_t* ids,
                                      const int32_t* slot_of, float* mapped_hidden,
                                      int32_t* mapped_ids, int width, int top_k,
                                      int n_expert, bool conditional) {
    const size_t row = blockIdx.x;
    bool miss = !conditional || slot_of == nullptr;
    for (int k = threadIdx.x; k < top_k; k += blockDim.x) {
        const size_t entry = row * top_k + k;
        const int32_t id = ids[entry];
        reinterpret_cast<volatile int32_t*>(mapped_ids)[entry] = id;
        if (conditional && slot_of != nullptr &&
            (id < 0 || id >= n_expert || slot_of[id] < 0)) miss = true;
    }
    if (__syncthreads_or(miss)) {
        // Copy stored bits, including signed zero/NaN payloads, without math.
        const uint32_t* in = reinterpret_cast<const uint32_t*>(hidden) + row * width;
        volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(mapped_hidden) + row * width;
        for (int i = threadIdx.x; i < width; i += blockDim.x) out[i] = in[i];
    }
    // Each writer orders its own system-memory stores before kernel/event
    // completion. A fence executed by thread zero alone would not suffice.
    __threadfence_system();
}

inline void publish(const float* hidden, const int32_t* ids, const int32_t* slot_of,
                    float* mapped_hidden, int32_t* mapped_ids, int rows, int width,
                    int top_k, int n_expert, bool conditional, cudaStream_t stream) {
    if (rows < 1 || rows > kMaxRows || width < 1 || top_k < 1 || n_expert < 1 ||
        !hidden || !ids || !mapped_hidden || !mapped_ids)
        throw std::invalid_argument("invalid hybrid routing publish shape or buffer");
    publish_kernel<<<rows, 128, 0, stream>>>(hidden, ids, slot_of, mapped_hidden,
                                           mapped_ids, width, top_k, n_expert, conditional);
}

} // namespace strata::qwen35::hybrid_publish
#endif
