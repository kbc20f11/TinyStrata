#pragma once

#include "strata/qwen35/hybrid_buffer_policy.hpp"
#include <cuda_runtime.h>

namespace strata::qwen35::hybrid_buffer_policy {

// cuda/hip aliases are supplied by the including engine/test translation unit.
inline cudaError_t upload(float* device_values, int32_t* device_map,
                          const float* host_values, const int32_t* host_map,
                          int capacity, int token_rows, size_t output_entries,
                          size_t top_k, size_t width, cudaStream_t stream) {
    Transfer bytes;
    if (!transfer(capacity, token_rows, output_entries, top_k, width, bytes) ||
        !device_map || !host_map ||
        (bytes.values_bytes && (!device_values || !host_values)))
    {
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
        return hipErrorInvalidValue;
#else
        return cudaErrorInvalidValue;
#endif
    }
    if (bytes.values_bytes) {
        const auto status = cudaMemcpyAsync(device_values, host_values, bytes.values_bytes,
                                             cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) return status;
    }
    return cudaMemcpyAsync(device_map, host_map, bytes.map_bytes,
                           cudaMemcpyHostToDevice, stream);
}

} // namespace strata::qwen35::hybrid_buffer_policy
