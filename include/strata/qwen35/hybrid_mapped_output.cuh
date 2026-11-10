// Optional eager CPU -> GPU result handoff. No device polling or graph waits.
#pragma once
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>

namespace strata::qwen35::hybrid_mapped_output {

// Explicit coherent mapped allocations are required: accepting plain pinned or
// non-coherent host memory would make repeat writes to the same rows unsafe.
// An unsupported flag/capability fails closed to the caller's ordinary DMA path.
inline bool allocate(void** host, void** device, size_t bytes, const char*& reason) {
    *host = nullptr;
    *device = nullptr;
#if defined(_WIN32) && (defined(__HIPCC__) || defined(STRATA_USE_HIP))
    int ordinal = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&ordinal) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess ||
        !properties.canMapHostMemory) {
        reason = "host mapping capability unavailable";
        (void) cudaGetLastError();
        return false;
    }
    constexpr unsigned wanted = hipHostMallocMapped | hipHostMallocCoherent;
    cudaError_t status = hipHostMalloc(host, bytes, wanted);
    unsigned actual = 0;
    if (status == cudaSuccess) status = hipHostGetFlags(&actual, *host);
    if (status == cudaSuccess &&
        ((actual & wanted) != wanted || (actual & hipHostMallocNonCoherent))) {
        reason = "coherent mapped allocation flags not confirmed";
        status = hipErrorInvalidValue;
    } else if (status != cudaSuccess) {
        reason = "coherent mapped allocation unavailable";
    }
    if (status == cudaSuccess) {
        status = cudaHostGetDevicePointer(device, *host, 0);
        if (status != cudaSuccess || !*device) reason = "mapped device alias unavailable";
    }
    if (status == cudaSuccess && *host && *device) {
        reason = "coherent mapped allocation verified";
        return true;
    }
    if (*host) (void) cudaFreeHost(*host);
    *host = *device = nullptr;
    (void) cudaGetLastError();
    return false;
#else
    (void) bytes;
    reason = "Windows HIP only";
    return false;
#endif
}

// All CPU pool writers have joined before this publication and the following
// stream submission. The allocation itself supplies system coherence; this
// fence only orders the already-complete host stores before that submission.
inline void publish_host_writes() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

} // namespace strata::qwen35::hybrid_mapped_output
