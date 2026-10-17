// GPU kernel launch helpers. Qwen3.6 uses ordinary stream-ordered launches.
#pragma once

#include <cuda_runtime.h>

#include <utility>

namespace strata::kernels {

#if (defined(__CUDACC__) || defined(__HIPCC__))

// STRATA_PDL_RESTRICT never goes on a __global__ signature: the host pass and the stub must see the same signature on
// every architecture (MSVC rejects the template stubs otherwise, and so does g++ when the stub is generated from a
// device pass).  A kernel takes plain pointers and declares `const T* STRATA_PDL_RESTRICT p = p_;` locals in its body.
// On CUDA the activation pointers lose __restrict__ so the compiler cannot load them before pdl_wait() - but only
// where PDL can run (sm_70 and newer here).  On Pascal and older (__CUDA_ARCH__ < 700, no PDL) `__restrict__` is what
// lets the compiler use the read-only data path (LDG.CI): without it decode on sm_61 halves (#1469).
#if defined(__HIPCC__)
#define STRATA_PDL_RESTRICT __restrict__
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define STRATA_PDL_RESTRICT __restrict__
#else
#define STRATA_PDL_RESTRICT
#endif

#if defined(__HIPCC__)
inline constexpr bool kPdlPrefetch = false;   // HIP: no PDL, the kernels keep their plain loops
#else
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
inline constexpr bool kPdlPrefetch = false;   // Pascal and older: no PDL, the prefetch only costs registers (#1469)
#else
inline constexpr bool kPdlPrefetch = true;    // CUDA: the weights a kernel can load before pdl_wait() are loaded there
#endif
#endif

__device__ __forceinline__ void pdl_wait() {
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

/// Ordinary launch with the same stream order as the upstream Qwen3.6 path.
template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args&&... args) {
    kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
}

#endif  // __CUDACC__ || __HIPCC__

}  // namespace strata::kernels
