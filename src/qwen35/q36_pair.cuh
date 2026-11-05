// The paired GDN projection schedule follows Basalt's independent qkv/z
// projection pairing; arithmetic remains q36k::q8_row_dot (see q36_kernels.cuh).
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "q36_kernels.cuh"
#include <cstdlib>
#include <cstring>

namespace q36k {
inline int q8_pair_max_warps() {
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
    // Both projections share one HIP grid. Keep its normal total at 1024
    // warps, matching the existing single-projection large-grid bound.
    // The diagnostic override is per projection and is read only once.
    static const int limit = [] {
        const char* value = std::getenv("TINYSTRATA_Q36_Q8_PAIR_MAX_WARPS");
        if (value && std::strcmp(value, "256") == 0) return 256;
        if (value && std::strcmp(value, "1024") == 0) return 1024;
        return 512;
    }();
    return limit;
#else
    return 1024;
#endif
}

template <int NC, int NJ>
__global__ void __launch_bounds__(256) k_q8_rows_pair(const uint8_t* __restrict__ wa,
                                                     const uint8_t* __restrict__ wb,
                                                     const QB* __restrict__ x,
                                                     float* __restrict__ ya,
                                                     float* __restrict__ yb,
                                                     int nb, int na, int nb_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int stride = gridDim.x * (blockDim.x >> 5);
    const bool second = blockIdx.y != 0;
    const int rows = second ? nb_out : na;
    const uint8_t* weights = second ? wb : wa;
    float* output = second ? yb : ya;
    for (int row = blockIdx.x * (blockDim.x >> 5) + warp; row < rows; row += stride) {
        float result[NC];
        q8_row_dot<NC, NJ, false>(weights, nullptr, x, row, nb, lane, result);
        if (lane == 0)
#pragma unroll
            for (int c = 0; c < NC; ++c) output[size_t(c) * rows + row] = result[c];
    }
}

template <int NC, int NJ>
inline void launch_q8_rows_pair(const void* wa, const void* wb, const void* x,
                                float* ya, float* yb, int nb, int na, int nb_out,
                                cudaStream_t stream) {
    const int maximum = na > nb_out ? na : nb_out;
    const int limit = q8_pair_max_warps();
    const int rows = maximum < limit ? maximum : limit;
    k_q8_rows_pair<NC, NJ><<<dim3(unsigned((rows + 7) / 8), 2), 256, 0, stream>>>(
        static_cast<const uint8_t*>(wa), static_cast<const uint8_t*>(wb),
        static_cast<const QB*>(x), ya, yb, nb, na, nb_out);
}
} // namespace q36k
#endif
