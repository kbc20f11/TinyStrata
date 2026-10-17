// Portable E4M3FN storage conversion, including GPUs without native FP8 math.
// Round to nearest, ties to even; saturate finite overflow at +/-448.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::qwen35 {
__device__ __forceinline__ uint8_t fp8_encode(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint8_t sign = (uint8_t) ((bits >> 24) & 0x80);
    const uint32_t mag = bits & 0x7fffffffu;
    if (mag > 0x7f800000u) return sign | 0x7f; // NaN
    if (mag >= 0x43e00000u) return sign | 0x7e; // 448, infinity
    if (mag < 0x3c800000u) // subnormal: spacing 2^-9
        return sign | (uint8_t) __float2int_rn(__uint_as_float(mag) * 512.0f);
    const uint32_t rounded = (mag + 0x7ffffu + ((mag >> 20) & 1u)) >> 20;
    return sign | (uint8_t) (rounded - 960u); // bias 127 -> 7
}

__device__ __forceinline__ float fp8_decode(uint8_t value) {
    const uint32_t sign = (uint32_t) (value & 0x80) << 24;
    const uint32_t mag = value & 0x7f;
    if (mag == 0x7f) return __uint_as_float(sign | 0x7fc00000u);
    if (mag < 8) {
        const float v = (float) mag * (1.0f / 512.0f);
        return __uint_as_float(__float_as_uint(v) | sign);
    }
    return __uint_as_float(sign | ((mag + 960u) << 20));
}
} // namespace strata::qwen35
