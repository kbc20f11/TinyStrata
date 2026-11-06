// Packed register bytes are extracted without taking the register's address.
// Adapted from jesdga95/basalt, MIT; see NOTICE.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace strata::kernels {

__device__ __forceinline__ uint32_t packed_byte(uint32_t word, int index) {
    return (word >> (8 * index)) & 0xffu;
}
__device__ __forceinline__ uint32_t packed_byte(int2 words, int index) {
    return packed_byte(index < 4 ? uint32_t(words.x) : uint32_t(words.y), index & 3);
}
__device__ __forceinline__ uint32_t packed_byte(uint2 words, int index) {
    return packed_byte(index < 4 ? words.x : words.y, index & 3);
}
__device__ __forceinline__ uint32_t pack_scale_min(const uint16_t aux[2]) {
    return uint32_t(aux[0]) | (uint32_t(aux[1]) << 16);
}

} // namespace strata::kernels
