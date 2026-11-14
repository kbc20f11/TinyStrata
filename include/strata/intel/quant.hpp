// Scalar GGUF block interpretation for the initial Intel port. The block
// definitions and codebooks are the project's pinned llama.cpp definitions.
// Formula reference: ggml-quants.c (MIT; third_party/ggml/LICENSE and NOTICE).
#pragma once
#include <sycl/sycl.hpp>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

namespace strata::intel::quant {
inline float half(uint16_t value) { return float(sycl::bit_cast<sycl::half>(value)); }
template<int Type> inline float value(const uint8_t* row, size_t col) {
    static_assert(Type == 0 || Type == 1 || Type == 30 || Type == 8 || Type == 10 || Type == 11 ||
                  Type == 14 || Type == 21 || Type == 22 || Type == 23, "Unsupported Intel GGUF block type");
    if constexpr (Type == 0) return reinterpret_cast<const float*>(row)[col];
    if constexpr (Type == 1) return half(reinterpret_cast<const uint16_t*>(row)[col]);
    if constexpr (Type == 30) return sycl::bit_cast<float>(uint32_t(reinterpret_cast<const uint16_t*>(row)[col]) << 16);
    if constexpr (Type == 8) {
        const auto& b = reinterpret_cast<const block_q8_0*>(row)[col / 32];
        return half(b.d) * b.qs[col % 32];
    }
    if constexpr (Type == 14) {
        const auto& b = reinterpret_cast<const block_q6_K*>(row)[col / 256];
        const int i = col % 256, segment = i / 128, part = (i % 128) / 32, lane = i % 32;
        const int low = (b.ql[segment * 64 + (part % 2) * 32 + lane] >> (part >= 2 ? 4 : 0)) & 15;
        const int high = (b.qh[segment * 32 + lane] >> (2 * part)) & 3;
        return half(b.d) * b.scales[segment * 8 + part * 2 + lane / 16] * ((low | high << 4) - 32);
    }
    if constexpr (Type == 10) {
        const auto& b = reinterpret_cast<const block_q2_K*>(row)[col / 256];
        const int i = col % 256;
        const int bits = (b.qs[(i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
        const uint8_t scale = b.scales[i / 16];
        return half(b.data.data.d) * (scale & 15) * bits - half(b.data.data.dmin) * (scale >> 4);
    }
    if constexpr (Type == 11) {
        const auto& b = reinterpret_cast<const block_q3_K*>(row)[col / 256];
        const int i = col % 256, s = i / 16;
        const int low = (b.scales[s % 8] >> (4 * (s / 8))) & 15;
        const int high = (b.scales[8 + s % 4] >> (2 * (s / 4))) & 3;
        const int bits = (b.qs[(i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
        const int correction = (b.hmask[i % 32] & (1 << (i / 32))) ? 0 : 4;
        return half(b.d) * ((low | high << 4) - 32) * (bits - correction);
    }
    if constexpr (Type == 22) {
        const auto& b = reinterpret_cast<const block_iq2_s*>(row)[col / 256];
        const int i = col % 256, group = i / 32, octet = (i % 32) / 8, lane = i % 8;
        const int index = b.qs[group * 4 + octet] | (((b.qh[group] >> (2 * octet)) & 3) << 8);
        const int grid = (iq2s_grid[index] >> (8 * lane)) & 255;
        const int scale = (b.scales[group] >> (octet >= 2 ? 4 : 0)) & 15;
        const float sign = (b.qs[32 + group * 4 + octet] & (1 << lane)) ? -1.f : 1.f;
        return half(b.d) * (.5f + scale) * .25f * grid * sign;
    }
    if constexpr (Type == 21) {
        const auto& b = reinterpret_cast<const block_iq3_s*>(row)[col / 256];
        const int i = col % 256, group = i / 32, octet = (i % 32) / 8, lane = i % 8;
        const int nibble = 2 * octet + lane / 4;
        const int index = b.qs[group * 8 + nibble] | (((b.qh[group] >> nibble) & 1) << 8);
        const int grid = (iq3s_grid[index] >> (8 * (lane % 4))) & 255;
        const int scale = (b.scales[group / 2] >> (4 * (group % 2))) & 15;
        const float sign = (b.signs[group * 4 + octet] & (1 << lane)) ? -1.f : 1.f;
        return half(b.d) * (1 + 2 * scale) * grid * sign;
    }
    if constexpr (Type == 23) {
        const auto& b = reinterpret_cast<const block_iq4_xs*>(row)[col / 256];
        const int i = col % 256, group = i / 32, lane = i % 32;
        const int scale = ((b.scales_l[group / 2] >> (4 * (group % 2))) & 15) | (((b.scales_h >> (2 * group)) & 3) << 4);
        const int code = (b.qs[group * 16 + lane % 16] >> (lane >= 16 ? 4 : 0)) & 15;
        return half(b.d) * (scale - 32) * kvalues_iq4nl[code];
    }
    return 0; // Type is restricted by the host dispatch, never a runtime fallback.
}
} // namespace strata::intel::quant
