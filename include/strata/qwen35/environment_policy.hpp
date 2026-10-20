// TinyStrata environment choices; explicit settings always take precedence.
#pragma once
#include <cstddef>
#include "strata/common/context_policy.hpp"

namespace strata::qwen35::environment_policy {
constexpr int kAutomaticKv = -1;
constexpr int kInt8Kv = 2;
constexpr int kNativeFp8Kv = 3;

// Installed capacity determines context. Free memory determines Expert placement.
constexpr int context_for_total_vram(std::size_t total_bytes) {
    return strata::context_policy::for_total_vram(total_bytes);
}

constexpr int resolve_context(int requested, std::size_t total_bytes) {
    return requested == 0 ? context_for_total_vram(total_bytes) : requested;
}

// `native_fp8_available` includes the active device, the compiled kernel image
// and the shared-memory requirement, not only the GPU product name.
constexpr int resolve_kv(int requested, bool supported_shape, bool native_fp8_available) {
    return requested != kAutomaticKv ? requested
         : supported_shape && native_fp8_available ? kNativeFp8Kv : kInt8Kv;
}
} // namespace strata::qwen35::environment_policy
