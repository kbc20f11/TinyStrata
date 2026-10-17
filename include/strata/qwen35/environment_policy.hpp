// TinyStrata environment choices; explicit settings always take precedence.
#pragma once
#include <cstddef>

namespace strata::qwen35::environment_policy {
constexpr int kAutomaticKv = -1;
constexpr int kInt8Kv = 2;
constexpr int kNativeFp8Kv = 3;

// The free memory is sampled before loading model weights. Other GPU users
// count against this budget, just as they do during automatic expert placement.
constexpr int context_for_free_vram(std::size_t free_bytes) {
    constexpr std::size_t gib = std::size_t{1} << 30;
    return free_bytes <= 8 * gib + gib / 2 ? 65536
         : free_bytes <= 12 * gib + gib / 2 ? 131072 : 262144;
}

constexpr int resolve_context(int requested, std::size_t free_bytes) {
    return requested == 0 ? context_for_free_vram(free_bytes) : requested;
}

// `native_fp8_available` includes the active device, the compiled kernel image
// and the shared-memory requirement, not only the GPU product name.
constexpr int resolve_kv(int requested, bool supported_shape, bool native_fp8_available) {
    return requested != kAutomaticKv ? requested
         : supported_shape && native_fp8_available ? kNativeFp8Kv : kInt8Kv;
}
} // namespace strata::qwen35::environment_policy
