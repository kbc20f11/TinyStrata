// Resident INT8 attention adapted from aflin/momoe (MIT); see NOTICE.
#pragma once
#include <cstdint>
#if defined(STRATA_USE_HIP)
#include <stdexcept>
#endif

namespace strata::qwen35::int8_attention {
struct Cache {
    const int8_t *k_q = nullptr, *v_q = nullptr;
    const uint16_t *k_scale = nullptr, *v_scale = nullptr;
    // Kept for the source kernel's compile-time format branches; no FP16 entry point is instantiated.
    const uint16_t *k_pool = nullptr, *v_pool = nullptr;
};
#if defined(STRATA_USE_HIP)
inline bool available() { return false; }
inline void run(const float*, Cache, const int32_t*, int, int, int, float*, void*) {
    throw std::runtime_error("the CUDA INT8 tiled attention kernel is unavailable on HIP; use the INT8 staging fallback");
}
#else
bool available(); // Call once before allocating prompt scratch or capturing graphs. Requires sm_80+.
void run(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
         float* output, void* stream);
#endif
} // namespace strata::qwen35::int8_attention
