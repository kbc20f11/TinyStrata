// Resident INT8 attention adapted from aflin/momoe (MIT); see NOTICE.
#pragma once
#include <cstdint>

namespace strata::qwen35::int8_attention {
struct Cache {
    const int8_t *k_q = nullptr, *v_q = nullptr;
    const uint16_t *k_scale = nullptr, *v_scale = nullptr;
    // Kept for the source kernel's compile-time format branches; no FP16 entry point is instantiated.
    const uint16_t *k_pool = nullptr, *v_pool = nullptr;
};
// Call before allocating prompt scratch or capturing graphs. CUDA needs sm_80+;
// HIP also checks the selected device against the compiled WMMA architectures.
bool available();
void run(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
         float* output, void* stream);
// Clover preview: every active Q row sees KV [0, visible_end), including the
// entire noisy block. The caller has prepared/mapped those KV rows already.
// The ordinary run() entry point retains its per-query causal upper bounds.
void run_preview(const float* q, Cache cache, const int32_t* pos0, int rows, int kv_heads, int context,
                 int visible_end, float* output, void* stream);
} // namespace strata::qwen35::int8_attention
