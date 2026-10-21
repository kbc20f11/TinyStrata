// Experimental dense native FP8 attention. See docs/fp8-native-attention.md.
#pragma once
#include <cstdint>
#if defined(STRATA_USE_HIP)
#include <stdexcept>
#endif
namespace strata::qwen35::fp8_attention {
struct Cache {
    const uint8_t *k_q, *v_q;
    const uint16_t *k_scale, *v_scale;
};
#if defined(STRATA_USE_HIP)
inline void run_split(const float *, Cache, const int32_t *, int, int, int, int, int, int, float *, float *, float *,
                      void *) { throw std::runtime_error("native FP8 attention is unavailable on HIP; use INT8 KV"); }
inline bool available() { return false; }
inline void run(const float *, Cache, const int32_t *, int, int, int, float *, void *) {
    throw std::runtime_error("native FP8 attention is unavailable on HIP; use INT8 KV");
}
#else
// Call before graph capture. Native FP8 MMA requires CUDA sm_89 or newer.
bool available();
// H = 8 * kv_heads, head dimension 256. Causal Q rows start at *pos0.
// KV is head-major E4M3, one stored FP16 scale per 64 elements.
// The caller owns bounds (0 <= *pos0, *pos0 + rows <= context).
void run(const float *, Cache, const int32_t *, int rows, int kv_heads, int context, float *, void *stream);
// Native FP8 split-K for decode/verification; 256 KV positions per chunk.
// Outputs match attention.cuh k_attn_merge (FP32 partials, natural-log maxima).
void run_split(const float *, Cache, const int32_t *, int row_offset, int rows, int kv_heads, int context,
               int grid_chunks, int chunk_stride, float *partial, float *maxima, float *sums, void *stream);
#endif
} // namespace strata::qwen35::fp8_attention
