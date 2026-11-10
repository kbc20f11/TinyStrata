#pragma once
#include <cstdint>

namespace strata::gemma4::prefill_attention {
inline constexpr int kKeyTile = 32; // shared with the independent tile-rounding unit oracle
// Checks the compiled image and active device; no allocation. Initial implementation supports HIP gfx1200.
bool available();

// D64/GQA8, full causal attention with a learned sink. Q/out: [row][head][64].
// INT8 KV codes: [key][kv_head][64], FP16 scales: [key][kv_head][2].
// Row r attends keys [0,pos+r]. Keys [0,pos+rows) must be mapped/initialized.
// Q, dequantized K/V, and softmax weights round to FP16; products and online sums stay FP32.
// No scratch or allocation. Unsupported/invalid host geometry returns false before any launch.
// The production caller keeps small decode/window rows on their existing path.
bool run(const float* q, const int8_t* k_codes, const uint16_t* k_scales,
         const int8_t* v_codes, const uint16_t* v_scales, const float* sinks, float* out,
         int heads, int kv_heads, int64_t pos, int rows, float scale, void* stream);
} // namespace strata::gemma4::prefill_attention
