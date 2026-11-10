#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::gemma4::long_prefill_attention {
inline constexpr int kKeyTile = 32;
inline constexpr int kQueryTile = 32;
inline constexpr std::size_t kSharedBytes = 62976;
// True only for an image/device supporting the gfx1200 WMMA specialization,
// 512 threads and the 62,976-byte static-LDS workgroup. No allocation.
bool available();

// Full causal D64/GQA8 Attention with one learned sink; caller chooses the long-context threshold.
// Q/out: [row][head][64]; INT8 KV: [key][kv_head][64]; FP16 scales: [key][kv_head][2].
// Row r reads keys [0,pos+r]. Only keys [0,pos+rows) must be mapped and initialized.
// Same FP16 Q/restored K,V/probability precision and FP32 accumulation as existing prefill.
// No persistent/global workspace or repack. Invalid/unsupported geometry returns false before launch.
// CUDA and other GPU architectures return false. Small decode/window calls retain the existing path.
bool run(const float* q,const int8_t* k_codes,const uint16_t* k_scales,
         const int8_t* v_codes,const uint16_t* v_scales,const float* sinks,float* out,
         int heads,int kv_heads,int64_t pos,int rows,float scale,void* stream);
} // namespace strata::gemma4::long_prefill_attention
