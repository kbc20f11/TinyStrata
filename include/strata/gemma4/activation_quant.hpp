#pragma once

#include "strata/gemma4/kernels.hpp"

namespace strata::gemma4::activation_quant {

// Initial implementation is HIP-only. CUDA returns false without launching;
// its native quantizer's fast-math compilation requires separate parity proof.
bool available();

// gate_up[row] = [width gate values | width up values]. ids[row] selects the
// expert's bias row, not an activation row. IDs must address valid bias rows.
// Same bytes as swiglu_oai_ids + gemma_native_quantize_q8_1 of the flattened
// activation rows. width must be a positive multiple of32. No scratch/allocation.
// Writes rows * width/32 *36 bytes. It does not materialize float activations.
bool q8_1_ids(const float* gate_up, void* quant, int width, int rows,
              const int32_t* ids, const float* bias_gate, const float* bias_up,
              float alpha, float limit, void* stream);

// Same native Q8_1 bytes for1..32 cached slots. Only eb.g/eb.u are used; their
// active entries are device pointers, copied by value as kernel arguments.
bool q8_1_slots(const float* gate_up, void* quant, int width, int rows,
                const ExpertBias& eb, float alpha, float limit, void* stream);

// Same bytes as swiglu_oai_ids + Mmq::quantize(type=MXFP4,ids=null,ld=width).
// width is a positive multiple of4. Output is the pinned ggml MMQ D4 layout:
// 128 codes +4 FP32 scales per144-byte block, block-major/row-interleaved,
// columns padded to512. Writes rows *pad512(width)/128 *144 bytes; the extra
//128-block allocation tail in Mmq::q8_bytes is left untouched. No scratch.
// Other weight formats/layouts must use the existing separate producer.
bool mmq_mxfp4_ids(const float* gate_up, void* quant, int width, int rows,
                   const int32_t* ids, const float* bias_gate, const float* bias_up,
                   float alpha, float limit, void* stream);

} // namespace strata::gemma4::activation_quant
