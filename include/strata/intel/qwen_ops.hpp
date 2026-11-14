#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::intel {
struct Q8Block { sycl::half scale; int8_t values[32]; };
static_assert(sizeof(Q8Block) == 34, "GGUF Q8_0 block layout");
// Initial, unfused port kernels. Pointer ownership stays with the caller;
// operations require one in-order queue and GPU-addressable USM allocations.
sycl::event rms_norm(sycl::queue& queue, const float* x, const float* weights,
                     float* y, int rows, int width, float epsilon);
sycl::event q8_matvec(sycl::queue& queue, const Q8Block* weights, const float* x,
                      float* y, int rows, int width);
sycl::event quantized_matvec(sycl::queue& queue, int type, const uint8_t* weights,
                             const float* x, float* y, int rows, int width);
// Token-major input/output. Reuse each decoded weight across four tokens.
sycl::event quantized_matmul(sycl::queue& queue, int type, const uint8_t* weights,
                             const float* x, float* y, int rows, int width, int tokens);
sycl::event dequantize(sycl::queue& queue, int type, const uint8_t* weights, float* output, size_t elements);
sycl::event dequantize_half(sycl::queue& queue, int type, const uint8_t* weights, sycl::half* output, size_t elements);
sycl::event convert_half(sycl::queue& queue, const float* input, sycl::half* output, size_t elements);
sycl::event embedding_lookup(sycl::queue& queue, int type, const uint8_t* weights,
                             const int32_t* ids, float* output, int width, int tokens);
// Exact causal visibility; online FP32 softmax, no materialized score matrix.
// Query/gate [token][head][256], persistent K/V [kv_head][context][256].
sycl::event causal_attention(sycl::queue& queue, const float* queries, const float* gates,
                             const float* keys, const float* values, float* output,
                             int heads, int kv_heads, int context, int position, int tokens);
// Single-query path. Scratch scores [head][context] and partial values
// [head][ceil(context/256)][256]. Position remains a GPU-readable graph input.
sycl::event decode_attention(sycl::queue& queue, const float* queries, const float* gates,
                             const float* keys, const float* values, float* scores, float* partial,
                             float* output, int heads, int kv_heads, int context,
                             const int32_t* position, int extent, sycl::event* stages = nullptr);
sycl::event selected_matvec(sycl::queue& queue, int type, const uint8_t* weights,
                            const int32_t* expert_ids, int picks, const float* input,
                            float* output, int rows, int width, bool separate_inputs, int tokens = 1);
struct ExpertPlan {
    int32_t *counts, *slots, *owners, *offsets, *tiles;
    int experts, tokens, picks;
    int tile_width = 4;
};
// Stable expert grouping; no CPU readback or duplicate expert weight arena.
sycl::event plan_experts(sycl::queue& queue, const int32_t* ids, ExpertPlan plan);
sycl::event selected_matmul(sycl::queue& queue, int type, const uint8_t* weights,
                            ExpertPlan plan, const float* input, float* output,
                            int rows, int width, bool separate_inputs);
// Candidate XMX path: unchanged GGUF storage, FP16 multiplicands, FP32 sums.
sycl::event quantized_matmul_xmx(sycl::queue& queue, int type, const uint8_t* weights,
                                 const float* input, float* output, int rows, int width, int tokens);
sycl::event selected_matmul_xmx(sycl::queue& queue, int type, const uint8_t* weights,
                                ExpertPlan plan, const float* input, float* output,
                                int rows, int width, bool separate_inputs);
// Qwen's normalized q|k|v input, state [key_row][value_head][value_column].
// gate is already softplus(alpha+dt)*ssm_a; beta is already sigmoid(beta).
// Writes raw GDN output; the gated normalization remains a separate operation.
sycl::event gdn_recurrence(sycl::queue& queue, float* state, const float* qkv,
                          const float* gate, const float* beta, float* output,
                          int key_heads, int value_heads, int tokens);
sycl::event gdn_recurrence_tiled(sycl::queue& queue, float* state, const float* qkv,
                                const float* gate, const float* beta, float* output,
                                int key_heads, int value_heads, int tokens);
} // namespace strata::intel
