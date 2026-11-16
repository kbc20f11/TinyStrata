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
// Elementwise SiLU(gate)*up permits output==gate; each cell is consumed first.
sycl::event silu_mul(sycl::queue& queue,const float* gate,const float* up,float* output,size_t count);
// Causal convolution consumes each channel's current raw input before writing
// that same cell, so output==input is valid. History remains a separate arena.
sycl::event conv_step_many(sycl::queue& queue,const float* input,float* history,const float* weights,
                           float* output,int key_heads,int value_heads,int tokens,float epsilon);
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
// Qwen head normalization, interleaved t/h/w M-RoPE and physical KV storage.
// The optional position map is [context][3]; rotary pair p uses axis p%3.
// KV addressing always uses the physical cell, including inside a picture.
sycl::event prepare_qwen_attention(sycl::queue& queue, const float* packed_query_gate,
    const float* keys, const float* values, const float* query_norm, const float* key_norm,
    float* queries, float* gates, float* rotated_keys, float* cache_keys, float* cache_values,
    int heads, int kv_heads, int context, int rotary_dimensions, int tokens,
    const int32_t* position, const int32_t* mrope, float base, float epsilon, bool quantized_kv);
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
inline bool xmx_weight_type_supported(int type) {
    return type==0||type==1||type==30||type==8||type==10||type==11||type==14||type==18||type==21||type==22||type==23;
}
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
