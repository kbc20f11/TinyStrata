#pragma once
#include "strata/intel/qwen_ops.hpp"
#include <memory>
namespace strata::intel {
class PrefillGemm {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit PrefillGemm(sycl::queue& queue, int context = 1, int max_tokens = 1024);
    ~PrefillGemm();
    sycl::event dense(int type, const uint8_t* weights, const float* input, float* output,
                      int rows, int width, int tokens);
    sycl::event dense_interleaved_q6(const uint8_t* weights, const float* input, float* output,
                      int rows, int width, int tokens);
    sycl::event selected(int type, const uint8_t* weights, size_t expert_bytes, ExpertPlan plan,
                         const float* input, float* output, int rows, int width, bool separate_inputs,
                         bool interleaved = false);
    sycl::event attention(const float* queries, const float* gates, const float* keys, const float* values,
                          float* output, int heads, int kv_heads, int context, int position, int tokens);
    // The KV tail through min(context, ceil((position+tokens)/256)*256) must be
    // zero, including V: zero probabilities must not multiply uninitialized NaNs.
    sycl::event attention_small(const float* queries, const float* gates, const float* keys, const float* values,
                          float* output, int heads, int kv_heads, int context, int position, int tokens);
};
} // namespace strata::intel
