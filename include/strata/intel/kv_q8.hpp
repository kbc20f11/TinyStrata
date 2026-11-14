#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace strata::intel {
struct Q8Kv {
    int8_t *key = nullptr, *value = nullptr;
    sycl::half *key_scale = nullptr, *value_scale = nullptr;
    int context = 0, heads = 0;
};
// Q8 codes with one FP16 scale per 32 coefficients. Token-major FP32 inputs,
// head-major cache; no model-weight conversion or persistent FP32 cache copy.
sycl::event store_q8_kv(sycl::queue& q, const float* key, const float* value, Q8Kv cache,
    const int32_t* position, int tokens, int32_t* error);
sycl::event dequantize_q8_kv(sycl::queue& q, Q8Kv cache, float* key, float* value, int visible);
// 1..8 token rows with independent causal positions. Query/gate/output are
// token-major; scratch needs tokens*heads*context scores and
// tokens*heads*ceil(context/256)*256 partial coefficients.
// actual_visible is the CPU-known exclusive visible end; 0 retains the old
// path. Graph capture/replay passes 0, never its padded extent as visibility.
sycl::event decode_attention_q8(sycl::queue& q, const float* query, const float* gate, Q8Kv cache,
    float* scores, float* partial, float* output, int heads, const int32_t* position,
    int extent, sycl::event* stages = nullptr, int tokens = 1, int actual_visible = 0);
}
