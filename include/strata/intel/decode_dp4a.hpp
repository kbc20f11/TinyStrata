#pragma once
#include <sycl/sycl.hpp>
#include <memory>
#include "strata/intel/qwen_ops.hpp"
namespace strata::intel {
// Explicit candidate: Q8 activations, unchanged GGUF weights, FP32 sums.
sycl::event quantize_decode_input(sycl::queue& q, const float* input, int8_t* values,
                                  float* scales, int width, int rows, int32_t* error = nullptr);
class DecodeDp4a {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit DecodeDp4a(sycl::queue& q, bool xmx_windows = false, bool dense_split_k = false);
    ~DecodeDp4a();
    static bool supports(int type) { return type == 8 || type == 10 || type == 11 || type == 14 || type == 18 || type == 21 || type == 22 || type == 23; }
    bool valid() const noexcept;
    sycl::event quantization_event() const;
    sycl::event dense(int type, const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens = 1);
    sycl::event dense_interleaved_q6(const uint8_t* weights, const float* input, float* output, int rows, int width,
                                   int tokens = 1, bool split_k = false, sycl::event* compute_stage = nullptr);
    // Draft proposal only. Caller supplies validated sorted/unique real IDs.
    sycl::event selected_vocabulary_q6(const uint8_t* weights, const float* input, float* output,
                                     int vocabulary, int width, const int32_t* ids, int count);
    sycl::event selected(int type, const uint8_t* weights, const int32_t* ids, int picks,
                         const float* input, float* output, int rows, int width, bool separate, int tokens = 1);
    sycl::event grouped(int type, const uint8_t* weights, ExpertPlan plan,
                        const float* input, float* output, int rows, int width, bool separate, bool interleaved = false);
};
} // namespace strata::intel
