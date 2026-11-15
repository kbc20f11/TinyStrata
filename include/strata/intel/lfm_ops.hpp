#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::intel {
// Experimental FP32 Q4_K decode only. No activation quantization, scratch,
// altered expert routing, or changes to the existing multi-token/MKL paths.
constexpr bool lfm_q4_fp32_eligible(int type, int tokens, bool enabled) {
    return enabled && type == 12 && tokens == 1;
}
sycl::event lfm_q4_fp32_matvec(sycl::queue&, const uint8_t* weights,
                              const float* input, float* output, int rows, int width);
sycl::event lfm_q4_fp32_selected(sycl::queue&, const uint8_t* weights, const int32_t* ids,
                                const float* input, float* output, int rows, int width,
                                bool separate_inputs);
// LFM's projection is [B,C,x], not Qwen's gated delta network. State stores
// the two preceding B*x values per channel, oldest first. No activation is
// applied to C or to the three-tap convolution.
sycl::event lfm_shortconv(sycl::queue&, const float* bcx, float* state,
                         const float* weights, float* output, int channels, int tokens);
// Selection uses sigmoid(logit)+bias; mixture weights use the unbiased
// sigmoid, with the model's optional normalization/scale. Ties select the
// lower expert index. The pinned LFM has 64 experts and retains all 4 picks.
sycl::event lfm_route(sycl::queue&, const float* logits, const float* bias,
                     int32_t* ids, float* weights, int tokens, int experts,
                     int picks, bool normalize = true, float scale = 0.f);
}
