#pragma once
#include "strata/intel/qwen_ops.hpp"
namespace strata::intel {
inline bool q8_xmx_supports(int type) { return type == 14 || type == 21 || type == 22; }
// INT8 XMX candidate. Q8 inputs and their scales are supplied by DecodeDp4a;
// original compressed weights are decoded per 16-row tile, without a replica.
// Uses a four-token expert plan. Dense DPAS regressed and is not selected.
sycl::event q8_xmx_product(sycl::queue& q, int type, const uint8_t* weights,
    const int8_t* input, const float* input_scales, float* output, int rows, int width,
    ExpertPlan plan, bool separate, const uint32_t* iq2, const uint32_t* iq3, bool interleaved = false);
sycl::event q6_interleaved_product(sycl::queue& q, const uint8_t* weights, const int8_t* input,
    const float* scales, float* output, int rows, int width, int tokens);
// Four K partitions; partial needs 4*tokens*rows floats, bounded to 1 MiB.
sycl::event q6_interleaved_splitk_product(sycl::queue& q, const uint8_t* weights, const int8_t* input,
    const float* scales, float* output, int rows, int width, int tokens, float* partial,
    sycl::event* compute_stage = nullptr);
sycl::event q6_interleaved_selected_product(sycl::queue& q, const uint8_t* weights, const int8_t* input,
    const float* scales, const int32_t* ids, float* output, int vocabulary, int width, int count);
}
