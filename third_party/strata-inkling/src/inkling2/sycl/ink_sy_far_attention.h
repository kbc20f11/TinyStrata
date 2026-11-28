// Far-global-attention approach adapted from Gufo (MIT), cead7b617d481beb61b84f76e6b316089fa2f294:
// https://github.com/gufo-org/gufo/commit/cead7b617d481beb61b84f76e6b316089fa2f294
// Copyright (c) 2026 gufo contributors. See LICENSE.gufo in this directory.
// Only the zero-bias/idle-rescale idea is shared; SYCL keeps its existing SG16
// dot products, causal key traversal, sink, FP16 cache and softmax reductions.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ink_sycl {

inline bool parse_far_attention_mode(const char* value) {
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("STRATA_REVIEW_INKLING_FAR_ATTN must be exactly 0 or 1");
}

inline bool far_attention_eligible(bool enabled, int tokens, bool swa, int relative_extent) {
    return enabled && !swa && tokens >= 16 && relative_extent == 1024;
}

inline bool far_attention_cell(int query_position, int key_position, int relative_extent) {
    return query_position >= relative_extent && key_position <= query_position - relative_extent;
}

struct FarAttentionDispatches {
    std::atomic<uint64_t> candidate{0};
    std::atomic<uint64_t> legacy{0};
};

inline FarAttentionDispatches& far_attention_dispatches() {
    static FarAttentionDispatches counters;
    return counters;
}

} // namespace ink_sycl
