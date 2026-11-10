// Full-vocabulary categorical sampling. The caller owns the RNG and all GPU storage.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::gemma4 {

enum FullVocabStatus : uint32_t {
    kSampleOk = 0,
    kSampleNonFinite = 1,
    kSampleInvalidTotal = 2,
};

struct FullVocabResult {
    int32_t token;
    uint32_t status;
    double total;
    double probability;
};
static_assert(sizeof(FullVocabResult) == 24);

// Storage is reusable and must be at least double-aligned. No allocation occurs in the launch wrapper.
size_t full_vocab_sample_workspace_bytes(int vocab);

// All finite logits contribute exp((logit - max) / temperature), in double precision, without a cutoff.
// uniform is one caller-supplied draw in [0, 1); zero selects the first positive mass.
// The three launches and result are ordered on stream. device_result remains on the GPU until the caller
// copies it. A nonfinite logit produces token=-1/status=kSampleNonFinite; it is never silently ignored.
// Invalid host arguments throw before any launch. A launch failure throws, without consuming RNG state.
void sample_full_vocab(const float* logits, int vocab, double temperature, double uniform,
                       void* workspace, size_t workspace_bytes, FullVocabResult* device_result, void* stream);

} // namespace strata::gemma4
