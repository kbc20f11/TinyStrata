// Full-vocabulary categorical sampling. The caller owns the RNG and all GPU storage.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::gemma4 {

enum FullVocabStatus : uint32_t {
    kSampleOk = 0,
    kSampleNonFinite = 1,
    kSampleInvalidTotal = 2,
    kSampleInvalidDraft = 3,
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

// Immediately after sample_full_vocab, preserve its normalized distribution on
// the device. The destination must not overlap the workspace. Invalid results
// produce NaNs so a later speculative verifier cannot silently accept them.
void copy_full_vocab_probabilities(const void* workspace, size_t workspace_bytes,
                                   const FullVocabResult* device_result, int vocab,
                                   double* device_out, void* stream);

// Exact speculative rejection for one sampled draft token and its normalized
// full-vocabulary distribution q. Accept with min(1,p(token)/q(token)); otherwise
// draw from normalized max(p-q,0). q must be finite/nonnegative, sum to one to
// double rounding (1e-10 tolerance), and assign positive mass to draft_token.
// The target p uses exactly the ordinary sampler's maximum/probability kernels.
// accepted is separate from result.status (errors set accepted=1 only to guard
// later kernels; callers MUST validate status before inspecting accepted).
// On rejection, result.total/probability describe the residual distribution.
// Both caller-supplied uniforms are finite in [0,1). No allocation or RNG occurs.
void verify_full_vocab(const float* target_logits, const double* q, int draft_token, int vocab,
                       double temperature, double uniform_accept, double uniform_residual,
                       void* workspace, size_t workspace_bytes, FullVocabResult* device_result,
                       uint32_t* accepted, void* stream);

} // namespace strata::gemma4
