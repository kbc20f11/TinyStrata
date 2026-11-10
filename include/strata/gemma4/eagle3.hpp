#pragma once

#include <cstdint>

namespace strata::gemma4 {

// q is already normalized over the draft vocabulary. The validated mapping is
// one-to-one and contains absolute target IDs. Every other target probability
// is exactly zero. Input, output and mapping must not alias.
void eagle3_scatter_probabilities(const double* q, const int32_t* draft_to_target,
                                  double* out, int draft_vocab, int target_vocab, void* stream);

}  // namespace strata::gemma4
