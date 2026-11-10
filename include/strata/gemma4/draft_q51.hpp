// External-draft Q5_1 x Q8_1: reuse each weight decode across up to eight rows,
// preserving the original iq_mmvq row's dot and reduction order.
// TINYSTRATA_DRAFT_Q51_READONCE=0 selects the existing generic row kernel.
#pragma once

namespace strata::gemma4 {

void draft_q51_mmvq(const void* weights, const void* q8_1, float* output,
                     int n_in, int n_out, int columns, void* stream);

}  // namespace strata::gemma4
