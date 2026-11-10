// Lossless grouping of repeated GPT-OSS Expert matrices in a small verify batch.
#pragma once
#include "strata/gemma4/matvec.hpp"
#include <cstddef>

namespace strata::kernels::grouped_mxfp4 {

// Fixed metadata for up to 32 raw Expert rows. Owned/charged by the caller;
// prepare and both products are ordered on the caller's existing GPU stream.
size_t workspace_bytes();
bool prepare_ids(const int32_t* ids, int count, int experts, void* workspace,
                  size_t bytes, void* stream);
bool prepare_slots(const MmvqSlots& slots, int count, void* workspace,
                    size_t bytes, void* stream);

// Grouping changes reads only. Outputs scatter to each original row's address,
// leaving activation, Expert bias/weight and ordered reduction unchanged.
// current_down_layout selects the existing mxfp4_down warp-32 dot for 2880->2880;
// false retains the existing native IDs/slots warp-128 dot for either product.
// Unsupported geometry returns false before any launch. Invalid device IDs
// produce NaNs without reading an out-of-range slot, so normal finite-logit
// validation rejects the request rather than using uninitialized output.
bool ids(int type, const MmvqIds& bundle, int count, int n_in, int n_out,
          const void* workspace, bool current_down_layout, void* stream);
bool slots(int type, const MmvqSlots& bundle, int count, int n_in, int n_out,
            const void* workspace, bool current_down_layout, void* stream);

} // namespace strata::kernels::grouped_mxfp4
