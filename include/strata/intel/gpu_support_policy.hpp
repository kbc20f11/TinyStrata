#pragma once
#include <stdexcept>
#include <string_view>

namespace strata::intel::gpu_support_policy {
// Sparse proposal support only. An explicit user's MTP0/draft0 remains off.
// Other devices can opt into the separately tested candidate without receiving
// a default performance claim from the B390 measurements.
inline bool enabled(bool validated_default, bool sparse_active, const char* requested) {
    if (requested) {
        const std::string_view value(requested);
        if (value == "0") return false;
        if (value == "1") return sparse_active;
        throw std::invalid_argument("TINYSTRATA_INTEL_GPU_SUPPORT must be 0 or 1");
    }
    return validated_default && sparse_active;
}
}
