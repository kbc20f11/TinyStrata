#pragma once

#include "strata/intel/memory_policy.hpp"
#include <cstdint>
#include <limits>

namespace ink_sycl {

// All GPU-readable allocations on UMA share physical RAM. The initial ceiling
// catches deferred backing; the refreshed ceiling catches other processes.
// Neither ceiling adds a second "shared VRAM" pool or subtracts reserves twice.
struct UmaLedger {
    uint64_t initial_budget = 0;
    uint64_t device_bytes = 0;
    uint64_t host_usm_bytes = 0;

    uint64_t owned_bytes() const { return device_bytes + host_usm_bytes; }
    uint64_t remaining(uint64_t refreshed_budget) const {
        const uint64_t owned = owned_bytes();
        const uint64_t initial_remaining = owned < initial_budget ? initial_budget - owned : 0;
        return std::min(initial_remaining, refreshed_budget);
    }
    bool fits(uint64_t bytes, uint64_t refreshed_budget) const {
        return bytes && bytes <= remaining(refreshed_budget);
    }
};

inline bool add_bytes(uint64_t& total, uint64_t bytes) {
    if (bytes > std::numeric_limits<uint64_t>::max() - total) return false;
    total += bytes;
    return true;
}

// The model loader rounds each compressed expert to a 4 KiB slab interval.
inline bool add_expert_bytes(uint64_t& total, uint64_t bytes) {
    if (bytes > std::numeric_limits<uint64_t>::max() - 4095) return false;
    return add_bytes(total, (bytes + 4095) & ~uint64_t{4095});
}

inline bool all_experts_fit(uint64_t expert_bytes, uint64_t runner_reserve,
                            uint64_t remaining) {
    return expert_bytes <= remaining && runner_reserve <= remaining - expert_bytes;
}

} // namespace ink_sycl
