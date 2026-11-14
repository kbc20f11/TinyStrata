#pragma once
#include <algorithm>
#include <cstdint>

namespace strata::intel {
constexpr uint64_t GiB = uint64_t{1} << 30;
struct MemoryBudget {
    uint64_t available_physical = 0;
    uint64_t available_commit = 0;
    uint64_t wddm_available = 0;
    bool wddm_known = false;
};
// UMA is one pool: never add "shared GPU memory" to physical memory. Leave
// headroom for Windows and the desktop, then respect the process's WDDM budget.
inline uint64_t allocation_budget(const MemoryBudget& m, uint64_t reserve = 4 * GiB) {
    const uint64_t available = std::min(m.available_physical, m.available_commit);
    uint64_t budget = available > reserve ? available - reserve : 0;
    if (m.wddm_known) budget = std::min(budget, m.wddm_available);
    return budget;
}
inline bool allocation_fits(const MemoryBudget& m, uint64_t bytes, uint64_t max_allocation) {
    return bytes != 0 && bytes <= max_allocation && bytes <= allocation_budget(m);
}
} // namespace strata::intel
