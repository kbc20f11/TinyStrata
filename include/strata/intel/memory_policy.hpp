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
    bool integrated = true;
    uint64_t wddm_nonlocal_available = 0;
    bool wddm_nonlocal_known = false;
};
enum class MemoryPool { Host, Shared, Device };
inline uint64_t system_allocation_budget(const MemoryBudget& m, uint64_t reserve = 4 * GiB) {
    const uint64_t available = std::min(m.available_physical, m.available_commit);
    return available > reserve ? available - reserve : 0;
}
// UMA is one pool: never add "shared GPU memory" to physical memory. Leave
// headroom for Windows and the desktop, then respect the process's WDDM budget.
inline uint64_t allocation_budget(const MemoryBudget& m, uint64_t reserve = 4 * GiB) {
    // A discrete GPU's local budget is VRAM, not another view of system RAM.
    // Keep modest local headroom; host staging/control is checked separately.
    if (!m.integrated) {
        constexpr uint64_t local_reserve = GiB / 4;
        return m.wddm_known && m.wddm_available > local_reserve ? m.wddm_available - local_reserve : 0;
    }
    const uint64_t available = std::min(m.available_physical, m.available_commit);
    uint64_t budget = available > reserve ? available - reserve : 0;
    if (m.wddm_known) budget = std::min(budget, m.wddm_available);
    return budget;
}
inline uint64_t allocation_budget(const MemoryBudget& m, MemoryPool pool, uint64_t reserve = 4 * GiB) {
    if (m.integrated) return allocation_budget(m, reserve);
    if (pool == MemoryPool::Device) return allocation_budget(m, reserve);
    // Migrating a shared weight arena between RAM/VRAM has not been validated.
    // Host-USM weights are explicitly GPU-readable and remain in one RAM arena.
    if (pool == MemoryPool::Shared || !m.wddm_nonlocal_known) return 0;
    const uint64_t host = system_allocation_budget(m, reserve);
    return std::min(host, m.wddm_nonlocal_available);
}
inline bool allocation_fits(const MemoryBudget& m, uint64_t bytes, uint64_t max_allocation) {
    return bytes != 0 && bytes <= max_allocation && bytes <= allocation_budget(m);
}
inline bool allocation_fits(const MemoryBudget& m, uint64_t bytes, uint64_t max_allocation, MemoryPool pool) {
    // max_mem_alloc_size describes the device limit. Host USM is allocated in
    // system memory; its allocator can still refuse an unsupported large arena.
    const bool limit = !m.integrated && pool == MemoryPool::Host ? true : bytes <= max_allocation;
    return bytes != 0 && limit && bytes <= allocation_budget(m, pool);
}
// device_bytes already includes shared/control USM's possible device backing.
// UMA counts that single RAM pool once, but ordinary CPU snapshots/work do
// not consume WDDM GPU backing. Discrete additionally admits its possible
// host backing and ordinary host-only snapshots/work without charging snapshots
// to VRAM. RAM and VRAM capacities are never added together.
inline bool execution_allocation_fits(const MemoryBudget& m, uint64_t device_bytes,
                                      uint64_t host_visible_bytes, uint64_t host_only_bytes) {
    const uint64_t device = allocation_budget(m);
    if (device_bytes > device) return false;
    if (m.integrated) {
        // The check above already admits device backing against WDDM and RAM.
        // Admit the additional ordinary host allocations against RAM/commit,
        // preserving the same one4GiB OS floor without charging them to WDDM.
        return host_only_bytes <= system_allocation_budget(m) - device_bytes;
    }
    const uint64_t host = system_allocation_budget(m);
    if (host_visible_bytes > host || host_only_bytes > host - host_visible_bytes) return false;
    if (host_visible_bytes && (!m.wddm_nonlocal_known || host_visible_bytes > m.wddm_nonlocal_available)) return false;
    return true;
}
} // namespace strata::intel
