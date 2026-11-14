#pragma once
#include "strata/intel/memory_policy.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>

namespace strata::intel {
struct DeviceInfo {
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    bool integrated = false;
    uint64_t reported_global_memory = 0;
    uint64_t max_allocation = 0;
    std::string name;
    std::string driver;
    std::string architecture = "unknown";
    // Actual Level Zero/DXGI identity. Empty when the optional LUID extension
    // is unavailable; a PCI device ID alone cannot identify an adapter.
    std::string adapter_luid;
    uint32_t adapter_node_mask = 0;
};
// Select an explicitly enumerated Intel Level Zero GPU, never a CPU fallback.
sycl::device select_device(unsigned index);
DeviceInfo inspect_device(const sycl::device& device);
MemoryBudget memory_budget(const DeviceInfo& device);
void print_memory(const char* stage, const DeviceInfo& device);
} // namespace strata::intel
