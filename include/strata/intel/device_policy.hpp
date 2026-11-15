#pragma once
#include <cstdint>

namespace strata::intel {
// Facts queried from the selected SYCL GPU; no device-name/PCI allowlist.
struct DeviceCapabilities {
    bool host_usm = false, shared_usm = false, device_usm = false;
    bool fp16 = false, atomic64 = false, subgroup16 = false, subgroup32 = false;
    bool esimd = false, matrix = false;
    uint64_t max_work_group = 0, local_memory = 0;
};
inline const char* baseline_limitation(const DeviceCapabilities& c) {
    if (!c.host_usm || !c.shared_usm || !c.device_usm) return "GPU needs host/shared/device USM allocations";
    if (!c.fp16) return "GPU needs FP16 weight/KV conversion support";
    if (!c.atomic64) return "GPU needs 64-bit atomics for weight visibility validation";
    if (!c.subgroup16 || !c.subgroup32) return "GPU needs subgroup sizes 16 and 32";
    if (c.max_work_group < 256) return "GPU needs a work-group size of at least 256";
    if (c.local_memory < 41984) return "GPU needs 41 KiB of local memory for tiled attention";
    return nullptr;
}
inline bool supports_xmx(const DeviceCapabilities& c) { return c.esimd && c.matrix; }
}
