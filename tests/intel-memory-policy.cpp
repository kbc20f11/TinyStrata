#include "strata/intel/memory_policy.hpp"
#include "strata/intel/device_policy.hpp"
#include "strata/intel/gpu_support_policy.hpp"
#include "strata/intel/host_profile.hpp"
#include <cstdio>
#include <stdexcept>
using namespace strata::intel;
void check(bool condition) { if (!condition) throw std::runtime_error("Intel memory policy regression"); }
int main() {
    MemoryBudget m{20 * GiB, 30 * GiB, 12 * GiB, true};
    check(allocation_budget(m) == 12 * GiB);
    m.wddm_known = false;
    check(allocation_budget(m) == 16 * GiB);
    m.available_commit = 8 * GiB;
    check(allocation_budget(m) == 4 * GiB);
    check(!allocation_fits(m, 4 * GiB + 1, 8 * GiB));
    check(!allocation_fits(m, GiB + 1, GiB));
    check(allocation_fits(m, GiB, GiB));
    check(!allocation_fits(m, 0, GiB));
    m.available_physical = GiB;
    check(allocation_budget(m) == 0);
    check(gpu_support_policy::enabled(true, true, nullptr));
    check(!gpu_support_policy::enabled(false, true, nullptr));
    check(!gpu_support_policy::enabled(true, true, "0"));
    check(gpu_support_policy::enabled(false, true, "1"));
    check(!gpu_support_policy::enabled(true, false, "1"));
    bool rejected_override = false;
    try { (void)gpu_support_policy::enabled(true, true, "true"); }
    catch (const std::invalid_argument&) { rejected_override = true; }
    check(rejected_override);
    namespace hp = host_profile;
    check(!hp::enabled && hp::now() == hp::Clock::time_point{});
    { hp::Scope scope(hp::TargetTotal); }
    check(hp::counters[hp::TargetTotal].calls == 0);
    hp::enabled = true;
    { hp::Scope scope(hp::TargetTotal); }
    check(hp::counters[hp::TargetTotal].calls == 1 && hp::counters[hp::TargetTotal].milliseconds >= 0);
    hp::enabled = false; hp::reset();
    check(hp::counters[hp::TargetTotal].calls == 0 && hp::now() == hp::Clock::time_point{});
    MemoryBudget discrete{24 * GiB, 30 * GiB, 8 * GiB, true, false, 16 * GiB, true};
    check(allocation_budget(discrete, MemoryPool::Host) == 16 * GiB);
    check(allocation_budget(discrete, MemoryPool::Device) == 8 * GiB - GiB / 4);
    check(allocation_budget(discrete, MemoryPool::Shared) == 0);
    check(allocation_fits(discrete, 14 * GiB, 2 * GiB, MemoryPool::Host));
    check(!allocation_fits(discrete, 14 * GiB, 16 * GiB, MemoryPool::Device));
    check(!allocation_fits(discrete, 3 * GiB, 2 * GiB, MemoryPool::Device));
    check(!allocation_fits(discrete, GiB, 4 * GiB, MemoryPool::Shared));
    discrete.available_commit = 10 * GiB;
    check(allocation_budget(discrete, MemoryPool::Host) == 6 * GiB);
    check(allocation_budget(discrete, MemoryPool::Device) == 8 * GiB - GiB / 4);
    discrete.wddm_nonlocal_known = false;
    check(allocation_budget(discrete, MemoryPool::Host) == 0);
    check(allocation_budget(discrete, MemoryPool::Device) > 0);
    discrete.wddm_known = false;
    check(allocation_budget(discrete, MemoryPool::Device) == 0);
    constexpr uint64_t snapshot = 63ull * 1024 * 1024;
    MemoryBudget uma_snapshot{4*GiB+GiB+snapshot,4*GiB+GiB+snapshot,GiB,true,true};
    check(execution_allocation_fits(uma_snapshot,GiB,GiB,snapshot)); // snapshot has no WDDM backing
    check(!execution_allocation_fits(uma_snapshot,GiB+1,0,snapshot)); // device is over WDDM
    check(!execution_allocation_fits(uma_snapshot,GiB,0,snapshot+1)); // one byte below RAM floor
    --uma_snapshot.available_commit;
    check(!execution_allocation_fits(uma_snapshot,GiB,0,snapshot)); // commit remains separate
    MemoryBudget separate{4 * GiB, 4 * GiB, GiB, true, false, GiB, true};
    check(!execution_allocation_fits(separate, GiB / 2, 1, snapshot)); // zero host room
    separate.available_physical += snapshot; separate.available_commit += snapshot;
    check(execution_allocation_fits(separate, GiB / 2, 0, snapshot));
    check(!execution_allocation_fits(separate, GiB / 2, 1, snapshot)); // one byte over host floor
    separate.available_physical += 1024 * 1024; separate.available_commit += 1024 * 1024;
    check(execution_allocation_fits(separate, GiB / 2, 1024 * 1024, snapshot));
    separate.wddm_nonlocal_available = 0;
    check(!execution_allocation_fits(separate, GiB / 2, 1, snapshot));
    separate.wddm_available = GiB / 4 + 1;
    check(execution_allocation_fits(separate, 1, 0, snapshot)); // host snapshot is not VRAM
    separate.available_commit = 4 * GiB;
    check(!execution_allocation_fits(separate, 1, 0, snapshot)); // commit limit also applies
    // Hardware facts, not a B390 name or PCI-ID allowlist, admit baseline.
    DeviceCapabilities caps{true, true, true, true, true, true, true, false, false, 256, 41984};
    check(!baseline_limitation(caps)); check(!supports_xmx(caps));
    caps.esimd = caps.matrix = true; check(supports_xmx(caps));
    caps.matrix = false; check(!supports_xmx(caps));
    caps.host_usm = false; check(baseline_limitation(caps)); caps.host_usm = true;
    caps.subgroup16 = false; check(baseline_limitation(caps)); caps.subgroup16 = true;
    caps.subgroup32 = false; check(baseline_limitation(caps)); caps.subgroup32 = true;
    caps.atomic64 = false; check(baseline_limitation(caps)); caps.atomic64 = true;
    caps.fp16 = false; check(baseline_limitation(caps)); caps.fp16 = true;
    caps.max_work_group = 255; check(baseline_limitation(caps)); caps.max_work_group = 256;
    caps.local_memory = 41983; check(baseline_limitation(caps));
    m.available_physical = 20 * GiB;
    m.wddm_known = true;
    m.wddm_available = 0;
    check(allocation_budget(m) == 0);
    std::puts("Intel memory policy: passed");
}
