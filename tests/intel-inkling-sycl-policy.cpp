#include "../third_party/strata-inkling/src/inkling2/sycl/ink_sy_memory_policy.h"
#include "../third_party/strata-inkling/src/inkling2/sycl/ink_sy_fit.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

static int checks = 0;
static void check(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("Inkling SYCL UMA policy regression");
}

int main() {
    using namespace strata::intel;
    using namespace ink_sycl;
    MemoryBudget measured{24 * GiB, 28 * GiB, 20 * GiB, true, true};
    check(allocation_budget(measured) == 20 * GiB - uma_wddm_headroom);
    UmaLedger ledger{allocation_budget(measured)};
    // Device and pinned host experts use one physical pool, counted once.
    ledger.device_bytes = 14 * GiB;
    ledger.host_usm_bytes = GiB;
    check(ledger.owned_bytes() == 15 * GiB);
    const uint64_t remaining = 5 * GiB - uma_wddm_headroom;
    check(ledger.remaining(20 * GiB) == remaining);
    check(ledger.fits(remaining, 20 * GiB));
    check(!ledger.fits(remaining + 1, 20 * GiB));
    check(!ledger.fits(0, 20 * GiB));
    // Refreshed availability is already net of current backing: do not
    // subtract the owned bytes or either headroom from that value again.
    check(ledger.remaining(GiB) == GiB);
    check(ledger.fits(GiB, GiB));
    check(!ledger.fits(GiB + 1, GiB));
    check(!ledger.fits(1, 0));
    ledger.device_bytes = ledger.initial_budget;
    ledger.host_usm_bytes = 0;
    check(ledger.remaining(20 * GiB) == 0);
    check(!ledger.fits(1, 20 * GiB));
    // Unknown WDDM is explicitly not zero measured capacity and never adds
    // reported global memory to the physical RAM/commit constraint.
    measured.wddm_known = false;
    check(allocation_budget(measured) == 20 * GiB);
    measured.available_commit = 10 * GiB;
    check(allocation_budget(measured) == 6 * GiB);
    measured.wddm_known = true;
    measured.wddm_available = uma_wddm_headroom;
    check(allocation_budget(measured) == 0);
    ++measured.wddm_available;
    check(allocation_budget(measured) == 1);

    uint64_t experts = 0;
    check(add_expert_bytes(experts, 1) && experts == 4096);
    check(add_expert_bytes(experts, 4096) && experts == 8192);
    check(add_expert_bytes(experts, 4097) && experts == 16384);
    check(all_experts_fit(experts, 8192, 24576));
    check(!all_experts_fit(experts, 8193, 24576));
    check(!all_experts_fit(experts, 0, experts - 1));
    const uint64_t before = experts;
    check(!add_expert_bytes(experts, std::numeric_limits<uint64_t>::max()));
    check(experts == before);
    uint64_t almost_full = std::numeric_limits<uint64_t>::max() - 2048;
    check(!add_expert_bytes(almost_full, 1));
    check(almost_full == std::numeric_limits<uint64_t>::max() - 2048);
    InkSyFitGeom fit;
    const auto language = ink_sy_runner_reserve(fit,1,8192,512);
    fit.vision = true;
    check(ink_sy_runner_reserve(fit,1,8192,512) == language + (700ull << 20));
    fit.resident_vision_bytes = 600ull << 20;
    check(ink_sy_runner_reserve(fit,1,8192,512) == language + (100ull << 20));
    fit.resident_vision_bytes = 800ull << 20;
    check(ink_sy_runner_reserve(fit,1,8192,512) == language);
    std::printf("Inkling SYCL UMA policy: %d checks passed\n", checks);
}
