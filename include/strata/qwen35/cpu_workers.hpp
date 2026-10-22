// Hybrid worker-count rule adapted from Niko1221/Strata (MIT),
// include/strata/kernels/cpu/pool_plan.hpp at 011c9e600a93bbc9fd451c563501b5831c32bbca.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
#pragma once

#include <algorithm>

namespace strata::qwen35 {

// Only Qwen's automatic hybrid-CPU count changes. The existing pool orders
// physical P-cores first and reserves one for the host. Using half the E-cores
// avoids the synchronization cost of putting a spinning worker on every core.
// Explicit counts and ordinary CPUs retain the existing pool behavior.
constexpr int resolve_cpu_workers(int requested, bool hybrid, int p_cores, int e_cores, int available_workers) {
    if (requested > 0 || !hybrid || p_cores <= 0 || e_cores <= p_cores || available_workers <= 0)
        return requested;
    return std::max(1, std::min(available_workers, p_cores - 1 + e_cores / 2));
}

} // namespace strata::qwen35
