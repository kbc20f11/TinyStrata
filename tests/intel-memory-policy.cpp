#include "strata/intel/memory_policy.hpp"
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
    m.available_physical = 20 * GiB;
    m.wddm_known = true;
    m.wddm_available = 0;
    check(allocation_budget(m) == 0);
    std::puts("Intel memory policy: passed");
}
