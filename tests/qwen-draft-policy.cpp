#include "strata/qwen35/draft_policy.hpp"
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <string>

void check_impl(bool value, const char* message, int line) {
    if (!value) throw std::runtime_error(std::string(message) + " (line " + std::to_string(line) + ")");
}
#define check(value, message) check_impl(value, message, __LINE__)
int run() {
    using namespace strata::qwen35;
    int checks = 0;
    for (int maximum = 0; maximum <= 7; ++maximum) {
        DraftDepthPolicy policy(maximum);
        bool probed_zero = false, probed_max = false;
        int alternatives = 0;
        for (int i = 0; i < 384; ++i) {
            const int d = policy.choose();
            check(d == 0 || d == 1 || d == maximum, "unconfigured depth");
            check(d <= maximum, "depth exceeds maximum");
            if (i < 95) check(d == maximum, "high acceptance paid an early cold probe");
            probed_zero |= d == 0; probed_max |= d == maximum;
            alternatives += d != maximum;
            check(policy.observe(d, d, d == maximum ? 1. : 4., false), "valid sample rejected");
            ++checks;
        }
        check(probed_zero && probed_max, "periodic recovery arm was never probed");
        check(alternatives <= 4 && policy.leader() == maximum, "high acceptance baseline displaced");
        check(!policy.observe(8, 0, 1, false), "unknown depth accepted");
        check(!policy.observe(0, 0, 0, false), "zero elapsed accepted");
        check(!policy.observe(0, 0, std::numeric_limits<double>::infinity(), false), "infinite elapsed accepted");
        const int before = policy.rate(0).samples;
        check(!policy.observe(0, 0, 1, true), "graph recording polluted measurements");
        check(policy.rate(0).samples == before, "graph recording changed sample count");
    }
    DraftDepthPolicy margin(3);
    for (int i = 0; i < 3; ++i) margin.observe(3, 3, 4, false); // 1 row/ms
    for (int i = 0; i < 3; ++i) margin.observe(0, 0, 1. / 1.04, false);
    check(margin.leader() == 3, "a sub-five-percent gain changed leader");
    DraftDepthPolicy confidence(3);
    for (int i = 0; i < 3; ++i) confidence.observe(3, 3, 4, false);
    for (int i = 0; i < 2; ++i) confidence.observe(0, 0, .5, false);
    check(confidence.leader() == 3, "leader changed without three candidate samples");
    confidence.observe(0, 0, .5, false);
    check(confidence.leader() == 0, "measured large gain did not change leader");
    // A serving policy must learn a poor regime, then recover when later
    // answer tokens become easier to predict, without discarding its evidence.
    DraftDepthPolicy recovery(3);
    for (int i = 0; i < 128; ++i) {
        const int d = recovery.choose();
        recovery.observe(d, 0, d == 3 ? 8. : d == 1 ? 1.3 : 1., false);
    }
    check(recovery.leader() == 0, "low acceptance did not learn the faster zero-proposal path");
    bool refreshed = false;
    for (int i = 0; i < 768; ++i) {
        const int d = recovery.choose();
        const int before = recovery.rate(3).samples;
        recovery.observe(d, d, d == 3 ? 1. : 4., false);
        refreshed |= before >= 3 && recovery.rate(3).samples == 1;
        if (refreshed && recovery.rate(3).samples < 3)
            check(recovery.leader() != 3, "stale-arm refresh promoted fewer than three fresh samples");
    }
    check(refreshed, "a strongly changed stale arm retained its obsolete cost history");
    check(recovery.leader() == 3, "periodic probes did not recover a faster maximum depth");
    LookupDraftGate gate;
    int used = 0, skipped = 0;
    for (int i = 0; i < 128; ++i) {
        const bool lookup = gate.choose();
        used += lookup; skipped += !lookup;
        gate.observe(lookup, lookup ? 1 : 4, lookup ? 7 : 3, lookup ? 0 : 3, 2, false);
    }
    check(used > 0 && skipped > used, "lookup gate did not retain baseline and recovery probes");
    std::printf("PASS draft scheduling: %d observations, conservative start, gain margin and regime recovery\n", checks);
    return 0;
}
int main() {
    try { return run(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL draft scheduling: %s\n", error.what());
        return 1;
    }
}
