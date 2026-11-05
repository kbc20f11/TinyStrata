// Standalone CPU policy test: no model files, GPU runtime or test framework.
#include "strata/qwen35/topic_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace TP = strata::qwen35::topic_policy;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

bool close(double a, double b) { return std::fabs(a - b) <= 1e-12 * (1.0 + std::fabs(b)); }

int main() {
    unsigned checks = 0;
    // An independent hand-calculated example: resident counts do not spend
    // the two new-entry places; the lower-ranked nonresident decays only.
    {
        std::vector<double> usage{10, 20, 30, 40, 50};
        const std::vector<uint64_t> hist{0, 20, 40, 10, 30};
        const std::vector<int> slots{0, -1, -1, 1, -1};
        TP::Parameters p; p.max_new_nonresident = 2;
        TP::SeedStats stats;
        REQUIRE(TP::seed_prompt(usage, hist, 64, slots, p, &stats));
        const std::vector<double> expected{3, 6, 49, 22, 45};
        for (size_t i = 0; i < usage.size(); ++i) REQUIRE(close(usage[i], expected[i]));
        REQUIRE(stats.resident_seeded == 1 && stats.nonresident_seeded == 2 && stats.scale == 1.0);
        ++checks;
    }
    // Identical routing proportions are independent of prompt length.
    for (uint64_t length : {64ull, 4096ull, 16384ull, 262144ull}) {
        std::vector<double> usage{4, 12, 20, 8};
        const std::vector<uint64_t> hist{length / 4, length, length / 2, length / 8};
        REQUIRE(TP::seed_prompt(usage, hist, length, {0, -1, -1, -1}));
        const std::vector<double> expected{17.2, 67.6, 38.0, 10.4};
        for (size_t i = 0; i < usage.size(); ++i) REQUIRE(close(usage[i], expected[i]));
        ++checks;
    }
    // Ranking ties resolve by Expert ID; the resident Expert remains seeded.
    {
        std::vector<double> usage(4, 0.0);
        TP::Parameters p; p.max_new_nonresident = 1;
        REQUIRE(TP::seed_prompt(usage, {10, 10, 10, 10}, 64, {0, -1, -1, -1}, p));
        REQUIRE((usage == std::vector<double>{10, 10, 0, 0}));
        ++checks;
    }
    // A new topic can replace the old signal without inheriting the raw size
    // of the new prompt. This is the same expectation at either input length.
    for (uint64_t length : {4096ull, 16384ull}) {
        std::vector<double> usage{100, 0};
        REQUIRE(TP::seed_prompt(usage, {0, length}, length, {0, -1}));
        REQUIRE(usage[0] == 30 && usage[1] == 64 && usage[1] > usage[0]);
        ++checks;
    }
    // The default bound is independent of VRAM size. Resident Experts do not
    // consume that bound; ordinary admission still chooses the actual slots.
    {
        std::vector<double> usage(256, 0.0);
        std::vector<uint64_t> hist(256, 1);
        std::vector<int> slots(256, -1);
        for (int i = 0; i < 4; ++i) slots[(size_t) i] = i;
        TP::SeedStats stats;
        REQUIRE(TP::seed_prompt(usage, hist, 64, slots, {}, &stats));
        REQUIRE(stats.resident_seeded == 4 && stats.nonresident_seeded == 32);
        for (size_t i = 0; i < usage.size(); ++i) REQUIRE(usage[i] == (i < 36 ? 1.0 : 0.0));
        ++checks;
    }
    {
        std::vector<double> usage{4, 8};
        TP::Parameters p; p.max_new_nonresident = 0;
        REQUIRE(TP::seed_prompt(usage, {64, 64}, 64, {0, -1}, p));
        REQUIRE(close(usage[0], 65.2) && close(usage[1], 2.4));
        ++checks;
    }
    // Cache hits and empty input do not decay otherwise useful prior usage.
    {
        std::vector<double> usage{7, 9};
        const auto before = usage;
        TP::SeedStats stats{1, 2, 3};
        REQUIRE(TP::seed_prompt(usage, {50, 10}, 0, {0, -1}, {}, &stats));
        REQUIRE(usage == before && stats.resident_seeded == 0 && stats.nonresident_seeded == 0 && stats.scale == 0);
        std::vector<double> empty;
        REQUIRE(TP::seed_prompt(empty, {}, 64, {}));
        ++checks;
    }
    // Accumulation has checked boundaries, and collecting separate chunks
    // produces the same normalized result as collecting a whole prompt.
    {
        std::vector<uint64_t> hist(3, 0);
        REQUIRE(TP::record(hist, 0, 2048) && TP::record(hist, 0, 2048));
        REQUIRE(TP::record(hist, 1, 1024) && TP::record(hist, 2, 512));
        const auto before = hist;
        REQUIRE(!TP::record(hist, 3) && hist == before);
        hist[2] = std::numeric_limits<uint64_t>::max();
        REQUIRE(!TP::record(hist, 2) && hist[2] == std::numeric_limits<uint64_t>::max());
        hist = before;
        std::vector<double> usage(3, 0.0);
        REQUIRE(TP::seed_prompt(usage, hist, 4096, {-1, -1, -1}));
        REQUIRE((usage == std::vector<double>{64, 16, 8}));
        ++checks;
    }
    // A failed validation never partially ages or changes the caller's usage.
    const std::vector<double> original{10, 20};
    const auto invalid = [&](std::vector<double> usage, std::vector<uint64_t> hist,
                             std::vector<int> slots, TP::Parameters p = {}) {
        const auto before = usage;
        REQUIRE(!TP::seed_prompt(usage, hist, 64, slots, p));
        for (size_t i = 0; i < usage.size(); ++i)
            REQUIRE((std::isnan(before[i]) && std::isnan(usage[i])) || before[i] == usage[i]);
        ++checks;
    };
    invalid(original, {1}, {-1, -1});
    invalid(original, {1, 2}, {-1});
    invalid(original, {1, 2}, {-2, -1});
    invalid(original, {1, 2}, {2, -1});
    invalid(original, {1, 2}, {0, 0});
    invalid({-1, 20}, {1, 2}, {-1, -1});
    invalid({std::numeric_limits<double>::infinity(), 20}, {1, 2}, {-1, -1});
    invalid({std::numeric_limits<double>::quiet_NaN(), 20}, {1, 2}, {-1, -1});
    TP::Parameters p;
    p.seed_tokens = 0; invalid(original, {1, 2}, {-1, -1}, p);
    p.seed_tokens = std::numeric_limits<double>::infinity(); invalid(original, {1, 2}, {-1, -1}, p);
    p = {}; p.history_keep = -0.1; invalid(original, {1, 2}, {-1, -1}, p);
    p.history_keep = 1.1; invalid(original, {1, 2}, {-1, -1}, p);
    p.history_keep = std::numeric_limits<double>::quiet_NaN(); invalid(original, {1, 2}, {-1, -1}, p);
    p = {}; p.seed_tokens = std::numeric_limits<double>::max();
    invalid(original, {std::numeric_limits<uint64_t>::max(), 2}, {-1, -1}, p);
    std::printf("PASS topic policy: %u normalization, bounds, topic shift and residency checks\n", checks);
    return 0;
}
