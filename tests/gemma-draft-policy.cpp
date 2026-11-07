// CPU-only bounds, calibration, recording exclusion and periodic re-probes.
#include "strata/gemma4/draft_policy.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
using strata::gemma4::DraftPolicy;
using strata::gemma4::LookupDraftGate;
using strata::gemma4::speculation_width;

int main() {
    REQUIRE(speculation_width(8, 100, 1000, 1000, false, 128) == 8);
    REQUIRE(speculation_width(8, 100, 1000, 103, true, 128) == 3);
    REQUIRE(speculation_width(8, 103, 1000, 103, true, 128) == 1);
    REQUIRE(speculation_width(8, 998, 1000, 1000, true, 128) == 2);
    REQUIRE(speculation_width(8, 100, 1000, 1000, false, 1) == 2);
    REQUIRE(speculation_width(4, 100, 1000, 1000, false, 128) == 4);
    // Legacy MTP grows at an exact prompt/KV boundary with its full window;
    // only lookup gets the physical-row cap before fallback to that MTP path.
    REQUIRE(speculation_width(4, 4096, 32768, 4096, false, 128) == 4);
    REQUIRE(speculation_width(4, 4096, 32768, 4096, true, 128) == 1);
    for (int maximum : {0, 1, 3, 7}) {
        DraftPolicy policy(maximum);
        int zero = 0, other = 0;
        for (int i = 0; i < 512; ++i) {
            const int depth = policy.choose(maximum);
            REQUIRE(depth == 0 || depth == 1 || depth == maximum);
            REQUIRE(depth >= 0 && depth <= maximum);
            depth ? ++other : ++zero;
            policy.observe(depth, 1, depth ? 4. * (depth + 1) : 1., false, depth, 0);
        }
        REQUIRE(zero > 360);
        if (maximum) REQUIRE(other >= 4); // Alternatives are revisited.
    }
    DraftPolicy policy(3);
    for (int i = 0; i < 20; ++i) {
        REQUIRE(policy.choose(3) == 3);
        policy.observe(3, 4, 100., true);
        policy.observe(3, 4, std::numeric_limits<double>::quiet_NaN(), false);
        policy.observe(3, 4, -1., false);
    }
    int maximum = 0;
    for (int i = 0; i < 512; ++i) {
        const int depth = policy.choose(3);
        maximum += depth == 3;
        policy.observe(depth, 1, depth == 3 ? 1. : 10., false, depth, 0);
    }
    REQUIRE(maximum > 420);
    int changed = 0;
    for (int i = 0; i < 4096; ++i) {
        const int depth = policy.choose(3);
        if (i >= 3000) changed += depth == 0;
        policy.observe(depth, 1, depth ? 100. : 1., false, depth, 0);
    }
    REQUIRE(changed > 900);
    DraftPolicy copying(3);
    copying.begin_request(4096);
    int alternates = 0;
    for (int i = 0; i < 256; ++i) {
        const int depth = copying.choose(3);
        alternates += depth != 3;
        copying.observe(depth, depth + 1, 1., false, depth, depth);
    }
    REQUIRE(alternates <= 3); // Strong-copy path pays no six-round calibration.
    REQUIRE(copying.acceptance() > 0.99);
    const int retained = copying.samples(3);
    copying.begin_request(4500);
    REQUIRE(copying.samples(3) == retained);
    copying.begin_request(16384);
    REQUIRE(copying.samples(3) == 0);
    REQUIRE(copying.choose(3) == 3);
    // A narrowed last/context window never proposes outside the available rows.
    for (int room = 0; room <= 7; ++room)
        REQUIRE(policy.choose(room) <= room);
    LookupDraftGate gate;
    gate.fallback(2, 1., false);
    for (int i = 0; i < 2; ++i) {
        REQUIRE(gate.allows(3));
        gate.observe(3, 3, 0, 1, 2., false);
    }
    for (int i = 0; i < 31; ++i) REQUIRE(!gate.allows(3));
    REQUIRE(gate.allows(3)); // One eligible retry after 32 opportunities.
    for (int i = 0; i < 128; ++i) {
        REQUIRE(gate.allows(8)); // Bad short matches never disable copying.
        gate.observe(8, 7, 7, 8, 2., false);
    }
    REQUIRE(gate.milliseconds_per_token(true) < gate.milliseconds_per_token(false));
    LookupDraftGate costly;
    costly.fallback(2, 1., false);
    for (int i = 0; i < 2; ++i) {
        REQUIRE(costly.allows(4));
        costly.observe(4, 3, 3, 4, 10., false);
    }
    REQUIRE(!costly.allows(4)); // Acceptance alone does not imply speed.
    LookupDraftGate recorded;
    for (int i = 0; i < 10; ++i) {
        REQUIRE(recorded.allows(3));
        recorded.observe(3, 3, 0, 1, 100., true);
    }
    std::puts("PASS Gemma draft policy and context/KV bounds");
}
