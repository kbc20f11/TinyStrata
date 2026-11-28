#include "ink_sy_far_attention.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

static unsigned checks = 0;
static void check(bool condition) {
    ++checks;
    if (!condition) throw std::runtime_error("Inkling far-attention policy regression");
}

int main() {
    using namespace ink_sycl;
    check(!parse_far_attention_mode(nullptr));
    check(!parse_far_attention_mode("0"));
    check(parse_far_attention_mode("1"));
    for (const char* invalid : {"", "true", "2", "01", " 1"}) {
        bool rejected = false;
        try { (void)parse_far_attention_mode(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
    check(!far_attention_eligible(false, 512, false, 1024));
    check(!far_attention_eligible(true, 512, true, 1024));
    check(!far_attention_eligible(true, 1, false, 1024));
    check(!far_attention_eligible(true, 15, false, 1024));
    check(far_attention_eligible(true, 16, false, 1024));
    check(far_attention_eligible(true, 512, false, 1024));
    check(!far_attention_eligible(true, 512, false, 1023));
    check(!far_attention_cell(1023, 0, 1024));
    check(far_attention_cell(1024, 0, 1024));
    check(!far_attention_cell(1024, 1, 1024));
    check(far_attention_cell(16383, 15359, 1024));
    check(!far_attention_cell(16383, 15360, 1024));
    // Verify each SG keeps precisely the old ascending stride-8 key sequence.
    // The far/non-far boundary never drops a key or adds a future key.
    for (int position : {0, 15, 1023, 1024, 1025, 1031, 4095, 16383}) {
        int far = 0, near = 0;
        for (int sg = 0; sg < 8; ++sg) {
            for (int key = sg; key <= position; key += 8) {
                const bool selected = far_attention_cell(position, key, 1024);
                check(selected == (position - key >= 1024));
                if (selected) ++far; else ++near;
            }
        }
        check(far == std::max(0, position - 1024 + 1));
        check(far + near == position + 1);
    }
    // The elided rescale is exactly one; l retains its old expression. This
    // scalar identity is not a claim of GPU compiler bit identity or speed.
    for (float maximum : {-100.f, -1.f, 0.f, 1.f, 100.f}) {
        check(std::exp(maximum - maximum) == 1.f);
        for (float accumulator : {-4.f, -.25f, .25f, 4.f}) {
            for (float term : {-.5f, 0.f, .5f}) {
                const float legacy = accumulator * std::exp(maximum - maximum) + term;
                const float candidate = accumulator + term;
                check(legacy == candidate);
            }
        }
    }
    std::printf("Inkling far-attention policy: %u checks passed; GPU comparison pending\n", checks);
}
