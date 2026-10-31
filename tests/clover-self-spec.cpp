// From the repository root: c++ -std=c++20 -O2 -Iinclude tests/clover-self-spec.cpp -o clover-self-spec && ./clover-self-spec
// Host-only policy tests: no model files, GPU runtime, or external dependencies.
#include "strata/qwen35/clover_decode.hpp"

#include <array>
#include <iostream>
#include <numeric>
#include <string>

namespace C = strata::qwen35::clover_decode;
namespace {
int checks = 0;
constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float nan = std::numeric_limits<float>::quiet_NaN();

void check(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
    ++checks;
}
void near(double actual, double expected, const char* message, double tolerance = 2e-7) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}
template <class Function> void invalid(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { ++checks; return; }
    throw std::runtime_error("Expected invalid_argument");
}
C::ThresholdStats distribution(std::initializer_list<float> values, int proposal) {
    const std::vector<float> logits(values);
    return C::threshold_stats(logits, proposal);
}
void accepted(const C::Accepted& result, int count, int next) {
    check(result.accepted == count, "Incorrect accepted prefix length");
    check(result.keep == count + 1, "Committed state must include anchor, exclude boundary token");
    check(result.next == next, "Incorrect boundary token");
}

void distributions() {
    const auto uniform = distribution({0, 0, 0, 0}, 3);
    check(uniform.selected == 0, "Uniform ties must select lowest token ID");
    near(uniform.probability, .25, "Uniform proposal probability");
    near(uniform.entropy, std::log(4.), "Uniform entropy");

    // Odds 3:1 have probabilities .75/.25 and a known binary entropy.
    const auto binary = distribution({std::log(3.f), 0}, 1);
    check(binary.selected == 0, "Binary argmax");
    near(binary.probability, .25, "Binary proposal probability");
    near(binary.entropy, -.75 * std::log(.75) - .25 * std::log(.25), "Binary entropy");

    // Integer shifts are exactly representable here: test overflow-safe
    // normalization without accidentally changing the input logit differences.
    const auto small = distribution({0, -1, -2}, 2);
    const auto large = distribution({10000, 9999, 9998}, 2);
    check(small.selected == large.selected, "Shift-invariant argmax");
    near(small.probability, .09003057317038046, "Known geometric distribution");
    near(small.entropy, .8323955818399389, "Known geometric entropy");
    near(large.probability, small.probability, "Shift-invariant probability");
    near(large.entropy, small.entropy, "Shift-invariant entropy");
    const auto extreme = distribution({3e38f, -3e38f}, 1);
    check(extreme.selected == 0, "Extreme finite logits argmax");
    near(extreme.probability, 0, "Underflowed probability");
    near(extreme.entropy, 0, "Extreme finite logits entropy");

    const auto masked = distribution({-inf, 0, -inf, 0}, 0);
    check(masked.selected == 1, "Masked ties must select first finite token");
    near(masked.probability, 0, "Masked proposal has zero probability");
    near(masked.entropy, std::log(2.), "Masks excluded from distribution");
    const auto only = distribution({-inf, -5, -inf}, 1);
    near(only.probability, 1, "Single visible token");
    near(only.entropy, 0, "Single visible token entropy");
    const auto bonus = distribution({0, -1, -2}, -1);
    check(bonus.selected == small.selected, "Bonus row still supplies argmax");
    near(bonus.probability, 0, "Bonus sentinel has no proposal probability");
    near(bonus.entropy, small.entropy, "Bonus row still supplies entropy");

    invalid([] { distribution({}, 0); });
    invalid([] { distribution({0}, -2); });
    invalid([] { distribution({0}, 1); });
    invalid([] { distribution({0, nan}, 0); });
    invalid([] { distribution({0, inf}, 0); });
    invalid([] { distribution({-inf, -inf}, 0); });
}

void prefix_policies() {
    const std::array<int32_t, 3> drafts{10, 11, 12};
    const std::array<int32_t, 4> same{10, 11, 12, 99};
    accepted(C::exact_prefix(drafts, same), 3, 99);
    for (int mismatch = 0; mismatch < 3; ++mismatch) {
        auto selected = same;
        selected[mismatch] = 90 + mismatch;
        accepted(C::exact_prefix(drafts, selected), mismatch, 90 + mismatch);
    }
    const std::array<int32_t, 0> empty{};
    const std::array<int32_t, 1> bonus{99};
    accepted(C::exact_prefix(empty, bonus), 0, 99);
    const std::array<int32_t, 2> short_selection{10, 11};
    invalid([&] { C::exact_prefix(drafts, short_selection); });

    // Threshold mode can accept a non-argmax draft, but only a contiguous
    // prefix. A later strong proposal cannot repair an earlier rejection.
    std::array<C::ThresholdStats, 4> stats{{{20, .9f, 0}, {21, .1f, 0}, {22, .9f, 0}, {99, 0, 0}}};
    accepted(C::threshold_prefix(drafts, stats), 1, 21);
    stats[1].probability = .9f;
    accepted(C::threshold_prefix(drafts, stats), 3, 99);
    stats[1].probability = .1f;
    stats[2] = {-1, nan, nan}; // Unvisited suffix belongs to a rejected history.
    accepted(C::threshold_prefix(drafts, stats), 1, 21);

    const std::array<int32_t, 1> one{10};
    std::array<C::ThresholdStats, 2> rows{{{20, .25f, 0}, {99, 0, 0}}};
    accepted(C::threshold_prefix(one, rows, .25f, 1), 1, 99); // Equality is admissible.
    rows[0].probability = std::nextafter(.25f, 0.f);
    accepted(C::threshold_prefix(one, rows, .25f, 1), 0, 20);
    rows[0] = {20, .21f, std::log(4.f)};
    accepted(C::threshold_prefix(one, rows, 1, .8f), 1, 99); // Entropy term limits to about .2.
    rows[0].probability = .19f;
    accepted(C::threshold_prefix(one, rows, 1, .8f), 0, 20);

    // A zero configured or underflowed threshold must not admit masked tokens.
    for (float entropy : {0.f, 1000.f}) {
        rows[0] = {20, 0, entropy};
        accepted(C::threshold_prefix(one, rows, 0, 0), 0, 20);
        accepted(C::threshold_prefix(one, rows, 1, 1), 0, 20);
        rows[0].probability = 1e-30f;
        accepted(C::threshold_prefix(one, rows, 0, 0), 1, 99);
    }
    rows[0] = {20, 1, 0};
    accepted(C::threshold_prefix(one, rows, 1, 1), 1, 99);
    rows[0].probability = .99f;
    accepted(C::threshold_prefix(one, rows, 1, 1), 0, 20);

    // Target EOS takes priority even if the non-EOS draft easily passes.
    rows[0] = {C::kEndOfText, 1, 0};
    accepted(C::threshold_prefix(one, rows, 0, 0), 0, C::kEndOfText);
    accepted(C::exact_prefix(one, std::array<int32_t, 2>{C::kEndOfTurn, 99}), 0, C::kEndOfTurn);
    const std::array<int32_t, 3> eos_drafts{10, C::kEndOfTurn, 12};
    stats = {{{20, .9f, 0}, {21, .9f, 0}, {-1, nan, nan}, {-1, nan, nan}}};
    accepted(C::threshold_prefix(eos_drafts, stats), 1, C::kEndOfTurn);
    stats[1].probability = .1f;
    accepted(C::threshold_prefix(eos_drafts, stats), 1, 21);
    const std::array<int32_t, 1> eos_only{C::kEndOfTurn};
    accepted(C::threshold_prefix(eos_only, rows, 0, 0), 0, C::kEndOfText);
    accepted(C::threshold_prefix(one, std::array<C::ThresholdStats, 2>{{{20, 1, 0}, {77, 0, 0}}}, 0, 0,
        [](int id) { return id == 20; }), 0, 20); // Caller-provided stop boundary.

    for (float bad : {-1.f, 1.01f, nan, inf}) {
        invalid([&] { C::threshold_prefix(one, rows, bad, .6f); });
        invalid([&] { C::threshold_prefix(one, rows, .9f, bad); });
    }
    for (C::ThresholdStats bad : {C::ThresholdStats{-1, .5f, 0}, {20, -1, 0}, {20, 1.1f, 0},
         {20, nan, 0}, {20, inf, 0}, {20, .5f, -1}, {20, .5f, nan}, {20, .5f, inf}}) {
        rows[0] = bad;
        invalid([&] { C::threshold_prefix(one, rows); });
    }
    invalid([&] { C::threshold_prefix(one, std::span<const C::ThresholdStats>()); });
    const std::array<int32_t, 1> negative{-1};
    rows[0] = {20, 1, 0};
    invalid([&] { C::threshold_prefix(negative, rows); });
}

void block_plans() {
    const auto full = C::plan(31, 96, 128);
    check(full.preview_rows == 32 && full.draft_count == 31, "32-token canvas at exact capacity");
    const auto limited = C::plan(31, 96, 128, 3);
    check(limited.preview_rows == 32 && limited.draft_count == 3, "Budget must not shrink learned preview canvas");
    check(!C::plan(31, 97, 128, 3), "Partial preview capacity must fall back to causal");
    check(!C::plan(31, 96, 128, 0) && !C::plan(0, 0, 128), "No drafts without generation budget");
    const auto end = std::numeric_limits<int64_t>::max();
    check(C::plan(31, end - 32, end).preview_rows == 32, "Capacity arithmetic near int64 limit");
    const auto canvas = C::masked_block(7, limited);
    check(canvas.size() == 32 && canvas[0] == 7 &&
        std::all_of(canvas.begin() + 1, canvas.end(), [](int id) { return id == C::kMask; }), "Known anchor and masked suffix");
    std::array<int32_t, 32> picks;
    std::iota(picks.begin(), picks.end(), 100);
    const auto drafts = C::drafts_from_preview(picks, limited);
    check(drafts == std::vector<int32_t>({101, 102, 103}), "Same-position preview excludes anchor row");
    check(C::verify_inputs(7, drafts) == std::vector<int32_t>({7, 101, 102, 103}), "Causal verification includes anchor");
    invalid([] { C::plan(32, 0, 128); });
    invalid([] { C::plan(-1, 0, 128); });
    invalid([] { C::plan(3, 129, 128); });
    invalid([] { C::plan(3, 0, 128, -2); });
    invalid([] { C::masked_block(7, C::Plan{32, 32}); });
    invalid([&] { C::drafts_from_preview(std::span<const int32_t>(picks.data(), 31), limited); });
    invalid([&] { C::verify_inputs(7, picks); });
}
} // namespace

int main() {
    try {
        distributions();
        prefix_policies();
        block_plans();
        std::cout << "Clover self-spec host tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Clover self-spec host tests failed: " << error.what() << '\n';
        return 1;
    }
}
