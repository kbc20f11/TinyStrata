#include "strata/intel/draft_vocabulary.hpp"
#include "strata/intel/sampling_policy.hpp"
#include <array>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

using strata::intel::make_draft_vocabulary;
using strata::intel::sampling_policy::Sampling;
using strata::intel::sampling_policy::sample;
using strata::intel::sampling_policy::sampling_distribution;
using strata::intel::speculative_distribution::Distribution;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1.e-12, message);
}
template<class Function> void invalid(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Invalid draft-vocabulary input was accepted");
}
void membership(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected) {
    require(actual == expected, "Draft vocabulary differs from manual expected membership");
    require(std::is_sorted(actual.begin(), actual.end()), "Draft IDs are not sorted");
    require(std::adjacent_find(actual.begin(), actual.end()) == actual.end(), "Draft IDs are not unique");
}
void weights(const Distribution& actual, const std::vector<int>& ids, const std::vector<double>& expected) {
    require(actual.tokens == ids && actual.probabilities.size() == expected.size(), "Sampling support/order changed");
    const double total = std::accumulate(expected.begin(), expected.end(), 0.);
    for (size_t i = 0; i < ids.size(); ++i) {
        close(actual.probabilities[i], expected[i], "Unnormalized sampling weight differs from manual calculation");
        close(actual.mass(ids[i]) / actual.total(), expected[i] / total, "Normalized probability differs from manual softmax");
    }
}

void vocabulary_cases() {
    const std::vector<float> raw{9, 8, 7, -100, 6, 5};
    const auto original = raw;
    membership(make_draft_vocabulary(raw.data(), 6, 4, {}), {0, 1, 2, 4});
    membership(make_draft_vocabulary(raw.data(), 6, 4, {1, 3, 5}), {0, 1, 2, 5});
    membership(make_draft_vocabulary(raw.data(), 6, 1, {4, 3, 3}), {3});
    membership(make_draft_vocabulary(raw.data(), 6, 1, {}), {0});
    membership(make_draft_vocabulary(raw.data(), 6, 6, {3, 3}), {0, 1, 2, 3, 4, 5});
    require(raw == original, "Draft selection mutated target logits");

    const std::vector<float> ties(6, 0);
    membership(make_draft_vocabulary(ties.data(), 6, 4, {}), {0, 1, 2, 3});
    membership(make_draft_vocabulary(ties.data(), 6, 4, {5}), {0, 1, 2, 5});

    // Only the last 128 tokens may supply reserved IDs, even if duplicate
    // recent tokens leave spare reserve capacity.
    std::vector<float> window_logits(16, -100);
    const std::array<int, 7> leaders{0, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < leaders.size(); ++i) window_logits[size_t(leaders[i])] = float(20 - i);
    std::vector<int> history{1, 2};
    history.insert(history.end(), 128, 15);
    membership(make_draft_vocabulary(window_logits.data(), 16, 8, history), {0, 3, 4, 5, 6, 7, 8, 15});
    history.assign(127, 15);
    history.insert(history.begin(), 1);  // Oldest allowed token is included.
    membership(make_draft_vocabulary(window_logits.data(), 16, 8, history), {0, 1, 3, 4, 5, 6, 7, 15});

    // The quarter-support rule changes at 512, where its 128-ID cap binds.
    std::vector<float> ascending(700);
    std::iota(ascending.begin(), ascending.end(), 0.f);
    history.resize(200);
    std::iota(history.begin(), history.end(), 0);
    std::vector<int32_t> expected;
    for (int i = 72; i <= 199; ++i) expected.push_back(i);
    for (int i = 316; i <= 699; ++i) expected.push_back(i);
    membership(make_draft_vocabulary(ascending.data(), 700, 512, history), expected);
    expected.erase(expected.begin());
    membership(make_draft_vocabulary(ascending.data(), 700, 511, history), expected);
}

void invalid_cases() {
    const std::array<float, 3> raw{0, 1, 2};
    for (int limit : {-1, 0, 4}) invalid([&] { make_draft_vocabulary(raw.data(), 3, limit, {}); });
    invalid([&] { make_draft_vocabulary(nullptr, 3, 2, {}); });
    invalid([&] { make_draft_vocabulary(raw.data(), 0, 1, {}); });
    invalid([&] { make_draft_vocabulary(raw.data(), -1, 1, {}); });
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        const std::array<float, 3> nonfinite{9, 8, bad};
        invalid([&] { make_draft_vocabulary(nonfinite.data(), 3, 1, {}); });
    }
    for (int bad : {-1, 3}) {
        invalid([&] { make_draft_vocabulary(raw.data(), 3, 2, {bad}); });
        std::vector<int> old_invalid(129, 0);
        old_invalid[0] = bad;
        invalid([&] { make_draft_vocabulary(raw.data(), 3, 2, old_invalid); });
    }
}

Sampling penalty_policy() {
    Sampling s;
    s.temperature = 1;
    s.top_k = 4;
    s.repeat = 2;
    s.frequency = .5;
    s.presence = 1;
    s.last_n = -1;
    return s;
}

void full_policy_cases() {
    const std::array<float, 4> raw{4, -2, 0, 3};
    const std::vector<int> history{0, 0, 1, 2};
    auto s = penalty_policy();
    // Manual adjusted logits: [0, -5.5, -1.5, 3].
    weights(sampling_distribution(raw.data(), 4, s, history), {3, 0, 2, 1},
            {1, std::exp(-3.), std::exp(-4.5), std::exp(-8.5)});
    s.temperature = 0;
    std::mt19937_64 rng(19), untouched = rng;
    s.seed = 123456;  // The caller owns RNG seeding; Sampling.seed does not reseed it.
    require(sample(raw.data(), 4, s, history, rng) == 3 && rng == untouched, "Greedy sampling consumed RNG or changed penalties");
    s.last_n = 2;
    require(sample(raw.data(), 4, s, history, rng) == 0, "last_n did not limit penalty history");
    s.last_n = 0;
    require(sample(raw.data(), 4, s, history, rng) == 0, "last_n=0 did not disable penalties");
    require(rng == untouched, "Greedy policy advanced RNG");

    const std::array<float, 4> ties{1, 1, 1, 1};
    s = Sampling{};
    require(sample(ties.data(), 4, s, {}, rng) == 0, "Greedy tie no longer picks first token");
    s.temperature = 1;
    s.top_k = 4;
    s.min_p = 1;
    const auto equal = sampling_distribution(ties.data(), 4, s, {});
    require(equal.tokens.size() == 4, "min_p equality incorrectly removed tied candidates");
    s.top_p = .5;
    require(sampling_distribution(ties.data(), 4, s, {}).tokens.size() == 2, "top_p equality must stop with >=");

    const std::array<float, 4> descending{3, 2, 1, 0};
    s.min_p = 0;
    s.top_p = .8f;
    weights(sampling_distribution(descending.data(), 4, s, {}), {0, 1}, {1, std::exp(-1.)});
    s.top_p = 1;
    s.min_p = .5;
    weights(sampling_distribution(descending.data(), 4, s, {}), {0}, {1});
    s.min_p = 2;  // Preserve at least the maximum even if no row passes min_p.
    require(sampling_distribution(descending.data(), 4, s, {}).tokens == std::vector<int>{0}, "Sampling lost keep>=1 safeguard");

    std::array<float, 70> many{};
    std::iota(many.begin(), many.end(), 0.f);
    s = Sampling{};
    s.temperature = 100;
    for (int top_k : {0, -1}) {
        s.top_k = top_k;
        const auto distribution = sampling_distribution(many.data(), 70, s, {});
        require(distribution.tokens.size() == 64 && distribution.tokens.front() == 69 && distribution.tokens.back() == 6,
                "top_k<=0 no longer means 64");
    }
    const float singleton = 0;
    require(sample(&singleton, 1, s, {}, rng) == 0 && rng != untouched, "Stochastic singleton did not consume RNG");
}

void compact_policy_cases() {
    const std::array<float, 4> raw{2, -1, 3, 0};
    const std::array<int32_t, 4> ids{2, 7, 10, 17};
    const std::vector<int> history{2, 2, 7, 7, 10, 17, 999, -3};
    auto s = penalty_policy();
    s.frequency = .25;
    s.presence = .5;
    // Real IDs -> adjusted scores: 2 -> 0, 7 -> -3, 10 -> .75,
    // 17 -> -.75. Out-of-support history IDs 999 and -3 are ignored.
    const auto distribution = sampling_distribution(raw.data(), 4, s, history, ids.data());
    weights(distribution, {10, 2, 17, 7}, {1, std::exp(-.75), std::exp(-1.5), std::exp(-3.75)});
    std::mt19937_64 rng(8), independent(8);
    for (int i = 0; i < 100; ++i)
        require(sample(raw.data(), 4, s, history, rng, ids.data()) == distribution.draw(independent),
                "Compact sample returned a row index or consumed different RNG");

    const std::array<float, 3> tied{4, -2, 0};
    const std::array<int32_t, 3> sparse_ids{7, 100, 900};
    s = penalty_policy();
    s.temperature = 0;
    const auto before = rng;
    require(sample(tied.data(), 3, s, {100, 7, 42, 7}, rng, sparse_ids.data()) == 7 && rng == before,
            "Compact greedy tie/real-ID/history mapping failed");
}

void identity_mapping_cases() {
    const std::array<float, 8> raw{1, 1, -2, 4, 0, 3, -1, 1};
    const std::array<int32_t, 8> identity{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<int> history{3, 3, 2, 7, 0, 3, 2};
    for (int last_n : {-1, 0, 3, 20}) for (int top_k : {0, 1, 4, 8}) for (float temperature : {0.f, .7f, 1.f}) {
        auto s = penalty_policy();
        s.last_n = last_n;
        s.top_k = top_k;
        s.temperature = temperature;
        s.top_p = .95f;
        s.min_p = .01f;
        const auto full = sampling_distribution(raw.data(), 8, s, history);
        const auto compact = sampling_distribution(raw.data(), 8, s, history, identity.data());
        require(full.tokens == compact.tokens && full.probabilities == compact.probabilities,
                "Identity compact mapping changed the full policy");
        std::mt19937_64 full_rng(2026), compact_rng(2026);
        for (int draw = 0; draw < 20; ++draw)
            require(sample(raw.data(), 8, s, history, full_rng) == sample(raw.data(), 8, s, history, compact_rng, identity.data()),
                    "Identity compact mapping changed seeded samples");
        require(full_rng == compact_rng, "Identity compact mapping changed RNG consumption");
    }
}

void sparse_proposal_residual_case() {
    // A full target and an arbitrary sparse proposal use real, nonconsecutive
    // IDs. Target-only IDs must remain reachable in the exact residual branch.
    const Distribution p{{2, 7, 10, 17}, {.2, .3, .1, .4}}, q{{2, 17}, {.75, .25}};
    const std::array<double, 4> residual{0, .3, .1, .15};
    const double rejection = .55;
    for (size_t i = 0; i < p.tokens.size(); ++i) {
        const int id = p.tokens[i];
        const double accepted = std::min(p.mass(id), q.mass(id));
        close(accepted + rejection * residual[i] / rejection, p.mass(id), "Sparse-q residual no longer reconstructs target");
    }
    std::mt19937_64 proposal(31), verification(32);
    std::array<int, 4> histogram{};
    constexpr int trials = 200000;
    for (int i = 0; i < trials; ++i) {
        const int output = strata::intel::speculative_distribution::verify(p, q, q.draw(proposal), verification);
        const auto found = std::find(p.tokens.begin(), p.tokens.end(), output);
        require(found != p.tokens.end(), "Sparse-q verification escaped target support");
        ++histogram[size_t(found - p.tokens.begin())];
    }
    for (size_t i = 0; i < p.tokens.size(); ++i)
        require(std::abs(double(histogram[i]) / trials - p.probabilities[i]) < .004,
                "Sparse-q residual sampling lost target-only token mass");
}
}

int main() {
    try {
        vocabulary_cases();
        invalid_cases();
        full_policy_cases();
        compact_policy_cases();
        identity_mapping_cases();
        sparse_proposal_residual_case();
        std::puts("Intel draft vocabulary: passed (selection, validation, full/compact sampling, RNG, sparse-q residual)");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Intel draft vocabulary test failed: %s\n", error.what());
        return 1;
    }
}
