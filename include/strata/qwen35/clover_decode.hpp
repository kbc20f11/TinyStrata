// Clover host-side self-drafting and response-only generation controls.
//
// The masked-block/causal-verify protocol was checked against llama.cpp
// PR #31306 (4838ffb), common_speculative_impl_draft_self (MIT), and SGLang
// PR #46927 (2243dc6), clover_utils.py. Penalty semantics were checked against
// Diffuspace/Clover-35B-A3B-Preview (4b0402b), sampling_clover_moe.py
// (Apache-2.0). This is an independent C++ implementation of those rules.
// Optional entropy acceptance uses the rule in llama.cpp PR #31306 (cb09d43),
// common/sampling.cpp (MIT), with caller-penalized logits. Unlike that sampler,
// a terminal target argmax always ends the prefix instead of being overridden.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::qwen35::clover_decode {

inline constexpr int kMaxDrafts = 31;
inline constexpr int32_t kMask = 151669;
inline constexpr int32_t kEndOfText = 248044;
inline constexpr int32_t kEndOfTurn = 248046;
inline constexpr int32_t kThinkEnd = 248069;

struct Plan {
    int preview_rows = 0;  // Fixed configured width, including the known anchor.
    int draft_count = 0;   // May be smaller than the preview width minus one.
    explicit operator bool() const { return draft_count > 0; }
};

// A shorter output budget, or a cache-maintenance boundary, can reduce the
// proposals verified without changing the block-dependent preview operation.
// If the entire configured preview cannot fit, use a normal causal step.
inline Plan plan(int configured_drafts, int64_t position, int64_t context,
                 int verify_budget = -1) {
    if (configured_drafts < 0 || configured_drafts > kMaxDrafts)
        throw std::invalid_argument("Clover drafts must be between 0 and 31");
    if (position < 0 || context < 0 || position > context || verify_budget < -1)
        throw std::invalid_argument("Invalid Clover decoding position or budget");
    const int count = verify_budget < 0 ? configured_drafts : std::min(configured_drafts, verify_budget);
    if (count == 0 || context - position < configured_drafts + 1) return {};
    return {configured_drafts + 1, count};
}

inline void validate_plan(const Plan& p) {
    if (p.preview_rows < 2 || p.preview_rows > kMaxDrafts + 1 ||
        p.draft_count < 1 || p.draft_count >= p.preview_rows)
        throw std::invalid_argument("Invalid Clover preview plan");
}

inline std::vector<int32_t> masked_block(int32_t anchor, const Plan& p, int32_t mask = kMask) {
    validate_plan(p);
    std::vector<int32_t> tokens(static_cast<size_t>(p.preview_rows), mask);
    tokens[0] = anchor;
    return tokens;
}

// Preview logits predict the SAME position. Row zero is the supplied anchor;
// proposals begin at row one. This differs from the causal verify alignment.
inline std::vector<int32_t> drafts_from_preview(std::span<const int32_t> preview_picks, const Plan& p) {
    validate_plan(p);
    if (preview_picks.size() < static_cast<size_t>(p.preview_rows))
        throw std::invalid_argument("Incomplete Clover preview picks");
    return {preview_picks.begin() + 1, preview_picks.begin() + 1 + p.draft_count};
}

inline std::vector<int32_t> verify_inputs(int32_t anchor, std::span<const int32_t> drafts) {
    if (drafts.size() > kMaxDrafts) throw std::invalid_argument("Too many Clover draft tokens");
    std::vector<int32_t> tokens;
    tokens.reserve(drafts.size() + 1);
    tokens.push_back(anchor);
    tokens.insert(tokens.end(), drafts.begin(), drafts.end());
    return tokens;
}

struct Accepted {
    int accepted = 0;
    int keep = 1;       // Causal inputs committed: anchor plus accepted drafts.
    int32_t next = 0;   // Selected boundary token, not processed or committed.
};

// selected[row] must be the target's token for that row's valid response
// prefix, with all requested penalties applied. Only a contiguous exact prefix
// may be retained; threshold acceptance is deliberately not implemented here.
// A terminal selected token is returned as the boundary, never pre-committed.
template <class Terminal>
Accepted exact_prefix(std::span<const int32_t> drafts, std::span<const int32_t> selected,
                      Terminal terminal) {
    if (drafts.size() > kMaxDrafts || selected.size() < drafts.size() + 1)
        throw std::invalid_argument("Incomplete Clover causal verification");
    size_t row = 0;
    while (row < drafts.size() && !terminal(selected[row]) && selected[row] == drafts[row]) ++row;
    return {static_cast<int>(row), static_cast<int>(row) + 1, selected[row]};
}

inline Accepted exact_prefix(std::span<const int32_t> drafts, std::span<const int32_t> selected) {
    return exact_prefix(drafts, selected, [](int32_t t) { return t == kEndOfText || t == kEndOfTurn; });
}

struct AcceptOptions {
    float ratio = 0.9f;
    float delta = 0.6f;
};

inline void validate_accept_options(const AcceptOptions& options) {
    if (!std::isfinite(options.ratio) || options.ratio < 0 || options.ratio > 1 ||
        !std::isfinite(options.delta) || options.delta < 0 || options.delta > 1)
        throw std::invalid_argument("Clover acceptance ratio and delta must be finite and in [0, 1]");
}

struct ThresholdStats {
    int32_t selected = 0;       // Target argmax, lowest token ID first on ties.
    float probability = 0;     // Full-vocabulary probability of the proposed token.
    float entropy = 0;         // Full-vocabulary entropy in natural logarithms.
};

// The caller applies its penalties before this T=1 distribution is measured.
// A masked -infinity has zero probability; NaN, +infinity and an entirely
// masked vocabulary are invalid. Proposal -1 denotes the bonus row: its
// probability is unused/zero, while argmax and entropy remain well-defined.
inline ThresholdStats threshold_stats(std::span<const float> logits, int32_t proposal) {
    if (logits.empty() || logits.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        proposal < -1 || (proposal >= 0 && static_cast<size_t>(proposal) >= logits.size()))
        throw std::invalid_argument("Invalid Clover acceptance logits or proposed token");
    float maximum = -std::numeric_limits<float>::infinity();
    int32_t selected = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float value = logits[i];
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
            throw std::invalid_argument("Nonfinite Clover acceptance logits");
        if (value > maximum) { maximum = value; selected = static_cast<int32_t>(i); }
    }
    if (!std::isfinite(maximum))
        throw std::invalid_argument("Clover acceptance logits have no finite token");
    double sum = 0, weighted_gap = 0;
    for (float value : logits) {
        if (!std::isfinite(value)) continue;
        const double gap = static_cast<double>(maximum) - value;
        const double mass = std::exp(-gap);
        sum += mass;
        weighted_gap += mass * gap;
    }
    // H = log(sum exp(x-max)) + E[max-x] avoids log(0) for tiny probabilities.
    const double entropy = std::log(sum) + weighted_gap / sum;
    const double probability = proposal < 0 ? 0.0 :
        std::exp(static_cast<double>(logits[static_cast<size_t>(proposal)]) - maximum) / sum;
    return {selected, static_cast<float>(probability), static_cast<float>(entropy)};
}

// This is approximate acceptance, not greedy equivalence or distribution-
// preserving rejection sampling. Only the contiguous accepted prefix matters;
// rows after a rejection are ignored because their drafted history is invalid.
// Both target-terminal and accepted draft-terminal tokens are returned as an
// uncommitted boundary. Target argmax terminals take priority, deliberately
// differing from upstream entropy acceptance which may override them.
template <class Terminal>
Accepted threshold_prefix(std::span<const int32_t> drafts, std::span<const ThresholdStats> stats,
                          float ratio, float delta, Terminal terminal) {
    const AcceptOptions options{ratio, delta};
    validate_accept_options(options);
    if (drafts.size() > kMaxDrafts || stats.size() < drafts.size() + 1)
        throw std::invalid_argument("Incomplete Clover threshold verification");
    const auto checked = [](const ThresholdStats& row) {
        if (row.selected < 0 || !std::isfinite(row.probability) || row.probability < 0 || row.probability > 1 ||
            !std::isfinite(row.entropy) || row.entropy < 0)
            throw std::invalid_argument("Invalid Clover acceptance statistics");
    };
    size_t row = 0;
    for (; row < drafts.size(); ++row) {
        checked(stats[row]);
        if (drafts[row] < 0) throw std::invalid_argument("Invalid Clover drafted token");
        if (terminal(stats[row].selected)) break;
        const float threshold = std::min(options.ratio, options.delta * std::exp(-stats[row].entropy));
        // A zero-probability proposal is never admissible, even when the
        // configured threshold is zero or exp(-entropy) underflows.
        if (!(stats[row].probability > 0) || stats[row].probability < threshold) break;
        if (terminal(drafts[row]))
            return {static_cast<int>(row), static_cast<int>(row) + 1, drafts[row]};
    }
    checked(stats[row]);
    return {static_cast<int>(row), static_cast<int>(row) + 1, stats[row].selected};
}

inline Accepted threshold_prefix(std::span<const int32_t> drafts, std::span<const ThresholdStats> stats,
                                 float ratio = 0.9f, float delta = 0.6f) {
    return threshold_prefix(drafts, stats, ratio, delta,
        [](int32_t t) { return t == kEndOfText || t == kEndOfTurn; });
}

struct PenaltyOptions {
    float presence = 0.9f;
    float frequency = 0.0f;
    float repetition = 1.05f;
    float length_strength = 10.0f;
    int64_t length_start = 8192, length_window = 8192;
    float think_strength = 10.0f;
    int64_t think_start = 4096, think_window = 4096;
    int32_t think_end = kThinkEnd;
    std::vector<int32_t> stops = {kEndOfText, kEndOfTurn};
};

inline float ramp(int64_t predicted_position, int64_t start, int64_t window, float strength) {
    if (window < 1 || start < 0 || predicted_position < 0 || !std::isfinite(strength) || strength < 0)
        throw std::invalid_argument("Invalid Clover stopping-bias ramp");
    const float fraction = (static_cast<float>(predicted_position) - static_cast<float>(start)) /
                           static_cast<float>(window);
    return std::clamp(fraction, 0.0f, 1.0f) * strength;
}

// Supply ONLY the generated response prefix for this causal prediction, never
// the prompt or an unverified speculative suffix. For a verification row that
// matched earlier drafts, its prefix includes those matched drafts exactly
// once. Calling this on fresh logits before sampling gives causal and exact
// verification the same generation controls. It does not implement HF's
// LowConfidence block-snapshot sampling or distribution-preserving sampling.
inline void apply_penalties(std::span<float> logits, std::span<const int32_t> response,
                            const PenaltyOptions& options = {}) {
    const auto valid = [&](int32_t id) { return id >= 0 && static_cast<size_t>(id) < logits.size(); };
    if (!std::isfinite(options.presence) || !std::isfinite(options.frequency) ||
        !std::isfinite(options.repetition) || options.repetition <= 0 ||
        (options.think_strength != 0 && !valid(options.think_end)))
        throw std::invalid_argument("Invalid Clover response penalties");
    const int64_t next_position = static_cast<int64_t>(response.size()) + 1;
    const float length_bias = ramp(next_position, options.length_start, options.length_window, options.length_strength);
    const float think_bias = ramp(next_position, options.think_start, options.think_window, options.think_strength);
    for (int32_t stop : options.stops)
        if (!valid(stop)) throw std::invalid_argument("Invalid Clover stop token");
    std::map<int32_t, int64_t> counts;
    for (int32_t id : response) if (valid(id)) ++counts[id];
    for (const auto& [id, count] : counts) {
        float& value = logits[static_cast<size_t>(id)];
        // Keep the model's frequency -> presence -> repetition operation order.
        if (options.frequency != 0) value -= static_cast<float>(count) * options.frequency;
        if (options.presence != 0) value -= options.presence;
        if (options.repetition != 1) value = value < 0 ? value * options.repetition : value / options.repetition;
    }
    // Duplicate stop IDs must not receive the same bias twice.
    std::vector<int32_t> stops = options.stops;
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    for (int32_t stop : stops) logits[static_cast<size_t>(stop)] += length_bias;
    if (think_bias != 0 && !counts.contains(options.think_end))
        logits[static_cast<size_t>(options.think_end)] += think_bias;
}

}  // namespace strata::qwen35::clover_decode
