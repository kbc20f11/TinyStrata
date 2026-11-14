#pragma once
#include "strata/intel/speculative_distribution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace strata::intel::sampling_policy {
// Same host sampling policy as the existing Qwen engine: penalties, top-k,
// temperature, min-p and top-p. This does not add a speculative approximation.
struct Sampling {
    float temperature = 0, top_p = 1, min_p = 0, repeat = 1, frequency = 0, presence = 0;
    int top_k = 20, last_n = 0;
    uint64_t seed = 0;
};

// A compact draft supplies sorted unique real token IDs, one per logit. A null
// mapping retains the full-vocabulary engine's original policy and tie order.
inline speculative_distribution::Distribution sampling_distribution(const float* logits, int count, const Sampling& s,
                                                                     const std::vector<int>& history,
                                                                     const int32_t* token_ids = nullptr) {
    std::vector<float> values(logits, logits + count);
    if (s.last_n && (s.repeat != 1 || s.frequency != 0 || s.presence != 0)) {
        std::map<int, int> counts;
        const size_t first = s.last_n < 0 || size_t(s.last_n) >= history.size() ? 0 : history.size() - size_t(s.last_n);
        for (size_t i = first; i < history.size(); ++i) ++counts[history[i]];
        for (const auto [id, seen] : counts) {
            size_t row = size_t(id);
            if (token_ids) {
                const auto position = std::lower_bound(token_ids, token_ids + count, id);
                if (position == token_ids + count || *position != id) continue;
                row = size_t(position - token_ids);
            }
            float& value = values[row];
            value = value > 0 ? value / s.repeat : value * s.repeat;
            value -= seen * s.frequency + s.presence;
        }
    }
    if (s.temperature <= 0) {
        const int row = int(std::max_element(values.begin(), values.end()) - values.begin());
        return {{token_ids ? token_ids[row] : row}, {1.}};
    }
    const int k = std::min(s.top_k > 0 ? s.top_k : 64, count);
    std::vector<int> indices(size_t(count), 0);
    for (int i = 0; i < count; ++i) indices[size_t(i)] = i;
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int a, int b) { return values[a] > values[b]; });
    std::vector<double> probabilities(size_t(k), 0);
    const double maximum = values[indices[0]];
    double total = 0;
    for (int i = 0; i < k; ++i) total += probabilities[size_t(i)] = std::exp((values[indices[size_t(i)]] - maximum) / s.temperature);
    int keep = k;
    double cumulative = 0;
    for (int i = 0; i < k; ++i) {
        if (probabilities[size_t(i)] < s.min_p * probabilities[0]) { keep = i; break; }
        cumulative += probabilities[size_t(i)] / total;
        if (cumulative >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max(1, keep);
    indices.resize(size_t(keep)); probabilities.resize(size_t(keep));
    if (token_ids) for (int& id : indices) id = token_ids[id];
    return {std::move(indices), std::move(probabilities)};
}

inline int sample(const float* logits, int count, const Sampling& s, const std::vector<int>& history,
                  std::mt19937_64& rng, const int32_t* token_ids = nullptr) {
    const auto distribution = sampling_distribution(logits, count, s, history, token_ids);
    return s.temperature <= 0 ? distribution.tokens[0] : distribution.draw(rng);
}
}
