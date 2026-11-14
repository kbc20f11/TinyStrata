#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::intel {
// Build only a draft proposal's support from raw target logits. Recent tokens
// reserve min(128, max(1, limit/4)) slots; the rest are the highest raw-logit
// IDs. Neither the target logits nor target sampler changes.
inline std::vector<int32_t> make_draft_vocabulary(const float* target, int vocabulary, int limit,
                                                const std::vector<int>& history) {
    if (!target || vocabulary <= 0 || limit <= 0 || limit > vocabulary)
        throw std::invalid_argument("Invalid draft vocabulary limit or target");
    for (int id = 0; id < vocabulary; ++id)
        if (!std::isfinite(target[id])) throw std::invalid_argument("Nonfinite target logit in draft vocabulary");
    for (int id : history)
        if (id < 0 || id >= vocabulary) throw std::invalid_argument("History ID is outside the target vocabulary");

    const int reserve = std::min(128, std::max(1, limit / 4));
    std::vector<int32_t> selected;
    selected.reserve(size_t(limit));
    const size_t first = history.size() > 128 ? history.size() - 128 : 0;
    for (size_t i = history.size(); i > first && int(selected.size()) < reserve; --i) {
        const int id = history[i - 1];
        if (std::find(selected.begin(), selected.end(), id) == selected.end()) selected.push_back(id);
    }
    std::sort(selected.begin(), selected.end());
    const auto reserved = selected;

    std::vector<int32_t> ranked(size_t(vocabulary), 0);
    std::iota(ranked.begin(), ranked.end(), 0);
    const auto higher = [target](int32_t a, int32_t b) {
        return target[a] > target[b] || (target[a] == target[b] && a < b);
    };
    // At most `limit` raw leaders suffice even if every reserved ID is among
    // them. Partition scratch, then order just that prefix; never mutate target.
    if (limit < vocabulary) std::nth_element(ranked.begin(), ranked.begin() + limit, ranked.end(), higher);
    ranked.resize(size_t(limit));
    std::sort(ranked.begin(), ranked.end(), higher);
    for (int32_t id : ranked) {
        if (int(selected.size()) == limit) break;
        if (!std::binary_search(reserved.begin(), reserved.end(), id)) selected.push_back(id);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}
}
