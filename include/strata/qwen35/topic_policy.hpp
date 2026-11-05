// Prompt-frequency seeding adapted from jesdga95/basalt (MIT),
// src/program/serve/tiers.cpp. Admission and Expert capacities stay with the caller.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::qwen35::topic_policy {

struct Parameters {
    double seed_tokens = 64.0;
    double history_keep = 0.3;
    size_t max_new_nonresident = 32;
};

struct SeedStats {
    size_t resident_seeded = 0;
    size_t nonresident_seeded = 0;
    double scale = 0.0;
};

// Count only newly read prompt rows. Keep this histogram separate from the
// retained decode usage, and discard it if the prompt is cancelled or fails.
inline bool record(std::vector<uint64_t>& histogram, size_t expert, uint64_t hits = 1) {
    if (expert >= histogram.size() || hits > std::numeric_limits<uint64_t>::max() - histogram[expert])
        return false;
    histogram[expert] += hits;
    return true;
}

// Apply once after a complete prompt, before the caller's ordinary admission.
// The cap is per layer because each layer owns a fixed-capacity Expert tier.
// Resident Experts keep their prompt signal without consuming the new-entry cap.
// Invalid input leaves usage unchanged; no newly read tokens is a no-op.
inline bool seed_prompt(std::vector<double>& usage, const std::vector<uint64_t>& histogram,
                        uint64_t newly_read_tokens, const std::vector<int>& slot_of,
                        const Parameters& parameters = {}, SeedStats* stats = nullptr) {
    if (stats) *stats = {};
    const size_t n = usage.size();
    if (histogram.size() != n || slot_of.size() != n ||
        !std::isfinite(parameters.seed_tokens) || parameters.seed_tokens <= 0.0 ||
        !std::isfinite(parameters.history_keep) || parameters.history_keep < 0.0 || parameters.history_keep > 1.0)
        return false;
    std::vector<uint8_t> seen(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(usage[i]) || usage[i] < 0.0 || slot_of[i] < -1 ||
            (slot_of[i] >= 0 && (size_t) slot_of[i] >= n))
            return false;
        if (slot_of[i] >= 0 && seen[(size_t) slot_of[i]]++) return false;
    }
    if (newly_read_tokens == 0 || n == 0) return true;

    std::vector<size_t> candidates;
    candidates.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (slot_of[i] < 0 && histogram[i] > 0) candidates.push_back(i);
    const size_t take = std::min(parameters.max_new_nonresident, candidates.size());
    if (take > 0) {
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(),
                          [&](size_t a, size_t b) {
                              return histogram[a] != histogram[b] ? histogram[a] > histogram[b] : a < b;
                          });
    }
    std::vector<uint8_t> selected(n, 0);
    for (size_t i = 0; i < take; ++i) selected[candidates[i]] = 1;

    SeedStats result;
    result.scale = parameters.seed_tokens / (double) newly_read_tokens;
    if (!std::isfinite(result.scale)) return false;
    std::vector<double> next(n);
    for (size_t i = 0; i < n; ++i) {
        const bool seed = slot_of[i] >= 0 || selected[i];
        next[i] = usage[i] * parameters.history_keep +
                  (seed ? (double) histogram[i] * result.scale : 0.0);
        if (!std::isfinite(next[i])) return false;
        if (seed && histogram[i] > 0) {
            if (slot_of[i] >= 0) ++result.resident_seeded;
            else ++result.nonresident_seeded;
        }
    }
    usage.swap(next);
    if (stats) *stats = result;
    return true;
}

} // namespace strata::qwen35::topic_policy
