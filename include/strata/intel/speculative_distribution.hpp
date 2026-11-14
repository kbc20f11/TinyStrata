#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace strata::intel::speculative_distribution {
// Sparse post-sampler distributions. Exact speculative rejection/residual
// sampling, Algorithm 1: https://arxiv.org/abs/2211.17192 (no lenience factor).
struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
    double mass(int token) const {
        const auto it = std::find(tokens.begin(), tokens.end(), token);
        return it == tokens.end() ? 0. : probabilities[size_t(it - tokens.begin())];
    }
    double total() const {
        if (tokens.empty() || tokens.size() != probabilities.size()) throw std::runtime_error("Invalid sampling distribution");
        double sum = 0;
        for (double p : probabilities) {
            if (!std::isfinite(p) || p < 0) throw std::runtime_error("Nonfinite or negative sampling probability");
            sum += p;
        }
        if (!(sum > 0) || !std::isfinite(sum)) throw std::runtime_error("Empty sampling probability mass");
        return sum;
    }
    template<class Random> int draw(Random& random) const {
        const double sum = total();
        double value = std::uniform_real_distribution<double>(0., sum)(random);
        int last = -1;
        for (size_t i = 0; i < tokens.size(); ++i) if (probabilities[i] > 0) {
            last = tokens[i];
            if ((value -= probabilities[i]) <= 0) return last;
        }
        return last; // Floating-point endpoint rounding: retain a positive-mass token.
    }
};
template<class Random> int verify(const Distribution& target, const Distribution& proposal, int token, Random& random) {
    const double p_total = target.total(), q_total = proposal.total();
    const double px = target.mass(token) / p_total, qx = proposal.mass(token) / q_total;
    if (!(qx > 0)) throw std::runtime_error("Draft token has no proposal probability");
    const double uniform = std::uniform_real_distribution<double>(0., 1.)(random);
    if (uniform < std::min(1., px / qx)) return token;
    Distribution residual;
    for (size_t i = 0; i < target.tokens.size(); ++i) {
        const double mass = std::max(0., target.probabilities[i] / p_total - proposal.mass(target.tokens[i]) / q_total);
        if (mass > 0) { residual.tokens.push_back(target.tokens[i]); residual.probabilities.push_back(mass); }
    }
    // A rejected proposal has zero residual mass. Never treat a rejection as
    // a kept proposal, or silently substitute unadjusted target sampling.
    const int replacement = residual.draw(random);
    if (replacement == token) throw std::runtime_error("Rejected proposal retained in residual distribution");
    return replacement;
}
}
