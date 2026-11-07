// Adaptive exact-verification depth; the policy changes work, never acceptance.
// Cost/probe ideas adapt PolyStrata's Guesses/DraftPolicy (MIT), a82999fd.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace strata::gemma4 {

class DraftPolicy {
public:
    explicit DraftPolicy(int maximum) : maximum_(std::clamp(maximum, 0, 7)), selected_(maximum_ > 1 ? 2 : maximum_) {}

    // Keep useful evidence across requests of similar size. Absolute timings
    // from a much shorter/longer attention prefix are not comparable.
    void begin_request(int64_t prompt) {
        if (previous_prompt_ && (prompt > 2 * previous_prompt_ + 512 || previous_prompt_ > 2 * prompt + 512)) {
            samples_ = {}; tokens_ = {}; ms_ = {}; at_ = {};
            accepted_ = offered_ = 0.;
            selected_ = maximum_ > 1 ? 2 : maximum_;
        }
        previous_prompt_ = std::max<int64_t>(prompt, 1);
    }

    // Start at the existing configured depth, not six cold exploration rounds.
    // High acceptance keeps it there; low acceptance revisits alternatives
    // quickly. Probes never change the target sampler's acceptance rule.
    int choose(int room) {
        ++round_;
        const int maximum = std::min(maximum_, std::max(room, 0));
        const int n = maximum > 1 ? 3 : maximum > 0 ? 2 : 1;
        const int depths[3] = {0, 1, maximum};
        if (n == 1) return 0;
        const int baseline = n - 1;
        if (maximum != maximum_ || samples_[baseline] < 4) return maximum;
        if (selected_ >= n || !fresh(selected_)) selected_ = baseline;
        if (acceptance() >= 0.7) selected_ = baseline;
        // Keep uncertain choices easy to overturn. A measured, clear winner
        // needs fewer probes; recheck the baseline before its 32-round evidence
        // expires, avoiding both stale costs and repeated cold calibration.
        const bool clear_lead = selected_ != baseline &&
            samples_[selected_] >= 3 && samples_[baseline] >= 3 &&
            fresh(selected_) && fresh(baseline) &&
            tokens_[selected_] * ms_[baseline] >
                1.10 * tokens_[baseline] * ms_[selected_];
        const int recovery = clear_lead ? 24 : 8;
        if (selected_ != baseline && round_ - at_[baseline] > recovery) return depths[baseline];
        const int period = acceptance() >= 0.7 ? 96 : clear_lead ? 16 : acceptance() < 0.6 ? 8 : 32;
        if (round_ % period == 0) {
            for (int i = 0; i < n; ++i) {
                const int probe = next_probe_++ % n;
                if (probe != selected_) return depths[probe];
            }
        }
        return depths[selected_];
    }

    void observe(int depth, int emitted, double ms, bool recorded, int offered = -1, int accepted = -1) {
        if (recorded || emitted < 1 || !std::isfinite(ms) || ms <= 0) return;
        const int slot = depth == 0 ? 0 : depth == 1 ? 1 : depth == maximum_ ? 2 : -1;
        if (slot < 0 || depth > maximum_) return;
        const bool recent = samples_[slot] && fresh(slot);
        const double keep = recent ? 0.8 : 0.;
        if (!recent) samples_[slot] = 0;
        tokens_[slot] = keep * tokens_[slot] + (1. - keep) * emitted;
        ms_[slot] = keep * ms_[slot] + (1. - keep) * ms;
        ++samples_[slot];
        at_[slot] = round_;
        if (depth == maximum_ && depth > 0) {
            offered = offered < 0 ? depth : offered;
            accepted = accepted < 0 ? std::clamp(emitted - 1, 0, offered) : accepted;
            if (offered > 0 && accepted >= 0 && accepted <= offered) {
                offered_ = 0.8 * offered_ + offered;
                accepted_ = 0.8 * accepted_ + accepted;
            }
        }
        if (samples_[selected_] < 3 || !fresh(selected_)) return;
        for (int i = 0; i < (maximum_ > 1 ? 3 : maximum_ + 1); ++i)
            if (i != selected_ && samples_[i] >= 3 && fresh(i) &&
                tokens_[i] * ms_[selected_] > 1.05 * tokens_[selected_] * ms_[i]) selected_ = i;
    }

    double acceptance() const { return offered_ > 0 ? accepted_ / offered_ : 0.; }
    int samples(int depth) const { const int slot = index(depth); return slot < 0 ? 0 : samples_[slot]; }
    double milliseconds_per_token(int depth) const {
        const int slot = index(depth);
        return slot < 0 || tokens_[slot] <= 0 ? 0. : ms_[slot] / tokens_[slot];
    }

private:
    int index(int depth) const { return depth == 0 ? 0 : depth == 1 ? 1 : depth == maximum_ ? 2 : -1; }
    bool fresh(int slot) const { return at_[slot] > 0 && round_ - at_[slot] <= 32; }
    int maximum_ = 0, next_probe_ = 0, selected_ = 0;
    int64_t round_ = 0, previous_prompt_ = 0;
    std::array<int, 3> samples_{};
    std::array<int64_t, 3> at_{};
    std::array<double, 3> tokens_{}, ms_{};
    double accepted_ = 0., offered_ = 0.;
};

// A short accidental match should not keep paying for rejected verify rows.
// Strong-copy matches keep their own evidence, so bad triples cannot disable
// the first long exact continuation later in the answer.
class LookupDraftGate {
public:
    bool allows(int match) {
        State& s = states_[match >= 8];
        ++s.opportunities;
        bool allow = s.windows < 2;
        if (!allow) {
            const double acceptance = s.offered > 0 ? s.accepted / s.offered : 0.;
            const bool pays = fallback_ms_ <= 0 ||
                s.emitted * fallback_ms_ >= 0.98 * fallback_emitted_ * s.ms;
            allow = acceptance >= 0.5 && pays;
            if (!allow) allow = s.opportunities - s.last_probe >= 32;
        }
        if (allow) s.last_probe = s.opportunities;
        return allow;
    }

    void observe(int match, int offered, int accepted, int emitted, double ms, bool recorded) {
        if (recorded || offered < 1 || accepted < 0 || accepted > offered || emitted < 1 ||
            !std::isfinite(ms) || ms <= 0) return;
        State& s = states_[match >= 8];
        const double keep = s.windows ? 0.8 : 0.;
        s.offered = keep * s.offered + offered;
        s.accepted = keep * s.accepted + accepted;
        s.emitted = keep * s.emitted + emitted;
        s.ms = keep * s.ms + ms;
        ++s.windows;
    }

    void fallback(int emitted, double ms, bool recorded) {
        if (recorded || emitted < 1 || !std::isfinite(ms) || ms <= 0) return;
        const double keep = fallback_ms_ > 0 ? 0.8 : 0.;
        fallback_emitted_ = keep * fallback_emitted_ + emitted;
        fallback_ms_ = keep * fallback_ms_ + ms;
    }

    double milliseconds_per_token(bool strong) const {
        const State& s = states_[strong];
        return s.emitted > 0 ? s.ms / s.emitted : 0.;
    }

private:
    struct State {
        int64_t opportunities = 0, last_probe = 0;
        int windows = 0;
        double offered = 0., accepted = 0., emitted = 0., ms = 0.;
    };
    std::array<State, 2> states_{};
    double fallback_emitted_ = 0., fallback_ms_ = 0.;
};

// Stop before an existing physical KV boundary; the single-token path grows
// there at the same position as ordinary causal decoding.
inline int speculation_width(int maximum, int64_t pos, int64_t context,
                             int64_t committed, bool growing, int remaining) {
    int64_t room = std::min<int64_t>({std::max(maximum, 1), context - pos,
                                    int64_t(std::max(remaining, 0)) + 1});
    if (growing && committed < context && committed > pos)
        room = std::min(room, committed - pos);
    else if (growing && committed <= pos && committed < context)
        room = std::min<int64_t>(room, 1);
    return int(std::max<int64_t>(1, room));
}

} // namespace strata::gemma4
