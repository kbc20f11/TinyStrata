// Measured proposal scheduling. Decisions never replace causal target verification.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace strata::qwen35 {

struct DraftRate {
    int samples = 0;
    double rows = 0, milliseconds = 0, accepted = 0, offered = 0;
    double rate() const { return milliseconds > 0 ? rows / milliseconds : 0; }
    bool add(int kept, int guesses, int matched, double ms, bool recorded) {
        if (recorded || kept < 1 || guesses < 0 || matched < 0 || matched > guesses ||
            !(ms > 0) || !std::isfinite(ms)) return false;
        ++samples;
        rows = .85 * rows + kept;
        milliseconds = .85 * milliseconds + ms;
        accepted = .85 * accepted + matched;
        offered = .85 * offered + guesses;
        return true;
    }
};

class DraftDepthPolicy {
public:
    explicit DraftDepthPolicy(int maximum) {
        if (maximum < 0 || maximum > 7) throw std::invalid_argument("invalid adaptive draft maximum");
        depths_[0] = 0;
        if (maximum > 0) depths_[count_++] = 1;
        if (maximum > 1) depths_[count_++] = maximum;
        leader_ = count_ - 1;
    }
    int choose() {
        ++rounds_;
        // Start with the configured depth. A fast, well-accepted baseline
        // should not pay six cold lower-depth trials on every answer.
        const auto& current = rates_[leader_];
        const double acceptance = current.offered > 0 ? current.accepted / current.offered : 0;
        const unsigned interval = acceptance >= .7 ? 96 : acceptance >= .6 ? 32 : 8;
        if (count_ > 1 && current.samples >= 3 && rounds_ - last_probe_ >= interval) {
            do { probe_ = (probe_ + 1) % count_; } while (probe_ == leader_);
            last_probe_ = rounds_;
            ++probes_;
            return depths_[probe_];
        }
        return depths_[leader_];
    }
    bool observe(int depth, int accepted, double milliseconds, bool recorded) {
        for (int i = 0; i < count_; ++i) if (depths_[i] == depth) {
            if (recorded || accepted < 0 || accepted > depth || !(milliseconds > 0) ||
                !std::isfinite(milliseconds)) return false;
            // Sparse probes must not average a new execution regime into an
            // indefinitely old cost history. A large change after a long idle
            // gap starts a fresh three-sample confirmation, never a direct
            // leader change from one unusually good probe.
            const double fresh_rate = (accepted + 1) / milliseconds;
            const double old_rate = rates_[i].rate();
            if (i != leader_ && rates_[i].samples >= 3 && rounds_ - measured_at_[i] >= 64 &&
                (fresh_rate > old_rate * 2 || fresh_rate < old_rate * .5))
                rates_[i] = DraftRate{};
            if (!rates_[i].add(accepted + 1, depth, accepted, milliseconds, recorded)) return false;
            measured_at_[i] = rounds_;
            if (rates_[leader_].samples >= 3) {
                int best = leader_;
                for (int j = 0; j < count_; ++j)
                    if (j != leader_ && rates_[j].samples >= 3 && rounds_ - measured_at_[j] <= 288 &&
                        rates_[j].rate() > rates_[leader_].rate() * 1.05 && rates_[j].rate() > rates_[best].rate())
                        best = j;
                leader_ = best;
            }
            return true;
        }
        return false;
    }
    int leader() const { return depths_[leader_]; }
    unsigned long long probes() const { return probes_; }
    const DraftRate& rate(int depth) const {
        for (int i = 0; i < count_; ++i) if (depths_[i] == depth) return rates_[i];
        throw std::invalid_argument("unmeasured draft depth");
    }
private:
    int count_ = 1, probe_ = 0, leader_ = 0;
    unsigned long long rounds_ = 0, last_probe_ = 0, probes_ = 0;
    std::array<int, 3> depths_{};
    std::array<DraftRate, 3> rates_{};
    std::array<unsigned long long, 3> measured_at_{};
};

class LookupDraftGate {
public:
    bool choose() {
        ++opportunities_;
        if (rates_[1].samples < 2) return true;
        if (rates_[0].samples < 2) return false;
        const bool lookup = rates_[1].rate() > rates_[0].rate();
        return opportunities_ % 16 == 0 ? !lookup : lookup;
    }
    void observe(bool lookup, int kept, int offered, int accepted, double milliseconds, bool recorded) {
        rates_[lookup ? 1 : 0].add(kept, offered, accepted, milliseconds, recorded);
    }
private:
    unsigned long long opportunities_ = 0;
    std::array<DraftRate, 2> rates_{};
};

} // namespace strata::qwen35
