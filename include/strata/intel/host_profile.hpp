#pragma once
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
namespace strata::intel::host_profile {
using Clock = std::chrono::steady_clock;
enum Phase { TargetTotal, TargetCopy, TargetPenalties, TargetIndices, TargetRank, TargetFilter,
             DraftTotal, DraftCopy, DraftPenalties, DraftIndices, DraftRank, DraftFilter,
             TargetDrawVerify, DraftDraw, SupportCpu, SupportWait, SupportCopy, Count };
inline constexpr const char* names[Count] = {
    "target_distribution", "target_copy", "target_penalties", "target_indices", "target_rank", "target_filter",
    "draft_distribution", "draft_copy", "draft_penalties", "draft_indices", "draft_rank", "draft_filter",
    "target_draw_verify", "draft_draw", "support_selection", "support_queue_wait", "support_ids_copy"};
struct Counter { uint64_t calls = 0; double milliseconds = 0; };
inline thread_local std::array<Counter, Count> counters{};
inline thread_local bool enabled = false;
inline Clock::time_point now() { return enabled ? Clock::now() : Clock::time_point{}; }
inline thread_local bool force_target = false;
struct TargetScope {
    bool previous = force_target;
    TargetScope() { force_target = true; }
    ~TargetScope() { force_target = previous; }
};
inline void add(Phase phase, Clock::time_point start, Clock::time_point end) {
    if (!enabled) return;
    auto& counter = counters[size_t(phase)];
    ++counter.calls;
    counter.milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
}
struct Scope {
    Phase phase;
    Clock::time_point start = now();
    explicit Scope(Phase value) : phase(value) {}
    ~Scope() { if (enabled) add(phase, start, Clock::now()); }
};
inline void reset() { counters = {}; }
inline void print() {
    if (!enabled) return;
    for (int phase = 0; phase < Count; ++phase) {
        const auto& counter = counters[size_t(phase)];
        std::fprintf(stderr, "host_profile phase=%s calls=%llu wall_ms=%.6f\n", names[phase],
                     static_cast<unsigned long long>(counter.calls), counter.milliseconds);
    }
}
}
