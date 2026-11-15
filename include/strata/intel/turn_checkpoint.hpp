#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strata::intel::turn_checkpoint {
enum class Source { reset, live, saved_turn };
struct Reuse { Source source = Source::reset; size_t tokens = 0; };
inline bool enabled(const char* requested) {
    if (!requested) return true;
    const std::string_view value(requested);
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::invalid_argument("TINYSTRATA_INTEL_TURN_CHECKPOINT must be 0 or 1");
}

// A recurrent snapshot cannot restore KV rows. Both the resident consumed
// sequence and the new request must retain its complete exact token prefix.
inline Reuse choose(const std::vector<int>& live, const std::vector<int>& prompt,
                    const std::vector<int>& saved, size_t saved_position,
                    bool checkpoint_enabled) {
    if (live.size() <= prompt.size() && std::equal(live.begin(), live.end(), prompt.begin()))
        return {Source::live, live.size()};
    if (checkpoint_enabled && saved_position && saved_position == saved.size() &&
        saved.size() <= live.size() && saved.size() <= prompt.size() &&
        std::equal(saved.begin(), saved.end(), live.begin()) &&
        std::equal(saved.begin(), saved.end(), prompt.begin()))
        return {Source::saved_turn, saved.size()};
    return {};
}
} // namespace strata::intel::turn_checkpoint
