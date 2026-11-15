#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strata::intel::moe_route_policy {
inline bool enabled(const char* setting, const char* name = "STRATA_REVIEW_INTEL_MOE_ROUTE") {
    if (!setting) return false;
    const std::string_view value(setting);
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::invalid_argument(std::string(name) + " requires 0 or 1");
}
inline bool measured_tokens(int tokens) { return tokens == 512 || tokens == 1024; }
// The measured Qwen IQ2_S gate/up and IQ3_S down projections only. No change
// to the selected experts, slot order, router weights, or short-window paths.
inline bool measured_projection(int type, int rows, int width, bool separate,
                                bool interleaved, int tokens, int experts, int picks) {
    return interleaved && measured_tokens(tokens) && experts == 256 && picks == 8 &&
        ((type == 22 && rows == 512 && width == 2048 && !separate) ||
         (type == 21 && rows == 2048 && width == 512 && separate));
}
struct Route {
    int tokens = 0, active = 0, peak = 0;
    uint32_t slots = 0, hot_slots = 0;
};
// Counts must have completed their producer before this CPU-only scan. Every
// token still selects all eight unique experts; counts alone cannot modify IDs.
inline Route analyze(const int32_t* counts, int experts, int tokens, int picks) {
    if (!counts || experts != 256 || picks != 8 || !measured_tokens(tokens))
        throw std::invalid_argument("Unsupported adaptive MoE route geometry");
    Route route; route.tokens = tokens;
    for (int expert = 0; expert < experts; ++expert) {
        const int count = counts[expert];
        if (count < 0 || count > tokens)
            throw std::invalid_argument("Invalid adaptive MoE expert count");
        route.active += count != 0;
        route.peak = std::max(route.peak, count);
        route.slots += uint32_t(count);
        if (count > tokens / 8) route.hot_slots += uint32_t(count);
    }
    if (route.slots != uint32_t(tokens * picks))
        throw std::invalid_argument("Incomplete adaptive MoE expert plan");
    return route;
}
inline bool prefer_packed_q8(const Route& route) {
    if (!measured_tokens(route.tokens) || route.slots != uint32_t(route.tokens * 8) ||
        route.active < 8 || route.active > 256 || route.peak < 1 || route.peak > route.tokens ||
        route.hot_slots > route.slots) return false;
    // Conservative experimental bounds anchored to three measured route
    // distributions, not a general T crossover. At T512 the mixed route's
    // two hot experts own 25% of slots; at T1024 that same distribution loses.
    if (route.tokens == 512)
        return route.active >= 192 && route.hot_slots <= route.slots / 4;
    if (route.tokens == 1024)
        return route.active >= 240 && route.peak <= route.tokens / 16;
    return false;
}
} // namespace strata::intel::moe_route_policy
