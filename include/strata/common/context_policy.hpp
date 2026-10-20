// Context capacity follows installed GPU VRAM; Expert placement uses free VRAM.
#pragma once
#include <cstddef>

namespace strata::context_policy {
// Compare capacity classes rather than requiring an exact 8/12/16 GiB report.
// Driver reservations and reporting differences can make a 16 GB card read 15.x.
constexpr int for_total_vram(std::size_t total_bytes, int model_limit = 262144) {
    constexpr std::size_t gib = std::size_t{1} << 30;
    const int capacity = total_bytes <= 8 * gib + gib / 2 ? 65536
                       : total_bytes <= 12 * gib + gib / 2 ? 131072 : 262144;
    return capacity < model_limit ? capacity : model_limit;
}
} // namespace strata::context_policy
