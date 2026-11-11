#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::qwen35::hybrid_buffer_policy {

inline bool multiply(size_t a, size_t b, size_t& result) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    result = a * b;
    return true;
}

struct Layout {
    size_t entries = 0, values_bytes = 0, map_bytes = 0, total_bytes = 0;
};

// The two GPU result buffers only. Host routing arrays also serve streamed
// prompt chunks and deliberately retain their full tmax allocation.
inline bool layout(size_t rows, size_t top_k, size_t width, Layout& result) {
    Layout next;
    size_t values = 0;
    if (!rows || !top_k || !width || !multiply(rows, top_k, next.entries) ||
        !multiply(next.entries, width, values) ||
        !multiply(values, sizeof(float), next.values_bytes) ||
        !multiply(next.entries, sizeof(int32_t), next.map_bytes) ||
        next.values_bytes > std::numeric_limits<size_t>::max() - next.map_bytes)
        return false;
    next.total_bytes = next.values_bytes + next.map_bytes;
    result = next;
    return true;
}

inline int capacity_rows(size_t full_need, size_t free_bytes, int tmax,
                         int hybrid_rows, int verify_rows,
                         bool main_hybrid, bool mmq, bool stage_loan) {
    if (full_need <= free_bytes || !main_hybrid || !mmq || stage_loan ||
        tmax <= 0 || hybrid_rows <= 0 || verify_rows <= 0)
        return tmax;
    return std::min(tmax, std::max(hybrid_rows, verify_rows));
}

struct Transfer {
    size_t values_bytes = 0, map_bytes = 0;
};

// output_entries can be compacted, but the map still has T * top_k entries.
// Validate both independently before submitting either asynchronous copy.
inline bool transfer(int capacity, int token_rows, size_t output_entries,
                     size_t top_k, size_t width, Transfer& result) {
    if (capacity <= 0 || token_rows <= 0 || token_rows > capacity) return false;
    Layout allocation, request;
    if (!layout((size_t) capacity, top_k, width, allocation) ||
        !layout((size_t) token_rows, top_k, width, request) ||
        output_entries > request.entries || output_entries > allocation.entries)
        return false;
    Transfer next;
    size_t values = 0;
    if (!multiply(output_entries, width, values) ||
        !multiply(values, sizeof(float), next.values_bytes)) return false;
    next.map_bytes = request.map_bytes;
    if (next.values_bytes > allocation.values_bytes || next.map_bytes > allocation.map_bytes)
        return false;
    result = next;
    return true;
}

} // namespace strata::qwen35::hybrid_buffer_policy
