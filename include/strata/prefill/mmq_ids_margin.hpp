// MMQ's row-map read bound, adapted from Niko1221/Strata f225844d19eec380cab7c3083cc644ec37e972c5 (MIT).
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
// llama.cpp's stream-k fixup loads a whole tile of IDs before checking its expert's row bound. An empty late
// expert can start at the last logical row and still read roundup(max_rows, J) IDs. Leave another tile after it.
// This changes storage only: Product::total_rows, routing, activations and output rows stay logical-sized.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::prefill::mmq {
constexpr int kMmqTileColsMax = 128;
constexpr int kMmqTileColsStep = 8;

inline bool mmq_ids_margin_rows(int64_t max_rows, int tile_cols_max, int64_t* out) {
    if (max_rows < 0 || tile_cols_max <= 0 || out == nullptr) return false;
    const int64_t J = tile_cols_max;
    const int64_t tiles = max_rows / J + (max_rows % J != 0 ? 1 : 0);
    if (tiles > INT64_MAX / J) return false;
    const int64_t rounded = tiles * J;
    if (rounded > INT64_MAX - J) return false;
    *out = rounded + J;
    return true;
}

inline uint64_t mmq_ids_table_rows(uint64_t logical_rows, uint64_t max_rows,
                                 int tile_cols_max = kMmqTileColsMax) {
    int64_t margin = 0;
    if (max_rows > (uint64_t) INT64_MAX || !mmq_ids_margin_rows((int64_t) max_rows, tile_cols_max, &margin) ||
        logical_rows > UINT64_MAX - (uint64_t) margin) return 0;
    return logical_rows + (uint64_t) margin;
}

struct IdsLayout {
    size_t logical_rows = 0, storage_rows = 0, storage_bytes = 0;
};

// IDs (including iota's initialized margin) are int32. Reject unrepresentable shapes rather than truncate an ID,
// byte size or initialization count; callers retain their existing admission/fallback behaviour on failure.
inline bool ids_layout(uint64_t logical_rows, uint64_t max_rows, IdsLayout& out) {
    const uint64_t storage = mmq_ids_table_rows(logical_rows, max_rows);
    if (storage == 0 || storage > (uint64_t) INT32_MAX ||
        storage > (uint64_t) std::numeric_limits<size_t>::max() / sizeof(int32_t)) return false;
    out = {(size_t) logical_rows, (size_t) storage, (size_t) storage * sizeof(int32_t)};
    return true;
}

// Gemma/LFM pack token, destination-ID, bound, weight and class tables together. The destination table also backs
// subspans (a staged group's d_place + row0), so its margin must precede the following bounds rather than alias them.
struct PackedIdsLayout {
    IdsLayout ids;
    size_t place = 0, bounds = 0, relative = 0, weights = 0, classes = 0, words = 0, bytes = 0;
};

inline bool packed_ids_layout(uint64_t logical_rows, uint64_t max_rows, uint64_t experts, uint64_t classes,
                              PackedIdsLayout& out) {
    PackedIdsLayout p;
    if (!ids_layout(logical_rows, max_rows, p.ids) || experts > (uint64_t) INT32_MAX ||
        classes > (uint64_t) INT32_MAX) return false;
    // All operands are now bounded by INT32_MAX. Even the class table fits uint64; check the final byte size
    // before converting offsets to size_t (including on a 32-bit host).
    const uint64_t place = logical_rows, bounds = place + p.ids.storage_rows;
    const uint64_t relative = bounds + experts + 1, weights = relative + 2 * experts + 2;
    const uint64_t class_offset = weights + logical_rows;
    const uint64_t words = class_offset + classes * (experts + 1);
    if (words > (uint64_t) std::numeric_limits<size_t>::max() / sizeof(int32_t)) return false;
    p.place = (size_t) place; p.bounds = (size_t) bounds; p.relative = (size_t) relative;
    p.weights = (size_t) weights; p.classes = (size_t) class_offset;
    p.words = (size_t) words; p.bytes = p.words * sizeof(int32_t);
    out = p;
    return true;
}
}  // namespace strata::prefill::mmq
