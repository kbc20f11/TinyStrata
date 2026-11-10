#pragma once
// Picture/text run scheduling adapts PolyStrata (MIT), commit
// f0aab99757ee49bf0d9b95385fa875048f23a90a. See LICENSE and NOTICE.
// Copyright (c) 2026 VecSzn and the PolyStrata contributors.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::gemma4 {

struct ImagePromptPiece {
    int limit;
    bool read;
};

// The existing stage planner fits floor(stage / (gate + up + down))
// non-resident Experts in one round. Use the same geometry, with overflow-safe
// diagnostics, rather than a GPU/VRAM-size heuristic or a promised stage loan.
inline bool image_text_stage_fits(size_t stage, size_t missing, size_t gate, size_t down,
                                  size_t* missing_bytes = nullptr) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    if (gate > (maximum - down) / 2 || (gate == 0 && down == 0)) {
        if (missing_bytes) *missing_bytes = maximum;
        return false;
    }
    const size_t one = 2 * gate + down;
    if (missing_bytes) *missing_bytes = missing > maximum / one ? maximum : missing * one;
    return missing <= stage / one;
}

// Positions and image spans are absolute, including any reused prompt prefix.
// remaining excludes the final logits token. Image must expose at and n, as
// Engine::Image does. A read() piece must never contain even one picture row.
template<class Image>
ImagePromptPiece image_prompt_piece(int64_t at, int remaining, int max_rows, int max_read,
                                   bool pictures, bool text_prefill, const Image* images,
                                   size_t n_images, int64_t pending_mark = -1) {
    bool read = !pictures;
    int text_run = remaining;
    if (pictures && text_prefill && images && n_images) {
        for (size_t k = 0; k < n_images; ++k) {
            const auto& im = images[k];
            if (at >= im.at && at < im.at + im.n) { text_run = 0; break; }
            if (im.at > at) text_run = (int) std::min<int64_t>(text_run, im.at - at);
        }
        read = text_run >= max_rows;
    }
    int limit = std::min(read ? text_run : remaining, read ? max_read : max_rows);
    // eval() cannot write an interior mark. read() already has mark support,
    // including a mark in the middle of a long interval of ordinary text.
    if (pictures && !read && pending_mark > at)
        limit = (int) std::min<int64_t>(limit, pending_mark - at);
    return {limit, read};
}

} // namespace strata::gemma4
