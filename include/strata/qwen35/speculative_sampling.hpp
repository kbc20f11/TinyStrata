#pragma once

#include <cstdint>

namespace strata::qwen35::speculative_sampling {

// Visit only the target rows whose contexts are known to be valid. A matching
// draft extends that context; a mismatch (or terminal token) ends the window.
// The last selected token is the next, as yet unprocessed token. In particular,
// discarded rows consume neither random draws nor repetition-penalty history.
template <class Select, class Stop>
int select_prefix(const int32_t* drafts, int count, int32_t* picks, Select select, Stop stop) {
    for (int row = 0;; ++row) {
        picks[row] = select(row);
        if (row == count || stop(picks[row]) || picks[row] != drafts[row]) return row;
    }
}

} // namespace strata::qwen35::speculative_sampling
