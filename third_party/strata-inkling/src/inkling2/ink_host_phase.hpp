// Publish a shared activation only after every gate/up row has been written.
#pragma once

#include <atomic>

namespace strata::ink {

template <class Prepare>
inline void finish_shared_activation(std::atomic<int>& remaining, Prepare prepare) noexcept {
    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        prepare();
        // Zero means that the last writer is still preparing; -1 publishes the bytes.
        remaining.store(-1, std::memory_order_release);
    }
}

inline bool split_activation_ready(const std::atomic<int>& remaining, bool shared) noexcept {
    return remaining.load(std::memory_order_acquire) <= (shared ? -1 : 0);
}

}  // namespace strata::ink
