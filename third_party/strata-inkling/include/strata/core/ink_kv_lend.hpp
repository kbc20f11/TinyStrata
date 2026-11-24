// CUDA Inkling 2 spare global KV rows can hold routed Expert weights until the context reaches them.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::core {

class InkKvLoanPlan {
public:
    struct Slot {
        int key;
        size_t buffer;
        size_t offset;
        size_t bytes;
        int first_row;
    };

    static constexpr int keep_rows = 8192;
    static constexpr int guard_rows = 1024;
    static constexpr size_t alignment = 256;

    void init(size_t buffers, int capacity, size_t row_bytes, size_t max_experts) {
        capacity_ = capacity;
        row_bytes_ = row_bytes;
        end_offset_ = ((size_t)capacity_ * row_bytes_) & ~(alignment - 1);
        top_.assign(buffers, end_offset_);
        slots_.reserve(max_experts);
    }

    bool active() const { return !top_.empty(); }
    size_t resident() const { return slots_.size(); }
    void reset() { written_end_ = 0; }

    template <typename Release>
    void prepare_write(int end, Release release) {
        written_end_ = std::max(written_end_, end);
        if (slots_.empty() || written_end_ + guard_rows <= first_loan_) return;
        const int keep_end = std::min(capacity_, written_end_ + keep_rows);
        const auto first_drop = std::remove_if(slots_.begin(), slots_.end(), [&](const Slot& slot) {
            if (slot.first_row >= keep_end) return false;
            release(slot);
            return true;
        });
        slots_.erase(first_drop, slots_.end());
        std::fill(top_.begin(), top_.end(), end_offset_);
        first_loan_ = capacity_;
        for (const auto& slot : slots_) {
            top_[slot.buffer] = std::min(top_[slot.buffer], slot.offset);
            first_loan_ = std::min(first_loan_, slot.first_row);
        }
    }

    const Slot* lend(int key, size_t bytes) {
        const int keep_end = std::min(capacity_, written_end_ + keep_rows);
        const size_t min_offset = (size_t)keep_end * row_bytes_;
        const size_t needed = (bytes + alignment - 1) & ~(alignment - 1);
        size_t best = top_.size();
        size_t best_offset = 0;
        for (size_t i = 0; i < top_.size(); ++i) {
            if (top_[i] < needed) continue;
            const size_t off = top_[i] - needed;
            if (off >= min_offset && (best == top_.size() || off > best_offset)) {
                best = i;
                best_offset = off;
            }
        }
        if (best == top_.size()) return nullptr;
        const int first_row = static_cast<int>(best_offset / row_bytes_);
        top_[best] = best_offset;
        first_loan_ = std::min(first_loan_, first_row);
        slots_.push_back({key, best, best_offset, needed, first_row});
        return &slots_.back();
    }

private:
    int capacity_ = 0;
    int written_end_ = 0;
    int first_loan_ = 1 << 30;
    size_t row_bytes_ = 0;
    size_t end_offset_ = 0;
    std::vector<size_t> top_;
    std::vector<Slot> slots_;
};

}  // namespace strata::core
