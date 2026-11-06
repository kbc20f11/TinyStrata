// Indexed prompt/output suffix proposals, adapted from PolyStrata's
// include/strata/spec/suffix_drafter.hpp and src/spec/suffix_drafter.cpp
// at a82999fd60c2b6b3dbfb3924df60f8ff756c4371 (MIT; see NOTICE).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::qwen35 {

// Proposals are untrusted token values; the caller must verify them with the
// causal model before appending accepted tokens to this history.
class SuffixLookup {
public:
    static constexpr int kMaxMatch = 64;
    static constexpr int kMaxDraft = 7;
    static constexpr size_t kMaxCapacity = size_t(1) << 20;

    explicit SuffixLookup(int min_match = 3, int max_match = kMaxMatch,
                          size_t capacity = size_t(1) << 18)
        : min_match_(min_match), max_match_(max_match), capacity_(capacity) {
        if (min_match < 3 || max_match < min_match || max_match > kMaxMatch ||
            capacity < 4 || capacity > kMaxCapacity)
            throw std::invalid_argument("invalid suffix lookup bounds");
        size_t slots = 1;
        while (slots < capacity * 2) slots <<= 1;
        table_.resize(slots);
        mask_ = slots - 1;
        history_.reserve(capacity);
        previous_.reserve(capacity);
    }

    void reset() {
        history_.clear();
        previous_.clear();
        std::fill(table_.begin(), table_.end(), Slot{});
        last_match_ = 0;
    }

    // Invalid replacements leave the old history intact.
    void rebuild(const int32_t* tokens, size_t n) {
        validate_input(tokens, n, capacity_);
        reset();
        append(tokens, n);
    }

    // The fixed capacity is a context bound, rather than an eviction policy.
    // The owner rebuilds this index whenever it changes the retained prefix.
    void append(const int32_t* tokens, size_t n) {
        validate_input(tokens, n, capacity_ - history_.size());
        if (!n) return;
        last_match_ = 0;
        for (size_t i = 0; i < n; ++i) {
            history_.push_back(tokens[i]);
            previous_.push_back(0);
            const size_t end = history_.size() - 1;
            if (end < 2) continue;
            Slot* slot = find(key_at(end), true);
            previous_[end] = slot->head;
            slot->head = uint32_t(end + 1);
        }
    }
    void append(int32_t token) { append(&token, 1); }

    // The longest previous suffix wins; equal matches keep the newest one.
    // Hash collisions are candidates only: every token of the match is checked.
    // Overlapping matches may read their continuation through the current end.
    int propose(int max_draft, int32_t* out) {
        last_match_ = 0;
        if (max_draft <= 0) return 0;
        if (!out) throw std::invalid_argument("null suffix proposal output");
        const size_t n = history_.size();
        if (n < size_t(min_match_) + 1) return 0;
        const size_t current = n - 1;
        const Slot* slot = find(key_at(current), false);
        if (!slot) return 0;
        size_t best_end = 0;
        int best_length = 0;
        for (uint32_t link = slot->head; link; ) {
            const size_t end = size_t(link - 1);
            link = previous_[end];
            if (end >= current) continue;
            int length = 0;
            while (length < max_match_ && size_t(length) <= end &&
                   history_[end - size_t(length)] == history_[current - size_t(length)]) ++length;
            if (length > best_length) {
                best_length = length;
                best_end = end;
                if (best_length == max_match_) break;
            }
        }
        if (best_length < min_match_) return 0;
        last_match_ = best_length;
        const int count = int(std::min<size_t>(size_t(std::min(max_draft, kMaxDraft)), current - best_end));
        std::copy_n(history_.data() + best_end + 1, count, out);
        return count;
    }

    size_t size() const { return history_.size(); }
    int last_match() const { return last_match_; }

private:
    struct Slot { uint64_t key = 0; uint32_t head = 0; };
    int min_match_, max_match_;
    size_t capacity_, mask_ = 0;
    int last_match_ = 0;
    std::vector<int32_t> history_;
    std::vector<uint32_t> previous_;
    std::vector<Slot> table_;

    static void validate_input(const int32_t* tokens, size_t n, size_t room) {
        if (n && !tokens) throw std::invalid_argument("null suffix history input");
        if (n > room) throw std::length_error("suffix history exceeds context capacity");
    }
    static uint64_t mix(uint64_t value) {
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdull;
        value ^= value >> 33;
        value *= 0xc4ceb9fe1a85ec53ull;
        return value ^ (value >> 33);
    }
    uint64_t key_at(size_t end) const {
        const uint64_t a = uint32_t(history_[end - 2]), b = uint32_t(history_[end - 1]), c = uint32_t(history_[end]);
        return mix(a * 0x9e3779b97f4a7c15ull ^ mix(b + 0x632be59bd9b4e019ull) ^ (c << 1)) | 1ull;
    }
    Slot* find(uint64_t key, bool insert) {
        for (size_t index = size_t(key) & mask_, probes = 0; probes < table_.size();
             index = (index + 1) & mask_, ++probes) {
            Slot& slot = table_[index];
            if (slot.key == key) return &slot;
            if (!slot.key) {
                if (!insert) return nullptr;
                slot.key = key;
                return &slot;
            }
        }
        // At most capacity-2 indexed positions occupy at least 2*capacity slots.
        throw std::logic_error("suffix lookup index exhausted");
    }
};

} // namespace strata::qwen35
