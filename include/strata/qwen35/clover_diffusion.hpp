// Clover LowConfidence block generation controls. This independent host
// implementation follows Diffuspace/Clover-35B-A3B-Preview (4b0402b),
// sampling_clover_moe.py and modeling_clover_moe.py (Apache-2.0).
#pragma once

#include "clover_decode.hpp"

#include <limits>
#include <utility>

namespace strata::qwen35::clover_diffusion {

using PenaltyOptions = clover_decode::PenaltyOptions;
inline constexpr int kMaxBlock = 32;
inline constexpr int32_t kMask = clover_decode::kMask;

inline void validate_block_size(int size) {
    if (size < 1 || size > kMaxBlock || kMaxBlock % size != 0)
        throw std::invalid_argument("Clover block size must be a positive divisor of 32");
}

inline void validate_threshold(float threshold) {
    if (!std::isfinite(threshold) || threshold < 0.f || threshold > 1.f)
        throw std::invalid_argument("Clover confidence threshold must be in [0, 1]");
}

// Every final partial block is still resolved at the complete configured width.
// Validate the entire request before prefill, just as HF generate() does.
inline int64_t preview_span(int64_t max_new, int block_size) {
    validate_block_size(block_size);
    if (max_new < 0 || max_new > std::numeric_limits<int64_t>::max() - (block_size - 1))
        throw std::invalid_argument("Invalid Clover output budget");
    return ((max_new + block_size - 1) / block_size) * block_size;
}

inline void validate_capacity(int block_size, int64_t position, int64_t max_new, int64_t context) {
    const int64_t preview = preview_span(max_new, block_size);
    if (position < 0 || context < 0 || position > context || preview > context - position)
        throw std::invalid_argument("Clover request exceeds context including full preview blocks");
}

// Tensor operations in the reference round frequency multiplication, each
// subtraction and repetition separately to FP32. Prevent host FMA contraction
// or reassociation from erasing those boundaries under a fast-math build.
inline float fp32(float value) {
    volatile float rounded = value;
    return rounded;
}

inline float row_ramp(int64_t position, int64_t start, int64_t window, float strength) {
    const float offset = fp32(static_cast<float>(position) - static_cast<float>(start));
    const float fraction = fp32(offset / static_cast<float>(window));
    return fp32(std::clamp(fraction, 0.f, 1.f) * strength);
}

class PenaltyState {
public:
    explicit PenaltyState(int vocab, PenaltyOptions options = {})
        : vocab_(vocab), options_(std::move(options)) {
        if (vocab_ < 1) throw std::invalid_argument("Invalid Clover vocabulary size");
        if (!std::isfinite(options_.presence) || !std::isfinite(options_.frequency) ||
            !std::isfinite(options_.repetition) || options_.repetition <= 0.f ||
            !std::isfinite(options_.length_strength) || options_.length_strength < 0.f ||
            !std::isfinite(options_.think_strength) || options_.think_strength < 0.f ||
            options_.length_start < 0 || options_.length_window < 1 ||
            options_.think_start < 0 || options_.think_window < 1)
            throw std::invalid_argument("Invalid Clover block penalties");
        if (options_.think_strength != 0.f && !valid(options_.think_end))
            throw std::invalid_argument("Invalid Clover thinking-end token");
        std::sort(options_.stops.begin(), options_.stops.end());
        options_.stops.erase(std::unique(options_.stops.begin(), options_.stops.end()), options_.stops.end());
        for (int32_t stop : options_.stops)
            if (!valid(stop)) throw std::invalid_argument("Invalid Clover stopping token");
        counts_.resize(static_cast<size_t>(vocab_));
    }

    int vocab_size() const { return vocab_; }
    int64_t length() const { return length_; }
    std::span<const int64_t> counts() const { return counts_; }
    const PenaltyOptions& options() const { return options_; }

    // Commit only tokens actually returned, once per completely resolved block.
    // Prompt IDs and uncommitted/frozen block positions never enter this state.
    void commit(std::span<const int32_t> returned) {
        if (returned.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max() - length_))
            throw std::invalid_argument("Clover response length overflow");
        for (int32_t token : returned)
            if (!valid(token)) throw std::invalid_argument("Invalid committed Clover token");
        for (int32_t token : returned) ++counts_[static_cast<size_t>(token)];
        length_ += static_cast<int64_t>(returned.size());
    }

    void reset() {
        std::fill(counts_.begin(), counts_.end(), int64_t{0});
        length_ = 0;
    }

    // logits are fresh [rows, vocab] preview logits, aligned to the SAME block
    // position. Call this once before sampling; do not reapply causal penalties
    // in the sampler. All rows share the committed-history snapshot. Visible
    // block tokens affect only thinking-end suppression, not repetition counts.
    void apply(std::span<float> logits, int rows, std::span<const int32_t> block,
               int32_t mask = kMask) const {
        validate_block_size(rows);
        if (block.size() != static_cast<size_t>(rows) ||
            logits.size() != static_cast<size_t>(rows) * static_cast<size_t>(vocab_) || !valid(mask))
            throw std::invalid_argument("Invalid Clover block-logit shape or mask");
        if (length_ > std::numeric_limits<int64_t>::max() - rows)
            throw std::invalid_argument("Clover block position overflow");
        bool close_seen = false;
        if (options_.think_strength != 0.f) {
            close_seen = counts_[static_cast<size_t>(options_.think_end)] != 0;
            for (int32_t token : block)
                close_seen = close_seen || (token != mask && token == options_.think_end);
        }
        const float minus_inf = -std::numeric_limits<float>::infinity();
        for (int row = 0; row < rows; ++row) {
            auto values = logits.subspan(static_cast<size_t>(row) * static_cast<size_t>(vocab_), static_cast<size_t>(vocab_));
            values[static_cast<size_t>(mask)] = minus_inf;
            if (options_.frequency != 0.f || options_.presence != 0.f || options_.repetition != 1.f) {
                for (int token = 0; token < vocab_; ++token) {
                    const int64_t count = counts_[static_cast<size_t>(token)];
                    float value = values[static_cast<size_t>(token)];
                    if (options_.frequency != 0.f)
                        value = fp32(value - fp32(static_cast<float>(count) * options_.frequency));
                    if (options_.presence != 0.f)
                        value = fp32(value - fp32(static_cast<float>(count != 0) * options_.presence));
                    if (count != 0 && options_.repetition != 1.f)
                        value = value < 0.f ? fp32(value * options_.repetition) : fp32(value / options_.repetition);
                    values[static_cast<size_t>(token)] = value;
                }
            }
            const int64_t position = length_ + row + 1;
            if (options_.length_strength != 0.f) {
                const float bias = row_ramp(position, options_.length_start, options_.length_window, options_.length_strength);
                for (int32_t stop : options_.stops)
                    values[static_cast<size_t>(stop)] = fp32(values[static_cast<size_t>(stop)] + bias);
            }
            if (options_.think_strength != 0.f) {
                const float bias = close_seen ? 0.f : row_ramp(position, options_.think_start, options_.think_window, options_.think_strength);
                const auto close = static_cast<size_t>(options_.think_end);
                values[close] = fp32(values[close] + bias);
            }
        }
    }

private:
    bool valid(int32_t token) const { return token >= 0 && token < vocab_; }
    int vocab_;
    PenaltyOptions options_;
    std::vector<int64_t> counts_;
    int64_t length_ = 0;
};

inline int32_t greedy(std::span<const float> logits) {
    if (logits.empty()) throw std::invalid_argument("Empty Clover logit row");
    int32_t best = 0;
    for (size_t token = 0; token < logits.size(); ++token) {
        const float value = logits[token];
        if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
            throw std::runtime_error("Nonfinite Clover preview logits");
        if (value > logits[static_cast<size_t>(best)]) best = static_cast<int32_t>(token);
    }
    if (!std::isfinite(logits[static_cast<size_t>(best)]))
        throw std::runtime_error("Clover preview has no finite candidate");
    return best;
}

// Confidence is the selected token's T=1 FULL-vocabulary softmax probability,
// before temperature/top-k/top-p filtering. Double accumulation of FP32 exp
// terms avoids the large serial-FP32 sum error for near-one probabilities.
// Tensor implementations use parallel FP32 reductions; a few ULP can differ.
inline float selected_probability(std::span<const float> logits, int32_t selected) {
    if (selected < 0 || static_cast<size_t>(selected) >= logits.size())
        throw std::invalid_argument("Invalid Clover prediction");
    const int32_t best = greedy(logits);
    const float maximum = logits[static_cast<size_t>(best)];
    double sum = 0.0;
    for (float value : logits) sum += static_cast<double>(std::exp(fp32(value - maximum)));
    const float numerator = std::exp(fp32(logits[static_cast<size_t>(selected)] - maximum));
    return fp32(numerator / static_cast<float>(sum));
}

// Return ascending positions to freeze. Known positions are never revisited.
// Strict '>' and first-position argmax tie-breaking match HF. Confidence for
// known positions is deliberately ignored, so a pending-only kernel may leave
// those entries unspecified. vocab=0 omits only the optional upper-ID check.
inline std::vector<int> transfer(std::span<const int32_t> block,
                                 std::span<const int32_t> predictions,
                                 std::span<const float> confidence, float threshold,
                                 int32_t mask = kMask, int vocab = 0) {
    validate_block_size(static_cast<int>(block.size()));
    validate_threshold(threshold);
    if (predictions.size() != block.size() || confidence.size() != block.size() || vocab < 0)
        throw std::invalid_argument("Invalid Clover LowConfidence predictions");
    std::vector<int> positions;
    int best = -1;
    for (size_t row = 0; row < block.size(); ++row) {
        if (block[row] != mask) continue;
        if (!std::isfinite(confidence[row]) || confidence[row] < 0.f || confidence[row] > 1.f ||
            predictions[row] < 0 || predictions[row] == mask || (vocab > 0 && predictions[row] >= vocab))
            throw std::runtime_error("Invalid pending Clover prediction or confidence");
        if (best < 0 || confidence[row] > confidence[static_cast<size_t>(best)]) best = static_cast<int>(row);
        if (confidence[row] > threshold) positions.push_back(static_cast<int>(row));
    }
    if (positions.empty() && best >= 0) positions.push_back(best);
    return positions;
}

}  // namespace strata::qwen35::clover_diffusion
