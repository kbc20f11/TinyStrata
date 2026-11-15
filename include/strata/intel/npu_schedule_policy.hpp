#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::intel::npu_policy {

// Independent CPU admission/control policy. It owns no model/runtime buffers
// and selects no experts, token IDs, arithmetic mode or sampling parameters.
enum class Projection { qkv, gate, state_output, attention_q, attention_output };
struct Shape {
    Projection projection;
    int weight_type;
    std::size_t rows, width, tokens;
};
struct RowRule {
    Projection projection;
    std::size_t min_tokens, max_tokens, npu_rows;
};
struct SplitLimits {
    int weight_type = 14;
    std::size_t width = 2048, min_tokens = 128, max_tokens = 1024;
    std::size_t row_alignment = 256;
    std::size_t max_prepared_weight_bytes = 128u * 1024u * 1024u;
};
struct Partition {
    std::size_t npu_rows = 0, gpu_rows = 0, prepared_weight_bytes = 0;
    bool offloaded() const noexcept { return npu_rows != 0; }
};

// Rules are explicit experiment/calibration inputs, not imported AMD defaults
// or an online tuner. An empty table leaves every projection entirely on GPU.
class SharePolicy {
    std::vector<RowRule> rules_;
    SplitLimits limits_;
public:
    explicit SharePolicy(std::vector<RowRule> rules = {}, SplitLimits limits = {})
        : rules_(std::move(rules)), limits_(limits) {
        if (!limits_.width || !limits_.row_alignment || !limits_.min_tokens ||
            limits_.min_tokens > limits_.max_tokens)
            throw std::invalid_argument("Invalid NPU split limits");
        for (std::size_t i = 0; i < rules_.size(); ++i) {
            const auto& a = rules_[i];
            if (!a.min_tokens || a.min_tokens > a.max_tokens ||
                a.npu_rows % limits_.row_alignment)
                throw std::invalid_argument("Invalid NPU row rule");
            for (std::size_t j = 0; j < i; ++j) {
                const auto& b = rules_[j];
                if (a.projection == b.projection &&
                    a.min_tokens <= b.max_tokens && b.min_tokens <= a.max_tokens)
                    throw std::invalid_argument("Ambiguous NPU token-range rules");
            }
        }
    }
    Partition select(const Shape& shape) const noexcept {
        Partition result{0, shape.rows, 0};
        if (shape.weight_type != limits_.weight_type || shape.width != limits_.width ||
            shape.tokens < limits_.min_tokens || shape.tokens > limits_.max_tokens)
            return result;
        for (const auto& rule : rules_) {
            if (rule.projection != shape.projection || shape.tokens < rule.min_tokens ||
                shape.tokens > rule.max_tokens) continue;
            // Keep both output intervals complete and disjoint. Backend fallback
            // must never lose an unaligned tail or silently move every row.
            if (!rule.npu_rows || rule.npu_rows >= shape.rows ||
                shape.width > std::numeric_limits<std::size_t>::max() / 2 ||
                rule.npu_rows > std::numeric_limits<std::size_t>::max() / (2 * shape.width))
                return result;
            const auto bytes = rule.npu_rows * shape.width * 2;
            if (bytes > limits_.max_prepared_weight_bytes) return result;
            return {rule.npu_rows, shape.rows - rule.npu_rows, bytes};
        }
        return result;
    }
};

struct PairResources {
    bool separate_GPU_banks = false;
    bool separate_NPU_request_buffers = false;
    bool extra_memory_admitted = false;
    bool ordered_state_dependencies = false;
    bool capture_or_vision = false;
};
struct ChunkPair {
    std::size_t start = 0, first_tokens = 0, second_tokens = 0;
    bool paired() const noexcept { return second_tokens != 0; }
};
inline ChunkPair choose_pair(std::size_t start, std::size_t remaining,
                             std::size_t chunk, std::size_t context,
                             const PairResources& resources,
                             std::size_t next_checkpoint = std::numeric_limits<std::size_t>::max()) {
    if (!chunk || start > context || remaining > context - start)
        throw std::invalid_argument("Invalid NPU pair context/chunk bounds");
    ChunkPair result{start, std::min(chunk, remaining), 0};
    // Initially pair only two original full chunks. Do not change a tail's GEMM
    // shape or cross an exact turn checkpoint merely to create another bank.
    if (remaining < chunk || remaining - chunk < chunk ||
        !resources.separate_GPU_banks || !resources.separate_NPU_request_buffers ||
        !resources.extra_memory_admitted || !resources.ordered_state_dependencies ||
        resources.capture_or_vision) return result;
    const auto end = start + chunk + chunk; // bounded above by start+remaining
    if (next_checkpoint > start && next_checkpoint < end) return result;
    result.second_tokens = chunk;
    return result;
}

enum class Stage : unsigned { linear_submit, state_apply, output_submit, finish_layer };
struct Action {
    unsigned bank;
    std::size_t layer;
    Stage stage;
    std::uint64_t epoch;
};
inline bool operator==(const Action& a, const Action& b) noexcept {
    return a.bank == b.bank && a.layer == b.layer && a.stage == b.stage && a.epoch == b.epoch;
}

// Advance only after the named operation has been submitted with all GPU event
// dependencies established. Retire an NPU slot only after inference completion
// and its output consumer/copy have finished. GDN/conv/KV state is shared: bank1
// state_apply cannot pass bank0's same-layer state_apply. It is not a second
// model state. The executor remains responsible for actual event dependencies.
class PairSchedule {
    std::array<std::size_t, 2> progress_{};
    std::array<bool, 2> pending_{};
    std::size_t layers_;
    std::uint64_t epoch_;
    unsigned turn_ = 0;
    bool cancelled_ = false;
    void check_bank(unsigned bank, std::uint64_t epoch) const {
        if (bank > 1 || epoch != epoch_) throw std::logic_error("Stale NPU bank/epoch");
    }
public:
    PairSchedule(std::size_t layers, std::uint64_t epoch) : layers_(layers), epoch_(epoch) {
        if (!layers || layers > std::numeric_limits<std::size_t>::max() / 4)
            throw std::invalid_argument("Invalid NPU pair layer count");
    }
    std::optional<Action> next() const noexcept {
        if (cancelled_) return std::nullopt;
        for (unsigned offset = 0; offset < 2; ++offset) {
            const auto bank = (turn_ + offset) % 2;
            if (progress_[bank] < layers_ * 4)
                return Action{bank, progress_[bank] / 4,
                    static_cast<Stage>(progress_[bank] % 4), epoch_};
        }
        return std::nullopt;
    }
    void submitted(unsigned bank, std::uint64_t epoch) {
        check_bank(bank, epoch);
        const auto action = next();
        if (cancelled_ || pending_[bank] || !action || action->bank != bank ||
            (action->stage != Stage::linear_submit && action->stage != Stage::output_submit))
            throw std::logic_error("NPU request buffer is not available");
        pending_[bank] = true;
    }
    void retire(unsigned bank, std::uint64_t epoch) {
        check_bank(bank, epoch);
        if (!pending_[bank]) throw std::logic_error("No matching NPU request to retire");
        pending_[bank] = false;
    }
    void advance(const Action& action) {
        check_bank(action.bank, action.epoch);
        const auto wanted = next();
        if (!wanted || !(action == *wanted)) throw std::logic_error("Out-of-order NPU pair action");
        if ((action.stage == Stage::state_apply || action.stage == Stage::finish_layer) && pending_[action.bank])
            throw std::logic_error("NPU output has not been retired before its consumer");
        if (action.bank == 1 && action.stage == Stage::state_apply &&
            progress_[0] <= action.layer * 4 + 1)
            throw std::logic_error("Later chunk cannot update recurrent/KV state first");
        ++progress_[action.bank];
        // Yield after async submits, keeping this bank's GPU consumer/state
        // work together until the next NPU submit. This can hide that submit
        // behind the partner's state/FFN GPU work instead of introducing an
        // artificial barrier after every individual GPU operation.
        if (action.stage == Stage::linear_submit || action.stage == Stage::output_submit ||
            progress_[action.bank] == layers_ * 4)
            turn_ = 1 - action.bank;
    }
    void cancel() noexcept { cancelled_ = true; }
    bool drained() const noexcept { return !pending_[0] && !pending_[1]; }
    bool publishable() const noexcept {
        return !cancelled_ && drained() && progress_[0] == layers_ * 4 && progress_[1] == layers_ * 4;
    }
};

} // namespace strata::intel::npu_policy
