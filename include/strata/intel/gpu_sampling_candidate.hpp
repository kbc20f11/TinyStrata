#pragma once
// Exact histogram reduction for GPU draft-support selection. Small CPU final
// selection retains raw-logit/ID ties and history inclusion. The target method
// remains an independent experiment fixture; normal inference samples target
// distributions on the CPU after its measured screen regressed.
#include "strata/intel/sampling_policy.hpp"
#include "strata/intel/draft_vocabulary.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::intel {
class GpuSamplingCandidate {
    static constexpr int bins = 65536, capacity = 8192;
    sycl::queue& queue_;
    int vocabulary_;
    float* values_ = nullptr;
    uint32_t* histogram_ = nullptr;
    uint32_t* counters_ = nullptr;
    int32_t* candidates_ = nullptr;
    int32_t* penalty_ids_ = nullptr;
    float* penalty_values_ = nullptr;
    uint64_t target_calls_ = 0, support_calls_ = 0, fallback_calls_ = 0;
    double total_ms_ = 0;
    static uint32_t bucket(float value) {
        const auto bits = sycl::bit_cast<uint32_t>(value);
        return ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u)) >> 16;
    }
    void release() noexcept {
        for (void* pointer : {static_cast<void*>(values_), static_cast<void*>(histogram_),
                              static_cast<void*>(counters_), static_cast<void*>(candidates_),
                              static_cast<void*>(penalty_ids_), static_cast<void*>(penalty_values_)})
            if (pointer) sycl::free(pointer, queue_);
    }
    // The result contains EVERY row in the threshold bucket and all higher
    // buckets. It is therefore exact, with a bounded fallback for crowded bins.
    bool leaders(const float* input, int needed, const sampling_policy::Sampling* policy,
                 const std::vector<int>& history, std::vector<int32_t>& result) {
        int penalty_count = 0;
        if (policy && policy->last_n && (policy->repeat != 1 || policy->frequency != 0 || policy->presence != 0)) {
            std::map<int, int> counts;
            const size_t first = policy->last_n < 0 || size_t(policy->last_n) >= history.size()
                ? 0 : history.size() - size_t(policy->last_n);
            for (size_t index = first; index < history.size(); ++index) ++counts[history[index]];
            for (const auto [id, seen] : counts) {
                if (id < 0 || id >= vocabulary_) throw std::invalid_argument("Invalid sampling history token");
                // Preserve the exact existing CPU float penalty arithmetic.
                float value = input[id];
                value = value > 0 ? value / policy->repeat : value * policy->repeat;
                value -= seen * policy->frequency + policy->presence;
                penalty_ids_[penalty_count] = id;
                penalty_values_[penalty_count++] = value;
            }
        }
        auto* values = values_;
        auto* histogram = histogram_;
        auto* counters = counters_;
        auto* output = candidates_;
        auto* ids = penalty_ids_;
        auto* penalties = penalty_values_;
        const int vocabulary = vocabulary_;
        queue_.memcpy(values, input, size_t(vocabulary) * sizeof(float));
        if (penalty_count) queue_.parallel_for(sycl::range<1>(size_t(penalty_count)), [=](sycl::id<1> item) {
            values[ids[item[0]]] = penalties[item[0]];
        });
        queue_.memset(histogram, 0, size_t(bins) * sizeof(uint32_t));
        queue_.memset(counters, 0, 2 * sizeof(uint32_t));
        queue_.parallel_for(sycl::range<1>(size_t(vocabulary)), [=](sycl::id<1> item) {
            const float value = values[item[0]];
            using Atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                sycl::memory_scope::device, sycl::access::address_space::global_space>;
            // Signed zero has equal CPU ordering but different ordered bit keys.
            // Preserve its original comparator behavior with a bounded fallback.
            if (!sycl::isfinite(value) || value == 0) { Atomic(counters[1]).fetch_add(1); return; }
            const auto bits = sycl::bit_cast<uint32_t>(value);
            const auto bin = ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u)) >> 16;
            Atomic(histogram[bin]).fetch_add(1);
        }).wait_and_throw();
        if (counters[1]) return false;
        uint32_t found = 0;
        int threshold = bins - 1;
        for (; threshold >= 0; --threshold) {
            found += histogram[threshold];
            if (found >= uint32_t(needed)) break;
        }
        if (threshold < 0 || found > uint32_t(capacity)) return false;
        const uint32_t cutoff = uint32_t(threshold);
        queue_.parallel_for(sycl::range<1>(size_t(vocabulary)), [=](sycl::id<1> item) {
            const float value = values[item[0]];
            const auto bits = sycl::bit_cast<uint32_t>(value);
            const auto bin = ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u)) >> 16;
            if (bin >= cutoff) {
                using Atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device, sycl::access::address_space::global_space>;
                const uint32_t slot = Atomic(counters[0]).fetch_add(1);
                if (slot < uint32_t(capacity)) output[slot] = int32_t(item[0]);
            }
        }).wait_and_throw();
        if (counters[0] != found || found > uint32_t(capacity)) return false;
        result.assign(output, output + found);
        return true;
    }
    struct Wall {
        double& total;
        std::chrono::steady_clock::time_point start = host_profile::now();
        ~Wall() { if (host_profile::enabled) total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
    };
public:
    GpuSamplingCandidate(sycl::queue& queue, int vocabulary) : queue_(queue), vocabulary_(vocabulary) {
        if (!queue.has_property<sycl::property::queue::in_order>() || vocabulary <= 0)
            throw std::invalid_argument("GPU sampling requires a positive vocabulary and in-order queue");
        try {
            values_ = sycl::malloc_shared<float>(size_t(vocabulary), queue_);
            histogram_ = sycl::malloc_shared<uint32_t>(bins, queue_);
            counters_ = sycl::malloc_shared<uint32_t>(2, queue_);
            candidates_ = sycl::malloc_shared<int32_t>(capacity, queue_);
            penalty_ids_ = sycl::malloc_shared<int32_t>(size_t(vocabulary), queue_);
            penalty_values_ = sycl::malloc_shared<float>(size_t(vocabulary), queue_);
            if (!values_ || !histogram_ || !counters_ || !candidates_ || !penalty_ids_ || !penalty_values_)
                throw std::bad_alloc();
        } catch (...) { release(); throw; }
    }
    ~GpuSamplingCandidate() { try { queue_.wait_and_throw(); } catch (...) {} release(); }
    GpuSamplingCandidate(const GpuSamplingCandidate&) = delete;
    GpuSamplingCandidate& operator=(const GpuSamplingCandidate&) = delete;
    static constexpr size_t storage_bytes(size_t vocabulary) { return vocabulary * 12 + size_t(bins) * 4 + size_t(capacity) * 4 + 8; }
    size_t bytes() const { return storage_bytes(size_t(vocabulary_)); }
    speculative_distribution::Distribution target(const float* input, const sampling_policy::Sampling& policy,
                                                  const std::vector<int>& history) {
        host_profile::TargetScope target_profile;
        Wall wall{total_ms_}; ++target_calls_;
        const int count = policy.temperature <= 0 ? 1 : std::min(policy.top_k > 0 ? policy.top_k : 64, vocabulary_);
        std::vector<int32_t> selected;
        if (!leaders(input, std::min(vocabulary_, count + 1), &policy, history, selected)) {
            ++fallback_calls_;
            return sampling_policy::sampling_distribution(input, vocabulary_, policy, history);
        }
        std::sort(selected.begin(), selected.end(), [&](int32_t a, int32_t b) {
            return values_[a] > values_[b] || (values_[a] == values_[b] && a < b);
        });
        // std::partial_sort's original equivalent-key ordering must remain
        // exactly unchanged. Any top-k/boundary tie uses that original path.
        const size_t inspect = std::min(selected.size(), size_t(count + 1));
        for (size_t index = 1; index < inspect; ++index) if (values_[selected[index - 1]] == values_[selected[index]]) {
            ++fallback_calls_;
            return sampling_policy::sampling_distribution(input, vocabulary_, policy, history);
        }
        selected.resize(size_t(count));
        std::sort(selected.begin(), selected.end());
        std::vector<float> compact; compact.reserve(selected.size());
        for (int32_t id : selected) compact.push_back(values_[id]);
        // History is already applied with precisely the host's float arithmetic.
        // Keeping empty history here prevents applying it a second time.
        return sampling_policy::sampling_distribution(compact.data(), int(compact.size()), policy, {}, selected.data());
    }
    std::vector<int32_t> support(const float* input, int limit, const std::vector<int>& history) {
        Wall wall{total_ms_}; ++support_calls_;
        if (!input || limit <= 0 || limit > vocabulary_) throw std::invalid_argument("Invalid draft support limit or target");
        for (int id : history) if (id < 0 || id >= vocabulary_) throw std::invalid_argument("Invalid draft support history");
        std::vector<int32_t> ranked;
        if (limit > capacity || !leaders(input, limit, nullptr, history, ranked)) {
            ++fallback_calls_;
            return make_draft_vocabulary(input, vocabulary_, limit, history);
        }
        std::sort(ranked.begin(), ranked.end(), [&](int32_t a, int32_t b) {
            return input[a] > input[b] || (input[a] == input[b] && a < b);
        });
        ranked.resize(size_t(limit));
        const int reserve = std::min(128, std::max(1, limit / 4));
        std::vector<int32_t> selected;
        selected.reserve(size_t(limit));
        const size_t first = history.size() > 128 ? history.size() - 128 : 0;
        for (size_t index = history.size(); index > first && int(selected.size()) < reserve; --index)
            if (std::find(selected.begin(), selected.end(), history[index - 1]) == selected.end()) selected.push_back(history[index - 1]);
        std::sort(selected.begin(), selected.end());
        const auto reserved = selected;
        for (int32_t id : ranked) {
            if (int(selected.size()) == limit) break;
            if (!std::binary_search(reserved.begin(), reserved.end(), id)) selected.push_back(id);
        }
        std::sort(selected.begin(), selected.end());
        return selected;
    }
    void print() const {
        if (!host_profile::enabled) return;
        std::fprintf(stderr, "gpu_sampling target_calls=%llu support_calls=%llu fallback_calls=%llu total_ms=%.6f scratch_bytes=%llu\n",
            static_cast<unsigned long long>(target_calls_), static_cast<unsigned long long>(support_calls_),
            static_cast<unsigned long long>(fallback_calls_), total_ms_, static_cast<unsigned long long>(bytes()));
    }
};
}
