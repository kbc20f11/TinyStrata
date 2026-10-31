// Clover LowConfidence greedy decisions over a committed-response snapshot.
// Equations and operation order follow the separately downloaded Clover HF
// sampling_clover_moe.py (Apache-2.0). This is an independent CUDA/HIP kernel.
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::qwen35::clover_diffusion_gpu {

struct Controls {
    float frequency = 0.f, presence = .9f, repetition = 1.05f;
    float length_strength = 10.f, think_strength = 10.f;
    int64_t length_start = 8192, length_window = 8192;
    int64_t think_start = 4096, think_window = 4096;
    int32_t mask = 151669, eos0 = 248044, eos1 = 248046, think_end = 248069;
};

__device__ __forceinline__ float multiply(float a, float b) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
    return a * b;
#else
    return __fmul_rn(a, b);
#endif
}
__device__ __forceinline__ float subtract(float a, float b) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
    return a - b;
#else
    return __fsub_rn(a, b);
#endif
}
__device__ __forceinline__ float add(float a, float b) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
    return a + b;
#else
    return __fadd_rn(a, b);
#endif
}
__device__ __forceinline__ float divide(float a, float b) {
#if defined(__HIPCC__)
    return a / b;
#else
    return __fdiv_rn(a, b);
#endif
}
__device__ __forceinline__ float ramp(int64_t position, int64_t start, int64_t window, float strength) {
    const float fraction = divide(subtract((float) position, (float) start), (float) window);
    return multiply(fminf(1.f, fmaxf(0.f, fraction)), strength);
}

template <int Threads>
__global__ void __launch_bounds__(Threads) greedy_kernel(
    float* logits, const int64_t* counts, const int32_t* block, int rows, int vocab,
    int64_t response_length, Controls controls, int32_t* prediction, float* confidence, int* bad) {
    const int row = (int) blockIdx.x, thread = (int) threadIdx.x;
    if (row >= rows || block[row] != controls.mask) return;
    __shared__ float values[Threads];
    __shared__ int indices[Threads];
    __shared__ float length_bias, think_bias;
    if (thread == 0) {
        bool visible_close = counts[controls.think_end] != 0;
        for (int i = 0; i < rows; ++i)
            visible_close |= block[i] != controls.mask && block[i] == controls.think_end;
        const int64_t position = response_length + row + 1;
        length_bias = controls.length_strength == 0 ? 0.f
            : ramp(position, controls.length_start, controls.length_window, controls.length_strength);
        think_bias = controls.think_strength == 0 || visible_close ? 0.f
            : ramp(position, controls.think_start, controls.think_window, controls.think_strength);
    }
    __syncthreads();
    float best = -INFINITY;
    int best_id = 0x7fffffff;
    bool invalid = false;
    for (int id = thread; id < vocab; id += Threads) {
        float value = logits[(size_t) row * vocab + id];
        if (id == controls.mask) value = -INFINITY;
        else {
            const int64_t count = counts[id];
            invalid |= count < 0;
            if (controls.frequency != 0) value = subtract(value, multiply((float) count, controls.frequency));
            if (controls.presence != 0 && count != 0) value = subtract(value, controls.presence);
            if (controls.repetition != 1 && count != 0)
                value = value < 0 ? multiply(value, controls.repetition) : divide(value, controls.repetition);
            if (id == controls.eos0 || id == controls.eos1) value = add(value, length_bias);
            if (id == controls.think_end) value = add(value, think_bias);
            if (isnan(value) || (isinf(value) && value > 0)) {
                invalid = true;
                value = -INFINITY;
            }
        }
        logits[(size_t) row * vocab + id] = value;
        if (value > best || (value == best && id < best_id)) { best = value; best_id = id; }
    }
    if (invalid) atomicExch(bad, 1);
    values[thread] = best;
    indices[thread] = best_id;
    __syncthreads();
    for (int width = Threads / 2; width; width >>= 1) {
        if (thread < width) {
            const float other = values[thread + width];
            const int other_id = indices[thread + width];
            if (other > values[thread] || (other == values[thread] && other_id < indices[thread])) {
                values[thread] = other;
                indices[thread] = other_id;
            }
        }
        __syncthreads();
    }
    const float maximum = values[0];
    const int selected = indices[0];
    if (!isfinite(maximum)) {
        if (thread == 0) { atomicExch(bad, 1); prediction[row] = -1; confidence[row] = NAN; }
        return;
    }
    // Exclude the chosen maximum from local sums, then add its exact exp(0)=1
    // once. This avoids repeatedly rounding tiny loser terms into 1 in the
    // winning thread when confidence is close to the transfer threshold.
    __syncthreads();
    float denominator = 0.f;
    for (int id = thread; id < vocab; id += Threads)
        if (id != selected) denominator = add(denominator, expf(subtract(logits[(size_t) row * vocab + id], maximum)));
    values[thread] = denominator;
    __syncthreads();
    for (int width = Threads / 2; width; width >>= 1) {
        if (thread < width) values[thread] = add(values[thread], values[thread + width]);
        __syncthreads();
    }
    if (thread == 0) {
        const float probability = divide(1.f, add(values[0], 1.f));
        if (!isfinite(probability) || probability <= 0 || probability > 1) atomicExch(bad, 1);
        prediction[row] = selected;
        confidence[row] = probability;
    }
}

// Counts are uploaded once per committed block. Each launch sees that frozen
// snapshot plus the current mask canvas. Filled-row outputs are left untouched.
// The caller reads `bad` after the stream completes; it is reset here per call.
inline cudaError_t launch_greedy(float* logits, const int64_t* counts, const int32_t* block,
    int rows, int vocab, int64_t response_length, const Controls& controls,
    int32_t* prediction, float* confidence, int* bad, cudaStream_t stream, int threads = 512) {
    if (!logits || !counts || !block || !prediction || !confidence || !bad || rows < 1 || rows > 32 || vocab < 1 ||
        response_length < 0 || controls.mask < 0 || controls.mask >= vocab || controls.think_end < 0 ||
        controls.think_end >= vocab || controls.eos0 < 0 || controls.eos0 >= vocab || controls.eos1 < 0 ||
        controls.eos1 >= vocab || controls.length_start < 0 || controls.think_start < 0 ||
        controls.length_window < 1 || controls.think_window < 1 ||
        response_length > std::numeric_limits<int64_t>::max() - rows ||
        !std::isfinite(controls.frequency) || !std::isfinite(controls.presence) ||
        !std::isfinite(controls.repetition) || controls.repetition <= 0 ||
        !std::isfinite(controls.length_strength) || controls.length_strength < 0 ||
        !std::isfinite(controls.think_strength) || controls.think_strength < 0 ||
        (threads != 256 && threads != 512 && threads != 1024))
        throw std::invalid_argument("Invalid Clover GPU greedy controls or shape");
    const auto status = cudaMemsetAsync(bad, 0, sizeof(int), stream);
    if (status != cudaSuccess) return status;
    if (threads == 256) greedy_kernel<256><<<rows, 256, 0, stream>>>(logits, counts, block, rows, vocab, response_length, controls, prediction, confidence, bad);
    else if (threads == 1024) greedy_kernel<1024><<<rows, 1024, 0, stream>>>(logits, counts, block, rows, vocab, response_length, controls, prediction, confidence, bad);
    else greedy_kernel<512><<<rows, 512, 0, stream>>>(logits, counts, block, rows, vocab, response_length, controls, prediction, confidence, bad);
    return cudaGetLastError();
}

// Self-speculation uses one committed-response count snapshot for every row.
// Keep the input logits untouched: the existing exact/sampled CPU path may
// still need their original values. Match the diffusion penalty operation
// order, but do not consult a LowConfidence canvas or skip filled rows.
__device__ __forceinline__ float self_spec_value(
    float value, int id, const int64_t* counts, const Controls& controls,
    float length_bias, float think_bias, bool& invalid) {
    if (id == controls.mask) return -INFINITY;
    const int64_t count = counts[id];
    invalid |= count < 0;
    if (controls.frequency != 0) value = subtract(value, multiply((float) count, controls.frequency));
    if (controls.presence != 0 && count != 0) value = subtract(value, controls.presence);
    if (controls.repetition != 1 && count != 0)
        value = value < 0 ? multiply(value, controls.repetition) : divide(value, controls.repetition);
    if (id == controls.eos0 || id == controls.eos1) value = add(value, length_bias);
    if (id == controls.think_end) value = add(value, think_bias);
    if (isnan(value) || (isinf(value) && value > 0)) {
        invalid = true;
        return -INFINITY;
    }
    return value;
}

template <int Threads, bool Statistics, bool Entropy>
__global__ void __launch_bounds__(Threads) self_spec_stats_kernel(
    const float* logits, const int64_t* counts, int64_t response_length,
    const int32_t* candidates, int rows, int vocab, Controls controls,
    int32_t* prediction, float* proposal_prob, float* entropy, int32_t* bad) {
    const int row = (int) blockIdx.x, thread = (int) threadIdx.x;
    if (row >= rows) return;
    __shared__ float values[Threads];
    __shared__ int indices[Threads];
    __shared__ float gap_sums[Entropy ? Threads : 1];
    __shared__ float length_bias, think_bias;
    if (thread == 0) {
        const int64_t position = response_length + row + 1;
        length_bias = controls.length_strength == 0 ? 0.f
            : ramp(position, controls.length_start, controls.length_window, controls.length_strength);
        // SGLang self-speculation biases only its first eligible row. Preview
        // row zero is the already committed anchor, so it is never eligible.
        // With this helper's aligned response_length, position > think_start
        // first occurs at row think_start - response_length (not at ramp zero).
        const int64_t minimum_row = Statistics ? 0 : 1;
        const int64_t first_eligible = controls.think_start - response_length > minimum_row
            ? controls.think_start - response_length : minimum_row;
        think_bias = controls.think_strength == 0 || counts[controls.think_end] != 0 || row != first_eligible ? 0.f
            : ramp(position, controls.think_start, controls.think_window, controls.think_strength);
        if constexpr (Statistics) {
            if (candidates[row] < -1 || candidates[row] >= vocab) atomicExch(bad, 1);
        }
    }
    __syncthreads();
    const float* source = logits + (size_t) row * vocab;
    float best = -INFINITY;
    int best_id = 0x7fffffff;
    bool invalid = false;
    for (int id = thread; id < vocab; id += Threads) {
        const float value = self_spec_value(source[id], id, counts, controls, length_bias, think_bias, invalid);
        if (value > best || (value == best && id < best_id)) { best = value; best_id = id; }
    }
    if (invalid) atomicExch(bad, 1);
    values[thread] = best;
    indices[thread] = best_id;
    __syncthreads();
    for (int width = Threads / 2; width; width >>= 1) {
        if (thread < width) {
            const float other = values[thread + width];
            const int other_id = indices[thread + width];
            if (other > values[thread] || (other == values[thread] && other_id < indices[thread])) {
                values[thread] = other;
                indices[thread] = other_id;
            }
        }
        __syncthreads();
    }
    const float maximum = values[0];
    const int selected = indices[0];
    if (!isfinite(maximum)) {
        if (thread == 0) {
            atomicExch(bad, 1);
            prediction[row] = -1;
            if constexpr (Statistics) proposal_prob[row] = NAN;
            if constexpr (Entropy) entropy[row] = NAN;
        }
        return;
    }
    if constexpr (!Statistics) {
        if (thread == 0) prediction[row] = selected;
        return;
    } else {
        // H = log(Z) + sum(exp(-gap) * gap) / Z. The selected maximum's
        // contribution is exactly (1, 0), added once after parallel reduction.
        // A finite-logit subtraction may overflow to +inf; exp(-inf) is zero,
        // so its weighted-gap contribution is zero rather than 0 * inf.
        __syncthreads();
        float denominator = 0.f, moment = 0.f;
        for (int id = thread; id < vocab; id += Threads) {
            if (id == selected) continue;
            bool unused = false;
            const float value = self_spec_value(source[id], id, counts, controls, length_bias, think_bias, unused);
            if (!isfinite(value)) continue;
            const float gap = subtract(maximum, value);
            const float weight = expf(-gap);
            denominator = add(denominator, weight);
            if constexpr (Entropy) {
                if (isfinite(gap)) moment = add(moment, multiply(weight, gap));
            }
        }
        values[thread] = denominator;
        if constexpr (Entropy) gap_sums[thread] = moment;
        __syncthreads();
        for (int width = Threads / 2; width; width >>= 1) {
            if (thread < width) {
                values[thread] = add(values[thread], values[thread + width]);
                if constexpr (Entropy) gap_sums[thread] = add(gap_sums[thread], gap_sums[thread + width]);
            }
            __syncthreads();
        }
        if (thread == 0) {
            const float normalizer = add(values[0], 1.f);
            const int proposed = candidates[row];
            float probability = 0.f;
            if (proposed >= 0 && proposed < vocab) {
                bool unused = false;
                const float value = self_spec_value(source[proposed], proposed, counts, controls,
                                                   length_bias, think_bias, unused);
                if (isfinite(value)) probability = divide(expf(subtract(value, maximum)), normalizer);
            }
            if (!isfinite(probability) || probability < 0 || probability > 1) atomicExch(bad, 1);
            prediction[row] = selected;
            proposal_prob[row] = probability;
            if constexpr (Entropy) {
                const float result = add(logf(normalizer), divide(gap_sums[0], normalizer));
                if (!isfinite(result) || result < 0) atomicExch(bad, 1);
                entropy[row] = result;
            }
        }
    }
}

// `candidates[row]` is the proposal scored by this row, not the anchor input.
// -1 is an unused final-row proposal: p=0, while argmax/H remain available for
// selecting a boundary bonus. A nullptr candidate array requests preview
// argmax only and does not touch probability/entropy outputs. For same-position
// preview pass response_length=committed response length-1; for causal verify
// pass its full length. The caller resets/uploads the count snapshot itself.
inline cudaError_t launch_self_spec_stats(
    const float* logits, const int64_t* counts, int64_t response_length,
    const int32_t* candidates, int rows, int vocab, const Controls& controls,
    int32_t* prediction, float* proposal_prob, float* entropy, int32_t* bad,
    cudaStream_t stream, int threads = 512) {
    if (!logits || !counts || !prediction || !bad || (candidates && !proposal_prob) ||
        rows < 1 || rows > 32 || vocab < 1 || response_length < 0 ||
        controls.mask < 0 || controls.mask >= vocab || controls.think_end < 0 || controls.think_end >= vocab ||
        controls.eos0 < 0 || controls.eos0 >= vocab || controls.eos1 < 0 || controls.eos1 >= vocab ||
        controls.length_start < 0 || controls.think_start < 0 || controls.length_window < 1 || controls.think_window < 1 ||
        response_length > std::numeric_limits<int64_t>::max() - rows ||
        !std::isfinite(controls.frequency) || !std::isfinite(controls.presence) ||
        !std::isfinite(controls.repetition) || controls.repetition <= 0 ||
        !std::isfinite(controls.length_strength) || controls.length_strength < 0 ||
        !std::isfinite(controls.think_strength) || controls.think_strength < 0 ||
        (threads != 256 && threads != 512 && threads != 1024))
        throw std::invalid_argument("Invalid Clover GPU self-spec controls or shape");
    const auto status = cudaMemsetAsync(bad, 0, sizeof(int32_t), stream);
    if (status != cudaSuccess) return status;
#define CLOVER_SELF_SPEC(THREADS) \
    if (!candidates) self_spec_stats_kernel<THREADS, false, false><<<rows, THREADS, 0, stream>>>(logits, counts, response_length, candidates, rows, vocab, controls, prediction, proposal_prob, entropy, bad); \
    else if (!entropy) self_spec_stats_kernel<THREADS, true, false><<<rows, THREADS, 0, stream>>>(logits, counts, response_length, candidates, rows, vocab, controls, prediction, proposal_prob, entropy, bad); \
    else self_spec_stats_kernel<THREADS, true, true><<<rows, THREADS, 0, stream>>>(logits, counts, response_length, candidates, rows, vocab, controls, prediction, proposal_prob, entropy, bad)
    if (threads == 256) { CLOVER_SELF_SPEC(256); }
    else if (threads == 1024) { CLOVER_SELF_SPEC(1024); }
    else { CLOVER_SELF_SPEC(512); }
#undef CLOVER_SELF_SPEC
    return cudaGetLastError();
}

}  // namespace strata::qwen35::clover_diffusion_gpu

namespace strata::qwen35::clover_diffusion_gpu {

// Exact verification applies the target's full causal-row response penalties:
// committed counts (including the emitted anchor), plus every preceding draft
// in this row's hypothetical prefix. Unlike threshold acceptance, this is not
// one fixed count snapshot across rows. The caller visits only a matching
// prefix and keeps terminal tokens as boundaries; unused later rows are inert.
__device__ __forceinline__ float self_spec_exact_value(
    float value, int id, int64_t count, const Controls& controls,
    float length_bias, float think_bias, bool& invalid) {
    // The causal CPU target does not exclude <|mask|>. Preserve that rule:
    // masking the target here would change exact verification at a mask argmax.
    if (controls.frequency != 0) value = subtract(value, multiply((float) count, controls.frequency));
    if (controls.presence != 0 && count != 0) value = subtract(value, controls.presence);
    if (controls.repetition != 1 && count != 0)
        value = value < 0 ? multiply(value, controls.repetition) : divide(value, controls.repetition);
    // The disjunction applies a duplicated stop ID's bias only once, matching
    // clover_decode::apply_penalties' deduplicated stop list.
    if (id == controls.eos0 || id == controls.eos1) value = add(value, length_bias);
    if (id == controls.think_end) value = add(value, think_bias);
    if (isnan(value) || (isinf(value) && value > 0)) {
        invalid = true;
        return -INFINITY;
    }
    return value;
}

template <int Threads>
__global__ void __launch_bounds__(Threads) self_spec_exact_kernel(
    const float* logits, const int64_t* counts, int64_t response_length,
    const int32_t* candidates, int rows, int vocab, Controls controls,
    int32_t* prediction, int32_t* bad) {
    const int row = (int) blockIdx.x, thread = (int) threadIdx.x;
    if (row >= rows) return;
    __shared__ float values[Threads];
    __shared__ int indices[Threads];
    __shared__ int32_t prefix[32];
    __shared__ float length_bias, think_bias;
    if (thread == 0) {
        bool visible_close = counts[controls.think_end] != 0;
        // Validate the complete proposals+bonus contract, not just the prefix
        // this block reads. The final -1 is a sentinel and is never counted.
        for (int i = 0; i < rows - 1; ++i) {
            const int32_t token = candidates[i];
            if (token < 0 || token >= vocab) atomicExch(bad, 1);
            if (i < row) {
                prefix[i] = token;
                visible_close |= token == controls.think_end;
            }
        }
        if (candidates[rows - 1] != -1) atomicExch(bad, 1);
        const int64_t position = response_length + row + 1;
        length_bias = controls.length_strength == 0 ? 0.f
            : ramp(position, controls.length_start, controls.length_window, controls.length_strength);
        // This is the causal rule: each eligible row is biased until </think>
        // occurs in its own prefix, not threshold mode's first-eligible-row rule.
        think_bias = controls.think_strength == 0 || visible_close ? 0.f
            : ramp(position, controls.think_start, controls.think_window, controls.think_strength);
    }
    __syncthreads();
    const float* source = logits + (size_t) row * vocab;
    float best = -INFINITY;
    int best_id = 0x7fffffff;
    bool invalid = false;
    for (int id = thread; id < vocab; id += Threads) {
        const int64_t base = counts[id];
        int increment = 0;
        for (int i = 0; i < row; ++i) increment += prefix[i] == id;
        int64_t count = base;
        if (base < 0 || base > 0x7fffffffffffffffLL - increment) invalid = true;
        else count += increment;
        const float value = self_spec_exact_value(source[id], id, count, controls,
                                                  length_bias, think_bias, invalid);
        if (value > best || (value == best && id < best_id)) { best = value; best_id = id; }
    }
    if (invalid) atomicExch(bad, 1);
    values[thread] = best;
    indices[thread] = best_id;
    __syncthreads();
    for (int width = Threads / 2; width; width >>= 1) {
        if (thread < width) {
            const float other = values[thread + width];
            const int other_id = indices[thread + width];
            if (other > values[thread] || (other == values[thread] && other_id < indices[thread])) {
                values[thread] = other;
                indices[thread] = other_id;
            }
        }
        __syncthreads();
    }
    if (thread == 0) {
        if (!isfinite(values[0])) { atomicExch(bad, 1); prediction[row] = -1; }
        else prediction[row] = indices[0];
    }
}

// candidates[0..rows-2] are valid vocabulary IDs, including a possible mask
// from the legacy raw preview; candidates[rows-1] must be -1 for the boundary
// bonus. Raw logits and base counts are read-only. Only the committed
// generated response contributes to response_length/counts; the prompt does not.
inline cudaError_t launch_self_spec_exact(
    const float* logits, const int64_t* counts, int64_t committed_response_length,
    const int32_t* candidates, int rows, int vocab, const Controls& controls,
    int32_t* prediction, int32_t* bad, cudaStream_t stream, int threads = 512) {
    if (!logits || !counts || !candidates || !prediction || !bad ||
        rows < 1 || rows > 32 || vocab < 1 || committed_response_length < 0 ||
        controls.mask < 0 || controls.mask >= vocab || controls.think_end < 0 || controls.think_end >= vocab ||
        controls.eos0 < 0 || controls.eos0 >= vocab || controls.eos1 < 0 || controls.eos1 >= vocab ||
        controls.length_start < 0 || controls.think_start < 0 || controls.length_window < 1 || controls.think_window < 1 ||
        committed_response_length > std::numeric_limits<int64_t>::max() - rows ||
        !std::isfinite(controls.frequency) || !std::isfinite(controls.presence) ||
        !std::isfinite(controls.repetition) || controls.repetition <= 0 ||
        !std::isfinite(controls.length_strength) || controls.length_strength < 0 ||
        !std::isfinite(controls.think_strength) || controls.think_strength < 0 ||
        (threads != 256 && threads != 512 && threads != 1024))
        throw std::invalid_argument("Invalid Clover GPU exact controls or shape");
    const auto status = cudaMemsetAsync(bad, 0, sizeof(int32_t), stream);
    if (status != cudaSuccess) return status;
    if (threads == 256) self_spec_exact_kernel<256><<<rows, 256, 0, stream>>>(logits, counts,
        committed_response_length, candidates, rows, vocab, controls, prediction, bad);
    else if (threads == 1024) self_spec_exact_kernel<1024><<<rows, 1024, 0, stream>>>(logits, counts,
        committed_response_length, candidates, rows, vocab, controls, prediction, bad);
    else self_spec_exact_kernel<512><<<rows, 512, 0, stream>>>(logits, counts,
        committed_response_length, candidates, rows, vocab, controls, prediction, bad);
    return cudaGetLastError();
}

} // namespace strata::qwen35::clover_diffusion_gpu
