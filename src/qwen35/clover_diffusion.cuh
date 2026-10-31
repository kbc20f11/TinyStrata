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

}  // namespace strata::qwen35::clover_diffusion_gpu
