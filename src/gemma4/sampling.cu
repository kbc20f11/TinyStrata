#include "strata/gemma4/sampling.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::gemma4 {
namespace {
constexpr int kThreads = 256;
constexpr int kItems = 8;
constexpr int kBlock = kThreads * kItems;

struct Layout {
    int blocks;
    size_t maxima, sums, invalid, bytes;
};
Layout layout(int vocab) {
    if (vocab <= 0) throw std::invalid_argument("full-vocabulary sampling needs a positive vocabulary");
    Layout l{};
    l.blocks = int((size_t(vocab) + kBlock - 1) / kBlock);
    l.maxima = size_t(vocab) * sizeof(double);
    l.sums = l.maxima + size_t(l.blocks) * sizeof(double);
    l.invalid = l.sums + size_t(l.blocks) * sizeof(double);
    l.bytes = (l.invalid + size_t(l.blocks) * sizeof(uint32_t) + 255) & ~size_t(255);
    return l;
}

__global__ void maximum_kernel(const float* __restrict__ logits, int vocab,
                                double* __restrict__ maxima, uint32_t* __restrict__ invalid) {
    __shared__ double top[kThreads];
    __shared__ uint32_t bad[kThreads];
    double maximum = -INFINITY;
    uint32_t error = 0;
    const size_t first = size_t(blockIdx.x) * kBlock;
    for (int j = 0; j < kItems; ++j) {
        const size_t i = first + threadIdx.x + j * kThreads;
        if (i < size_t(vocab)) {
            const double x = double(logits[i]);
            if (!isfinite(x)) error = 1;
            else maximum = fmax(maximum, x);
        }
    }
    top[threadIdx.x] = maximum;
    bad[threadIdx.x] = error;
    __syncthreads();
    for (int d = kThreads / 2; d; d >>= 1) {
        if (threadIdx.x < d) {
            top[threadIdx.x] = fmax(top[threadIdx.x], top[threadIdx.x + d]);
            bad[threadIdx.x] |= bad[threadIdx.x + d];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maxima[blockIdx.x] = top[0];
        invalid[blockIdx.x] = bad[0];
    }
}

__global__ void probability_kernel(const float* __restrict__ logits, int vocab, double temperature,
                                    const double* __restrict__ maxima, const uint32_t* __restrict__ invalid,
                                    int blocks, double* __restrict__ probabilities, double* __restrict__ sums) {
    __shared__ double maximum, segments[kThreads];
    __shared__ uint32_t bad;
    if (threadIdx.x == 0) {
        maximum = -INFINITY;
        bad = 0;
        for (int b = 0; b < blocks; ++b) {
            maximum = fmax(maximum, maxima[b]);
            bad |= invalid[b];
        }
    }
    __syncthreads();
    if (bad) {
        if (threadIdx.x == 0) sums[blockIdx.x] = 0.0;
        return;
    }
    // Each thread owns eight consecutive token IDs. Preserve that order both here and in the final CDF.
    const size_t first = size_t(blockIdx.x) * kBlock + size_t(threadIdx.x) * kItems;
    double sum = 0.0;
    for (int j = 0; j < kItems; ++j) {
        const size_t i = first + j;
        if (i < size_t(vocab)) {
            const double probability = exp((double(logits[i]) - maximum) / temperature);
            probabilities[i] = probability;
            sum += probability;
        }
    }
    segments[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        for (int t = 0; t < kThreads; ++t) total += segments[t];
        sums[blockIdx.x] = total;
    }
}

__global__ void select_kernel(const double* __restrict__ probabilities, int vocab,
                               const double* __restrict__ sums, const uint32_t* __restrict__ invalid,
                               int blocks, double uniform, FullVocabResult* result) {
    __shared__ double segments[kThreads], target, total;
    __shared__ int selected_block, selected_segment;
    if (threadIdx.x == 0) {
        uint32_t bad = 0;
        total = 0.0;
        for (int b = 0; b < blocks; ++b) { bad |= invalid[b]; total += sums[b]; }
        selected_block = -1;
        if (bad) *result = {-1, kSampleNonFinite, 0.0, 0.0};
        else if (!(total > 0.0) || !isfinite(total)) *result = {-1, kSampleInvalidTotal, total, 0.0};
        else {
            // Multiplication may round a draw immediately below 1 to total. Stay below the upper endpoint.
            const double threshold = fmin(uniform * total, nextafter(total, 0.0));
            double before = 0.0;
            for (int b = 0; b < blocks; ++b) {
                const double after = before + sums[b];
                if (sums[b] > 0.0 && threshold < after) {
                    selected_block = b;
                    target = fmin(fmax(threshold - before, 0.0), nextafter(sums[b], 0.0));
                    break;
                }
                before = after;
            }
            if (selected_block < 0) *result = {-1, kSampleInvalidTotal, total, 0.0};
        }
    }
    __syncthreads();
    if (selected_block < 0) return;

    const size_t first = size_t(selected_block) * kBlock + size_t(threadIdx.x) * kItems;
    double local = 0.0;
    for (int j = 0; j < kItems && first + j < size_t(vocab); ++j) local += probabilities[first + j];
    segments[threadIdx.x] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        selected_segment = -1;
        double before = 0.0;
        for (int t = 0; t < kThreads; ++t) {
            const double after = before + segments[t];
            if (segments[t] > 0.0 && target < after) {
                selected_segment = t;
                target = fmin(fmax(target - before, 0.0), nextafter(segments[t], 0.0));
                break;
            }
            before = after;
        }
        if (selected_segment < 0) *result = {-1, kSampleInvalidTotal, total, 0.0};
    }
    __syncthreads();
    if (int(threadIdx.x) != selected_segment) return;
    double before = 0.0;
    int last_positive = -1;
    for (int j = 0; j < kItems && first + j < size_t(vocab); ++j) {
        const double p = probabilities[first + j];
        const double after = before + p;
        if (p > 0.0) {
            last_positive = int(first + j);
            if (target < after) {
                *result = {int32_t(first + j), kSampleOk, total, p / total};
                return;
            }
        }
        before = after;
    }
    // Only endpoint rounding can reach this after the segment's positive-mass check.
    if (last_positive >= 0)
        *result = {last_positive, kSampleOk, total, probabilities[last_positive] / total};
    else *result = {-1, kSampleInvalidTotal, total, 0.0};
}
} // namespace

size_t full_vocab_sample_workspace_bytes(int vocab) { return layout(vocab).bytes; }

void sample_full_vocab(const float* logits, int vocab, double temperature, double uniform,
                       void* workspace, size_t workspace_bytes, FullVocabResult* device_result, void* stream) {
    const Layout l = layout(vocab);
    if (!logits || !workspace || !device_result || workspace_bytes < l.bytes ||
        reinterpret_cast<uintptr_t>(workspace) % alignof(double) ||
        !(temperature > 0.0) || !std::isfinite(temperature) ||
        !(uniform >= 0.0 && uniform < 1.0) || !std::isfinite(uniform))
        throw std::invalid_argument("invalid full-vocabulary sampling input or workspace");
    auto* base = static_cast<uint8_t*>(workspace);
    auto* probabilities = reinterpret_cast<double*>(base);
    auto* maxima = reinterpret_cast<double*>(base + l.maxima);
    auto* sums = reinterpret_cast<double*>(base + l.sums);
    auto* invalid = reinterpret_cast<uint32_t*>(base + l.invalid);
    const auto s = static_cast<cudaStream_t>(stream);
    maximum_kernel<<<l.blocks, kThreads, 0, s>>>(logits, vocab, maxima, invalid);
    probability_kernel<<<l.blocks, kThreads, 0, s>>>(logits, vocab, temperature, maxima, invalid, l.blocks, probabilities, sums);
    select_kernel<<<1, kThreads, 0, s>>>(probabilities, vocab, sums, invalid, l.blocks, uniform, device_result);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        throw std::runtime_error(std::string("full-vocabulary sampling launch: ") + cudaGetErrorString(error));
}

} // namespace strata::gemma4
