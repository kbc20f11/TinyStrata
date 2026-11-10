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
                               int blocks, double uniform, FullVocabResult* result,
                               const uint32_t* accepted = nullptr) {
    if (accepted && *accepted) return;
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

__global__ void copy_probabilities_kernel(const double* probabilities, int vocab,
                                           const FullVocabResult* result, double* output) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(vocab))
        output[i] = result->status == kSampleOk && result->total > 0.0 && isfinite(result->total)
            ? probabilities[i] / result->total : NAN;
}

// maximum scratch is dead after probability_kernel; use it for q's segment
// sums while preserving all target masses/sums until the decision is made.
__global__ void validate_draft_kernel(const double* q, int vocab, double* q_sums, uint32_t* invalid) {
    __shared__ double segments[kThreads];
    __shared__ uint32_t errors[kThreads];
    const size_t first = size_t(blockIdx.x) * kBlock + size_t(threadIdx.x) * kItems;
    double sum = 0.0;
    uint32_t bad = 0;
    for (int j = 0; j < kItems; ++j) {
        const size_t i = first + j;
        if (i < size_t(vocab)) {
            const double value = q[i];
            if (!(value >= 0.0 && value <= 1.0) || !isfinite(value)) bad = 2;
            else sum += value;
        }
    }
    segments[threadIdx.x] = sum;
    errors[threadIdx.x] = bad;
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        uint32_t error = invalid[blockIdx.x];
        for (int t = 0; t < kThreads; ++t) { total += segments[t]; error |= errors[t]; }
        q_sums[blockIdx.x] = total;
        invalid[blockIdx.x] = error;
    }
}

__global__ void verify_decision_kernel(const double* probabilities, const double* p_sums,
                                        const double* q, const double* q_sums,
                                        const uint32_t* invalid, int blocks, int token,
                                        double uniform, FullVocabResult* result, uint32_t* accepted) {
    if (threadIdx.x != 0) return;
    double p_total = 0.0, q_total = 0.0;
    uint32_t bad = 0;
    for (int b = 0; b < blocks; ++b) { p_total += p_sums[b]; q_total += q_sums[b]; bad |= invalid[b]; }
    // An error is not an accepted token. One is used solely as the stop flag
    // shared with the residual/select kernels; status always takes precedence.
    *accepted = 1;
    if (bad & 1) { *result = {-1, kSampleNonFinite, 0.0, 0.0}; return; }
    if (!(p_total > 0.0) || !isfinite(p_total)) {
        *result = {-1, kSampleInvalidTotal, p_total, 0.0}; return;
    }
    if ((bad & 2) || !isfinite(q_total) || fabs(q_total - 1.0) > 1e-10 ||
        !(q[token] > 0.0) || !isfinite(q[token])) {
        *result = {-1, kSampleInvalidDraft, q_total, 0.0}; return;
    }
    const double p = probabilities[token] / p_total;
    const double ratio = fmin(1.0, p / q[token]);
    *result = {token, kSampleOk, p_total, p};
    *accepted = uint32_t(uniform < ratio);
}

__global__ void residual_kernel(double* probabilities, int vocab, const double* q,
                                 double* sums, const FullVocabResult* result, const uint32_t* accepted) {
    if (*accepted) return;
    __shared__ double segments[kThreads];
    const size_t first = size_t(blockIdx.x) * kBlock + size_t(threadIdx.x) * kItems;
    double sum = 0.0;
    const double total = result->total;
    for (int j = 0; j < kItems; ++j) {
        const size_t i = first + j;
        if (i < size_t(vocab)) {
            const double residual = fmax(probabilities[i] / total - q[i], 0.0);
            probabilities[i] = residual;
            sum += residual;
        }
    }
    segments[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int t = 0; t < kThreads; ++t) sum += segments[t];
        sums[blockIdx.x] = sum;
    }
}

bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
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

void copy_full_vocab_probabilities(const void* workspace, size_t workspace_bytes,
                                   const FullVocabResult* device_result, int vocab,
                                   double* device_out, void* stream) {
    const Layout l = layout(vocab);
    if (!workspace || !device_result || !device_out || workspace_bytes < l.bytes ||
        reinterpret_cast<uintptr_t>(workspace) % alignof(double) ||
        reinterpret_cast<uintptr_t>(device_out) % alignof(double) ||
        overlaps(workspace, l.bytes, device_out, size_t(vocab) * sizeof(double)) ||
        overlaps(device_result, sizeof(FullVocabResult), device_out, size_t(vocab) * sizeof(double)))
        throw std::invalid_argument("invalid full-vocabulary probability copy or workspace");
    copy_probabilities_kernel<<<(size_t(vocab) + kThreads - 1) / kThreads, kThreads, 0,
                                static_cast<cudaStream_t>(stream)>>>(
        static_cast<const double*>(workspace), vocab, device_result, device_out);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        throw std::runtime_error(std::string("full-vocabulary probability copy: ") + cudaGetErrorString(error));
}

void verify_full_vocab(const float* target_logits, const double* q, int draft_token, int vocab,
                       double temperature, double uniform_accept, double uniform_residual,
                       void* workspace, size_t workspace_bytes, FullVocabResult* device_result,
                       uint32_t* accepted, void* stream) {
    const Layout l = layout(vocab);
    if (!target_logits || !q || !workspace || !device_result || !accepted || workspace_bytes < l.bytes ||
        draft_token < 0 || draft_token >= vocab || reinterpret_cast<uintptr_t>(workspace) % alignof(double) ||
        reinterpret_cast<uintptr_t>(q) % alignof(double) || reinterpret_cast<uintptr_t>(accepted) % alignof(uint32_t) ||
        !(temperature > 0.0) || !std::isfinite(temperature) ||
        !(uniform_accept >= 0.0 && uniform_accept < 1.0) || !std::isfinite(uniform_accept) ||
        !(uniform_residual >= 0.0 && uniform_residual < 1.0) || !std::isfinite(uniform_residual) ||
        overlaps(workspace, l.bytes, q, size_t(vocab) * sizeof(double)) ||
        overlaps(workspace, l.bytes, device_result, sizeof(FullVocabResult)) ||
        overlaps(workspace, l.bytes, accepted, sizeof(uint32_t)) ||
        overlaps(device_result, sizeof(FullVocabResult), accepted, sizeof(uint32_t)) ||
        overlaps(q, size_t(vocab) * sizeof(double), device_result, sizeof(FullVocabResult)) ||
        overlaps(q, size_t(vocab) * sizeof(double), accepted, sizeof(uint32_t)))
        throw std::invalid_argument("invalid full-vocabulary speculative verification input or workspace");
    auto* base = static_cast<uint8_t*>(workspace);
    auto* probabilities = reinterpret_cast<double*>(base);
    auto* maxima = reinterpret_cast<double*>(base + l.maxima);
    auto* sums = reinterpret_cast<double*>(base + l.sums);
    auto* invalid = reinterpret_cast<uint32_t*>(base + l.invalid);
    const auto s = static_cast<cudaStream_t>(stream);
    maximum_kernel<<<l.blocks, kThreads, 0, s>>>(target_logits, vocab, maxima, invalid);
    probability_kernel<<<l.blocks, kThreads, 0, s>>>(target_logits, vocab, temperature, maxima, invalid,
                                                 l.blocks, probabilities, sums);
    validate_draft_kernel<<<l.blocks, kThreads, 0, s>>>(q, vocab, maxima, invalid);
    verify_decision_kernel<<<1, 1, 0, s>>>(probabilities, sums, q, maxima, invalid, l.blocks,
                                          draft_token, uniform_accept, device_result, accepted);
    residual_kernel<<<l.blocks, kThreads, 0, s>>>(probabilities, vocab, q, sums, device_result, accepted);
    select_kernel<<<1, kThreads, 0, s>>>(probabilities, vocab, sums, invalid, l.blocks, uniform_residual,
                                        device_result, accepted);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        throw std::runtime_error(std::string("full-vocabulary speculative verification launch: ") + cudaGetErrorString(error));
}

} // namespace strata::gemma4
