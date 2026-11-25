// FP32 image-encoder operations for Inkling 2 Mini's native Vision shard.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

namespace ink2::cuda_vision_ops {

constexpr int kRowThreads = 256;

inline __device__ std::size_t element_index() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

template<bool Maximum>
inline __device__ float combine(float left, float right) {
    if constexpr (Maximum) return fmaxf(left, right);
    return left + right;
}

template<bool Maximum>
inline __device__ float warp_reduce(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = combine<Maximum>(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// One result per 256-thread row. The final barrier protects the result before
// a second call reuses the scratch for variance or the softmax denominator.
template<bool Maximum>
inline __device__ float row_reduce(float value, float* partial) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce<Maximum>(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (warp == 0) {
        const float identity = Maximum ? -CUDART_INF_F : 0.0f;
        value = warp_reduce<Maximum>(lane < kRowThreads / 32 ? partial[lane] : identity);
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}

__global__ void unpack_f16(const uint16_t* source, float* output, std::size_t count) {
    const std::size_t i = element_index();
    if (i < count) output[i] = __half2float(__ushort_as_half(source[i]));
}

__global__ void unpack_bf16(const uint16_t* source, float* output, std::size_t count) {
    const std::size_t i = element_index();
    if (i < count) output[i] = __uint_as_float(static_cast<uint32_t>(source[i]) << 16);
}

__global__ void add_bias(float* values, const float* bias, std::size_t count, int width) {
    const std::size_t i = element_index();
    if (i < count) values[i] += bias[i % width];
}

__global__ void add_position(float* values, const float* position, std::size_t count) {
    const std::size_t i = element_index();
    if (i < count) values[i] += position[i];
}

// Input and output may alias. All input reads for the statistics precede the writes.
__global__ void __launch_bounds__(kRowThreads) layer_norm(const float* input, const float* gain,
                                                         const float* bias, float* output,
                                                         int width, float eps) {
    __shared__ float partial[kRowThreads / 32];
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * width;
    const float* row = input + offset;
    float* result = output + offset;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += kRowThreads) sum += row[col];
    const float mean = row_reduce<false>(sum, partial) / static_cast<float>(width);
    float squares = 0.0f;
    for (int col = threadIdx.x; col < width; col += kRowThreads) {
        const float delta = row[col] - mean;
        squares = __fadd_rn(squares, __fmul_rn(delta, delta));
    }
    const float variance = row_reduce<false>(squares, partial) / static_cast<float>(width);
    const float inv = 1.0f / sqrtf(variance + eps);
    for (int col = threadIdx.x; col < width; col += kRowThreads) {
        const float normalized = (row[col] - mean) * inv;
        result[col] = __fadd_rn(__fmul_rn(normalized, gain[col]), bias ? bias[col] : 0.0f);
    }
}

__global__ void rope_2d(float* queries, float* keys, const float* cosine, const float* sine,
                        int heads, int head_dim) {
    const std::size_t token = blockIdx.y;
    const std::size_t offset = (token * heads + blockIdx.x) * head_dim;
    const float* rc = cosine + token * head_dim;
    const float* rs = sine + token * head_dim;
    float* q = queries + offset;
    float* k = keys + offset;
    for (int channel = 2 * threadIdx.x; channel + 1 < head_dim; channel += 2 * blockDim.x) {
        const float q0 = q[channel];
        const float q1 = q[channel + 1];
        const float k0 = k[channel];
        const float k1 = k[channel + 1];
        q[channel] = __fsub_rn(__fmul_rn(q0, rc[channel]), __fmul_rn(q1, rs[channel]));
        q[channel + 1] = __fadd_rn(__fmul_rn(q1, rc[channel + 1]), __fmul_rn(q0, rs[channel + 1]));
        k[channel] = __fsub_rn(__fmul_rn(k0, rc[channel]), __fmul_rn(k1, rs[channel]));
        k[channel + 1] = __fadd_rn(__fmul_rn(k1, rc[channel + 1]), __fmul_rn(k0, rs[channel + 1]));
    }
}

// Scores are already scaled. Each query's `cols` keys are contiguous, with
// head_stride measured in float elements and arbitrary padding between heads.
__global__ void __launch_bounds__(kRowThreads) softmax(float* scores, int cols, std::size_t head_stride,
                                                       int tile_rows) {
    if (blockIdx.x >= static_cast<unsigned>(tile_rows)) return;
    __shared__ float partial[kRowThreads / 32];
    float* row = scores + static_cast<std::size_t>(blockIdx.y) * head_stride
                       + static_cast<std::size_t>(blockIdx.x) * cols;
    float max_score = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += kRowThreads) max_score = fmaxf(max_score, row[col]);
    max_score = row_reduce<true>(max_score, partial);
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += kRowThreads) {
        const float probability = expf(row[col] - max_score);
        row[col] = probability;
        sum += probability;
    }
    const float inv = 1.0f / row_reduce<false>(sum, partial);
    for (int col = threadIdx.x; col < cols; col += kRowThreads) row[col] *= inv;
}

__global__ void gelu(float* values, std::size_t count) {
    const std::size_t i = element_index();
    if (i < count) {
        const float x = values[i];
        values[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
}

__global__ void scaled_residual(float* values, const float* delta, const float* scale,
                                std::size_t count, int width) {
    const std::size_t i = element_index();
    if (i < count) values[i] = __fadd_rn(values[i], __fmul_rn(scale[i % width], delta[i]));
}

}  // namespace ink2::cuda_vision_ops
