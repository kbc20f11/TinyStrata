// Shared E4M3 Tensor Core attention for token-major KV caches.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::fp8_attention {
struct Cache {
    const std::uint8_t *k, *v;
    const std::uint16_t *ks, *vs;
};

// Call before graph capture. Checks both the GPU and the selected compiled image
// and configures dynamic shared memory. Supports head dimensions 64, 256 and 512.
bool available(int head_dim = 0);

// Gemma 4's 256/512-dimensional heads share QK/softmax across cooperating
// output warps and use three E4M3 components for Q/P. Prepare before capture.
bool available_gemma();

// Codes are E4M3, with one FP16 multiplier per 32 contiguous elements.
// src is [rows,width]; cache is [position,width]. A device position, when
// supplied, overrides pos; its first int64 element is the first query position.
// ring == 0 is a non-circular cache, otherwise positions wrap modulo ring.
void store(const float *src, std::uint8_t *codes, std::uint16_t *scales, int width,
           std::int64_t pos, int rows, const std::int64_t *device_pos, int ring, void *stream);

// Optional reusable storage for split-K decoding/verification (rows <= 8).
// Larger batches use the dense tiled path without this storage.
std::size_t scratch_bytes(int rows, int q_heads, int head_dim);

// Q/output: [rows,q_heads,head_dim]. KV: [capacity,kv_heads,head_dim].
// GQA ratios 1,2,4,8,16 are supported. All visible positions are evaluated.
// window == 0 is full attention. whole allows all queries in this call to see
// each other (Gemma image blocks), while retaining each query's window start.
// scale is the model's QK multiplier (not inferred from head_dim).
void run(const float *q, Cache cache, std::int64_t pos, int rows, int q_heads, int kv_heads,
         int head_dim, int context, float scale, float *output,
         const std::int64_t *device_pos, int ring, int window, bool whole,
         void *scratch, std::size_t scratch_size, void *stream);

// Same layout/masking contract and scratch allocation as run(). This separate
// experimental entry point leaves the shared LFM path unchanged.
void run_gemma(const float *q, Cache cache, std::int64_t pos, int rows, int q_heads, int kv_heads,
               int head_dim, int context, float scale, float *output,
               const std::int64_t *device_pos, int ring, int window, bool whole,
               void *scratch, std::size_t scratch_size, void *stream);
} // namespace strata::fp8_attention
