// Compensated CUDA decode for Tiny's head-major INT8 KV.
#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::qwen35::int8_fdec {
struct Cache {
    const std::int8_t *key = nullptr, *value = nullptr;
    const std::uint16_t *key_scale = nullptr, *value_scale = nullptr;
};
// These are float counts, not bytes. Zero denotes unsupported/overflowing shape.
std::size_t part_floats(int context, int kv_heads);
std::size_t meta_floats(int context, int kv_heads);
// Call once before graph capture; checks compiled code and active SM80+ device.
bool available();
// n=1..4, HD256/GQA8, logical context>=4096. Device position must be
// in [0, context-n]. Eager grids use host_pos+n; captured grids cover context
// and mask before dereferencing future/unmapped rows. No allocations on run.
// Ineligible calls launch nothing. Launch errors throw; no partial fallback.
bool run(const float* query, Cache cache, const std::int32_t* position, std::int64_t host_pos,
         int n, int heads, int kv_heads, int context, float scale,
         float* part, std::size_t part_capacity, float* meta, std::size_t meta_capacity,
         const float* gate, float* output, void* stream, bool graph_grid = false);
} // namespace strata::qwen35::int8_fdec
