#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::gemma4 {
// One to eight query rows, head_dim=64, GQA=8, full attention. Each 512-key split stores 64 unnormalized values/max/den.
size_t sink_attention_workspace_floats(int heads, int max_keys, int rows = 1);

// Queries stay FP32 and the existing INT8/FP16-scale KV layout is read unchanged. No allocation occurs.
// With at, room is the fixed graph capacity and at[2*r] is query row r's actual position.
// Otherwise row r attends through pos+r. Query/output rows and scratch are independent; future keys are masked.
// Unsupported shapes, invalid host arguments or insufficient scratch return false without a launch.
bool sink_attention_q8(const float* q, const int8_t* kcodes, const uint16_t* kscales,
                        const int8_t* vcodes, const uint16_t* vscales, const float* sinks,
                        float* out, float* part, size_t part_floats, int heads, int kv_heads,
                        int64_t pos, const long long* at, int room, float scale, void* stream, int rows = 1);
} // namespace strata::gemma4
