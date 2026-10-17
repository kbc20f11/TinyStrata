#pragma once
#include "strata/artifact/model_family.hpp"
#include <cstdint>
namespace strata::core {
// Qwen3.6 defaults; required GGUF metadata supplies the runtime dimensions.
struct ModelGeometry {
    int64_t n_embd = 2048, n_layers = 40, full_attention_interval = 4, n_vocab = 248320;
    double rms_eps = 1e-6;
    int64_t ssm_state_size = 128, ssm_k_heads = 16, ssm_v_heads = 32, ssm_d_conv = 4;
    int64_t ssm_conv_channels = 8192, ssm_value_dim = 4096;
    int64_t n_head = 16, n_head_kv = 2, head_dim = 256, n_rot = 64;
    int64_t n_expert = 256, n_expert_used = 8, n_ff = 512, n_ff_shexp = 512;
    ModelFamily family = ModelFamily::Qwen35Moe;
    int64_t n_mtp_layers = 0;
    constexpr int64_t n_attention_layers() const { return n_layers / full_attention_interval; }
    constexpr int64_t n_gdn_layers() const { return n_layers - n_attention_layers(); }
};
inline constexpr ModelGeometry kCompiledGeometry{};
inline bool is_attention_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.full_attention_interval == g.full_attention_interval - 1;
}
}  // namespace strata::core
