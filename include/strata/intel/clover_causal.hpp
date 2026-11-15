#pragma once
#include "strata/qwen35/clover_decode.hpp"
#include <cmath>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strata::intel::clover_causal {
// Clover exports A_log. Qwen's exported ssm_a already contains -exp(A_log).
// Convert the small F32 scalar tensor in the one weight arena, never the GGUF.
inline float causal_a(float a_log) {
    if (!std::isfinite(a_log)) throw std::runtime_error("Non-finite Clover A_log");
    const float value = -std::exp(a_log);
    if (!std::isfinite(value)) throw std::runtime_error("Clover A_log overflows FP32");
    return value;
}
inline void convert_a_log(std::span<float> values) {
    // Check the whole tensor before modifying it.
    for (float value : values) (void)causal_a(value);
    for (float& value : values) value = causal_a(value);
}
inline bool terminal(int token) {
    return token == strata::qwen35::clover_decode::kEndOfText ||
           token == strata::qwen35::clover_decode::kEndOfTurn;
}
inline int tensor_type(std::string_view name) {
    if (name == "output.weight" || name == "token_embd.weight") return 8;
    if (name == "output_norm.weight") return 0;
    if (!name.starts_with("blk.")) return -1;
    const auto dot = name.find('.', 4);
    if (dot == std::string_view::npos) return -1;
    const auto suffix = name.substr(dot + 1);
    if (suffix == "ffn_gate_exps.weight" || suffix == "ffn_up_exps.weight") return 18;
    if (suffix == "ffn_down_exps.weight") return 21;
    for (auto scalar : {"attn_norm.weight", "post_attention_norm.weight", "ffn_gate_inp.weight",
                       "ffn_gate_inp_shexp.weight", "ssm_a", "ssm_conv1d.weight", "ssm_conv1d.weight_right",
                       "ssm_dt.bias", "ssm_norm.weight", "ssm_bd_logit_scale.weight", "ssm_bd_history.bias",
                       "ssm_bd_history.weight", "attn_q_norm.weight", "attn_k_norm.weight"})
        if (suffix == scalar) return 0;
    for (auto matrix : {"attn_gate.weight", "attn_qkv.weight", "ssm_alpha.weight", "ssm_beta.weight",
                       "ssm_out.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
        if (suffix == matrix) return 8;
    return -1;
}
inline void validate_launch(bool int8_kv, int mtp, int draft_vocabulary,
                            bool stochastic_mtp, bool npu,
                            std::string_view mode = "causal", int drafts = 0) {
    if (!int8_kv) throw std::runtime_error("Experimental Intel Clover requires INT8 KV");
    if (mtp || draft_vocabulary || stochastic_mtp || mode != "causal" || drafts)
        throw std::runtime_error("Experimental Intel Clover supports causal D0 only; Qwen MTP is not Clover drafting");
    if (npu) throw std::runtime_error("Intel Clover NPU assistance is unvalidated");
}
inline std::vector<int> request_history(bool clover, const std::vector<int>& prompt) {
    // Clover's penalties and stopping ramps count this response only.
    return clover ? std::vector<int>{} : prompt;
}
} // namespace strata::intel::clover_causal
