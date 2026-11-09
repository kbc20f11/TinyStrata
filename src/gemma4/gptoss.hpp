// src/gemma4/gptoss.hpp - OpenAI's gpt-oss (llama.cpp architecture "gpt-oss", 20B: 24 layers, 32 experts, 4 used).
//
// A prototype on this engine's MoE paths.  Every layer: RMS norm, attention with biases, an attention sink a head and
// YaRN rotary positions (a window layer in every other one, from the first), the residual, a second RMS norm
// (the file calls it post_attention_norm), and a router with a bias, four experts whose gate, up and down have
// biases of their own and the unit min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, +-7) + 1).  No dense MLP and
// no shared expert.  The reference for the math is llama.cpp's openai-moe graph.
#pragma once

#include "engine.hpp"

namespace strata::gemma4 {

class GptOss final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void router_scores(const Layer& L, int n) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;

private:
    struct Block {
        bool swa = false;   ///< attends a sliding window; its caches are rings
        Dev wq, wk, wv, wo, bq, bk, bv, bo, sinks;
        uint16_t* keys = nullptr;
        uint16_t* vals = nullptr;
        int8_t *k_codes = nullptr, *v_codes = nullptr;
        uint16_t *k_scales = nullptr, *v_scales = nullptr;
    };

    void attn(Block& B, int64_t pos, int n);
    void store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;
    std::vector<uint8_t> swa_;
    int n_head_ = 0, n_kv_ = 0, dim_ = 0, n_rot_ = 0;
    float theta_scale_ = 0, mscale_ = 1;
    int window_ = 0;
    int ring_ = 0;
    int keep_ = 0;
    int chunk_ = 1;
    std::vector<float> yarn_freq_;   ///< the rotary pairs' divisors: YaRN's blend of interpolation and extrapolation
    float* freq_dev_ = nullptr;

    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr,
          *att_ = nullptr, *score_ = nullptr;
};

}  // namespace strata::gemma4
