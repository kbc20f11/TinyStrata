// Native one-layer EAGLE3 for the matching GPT-OSS 20B checkpoint.
// Draft proposals are always verified by the unchanged target model.
#pragma once

#include "engine.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace strata::gemma4 {

struct FullVocabResult;

class Eagle3 {
public:
    explicit Eagle3(Engine& target);
    ~Eagle3();
    Eagle3(const Eagle3&) = delete;
    Eagle3& operator=(const Eagle3&) = delete;

    void load(const std::string& path, int max_context);
    const std::vector<int>& tap_layers() const { return taps_; }
    int tap_width() const { return int(taps_.size()) * hidden_; }
    int block_size() const { return kDepth + 1; }
    int64_t pos() const { return pos_; }
    size_t vram_bytes() const;

    // Features are device [n, tap_width()], tokens are host target-token IDs.
    // KV position P pairs token[P+1] with FC(features[P]). The final feature
    // stays pending until the target's next token is known.
    void append(const float* features, const int32_t* tokens, int64_t pos0, int n);
    void reset();
    void rewind(int64_t keep);
    int propose(int32_t anchor, int32_t* out, int cap, float* probs = nullptr);
    int sample_propose(int32_t anchor, int32_t* out, int cap, double temperature, const double* uniforms);
    const double* probabilities(int row) const;
    bool verify(const float* target_logits, int row, int proposed, double temperature,
                double uniform_accept, double uniform_residual, bool* accepted, int32_t* out);

private:
    static constexpr int kDepth = 2, kChunk = 32, kHistory = 8;
    Dev upload(const std::string& name, int n_in, int n_out, bool vector = false);
    void* allocate(size_t bytes, const char* what);
    float* floats(size_t n, const char* what);
    void linear(const Dev& w, const float* x, float* y, int n);
    void normalize_inputs(const float* features, const int32_t* tokens, int n);
    void project_kv(int64_t pos0, int n);
    int prepare(int32_t anchor, int cap);
    void decode(int32_t token, int step);
    void finish_dump();
    void dump(const std::string& name, const float* device, int rows, int width);
    void dump_manifest();

    Engine& target_;
    std::unique_ptr<Shards> file_;
    std::vector<int> taps_;
    std::vector<int32_t> d2t_host_;
    Dev fc_, input_norm_, hidden_norm_, ffn_norm_, final_norm_;
    Dev q_weight_, k_weight_, v_weight_, o_weight_, gate_weight_, up_weight_, down_weight_, head_;
    int hidden_ = 0, inter_ = 0, heads_ = 0, kv_heads_ = 0, dim_ = 0, vocab_ = 0, max_context_ = 0;
    int64_t pos_ = 0;
    bool valid_ = true;
    float eps_ = 0, theta_scale_ = 0, mscale_ = 0;
    size_t allocated_ = 0, kv_begin_ = 0, kv_end_ = 0;
    uint16_t *keys_ = nullptr, *values_ = nullptr;
    int32_t* d2t_ = nullptr;
    float *freq_ = nullptr, *encoded_ = nullptr, *shifted_ = nullptr, *pending_ = nullptr, *history_ = nullptr;
    std::array<int64_t, kHistory> history_pos_{};
    float *embedding_ = nullptr, *concat_ = nullptr, *state_ = nullptr, *norm_ = nullptr;
    float *qraw_ = nullptr, *kraw_ = nullptr, *v_ = nullptr, *q_ = nullptr, *k_ = nullptr;
    float *attention_ = nullptr, *out_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *act_ = nullptr;
    float *logits_ = nullptr, *score_ = nullptr;
    void* quant_ = nullptr;
    float* embedding_host_ = nullptr;
    Likeliest *pick_ = nullptr, *pick_host_ = nullptr;
    void* sample_workspace_ = nullptr;
    size_t sample_workspace_bytes_ = 0;
    FullVocabResult *sample_result_ = nullptr, *sample_result_host_ = nullptr;
    double *draft_probabilities_ = nullptr, *probabilities_ = nullptr;
    int probability_rows_ = 0;
    uint32_t *verify_accepted_ = nullptr, *verify_accepted_host_ = nullptr;
    std::array<int32_t, kDepth> sampled_tokens_{};
    double sampled_temperature_ = 0;

    std::string dump_dir_;
    struct Dump { std::string name, file; int rows, width; };
    std::vector<Dump> dumps_;
    bool dumping_ = false, did_dump_ = false, dumped_context_ = false;
    int dump_context_rows_ = 0, dump_steps_ = 0;
    int32_t dump_anchor_ = -1;
    int64_t dump_context_pos_ = 0, dump_block_pos_ = 0;
    std::vector<int32_t> dump_tokens_, dump_step_tokens_;
};

}  // namespace strata::gemma4
