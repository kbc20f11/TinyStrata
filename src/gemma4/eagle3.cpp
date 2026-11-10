// One-layer GPT-OSS EAGLE3 equations follow SafeAILab/EAGLE (Apache-2.0)
// and llama.cpp's eagle3 graph (MIT). The Nebius checkpoint is CC-BY-4.0.
// This runner uses TinyStrata's existing math primitives. Target computation,
// sampling and verification are unchanged. See LICENSE and NOTICE.
#include "eagle3.hpp"

#include "runtime.hpp"
#include "strata/gemma4/eagle3.hpp"
#include "strata/gemma4/kernels.hpp"
#include "strata/gemma4/matvec.hpp"
#include "strata/gemma4/draft_q51.hpp"
#include "strata/gemma4/sampling.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace strata::gemma4 {
namespace {

void need(bool okay, const std::string& what) {
    if (!okay) throw std::runtime_error("EAGLE3: " + what);
}

std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (unsigned char ch : text) {
        if (ch == '\\' || ch == '\"') out += '\\';
        if (ch >= 32) out += char(ch);
    }
    return out + "\"";
}

}  // namespace

Eagle3::Eagle3(Engine& target) : target_(target) { history_pos_.fill(-1); }
Eagle3::~Eagle3() = default;  // The Engine owns all allocations and growing KV mappings.

size_t Eagle3::vram_bytes() const {
    size_t kv_bytes = 0;
    if (target_.vmm_ && target_.opt_.kv_grow) {
        for (size_t i = kv_begin_; i < kv_end_; ++i) kv_bytes += target_.kv_.at(i).mapped;
    } else {
        kv_bytes = 2 * size_t(max_context_) * kv_heads_ * dim_ * sizeof(uint16_t);
    }
    return allocated_ + kv_bytes;
}

void* Eagle3::allocate(size_t bytes, const char* what) {
    void* p = target_.dmalloc(bytes, what);
    allocated_ += std::max<size_t>(bytes, 256);
    return p;
}

float* Eagle3::floats(size_t n, const char* what) { return static_cast<float*>(allocate(n * sizeof(float), what)); }

Dev Eagle3::upload(const std::string& name, int n_in, int n_out, bool vector) {
    const TensorInfo* t = file_->find(name);
    need(t && !t->shape.empty() && t->shape[0] == uint64_t(n_in) &&
             t->elements() == uint64_t(n_in) * uint64_t(n_out), "missing tensor or unexpected shape: " + name);
    const size_t bytes = file_->payload(*t);
    const ggml_type_traits* traits = ggml_get_type_traits(ggml_type(t->type));
    need(traits != nullptr, "unsupported tensor type for " + name);
    const bool native = !vector && (t->type == kTypeF32 || t->type == kTypeBF16 ||
                                    kernels::gemma_native_mmvq_supported(int(t->type)));
    const bool undo_qk_permutation = name == "blk.0.attn_q.weight" || name == "blk.0.attn_k.weight";
    const uint32_t type = native ? t->type : kTypeF32;
    const size_t stored = native ? bytes : size_t(t->elements()) * sizeof(float);
    Dev out{allocate(pad(stored) + kTail, "EAGLE3 weights"), type, n_in, n_out};
    if (native) {
        if (undo_qk_permutation) {
            // llama.cpp's Llama-family converter stores consecutive rotary
            // pairs. Restore the HF half-split row order for rope_bias's NEOX
            // contract by copying entire stored rows, including quantized bits.
            // No weight is decoded or requantized by this permutation.
            need(n_out % dim_ == 0 && bytes % size_t(n_out) == 0, "invalid Q/K stored row layout");
            const size_t row_bytes = bytes / size_t(n_out);
            const auto* source = static_cast<const uint8_t*>(file_->data(*t));
            std::vector<uint8_t> reordered(bytes);
            for (int row = 0; row < n_out; ++row) {
                const int i = row % dim_;
                const int saved = row - i + 2 * (i % (dim_ / 2)) + (i >= dim_ / 2 ? 1 : 0);
                std::memcpy(reordered.data() + size_t(row) * row_bytes, source + size_t(saved) * row_bytes, row_bytes);
            }
            target_.h2d(out.p, reordered.data(), bytes);
            ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 rotary row permutation upload");
        } else target_.h2d(out.p, file_->data(*t), bytes);
    } else {
        std::vector<float> values(size_t(t->elements()));
        if (t->type == kTypeF32) std::memcpy(values.data(), file_->data(*t), bytes);
        else {
            need(traits->to_float != nullptr, "cannot decode " + name);
            traits->to_float(file_->data(*t), values.data(), int64_t(values.size()));
        }
        if (undo_qk_permutation) {
            const std::vector<float> saved = values;
            for (int row = 0; row < n_out; ++row) {
                const int i = row % dim_;
                const int from = row - i + 2 * (i % (dim_ / 2)) + (i >= dim_ / 2 ? 1 : 0);
                std::memcpy(values.data() + size_t(row) * n_in, saved.data() + size_t(from) * n_in,
                            size_t(n_in) * sizeof(float));
            }
        }
        for (float v : values) need(std::isfinite(v), "nonfinite weight in " + name);
        target_.h2d(out.p, values.data(), stored);
        ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 converted weight upload");
    }
    target_.zero_vram(static_cast<uint8_t*>(out.p) + stored, pad(stored) + kTail - stored);
    return out;
}

void Eagle3::load(const std::string& path, int max_context) {
    need(!file_, "the drafter was loaded twice");
    need(target_.oai_ && !target_.split_ && !target_.next_ && target_.opt_.batch == 0,
         "only one GPT-OSS sequence on one GPU is supported");
    file_ = std::make_unique<Shards>(path);
    const auto text = [&](const char* key) {
        const MetaValue* v = file_->get(key);
        need(v && v->type == MetaType::STRING, std::string("missing metadata ") + key);
        return v->s;
    };
    const auto integer = [&](const char* key, int fallback = -1) {
        const MetaValue* v = file_->get(key);
        if (!v && fallback >= 0) return fallback;
        need(v && v->is_num() && v->u < uint64_t(std::numeric_limits<int>::max()),
             std::string("missing or invalid metadata ") + key);
        return int(v->u);
    };
    const auto scalar = [&](const char* key, float fallback = -1.f) {
        const MetaValue* v = file_->get(key);
        if (!v && fallback >= 0) return fallback;
        need(v && v->is_num(), std::string("missing metadata ") + key);
        const float f = float(v->num());
        need(std::isfinite(f) && f > 0, std::string("invalid metadata ") + key);
        return f;
    };
    need(text("general.architecture") == "eagle3", "the GGUF architecture is not eagle3");
    hidden_ = integer("eagle3.embedding_length");
    inter_ = integer("eagle3.feed_forward_length");
    heads_ = integer("eagle3.attention.head_count");
    kv_heads_ = integer("eagle3.attention.head_count_kv");
    dim_ = integer("eagle3.attention.key_length");
    max_context_ = std::min(max_context, integer("eagle3.context_length"));
    eps_ = scalar("eagle3.attention.layer_norm_rms_epsilon");
    need(hidden_ == 2880 && inter_ == 11520 && heads_ == 64 && kv_heads_ == 8 && dim_ == 64 &&
             integer("eagle3.block_count") == 1 && target_.n_embd_ == hidden_ && target_.n_vocab_ == 201088,
         "only the matching one-layer Nebius GPT-OSS 20B EAGLE3 is supported");
    need(max_context_ == max_context && max_context > kDepth, "context length mismatch");
    need(integer("eagle3.target_hidden_size", hidden_) == hidden_, "target hidden width mismatch");
    const MetaValue* norm_residual = file_->get("eagle3.norm_before_residual");
    const MetaValue* norm_fc = file_->get("eagle3.norm_before_fc");
    need((!norm_residual || norm_residual->u == 0) && (!norm_fc || norm_fc->u == 0),
         "normalized residual and normalized FC input variants are unsupported");
    need(!file_->find("token_embd.weight") && !file_->find("enc.output_norm.weight") &&
             !file_->find("blk.0.attn_q.bias") && !file_->find("blk.0.attn_q_norm.weight"),
         "private embeddings, FC input norm, biased attention and Q/K norm variants are unsupported");

    const MetaValue* tap = file_->get("eagle3.target_layers");
    need(tap && tap->type == MetaType::ARRAY && tap->items.size() == 3, "expected three target hidden-state taps");
    const int expected[] = {2, 12, 21};
    for (size_t i = 0; i < tap->items.size(); ++i) {
        need(tap->items[i].is_num() && tap->items[i].u == uint64_t(expected[i]), "target hidden-state taps mismatch");
        // GGUF describes layer inputs. These are outputs of blocks 1, 11, 20.
        taps_.push_back(expected[i] - 1);
    }

    need(text("eagle3.rope.scaling.type") == "yarn", "only the checkpoint's YaRN is supported");
    const float base = scalar("eagle3.rope.freq_base");
    const float factor = scalar("eagle3.rope.scaling.factor");
    const float original = float(integer("eagle3.rope.scaling.original_context_length"));
    const float fast = scalar("eagle3.rope.scaling.yarn_beta_fast", 32);
    const float slow = scalar("eagle3.rope.scaling.yarn_beta_slow", 1);
    need(base == 150000 && factor == 32 && original == 4096 && fast == 32 && slow == 1 &&
             integer("eagle3.rope.dimension_count", dim_) == dim_, "unexpected rotary parameters");
    // The source config explicitly uses truncate=false: retain fractional bounds.
    const auto correction = [&](double beta) {
        return double(dim_) * std::log(double(original) / (beta * 2.0 * 3.14159265358979323846)) /
               (2.0 * std::log(double(base)));
    };
    const float low = float(std::max(0.0, correction(fast)));
    const float high = float(std::min(double(dim_ - 1), correction(slow)));
    std::vector<float> frequencies(size_t(dim_ / 2));
    for (int j = 0; j < dim_ / 2; ++j) {
        const float ramp = 1.f - std::clamp((float(j) - low) / std::max(0.001f, high - low), 0.f, 1.f);
        frequencies[size_t(j)] = 1.f / ((1.f - ramp) / factor + ramp);
    }
    theta_scale_ = std::pow(base, -2.f / float(dim_));
    mscale_ = 1.f + 0.1f * std::log(factor);
    freq_ = floats(frequencies.size(), "EAGLE3 YaRN frequencies");
    target_.h2d(freq_, frequencies.data(), frequencies.size() * sizeof(float));

    fc_ = upload("fc.weight", tap_width(), hidden_);
    input_norm_ = upload("blk.0.attn_norm.weight", hidden_, 1, true);
    hidden_norm_ = upload("blk.0.attn_norm_2.weight", hidden_, 1, true);
    ffn_norm_ = upload("blk.0.ffn_norm.weight", hidden_, 1, true);
    final_norm_ = upload("output_norm.weight", hidden_, 1, true);
    q_weight_ = upload("blk.0.attn_q.weight", 2 * hidden_, heads_ * dim_);
    k_weight_ = upload("blk.0.attn_k.weight", 2 * hidden_, kv_heads_ * dim_);
    v_weight_ = upload("blk.0.attn_v.weight", 2 * hidden_, kv_heads_ * dim_);
    o_weight_ = upload("blk.0.attn_output.weight", heads_ * dim_, hidden_);
    gate_weight_ = upload("blk.0.ffn_gate.weight", hidden_, inter_);
    up_weight_ = upload("blk.0.ffn_up.weight", hidden_, inter_);
    down_weight_ = upload("blk.0.ffn_down.weight", inter_, hidden_);
    const TensorInfo* head = file_->find("output.weight");
    need(head && head->shape.size() == 2 && head->shape[1] == 64000, "expected the 64000-token draft head");
    vocab_ = int(head->shape[1]);
    head_ = upload("output.weight", hidden_, vocab_);

    // llama.cpp's converter adds arange to the source HF offset table. GGUF d2t
    // is therefore an absolute target ID, never guess its convention from values.
    const TensorInfo* mapping = file_->find("d2t");
    need(mapping && mapping->shape.size() == 1 && mapping->elements() == uint64_t(vocab_) &&
             mapping->type == GGML_TYPE_I64, "expected the converted GGUF I64 absolute vocabulary map");
    std::vector<uint8_t> seen(size_t(target_.n_vocab_), 0);
    d2t_host_.resize(size_t(vocab_));
    const auto* mapping_bytes = static_cast<const uint8_t*>(file_->data(*mapping));
    for (int i = 0; i < vocab_; ++i) {
        int64_t id;
        std::memcpy(&id, mapping_bytes + size_t(i) * sizeof(id), sizeof(id));
        need(id >= 0 && id < target_.n_vocab_ && !seen[size_t(id)], "invalid or duplicate target vocabulary ID");
        seen[size_t(id)] = 1;
        d2t_host_[size_t(i)] = int32_t(id);
    }
    d2t_ = static_cast<int32_t*>(allocate(size_t(vocab_) * sizeof(int32_t), "EAGLE3 vocabulary map"));
    target_.h2d(d2t_, d2t_host_.data(), size_t(vocab_) * sizeof(int32_t));

    kv_begin_ = target_.kv_.size();
    const size_t row_bytes = size_t(kv_heads_) * dim_ * sizeof(uint16_t);
    keys_ = static_cast<uint16_t*>(target_.kv_alloc(row_bytes, 1));
    values_ = static_cast<uint16_t*>(target_.kv_alloc(row_bytes, 1));
    kv_end_ = target_.kv_.size();
    target_.kv_rows_ = 0;
    target_.cover(1);
    encoded_ = floats(size_t(kChunk) * hidden_, "EAGLE3 fused features");
    shifted_ = floats(size_t(kChunk) * hidden_, "EAGLE3 shifted features");
    pending_ = floats(hidden_, "EAGLE3 deferred feature");
    history_ = floats(size_t(kHistory) * hidden_, "EAGLE3 short rewind history");
    embedding_ = floats(size_t(kChunk) * hidden_, "EAGLE3 embeddings");
    concat_ = floats(size_t(kChunk) * 2 * hidden_, "EAGLE3 normalized concatenation");
    state_ = floats(hidden_, "EAGLE3 autoregressive hidden state");
    norm_ = floats(hidden_, "EAGLE3 FFN and final norm");
    qraw_ = floats(size_t(heads_) * dim_, "EAGLE3 queries");
    kraw_ = floats(size_t(kChunk) * kv_heads_ * dim_, "EAGLE3 keys");
    v_ = floats(size_t(kChunk) * kv_heads_ * dim_, "EAGLE3 values");
    q_ = floats(size_t(heads_) * dim_, "EAGLE3 rotary queries");
    k_ = floats(size_t(kChunk) * kv_heads_ * dim_, "EAGLE3 rotary keys");
    attention_ = floats(size_t(heads_) * dim_, "EAGLE3 attention output");
    out_ = floats(hidden_, "EAGLE3 layer output");
    gate_ = floats(inter_, "EAGLE3 gate");
    up_ = floats(inter_, "EAGLE3 up projection");
    act_ = floats(inter_, "EAGLE3 activation");
    logits_ = floats(size_t(kDepth) * vocab_, "EAGLE3 reduced-vocabulary logits");
    score_ = floats(size_t(heads_) * max_context_, "EAGLE3 attention scores");
    quant_ = allocate(kChunk * kernels::gemma_native_q8_1_bytes(std::max(tap_width(), inter_)),
                      "EAGLE3 activation quantization");
    embedding_host_ = static_cast<float*>(target_.hmalloc(size_t(kChunk) * hidden_ * sizeof(float),
                                                         "EAGLE3 embedding staging"));
    pick_ = static_cast<Likeliest*>(allocate(sizeof(Likeliest), "EAGLE3 proposal pick"));
    pick_host_ = static_cast<Likeliest*>(target_.hmalloc(sizeof(Likeliest), "EAGLE3 pick staging"));
    sample_workspace_bytes_ = full_vocab_sample_workspace_bytes(target_.n_vocab_);
    sample_workspace_ = allocate(sample_workspace_bytes_, "EAGLE3 categorical workspace");
    sample_result_ = static_cast<FullVocabResult*>(allocate(sizeof(FullVocabResult), "EAGLE3 categorical result"));
    sample_result_host_ = static_cast<FullVocabResult*>(target_.hmalloc(sizeof(FullVocabResult),
                                                                      "EAGLE3 categorical staging"));
    draft_probabilities_ = static_cast<double*>(allocate(size_t(vocab_) * sizeof(double), "EAGLE3 reduced probabilities"));
    probabilities_ = static_cast<double*>(allocate(size_t(kDepth) * target_.n_vocab_ * sizeof(double),
                                                   "EAGLE3 target vocabulary probabilities"));
    verify_accepted_ = static_cast<uint32_t*>(allocate(sizeof(uint32_t), "EAGLE3 rejection flag"));
    verify_accepted_host_ = static_cast<uint32_t*>(target_.hmalloc(sizeof(uint32_t), "EAGLE3 rejection staging"));
    if (const char* dir = std::getenv("TINYSTRATA_EAGLE3_DUMP_DIR")) {
        dump_dir_ = dir;
        if (!dump_dir_.empty()) std::filesystem::create_directories(dump_dir_);
    }
    ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 load completion");
    reset();
}

void Eagle3::linear(const Dev& w, const float* x, float* y, int n) {
    if (w.type == kTypeF32) return gemv_f32(fp(w), x, y, w.n_in, w.n_out, w.n_out, n, target_.st_);
    if (w.type == kTypeBF16)
        return gemv_bf16(static_cast<const uint16_t*>(w.p), x, y, w.n_in, w.n_out, w.n_out, n, target_.st_);
    kernels::gemma_native_quantize_q8_1(x, quant_, w.n_in * n, 1, target_.st_);
    const size_t row_bytes = kernels::gemma_native_q8_1_bytes(w.n_in);
    for (int row = 0; row < n; row += 8) {
        const void* xq = static_cast<const uint8_t*>(quant_) + size_t(row) * row_bytes;
        float* out = y + size_t(row) * w.n_out;
        const int count = std::min(8, n - row);
        if (w.type == 7) draft_q51_mmvq(w.p, xq, out, w.n_in, w.n_out, count, target_.st_);
        else kernels::gemma_native_mmvq(int(w.type), w.p, xq, out, w.n_in, w.n_out, count, target_.st_);
    }
}

void Eagle3::normalize_inputs(const float* features, const int32_t* tokens, int n) {
    // Reusing pinned staging is safe only after its previous asynchronous copy.
    ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 before embedding staging");
    for (int r = 0; r < n; ++r) {
        need(tokens[r] >= 0 && tokens[r] < target_.n_vocab_, "invalid target input token");
        target_.embed(tokens[r], embedding_host_ + size_t(r) * hidden_);
    }
    target_.h2d(embedding_, embedding_host_, size_t(n) * hidden_ * sizeof(float));
    rms_norm(embedding_, fp(input_norm_), concat_, hidden_, n, eps_, target_.st_, 2 * hidden_);
    rms_norm(features, fp(hidden_norm_), concat_ + hidden_, hidden_, n, eps_, target_.st_, 2 * hidden_);
}

void Eagle3::project_kv(int64_t pos0, int n) {
    linear(k_weight_, concat_, kraw_, n);
    linear(v_weight_, concat_, v_, n);
    rope_bias(kraw_, nullptr, k_, dim_, kv_heads_, dim_, theta_scale_, freq_, mscale_, pos0, n, target_.st_);
    store_f16(k_, keys_, kv_heads_ * dim_, pos0, n, target_.st_);
    store_f16(v_, values_, kv_heads_ * dim_, pos0, n, target_.st_);
}

void Eagle3::reset() {
    pos_ = 0;
    valid_ = true;
    probability_rows_ = 0;
    history_pos_.fill(-1);
}

void Eagle3::rewind(int64_t keep) {
    need(keep >= 0, "negative rewind position");
    probability_rows_ = 0;
    if (keep == 0) { reset(); return; }
    if (!valid_ || keep > pos_ || history_pos_[size_t((keep - 1) % kHistory)] != keep - 1) {
        valid_ = false;
        return;
    }
    ck(cudaMemcpyAsync(pending_, history_ + size_t((keep - 1) % kHistory) * hidden_,
                       size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToDevice, cs(target_.st_)),
       "EAGLE3 restore deferred feature");
    pos_ = keep;
    for (int64_t& p : history_pos_) if (p >= keep) p = -1;
}

void Eagle3::append(const float* features, const int32_t* tokens, int64_t pos0, int n) {
    need(file_ && features && tokens && n > 0 && pos0 >= 0 && pos0 + n <= max_context_, "invalid context append");
    probability_rows_ = 0;
    if (!valid_ || pos0 != pos_) { valid_ = false; return; }
    const bool capture = !dump_dir_.empty() && !dumped_context_ && pos0 == 0 && n <= kChunk;
    if (capture) {
        dump_context_rows_ = n;
        dump_context_pos_ = pos0;
        dump_tokens_.assign(tokens, tokens + n);
        dump("context.features", features, n, tap_width());
    }
    for (int start = 0; start < n; start += kChunk) {
        const int rows = std::min(kChunk, n - start);
        const int64_t first = pos0 + start;
        linear(fc_, features + size_t(start) * tap_width(), encoded_, rows);
        if (capture) dump("context.fc", encoded_, rows, hidden_);
        const bool have_boundary = first > 0;
        const int count = rows - (have_boundary ? 0 : 1);
        if (count > 0) {
            if (have_boundary) {
                ck(cudaMemcpyAsync(shifted_, pending_, size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToDevice,
                                   cs(target_.st_)), "EAGLE3 cross-chunk boundary feature");
                if (rows > 1)
                    ck(cudaMemcpyAsync(shifted_ + hidden_, encoded_, size_t(rows - 1) * hidden_ * sizeof(float),
                                       cudaMemcpyDeviceToDevice, cs(target_.st_)), "EAGLE3 shifted context features");
            } else {
                ck(cudaMemcpyAsync(shifted_, encoded_, size_t(count) * hidden_ * sizeof(float),
                                   cudaMemcpyDeviceToDevice, cs(target_.st_)), "EAGLE3 initial context features");
            }
            normalize_inputs(shifted_, tokens + start + (have_boundary ? 0 : 1), count);
            project_kv(first - (have_boundary ? 1 : 0), count);
            if (capture) {
                dump("context.concat", concat_, count, 2 * hidden_);
                dump("context.k_raw", kraw_, count, kv_heads_ * dim_);
                dump("context.k_rope", k_, count, kv_heads_ * dim_);
                dump("context.v", v_, count, kv_heads_ * dim_);
            }
        }
        ck(cudaMemcpyAsync(pending_, encoded_ + size_t(rows - 1) * hidden_, size_t(hidden_) * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(target_.st_)), "EAGLE3 defer final context feature");
        // Only a few recent true features are needed for bounded window rollback.
        // Deep prefix restores fall back until a fresh reset/rebuild.
        for (int row = std::max(0, rows - kHistory); row < rows; ++row) {
            const int64_t p = first + row;
            const size_t slot = size_t(p % kHistory);
            ck(cudaMemcpyAsync(history_ + slot * hidden_, encoded_ + size_t(row) * hidden_,
                               size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToDevice, cs(target_.st_)),
               "EAGLE3 remember true context feature");
            history_pos_[slot] = p;
        }
    }
    pos_ += n;
    if (capture) { dumped_context_ = true; dump_manifest(); }
}

int Eagle3::prepare(int32_t anchor, int cap) {
    probability_rows_ = 0;
    if (!file_ || !valid_ || pos_ <= 0 || pos_ != target_.pos_ || cap <= 0) return 0;
    need(anchor >= 0 && anchor < target_.n_vocab_, "invalid proposal anchor");
    cap = int(std::min<int64_t>(std::min(cap, kDepth), max_context_ - pos_ - 1));
    if (cap <= 0) return 0;
    target_.cover(pos_ + cap + 1);
    ck(cudaMemcpyAsync(state_, pending_, size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToDevice,
                       cs(target_.st_)), "EAGLE3 seed deferred feature");
    dumping_ = !dump_dir_.empty() && !did_dump_ && dumped_context_ && pos_ == dump_context_pos_ + dump_context_rows_;
    if (dumping_) {
        dump_block_pos_ = pos_;
        dump_anchor_ = anchor;
        dump_steps_ = 0;
        dump_step_tokens_.clear();
    }
    return cap;
}

void Eagle3::decode(int32_t token, int step) {
    const int64_t at = pos_ - 1 + step;
    const std::string prefix = "step" + std::to_string(step);
    if (dumping_) dump(prefix + ".g", state_, 1, hidden_);
    normalize_inputs(state_, &token, 1);
    if (dumping_) {
        dump_step_tokens_.push_back(token);
        dump_steps_ = step + 1;
        dump(prefix + ".embedding", embedding_, 1, hidden_);
        dump(prefix + ".concat", concat_, 1, 2 * hidden_);
    }
    linear(q_weight_, concat_, qraw_, 1);
    rope_bias(qraw_, nullptr, q_, dim_, heads_, dim_, theta_scale_, freq_, mscale_, at, 1, target_.st_);
    project_kv(at, 1);
    if (dumping_) {
        dump(prefix + ".q_raw", qraw_, 1, heads_ * dim_);
        dump(prefix + ".q_rope", q_, 1, heads_ * dim_);
        dump(prefix + ".k_raw", kraw_, 1, kv_heads_ * dim_);
        dump(prefix + ".k_rope", k_, 1, kv_heads_ * dim_);
        dump(prefix + ".v", v_, 1, kv_heads_ * dim_);
    }
    attend(q_, keys_, values_, at, 1, score_, 1, attention_, dim_, heads_, kv_heads_,
           1.f / std::sqrt(float(dim_)), target_.st_);
    linear(o_weight_, attention_, out_, 1);
    if (dumping_) {
        dump(prefix + ".attention", attention_, 1, heads_ * dim_);
        dump(prefix + ".projected", out_, 1, hidden_);
    }
    axpy(out_, 1.f, state_, hidden_, true, target_.st_);
    if (dumping_) dump(prefix + ".attention_residual", state_, 1, hidden_);
    rms_norm(state_, fp(ffn_norm_), norm_, hidden_, 1, eps_, target_.st_);
    linear(gate_weight_, norm_, gate_, 1);
    linear(up_weight_, norm_, up_, 1);
    swiglu_clamp(gate_, up_, act_, inter_, INFINITY, target_.st_);
    linear(down_weight_, act_, out_, 1);
    if (dumping_) {
        dump(prefix + ".ffn_norm", norm_, 1, hidden_);
        dump(prefix + ".gate", gate_, 1, inter_);
        dump(prefix + ".up", up_, 1, inter_);
        dump(prefix + ".activation", act_, 1, inter_);
        dump(prefix + ".down", out_, 1, hidden_);
    }
    axpy(out_, 1.f, state_, hidden_, true, target_.st_);
    rms_norm(state_, fp(final_norm_), norm_, hidden_, 1, eps_, target_.st_);
    linear(head_, norm_, logits_ + size_t(step) * vocab_, 1);
    if (dumping_) {
        dump(prefix + ".prenorm", state_, 1, hidden_);
        dump(prefix + ".final", norm_, 1, hidden_);
        dump(prefix + ".logits", logits_ + size_t(step) * vocab_, 1, vocab_);
    }
}

int Eagle3::propose(int32_t anchor, int32_t* result, int cap, float* probs) {
    if (cap > 0) need(result != nullptr, "invalid proposal destination");
    cap = prepare(anchor, cap);
    int32_t token = anchor;
    for (int row = 0; row < cap; ++row) {
        decode(token, row);
        likeliest(logits_ + size_t(row) * vocab_, vocab_, pick_, target_.st_);
        ck(cudaMemcpyAsync(pick_host_, pick_, sizeof(Likeliest), cudaMemcpyDeviceToHost, cs(target_.st_)),
           "EAGLE3 proposal download");
        ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 proposal completion");
        need(pick_host_->token >= 0 && pick_host_->token < vocab_ && std::isfinite(pick_host_->prob),
             "invalid draft logits");
        token = d2t_host_[size_t(pick_host_->token)];
        result[row] = token;
        if (probs) probs[row] = pick_host_->prob;
    }
    finish_dump();
    return cap;
}

int Eagle3::sample_propose(int32_t anchor, int32_t* result, int cap, double temperature, const double* uniforms) {
    probability_rows_ = 0;
    if (cap <= 0) return 0;
    need(result && uniforms && std::isfinite(temperature) && temperature > 0, "invalid categorical proposal arguments");
    for (int row = 0; row < std::min(cap, kDepth); ++row)
        need(std::isfinite(uniforms[row]) && uniforms[row] >= 0 && uniforms[row] < 1,
             "a draft uniform must be in [0,1)");
    cap = prepare(anchor, cap);
    int32_t token = anchor;
    for (int row = 0; row < cap; ++row) {
        decode(token, row);
        sample_full_vocab(logits_ + size_t(row) * vocab_, vocab_, temperature, uniforms[row],
                          sample_workspace_, sample_workspace_bytes_, sample_result_, target_.st_);
        copy_full_vocab_probabilities(sample_workspace_, sample_workspace_bytes_, sample_result_, vocab_,
                                      draft_probabilities_, target_.st_);
        eagle3_scatter_probabilities(draft_probabilities_, d2t_, probabilities_ + size_t(row) * target_.n_vocab_,
                                     vocab_, target_.n_vocab_, target_.st_);
        ck(cudaMemcpyAsync(sample_result_host_, sample_result_, sizeof(FullVocabResult), cudaMemcpyDeviceToHost,
                           cs(target_.st_)), "EAGLE3 categorical result download");
        ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 categorical completion");
        const FullVocabResult& got = *sample_result_host_;
        need(got.status == kSampleOk && got.token >= 0 && got.token < vocab_ &&
                 std::isfinite(got.total) && got.total > 0 && std::isfinite(got.probability) &&
                 got.probability > 0 && got.probability <= 1, "invalid categorical draft distribution");
        token = d2t_host_[size_t(got.token)];
        result[row] = token;
        sampled_tokens_[size_t(row)] = token;
    }
    sampled_temperature_ = temperature;
    probability_rows_ = cap;
    finish_dump();
    return cap;
}

const double* Eagle3::probabilities(int row) const {
    if (row < 0 || row >= probability_rows_)
        throw std::out_of_range("EAGLE3: no current sampled proposal probability row");
    return probabilities_ + size_t(row) * target_.n_vocab_;
}

bool Eagle3::verify(const float* target_logits, int row, int proposed, double temperature,
                    double uniform_accept, double uniform_residual, bool* accepted, int32_t* out) {
    const double* q = probabilities(row);
    need(target_logits && accepted && out, "invalid rejection verifier pointers");
    need(proposed >= 0 && proposed < target_.n_vocab_ && proposed == sampled_tokens_[size_t(row)] &&
             temperature == sampled_temperature_, "the verifier must use the sampled draft token and temperature");
    verify_full_vocab(target_logits, q, proposed, target_.n_vocab_, temperature, uniform_accept, uniform_residual,
                      sample_workspace_, sample_workspace_bytes_, sample_result_, verify_accepted_, target_.st_);
    ck(cudaMemcpyAsync(sample_result_host_, sample_result_, sizeof(FullVocabResult), cudaMemcpyDeviceToHost,
                       cs(target_.st_)), "EAGLE3 rejection result download");
    ck(cudaMemcpyAsync(verify_accepted_host_, verify_accepted_, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                       cs(target_.st_)), "EAGLE3 rejection flag download");
    ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 rejection completion");
    const FullVocabResult& got = *sample_result_host_;
    need(got.status == kSampleOk, "invalid target or draft rejection distribution (status=" + std::to_string(got.status) + ")");
    need(*verify_accepted_host_ <= 1 && got.token >= 0 && got.token < target_.n_vocab_ &&
             std::isfinite(got.total) && got.total > 0 && std::isfinite(got.probability) &&
             got.probability > 0 && got.probability <= 1, "invalid rejection result");
    need(!*verify_accepted_host_ || got.token == proposed, "accepted rejection changed the proposal");
    *accepted = *verify_accepted_host_ != 0;
    *out = got.token;
    return true;
}

void Eagle3::dump(const std::string& name, const float* device, int rows, int width) {
    const std::string file = name + ".f32";
    std::vector<float> values(size_t(rows) * width);
    ck(cudaMemcpyAsync(values.data(), device, values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                       cs(target_.st_)), "EAGLE3 diagnostic copy");
    ck(cudaStreamSynchronize(cs(target_.st_)), "EAGLE3 diagnostic completion");
    const auto path = std::filesystem::path(dump_dir_) / file;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    need(bool(out), "cannot write diagnostic " + path.string());
    out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
    need(bool(out), "short diagnostic write " + path.string());
    dumps_.push_back({name, file, rows, width});
}

void Eagle3::finish_dump() {
    if (dumping_) { did_dump_ = true; dump_manifest(); }
    dumping_ = false;
}

void Eagle3::dump_manifest() {
    const auto path = std::filesystem::path(dump_dir_) / "fixture.json";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    need(bool(out), "cannot write diagnostic manifest");
    out << "{\"context_pos\":" << dump_context_pos_ << ",\"context_rows\":" << dump_context_rows_
        << ",\"target_pos\":" << dump_block_pos_ << ",\"anchor\":" << dump_anchor_
        << ",\"steps\":" << dump_steps_ << ",\"kv_rounding\":\"f16\",\"rope_truncate\":false,\"tokens\":[";
    for (size_t i = 0; i < dump_tokens_.size(); ++i) out << (i ? "," : "") << dump_tokens_[i];
    out << "],\"step_tokens\":[";
    for (size_t i = 0; i < dump_step_tokens_.size(); ++i) out << (i ? "," : "") << dump_step_tokens_[i];
    out << "],\"arrays\":{";
    for (size_t i = 0; i < dumps_.size(); ++i) {
        const Dump& d = dumps_[i];
        out << (i ? "," : "") << quoted(d.name) << ":{\"file\":" << quoted(d.file)
            << ",\"shape\":[" << d.rows << "," << d.width << "],\"dtype\":\"float32\"}";
    }
    out << "}}\n";
    need(bool(out), "short diagnostic manifest write");
}

}  // namespace strata::gemma4
