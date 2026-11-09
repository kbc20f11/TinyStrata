// src/gemma4/gptoss.cpp - gpt-oss's layers.  The reference for the math is llama.cpp's openai-moe graph.
#include "gptoss.hpp"

#include "runtime.hpp"
#include "strata/gemma4/kernels.hpp"
#include "strata/gemma4/matvec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::gemma4 {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr float kOaiLimit = 7.0f;                  // llama.cpp's build_moe_ffn_from_probs for openai-moe
constexpr float kOaiAlpha = 1.702f;

}  // namespace

void GptOss::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
}

void GptOss::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    window_ = shape_i("attention.sliding_window");
    dim_ = shape_i("attention.key_length");
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : dim_;
    const float base = shape_f("rope.freq_base");
    need(window_ > 0 && dim_ > 0 && n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256,
         "bad attention shape");
    need(shape_i("attention.value_length") == dim_, "the attention's keys and values differ in length");
    need(n_rot_ % 2 == 0 && n_rot_ <= dim_ && dim_ % 32 == 0, "bad rope.dimension_count");
    need(n_expert_ == 32 && n_used_ == 4 && n_ff_exp_ == n_embd_, "an expert layout other than gpt-oss-20b's");
    n_ff_ = n_ff_exp_;
    swa_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) swa_[(size_t) il] = il % 2 == 0;   // llama.cpp's set_swa_pattern(2)

    // YaRN (ggml's rope_yarn): pairs below `low` keep their frequency, pairs above `high` are interpolated by the
    // factor, and the ones between blend.  The cosines and sines are scaled by 1 + 0.1 ln(factor).
    const MetaValue* kind = find("rope.scaling.type");
    need(kind && kind->s == "yarn", "a rope other than YaRN");
    const float factor = shape_f("rope.scaling.factor");
    const float orig = (float) shape_i("rope.scaling.original_context_length")   /* an integer: shape_f would read 0 */;
    const float fast = shape_f("rope.scaling.yarn_beta_fast"), slow = shape_f("rope.scaling.yarn_beta_slow");
    const auto corr = [&](float beta) {
        return n_rot_ * std::log(orig / (beta * 2.f * 3.14159265358979f)) / (2.f * std::log(base));
    };
    const float low = std::max(0.f, std::floor(corr(fast))), high = std::min((float) n_rot_ - 1.f, std::ceil(corr(slow)));
    theta_scale_ = std::pow(base, -2.f / (float) n_rot_);
    yarn_freq_.resize((size_t) n_rot_ / 2);
    for (int j = 0; j < n_rot_ / 2; ++j) {
        const float ramp = 1.f - std::clamp(((float) j - low) / std::max(0.001f, high - low), 0.f, 1.f);   // extrapolation's share
        yarn_freq_[(size_t) j] = 1.f / ((1.f - ramp) / factor + ramp);   // theta = turn / freq
    }
    mscale_ = 1.f + 0.1f * std::log(factor);

    // YaRN stretches 4096 positions by 32: the model's own limit.  The automatic context is sized by the card.
    const int trained = shape_i("context_length");
    if (trained > 0 && opt_.max_context > trained) {
        std::fprintf(stderr, "gpt-oss: the context is limited to the model's %d tokens (asked for %d)\n", trained,
                     opt_.max_context);
        opt_.max_context = trained;
    }
    if (opt_.kv_format == KvFmt::kAuto) opt_.kv_format = KvFmt::kQ8_0;
    need(opt_.kv_format == KvFmt::kFp16 || opt_.kv_format == KvFmt::kQ8_0,
         "a KV format other than fp16 or int8 (the sink is not in the FP8 attention yet)");

    oai_ = true;
    oai_alpha_ = kOaiAlpha;
    clamp_exp_.assign((size_t) n_layer_, kOaiLimit);
    clamp_shexp_ = clamp_exp_;
}

void GptOss::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    B.swa = swa_[(size_t) il] != 0;
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    const int64_t q = (int64_t) n_head_ * dim_, kv = (int64_t) n_kv_ * dim_;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, q);
    B.bq = f32("attn_q.bias", q);
    B.wk = mat("attn_k.weight", n_embd_, kv);
    B.bk = f32("attn_k.bias", kv);
    B.wv = mat("attn_v.weight", n_embd_, kv);
    B.bv = f32("attn_v.bias", kv);
    B.wo = mat("attn_output.weight", q, n_embd_);
    B.bo = f32("attn_output.bias", n_embd_);
    B.sinks = f32("attn_sinks.weight", n_head_);
    L.ffn_norm = f32("post_attention_norm.weight", n_embd_);   // ahead of the experts: this file's name for it
    L.b_router = fp(f32("ffn_gate_inp.bias", n_expert_));
    load_moe(il, false);
    L.b_gate = fp(f32("ffn_gate_exps.bias", n_ff_exp_, n_expert_));
    L.b_up = fp(f32("ffn_up_exps.bias", n_ff_exp_, n_expert_));
    L.b_down = fp(f32("ffn_down_exps.bias", n_embd_, n_expert_));
    // the CPU's experts read the same values from the file
    const auto host = [&](const char* n) {
        const TensorInfo* t = tensor(il, n);
        if (!t || t->type != kTypeF32) throw std::runtime_error(blk(il, n) + " is not F32");
        home(*t).payload(*t);
        return (const float*) home(*t).data(*t);
    };
    L.hb_gate = host("ffn_gate_exps.bias");
    L.hb_up = host("ffn_up_exps.bias");
    L.hb_down = host("ffn_down_exps.bias");
}

int GptOss::widest() const { return n_head_ * dim_; }

// The caches: a full layer keeps a row a position, a window layer a ring of the rows a call can read.
void GptOss::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        const size_t row = (size_t) n_kv_ * dim_;
        const auto buffer = [&](size_t bytes) -> void* {
            return B.swa ? dmalloc(bytes * ring_, "the window attention cache") : kv_alloc(bytes, 1);
        };
        if (opt_.kv_format != KvFmt::kFp16) {
            B.k_codes = (int8_t*) buffer(row); B.v_codes = (int8_t*) buffer(row);
            B.k_scales = (uint16_t*) buffer(row / 32 * sizeof(uint16_t));
            B.v_scales = (uint16_t*) buffer(row / 32 * sizeof(uint16_t));
        } else {
            B.keys = (uint16_t*) buffer(row * sizeof(uint16_t));
            B.vals = (uint16_t*) buffer(row * sizeof(uint16_t));
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void GptOss::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * widest(), kv = n * n_kv_ * dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    vraw_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    freq_dev_ = fbuf(yarn_freq_.size());
    if (freq_dev_)   // the first pass only counts: there is no memory yet
        ck(cudaMemcpy(freq_dev_, yarn_freq_.data(), yarn_freq_.size() * sizeof(float), cudaMemcpyHostToDevice),
           "rope freqs");
}

// A sequence's rows are the full layers': a window layer's are part of its state (save_layer()).
size_t GptOss::card_row_bytes() const {
    const size_t row = (size_t) n_kv_ * dim_;
    size_t b = 0;
    for (const Block& B : blocks_) if (!B.swa) b += opt_.kv_format != KvFmt::kFp16 ? 2 * (row + row / 32 * 2) : 2 * row * 2;
    return b;
}

template<class F>
static void cache_spans(bool q8, uint16_t* k, uint16_t* v, int8_t* kc, int8_t* vc,
                        uint16_t* ks, uint16_t* vs, size_t width, F&& f) {
    if (q8) {
        f((uint8_t*) kc, width); f((uint8_t*) ks, width / 32 * 2);
        f((uint8_t*) vc, width); f((uint8_t*) vs, width / 32 * 2);
    } else { f((uint8_t*) k, width * 2); f((uint8_t*) v, width * 2); }
}

void GptOss::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first > opt_.max_context - n) throw std::logic_error("save_rows: outside context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) if (!B.swa) {
        cache_spans(opt_.kv_format != KvFmt::kFp16, B.keys, B.vals, B.k_codes, B.v_codes, B.k_scales, B.v_scales,
                    (size_t) n_kv_ * dim_, [&](uint8_t* p, size_t row) {
            ck(cudaMemcpy(dst, p + (size_t) first * row, (size_t) n * row, cudaMemcpyDeviceToHost), "save rows");
            dst += (size_t) n * row;
        });
    }
}

void GptOss::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first > from + count - n || first > opt_.max_context - n)
        throw std::logic_error("load_rows: outside saved context");
    cover(first + n); ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) if (!B.swa) {
        cache_spans(opt_.kv_format != KvFmt::kFp16, B.keys, B.vals, B.k_codes, B.v_codes, B.k_scales, B.v_scales,
                    (size_t) n_kv_ * dim_, [&](uint8_t* p, size_t row) {
            ck(cudaMemcpy(p + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row,
                          cudaMemcpyHostToDevice), "load rows");
            src += (size_t) count * row;
        });
    }
}

template<class F>
void GptOss::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m); p += m;
    }
}

size_t GptOss::extra_bytes() const {
    const size_t width = (size_t) n_kv_ * dim_;
    size_t b = 0;
    for (const Block& B : blocks_) if (B.swa)
        b += (size_t) keep_ * (opt_.kv_format != KvFmt::kFp16 ? 2 * (width + width / 32 * 2) : 4 * width);
    return b;
}

void GptOss::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    dst += state_n_ * sizeof(float);
    for (int j = 0; j <= il; ++j) {
        const Block& B = blocks_[(size_t) j]; if (!B.swa) continue;
        cache_spans(opt_.kv_format != KvFmt::kFp16, B.keys, B.vals, B.k_codes, B.v_codes, B.k_scales, B.v_scales,
                    (size_t) n_kv_ * dim_, [&](uint8_t* p, size_t row) {
            if (j == il) {
                std::memset(dst, 0, (size_t) keep_ * row);
                window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                    ck(cudaMemcpy(dst + slot * row, p + at * row, m * row, cudaMemcpyDeviceToHost), "save state");
                });
            }
            dst += (size_t) keep_ * row;
        });
    }
}

void GptOss::load_extra(const uint8_t* src, int64_t pos) {
    for (const Block& B : blocks_) if (B.swa) {
        cache_spans(opt_.kv_format != KvFmt::kFp16, B.keys, B.vals, B.k_codes, B.v_codes, B.k_scales, B.v_scales,
                    (size_t) n_kv_ * dim_, [&](uint8_t* p, size_t row) {
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy(p + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice), "load state");
            }); src += (size_t) keep_ * row;
        });
    }
}

int GptOss::head_key(int il, int n) const { return swa_[(size_t) il] ? 0 : room_key(n); }

void GptOss::store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n) {
    const int width = n_kv_ * dim_;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// Queries, keys and values have their biases, queries and keys turn by YaRN's rotation; the scores are scaled by
// 1 / sqrt(head width), and each head's sink is one more weight in the softmax that has no value.
void GptOss::attn(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin(B.wq, xn_, qraw_, n, xq_);
    rope_bias(qraw_, fp(B.bq), q_, dim_, n_head_, n_rot_, theta_scale_, freq_dev_, mscale_, pos, n, st_, at);
    lin(B.wk, xn_, kraw_, n, xq_);
    rope_bias(kraw_, fp(B.bk), k_, dim_, n_kv_, n_rot_, theta_scale_, freq_dev_, mscale_, pos, n, st_, at);
    lin(B.wv, xn_, vraw_, n, xq_);
    rope_bias(vraw_, fp(B.bv), v_, dim_, n_kv_, 0, 1.f, nullptr, 1.f, pos, n, st_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    if (q8) {
        const auto write = [&](const float* x, int8_t* c, uint16_t* sc) {
            if (at) return store_q8_at(x, c, sc, n_kv_ * dim_, at, n, B.swa ? ring_ : 0, st_);
            if (!B.swa) return store_q8(x, c, sc, n_kv_ * dim_, pos, n, st_);
            const int head = std::min(n, ring_ - (int) (pos % ring_));
            store_q8(x, c, sc, n_kv_ * dim_, pos % ring_, head, st_);
            if (head < n) store_q8(x + (size_t) head * n_kv_ * dim_, c, sc, n_kv_ * dim_, 0, n - head, st_);
        };
        write(k_, B.k_codes, B.k_scales); write(v_, B.v_codes, B.v_scales);
    } else { store(k_, B.keys, B, pos, n); store(v_, B.vals, B, pos, n); }
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    // The kernels that handle the sink for any number of tokens; a prompt is not sent through cuBLAS yet.
    attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, dim_, n_head_, n_kv_, 1.f / std::sqrt((float) dim_), st_,
           window, ring, false, at, !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n), 0,
           q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{}, q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{}, fp(B.sinks));
    lin(B.wo, att_, ao_, n);
    rope_bias(ao_, fp(B.bo), ao_, n_embd_, 1, 0, 1.f, nullptr, 1.f, 0, n, st_);
}

// A layer from its input to its router's copy: r_ holds the residual with the attention added, normed into xn_
// for the router and the experts.
void GptOss::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int all = n * n_embd_;
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::gemma_native_quantize_q8_1(xn_, xq_, all, 1, st_);
    attn(B, pos_, n);
    axpy(ao_, 1.f, r_, all, true, st_);
    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::gemma_native_quantize_q8_1(xn_, xq_, all, 1, st_);
    if (!batched(L, n)) moe_router(L, n);
}

// The router's scores plus its bias.
void GptOss::router_scores(const Layer& L, int n) {
    lin(L.gate_inp, xn_, rl_, n, xq_);
    rope_bias(rl_, L.b_router, rl_, n_expert_, 1, 0, 1.f, nullptr, 1.f, 0, n, st_);
}

// No expert is shared: the routed experts' sum starts from nothing.
void GptOss::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

void GptOss::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void GptOss::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: the router's scores with their bias, the n_used largest, and a softmax
// over those (the same as a softmax over every expert with the chosen ones' weights made to sum to one).
void GptOss::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    for (int e = 0; e < n_expert_; ++e) probs[e] = logits[e];
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = std::exp(probs[best] - mx);
        sum += w[k];
        probs[best] = -INFINITY;
    }
    for (int k = 0; k < n_used_; ++k) w[k] /= sum;
}

bool GptOss::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, true, 0.f, ids, w, st_);
    return true;
}

}  // namespace strata::gemma4
