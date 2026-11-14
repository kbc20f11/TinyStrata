// Initial causal Qwen3.6 driver for the optional Windows SYCL backend.
// Model/layer semantics follow src/qwen35/q36_engine.cu (Strata/TinyStrata,
// MIT; LICENSE and NOTICE). Optional arithmetic/MTP candidates are explicit;
// compressed weights and FP32 recurrent/KV state are retained. No image route.
#include "strata/intel/weight_store.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/decode_dp4a.hpp"
#include "strata/intel/q8_xmx.hpp"
#include "strata/intel/kv_q8.hpp"
#include "strata/intel/execution_request.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include "strata/intel/prefill_gemm.hpp"
#endif
#ifdef TINYSTRATA_INTEL_NPU
#include "strata/intel/npu.hpp"
#endif
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include "strata/common/command_inbox.hpp"
#include "strata/qwen35/speculative_sampling.hpp"
#include "strata/intel/speculative_distribution.hpp"
#include "strata/intel/sampling_policy.hpp"
#include "strata/intel/draft_vocabulary.hpp"
#include <functional>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <random>
#include <optional>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <stdexcept>

namespace strata::intel {
namespace {
namespace graph_api = sycl::ext::oneapi::experimental;
struct LayerState { float* recurrent = nullptr; float* convolution = nullptr; float* key = nullptr; float* value = nullptr; Q8Kv q8; };
struct VerifyInputs { float *qkv = nullptr, *conv = nullptr, *alpha = nullptr, *beta = nullptr; };
class Engine {
    sycl::queue q_;
    std::unique_ptr<WeightStore> weights_;
    std::unique_ptr<DecodeDp4a> dp4a_;
#ifdef TINYSTRATA_INTEL_MKL
    std::unique_ptr<PrefillGemm> gemm_;
#endif
#ifdef TINYSTRATA_INTEL_NPU
    std::unique_ptr<NpuProjection> npu_;
#endif
    std::vector<void*> allocations_;
    std::vector<LayerState> layers_, checkpoints_;
    std::vector<VerifyInputs> verify_inputs_;
    LayerState mtp_state_;
    int mtp_depth_ = 0, draft_vocab_limit_ = 0;
    int32_t* draft_vocab_ids_ = nullptr;
    std::vector<int32_t> draft_vocab_host_;
    uint64_t draft_vocab_updates_ = 0;
    double draft_vocab_prepare_ms_ = 0;
    uint64_t mtp_windows_ = 0, mtp_offered_ = 0, mtp_accepted_ = 0;
    double mtp_draft_ms_ = 0, mtp_verify_ms_ = 0, mtp_commit_ms_ = 0;
    bool verifying_ = false;
    float *mtp_cat_ = nullptr, *mtp_main_raw_ = nullptr, *mtp_main_norm_ = nullptr, *mtp_seeds_ = nullptr;
    float *mtp_norm_ = nullptr, *mtp_last_norm_ = nullptr, *checkpoint_norm_ = nullptr, *mtp_logits_ = nullptr;
    float *verify_logits_ = nullptr, *verify_raw_ = nullptr, *verify_norm_ = nullptr;
    bool kv_int8_ = false;
    int32_t* kv_error_ = nullptr;
    float *kv_float_key_ = nullptr, *kv_float_value_ = nullptr;
    std::vector<int> mtp_pending_;
    float* checkpoint_hidden_ = nullptr;
    int n_, hk_, hv_, heads_, kv_heads_, experts_, picks_, ff_, shared_ff_, context_, n_rot_;
    float epsilon_, rope_base_;
    float *x_, *h_, *mixer_, *qkv_, *z_, *alpha_, *beta_, *conv_, *recurrent_output_, *normalized_, *router_, *expert_weights_;
    float *expert_gate_, *expert_up_, *expert_hidden_, *expert_down_, *shared_gate_, *shared_up_, *shared_hidden_, *shared_down_, *shared_scalar_;
    float *query_, *query_gate_, *key_, *value_, *scores_, *attention_partial_, *attention_, *logits_;
    int32_t* expert_ids_;
    int32_t *expert_counts_, *expert_slots_, *tile_owners_, *tile_offsets_, *tile_count_;
    int32_t *token_input_ = nullptr, *position_input_ = nullptr;
    int graph_extent_ = 128;
    bool graph_enabled_ = true;
    bool recording_ = false, profiling_ = false, prefill_xmx_ = false, verify_xmx_ = false, prefill_int8_xmx_ = false;
    bool verify_mkl_attention_ = false, dense_split_k_ = false;
    bool device_state_ = false;
    bool verify_graph_enabled_ = false;
    int recording_extent_ = 128;
    std::string profile_phase_;
    std::vector<std::pair<std::string, sycl::event>> profile_events_;
    std::map<std::string, double> profile_ms_;
    void remember(const std::string& name, const sycl::event& event) {
        if (profiling_) profile_events_.emplace_back(profile_phase_ + name, event);
    }
    template<class Range, class Kernel> sycl::event dispatch(Range range, Kernel kernel) {
        auto event = q_.parallel_for(range, kernel);
        remember("elementwise_attention", event);
        return event;
    }
    void collect_profile() {
        for (const auto& [name, event] : profile_events_) {
            const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            profile_ms_[name] += double(end - start) * 1e-6;
        }
        profile_events_.clear();
    }
    std::optional<graph_api::command_graph<graph_api::graph_state::executable>> decode_graph_;
    using ExecutableGraph = graph_api::command_graph<graph_api::graph_state::executable>;
    std::map<std::pair<int, int>, std::unique_ptr<ExecutableGraph>> verify_graphs_;
    static constexpr int max_chunk = 1024, verify_capacity = 8;
    int chunk_ = 128, last_hidden_row_ = 0;
    int position_ = 0;
    int terminal_token_ = -1;
    bool logits_valid_ = false;
    void free_allocations() noexcept {
        for (void* allocation : allocations_) {
            try { sycl::free(allocation, q_); }
            catch (const std::exception& error) { std::fprintf(stderr, "Intel state release failed: %s\n", error.what()); }
        }
        allocations_.clear();
    }
    template<class T> T* alloc(size_t count, bool cpu_visible = false) {
        T* result = device_state_ && !cpu_visible ? sycl::malloc_device<T>(count, q_) : sycl::malloc_shared<T>(count, q_);
        if (!result) throw std::runtime_error("Intel activation/state allocation refused");
        try { allocations_.push_back(result); } catch (...) { sycl::free(result, q_); throw; }
        q_.memset(result, 0, count * sizeof(T));
        return result;
    }
    void allocate_kv(LayerState& state) {
        const size_t count = size_t(kv_heads_) * context_ * 256;
        if (kv_int8_) state.q8 = {alloc<int8_t>(count), alloc<int8_t>(count), alloc<sycl::half>(count / 32),
            alloc<sycl::half>(count / 32), context_, kv_heads_};
        else { state.key = alloc<float>(count); state.value = alloc<float>(count); }
    }
    void check_numeric_flags() const {
        if (dp4a_ && !dp4a_->valid()) throw std::runtime_error("Non-finite Intel Q8 activation");
        if (kv_error_ && kv_error_[0]) throw std::runtime_error("Invalid/non-finite Intel INT8 KV input");
    }
    const ResidentTensor& tensor(const std::string& name) const {
        const auto it = weights_->tensors().find(name);
        if (it == weights_->tensors().end()) throw std::runtime_error("Missing Qwen tensor: " + name);
        return it->second;
    }
    const float* f32(const std::string& name) const {
        const auto& t = tensor(name);
        if (t.info.type != 0) throw std::runtime_error("Initial Qwen Intel scalar tensors must be F32: " + name);
        return reinterpret_cast<const float*>(t.data);
    }
    double metadata_number(const std::string& name, double fallback) const {
        const auto it = weights_->metadata().find("qwen35moe." + name);
        return it == weights_->metadata().end() ? fallback : it->second.num();
    }
    void matrix(const std::string& name, const float* input, float* output, int tokens = 1) {
        const auto& t = tensor(name);
        const int width = int(t.info.shape[0]);
        const int rows = t.info.shape.size() < 2 ? 1 : int(t.info.shape[1]);
        if (t.layout == WeightLayout::Q6Interleaved16) {
            if (!dp4a_) throw std::runtime_error("Interleaved vocabulary requires the Q8 input engine");
            if (tokens <= verify_capacity) {
                const bool split = dense_split_k_ && rows == 2048 && width == 4096;
                sycl::event compute;
                const auto event = dp4a_->dense_interleaved_q6(t.data, input, output, rows, width, tokens, split, profiling_ && split ? &compute : nullptr);
                if (profiling_ && split) remember("splitk_xmx_" + name, compute);
                if (profiling_) { remember("q8_input_quantization", dp4a_->quantization_event()); remember("interleaved_xmx_" + name, event); }
            } else {
#ifdef TINYSTRATA_INTEL_MKL
                if (!gemm_) throw std::runtime_error("Interleaved dense prefill needs oneMKL");
                remember("mkl_interleaved_" + name, gemm_->dense_interleaved_q6(t.data, input, output, rows, width, tokens));
#else
                throw std::runtime_error("Interleaved dense prefill needs a oneMKL build");
#endif
            }
            return;
        }
        if (dp4a_ && tokens <= verify_capacity && DecodeDp4a::supports(int(t.info.type))) {
            const auto event = dp4a_->dense(int(t.info.type), t.data, input, output, rows, width, tokens);
            if (profiling_) {
                remember("q8_input_quantization", dp4a_->quantization_event());
                remember("dp4a_" + (name.rfind("blk.", 0) == 0 ? name.substr(name.find('.', 4) + 1) : name), event);
            }
            return;
        }
#ifdef TINYSTRATA_INTEL_NPU
        if (npu_ && tokens >= 128 && t.info.type != 0) {
            remember("npu_output_copy", npu_->run(int(t.info.type), t.data, input, output, rows, width, tokens));
            return;
        }
#endif
#ifdef TINYSTRATA_INTEL_MKL
        if (gemm_ && tokens >= 16 && t.info.type != 0) {
            remember("mkl_dense", gemm_->dense(int(t.info.type), t.data, input, output, rows, width, tokens));
            return;
        }
#endif
        const auto event = prefill_xmx_ && tokens >= 16 && t.info.type != 0
            ? quantized_matmul_xmx(q_, int(t.info.type), t.data, input, output, rows, width, tokens)
            : quantized_matmul(q_, int(t.info.type), t.data, input, output, rows, width, tokens);
        remember(name.rfind("blk.", 0) == 0 ? name.substr(name.find('.', 4) + 1) : name, event);
    }
    void add(float* destination, const float* source, int count) {
        dispatch(sycl::range<1>(size_t(count)), [=](sycl::id<1> i) { destination[i] += source[i]; });
    }
    void swiglu(const float* gate, const float* up, float* output, int count) {
        dispatch(sycl::range<1>(size_t(count)), [=](sycl::id<1> i) {
            output[i] = gate[i] / (1.f + sycl::exp(-gate[i])) * up[i];
        });
    }
    sycl::event recurrent_state(float* state, const float* qkv, const float* alpha, const float* beta, float* output, int tokens) {
        return gdn_recurrence_tiled(q_, state, qkv, alpha, beta, output, hk_, hv_, tokens);
    }
    void gdn(const std::string& prefix, const LayerState& state, int tokens, size_t layer) {
        matrix(prefix + "attn_qkv.weight", h_, qkv_, tokens);
        matrix(prefix + "attn_gate.weight", h_, z_, tokens);
        matrix(prefix + "ssm_alpha.weight", h_, alpha_, tokens);
        matrix(prefix + "ssm_beta.weight", h_, beta_, tokens);
        const float *dt = f32(prefix + "ssm_dt.bias"), *ssm_a = f32(prefix + "ssm_a");
        float *alpha = alpha_, *beta = beta_;
        const int hv = hv_, hk = hk_, channels = (2 * hk + hv) * 128;
        dispatch(sycl::range<1>(size_t(tokens) * hv), [=](sycl::id<1> i) {
            const float value = alpha[i] + dt[i[0] % hv];
            alpha[i] = (value > 20.f ? value : sycl::log1p(sycl::exp(value))) * ssm_a[i[0] % hv];
            beta[i] = 1.f / (1.f + sycl::exp(-beta[i]));
        });
        if (verifying_) {
            const auto& saved = verify_inputs_[layer];
            q_.memcpy(saved.qkv, qkv_, size_t(tokens) * channels * sizeof(float));
            q_.memcpy(saved.alpha, alpha_, size_t(tokens) * hv * sizeof(float));
            q_.memcpy(saved.beta, beta_, size_t(tokens) * hv * sizeof(float));
        }
        float *history = state.convolution, *output = conv_;
        const float *input = qkv_, *weight = f32(prefix + "ssm_conv1d.weight");
        const float epsilon = epsilon_;
        dispatch(sycl::nd_range<1>(size_t(channels), 128), [=](sycl::nd_item<1> it) {
            const size_t channel = it.get_global_linear_id();
            for (int token = 0; token < tokens; ++token) {
            const float current = input[size_t(token) * channels + channel];
            float value = history[channel * 3] * weight[channel * 4] + history[channel * 3 + 1] * weight[channel * 4 + 1]
                + history[channel * 3 + 2] * weight[channel * 4 + 2] + current * weight[channel * 4 + 3];
            history[channel * 3] = history[channel * 3 + 1];
            history[channel * 3 + 1] = history[channel * 3 + 2];
            history[channel * 3 + 2] = current;
            value = value / (1.f + sycl::exp(-value));
            if (it.get_group_linear_id() < size_t(2 * hk)) {
                const float sum = sycl::reduce_over_group(it.get_group(), value * value, sycl::plus<float>());
                value *= sycl::rsqrt(sum + epsilon);
            }
            output[size_t(token) * channels + channel] = value;
            }
        });
        if (verifying_) q_.memcpy(verify_inputs_[layer].conv, conv_, size_t(tokens) * channels * sizeof(float));
        remember("gdn_recurrence_tiled",
            recurrent_state(state.recurrent, conv_, alpha_, beta_, recurrent_output_, tokens));
        remember("rms_norm", rms_norm(q_, recurrent_output_, f32(prefix + "ssm_norm.weight"), normalized_, tokens * hv, 128, epsilon_));
        float* normalized = normalized_;
        const float* z = z_;
        dispatch(sycl::range<1>(size_t(tokens) * hv * 128), [=](sycl::id<1> i) {
            normalized[i] *= z[i] / (1.f + sycl::exp(-z[i]));
        });
        matrix(prefix + "ssm_out.weight", normalized_, mixer_, tokens);
    }
    void attention_row(const LayerState& state, int token, int extent) {
        sycl::event stages[3];
        const size_t offset = size_t(token) * heads_ * 256;
        const auto event = kv_int8_ ? decode_attention_q8(q_, query_ + offset, query_gate_ + offset, state.q8, scores_,
            attention_partial_, attention_ + offset, heads_, position_input_ + token, extent, profiling_ ? stages : nullptr)
            : decode_attention(q_, query_ + offset, query_gate_ + offset, state.key, state.value, scores_,
                attention_partial_, attention_ + offset, heads_, kv_heads_, context_, position_input_ + token, extent, profiling_ ? stages : nullptr);
        remember("decode_attention_join", event);
        if (profiling_) {
            remember("decode_attention_qk", stages[0]); remember("decode_attention_softmax", stages[1]);
            remember("decode_attention_pv", stages[2]);
        }
    }
    void attention(const std::string& prefix, const LayerState& state, int tokens) {
        matrix(prefix + "attn_q.weight", h_, qkv_, tokens);
        matrix(prefix + "attn_k.weight", h_, key_, tokens);
        matrix(prefix + "attn_v.weight", h_, value_, tokens);
        const int heads = heads_, kv_heads = kv_heads_, context = context_, position = position_, n_rot = n_rot_;
        const int extent = recording_ ? recording_extent_ : position + tokens;
        const int32_t* position_input = position_input_;
        const float base = rope_base_, epsilon = epsilon_;
        const float *qw = f32(prefix + "attn_q_norm.weight"), *kw = f32(prefix + "attn_k_norm.weight");
        const float *packed = qkv_, *keys = key_, *values = value_;
        float *queries = query_, *gates = query_gate_, *cache_k = state.key, *cache_v = state.value, *rotated_keys = key_;
        const bool quantized_kv = kv_int8_;
        q_.submit([&](sycl::handler& handler) {
            sycl::local_accessor<float, 1> normalized(256, handler);
            handler.parallel_for(sycl::nd_range<1>(size_t(tokens) * (heads + kv_heads) * 256, 256), [=](sycl::nd_item<1> it) {
                const size_t entry = it.get_group_linear_id(), token = entry / (heads + kv_heads);
                const size_t head = entry % (heads + kv_heads), col = it.get_local_linear_id();
                const bool query = head < size_t(heads);
                const float* input = query ? packed + (token * heads + head) * 512 : keys + (token * kv_heads + head - heads) * 256;
                const float value = input[col];
                const float sum = sycl::reduce_over_group(it.get_group(), value * value, sycl::plus<float>());
                normalized[col] = value * sycl::rsqrt(sum / 256.f + epsilon) * (query ? qw[col] : kw[col]);
                sycl::group_barrier(it.get_group());
                float result = normalized[col];
                const int half = n_rot / 2;
                if (col < size_t(n_rot)) {
                    const size_t pair = col < size_t(half) ? col : col - half;
                    const float theta = float(position_input[0] + token) * sycl::pow(base, -2.f * float(pair) / float(n_rot));
                    const float sine = sycl::sin(theta), cosine = sycl::cos(theta);
                    result = col < size_t(half) ? normalized[col] * cosine - normalized[col + half] * sine
                                              : normalized[col - half] * sine + normalized[col] * cosine;
                }
                if (query) {
                    queries[(token * heads + head) * 256 + col] = result;
                    gates[(token * heads + head) * 256 + col] = packed[(token * heads + head) * 512 + 256 + col];
                } else {
                    const size_t key_head = head - heads;
                    const size_t index = (key_head * context + position_input[0] + token) * 256 + col;
                    if (quantized_kv) rotated_keys[(token * kv_heads + key_head) * 256 + col] = result;
                    else {
                        cache_k[index] = result;
                        cache_v[index] = values[(token * kv_heads + key_head) * 256 + col];
                    }
                }
            });
        });
        const float *visible_keys = state.key, *visible_values = state.value;
        bool small_gemm_attention = false;
#ifdef TINYSTRATA_INTEL_MKL
        small_gemm_attention = verify_mkl_attention_ && gemm_ && tokens > 1 && tokens <= verify_capacity && position >= 512;
#endif
        if (kv_int8_) {
            remember("store_int8_kv", store_q8_kv(q_, key_, value_, state.q8, position_input_, tokens, kv_error_));
            if (tokens > verify_capacity || small_gemm_attention) {
                remember("dequantize_int8_kv", dequantize_q8_kv(q_, state.q8, kv_float_key_, kv_float_value_, position + tokens));
                visible_keys = kv_float_key_; visible_values = kv_float_value_;
            }
        }
        if (tokens > 1) {
#ifdef TINYSTRATA_INTEL_MKL
            if (small_gemm_attention) {
                const int first = position + tokens, end = std::min(context, (first + 255) / 256 * 256), tail = end - first;
                float* keys = kv_int8_ ? kv_float_key_ : state.key;
                float* values = kv_int8_ ? kv_float_value_ : state.value;
                if (tail) remember("clear_grouped_attention_padding", q_.parallel_for(sycl::range<1>(size_t(kv_heads) * tail * 256), [=](sycl::id<1> id) {
                    const size_t cell = id[0] / 256, col = id[0] % 256;
                    const size_t index = ((cell / tail) * context + first + cell % tail) * 256 + col;
                    keys[index] = values[index] = 0.f;
                }));
            }
            if (small_gemm_attention) remember("mkl_grouped_attention", gemm_->attention_small(query_, query_gate_, visible_keys, visible_values,
                attention_, heads, kv_heads, context, position, tokens));
            else
#endif
            if (tokens <= verify_capacity && (kv_int8_ || position >= 512)) {
                // A short verification window needs parallel work across the
                // existing KV prefix, rather than one serial tiled walk/head.
                if (kv_int8_) {
                    sycl::event stages[3];
                    remember("decode_attention_join", decode_attention_q8(q_, query_, query_gate_, state.q8, scores_,
                        attention_partial_, attention_, heads_, position_input_, extent, profiling_ ? stages : nullptr, tokens));
                    if (profiling_) {
                        remember("decode_attention_qk", stages[0]); remember("decode_attention_softmax", stages[1]);
                        remember("decode_attention_pv", stages[2]);
                    }
                } else for (int token = 0; token < tokens; ++token) attention_row(state, token, recording_ ? recording_extent_ : position + token + 1);
            } else {
#ifdef TINYSTRATA_INTEL_MKL
            if (gemm_ && tokens >= 1024) remember("mkl_attention_gate", gemm_->attention(query_, query_gate_, visible_keys, visible_values,
                attention_, heads, kv_heads, context, position, tokens));
            else
#endif
            remember("causal_attention", causal_attention(q_, query_, query_gate_, visible_keys, visible_values,
                attention_, heads, kv_heads, context, position, tokens));
            }
            matrix(prefix + "attn_output.weight", attention_, mixer_, tokens);
            return;
        }
        attention_row(state, 0, extent);
        matrix(prefix + "attn_output.weight", attention_, mixer_, tokens);
    }
    void moe(const std::string& prefix, int tokens) {
        matrix(prefix + "ffn_gate_inp.weight", h_, router_, tokens);
        matrix(prefix + "ffn_gate_inp_shexp.weight", h_, shared_scalar_, tokens);
        const int experts = experts_, picks = picks_, ff = ff_, n = n_;
        const float* logits = router_;
        int32_t* ids = expert_ids_;
        float* weights = expert_weights_;
        dispatch(sycl::nd_range<1>(size_t(tokens) * 128, 128), [=](sycl::nd_item<1> it) {
            const size_t token = it.get_group_linear_id(), lane = it.get_local_linear_id();
            const float* token_logits = logits + token * experts;
            int selected[8];
            for (int pick = 0; pick < picks; ++pick) {
                float best_value = -INFINITY;
                int best_id = experts;
                for (int expert = int(lane); expert < experts; expert += 128) {
                    bool used = false;
                    for (int earlier = 0; earlier < pick; ++earlier) used = used || selected[earlier] == expert;
                    if (!used && (token_logits[expert] > best_value || (token_logits[expert] == best_value && expert < best_id))) {
                        best_value = token_logits[expert]; best_id = expert;
                    }
                }
                const float maximum = sycl::reduce_over_group(it.get_group(), best_value, sycl::maximum<float>());
                selected[pick] = sycl::reduce_over_group(it.get_group(), best_value == maximum ? best_id : experts, sycl::minimum<int>());
                if (lane == 0) ids[token * picks + pick] = selected[pick];
            }
            if (lane == 0) {
                float total = 0;
                for (int pick = 0; pick < picks; ++pick) {
                    weights[token * picks + pick] = sycl::exp(token_logits[selected[pick]] - token_logits[selected[0]]);
                    total += weights[token * picks + pick];
                }
                for (int pick = 0; pick < picks; ++pick) weights[token * picks + pick] /= total;
            }
        });
        const auto& gate = tensor(prefix + "ffn_gate_exps.weight");
        const auto& up = tensor(prefix + "ffn_up_exps.weight");
        const auto& down = tensor(prefix + "ffn_down_exps.weight");
        const bool interleaved_experts = gate.layout == WeightLayout::ExpertInterleaved16;
        if ((up.layout == WeightLayout::ExpertInterleaved16) != interleaved_experts || (down.layout == WeightLayout::ExpertInterleaved16) != interleaved_experts)
            throw std::runtime_error("Mixed original/interleaved expert matrices");
        ExpertPlan plan{expert_counts_, expert_slots_, tile_owners_, tile_offsets_, tile_count_, experts, tokens, picks};
        const bool small_dp = dp4a_ && tokens > 1 && tokens <= verify_capacity && DecodeDp4a::supports(int(gate.info.type))
            && DecodeDp4a::supports(int(up.info.type)) && DecodeDp4a::supports(int(down.info.type));
        const bool prefill_dp = prefill_int8_xmx_ && tokens > verify_capacity && tokens <= 128
            && q8_xmx_supports(int(gate.info.type)) && q8_xmx_supports(int(up.info.type)) && q8_xmx_supports(int(down.info.type));
        const bool grouped = tokens >= 64 || small_dp || prefill_dp || interleaved_experts;
        plan.tile_width = interleaved_experts ? (tokens == 1 ? 1 : 4) : prefill_dp ? 4 : small_dp ? (verify_xmx_ ? 4 : 2) : prefill_xmx_ && tokens >= 64 ? 8 : 4;
        if (grouped) remember("expert_plan", plan_experts(q_, ids, plan));
        auto product = [&](const ResidentTensor& weight, const float* input, float* output, int rows, int width, bool separate) {
#ifdef TINYSTRATA_INTEL_MKL
            if (interleaved_experts && gemm_ && tokens >= 512) {
                remember("mkl_interleaved_experts", gemm_->selected(int(weight.info.type), weight.data, weight.bytes / experts,
                    plan, input, output, rows, width, separate, true));
                return;
            }
#endif
            if (interleaved_experts) {
                if (!dp4a_) throw std::runtime_error("Interleaved experts require Q8 input kernels");
                const auto event = dp4a_->grouped(int(weight.info.type), weight.data, plan, input, output, rows, width, separate, true);
                if (profiling_) { remember("q8_input_quantization", dp4a_->quantization_event()); remember("interleaved_xmx_experts", event); }
                return;
            }
            if (dp4a_ && (tokens <= verify_capacity || prefill_dp) && DecodeDp4a::supports(int(weight.info.type))) {
                const auto event = grouped ? dp4a_->grouped(int(weight.info.type), weight.data, plan, input, output, rows, width, separate)
                    : dp4a_->selected(int(weight.info.type), weight.data, ids, picks, input, output, rows, width, separate, tokens);
                if (profiling_) { remember("q8_input_quantization", dp4a_->quantization_event()); remember((verify_xmx_ || prefill_dp) && grouped && q8_xmx_supports(int(weight.info.type)) ? "int8_xmx_experts" : "dp4a_experts", event); }
                return;
            }
#ifdef TINYSTRATA_INTEL_MKL
            if (gemm_ && tokens >= 512) {
                remember("mkl_experts", gemm_->selected(int(weight.info.type), weight.data, weight.bytes / experts,
                    plan, input, output, rows, width, separate));
                return;
            }
#endif
            if (grouped && prefill_xmx_) remember("xmx_experts", selected_matmul_xmx(q_, int(weight.info.type), weight.data, plan, input, output, rows, width, separate));
            else if (grouped) remember("grouped_experts", selected_matmul(q_, int(weight.info.type), weight.data, plan, input, output, rows, width, separate));
            else remember("selected_matvec", selected_matvec(q_, int(weight.info.type), weight.data, ids, picks, input, output, rows, width, separate, tokens));
        };
        product(gate, h_, expert_gate_, ff, n, false);
        product(up, h_, expert_up_, ff, n, false);
        swiglu(expert_gate_, expert_up_, expert_hidden_, tokens * picks * ff);
        product(down, expert_hidden_, expert_down_, n, ff, true);
        matrix(prefix + "ffn_gate_shexp.weight", h_, shared_gate_, tokens);
        matrix(prefix + "ffn_up_shexp.weight", h_, shared_up_, tokens);
        swiglu(shared_gate_, shared_up_, shared_hidden_, tokens * shared_ff_);
        matrix(prefix + "ffn_down_shexp.weight", shared_hidden_, shared_down_, tokens);
        const float *routed = expert_down_, *shared = shared_down_, *scalar = shared_scalar_;
        float* x = x_;
        dispatch(sycl::range<1>(size_t(tokens) * n), [=](sycl::id<1> i) {
            const size_t token = i[0] / n, col = i[0] % n;
            float sum = weights[token * picks] * routed[token * picks * n + col];
            for (int pick = 1; pick < picks; ++pick) sum = sycl::fma(weights[token * picks + pick], routed[(token * picks + pick) * n + col], sum);
            x[i] += sum + shared[i] / (1.f + sycl::exp(-scalar[token]));
        });
    }
    bool enqueue(int count, bool output_logits, const std::atomic<bool>* cancelled = nullptr) {
        const auto& embedding = tensor("token_embd.weight");
        remember("embedding_lookup", embedding_lookup(q_, int(embedding.info.type), embedding.data, token_input_, x_, n_, count));
        for (size_t layer = 0; layer < layers_.size(); ++layer) {
            if (cancelled && cancelled->load()) return false;
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            remember("rms_norm", rms_norm(q_, x_, f32(prefix + "attn_norm.weight"), h_, count, n_, epsilon_));
            if (layers_[layer].key || layers_[layer].q8.key) attention(prefix, layers_[layer], count); else gdn(prefix, layers_[layer], count, layer);
            add(x_, mixer_, count * n_);
            remember("rms_norm", rms_norm(q_, x_, f32(prefix + "post_attention_norm.weight"), h_, count, n_, epsilon_));
            moe(prefix, count);
        }
        if (cancelled && cancelled->load()) return false;
        if (output_logits) {
            const int rows = verifying_ ? count : 1;
            remember("rms_norm", rms_norm(q_, x_ + (verifying_ ? 0 : size_t(count - 1) * n_), f32("output_norm.weight"), h_, rows, n_, epsilon_));
            if (verifying_) {
                q_.memcpy(verify_raw_, x_, size_t(count) * n_ * sizeof(float));
                q_.memcpy(verify_norm_, h_, size_t(count) * n_ * sizeof(float));
            }
            matrix("output.weight", h_, verifying_ ? verify_logits_ : logits_, rows);
        }
        return true;
    }
    void checkpoint(bool restore, bool hidden = true) {
        for (size_t i = 0; i < layers_.size(); ++i) if (layers_[i].recurrent) {
            const auto& state = layers_[i]; const auto& saved = checkpoints_[i];
            q_.memcpy(restore ? state.recurrent : saved.recurrent, restore ? saved.recurrent : state.recurrent,
                size_t(hv_) * 128 * 128 * sizeof(float));
            q_.memcpy(restore ? state.convolution : saved.convolution, restore ? saved.convolution : state.convolution,
                size_t(2 * hk_ + hv_) * 128 * 3 * sizeof(float));
        }
        if (!hidden) return;
        if (restore) {
            q_.memcpy(x_, checkpoint_hidden_, size_t(n_) * sizeof(float));
            if (mtp_depth_) q_.memcpy(mtp_last_norm_, checkpoint_norm_, size_t(n_) * sizeof(float));
            last_hidden_row_ = 0;
            logits_valid_ = false;
        } else {
            q_.memcpy(checkpoint_hidden_, x_ + size_t(last_hidden_row_) * n_, size_t(n_) * sizeof(float));
            if (mtp_depth_) q_.memcpy(checkpoint_norm_, mtp_last_norm_, size_t(n_) * sizeof(float));
        }
    }
    void mtp_forward(const int* tokens, int count, const float* seeds, int position, bool head) {
        q_.wait_and_throw();
        const int main_position = position_;
        const auto previous_phase = profile_phase_;
        if (profiling_) profile_phase_ = head ? "draft." : "mtp_warm.";
        position_ = position;
        for (int t = 0; t < count; ++t) position_input_[t] = position + t;
        std::copy_n(tokens, count, token_input_);
        const std::string prefix = "blk.40.";
        const auto& embedding = tensor("token_embd.weight");
        embedding_lookup(q_, int(embedding.info.type), embedding.data, token_input_, x_, n_, count);
        rms_norm(q_, x_, f32(prefix + "nextn.enorm.weight"), h_, count, n_, epsilon_);
        float* cat = mtp_cat_; const float* normalized = h_; const int n = n_;
        q_.parallel_for(sycl::range<1>(size_t(count) * n), [=](sycl::id<1> i) {
            cat[(i[0] / n) * 2 * n + i[0] % n] = normalized[i];
        });
        rms_norm(q_, seeds, f32(prefix + "nextn.hnorm.weight"), h_, count, n_, epsilon_);
        q_.parallel_for(sycl::range<1>(size_t(count) * n), [=](sycl::id<1> i) {
            cat[(i[0] / n) * 2 * n + n + i[0] % n] = normalized[i];
        });
        matrix(prefix + "nextn.eh_proj.weight", mtp_cat_, x_, count);
        rms_norm(q_, x_, f32(prefix + "attn_norm.weight"), h_, count, n_, epsilon_);
        attention(prefix, mtp_state_, count);
        add(x_, mixer_, count * n_);
        rms_norm(q_, x_, f32(prefix + "post_attention_norm.weight"), h_, count, n_, epsilon_);
        moe(prefix, count);
        rms_norm(q_, x_, f32(prefix + "nextn.shared_head_norm.weight"), mtp_norm_, count, n_, epsilon_);
        if (head) {
            const float* normalized = mtp_norm_ + size_t(count - 1) * n_;
            if (draft_vocab_limit_) {
                const auto& head_weight = tensor("output.weight");
                if (draft_vocab_host_.empty() || !dp4a_ || head_weight.info.type != 14 || head_weight.layout != WeightLayout::Q6Interleaved16)
                    throw std::runtime_error("Selected draft vocabulary needs prepared IDs and an interleaved Q6 head");
                const auto event = dp4a_->selected_vocabulary_q6(head_weight.data, normalized, mtp_logits_, vocabulary(), n_,
                    draft_vocab_ids_, int(draft_vocab_host_.size()));
                remember("draft_vocab_q6", event);
                if (profiling_) remember("q8_input_quantization", dp4a_->quantization_event());
            } else matrix("output.weight", normalized, mtp_logits_);
        }
        q_.wait_and_throw();
        profile_phase_ = previous_phase;
        position_ = main_position; position_input_[0] = main_position;
        check_numeric_flags();
    }
    void repair_mtp() {
        if (mtp_pending_.empty()) return;
        const size_t bytes = size_t(last_hidden_row_ + 1) * n_ * sizeof(float);
        q_.memcpy(mtp_main_raw_, x_, bytes);
        mtp_forward(mtp_pending_.data(), int(mtp_pending_.size()), verify_norm_, position_ - int(mtp_pending_.size()), false);
        q_.memcpy(x_, mtp_main_raw_, bytes);
        mtp_pending_.clear();
    }
    void warm_mtp(const int* tokens, int count) {
        q_.memcpy(mtp_main_raw_, x_, size_t(count) * n_ * sizeof(float));
        rms_norm(q_, x_, f32("output_norm.weight"), mtp_main_norm_, count, n_, epsilon_);
        q_.memcpy(mtp_seeds_, mtp_last_norm_, size_t(n_) * sizeof(float));
        if (count > 1) q_.memcpy(mtp_seeds_ + n_, mtp_main_norm_, size_t(count - 1) * n_ * sizeof(float));
        mtp_forward(tokens, count, mtp_seeds_, position_, false);
        q_.memcpy(mtp_last_norm_, mtp_main_norm_ + size_t(count - 1) * n_, size_t(n_) * sizeof(float));
        q_.memcpy(x_, mtp_main_raw_, size_t(count) * n_ * sizeof(float));
        q_.wait_and_throw();
    }
    void commit_verify(int keep, int count) {
        if (keep < count) {
            checkpoint(true, false);
            const int hv = hv_, channels = (2 * hk_ + hv_) * 128;
            for (size_t i = 0; i < layers_.size(); ++i) if (layers_[i].recurrent) {
                float* history = layers_[i].convolution;
                const float* raw = verify_inputs_[i].qkv;
                q_.parallel_for(sycl::range<1>(size_t(channels)), [=](sycl::id<1> id) {
                    const size_t channel = id[0];
                    for (int t = 0; t < keep; ++t) {
                        history[channel * 3] = history[channel * 3 + 1];
                        history[channel * 3 + 1] = history[channel * 3 + 2];
                        history[channel * 3 + 2] = raw[size_t(t) * channels + channel];
                    }
                });
                const auto& saved = verify_inputs_[i];
                recurrent_state(layers_[i].recurrent, saved.conv, saved.alpha, saved.beta, recurrent_output_, keep);
            }
        }
        q_.memcpy(x_, verify_raw_ + size_t(keep - 1) * n_, size_t(n_) * sizeof(float));
        q_.memcpy(mtp_last_norm_, verify_norm_ + size_t(keep - 1) * n_, size_t(n_) * sizeof(float));
        q_.memcpy(logits_, verify_logits_ + size_t(keep - 1) * vocabulary(), size_t(vocabulary()) * sizeof(float));
        q_.wait_and_throw();
        last_hidden_row_ = 0; position_ += keep; logits_valid_ = true;
    }
    void prepare_graph() {
        q_.wait_and_throw();
        decode_graph_.reset();
        graph_api::command_graph<graph_api::graph_state::modifiable> graph(q_.get_context(), q_.get_device());
        bool recording = false;
        try {
            recording_ = true;
            recording_extent_ = graph_extent_;
            graph.begin_recording(q_);
            recording = true;
            enqueue(1, true);
            graph.end_recording(q_);
            recording = false;
            decode_graph_.emplace(graph.finalize());
            std::fprintf(stderr, "Intel decode graph=enabled\n");
        } catch (const std::exception& error) {
            if (recording) graph.end_recording(q_);
            graph_enabled_ = false;
            std::fprintf(stderr, "Intel decode graph=unavailable; eager fallback: %s\n", error.what());
        }
        recording_ = false;
    }
    ExecutableGraph* verification_graph(int count) {
        int extent = std::min(context_, 128);
        while (extent < position_ + count) extent = std::min(context_, extent * 2);
        const auto key = std::make_pair(count, extent);
        if (const auto it = verify_graphs_.find(key); it != verify_graphs_.end()) return it->second.get();
        q_.wait_and_throw();
        graph_api::command_graph<graph_api::graph_state::modifiable> graph(q_.get_context(), q_.get_device());
        bool recording = false;
        try {
            recording_ = true; recording_extent_ = extent;
            graph.begin_recording(q_); recording = true;
            enqueue(count, true);
            graph.end_recording(q_); recording = false;
            auto executable = std::make_unique<ExecutableGraph>(graph.finalize());
            // A bounded cache retains common counts/extents across prompt replay.
            if (verify_graphs_.size() >= 16) verify_graphs_.erase(verify_graphs_.begin());
            auto* result = executable.get(); verify_graphs_.emplace(key, std::move(executable));
            recording_ = false;
            std::fprintf(stderr, "Intel verify_graph=enabled tokens=%d extent=%d cached=%zu\n", count, extent, verify_graphs_.size());
            return result;
        } catch (const std::exception& error) {
            if (recording) graph.end_recording(q_);
            recording_ = false; verify_graph_enabled_ = false;
            std::fprintf(stderr, "Intel verify_graph=unavailable; eager fallback: %s\n", error.what());
            return nullptr;
        }
    }
public:
    Engine(const std::string& path, int context, unsigned device_index, WeightMemory memory = WeightMemory::Host, int chunk = 128, bool use_graph = true, bool profiling = false, bool prefill_xmx = false, bool prefill_npu = false, bool prefill_mkl = false, bool decode_dp4a = false, int mtp_depth = 0, bool verify_xmx = false, bool kv_int8 = false, bool prefill_int8_xmx = false, bool vocab_xmx = false, bool verify_mkl_attention = false, bool expert_layout = false, bool device_state = false, bool verify_graph = false, bool dense_layout = false, bool dense_split_k = false, int draft_vocab_limit = 0) :
        q_(select_device(device_index), [](sycl::exception_list errors) { for (const auto& error : errors) std::rethrow_exception(error); },
           profiling ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
                     : sycl::property_list{sycl::property::queue::in_order{}}), profiling_(profiling), prefill_xmx_(prefill_xmx), context_(context), chunk_(chunk) {
        graph_enabled_ = use_graph;
        kv_int8_ = kv_int8;
        verify_xmx_ = verify_xmx;
        verify_mkl_attention_ = verify_mkl_attention;
        device_state_ = device_state;
        dense_split_k_ = dense_split_k;
        verify_graph_enabled_ = verify_graph;
        if (device_state_) std::fprintf(stderr, "Intel state_memory=device cpu_control_and_logits=shared\n");
        prefill_int8_xmx_ = prefill_int8_xmx;
        mtp_depth_ = mtp_depth;
        draft_vocab_limit_ = draft_vocab_limit;
        graph_extent_ = std::min(context_, 128);
        if (chunk < 1 || chunk > max_chunk) throw std::runtime_error("Intel chunk must be in 1..1024");
        if (mtp_depth < 0 || mtp_depth >= verify_capacity || (mtp_depth && chunk < std::max(4, mtp_depth + 1)))
            throw std::runtime_error("Experimental Intel MTP needs depth 1..7 and chunk >= max(4, depth+1)");
        try {
#ifndef TINYSTRATA_INTEL_INT8_XMX
            if (verify_xmx) throw std::runtime_error("--verify-xmx requires TINYSTRATA_INTEL_INT8_XMX=ON");
            if (prefill_int8_xmx) throw std::runtime_error("--prefill-int8-xmx requires TINYSTRATA_INTEL_INT8_XMX=ON");
            if (vocab_xmx) throw std::runtime_error("--vocab-xmx requires TINYSTRATA_INTEL_INT8_XMX=ON");
            if (expert_layout) throw std::runtime_error("--expert-xmx-layout requires TINYSTRATA_INTEL_INT8_XMX=ON");
            if (dense_layout) throw std::runtime_error("--dense-xmx-layout requires TINYSTRATA_INTEL_INT8_XMX=ON");
#endif
            if (verify_xmx && !decode_dp4a) throw std::runtime_error("--verify-xmx requires --decode-dp4a");
            if (prefill_int8_xmx && !decode_dp4a) throw std::runtime_error("--prefill-int8-xmx requires --decode-dp4a");
            if (vocab_xmx && !decode_dp4a) throw std::runtime_error("--vocab-xmx requires --decode-dp4a");
            if (expert_layout && !decode_dp4a) throw std::runtime_error("--expert-xmx-layout requires --decode-dp4a");
            if (dense_split_k && !dense_layout) throw std::runtime_error("--dense-split-k requires --dense-xmx-layout");
            if (dense_layout && (!decode_dp4a || !prefill_mkl)) throw std::runtime_error("--dense-xmx-layout requires --decode-dp4a and --prefill-mkl");
#ifndef TINYSTRATA_INTEL_NPU
            if (prefill_npu) throw std::runtime_error("--prefill-npu requires a build with TINYSTRATA_INTEL_NPU=ON");
#endif
#ifndef TINYSTRATA_INTEL_MKL
            if (prefill_mkl || verify_mkl_attention) throw std::runtime_error("oneMKL attention/prefill requires TINYSTRATA_INTEL_MKL=ON");
#endif
            if (verify_graph && (!kv_int8 || !mtp_depth || !use_graph || profiling || verify_mkl_attention))
                throw std::runtime_error("--verify-graph requires INT8 KV, MTP, graphs enabled, and native attention without profiling");
            if (verify_mkl_attention && (!prefill_mkl || chunk < 64))
                throw std::runtime_error("--verify-mkl-attention requires --prefill-mkl and chunk >= 64");
            ExecutionRequest loading;
            if (draft_vocab_limit && (!mtp_depth || !vocab_xmx || draft_vocab_limit < 64 || draft_vocab_limit > 4096))
                throw std::runtime_error("--draft-vocab requires MTP, --vocab-xmx and a limit in 64..4096");
            weights_ = WeightStore::load(q_, path, memory, vocab_xmx, expert_layout, dense_layout, dense_split_k);
            if (expert_layout) std::fprintf(stderr, "Intel expert_layout=interleaved16 Q8_activations=1 single_weight_store=1\n");
            if (vocab_xmx) std::fprintf(stderr, "Intel vocabulary=interleaved_Q6_K XMX unchanged_quantization=1 single_weight_store=1\n");
            const auto& g = weights_->geometry();
            if (draft_vocab_limit_ > int(g.n_vocab)) throw std::runtime_error("Draft vocabulary limit exceeds the real vocabulary");
            if (mtp_depth && g.n_mtp_layers < 1) throw std::runtime_error("MTP requested but this GGUF has no draft block");
            if (g.n_embd != 2048 || g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 ||
                g.ssm_k_heads != 16 || g.ssm_v_heads != 32 || g.n_head != 16 || g.n_head_kv != 2 ||
                g.n_expert != 256 || g.n_expert_used != 8 || g.n_ff != 512 || g.n_ff_shexp != 512 ||
                g.full_attention_interval != 4 || g.n_layers != 40 || g.n_vocab <= 0 || g.n_vocab > 1000000 ||
                context < 1 || context > 32768)
                throw std::runtime_error("Initial Intel Qwen engine does not support this geometry/context");
            n_ = int(g.n_embd); hk_ = int(g.ssm_k_heads); hv_ = int(g.ssm_v_heads);
            heads_ = int(g.n_head); kv_heads_ = int(g.n_head_kv); experts_ = int(g.n_expert); picks_ = int(g.n_expert_used);
            ff_ = int(g.n_ff); shared_ff_ = int(g.n_ff_shexp);
            epsilon_ = float(metadata_number("attention.layer_norm_rms_epsilon", 1e-6));
            if (const auto it = weights_->metadata().find("tokenizer.ggml.eos_token_id"); it != weights_->metadata().end())
                terminal_token_ = int(it->second.u);
            rope_base_ = float(metadata_number("rope.freq_base", 1e7)); n_rot_ = int(metadata_number("rope.dimension_count", 64));
            if (n_rot_ <= 0 || n_rot_ > 256 || n_rot_ % 2 || !(epsilon_ > 0) || !(rope_base_ > 0))
                throw std::runtime_error("Invalid Qwen normalization/rotary settings");
            const uint64_t attention_layers = uint64_t(g.n_layers / g.full_attention_interval);
            const uint64_t recurrent_layers = uint64_t(g.n_layers) - attention_layers;
            const uint64_t cache_cell_bytes = kv_int8_ ? 256 * 2 + 8 * 2 * sizeof(sycl::half) : 256 * 2 * sizeof(float);
            const uint64_t cache_bytes = (attention_layers + (mtp_depth ? 1 : 0)) * uint64_t(kv_heads_) * context * cache_cell_bytes;
            const uint64_t state_bytes = recurrent_layers * (uint64_t(hv_) * 128 * 128 + uint64_t(2 * hk_ + hv_) * 128 * 3) * sizeof(float);
            const uint64_t scratch_reserve = uint64_t(chunk_) * 384 * 1024 + (64ull + (prefill_npu ? 96ull : 0ull) + (prefill_mkl ? 96ull : 0ull) + (mtp_depth ? 96ull : 0ull) + (verify_graph ? 128ull : 0ull)) * 1024 * 1024 + (prefill_mkl ? uint64_t(chunk_) * context * sizeof(float) : 0)
                + (dense_split_k ? 1024ull * 1024 : 0) + uint64_t(draft_vocab_limit_) * sizeof(int32_t)
                + (kv_int8_ ? uint64_t(kv_heads_) * context * 256 * 2 * sizeof(float)
                    + uint64_t(std::min(chunk_, verify_capacity)) * heads_ * (context + ((context + 255) / 256) * 256) * sizeof(float) : 0);
            if (cache_bytes + 2 * state_bytes + scratch_reserve > allocation_budget(memory_budget(inspect_device(q_.get_device()))))
                throw std::runtime_error("Intel recurrent/KV/scratch buffers exceed the remaining memory budget");
            x_ = alloc<float>(size_t(chunk_) * (n_)); h_ = alloc<float>(size_t(chunk_) * (n_)); mixer_ = alloc<float>(size_t(chunk_) * (n_));
            qkv_ = alloc<float>(size_t(chunk_) * (std::max((2 * hk_ + hv_) * 128, heads_ * 512)));
            z_ = alloc<float>(size_t(chunk_) * (hv_ * 128)); alpha_ = alloc<float>(size_t(chunk_) * (hv_)); beta_ = alloc<float>(size_t(chunk_) * (hv_));
            conv_ = alloc<float>(size_t(chunk_) * ((2 * hk_ + hv_) * 128)); recurrent_output_ = alloc<float>(size_t(chunk_) * (hv_ * 128)); normalized_ = alloc<float>(size_t(chunk_) * (hv_ * 128));
            router_ = alloc<float>(size_t(chunk_) * (experts_)); expert_ids_ = alloc<int32_t>(size_t(chunk_) * (picks_)); expert_weights_ = alloc<float>(size_t(chunk_) * (picks_));
            expert_gate_ = alloc<float>(size_t(chunk_) * (picks_ * ff_)); expert_up_ = alloc<float>(size_t(chunk_) * (picks_ * ff_)); expert_hidden_ = alloc<float>(size_t(chunk_) * (picks_ * ff_));
            expert_down_ = alloc<float>(size_t(chunk_) * (picks_ * n_)); shared_gate_ = alloc<float>(size_t(chunk_) * (shared_ff_)); shared_up_ = alloc<float>(size_t(chunk_) * (shared_ff_));
            shared_hidden_ = alloc<float>(size_t(chunk_) * (shared_ff_)); shared_down_ = alloc<float>(size_t(chunk_) * (n_)); shared_scalar_ = alloc<float>(size_t(chunk_) * (1));
            query_ = alloc<float>(size_t(chunk_) * (heads_ * 256)); query_gate_ = alloc<float>(size_t(chunk_) * (heads_ * 256));
            key_ = alloc<float>(size_t(chunk_) * (kv_heads_ * 256)); value_ = alloc<float>(size_t(chunk_) * (kv_heads_ * 256));
            const size_t attention_rows = size_t(heads_) * (kv_int8_ ? std::min(chunk_, verify_capacity) : 1);
            scores_ = alloc<float>(attention_rows * context);
            attention_partial_ = alloc<float>(attention_rows * ((context + 255) / 256) * 256); attention_ = alloc<float>(size_t(chunk_) * (heads_ * 256));
            logits_ = alloc<float>(size_t(g.n_vocab), true);
            expert_counts_ = alloc<int32_t>(experts_, true);
            expert_slots_ = alloc<int32_t>(experts_ * chunk_);
            const int max_tiles = (chunk_ * picks_ + 3) / 4 + experts_;
            tile_owners_ = alloc<int32_t>(max_tiles); tile_offsets_ = alloc<int32_t>(max_tiles); tile_count_ = alloc<int32_t>(1);
            token_input_ = alloc<int32_t>(chunk_, true);
            position_input_ = alloc<int32_t>(chunk_, true);
            checkpoint_hidden_ = alloc<float>(n_);
            if (kv_int8_) {
                kv_error_ = alloc<int32_t>(1, true);
                kv_float_key_ = alloc<float>(size_t(kv_heads_) * context * 256);
                kv_float_value_ = alloc<float>(size_t(kv_heads_) * context * 256);
            }
            layers_.resize(size_t(g.n_layers));
            checkpoints_.resize(layers_.size());
            if (mtp_depth_) {
                verify_inputs_.resize(layers_.size());
                mtp_cat_ = alloc<float>(size_t(chunk_) * 2 * n_);
                mtp_main_raw_ = alloc<float>(size_t(chunk_) * n_); mtp_main_norm_ = alloc<float>(size_t(chunk_) * n_);
                mtp_seeds_ = alloc<float>(size_t(chunk_) * n_); mtp_norm_ = alloc<float>(size_t(chunk_) * n_);
                mtp_last_norm_ = alloc<float>(n_); checkpoint_norm_ = alloc<float>(n_);
                mtp_logits_ = alloc<float>(size_t(draft_vocab_limit_ ? draft_vocab_limit_ : int(g.n_vocab)), true);
                if (draft_vocab_limit_) draft_vocab_ids_ = alloc<int32_t>(draft_vocab_limit_, true);
                verify_logits_ = alloc<float>(verify_capacity * size_t(g.n_vocab), true);
                verify_raw_ = alloc<float>(verify_capacity * n_); verify_norm_ = alloc<float>(verify_capacity * n_);
                allocate_kv(mtp_state_);
            }
            for (size_t layer = 0; layer < layers_.size(); ++layer) {
                if ((layer + 1) % size_t(g.full_attention_interval) == 0) {
                    allocate_kv(layers_[layer]);
                } else {
                    layers_[layer].recurrent = alloc<float>(size_t(hv_) * 128 * 128);
                    layers_[layer].convolution = alloc<float>(size_t(2 * hk_ + hv_) * 128 * 3);
                    checkpoints_[layer].recurrent = alloc<float>(size_t(hv_) * 128 * 128);
                    checkpoints_[layer].convolution = alloc<float>(size_t(2 * hk_ + hv_) * 128 * 3);
                    if (mtp_depth_) {
                        auto& saved = verify_inputs_[layer];
                        saved.qkv = alloc<float>(verify_capacity * size_t(2 * hk_ + hv_) * 128);
                        saved.conv = alloc<float>(verify_capacity * size_t(2 * hk_ + hv_) * 128);
                        saved.alpha = alloc<float>(verify_capacity * hv_); saved.beta = alloc<float>(verify_capacity * hv_);
                    }
                }
            }
            q_.wait_and_throw();
#ifdef TINYSTRATA_INTEL_NPU
            if (prefill_npu) npu_ = std::make_unique<NpuProjection>(q_);
#endif
#ifdef TINYSTRATA_INTEL_MKL
            if (prefill_mkl) gemm_ = std::make_unique<PrefillGemm>(q_, context_, chunk_);
#endif
            if (decode_dp4a) dp4a_ = std::make_unique<DecodeDp4a>(q_, verify_xmx || prefill_int8_xmx || expert_layout, dense_split_k);
            if (use_graph && !profiling) prepare_graph();
            const auto device = inspect_device(q_.get_device());
            std::fprintf(stderr, "Intel causal Qwen: %s, architecture=%s, context=%d, KV=%s, MTP=%d, images=off, weight_memory=%s\n",
                         device.name.c_str(), device.architecture.c_str(), context, kv_int8_ ? "INT8" : "FP32",
                         mtp_depth_, memory == WeightMemory::Host ? "host" : memory == WeightMemory::Shared ? "shared" : "device");
            print_memory("engine_ready", device);
        } catch (...) {
            try { q_.wait_and_throw(); } catch (...) {}
            decode_graph_.reset();
            verify_graphs_.clear();
            free_allocations();
            throw;
        }
    }
    ~Engine() {
        try { q_.wait_and_throw(); } catch (...) {}
        decode_graph_.reset();
        verify_graphs_.clear();
        for (const auto& [name, milliseconds] : profile_ms_) std::fprintf(stderr, "profile kernel=%s gpu_ms=%.6f\n", name.c_str(), milliseconds);
        free_allocations();
    }
    const float* step(int token, bool output_logits = true) { return step_many(&token, 1, output_logits); }
    int chunk() const { return chunk_; }
    DeviceInfo device_info() const { return inspect_device(q_.get_device()); }
    const float* step_many(const int* tokens, int count, bool output_logits = true, const std::atomic<bool>* cancelled = nullptr) {
        const auto& g = weights_->geometry();
        if (count < 1 || count > chunk_ || position_ + count > context_) throw std::runtime_error("Intel chunk/context limit reached");
        for (int i = 0; i < count; ++i)
            if (tokens[i] < 0 || tokens[i] >= g.n_vocab) throw std::runtime_error("Token ID outside vocabulary");
        if (cancelled && cancelled->load()) return nullptr;
        if (mtp_depth_) repair_mtp();
        if (cancelled) checkpoint(false);
        logits_valid_ = false;
        std::copy_n(tokens, count, token_input_);
        for (int t = 0; t < count; ++t) position_input_[t] = position_ + t;
        last_hidden_row_ = count - 1;
        if (graph_enabled_ && !profiling_ && count == 1 && output_logits) {
            int extent = std::min(context_, 128);
            while (extent <= position_) extent = std::min(context_, extent * 2);
            if (!decode_graph_ || extent != graph_extent_) { graph_extent_ = extent; prepare_graph(); }
        }
        bool complete = true;
        if (decode_graph_ && count == 1 && output_logits) q_.ext_oneapi_graph(*decode_graph_);
        else complete = enqueue(count, output_logits, cancelled);
        q_.wait_and_throw();
        if (mtp_depth_ && complete && !(cancelled && cancelled->load())) warm_mtp(tokens, count);
        collect_profile();
        check_numeric_flags();
        if (cancelled && (!complete || cancelled->load())) {
            checkpoint(true);
            q_.wait_and_throw();
            return nullptr;
        }
        if (output_logits) for (int64_t i = 0; i < g.n_vocab; ++i) if (!std::isfinite(logits_[i])) throw std::runtime_error("Non-finite Intel logits");
        logits_valid_ = output_logits;
        position_ += count;
        return logits_;
    }
    int vocabulary() const { return int(weights_->geometry().n_vocab); }
    int mtp_depth() const { return mtp_depth_; }
    const std::vector<int32_t>& draft_vocabulary() const { return draft_vocab_host_; }
    void prepare_draft_vocabulary(const float* target, const std::vector<int>& history) {
        if (!draft_vocab_limit_) return;
        const auto start = std::chrono::steady_clock::now();
        auto ids = make_draft_vocabulary(target, vocabulary(), draft_vocab_limit_, history);
        q_.wait_and_throw();
        std::copy(ids.begin(), ids.end(), draft_vocab_ids_);
        draft_vocab_host_ = std::move(ids);
        ++draft_vocab_updates_;
        draft_vocab_prepare_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    const char* kv_type() const { return kv_int8_ ? "int8" : "fp32"; }
    std::pair<uint64_t, uint64_t> speculation_counts() const { return {mtp_accepted_, mtp_offered_}; }
    struct SpecResult { std::vector<int> inputs, output; };
    template<class Select> SpecResult speculate(int anchor, int remaining, Select select, const std::atomic<bool>* cancelled = nullptr,
        const std::function<int(const float*)>& propose = {}) {
        SpecResult result;
        if (cancelled && cancelled->load()) return result;
        const int depth = std::min({mtp_depth_, remaining - 1, context_ - position_ - 1});
        if (depth <= 0) {
            if (const float* row = step_many(&anchor, 1, true, cancelled)) {
                result.inputs.push_back(anchor); result.output.push_back(select(row));
            }
            return result;
        }
        repair_mtp();
        const auto draft_start = std::chrono::steady_clock::now();
        q_.memcpy(mtp_main_raw_, x_, size_t(last_hidden_row_ + 1) * n_ * sizeof(float));
        q_.memcpy(mtp_seeds_, mtp_last_norm_, size_t(n_) * sizeof(float));
        std::vector<int> drafts;
        int input = anchor;
        for (int i = 0; i < depth; ++i) {
            mtp_forward(&input, 1, mtp_seeds_, position_ + i, true);
            const int draft_rows = draft_vocab_limit_ ? int(draft_vocab_host_.size()) : vocabulary();
            if (!draft_rows) throw std::runtime_error("Draft vocabulary was not prepared");
            for (int v = 0; v < draft_rows; ++v) if (!std::isfinite(mtp_logits_[v])) throw std::runtime_error("Non-finite Intel draft logits");
            if (propose) input = propose(mtp_logits_);
            else {
                const int row = int(std::max_element(mtp_logits_, mtp_logits_ + draft_rows) - mtp_logits_);
                input = draft_vocab_limit_ ? draft_vocab_host_[row] : row;
            }
            drafts.push_back(input);
            q_.memcpy(mtp_seeds_, mtp_norm_, size_t(n_) * sizeof(float));
            if (cancelled && cancelled->load()) break;
        }
        q_.memcpy(x_, mtp_main_raw_, size_t(last_hidden_row_ + 1) * n_ * sizeof(float));
        if (cancelled && cancelled->load()) { q_.wait_and_throw(); return result; }
        checkpoint(false);
        std::vector<int> window{anchor}; window.insert(window.end(), drafts.begin(), drafts.end());
        q_.wait_and_throw();
        const auto verify_start = std::chrono::steady_clock::now();
        std::copy(window.begin(), window.end(), token_input_);
        for (size_t t = 0; t < window.size(); ++t) position_input_[t] = position_ + int(t);
        verifying_ = true;
        if (profiling_) profile_phase_ = "verify.";
        bool complete;
        try {
            auto* graph = verify_graph_enabled_ ? verification_graph(int(window.size())) : nullptr;
            if (cancelled && cancelled->load()) complete = false;
            else if (graph) { q_.ext_oneapi_graph(*graph); complete = true; }
            else complete = enqueue(int(window.size()), true, cancelled);
            q_.wait_and_throw();
        }
        catch (...) { verifying_ = false; profile_phase_.clear(); throw; }
        verifying_ = false;
        profile_phase_.clear();
        check_numeric_flags();
        if (!complete || (cancelled && cancelled->load())) { checkpoint(true); q_.wait_and_throw(); return result; }
        for (size_t i = 0; i < window.size() * size_t(vocabulary()); ++i)
            if (!std::isfinite(verify_logits_[i])) throw std::runtime_error("Non-finite Intel verification logits");
        int32_t picks[verify_capacity];
        const int accepted = strata::qwen35::speculative_sampling::select_prefix(drafts.data(), int(drafts.size()), picks,
            [&](int row) { return select(verify_logits_ + size_t(row) * vocabulary()); }, [&](int token) { return terminal(token); });
        const auto commit_start = std::chrono::steady_clock::now();
        commit_verify(accepted + 1, int(window.size()));
        result.inputs.assign(window.begin(), window.begin() + accepted + 1);
        result.output.assign(picks, picks + accepted + 1);
        mtp_pending_.assign(drafts.begin(), drafts.begin() + accepted);
        ++mtp_windows_; mtp_offered_ += drafts.size(); mtp_accepted_ += accepted;
        mtp_draft_ms_ += std::chrono::duration<double, std::milli>(verify_start - draft_start).count();
        mtp_verify_ms_ += std::chrono::duration<double, std::milli>(commit_start - verify_start).count();
        mtp_commit_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commit_start).count();
        collect_profile();
        return result;
    }
    void print_spec_stats() const {
        if (mtp_depth_) std::fprintf(stderr, "mtp_windows=%llu mtp_offered=%llu mtp_accepted=%llu mtp_draft_ms=%.3f mtp_verify_ms=%.3f mtp_commit_ms=%.3f\n",
            static_cast<unsigned long long>(mtp_windows_), static_cast<unsigned long long>(mtp_offered_), static_cast<unsigned long long>(mtp_accepted_), mtp_draft_ms_, mtp_verify_ms_, mtp_commit_ms_);
        if (draft_vocab_limit_) std::fprintf(stderr, "draft_vocab_limit=%d draft_vocab_updates=%llu draft_vocab_prepare_ms=%.3f target_full_vocabulary=1\n",
            draft_vocab_limit_, static_cast<unsigned long long>(draft_vocab_updates_), draft_vocab_prepare_ms_);
    }
    bool terminal(int token) const { return token == terminal_token_; }
    int context() const { return context_; }
    const float* logits() {
        if (!logits_valid_) {
            if (!position_) throw std::runtime_error("No processed prompt for Intel logits");
            remember("rms_norm", rms_norm(q_, x_ + size_t(last_hidden_row_) * n_, f32("output_norm.weight"), h_, 1, n_, epsilon_));
            matrix("output.weight", h_, logits_);
            q_.wait_and_throw();
            check_numeric_flags();
            for (int i = 0; i < vocabulary(); ++i) if (!std::isfinite(logits_[i])) throw std::runtime_error("Non-finite Intel logits");
            logits_valid_ = true;
        }
        return logits_;
    }
    void reset() {
        for (const auto& layer : layers_) {
            if (layer.recurrent) q_.memset(layer.recurrent, 0, size_t(hv_) * 128 * 128 * sizeof(float));
            if (layer.convolution) q_.memset(layer.convolution, 0, size_t(2 * hk_ + hv_) * 128 * 3 * sizeof(float));
        }
        q_.wait_and_throw();
        if (mtp_depth_) { q_.memset(mtp_last_norm_, 0, size_t(n_) * sizeof(float)); q_.wait_and_throw(); mtp_pending_.clear(); }
        position_ = 0;
        logits_valid_ = false;
    }
};
std::vector<int> parse_tokens(std::string text) {
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream stream(text);
    std::vector<int> tokens;
    int64_t token;
    while (stream >> token) {
        if (token < 0 || token > std::numeric_limits<int>::max()) throw std::runtime_error("Invalid token ID");
        tokens.push_back(int(token));
    }
    if (!stream.eof() || tokens.empty()) throw std::runtime_error("Expected a nonempty token ID list");
    return tokens;
}
int number(const char* input, int maximum) {
    const std::string text(input);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("Expected a nonnegative integer");
    const auto result = std::stoull(text);
    if (result > uint64_t(maximum)) throw std::runtime_error("Integer option is out of range");
    return int(result);
}
// Same host sampling policy as the existing Qwen engine: penalties, top-k,
// temperature, min-p and top-p. This does not add a speculative approximation.
using sampling_policy::Sampling;
using sampling_policy::sampling_distribution;
using sampling_policy::sample;

float real_option(const std::string& text) {
    size_t end = 0;
    const float value = std::stof(text, &end);
    if (end != text.size() || !std::isfinite(value)) throw std::runtime_error("Invalid finite sampling value");
    return value;
}
void sampling_option(Sampling& s, const std::string& field) {
    const auto equals = field.find('=');
    const std::string key = field.substr(0, equals), text = field.substr(equals + 1);
    const float value = real_option(text);
    if (key == "temperature" && value >= 0) s.temperature = value;
    else if (key == "top_p" && value > 0 && value <= 1) s.top_p = value;
    else if (key == "min_p" && value >= 0 && value <= 1) s.min_p = value;
    else if (key == "penalty_repeat" && value > 0) s.repeat = value;
    else if (key == "penalty_freq" && value >= 0) s.frequency = value;
    else if (key == "penalty_present" && value >= 0) s.presence = value;
    else if (key == "top_k") s.top_k = number(text.c_str(), 64);
    else if (key == "penalty_last_n" && value >= -1 && value <= 16384 && value == std::floor(value)) s.last_n = int(value);
    else if (key == "seed") {
        size_t end = 0;
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("Invalid random seed");
        s.seed = std::stoull(text, &end);
    } else if ((key == "mtp_auto" || key == "lookup_draft") && value == 0) {
        // Explicitly disabled. Nonzero speculation remains unsupported.
    } else if (key == "ckpt" || key == "turn_at") {
        // A hint for turn-state reuse. Edits replay from a clean recurrent
        // state; no on-disk session/snapshot capability is advertised.
        if (value < 0 || value != std::floor(value)) throw std::runtime_error("Invalid turn-cache hint");
    } else throw std::runtime_error("Unsupported Intel request option: " + field);
}
int serve(Engine& engine, bool stochastic_mtp = false) {
    const auto device = engine.device_info();
    std::printf("INFO backend=sycl experimental=1 kv=%s mtp=%d vision=0 memory_model=uma single_weight_store=1 device_vendor=%u device_id=%u device_integrated=%d device_architecture=%s adapter_luid=%s adapter_node_mask=%u\nREADY %d stop\n",
        engine.kv_type(), engine.mtp_depth(), device.vendor_id, device.device_id, int(device.integrated),
        device.architecture.c_str(), device.adapter_luid.empty() ? "unknown" : device.adapter_luid.c_str(),
        device.adapter_node_mask, engine.context());
    std::fflush(stdout);
    strata::common::CommandInbox inbox;
    std::vector<int> live;
    std::string line;
    while (inbox.next(line)) {
        if (line == "QUIT") return 0;
        if (line.empty()) continue;
        std::vector<int> prompt;
        Sampling sampling;
        int max_new = 0;
        try {
            if (line.rfind("GEN ", 0) != 0) throw std::runtime_error("Intel supports text GEN requests only");
            std::istringstream fields(line.substr(4));
            std::string field, ids;
            if (!(fields >> field)) throw std::runtime_error("Missing output count");
            max_new = number(field.c_str(), engine.context());
            while (fields >> field) {
                if (field.find('=') != std::string::npos && ids.empty()) sampling_option(sampling, field);
                else { if (!ids.empty()) ids += ' '; ids += field; }
            }
            prompt = parse_tokens(ids);
            if (!max_new || prompt.size() + size_t(max_new) > size_t(engine.context())) throw std::runtime_error("Request exceeds Intel context capacity");
            for (int token : prompt) if (token >= engine.vocabulary()) throw std::runtime_error("Token ID outside vocabulary");
        } catch (const std::exception& error) {
            std::printf("ERR %s\n", error.what());
            std::fflush(stdout);
            continue; // Invalid requests leave the resident conversation intact.
        }
        const auto start = std::chrono::steady_clock::now();
        const auto speculation_start = engine.speculation_counts();
        ExecutionRequest active_request;
        const bool reuse = live.size() <= prompt.size() && std::equal(live.begin(), live.end(), prompt.begin());
        if (!reuse) { engine.reset(); live.clear(); }
        const size_t reused = live.size();
        std::printf("RESUME %zu\n", reused);
        std::fflush(stdout);
        const float* logits = nullptr;
        for (size_t index = reused; index < prompt.size() && !inbox.cancelled().load();) {
            const int count = int(std::min(size_t(engine.chunk()), prompt.size() - index));
            logits = engine.step_many(prompt.data() + index, count, index + count == prompt.size(), &inbox.cancelled());
            if (!logits) break;
            live.insert(live.end(), prompt.begin() + index, prompt.begin() + index + count);
            index += count;
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::printf("PP %zu %zu %.3f\n", index, prompt.size(), elapsed);
            std::fflush(stdout);
        }
        const auto decode_start = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(decode_start - start).count();
        const size_t fresh = live.size() - reused;
        int produced = 0;
        const char* finish = "length";
        std::vector<int> history(prompt);
        const uint64_t generation_seed = sampling.seed ? sampling.seed : std::random_device{}();
        std::mt19937_64 random(generation_seed), draft_random(generation_seed ^ 0x9e3779b97f4a7c15ull);
        if (inbox.cancelled().load()) finish = "cancel";
        else if (engine.mtp_depth()) {
            const float* last_target_logits = engine.logits();
            std::vector<int> ready{sample(last_target_logits, engine.vocabulary(), sampling, history, random)};
            history.push_back(ready[0]);
            while (!ready.empty()) {
                bool done = false;
                for (int token : ready) {
                    if (inbox.cancelled().load()) { finish = "cancel"; done = true; break; }
                    std::printf("T %d\n", token); std::fflush(stdout); ++produced;
                    if (engine.terminal(token)) { finish = "stop"; done = true; break; }
                    if (produced == max_new) { done = true; break; }
                }
                if (done) break;
                engine.prepare_draft_vocabulary(last_target_logits, history);
                const bool rejection_sampling = stochastic_mtp && sampling.temperature > 0;
                std::vector<speculative_distribution::Distribution> draft_distributions;
                std::vector<int> draft_tokens, draft_history(history);
                std::function<int(const float*)> propose;
                if (rejection_sampling) propose = [&](const float* row) {
                    const auto& ids = engine.draft_vocabulary();
                    auto distribution = sampling_distribution(row, ids.empty() ? engine.vocabulary() : int(ids.size()), sampling,
                        draft_history, ids.empty() ? nullptr : ids.data());
                    const int token = distribution.draw(draft_random);
                    draft_tokens.push_back(token); draft_history.push_back(token);
                    draft_distributions.push_back(std::move(distribution));
                    return token;
                };
                size_t selected_row = 0;
                auto next = engine.speculate(ready.back(), max_new - produced, [&](const float* row) {
                    last_target_logits = row;
                    const auto target = sampling_distribution(row, engine.vocabulary(), sampling, history);
                    const int token = rejection_sampling && selected_row < draft_distributions.size()
                        ? speculative_distribution::verify(target, draft_distributions[selected_row], draft_tokens[selected_row], random)
                        : sampling.temperature <= 0 ? target.tokens[0] : target.draw(random);
                    ++selected_row;
                    history.push_back(token); return token;
                }, &inbox.cancelled(), propose);
                if (next.output.empty()) { finish = "cancel"; break; }
                live.insert(live.end(), next.inputs.begin(), next.inputs.end());
                ready = std::move(next.output);
            }
        }
        else for (int index = 0; index < max_new; ++index) {
            if (inbox.cancelled().load()) { finish = "cancel"; break; }
            if (index == 0) logits = engine.logits();
            const int token = sample(logits, engine.vocabulary(), sampling, history, random);
            std::printf("T %d\n", token);
            std::fflush(stdout);
            ++produced;
            history.push_back(token);
            if (engine.terminal(token)) { finish = "stop"; break; }
            if (index + 1 < max_new) { logits = engine.step(token); live.push_back(token); }
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
        engine.print_spec_stats();
        const auto speculation_end = engine.speculation_counts();
        std::printf("DONE %d %zu %.3f %.3f %s %llu %llu %zu 0 0 0 0 0.0 %zu 0\n", produced, prompt.size(), prompt_ms, decode_ms, finish,
            static_cast<unsigned long long>(speculation_end.first - speculation_start.first),
            static_cast<unsigned long long>(speculation_end.second - speculation_start.second), reused, fresh);
        std::fflush(stdout);
    }
    return 0;
}
} // namespace
} // namespace strata::intel

int main(int argc, char** argv) {
    try {
        strata::paths::Utf8Arguments arguments(argc, argv);
        argc = arguments.argc(); argv = arguments.argv();
        std::string path, token_list, token_file, dump_path;
        int count = 32, context = 4096, device = 0, chunk = 128, mtp_depth = 0, draft_vocab = 0;
        bool serve = false, use_graph = true, profiling = false, prefill_xmx = false, prefill_npu = false, prefill_mkl = false, decode_dp4a = false, verify_xmx = false, kv_int8 = false, prefill_int8_xmx = false, vocab_xmx = false, verify_mkl_attention = false, stochastic_mtp = false, expert_layout = false, device_state = false, verify_graph = false, dense_layout = false, dense_split_k = false;
        strata::intel::WeightMemory memory = strata::intel::WeightMemory::Host;
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--serve") { serve = true; continue; }
            if (option == "--decode-dp4a") { decode_dp4a = true; continue; }
            if (option == "--verify-xmx") { verify_xmx = true; continue; }
            if (option == "--prefill-int8-xmx") { prefill_int8_xmx = true; continue; }
            if (option == "--vocab-xmx") { vocab_xmx = true; continue; }
            if (option == "--verify-mkl-attention") { verify_mkl_attention = true; continue; }
            if (option == "--mtp-stochastic") { stochastic_mtp = true; continue; }
            if (option == "--expert-xmx-layout") { expert_layout = true; continue; }
            if (option == "--device-state") { device_state = true; continue; }
            if (option == "--verify-graph") { verify_graph = true; continue; }
            if (option == "--dense-split-k") { dense_split_k = true; continue; }
            if (option == "--dense-xmx-layout") { dense_layout = true; continue; }
            if (option == "--prefill-mkl") { prefill_mkl = true; continue; }
            if (option == "--prefill-npu") { prefill_npu = true; continue; }
            if (option == "--prefill-xmx") { prefill_xmx = true; continue; }
            if (option == "--profile") { profiling = true; continue; }
            if (option == "--no-graph") { use_graph = false; continue; }
            if (option == "--help") {
                std::puts("tinystrata-intel --model MODEL [--serve | --tokens IDS | --tokens-file FILE] [--n N] [--max-context N] [--device N] [--chunk 1..1024] [--mtp 0..7] [--kv-type fp32|int8] [--draft-vocab 0|64..4096] [--dump-full-logits FILE]\nExperimental text driver. KV defaults to FP32; MTP and draft vocabulary selection default to 0 (off); images are unsupported. A nonzero draft vocabulary requires --serve --mtp-stochastic --mtp --vocab-xmx.");
                return 0;
            }
            if (option == "--capabilities") {
                std::puts("{\"backend\":\"sycl\",\"experimental\":true,\"model_families\":[],\"kv_types\":[\"fp32\",\"int8\"],\"images\":false,\"mtp\":true,\"experimental_mtp_max_depth\":7,\"serve\":true}");
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing argument for " + option);
            if (option == "--gguf" || option == "--model") path = argv[++i];
            else if (option == "--tokens") token_list = argv[++i];
            else if (option == "--tokens-file") token_file = argv[++i];
            else if (option == "--dump-full-logits") dump_path = argv[++i];
            else if (option == "--n") count = strata::intel::number(argv[++i], 16384);
            else if (option == "--context" || option == "--max-context") context = strata::intel::number(argv[++i], 32768);
            else if (option == "--chunk") chunk = strata::intel::number(argv[++i], 1024);
            else if (option == "--device") device = strata::intel::number(argv[++i], 1024);
            else if (option == "--weight-memory") {
                const std::string kind(argv[++i]);
                if (kind == "host") memory = strata::intel::WeightMemory::Host;
                else if (kind == "shared") memory = strata::intel::WeightMemory::Shared;
                else if (kind == "device") memory = strata::intel::WeightMemory::Device;
                else throw std::runtime_error("Unsupported weight memory kind");
            }
            else if (option == "--kv-type") {
                const std::string kind(argv[++i]);
                if (kind != "fp32" && kind != "int8") throw std::runtime_error("Intel supports explicit --kv-type fp32 or int8");
                kv_int8 = kind == "int8";
            }
            else if (option == "--draft-vocab") draft_vocab = strata::intel::number(argv[++i], 4096);
            else if (option == "--mtp") mtp_depth = strata::intel::number(argv[++i], 7);
            else if (option == "--batch") { if (strata::intel::number(argv[++i], 16) != 1) throw std::runtime_error("The initial Intel route serves one request at a time"); }
            else throw std::runtime_error("Unsupported Intel option: " + option);
        }
        if (path.empty()) throw std::runtime_error("--gguf is required");
        if (draft_vocab && (!serve || !stochastic_mtp || !mtp_depth || !vocab_xmx || draft_vocab < 64))
            throw std::runtime_error("--draft-vocab requires --serve, --mtp-stochastic, --mtp, --vocab-xmx and 64..4096 rows");
        if (stochastic_mtp && (!serve || !mtp_depth)) throw std::runtime_error("--mtp-stochastic requires --serve and --mtp >= 1");
        if (serve) {
            if (!token_list.empty() || !token_file.empty() || !dump_path.empty()) throw std::runtime_error("Token CLI options cannot accompany --serve");
            std::optional<strata::intel::ExecutionRequest> loading(std::in_place);
            strata::intel::Engine engine(path, context, unsigned(device), memory, chunk, use_graph, profiling, prefill_xmx, prefill_npu, prefill_mkl, decode_dp4a, mtp_depth, verify_xmx, kv_int8, prefill_int8_xmx, vocab_xmx, verify_mkl_attention, expert_layout, device_state, verify_graph, dense_layout, dense_split_k, draft_vocab);
            loading.reset();
            return strata::intel::serve(engine, stochastic_mtp);
        }
        if (!token_file.empty()) {
            if (!token_list.empty()) throw std::runtime_error("Choose --tokens or --tokens-file");
            std::ifstream input(strata::paths::native_path(token_file));
            if (!input) throw std::runtime_error("Cannot open token ID file");
            token_list.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        auto tokens = strata::intel::parse_tokens(token_list);
        if (tokens.size() + size_t(count) > size_t(context)) throw std::runtime_error("Prompt and requested output exceed --context");
        strata::intel::ExecutionRequest active_request;
        strata::intel::Engine engine(path, context, unsigned(device), memory, chunk, use_graph, profiling, prefill_xmx, prefill_npu, prefill_mkl, decode_dp4a, mtp_depth, verify_xmx, kv_int8, prefill_int8_xmx, vocab_xmx, verify_mkl_attention, expert_layout, device_state, verify_graph, dense_layout, dense_split_k, draft_vocab);
        const float* logits = nullptr;
        const auto start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < tokens.size();) {
            const int batch = int(std::min(size_t(engine.chunk()), tokens.size() - index));
            logits = engine.step_many(tokens.data() + index, batch, index + batch == tokens.size());
            index += batch;
            if (tokens.size() >= 128 && (index % 64 == 0 || index == tokens.size())) {
                std::fprintf(stderr, "prompt_progress=%zu/%zu\n", index, tokens.size());
                std::fflush(stderr);
            }
        }
        const auto prefill_end = std::chrono::steady_clock::now();
        if (!dump_path.empty()) {
            std::ofstream output(strata::paths::native_path(dump_path), std::ios::binary);
            if (!output.write(reinterpret_cast<const char*>(logits), size_t(engine.vocabulary()) * sizeof(float)))
                throw std::runtime_error("Could not write full logits");
        }
        int produced = 0, decode_steps = 0;
        if (engine.mtp_depth() && count > 0) {
            auto greedy = [&](const float* row) { return int(std::max_element(row, row + engine.vocabulary()) - row); };
            std::vector<int> ready{greedy(logits)};
            while (!ready.empty()) {
                bool done = false;
                for (int token : ready) {
                    std::printf("TOKEN %d\n", token); std::fflush(stdout); ++produced;
                    if (engine.terminal(token) || produced == count) { done = true; break; }
                }
                if (done) break;
                auto next = engine.speculate(ready.back(), count - produced, greedy);
                decode_steps += int(next.output.size()); ready = std::move(next.output);
            }
        } else for (int index = 0; index < count; ++index) {
            const int token = int(std::max_element(logits, logits + engine.vocabulary()) - logits);
            std::printf("TOKEN %d\n", token);
            std::fflush(stdout);
            ++produced;
            if (engine.terminal(token)) break;
            if (index + 1 < count) { logits = engine.step(token); ++decode_steps; }
        }
        const auto end = std::chrono::steady_clock::now();
        engine.print_spec_stats();
        std::fprintf(stderr, "prompt_tokens=%zu prefill_seconds=%.6f output_tokens=%d decode_steps=%d decode_seconds=%.6f\n", tokens.size(),
            std::chrono::duration<double>(prefill_end - start).count(), produced, decode_steps, std::chrono::duration<double>(end - prefill_end).count());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Intel Qwen engine failed: %s\n", error.what());
        return 1;
    }
}
