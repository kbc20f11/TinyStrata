// Causal Qwen3.6/Tiel and opt-in Clover D0 driver for Windows SYCL.
// Model/layer semantics follow src/qwen35/q36_engine.cu (Strata/TinyStrata,
// MIT; LICENSE and NOTICE). Optional arithmetic/MTP candidates are explicit;
// compressed weights and FP32 recurrent/KV state are retained.
#include "strata/intel/weight_store.hpp"
#include "strata/intel/context_policy.hpp"
#include "strata/intel/graph_visibility.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/decode_dp4a.hpp"
#include "strata/intel/q8_xmx.hpp"
#include "strata/intel/moe_route_policy.hpp"
#include "strata/intel/kv_q8.hpp"
#include "strata/intel/execution_request.hpp"
#include "strata/intel/turn_checkpoint.hpp"
#include "strata/intel/qwen_vision.hpp"
#include "strata/intel/clover_causal.hpp"
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
#include "strata/intel/gpu_sampling_candidate.hpp"
#include "strata/intel/gpu_support_policy.hpp"
#include <cstdlib>
#include <functional>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
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
    std::unique_ptr<GpuSamplingCandidate> gpu_sampling_candidate_;
    bool gpu_support_sampling_ = false;
    GpuSamplingCandidate& gpu_sampling_candidate() {
        if (!gpu_sampling_candidate_) gpu_sampling_candidate_ = std::make_unique<GpuSamplingCandidate>(q_, vocabulary());
        return *gpu_sampling_candidate_;
    }
    uint64_t mtp_windows_ = 0, mtp_offered_ = 0, mtp_accepted_ = 0;
    double mtp_draft_ms_ = 0, mtp_verify_ms_ = 0, mtp_commit_ms_ = 0;
    bool verifying_ = false;
    float *mtp_cat_ = nullptr, *mtp_main_raw_ = nullptr, *mtp_main_norm_ = nullptr, *mtp_seeds_ = nullptr;
    float *mtp_norm_ = nullptr, *mtp_last_norm_ = nullptr, *checkpoint_norm_ = nullptr, *mtp_logits_ = nullptr;
    float *verify_logits_ = nullptr, *verify_raw_ = nullptr, *verify_norm_ = nullptr;
    bool kv_int8_ = false;
    bool kv_fp8_ = false; // kv_int8_ selects shared byte-block cache paths; this flag identifies their codec.
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
    bool vision_enabled_ = false;
    bool clover_vision_workspace_aliases_ = false;
    int image_pad_ = -1;
    int32_t* mrope_ = nullptr;
    qwen_vision::Input image_input_;
    int graph_extent_ = 128;
    bool graph_enabled_ = true;
    bool recording_ = false, profiling_ = false, prefill_xmx_ = false, verify_xmx_ = false, prefill_int8_xmx_ = false;
    bool verify_mkl_attention_ = false, dense_split_k_ = false;
    moe_route_policy::Policy moe_route_policy_;
    uint64_t moe_route_observed_ = 0, moe_route_eligible_ = 0, moe_route_q8_projections_ = 0;
    uint64_t moe_route_observed_512_ = 0, moe_route_eligible_512_ = 0;
    int moe_route_active_min_ = 256, moe_route_active_max_ = 0, moe_route_peak_max_ = 0;
    uint32_t moe_route_hot_slots_max_ = 0;
    bool device_state_ = false;
    bool verify_graph_enabled_ = false;
    int recording_extent_ = 128;
    int recording_minimum_visible_ = 0;
    bool clover_fused_t1_hint_ = false;
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
    bool clover_ = false;
    // One independent host snapshot; never alias speculative/cancellation
    // scratch. KV rows remain resident and are guarded by both exact prefixes.
    struct TurnSnapshot {
        std::vector<uint8_t> bytes;
        std::vector<int> tokens;
        int position = 0;
        void invalidate() { position = 0; tokens.clear(); }
    } turn_snapshot_;
    bool turn_checkpoint_enabled_ = false;
    size_t turn_checkpoint_bytes_ = 0;
    context_policy::Footprint actual_allocations_;
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
        actual_allocations_.allocation(count,sizeof(T),!device_state_||cpu_visible);
        q_.memset(result, 0, count * sizeof(T));
        return result;
    }
    void allocate_kv(LayerState& state) {
        const size_t count = size_t(kv_heads_) * context_ * 256;
        if (kv_int8_) state.q8 = {alloc<int8_t>(count), alloc<int8_t>(count), alloc<sycl::half>(count / 32),
            alloc<sycl::half>(count / 32), context_, kv_heads_, kv_fp8_};
        else { state.key = alloc<float>(count); state.value = alloc<float>(count); }
    }
    void check_numeric_flags() const {
        if (dp4a_ && !dp4a_->valid()) throw std::runtime_error("Non-finite Intel Q8 activation");
        if (kv_error_ && kv_error_[0]) throw std::runtime_error(kv_fp8_ ? "Invalid/non-finite Intel FP8 KV input" : "Invalid/non-finite Intel INT8 KV input");
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
        const auto it = weights_->metadata().find(std::string(clover_ ? "clover_moe." : "qwen35moe.") + name);
        return it == weights_->metadata().end() ? fallback : it->second.num();
    }
    // SVE1 rows remain owned by the active request until all queued copies
    // complete. Main and MTP prefill receive the same replacement embeddings.
    void image_rows(float* destination, int count, int position) {
        for (const auto& image : image_input_.images) {
            const size_t begin = std::max(size_t(position), image.cell);
            const size_t end = std::min(size_t(position + count), image.cell + size_t(image.rows));
            if (begin >= end) continue;
            q_.memcpy(destination + (begin - size_t(position)) * size_t(n_),
                image_input_.embeddings.data() + image.offset + (begin - image.cell) * size_t(n_),
                (end - begin) * size_t(n_) * sizeof(float));
        }
    }
    void matrix(const std::string& name, const float* input, float* output, int tokens = 1) {
        const auto& t = tensor(name);
        const int width = int(t.info.shape[0]);
        const int rows = t.info.shape.size() < 2 ? 1 : int(t.info.shape[1]);
#ifdef TINYSTRATA_INTEL_NPU
        const char* split_mode = std::getenv("STRATA_REVIEW_NPU_SPLIT");
#ifdef TINYSTRATA_INTEL_MKL
        if (npu_ && gemm_ && split_mode && *split_mode && tokens >= 128 &&
            (name.find("attn_qkv.weight") != std::string::npos || name.find("attn_gate.weight") != std::string::npos)) {
            sycl::event completion;
            if (npu_->run_split(*gemm_, int(t.info.type), t.data, input, output, rows, width, tokens,
                               1024, std::string(split_mode) != "serial", std::string(split_mode) == "remote", completion,
                               t.layout == WeightLayout::Q6Interleaved16)) {
                remember("npu_split_join", completion);
                return;
            }
        }
#endif
#endif
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
        if (npu_ && !split_mode && tokens >= 128 && t.info.type != 0) {
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
        if(clover_vision_workspace_aliases_){
            remember("elementwise_attention",silu_mul(q_,gate,up,output,size_t(count)));
            return;
        }
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
        if(clover_vision_workspace_aliases_)
            remember("elementwise_attention",conv_step_many(q_,input,history,weight,output,hk,hv,tokens,epsilon));
        else dispatch(sycl::nd_range<1>(size_t(channels), 128), [=](sycl::nd_item<1> it) {
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
        const int visible_hint = clover_ && clover_fused_t1_hint_
            ? (recording_ ? recording_minimum_visible_ : position_ + token + 1) : 0;
        const auto event = kv_int8_ ? decode_attention_q8(q_, query_ + offset, query_gate_ + offset, state.q8, scores_,
            attention_partial_, attention_ + offset, heads_, position_input_ + token, extent, profiling_ ? stages : nullptr, 1, visible_hint)
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
        remember("prepare_attention", prepare_qwen_attention(q_, packed, keys, values, qw, kw,
            queries, gates, rotated_keys, cache_k, cache_v, heads, kv_heads, context, n_rot, tokens,
            position_input, mrope_, base, epsilon, quantized_kv));
        const float *visible_keys = state.key, *visible_values = state.value;
        bool small_gemm_attention = false;
#ifdef TINYSTRATA_INTEL_MKL
        small_gemm_attention = verify_mkl_attention_ && gemm_ && tokens > 1 && tokens <= verify_capacity && position >= 512;
#endif
        if (kv_int8_) {
            remember(kv_fp8_ ? "store_fp8_kv" : "store_int8_kv", store_q8_kv(q_, key_, value_, state.q8, position_input_, tokens, kv_error_));
            if (tokens > verify_capacity || small_gemm_attention) {
                remember(kv_fp8_ ? "dequantize_fp8_kv" : "dequantize_int8_kv", dequantize_q8_kv(q_, state.q8, kv_float_key_, kv_float_value_, position + tokens));
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
                        attention_partial_, attention_, heads_, position_input_, extent, profiling_ ? stages : nullptr, tokens,
                        recording_ || verify_graph_enabled_ ? 0 : position + tokens));
                    if (profiling_) {
                        remember("decode_attention_qk", stages[0]); remember("decode_attention_softmax", stages[1]);
                        remember("decode_attention_pv", stages[2]);
                    }
                } else for (int token = 0; token < tokens; ++token) attention_row(state, token, recording_ ? recording_extent_ : position + token + 1);
            } else {
#ifdef TINYSTRATA_INTEL_MKL
            if (gemm_ && tokens >= 1008) remember("mkl_attention_gate", gemm_->attention(query_, query_gate_, visible_keys, visible_values,
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
        bool route_packed_q8 = false;
        if (grouped) {
            auto expert_plan = plan_experts(q_, ids, plan);
            remember("expert_plan", expert_plan);
#ifdef TINYSTRATA_INTEL_MKL
        if (moe_route_policy_.observe_tokens(tokens) && !recording_ && !verifying_ && prefix != "blk.40." && gemm_ && dp4a_ &&
            moe_route_policy::measured_projection(int(gate.info.type),ff,n,false,interleaved_experts,tokens,experts,picks) &&
            moe_route_policy::measured_projection(int(up.info.type),ff,n,false,interleaved_experts,tokens,experts,picks) &&
            moe_route_policy::measured_projection(int(down.info.type),n,ff,true,interleaved_experts,tokens,experts,picks)) {
            // The plan already produces these shared control counts. One
            // scoped completion and one bounded scan serve all projections;
            // no IDs/weights/slots are copied, filtered, or reordered.
            expert_plan.wait_and_throw();
            const auto route = moe_route_policy::analyze(expert_counts_,experts,tokens,picks);
            const bool eligible = moe_route_policy::prefer_packed_q8(route);
            ++moe_route_observed_; moe_route_eligible_ += eligible;
            if (tokens == 512) { ++moe_route_observed_512_; moe_route_eligible_512_ += eligible; }
            moe_route_active_min_ = std::min(moe_route_active_min_,route.active);
            moe_route_active_max_ = std::max(moe_route_active_max_,route.active);
            moe_route_peak_max_ = std::max(moe_route_peak_max_,route.peak);
            moe_route_hot_slots_max_ = std::max(moe_route_hot_slots_max_,route.hot_slots);
            route_packed_q8 = moe_route_policy::select_packed_q8(moe_route_policy_,route);
        }
#endif
        }
        auto product = [&](const ResidentTensor& weight, const float* input, float* output, int rows, int width, bool separate) {
#ifdef TINYSTRATA_INTEL_MKL
            if (interleaved_experts && gemm_ && tokens >= 512 && !route_packed_q8) {
                remember("mkl_interleaved_experts", gemm_->selected(int(weight.info.type), weight.data, weight.bytes / experts,
                    plan, input, output, rows, width, separate, true));
                return;
            }
#endif
            if (interleaved_experts) {
                if (!dp4a_) throw std::runtime_error("Interleaved experts require Q8 input kernels");
                const auto event = dp4a_->grouped(int(weight.info.type), weight.data, plan, input, output, rows, width, separate, true, route_packed_q8);
                moe_route_q8_projections_ += route_packed_q8;
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
        image_rows(x_, count, position_);
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
        image_rows(x_, count, position);
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
            recording_minimum_visible_ = clover_ && clover_fused_t1_hint_
                ? graph_minimum_visible(context_,graph_extent_) : 0;
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
            recording_ = true; recording_extent_ = extent; recording_minimum_visible_ = 0;
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
    Engine(const std::string& path, int context, unsigned device_index, WeightMemory memory = WeightMemory::Host, int chunk = 128, bool use_graph = true, bool profiling = false, bool prefill_xmx = false, bool prefill_npu = false, bool prefill_mkl = false, bool decode_dp4a = false, int mtp_depth = 0, bool verify_xmx = false, bool kv_int8 = false, bool prefill_int8_xmx = false, bool vocab_xmx = false, bool verify_mkl_attention = false, bool expert_layout = false, bool device_state = false, bool verify_graph = false, bool dense_layout = false, bool dense_split_k = false, int draft_vocab_limit = 0, bool experimental_tiel = false, bool experimental_clover = false, bool vision = false, bool kv_fp8 = false) :
        q_(select_device(device_index), [](sycl::exception_list errors) { for (const auto& error : errors) std::rethrow_exception(error); },
           profiling ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
                     : sycl::property_list{sycl::property::queue::in_order{}}), profiling_(profiling), prefill_xmx_(prefill_xmx), context_(context), chunk_(chunk) {
        const auto selected_device = inspect_device(q_.get_device());
        if (const char* limitation = baseline_limitation(selected_device.capabilities)) throw std::runtime_error(limitation);
#ifdef TINYSTRATA_INTEL_AOT_TARGET
        if (std::string(TINYSTRATA_INTEL_AOT_TARGET) == "intel_gpu_ptl_h" && !selected_device.ptl_h_aot)
            throw std::runtime_error("This PTL-H AOT runtime needs Panther Lake; select the generic SPIR-V runtime for this Intel GPU");
#endif
        if (!supports_xmx(selected_device.capabilities)) {
            if (prefill_xmx || verify_xmx || prefill_int8_xmx || vocab_xmx || expert_layout || dense_layout)
                std::fprintf(stderr, "Intel matrix/ESIMD capability unavailable; using original-layout GPU kernels\n");
            prefill_xmx = prefill_xmx_ = verify_xmx = prefill_int8_xmx = vocab_xmx = false;
            expert_layout = dense_layout = dense_split_k = false;
            draft_vocab_limit = 0;
        }
        if (!selected_device.capabilities.esimd) decode_dp4a = false;
        if (!selected_device.integrated) device_state = true;
        graph_enabled_ = use_graph;
        if (kv_fp8 && (experimental_tiel || experimental_clover)) throw std::runtime_error("FP8 KV is experimental Qwen3.6 only");
        kv_fp8_ = kv_fp8;
        kv_int8 = kv_int8 || kv_fp8;
        kv_int8_ = kv_int8;
        verify_xmx_ = verify_xmx;
        verify_mkl_attention_ = verify_mkl_attention;
        device_state_ = device_state;
        dense_split_k_ = dense_split_k;
        verify_graph_enabled_ = verify_graph;
        if (device_state_) std::fprintf(stderr, "Intel state_memory=device cpu_control_and_logits=shared\n");
        prefill_int8_xmx_ = prefill_int8_xmx;
        moe_route_policy::Recipe moe_recipe;
        moe_recipe.qwen = !experimental_tiel && !experimental_clover;
        moe_recipe.interleaved = expert_layout; moe_recipe.dp4a = decode_dp4a;
#ifdef TINYSTRATA_INTEL_MKL
        moe_recipe.mkl = prefill_mkl;
#endif
#ifdef TINYSTRATA_INTEL_INT8_XMX
        moe_recipe.int8_xmx = prefill_int8_xmx;
#endif
        bool moe_ptl_h_aot = false;
#ifdef TINYSTRATA_INTEL_AOT_TARGET
        moe_ptl_h_aot = std::string(TINYSTRATA_INTEL_AOT_TARGET) == "intel_gpu_ptl_h" && selected_device.ptl_h_aot;
#endif
        moe_route_policy_ = moe_route_policy::configure(std::getenv("STRATA_REVIEW_INTEL_MOE_ROUTE"),
            std::getenv("STRATA_REVIEW_INTEL_MOE_ROUTE_STATS"),moe_recipe,
            {selected_device.vendor_id,selected_device.device_id,selected_device.integrated,moe_ptl_h_aot});
        mtp_depth_ = mtp_depth;
        vision_enabled_ = vision;
        bool measured_clover_runtime = false;
#ifdef TINYSTRATA_INTEL_AOT_TARGET
        measured_clover_runtime = std::string(TINYSTRATA_INTEL_AOT_TARGET) == "intel_gpu_ptl_h";
#endif
        clover_fused_t1_hint_ = experimental_clover && measured_clover_runtime &&
            selected_device.vendor_id == 0x8086 && selected_device.device_id == 0xB082 &&
            selected_device.integrated && selected_device.ptl_h_aot;
        if(const char* setting=experimental_clover?std::getenv("STRATA_INTEL_CLOVER_FUSED_T1"):nullptr){
            const std::string value(setting);
            if(value!="0"&&value!="1")throw std::runtime_error("STRATA_INTEL_CLOVER_FUSED_T1 requires0 or1");
            clover_fused_t1_hint_=experimental_clover&&value=="1";
        }
        draft_vocab_limit_ = draft_vocab_limit;
        const char* turn_setting = std::getenv("TINYSTRATA_INTEL_TURN_CHECKPOINT");
        turn_checkpoint_enabled_ = turn_checkpoint::enabled(turn_setting);
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
            if (experimental_clover) clover_causal::validate_launch(kv_int8, mtp_depth, draft_vocab_limit, false, prefill_npu);
            weights_ = WeightStore::load(q_, path, memory, vocab_xmx, expert_layout, dense_layout, dense_split_k, experimental_tiel, experimental_clover, vision);
            clover_ = weights_->geometry().family == strata::ModelFamily::CloverMoe;
            clover_vision_workspace_aliases_=qwen_vision::clover_workspace_aliases(clover_,vision_enabled_,mtp_depth_);
            if (expert_layout) std::fprintf(stderr, "Intel expert_layout=interleaved16 Q8_activations=1 single_weight_store=1\n");
            if (vocab_xmx) {
                if (weights_->tensors().at("output.weight").layout == WeightLayout::Q6Interleaved16)
                    std::fprintf(stderr, "Intel vocabulary=interleaved_Q6_K XMX unchanged_quantization=1 single_weight_store=1\n");
                else
                    std::fprintf(stderr, "Intel vocabulary=original_GGUF requested_vocab_xmx=1 q6_interleaving=0 unchanged_quantization=1 single_weight_store=1\n");
            }
            const auto& g = weights_->geometry();
            if (draft_vocab_limit_ > int(g.n_vocab)) throw std::runtime_error("Draft vocabulary limit exceeds the real vocabulary");
            if (mtp_depth && g.n_mtp_layers < 1) throw std::runtime_error("MTP requested but this GGUF has no draft block");
            if (g.n_embd != 2048 || g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 ||
                g.ssm_k_heads != 16 || g.ssm_v_heads != 32 || g.n_head != 16 || g.n_head_kv != 2 ||
                g.n_expert != 256 || g.n_expert_used != 8 || g.n_ff != 512 || g.n_ff_shexp != 512 ||
                g.full_attention_interval != 4 || g.n_layers != 40 || g.n_vocab <= 0 || g.n_vocab > 1000000 ||
                context < 0 || context > context_policy::full_limit)
                throw std::runtime_error("Initial Intel Qwen engine does not support this geometry/context");
            const bool measured_support_default = selected_device.vendor_id == 0x8086 && selected_device.device_id == 0xB082 &&
                selected_device.integrated && selected_device.ptl_h_aot && g.n_vocab == 248320;
            gpu_support_sampling_ = gpu_support_policy::enabled(measured_support_default,
                mtp_depth_ > 0 && draft_vocab_limit_ > 0, std::getenv("TINYSTRATA_INTEL_GPU_SUPPORT"));
            if (gpu_support_sampling_) std::fprintf(stderr,
                "Intel draft_support_selection=gpu_histogram limit=%d scratch_bytes=%zu target_sampling=cpu\n",
                draft_vocab_limit_, GpuSamplingCandidate::storage_bytes(size_t(g.n_vocab)));
            n_ = int(g.n_embd); hk_ = int(g.ssm_k_heads); hv_ = int(g.ssm_v_heads);
            heads_ = int(g.n_head); kv_heads_ = int(g.n_head_kv); experts_ = int(g.n_expert); picks_ = int(g.n_expert_used);
            ff_ = int(g.n_ff); shared_ff_ = int(g.n_ff_shexp);
            epsilon_ = float(metadata_number("attention.layer_norm_rms_epsilon", 1e-6));
            if (const auto it = weights_->metadata().find("tokenizer.ggml.eos_token_id"); it != weights_->metadata().end())
                terminal_token_ = int(it->second.u);
            rope_base_ = float(metadata_number("rope.freq_base", 1e7)); n_rot_ = int(metadata_number("rope.dimension_count", 64));
            if (n_rot_ <= 0 || n_rot_ > 256 || n_rot_ % 2 || !(epsilon_ > 0) || !(rope_base_ > 0))
                throw std::runtime_error("Invalid Qwen normalization/rotary settings");
            if (vision_enabled_) {
                // WeightStore admitted the image metadata before allocating
                // its one arena, using the already open tokenizer mapping.
                image_pad_ = weights_->image_pad();
                if (n_rot_ != 64 || image_pad_ < 0 || image_pad_ >= g.n_vocab)
                    throw std::logic_error("Intel image weight admission differs from engine geometry");
            }
            const uint64_t attention_layers = uint64_t(g.n_layers / g.full_attention_interval);
            const uint64_t recurrent_layers = uint64_t(g.n_layers) - attention_layers;
            const uint64_t state_bytes = recurrent_layers * (uint64_t(hv_) * 128 * 128 + uint64_t(2 * hk_ + hv_) * 128 * 3) * sizeof(float);
            turn_checkpoint_bytes_ = turn_checkpoint_enabled_ ? size_t(state_bytes + uint64_t(n_) * sizeof(float) * (mtp_depth_ ? 2 : 1)) : 0;
            const auto remaining_memory = memory_budget(selected_device);
            context_policy::QwenSpec context_shape;
            context_shape.chunk=chunk_;context_shape.vocabulary=g.n_vocab;context_shape.draft_vocabulary=draft_vocab_limit_;
            context_shape.int8=kv_int8_&&!kv_fp8_;context_shape.fp8_e4m3=kv_fp8_;context_shape.mtp=mtp_depth_>0;context_shape.mkl=prefill_mkl;context_shape.dp4a=decode_dp4a;
            context_shape.dense_split=dense_split_k;context_shape.support=gpu_support_sampling_;context_shape.device_state=device_state_;
            context_shape.turn_checkpoint=turn_checkpoint_enabled_;context_shape.graph=use_graph&&!profiling;context_shape.verify_graph=verify_graph;
            context_shape.npu=prefill_npu;context_shape.npu_split=std::getenv("STRATA_REVIEW_NPU_SPLIT")!=nullptr;
            context_shape.vision=vision_enabled_;
            context_shape.clover_vision_aliases=clover_vision_workspace_aliases_;
            const int context_limit=context_policy::model_limit(metadata_number("context_length",0),clover_?context_policy::lfm_limit:context_policy::full_limit);
            const auto chosen=context_policy::select(context,context_limit,remaining_memory,selected_device.max_allocation,
                [&](uint64_t tokens){return context_policy::qwen(tokens,context_shape);});
            context=context_=chosen.tokens;graph_extent_=std::min(context_,128);
            const auto phase=clover_vision_workspace_aliases_
                ? context_policy::clover_phase_layout(chunk_,std::max((2*hk_+hv_)*128,heads_*512),heads_*256,
                    hv_*128,picks_*n_,picks_*ff_,hv_*128)
                : context_policy::CloverPhaseLayout{};
            x_ = alloc<float>(size_t(chunk_) * (n_)); h_ = alloc<float>(size_t(chunk_) * (n_)); mixer_ = alloc<float>(size_t(chunk_) * (n_));
            if(clover_vision_workspace_aliases_){
                expert_down_=alloc<float>(size_t(phase.routed_count));
                expert_gate_=alloc<float>(size_t(phase.gate_count));
            }
            qkv_ = clover_vision_workspace_aliases_ ? expert_down_
                : alloc<float>(size_t(chunk_) * (std::max((2 * hk_ + hv_) * 128, heads_ * 512)));
            z_ = clover_vision_workspace_aliases_ ? expert_gate_ : alloc<float>(size_t(chunk_) * (hv_ * 128));
            alpha_ = alloc<float>(size_t(chunk_) * (hv_)); beta_ = alloc<float>(size_t(chunk_) * (hv_));
            // These aliases are enabled only for Clover image MTP0. Each source
            // is dead after its consumer; aliased pointers never acquire ownership.
            conv_ = clover_vision_workspace_aliases_ ? qkv_ : alloc<float>(size_t(chunk_) * ((2 * hk_ + hv_) * 128));
            recurrent_output_ = clover_vision_workspace_aliases_ ? expert_down_+phase.query_offset
                : alloc<float>(size_t(chunk_) * (hv_ * 128));
            normalized_ = clover_vision_workspace_aliases_ ? recurrent_output_ : alloc<float>(size_t(chunk_) * (hv_ * 128));
            router_ = alloc<float>(size_t(chunk_) * (experts_)); expert_ids_ = alloc<int32_t>(size_t(chunk_) * (picks_)); expert_weights_ = alloc<float>(size_t(chunk_) * (picks_));
            if(!clover_vision_workspace_aliases_)expert_gate_ = alloc<float>(size_t(chunk_) * (picks_ * ff_));
            expert_up_ = alloc<float>(size_t(chunk_) * (picks_ * ff_));
            expert_hidden_ = clover_vision_workspace_aliases_ ? expert_gate_ : alloc<float>(size_t(chunk_) * (picks_ * ff_));
            if(!clover_vision_workspace_aliases_)expert_down_ = alloc<float>(size_t(chunk_) * (picks_ * n_));
            shared_gate_ = alloc<float>(size_t(chunk_) * (shared_ff_)); shared_up_ = alloc<float>(size_t(chunk_) * (shared_ff_));
            shared_hidden_ = clover_vision_workspace_aliases_ ? shared_gate_ : alloc<float>(size_t(chunk_) * (shared_ff_));
            shared_down_ = alloc<float>(size_t(chunk_) * (n_)); shared_scalar_ = alloc<float>(size_t(chunk_) * (1));
            // enqueue() completes attention/GDN before moe(). At that boundary
            // all these views are dead; the expert projections replace them.
            query_ = clover_vision_workspace_aliases_ ? expert_down_+phase.query_offset
                : alloc<float>(size_t(chunk_) * (heads_ * 256));
            // attention() and gdn() are alternative branches of one layer.
            // Each writes its gate before consuming it on the in-order queue.
            query_gate_ = clover_vision_workspace_aliases_
                ? qwen_vision::clover_layer_gate_alias(z_,size_t(heads_)*256,size_t(hv_)*128)
                : alloc<float>(size_t(chunk_) * (heads_ * 256));
            key_ = alloc<float>(size_t(chunk_) * (kv_heads_ * 256)); value_ = alloc<float>(size_t(chunk_) * (kv_heads_ * 256));
            const size_t attention_rows = size_t(heads_) * (kv_int8_ ? std::min(chunk_, verify_capacity) : 1);
            scores_ = alloc<float>(attention_rows * context);
            attention_partial_ = alloc<float>(attention_rows * ((context + 255) / 256) * 256);
            attention_ = clover_vision_workspace_aliases_ ? expert_down_+phase.attention_offset
                : alloc<float>(size_t(chunk_) * (heads_ * 256));
            logits_ = alloc<float>(size_t(g.n_vocab), true);
            expert_counts_ = alloc<int32_t>(experts_, true);
            expert_slots_ = alloc<int32_t>(experts_ * chunk_);
            const int max_tiles = (chunk_ * picks_ + 3) / 4 + experts_;
            tile_owners_ = alloc<int32_t>(max_tiles); tile_offsets_ = alloc<int32_t>(max_tiles); tile_count_ = alloc<int32_t>(1);
            token_input_ = alloc<int32_t>(chunk_, true);
            position_input_ = alloc<int32_t>(chunk_, true);
            if (vision_enabled_) {
                mrope_ = alloc<int32_t>(size_t(context_) * 3);
                const auto identity = qwen_vision::canonical::identity_positions(context_);
                q_.memcpy(mrope_, identity.data(), identity.size() * sizeof(int32_t)).wait_and_throw();
            }
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
            if (prefill_npu || std::getenv("STRATA_REVIEW_NPU_SPLIT")) {
                if (inspect_device(q_.get_device()).device_id != 0xB082)
                    throw std::runtime_error("NPU assistance is validated only with Arc B390");
                npu_ = std::make_unique<NpuProjection>(q_);
            }
#endif
            if(actual_allocations_.device!=chosen.bytes.owned_device||actual_allocations_.host_visible!=chosen.bytes.owned_host_visible||actual_allocations_.largest>chosen.bytes.largest)
                throw std::logic_error("Qwen context planner differs from actual engine allocations");
#ifdef TINYSTRATA_INTEL_MKL
            if (prefill_mkl) gemm_ = std::make_unique<PrefillGemm>(q_, context_, chunk_);
#endif
            if (decode_dp4a) dp4a_ = std::make_unique<DecodeDp4a>(q_, verify_xmx || prefill_int8_xmx || expert_layout, dense_split_k);
            if (use_graph && !profiling) prepare_graph();
            const auto device = inspect_device(q_.get_device());
            std::fprintf(stderr, "Intel causal Qwen: %s, architecture=%s, context=%d, KV=%s, MTP=%d, images=%s, weight_memory=%s\n",
                         device.name.c_str(), device.architecture.c_str(), context, kv_fp8_ ? "FP8_E4M3FN" : kv_int8_ ? "INT8" : "FP32",
                         mtp_depth_, vision_enabled_ ? "on" : "off", memory == WeightMemory::Host ? "host" : memory == WeightMemory::Shared ? "shared" : "device");
            print_memory("engine_ready", device);
            context_policy::report(chosen,context_limit,remaining_memory,selected_device.max_allocation);
            if(clover_vision_workspace_aliases_)std::fprintf(stderr,"Intel clover_image_workspace_aliases=1 saved_bytes=%llu MTP=0 single_owned_allocation_per_alias=1\n",
                (unsigned long long)(uint64_t(chunk_)*45568*sizeof(float)));
            print_moe_routes();
            if(clover_){std::printf("INFO clover_fused_t1_hint=%d\n",int(clover_fused_t1_hint_));std::fflush(stdout);}
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
        if (graph_enabled_ && !profiling_ && count == 1 && output_logits &&
            !qwen_vision::overlaps(image_input_, size_t(position_), size_t(count))) {
            int extent = std::min(context_, 128);
            while (extent <= position_) extent = std::min(context_, extent * 2);
            if (!decode_graph_ || extent != graph_extent_) { graph_extent_ = extent; prepare_graph(); }
        }
        bool complete = true;
        if (decode_graph_ && count == 1 && output_logits &&
            !qwen_vision::overlaps(image_input_, size_t(position_), size_t(count))) q_.ext_oneapi_graph(*decode_graph_);
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
    void print_moe_routes() const {
        std::printf("INFO moe_route_adaptive=%d\nINFO moe_route_observe=%d\nINFO moe_route_auto512=%d\n",
            int(moe_route_policy_.adaptive()),int(moe_route_policy_.observing()),int(moe_route_policy_.mode == moe_route_policy::Mode::Auto512));
        if (moe_route_policy_.observing()) std::printf(
            "INFO moe_route_observed=%llu\nINFO moe_route_eligible=%llu\nINFO moe_route_q8_projections=%llu\n"
            "INFO moe_route_observed_512=%llu\nINFO moe_route_eligible_512=%llu\n"
            "INFO moe_route_active_min=%d\nINFO moe_route_active_max=%d\nINFO moe_route_peak_max=%d\nINFO moe_route_hot_slots_max=%u\n",
            (unsigned long long)moe_route_observed_,(unsigned long long)moe_route_eligible_,(unsigned long long)moe_route_q8_projections_,
            (unsigned long long)moe_route_observed_512_,(unsigned long long)moe_route_eligible_512_,
            moe_route_observed_ ? moe_route_active_min_ : 0,moe_route_active_max_,moe_route_peak_max_,moe_route_hot_slots_max_);
        std::fflush(stdout);
    }
    int mtp_depth() const { return mtp_depth_; }
    const std::vector<int32_t>& draft_vocabulary() const { return draft_vocab_host_; }
    void prepare_draft_vocabulary(const float* target, const std::vector<int>& history) {
        if (!draft_vocab_limit_) return;
        const auto start = std::chrono::steady_clock::now();
        auto ids = gpu_support_sampling_ ? gpu_sampling_candidate().support(target, draft_vocab_limit_, history)
            : make_draft_vocabulary(target, vocabulary(), draft_vocab_limit_, history);
        const auto support_rank_done = host_profile::now();
        host_profile::add(host_profile::SupportCpu, start, support_rank_done);
        q_.wait_and_throw();
        const auto support_wait_done = host_profile::now();
        host_profile::add(host_profile::SupportWait, support_rank_done, support_wait_done);
        std::copy(ids.begin(), ids.end(), draft_vocab_ids_);
        draft_vocab_host_ = std::move(ids);
        host_profile::add(host_profile::SupportCopy, support_wait_done, host_profile::now());
        ++draft_vocab_updates_;
        draft_vocab_prepare_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    const char* kv_type() const { return kv_fp8_ ? "fp8" : kv_int8_ ? "int8" : "fp32"; }
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
        if (gpu_sampling_candidate_) gpu_sampling_candidate_->print();
        if (mtp_depth_) std::fprintf(stderr, "mtp_windows=%llu mtp_offered=%llu mtp_accepted=%llu mtp_draft_ms=%.3f mtp_verify_ms=%.3f mtp_commit_ms=%.3f\n",
            static_cast<unsigned long long>(mtp_windows_), static_cast<unsigned long long>(mtp_offered_), static_cast<unsigned long long>(mtp_accepted_), mtp_draft_ms_, mtp_verify_ms_, mtp_commit_ms_);
        if (draft_vocab_limit_) std::fprintf(stderr, "draft_vocab_limit=%d draft_vocab_updates=%llu draft_vocab_prepare_ms=%.3f target_full_vocabulary=1\n",
            draft_vocab_limit_, static_cast<unsigned long long>(draft_vocab_updates_), draft_vocab_prepare_ms_);
    }
    bool clover() const { return clover_; }
    bool vision() const { return vision_enabled_; }
    int image_pad() const { return image_pad_; }
    int embedding_width() const { return n_; }
    const qwen_vision::Input& images() const { return image_input_; }
    void images(qwen_vision::Input next) {
        if (!vision_enabled_) {
            if (!next.images.empty()) throw std::runtime_error("Intel engine was started without --vision");
            return;
        }
        if ((!next.images.empty() && next.positions.size() != size_t(context_) * 3) ||
            next.embeddings.size() > qwen_vision::max_embedding_bytes / sizeof(float))
            throw std::runtime_error("Invalid bounded Intel image input");
        q_.wait_and_throw();
        if (!next.positions.empty())
            q_.memcpy(mrope_, next.positions.data(), next.positions.size() * sizeof(int32_t)).wait_and_throw();
        else if (!image_input_.positions.empty()) {
            const auto identity = qwen_vision::canonical::identity_positions(context_);
            q_.memcpy(mrope_, identity.data(), identity.size() * sizeof(int32_t)).wait_and_throw();
        }
        image_input_ = std::move(next);
    }
    bool terminal(int token) const { return clover_ ? clover_causal::terminal(token) : token == terminal_token_; }
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
    bool turn_checkpoint_enabled() const { return turn_checkpoint_enabled_; }
    const std::vector<int>& turn_prefix() const { return turn_snapshot_.tokens; }
    size_t turn_position() const { return size_t(turn_snapshot_.position); }
    int consumed_tokens() const { return position_; }
    size_t turn_checkpoint_bytes() const { return turn_checkpoint_bytes_; }
    bool save_turn(const std::vector<int>& live) {
        if (!turn_checkpoint_enabled_ || live.empty() || live.size() != size_t(position_) || live.size() > size_t(context_))
            return false;
        // repair_mtp fills deferred MTP KV rows before they become part of the
        // snapshot's preserved prefix. It also preserves the raw last hidden.
        if (mtp_depth_) repair_mtp();
        q_.wait_and_throw();
        turn_snapshot_.invalidate();
        try {
            turn_snapshot_.bytes.resize(turn_checkpoint_bytes_);
            turn_snapshot_.tokens.reserve(size_t(context_));
            turn_snapshot_.tokens.assign(live.begin(), live.end());
        } catch (const std::bad_alloc&) {
            turn_snapshot_.invalidate();
            std::fprintf(stderr, "Intel turn_checkpoint skipped: insufficient host memory\n");
            return false;
        }
        size_t at = 0;
        const auto copy = [&](const float* source, size_t bytes) {
            if (bytes > turn_snapshot_.bytes.size() - at) throw std::runtime_error("Intel turn checkpoint exceeds its bound");
            q_.memcpy(turn_snapshot_.bytes.data() + at, source, bytes);
            at += bytes;
        };
        for (const auto& layer : layers_) if (layer.recurrent) {
            copy(layer.recurrent, size_t(hv_) * 128 * 128 * sizeof(float));
            copy(layer.convolution, size_t(2 * hk_ + hv_) * 128 * 3 * sizeof(float));
        }
        copy(x_ + size_t(last_hidden_row_) * n_, size_t(n_) * sizeof(float));
        if (mtp_depth_) copy(mtp_last_norm_, size_t(n_) * sizeof(float));
        if (at != turn_snapshot_.bytes.size()) throw std::runtime_error("Intel turn checkpoint size differs");
        q_.wait_and_throw();
        // Publish validity only after all independent copies have completed.
        turn_snapshot_.position = position_;
        return true;
    }
    void restore_turn(const std::vector<int>& live, const std::vector<int>& prompt) {
        // A cancelled chunk can require a rewind even when its live tokens
        // are a valid prefix. Validate the saved prefix independently of the
        // usual preference for direct live reuse.
        if (!qwen_vision::saved_prefix(live, prompt, turn_snapshot_.tokens,
                turn_position(), turn_checkpoint_enabled_) || turn_snapshot_.bytes.size() != turn_checkpoint_bytes_)
            throw std::runtime_error("Intel turn checkpoint prefix is no longer resident");
        q_.wait_and_throw();
        size_t at = 0;
        const auto copy = [&](float* target, size_t bytes) {
            if (bytes > turn_snapshot_.bytes.size() - at) throw std::runtime_error("Intel turn restore exceeds its bound");
            q_.memcpy(target, turn_snapshot_.bytes.data() + at, bytes);
            at += bytes;
        };
        for (auto& layer : layers_) if (layer.recurrent) {
            copy(layer.recurrent, size_t(hv_) * 128 * 128 * sizeof(float));
            copy(layer.convolution, size_t(2 * hk_ + hv_) * 128 * 3 * sizeof(float));
        }
        copy(x_, size_t(n_) * sizeof(float));
        if (mtp_depth_) copy(mtp_last_norm_, size_t(n_) * sizeof(float));
        if (at != turn_snapshot_.bytes.size()) throw std::runtime_error("Intel turn restore size differs");
        q_.wait_and_throw();
        position_ = turn_snapshot_.position;
        position_input_[0] = position_;
        mtp_pending_.clear();
        last_hidden_row_ = 0;
        logits_valid_ = false;
    }
    void reset() {
        turn_snapshot_.invalidate();
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

Sampling model_sampling(bool clover) {
    Sampling result;
    if (clover) {
        result.clover = true;
        result.top_k = 64;
        result.repeat = 1.05f;
        result.presence = .9f;
    }
    return result;
}

float real_option(const std::string& text) {
    size_t end = 0;
    const float value = std::stof(text, &end);
    if (end != text.size() || !std::isfinite(value)) throw std::runtime_error("Invalid finite sampling value");
    return value;
}
void sampling_option(Sampling& s, const std::string& field) {
    const auto equals = field.find('=');
    const std::string key = field.substr(0, equals), text = field.substr(equals + 1);
    if (key == "img_keys") {
        if (!qwen_vision::valid_image_keys(text)) throw std::runtime_error("Invalid Intel image cache keys");
        return; // Hints only: exact SVE1 bytes/grids/positions decide prefix identity.
    }
    if (key == "clover_mode" && s.clover) {
        if (text != "causal") throw std::runtime_error("Intel Clover supports causal D0 only");
        return;
    }
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
    } else if ((key == "mtp_auto" || key == "lookup_draft" ||
                (s.clover && (key == "clover_draft" || key == "clover_lookup"))) && value == 0) {
        // Explicitly disabled. Nonzero speculation remains unsupported.
    } else if (key == "ckpt" || key == "turn_at") {
        // Turn reuse hints. Only exact prefixes can restore the bounded,
        // independent in-memory snapshot; no on-disk session is advertised.
        if (value < 0 || value != std::floor(value)) throw std::runtime_error("Invalid turn-cache hint");
    } else throw std::runtime_error("Unsupported Intel request option: " + field);
}
int serve(Engine& engine, bool stochastic_mtp = false) {
    const auto device = engine.device_info();
    std::printf("INFO backend=sycl experimental=1 model_family=%s kv=%s mtp=%d vision=%d memory_model=%s single_weight_store=1 device_vendor=%u device_id=%u device_integrated=%d device_architecture=%s adapter_luid=%s adapter_node_mask=%u kv_used_tokens=%d\nREADY %d stop\n",
        engine.clover() ? "clover_moe" : "qwen35moe", engine.kv_type(), engine.mtp_depth(), int(engine.vision()), device.integrated ? "uma" : "discrete", device.vendor_id, device.device_id, int(device.integrated),
        device.architecture.c_str(), device.adapter_luid.empty() ? "unknown" : device.adapter_luid.c_str(),
        device.adapter_node_mask, engine.consumed_tokens(), engine.context());
    std::fflush(stdout);
    // Publish only committed model positions. Generated T tokens, transient MTP
    // verification rows, and the allocated KV capacity are different quantities.
    // The Python stdout pump already forwards dynamic INFO facts to /metrics.
    int reported_position = engine.consumed_tokens();
    const auto report_context_usage = [&] {
        const int position = engine.consumed_tokens();
        if (position == reported_position) return;
        std::printf("INFO kv_used_tokens=%d\n", position);
        std::fflush(stdout);
        reported_position = position;
    };
    strata::common::CommandInbox inbox;
    std::vector<int> live;
    if (engine.turn_checkpoint_enabled())
        std::fprintf(stderr, "Intel turn_checkpoint=enabled snapshot_bytes=%zu max_prefix_tokens=%d separate_mtp_rollback=1\n",
                     engine.turn_checkpoint_bytes(), engine.context());
    std::string line;
    while (inbox.next(line)) {
        if (line == "QUIT") return 0;
        if (line.empty()) continue;
        std::vector<int> prompt;
        qwen_vision::Input image_input;
        const bool image_request = line.rfind("GENI ", 0) == 0;
        Sampling sampling = model_sampling(engine.clover());
        int max_new = 0;
        size_t turn_at = 0;
        bool checkpoint_requested = true;
        try {
            if (!image_request && line.rfind("GEN ", 0) != 0)
                throw std::runtime_error("Expected GEN <max_new> [key=value ...] <ids> or GENI <max_new> [key=value ...] <file> <ids>");
            if (image_request && !engine.vision()) throw std::runtime_error("Intel engine was started without --vision");
            std::istringstream fields(line.substr(image_request ? 5 : 4));
            std::string field, ids, embeddings_path;
            if (!(fields >> field)) throw std::runtime_error("Missing output count");
            max_new = number(field.c_str(), engine.context());
            while (fields >> std::ws && fields.peek() != std::char_traits<char>::eof()) {
                const bool quoted_path = image_request && embeddings_path.empty() && fields.peek() == '"';
                if (!(fields >> std::quoted(field, '"', '\0'))) throw std::runtime_error("Invalid request field");
                if (!quoted_path && field.find('=') != std::string::npos && ids.empty() && embeddings_path.empty()) {
                    sampling_option(sampling, field);
                    if (engine.turn_checkpoint_enabled() && field.rfind("turn_at=", 0) == 0) {
                        const float value = real_option(field.substr(8));
                        if (value > engine.context()) throw std::runtime_error("Turn-cache boundary exceeds context");
                        turn_at = size_t(value);
                    } else if (engine.turn_checkpoint_enabled() && field.rfind("ckpt=", 0) == 0)
                        checkpoint_requested = real_option(field.substr(5)) != 0;
                }
                else if (image_request && embeddings_path.empty()) embeddings_path = field;
                else { if (!ids.empty()) ids += ' '; ids += field; }
            }
            prompt = parse_tokens(ids);
            if (!max_new || prompt.size() + size_t(max_new) > size_t(engine.context())) throw std::runtime_error("Request exceeds Intel context capacity");
            for (int token : prompt) if (token >= engine.vocabulary()) throw std::runtime_error("Token ID outside vocabulary");
            if (engine.turn_checkpoint_enabled() && turn_at > prompt.size()) throw std::runtime_error("Turn-cache boundary exceeds prompt");
            if (image_request) {
                std::string error;
                if (!qwen_vision::load(embeddings_path, prompt, engine.embedding_width(), engine.context(),
                        engine.image_pad(), image_input, error)) throw std::runtime_error(error);
            }
            if (!qwen_vision::canonical::whole_prefix(image_input, turn_at))
                throw std::runtime_error("Turn-cache boundary cannot split an image");
        } catch (const std::exception& error) {
            std::printf("ERR %s\n", error.what());
            std::fflush(stdout);
            continue; // Invalid requests leave the resident conversation intact.
        }
        const auto start = std::chrono::steady_clock::now();
        host_profile::reset();
        const auto speculation_start = engine.speculation_counts();
        ExecutionRequest active_request;
        if (engine.consumed_tokens() != int(live.size())) throw std::runtime_error("Intel consumed history differs from model position");
        const auto reuse = qwen_vision::choose(live, prompt, engine.turn_prefix(), engine.turn_position(),
            engine.turn_checkpoint_enabled() && checkpoint_requested, engine.images(), image_input, engine.embedding_width());
        const bool restored_turn = reuse.source == turn_checkpoint::Source::saved_turn;
        if (restored_turn) {
            engine.restore_turn(live, prompt);
            live.assign(engine.turn_prefix().begin(), engine.turn_prefix().end());
        } else if (reuse.source == turn_checkpoint::Source::reset) { engine.reset(); live.clear(); }
        engine.images(std::move(image_input));
        report_context_usage();
        const size_t reused = live.size();
        std::printf("RESUME %zu\n", reused);
        std::fflush(stdout);
        const float* logits = nullptr;
        for (size_t index = reused; index < prompt.size() && !inbox.cancelled().load();) {
            size_t end = std::min(index + size_t(engine.chunk()), prompt.size());
            if (engine.turn_checkpoint_enabled() && checkpoint_requested && index < turn_at && turn_at < end)
                end = turn_at;
            const int count = int(end - index);
            logits = engine.step_many(prompt.data() + index, count, index + count == prompt.size(), &inbox.cancelled());
            if (!logits) break;
            live.insert(live.end(), prompt.begin() + index, prompt.begin() + index + count);
            index += count;
            report_context_usage();
            if (engine.turn_checkpoint_enabled() && checkpoint_requested && turn_at && index == turn_at && !inbox.cancelled().load())
                engine.save_turn(live);
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::printf("PP %zu %zu %.3f\n", index, prompt.size(), elapsed);
            std::fflush(stdout);
        }
        const auto decode_start = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(decode_start - start).count();
        // A partially consumed image cannot act as an independently reusable
        // prefix. Restore the last complete turn, or discard the partial state.
        if (inbox.cancelled().load() && !qwen_vision::canonical::whole_prefix(engine.images(), live.size())) {
            if (qwen_vision::saved_prefix(live, prompt, engine.turn_prefix(), engine.turn_position(),
                    engine.turn_checkpoint_enabled()) &&
                qwen_vision::canonical::whole_prefix(engine.images(), engine.turn_position())) {
                engine.restore_turn(live, prompt);
                live.assign(engine.turn_prefix().begin(), engine.turn_prefix().end());
            } else { engine.reset(); live.clear(); engine.images({}); }
            report_context_usage();
        }
        const size_t committed_reused = std::min(reused, live.size());
        const size_t fresh = live.size() - committed_reused;
        int produced = 0;
        const char* finish = "length";
        std::vector<int> history = clover_causal::request_history(engine.clover(), prompt);
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
                    const int token = [&] {
                        host_profile::Scope draw(host_profile::DraftDraw);
                        return distribution.draw(draft_random);
                    }();
                    draft_tokens.push_back(token); draft_history.push_back(token);
                    draft_distributions.push_back(std::move(distribution));
                    return token;
                };
                size_t selected_row = 0;
                auto next = engine.speculate(ready.back(), max_new - produced, [&](const float* row) {
                    last_target_logits = row;
                    const auto target = sampling_distribution(row, engine.vocabulary(), sampling, history);
                    const int token = [&] {
                        host_profile::Scope draw(host_profile::TargetDrawVerify);
                        return rejection_sampling && selected_row < draft_distributions.size()
                            ? speculative_distribution::verify(target, draft_distributions[selected_row], draft_tokens[selected_row], random)
                            : sampling.temperature <= 0 ? target.tokens[0] : target.draw(random);
                    }();
                    ++selected_row;
                    history.push_back(token); return token;
                }, &inbox.cancelled(), propose);
                // speculate() has committed its accepted prefix or rolled back
                // an interrupted window before returning; never report inside it.
                report_context_usage();
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
            if (index + 1 < max_new) {
                logits = engine.step(token);
                live.push_back(token);
                report_context_usage();
            }
        }
        report_context_usage(); // Also covers cancelled/empty prefill and terminal output.
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
        engine.print_spec_stats();
        engine.print_moe_routes();
        host_profile::print();
        const auto speculation_end = engine.speculation_counts();
        std::printf("DONE %d %zu %.3f %.3f %s %llu %llu %zu %d 0 0 0 0.0 %zu 0\n", produced, prompt.size(), prompt_ms, decode_ms, finish,
            static_cast<unsigned long long>(speculation_end.first - speculation_start.first),
            static_cast<unsigned long long>(speculation_end.second - speculation_start.second), committed_reused, int(restored_turn), fresh);
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
        // Parse this isolated probe before ordinary early-return options, so
        // --device N selects the same device on either side of --device-info.
        if (std::any_of(argv + 1, argv + argc, [](const char* arg) { return std::string(arg) == "--device-info"; })) {
            unsigned selected = 0;
            for (int i = 1; i < argc; ++i) {
                const std::string option(argv[i]);
                if (option == "--device-info") continue;
                if (option != "--device" || i + 1 >= argc)
                    throw std::runtime_error("--device-info accepts only an optional --device N");
                selected = unsigned(strata::intel::number(argv[++i], 1024));
            }
            // Device enumeration/properties only: no queue, model, USM or
            // memory-budget query is created by this command.
            const auto info = strata::intel::inspect_device(strata::intel::select_device(selected));
            const auto quoted = [](const std::string& value) {
                constexpr char hex[] = "0123456789abcdef";
                std::string out = "\"";
                for (unsigned char c : value) {
                    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
                    else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
                    else out += char(c);
                }
                return out + '"';
            };
            std::printf("{\"schema\":1,\"backend\":\"sycl\",\"device_index\":%u,\"vendor_id\":%u,\"device_id\":%u,\"integrated\":%s,"
                        "\"architecture\":%s,\"adapter_luid\":%s,\"adapter_node_mask\":%u,\"name\":%s,\"driver\":%s,"
                        "\"capabilities\":{\"baseline\":%s,\"esimd\":%s,\"matrix\":%s,\"local_memory\":%llu,\"max_work_group\":%llu}}\n",
                selected, info.vendor_id, info.device_id, info.integrated ? "true" : "false", quoted(info.architecture).c_str(),
                quoted(info.adapter_luid).c_str(), info.adapter_node_mask, quoted(info.name).c_str(), quoted(info.driver).c_str(),
                baseline_limitation(info.capabilities) ? "false" : "true", info.capabilities.esimd ? "true" : "false",
                info.capabilities.matrix ? "true" : "false", (unsigned long long)info.capabilities.local_memory,
                (unsigned long long)info.capabilities.max_work_group);
            return 0;
        }
        std::string path, token_list, token_file, dump_path;
        int count = 32, context = 0, device = 0, chunk = 128, mtp_depth = 0, draft_vocab = 0;
        bool serve = false, use_graph = true, profiling = false, prefill_xmx = false, prefill_npu = false, prefill_mkl = false, decode_dp4a = false, verify_xmx = false, kv_int8 = false, prefill_int8_xmx = false, vocab_xmx = false, verify_mkl_attention = false, stochastic_mtp = false, expert_layout = false, device_state = false, verify_graph = false, dense_layout = false, dense_split_k = false, experimental_tiel = false, experimental_clover = false, explicit_kv = false, vision = false, kv_fp8 = false;
        std::string clover_mode = "causal";
        int clover_drafts = 0;
        bool clover_options = false;
        strata::intel::WeightMemory memory = strata::intel::WeightMemory::Host;
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--serve") { serve = true; continue; }
            if (option == "--vision") { vision = true; continue; }
            if (option == "--experimental-tiel") { experimental_tiel = true; continue; }
            if (option == "--experimental-clover") { experimental_clover = true; continue; }
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
            if (option == "--host-profile") { strata::intel::host_profile::enabled = true; continue; }
            if (option == "--no-graph") { use_graph = false; continue; }
            if (option == "--help") {
                std::puts("tinystrata-intel --model MODEL [--serve | --tokens IDS | --tokens-file FILE] [--n N] [--max-context auto|N] [--device N] [--chunk 1..1024] [--mtp 0..7] [--kv-type fp32|int8|fp8] [--draft-vocab 0|64..4096] [--dump-full-logits FILE] [--experimental-tiel | --experimental-clover] [--vision] [--host-profile]\ntinystrata-intel --device-info [--device N]\nExperimental Intel multimodal driver. Explicit FP8 uses E4M3FN with FP16 block32 scales for Qwen3.6 only. Qwen/Tiel KV defaults to FP32; MTP and draft vocabulary selection default to 0 (off); --vision enables bounded SVE1 image input through GENI; ordinary GEN remains available. A nonzero draft vocabulary requires --serve --mtp-stochastic --mtp --vocab-xmx. --host-profile enables host phase wall timers without changing decode graphs. --experimental-tiel selects the separately pinned Ornith IQ3_XXS artifact. --experimental-clover selects the pinned Clover AD-IQ3_S-IQ3_XXS artifact with INT8 KV and causal D0 only (--clover-mode causal --clover-draft 0). It preserves Clover response-only penalties and stopping ramps; Clover MTP remains disabled; Clover images use the Qwen-compatible projector and interleaved M-RoPE.");
                return 0;
            }
            if (option == "--capabilities") {
#ifdef TINYSTRATA_INTEL_MKL
                constexpr const char* mkl = "true";
#else
                constexpr const char* mkl = "false";
#endif
#ifdef TINYSTRATA_INTEL_INT8_XMX
                constexpr const char* int8_xmx = "true";
                constexpr const char* xmx_types = "[14,18,21,22]";
#else
                constexpr const char* int8_xmx = "false";
                constexpr const char* xmx_types = "[]";
#endif
#if defined(TINYSTRATA_INTEL_MKL) && defined(TINYSTRATA_INTEL_INT8_XMX)
                constexpr const char* profiles = "{\"b390-qwen36\":1,\"b390-tiel-coder\":1,\"b390-clover\":1}";
                constexpr const char* models = "[\"qwen3.6-35b-a3b\",\"tiel-coder-35b-a3b\",\"clover-35b-a3b-preview\"]";
#else
                constexpr const char* profiles = "{}";
                constexpr const char* models = "[\"qwen3.6-35b-a3b\",\"clover-35b-a3b-preview\"]";
#endif
#ifdef TINYSTRATA_INTEL_AOT_TARGET
                constexpr const char* aot_target = TINYSTRATA_INTEL_AOT_TARGET;
#else
                constexpr const char* aot_target = "";
#endif
                std::printf("{\"backend\":\"sycl\",\"experimental\":true,\"model_families\":[],\"kv_types\":[\"fp32\",\"int8\",\"fp8\"],\"experimental_kv_models\":{\"fp8\":[\"qwen3.6-35b-a3b\"]},\"fp8_kv_format\":\"e4m3fn_block32_fp16_scale\",\"images\":true,\"image_protocol\":\"geni_sve1\",\"vision_contract_version\":1,\"vision_max_embedding_bytes\":268435456,\"mtp\":true,\"experimental_mtp_max_depth\":7,\"serve\":true,"
                            "\"capabilities_schema\":1,\"context_auto\":true,\"max_context\":262144,\"experimental_models\":%s,\"experimental_profiles\":%s,"
                            "\"experimental_model_contracts\":{\"clover-35b-a3b-preview\":{\"version\":2,\"architecture\":\"clover_moe\",\"kv_types\":[\"int8\"],\"mtp\":false,\"experimental_mtp_max_depth\":0,\"images\":true,\"thinking\":\"on_off\",\"mode\":\"causal_d0\",\"context_max\":131072}},"
                            "\"build\":{\"mkl\":%s,\"int8_xmx\":%s,\"aot_target\":\"%s\"},"
                            "\"weight_types\":{\"scalar\":[0,1,8,10,11,14,18,21,22,23,30],\"dp4a\":[8,10,11,14,18,21,22,23],\"int8_xmx\":%s}}\n",
                            models, profiles, mkl, int8_xmx, aot_target, xmx_types);
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing argument for " + option);
            if (option == "--gguf" || option == "--model") path = argv[++i];
            else if (option == "--tokens") token_list = argv[++i];
            else if (option == "--tokens-file") token_file = argv[++i];
            else if (option == "--dump-full-logits") dump_path = argv[++i];
            else if (option == "--n") count = strata::intel::number(argv[++i], 16384);
            else if (option == "--context" || option == "--max-context") context = strata::intel::context_policy::parse(argv[++i],strata::intel::context_policy::full_limit);
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
                if (kind != "fp32" && kind != "int8" && kind != "fp8") throw std::runtime_error("Intel supports explicit --kv-type fp32, int8 or experimental Qwen3.6 fp8");
                kv_fp8 = kind == "fp8";
                kv_int8 = kind != "fp32";
                explicit_kv = true;
            }
            else if (option == "--clover-mode") { clover_mode = argv[++i]; clover_options = true; }
            else if (option == "--clover-draft") { clover_drafts = strata::intel::number(argv[++i], 31); clover_options = true; }
            else if (option == "--draft-vocab") draft_vocab = strata::intel::number(argv[++i], 4096);
            else if (option == "--mtp") mtp_depth = strata::intel::number(argv[++i], 7);
            else if (option == "--batch") { if (strata::intel::number(argv[++i], 16) != 1) throw std::runtime_error("The initial Intel route serves one request at a time"); }
            else throw std::runtime_error("Unsupported Intel option: " + option);
        }
        if (kv_fp8 && (experimental_tiel || experimental_clover)) throw std::runtime_error("FP8 KV is experimental Qwen3.6 only");
        if (path.empty()) throw std::runtime_error("--gguf is required");
        if (vision && !serve) throw std::runtime_error("Intel --vision requires --serve and GENI image input");
        if (experimental_clover) {
            if (experimental_tiel) throw std::runtime_error("Clover and Tiel require separate model selections");
            if (!explicit_kv) kv_int8 = true;
            strata::intel::clover_causal::validate_launch(kv_int8, mtp_depth, draft_vocab, stochastic_mtp,
                                                        prefill_npu, clover_mode, clover_drafts);
        } else if (clover_options) throw std::runtime_error("Clover options require --experimental-clover");
        if (draft_vocab && (!serve || !stochastic_mtp || !mtp_depth || !vocab_xmx || draft_vocab < 64))
            throw std::runtime_error("--draft-vocab requires --serve, --mtp-stochastic, --mtp, --vocab-xmx and 64..4096 rows");
        if (stochastic_mtp && (!serve || !mtp_depth)) throw std::runtime_error("--mtp-stochastic requires --serve and --mtp >= 1");
        if (serve) {
            if (!token_list.empty() || !token_file.empty() || !dump_path.empty()) throw std::runtime_error("Token CLI options cannot accompany --serve");
            std::optional<strata::intel::ExecutionRequest> loading(std::in_place);
            strata::intel::Engine engine(path, context, unsigned(device), memory, chunk, use_graph, profiling, prefill_xmx, prefill_npu, prefill_mkl, decode_dp4a, mtp_depth, verify_xmx, kv_int8, prefill_int8_xmx, vocab_xmx, verify_mkl_attention, expert_layout, device_state, verify_graph, dense_layout, dense_split_k, draft_vocab, experimental_tiel, experimental_clover, vision, kv_fp8);
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
        if (context>0&&tokens.size() + size_t(count) > size_t(context)) throw std::runtime_error("Prompt and requested output exceed --context");
        strata::intel::ExecutionRequest active_request;
        strata::intel::Engine engine(path, context, unsigned(device), memory, chunk, use_graph, profiling, prefill_xmx, prefill_npu, prefill_mkl, decode_dp4a, mtp_depth, verify_xmx, kv_int8, prefill_int8_xmx, vocab_xmx, verify_mkl_attention, expert_layout, device_state, verify_graph, dense_layout, dense_split_k, draft_vocab, experimental_tiel, experimental_clover, vision, kv_fp8);
        if(tokens.size()+size_t(count)>size_t(engine.context()))throw std::runtime_error("Prompt and requested output exceed allocated Intel context");
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
        } else {
          const auto sampling = strata::intel::model_sampling(engine.clover());
          std::vector<int> response;
          std::mt19937_64 random(0);
          for (int index = 0; index < count; ++index) {
            const int token = engine.clover()
                ? strata::intel::sample(logits, engine.vocabulary(), sampling, response, random)
                : int(std::max_element(logits, logits + engine.vocabulary()) - logits);
            std::printf("TOKEN %d\n", token);
            std::fflush(stdout);
            ++produced;
            response.push_back(token);
            if (engine.terminal(token)) break;
            if (index + 1 < count) { logits = engine.step(token); ++decode_steps; }
          }
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
