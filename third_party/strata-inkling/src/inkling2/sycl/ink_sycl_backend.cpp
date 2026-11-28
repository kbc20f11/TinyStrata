// engine/src/inkling2/sycl/ink_sycl_backend.cpp - InkBackend plugin for Intel Arc SYCL (dg2 + bmg_g31 B70)
// CRITICAL BATTLEMAGE B70 RULE: Zero reqd_sub_group_size(8); strictly SG16!
// MIT License - Copyright (c) 2026 Strata contributors
#include "../ink_backend.h"
#include "../ink_vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define INK_SYCL_WITH_MODEL_FILE 1
#include "ink_sy_fit.h"
#include "ink_sy_runner.h"

namespace {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor(101)))
#endif
void ink_sycl_early_l0_v1_init() {
    ink_sycl::ensure_l0_v1_default();
}
struct InkSyclEarlyInitGuard {
    InkSyclEarlyInitGuard() { ink_sycl::ensure_l0_v1_default(); }
};
static InkSyclEarlyInitGuard g_ink_sycl_early_init_guard;
}  // namespace

namespace ink2 {

class SyclInkBackend final : public InkBackend {
public:
    ~SyclInkBackend() override {
        if (far_mode_initialized_) {
            const auto& counters = ink_sycl::far_attention_dispatches();
            std::fprintf(stderr, "[ink_sycl] far_attn_mode=%d candidate_kernel_dispatches=%llu legacy_kernel_dispatches=%llu "
                "counter_scope=enabled_mode_dispatches skipped_cells=not_measured\n", (int)far_mode_,
                (unsigned long long)counters.candidate.load(std::memory_order_relaxed),
                (unsigned long long)counters.legacy.load(std::memory_order_relaxed));
        }
    }
    const char* backend_name() const noexcept override { return mod_.ep2_active ? "sycl-ep2" : "sycl"; }

    bool dspark_ready() const override { return runner_.use_dspark; }

    std::string device_info() const override {
        const auto& d = ink_sycl::dev_caps();
        std::string info = d.name + " (" + d.platform + ", " +
               (ink_sycl::is_uma() ? std::string("UMA physical RAM pool") :
                std::to_string((int)(d.global_mem / (1024ull * 1024ull * 1024ull))) + " GiB") + ", SG16)";
        if (mod_.ep2_active) {
            info += " + " + mod_.ep2_dev1_name + " (" +
                    std::to_string((int)(mod_.ep2_dev1_mem / (1024ull * 1024ull * 1024ull))) +
                    " GiB) [EP-2GPU split=" + std::to_string(mod_.ep2_split) + ", " +
                    (mod_.ep2_p2p ? "p2p" : "pinned") + "]";
        }
        return info;
    }

    BackendStats stats() const override {
        BackendStats s;
        s.total_experts = (size_t)mod_.man.c.n_phys * (size_t)mod_.man.c.n_experts;
        s.vram_experts_loaded = (size_t)mod_.n_vram_experts;
        s.ram_experts_loaded = (size_t)mod_.n_host_experts;
        // These legacy fields describe dedicated VRAM. UMA is not another RAM
        // capacity; leave them unavailable (zero) and report owned bytes below.
        if (!ink_sycl::is_uma()) {
            s.vram_used_bytes = ink_sycl::dev_bytes_allocated() + mod_.vram_expert_bytes_g1;
            s.vram_total_bytes = ink_sycl::dev_caps().global_mem + (mod_.ep2_active ? mod_.ep2_dev1_mem : 0);
        }
        s.active_slots = active_slots_;
        s.max_slots = runner_.max_slots;
        s.kv_dtype = kv_dtype_;
        return s;
    }

    std::optional<BackendMemoryStats> memory_stats() const override {
        if (!ink_sycl::is_uma()) return std::nullopt;
        BackendMemoryStats s;
        s.expert_cache_bytes = mod_.vram_expert_bytes;
        s.expert_cache_full_bytes = mod_.vram_expert_bytes;
        s.kv_logical_bytes = runner_.kv_storage_bytes;
        s.kv_mapped_bytes = runner_.kv_storage_bytes;
        s.kv_grow = false;
        for (int rows : slot_rows_) s.kv_committed_rows += rows;
        return s;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        if (!opts.kv_dtype_auto && opts.kv_dtype != KvDtype::FP16) {
            err = "sycl: this backend stores FP16 K/V only; use --kv fp16 (FP8/k8v8/k8v4 are not implemented)";
            return false;
        }
        ink_sycl::apply_gpu_index(opts.gpu_index, opts.gpus);
        far_mode_ = ink_sycl::far_attention_requested();
        far_mode_initialized_ = true;
        std::fprintf(stderr, "[ink_sycl] far_attn_mode=%d admission=global_prefill_Tge16_rel1024 baseline_default=0\n", (int)far_mode_);
        std::printf("INFO inkling_far_attn=%d\n", (int)far_mode_);
        kv_dtype_ = KvDtype::FP16;
        if (ink_sycl::is_uma() && opts.gpus != 1) {
            err = "sycl: UMA qualification requires one selected GPU and one compressed weight store";
            return false;
        }
        const int par = std::max(1, std::min(64, opts.parallel));
        const int ctx = std::max(256, opts.max_ctx);
        const int batch_cap = std::max(256, std::max(opts.chunk, par));
        // Exact runner + KV + DSpark + Vision + XMX/Hybrid reserve (ink_sy_fit.h), so an 8-16 GB Arc card is told BEFORE the runner allocates
        // that the --parallel / --max-context asked for does not fit (a start used to die with "dalloc ...: out of device memory").
        const auto& mc = model.m.c;
        ink_sycl::InkSyFitGeom fit;
        const int core_len = mc.core_end - mc.core_start;
        for (int v = 0; v < mc.n_visits; ++v) {
            const int ip = (v < mc.core_start) ? v
                         : (v < mc.core_start + core_len * mc.num_passes)
                             ? (mc.core_start + (v - mc.core_start) % core_len)
                             : (mc.core_end + (v - mc.core_start - core_len * mc.num_passes));
            fit.visit_is_swa.push_back(mc.is_swa[ip] ? 1 : 0);
        }
        fit.swa_win = mc.swa_win;
        fit.n_kv = mc.n_kv;
        fit.head_dim = mc.head_dim;
        fit.hidden = mc.hidden;
        fit.dspark = opts.enable_dspark && model.m.dspark.present;
        fit.vision = !opts.vision_path.empty();
        ink_sycl::ensure_l0_v1_default();
#ifdef INK_SY_TINYSTRATA_MEMORY
        if (fit.vision && ink_sycl::is_uma()) {
            // Admit the resident encoder before choosing language KV/Expert
            // capacity. The estimate keeps only its remaining scratch reserve.
            (void)ink_sycl::uma_ledger();
            const size_t before = ink_sycl::dev_bytes_allocated();
            if (!runner_.vis.load(opts.vision_path, err)) return false;
            fit.resident_vision_bytes = ink_sycl::dev_bytes_allocated() - before;
        }
#endif
        const size_t extra_reserve = ink_sycl::ink_sy_runner_reserve(fit, par, ctx, opts.chunk);
        if (model.kind == ModelSourceKind::Strata) {
            if (!mod_.load(model.path, opts.vram_experts_gb, err, extra_reserve, opts.gpus, opts.gpu_ids, opts.ep_split)) return false;
        } else {
            if (!mod_.load_from_model_file(model, opts.vram_experts_gb, err, extra_reserve)) return false;
        }
        {
            const size_t gpu_mem = ink_sycl::dev_caps().global_mem;
            const size_t used = ink_sycl::dev_bytes_allocated();
            size_t avail = gpu_mem > used ? gpu_mem - used : 0;
#ifdef INK_SY_TINYSTRATA_MEMORY
            if (ink_sycl::is_uma()) avail = (size_t)ink_sycl::uma_remaining_bytes();
#endif
            const std::string why = ink_sycl::ink_sy_fit_message(fit, par, ctx, opts.chunk, avail);
            if (!why.empty()) {
                err = why;
                return false;
            }
        }
        if (!runner_.init(mod_, par, ctx, batch_cap, opts.enable_dspark, err)) {
            return false;
        }
        if (fit.vision && !runner_.vis.loaded) {
            if (!runner_.vis.load(opts.vision_path, err)) return false;
        }
        slot_used_.assign((size_t)par, 0);
        slot_rows_.assign((size_t)par, 0);
#ifdef INK_SY_TINYSTRATA_MEMORY
        if (ink_sycl::is_uma()) {
            const auto& device = ink_sycl::selected_memory_device();
            const auto memory = strata::intel::memory_budget(device);
            const auto& ledger = ink_sycl::uma_ledger();
            std::fprintf(stderr, "[ink_sycl] memory_model=uma adapter_luid=%s adapter_node_mask=%u "
                "owned_device_bytes=%llu owned_host_usm_bytes=%llu owned_total_bytes=%llu "
                "initial_allocation_budget=%llu remaining_allocation_budget=%llu "
                "available_physical=%llu available_commit=%llu wddm_known=%d wddm_available=%llu "
                "os_headroom_bytes=%llu wddm_headroom_bytes=%llu kv_storage=fp16 kv_storage_bytes=%zu\n",
                device.adapter_luid.empty() ? "unknown" : device.adapter_luid.c_str(), device.adapter_node_mask,
                (unsigned long long)ledger.device_bytes, (unsigned long long)ledger.host_usm_bytes,
                (unsigned long long)ledger.owned_bytes(), (unsigned long long)ledger.initial_budget,
                (unsigned long long)ink_sycl::uma_remaining_bytes(),
                (unsigned long long)memory.available_physical, (unsigned long long)memory.available_commit,
                (int)memory.wddm_known, (unsigned long long)memory.wddm_available,
                (unsigned long long)(4 * strata::intel::GiB),
                (unsigned long long)strata::intel::uma_wddm_headroom, runner_.kv_storage_bytes);
        }
#endif
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= runner_.max_slots) { err = "sycl slot out of range"; return false; }
        if (!slot_used_[slot]) { slot_used_[slot] = 1; ++active_slots_; }
        runner_.reset_slot(slot);
        slot_rows_[slot] = 0;
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < runner_.max_slots && slot_used_[slot]) {
            slot_used_[slot] = 0;
            --active_slots_;
            runner_.reset_slot(slot);
            slot_rows_[slot] = 0;
        }
    }

    void reset_slot(int slot) override { runner_.reset_slot(slot); slot_rows_.at(slot) = 0; }
    void snapshot_slot(int slot) override { runner_.snapshot_slot(slot); }
    void rollback_slot(int slot, int keep_len) override {
        runner_.rollback_slot(slot, keep_len);
        slot_rows_.at(slot) = keep_len;
    }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        if (batch.n_tokens <= 0) return true;
        const int M_tot = batch.n_tokens;
        const int D = mod_.man.c.hidden;
        const int V = mod_.man.c.vocab_unpadded;
        const int cap = runner_.max_batch;

        int dst_logit_row = 0;
        for (int off = 0; off < M_tot; off += cap) {
            const int M = std::min(cap, M_tot - off);
            std::vector<uint8_t> is_img;
            std::vector<float> img_emb;
            bool any_img = false;
            for (int m = 0; m < M; ++m) if (batch.img_embed_idx[off + m] >= 0) { any_img = true; break; }
            if (any_img && batch.img_embeds) {
                is_img.assign((size_t)M, 0);
                img_emb.assign((size_t)M * D, 0.0f);
                for (int m = 0; m < M; ++m) {
                    const int idx = batch.img_embed_idx[off + m];
                    if (idx >= 0) {
                        is_img[m] = 1;
                        std::memcpy(img_emb.data() + (size_t)m * D,
                                    batch.img_embeds + (size_t)idx * D,
                                    (size_t)D * sizeof(float));
                    }
                }
            }

            std::vector<int> want_idx;
            if (logits_out != nullptr) {
                for (int m = 0; m < M; ++m) if (batch.want_logits[off + m] != 0) want_idx.push_back(m);
            }
            const int n_want = (int)want_idx.size();
            float* chunk_lg = (n_want > 0 && logits_out != nullptr) ? (logits_out + (size_t)dst_logit_row * V) : nullptr;
            int mode = 0;
            if (n_want == 1 && want_idx[0] == M - 1) mode = 1;
            else if (n_want > 0) mode = 2;

            if (!runner_.forward_batch(batch.token_ids.data() + off, batch.slot_ids.data() + off, batch.pos.data() + off, M,
                                       any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                       mode, chunk_lg, nullptr, batch.capture_taps,
                                       (mode == 2) ? want_idx.data() : nullptr, n_want)) {
                err = "sycl forward_batch failed"; return false;
            }
            dst_logit_row += n_want;
            for (int row = off; row < off + M; ++row) {
                const int slot = batch.slot_ids[row];
                slot_rows_.at(slot) = std::max(slot_rows_.at(slot), batch.pos[row] + 1);
            }
        }
        (void)taps_out;
        return true;
    }

    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft,
                     int32_t* out_draft_ids, std::string& /*err*/) override {
        if (!runner_.use_dspark || max_draft <= 0) return 0;
        int tmp[7]{};
        int n = runner_.dspark_draft_block8(slot, anchor_tok, anchor_pos, tmp);
        int ret = std::min(n, max_draft);
        for (int i = 0; i < ret; ++i) out_draft_ids[i] = tmp[i];
        return ret;
    }

    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (!runner_.vis.loaded || img.file.empty()) return false;
        int ntok = 0;
        return runner_.vis.encode_image_file(img.file, out_embeds, ntok, err);
    }

    bool selfcheck(std::string& err) override {
        ink_sycl::apply_gpu_index(pending_gpu_index_, pending_gpus_);
        if (mod_.path.empty() && runner_.mod == nullptr) {
            return selfcheck_device(err);
        }
        StepBatch b;
        b.push(0, 0, 1, true);
        std::vector<float> lg((size_t)mod_.man.c.vocab_unpadded);
        runner_.reset_slot(0);
        if (!forward_batch(b, lg.data(), nullptr, err)) return false;
        for (float v : lg) {
            if (!std::isfinite(v)) { err = "sycl selfcheck produced non-finite logits"; return false; }
        }
        runner_.reset_slot(0);
        return true;
    }

    bool selfcheck_device(std::string& err) {
        constexpr int N = 32, K = 64, M = 2;
        std::vector<float> w((size_t)N * K), x((size_t)M * K);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float)(i % 97)) * 0.5f - 0.25f;
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i % 89)) * 0.5f;
        float* wd = sycl::malloc_device<float>(w.size(), ink_sycl::Q());
        float* xd = sycl::malloc_device<float>(x.size(), ink_sycl::Q());
        float* yd = sycl::malloc_device<float>((size_t)M * N, ink_sycl::Q());
        if (!wd || !xd || !yd) {
            if (wd) sycl::free(wd, ink_sycl::Q());
            if (xd) sycl::free(xd, ink_sycl::Q());
            if (yd) sycl::free(yd, ink_sycl::Q());
            err = "sycl selfcheck: device USM allocation failed";
            return false;
        }
        ink_sycl::h2d_sync(wd, w.data(), w.size() * sizeof(float));
        ink_sycl::h2d_sync(xd, x.data(), x.size() * sizeof(float));
        ink_sycl::dzero(yd, (size_t)M * N * sizeof(float));
        ink_sycl::k_linear_f32_sg16(wd, N, K, xd, yd, M);
        std::vector<float> y((size_t)M * N, 0.0f);
        ink_sycl::d2h(y.data(), yd, y.size() * sizeof(float));
        ink_sycl::wait();
        sycl::free(wd, ink_sycl::Q());
        sycl::free(xd, ink_sycl::Q());
        sycl::free(yd, ink_sycl::Q());
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k) ref += w[(size_t)n * K + k] * x[(size_t)m * K + k];
                const float got = y[(size_t)m * N + n];
                if (!std::isfinite(got) || std::fabs(ref - got) > 1e-4f * std::max(1.0f, std::fabs(ref))) {
                    err = "sycl selfcheck F32 SG16 linear mismatch at (" + std::to_string(m) + "," + std::to_string(n) +
                          "): ref=" + std::to_string(ref) + " got=" + std::to_string(got);
                    return false;
                }
            }
        }
        return true;
    }

private:
    friend std::unique_ptr<InkBackend> make_sycl_backend(const BackendOpts* opts);
    ink_sycl::InkSyModel mod_;
    ink_sycl::InkSyRunner runner_;
    std::vector<uint8_t> slot_used_;
    std::vector<int> slot_rows_;
    int active_slots_ = 0;
    KvDtype kv_dtype_ = KvDtype::FP16;
    int pending_gpu_index_ = 0;
    int pending_gpus_ = 1;
    bool far_mode_ = false;
    bool far_mode_initialized_ = false;
};

std::unique_ptr<InkBackend> make_sycl_backend(const BackendOpts* opts) {
    auto b = std::make_unique<SyclInkBackend>();
    if (opts) {
        b->pending_gpu_index_ = opts->gpu_index;
        b->pending_gpus_ = opts->gpus;
    }
    return b;
}

}  // namespace ink2
