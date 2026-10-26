// TinyStrata's LFM2.5 text engine and API line protocol.
// Sampling and protocol conventions adapt PolyStrata/Strata (MIT), commit
// 26e4cb5df7996edfb5ff549a63b82d8581e1e658. See LICENSE and NOTICE.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
#include "engine.hpp"
#include "strata/common/context_policy.hpp"
#include "strata/lfm2/vram.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
using strata::lfm2::Engine;
using strata::lfm2::Options;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Sampling {
    double temperature = 0, top_p = 1, min_p = 0, repeat = 1, frequency = 0, presence = 0;
    int top_k = 50, last_n = 0;
    uint64_t seed = 0;
};

int sample(const float* logits, int vocab, const Sampling& s, const std::vector<int32_t>& history,
           std::mt19937_64& rng) {
    std::vector<double> scores(logits, logits + vocab);
    for (double x : scores)
        if (!std::isfinite(x)) throw std::runtime_error("nonfinite logits");
    if (s.last_n && (s.repeat != 1 || s.frequency != 0 || s.presence != 0)) {
        std::unordered_map<int, int> counts;
        const size_t start = s.last_n < 0 || (size_t) s.last_n >= history.size() ? 0 : history.size() - s.last_n;
        for (size_t i = start; i < history.size(); ++i) ++counts[history[i]];
        for (auto [id, count] : counts) {
            double& v = scores[(size_t) id];
            v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= count * s.frequency + s.presence;
        }
    }
    for (double x : scores)
        if (!std::isfinite(x)) throw std::runtime_error("nonfinite penalized logits");
    if (s.temperature <= 0) return (int) (std::max_element(scores.begin(), scores.end()) - scores.begin());
    std::vector<int> ids((size_t) vocab);
    std::iota(ids.begin(), ids.end(), 0);
    const int k = s.top_k > 0 ? std::min(s.top_k, vocab) : vocab;
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    });
    ids.resize((size_t) k);
    // The standard sampler order: penalties, top-k, top-p, min-p, temperature.
    if (s.top_p < 1) {
        double sum = 0, cum = 0;
        for (int id : ids) sum += std::exp(scores[id] - scores[ids[0]]);
        size_t n = 0;
        do { cum += std::exp(scores[ids[n++]] - scores[ids[0]]) / sum; } while (n < ids.size() && cum < s.top_p);
        ids.resize(n);
    }
    if (s.min_p > 0) {
        const double floor = scores[ids[0]] + std::log(s.min_p);
        while (ids.size() > 1 && scores[ids.back()] < floor) ids.pop_back();
    }
    std::vector<double> probs;
    for (int id : ids) probs.push_back(std::exp((scores[id] - scores[ids[0]]) / s.temperature));
    return ids[std::discrete_distribution<size_t>(probs.begin(), probs.end())(rng)];
}

long long integer(const std::string& value, const char* name) {
    size_t used = 0;
    long long n = std::stoll(value, &used);
    if (used != value.size()) throw std::invalid_argument(std::string(name) + ": invalid integer");
    return n;
}

int bounded_integer(const std::string& value, const char* name, int lo, int hi) {
    const long long n = integer(value, name);
    if (n < lo || n > hi) throw std::invalid_argument(std::string(name) + ": integer outside supported range");
    return (int) n;
}

uint64_t seed_integer(const std::string& value) {
    if (value.empty() || value[0] == '-') throw std::invalid_argument("seed must be an unsigned integer");
    size_t used = 0;
    const uint64_t n = std::stoull(value, &used);
    if (used != value.size()) throw std::invalid_argument("invalid seed");
    return n;
}

std::vector<int> stop_tokens(const Engine& e) {
    std::vector<int> ids;
    for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                            "tokenizer.ggml.eom_token_id"}) {
        const int64_t id = e.meta_u(key, key == std::string("tokenizer.ggml.eos_token_id") ? 124900 : -1);
        if (id >= 0 && id < e.n_vocab()) ids.push_back((int) id);
    }
    // LFM2.5's end-of-text is separate from its ChatML end-of-message EOS.
    if (124895 < e.n_vocab()) ids.push_back(124895);
    return ids;
}

bool is_stop(int token, const std::vector<int>& ids) {
    return std::find(ids.begin(), ids.end(), token) != ids.end();
}

std::vector<int32_t> parse_ids(std::string text) {
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::string word;
    std::vector<int32_t> result;
    while (in >> word) {
        const auto n = integer(word, "token ID");
        if (n < 0 || n > INT32_MAX) throw std::invalid_argument("invalid token ID");
        result.push_back((int32_t) n);
    }
    return result;
}

void check_ids(const std::vector<int32_t>& ids, const Engine& e) {
    if (ids.empty()) throw std::invalid_argument("empty prompt");
    for (int32_t id : ids)
        if (id < 0 || id >= e.n_vocab()) throw std::invalid_argument("token ID outside vocabulary");
}

// The final token is evaluated with logits; preceding tokens can use the
// layer-major read path, staging each missing Expert once for several chunks.
bool read_prompt_prefix(Engine& e, const int32_t* ids, int n, const std::atomic<bool>& stop, Clock::time_point start,
                        bool report) {
    int done = 0;
    while (done < n - 1) {
        if (stop.load()) return false;
        const int count = std::min(n - 1 - done, e.max_read());
        const std::function<bool(int, int)> progress = [&](int layers, int total) {
            if (report) {
                const int at = done + (int) ((int64_t) count * layers / std::max(total, 1));
                const double ms = elapsed(start);
                std::printf("PP %d %d %.1f %.1f\n", at, n, ms, at * 1000.0 / std::max(ms, 0.001));
                std::fflush(stdout);
            }
            return !stop.load();
        };
        if (!e.read(ids + done, count, nullptr, 0, &progress)) return false;
        done += count;
    }
    return !stop.load();
}

// Same phase ownership as the Gemma front end: the existing Engine stage loan
// ends before final logits, including cancellation/exception and MTP-free runs.
bool prompt(Engine& e, const int32_t* ids, int n, const std::atomic<bool>& stop, Clock::time_point start,
            bool report, bool trace_stage = false) {
    const size_t before_stage = trace_stage ? e.staging_bytes() : 0;
    const double before_cache = trace_stage ? e.cached_gib() * 1024 : 0;
    bool ended = false;
    const auto end_read = [&] {
        const size_t during_stage = trace_stage ? e.staging_bytes() : 0;
        const double during_cache = trace_stage ? e.cached_gib() * 1024 : 0;
        const auto t0 = Clock::now();
        ended = true;
        e.end_read();
        if (trace_stage) {
            static bool announced = false;
            if (during_stage > before_stage && !announced) {
                announced = true;
                std::fprintf(stderr, "tinystrata-lfm2: full-layer stage active (CUDA long read)\n");
            }
            std::fprintf(stderr, "lfm2-stage-read: prompt=%d stage_bytes_before=%zu stage_bytes_during=%zu "
                                 "stage_bytes_after=%zu cache_mib_before=%.3f cache_mib_during=%.3f "
                                 "cache_mib_after=%.3f end_read_ms=%.3f\n", n, before_stage, during_stage,
                         e.staging_bytes(), before_cache, during_cache, e.cached_gib() * 1024, elapsed(t0));
        }
    };
    try {
        const bool complete = read_prompt_prefix(e, ids, n, stop, start, report);
        end_read();
        if (!complete || stop.load()) return false;
        e.eval(ids + n - 1, 1, true);
        if (report) {
            const double ms = elapsed(start);
            std::printf("PP %d %d %.1f %.1f\n", n, n, ms, n * 1000.0 / std::max(ms, 0.001));
            std::fflush(stdout);
        }
        return true;
    } catch (...) {
        if (!ended) end_read();
        throw;
    }
}

void sampling_key(Sampling& s, const std::string& word) {
    const size_t eq = word.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("expected sampling key=value");
    const std::string key = word.substr(0, eq), value = word.substr(eq + 1);
    if (key == "top_k") { s.top_k = bounded_integer(value, "top_k", 0, INT32_MAX); return; }
    if (key == "penalty_last_n") { s.last_n = bounded_integer(value, "penalty_last_n", -1, INT32_MAX); return; }
    if (key == "seed") { s.seed = seed_integer(value); return; }
    size_t used = 0;
    const double v = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(v)) throw std::invalid_argument("invalid sampling value");
    if (key == "temperature") s.temperature = v;
    else if (key == "top_p") s.top_p = v;
    else if (key == "min_p") s.min_p = v;
    else if (key == "penalty_repeat") s.repeat = v;
    else if (key == "penalty_freq") s.frequency = v;
    else if (key == "penalty_present") s.presence = v;
    else throw std::invalid_argument("unsupported sampling key: " + key);
    if (s.temperature < 0 || s.top_p <= 0 || s.top_p > 1 || s.min_p < 0 || s.min_p > 1 || s.repeat <= 0 || s.top_k < 0)
        throw std::invalid_argument("sampling value outside range");
}

void memory_info(const Engine& e) {
    std::printf("INFO expert_slots=%d expert_cache_mib=%lld kv_grow=%d kv_committed_rows=%lld "
                "kv_logical_bytes=%llu kv_mapped_bytes=%llu\n", e.cached_experts(),
                (long long) (e.cached_gib() * 1024), e.kv_growing(), (long long) e.kv_committed_rows(),
                (unsigned long long) e.kv_logical_bytes(), (unsigned long long) e.kv_mapped_bytes());
}

int serve(Engine& e, const Options& o) {
    const char* kv = e.kv_format() == strata::lfm2::KvFmt::kFp8 ? "fp8-mma"
        : e.kv_format() == strata::lfm2::KvFmt::kQ8_0 ? "int8" : "fp16";
    std::printf("INFO version=0.1.0 family=lfm2moe kv=%s mtp=0 vision=0 expert_slots=%d "
                "expert_cache_mib=%lld expert_auto=%d chunk=%d vram_reserve_mib=%d graph=%d prompt_logits_first=1 "
                "kv_grow=%d kv_committed_rows=%lld kv_logical_bytes=%llu kv_mapped_bytes=%llu\n",
                kv, e.cached_experts(), (long long) (e.cached_gib() * 1024), o.expert_vram_gib < 0 && o.gpu_experts,
                e.max_rows(), e.reserve_mib(), o.graphs, e.kv_growing(), (long long) e.kv_committed_rows(),
                (unsigned long long) e.kv_logical_bytes(), (unsigned long long) e.kv_mapped_bytes());
    std::printf("READY %d stop\n", e.max_context());
    std::fflush(stdout);
    struct Inbox {
        struct Command { std::string text; uint64_t cancel_epoch; };
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<Command> lines;
        std::atomic<bool> stop{false};
        bool eof = false;
        uint64_t cancel_epoch = 0;
    };
    const auto eos = stop_tokens(e);
    auto box = std::make_shared<Inbox>();
    std::thread reader([box] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::lock_guard<std::mutex> guard(box->mutex);
            if (line == "STOP") {
                ++box->cancel_epoch;
                box->stop.store(true);
                continue;
            }
            box->lines.push_back({line, box->cancel_epoch});
            box->changed.notify_one();
            if (line == "QUIT") break;
        }
        std::lock_guard<std::mutex> guard(box->mutex);
        box->eof = true;
        box->changed.notify_one();
    });
    std::vector<int32_t> live;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> guard(box->mutex);
            box->changed.wait(guard, [&] { return box->eof || !box->lines.empty(); });
            if (box->lines.empty()) {
                guard.unlock();
                reader.join();
                return 0;
            }
            auto command = std::move(box->lines.front());
            line = std::move(command.text);
            box->lines.pop_front();
            // STOP may already have arrived after this GEN was queued. Clearing
            // it unconditionally here would silently lose a quick cancellation.
            box->stop.store(command.cancel_epoch != box->cancel_epoch);
        }
        if (line == "QUIT") { reader.join(); return 0; }
        if (line.empty()) continue;
        try {
            if (line == "RESET") {
                e.reset(); live.clear();
                std::puts("OK RESET"); std::fflush(stdout);
                continue;
            }
            if (line.rfind("GEN ", 0) != 0) throw std::invalid_argument("expected GEN <max_new> [key=value ...] <ids>; text only");
            std::istringstream input(line.substr(4));
            std::string word;
            input >> word;
            const auto max_new = integer(word, "max_new");
            if (max_new < 1 || max_new > e.max_context()) throw std::invalid_argument("max_new outside context capacity");
            Sampling smp;
            std::string tokens;
            while (input >> word) {
                if (word.find('=') != std::string::npos && tokens.empty()) sampling_key(smp, word);
                else tokens += (tokens.empty() ? "" : " ") + word;
            }
            const auto ids = parse_ids(tokens);
            check_ids(ids, e);
            if (ids.size() > (size_t) e.max_context() || max_new > e.max_context() - (int64_t) ids.size())
                throw std::invalid_argument("prompt + max_new exceeds context");
            size_t reuse = 0;
            if (live.size() < ids.size() && e.pos() == (int64_t) live.size() &&
                std::equal(live.begin(), live.end(), ids.begin())) reuse = live.size();
            else { e.reset(); live.clear(); }
            // Between requests, unused KV pages can go back to the Expert cache.
            // trim_kv also keeps the preceding request's reach to avoid churn.
            e.trim_kv((int64_t) ids.size());
            e.mind_vram();
            const auto before = e.stats();
            const auto p0 = Clock::now();
            const int fresh = (int) (ids.size() - reuse);
            if (!prompt(e, ids.data() + reuse, fresh, box->stop, p0, true, o.stage_read_mib > 0)) {
                e.reset(); live.clear();
                memory_info(e);
                std::printf("DONE 0 %zu %.1f 0 cancel 0 0 %zu 0 0 0 0 0.0 0 0\n", ids.size(), elapsed(p0), reuse);
                std::fflush(stdout);
                continue;
            }
            live = ids;
            const double pp = elapsed(p0);
            std::vector<int32_t> history = ids;
            std::mt19937_64 rng(smp.seed);
            const auto d0 = Clock::now();
            int produced = 0;
            int evaluated = 0;
            const char* finish = "length";
            for (; produced < max_new;) {
                if (box->stop.load()) { finish = "cancel"; break; }
                const int token = sample(e.logits(), e.n_vocab(), smp, history, rng);
                std::printf("T %d\n", token);
                std::fflush(stdout);
                ++produced;
                history.push_back(token);
                if (is_stop(token, eos)) { finish = "stop"; break; }
                if (produced == max_new) break;
                e.eval(&history.back(), 1, true);
                ++evaluated;
                live.push_back(token);
            }
            if ((int64_t) live.size() != e.pos()) throw std::logic_error("history/state mismatch");
            memory_info(e);
            std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %d 0\n", produced, ids.size(), pp,
                        elapsed(d0), finish, reuse, fresh);
            std::fflush(stdout);
            const auto& st = e.stats();
            std::fprintf(stderr, "lfm2: generated=%d decode_evaluated=%d prompt_logits_first=1 "
                         "experts gpu=%lld cpu=%lld staged=%lld cached_batch=%lld swaps=%lld "
                         "kv_grow=%d kv_committed_rows=%lld kv_logical_bytes=%llu kv_mapped_bytes=%llu expert_cache_mib=%lld\n", produced, evaluated,
                         (long long) (st.gpu_experts - before.gpu_experts),
                         (long long) (st.cpu_experts - before.cpu_experts),
                         (long long) (st.batch_staged - before.batch_staged),
                         (long long) (st.batch_cached - before.batch_cached),
                         (long long) (st.swapped_experts - before.swapped_experts),
                         e.kv_growing(), (long long) e.kv_committed_rows(),
                         (unsigned long long) e.kv_logical_bytes(), (unsigned long long) e.kv_mapped_bytes(),
                         (long long) (e.cached_gib() * 1024));
        } catch (const std::exception& ex) {
            // A failed/cancelled read may leave only some recurrent layers advanced.
            // Reset before accepting another request, rather than reuse that state.
            e.reset(); live.clear();
            std::printf("ERR %s\n", ex.what());
            std::fflush(stdout);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options opt;
        std::string ids_text, force_text, dump, ctx = "auto", chunk = "auto", kv = "auto";
        int n_gen = 64;
        bool server = false;
        opt.draft = false;
        opt.window = 1;
        opt.vram_reserve_mib = 768;
        opt.prompt_batch = 256;
        opt.stage_mib = 256;
#if !defined(STRATA_USE_HIP) && !defined(__HIPCC__)
        opt.stage_read_mib = 4096;
        if (const char* setting = std::getenv("TINYSTRATA_LFM_STAGE_READ"))
            if (std::strcmp(setting, "0") == 0) opt.stage_read_mib = 0;
#endif
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
        // The HIP growable path is disabled even for direct native launches (#1).
        opt.kv_grow = false;
#endif
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (++i == argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--capabilities") {
#if defined(STRATA_USE_HIP)
                std::puts("{\"backend\":\"hip\",\"model_family\":\"lfm2\",\"kv_types\":[\"int8\"],\"images\":false,\"mtp\":false}");
#else
                std::puts("{\"backend\":\"cuda\",\"model_family\":\"lfm2\",\"kv_types\":[\"int8\",\"fp8-mma\",\"fp16\"],\"images\":false,\"mtp\":false}");
#endif
                return 0;
            }
            if (arg == "--help" || arg == "-h") {
                std::puts("TinyStrata LFM2.5-24B-A2B (text only, no MTP)\n"
                          "  --model GGUF --serve\n"
                          "  --model GGUF --tokens IDS --n N [--dump-full-logits FILE] [--force IDS]\n"
                          "  --max-context auto|N   Context capacity; auto by total VRAM: 8 GB 64K, 12+ GB 128K (model limit)\n"
                          "  --chunk auto|N         Prompt chunk size (auto: 256)\n"
                          "  --expert-vram auto|GiB Expert GPU cache budget; overflow runs on CPU\n"
                          "  --vram-reserve-mib N   Leave VRAM headroom (default 768 MiB)\n"
                          "  --cpu-threads N --no-graph --no-gpu-experts\n"
                          "  --kv-type auto|fp8|int8|fp16\n"
                          "                        Auto: native FP8 Attention, otherwise INT8 KV\n"
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
                          "  --kv-grow             Not supported on the AMD HIP backend (issue #1)\n"
                          "  --kv-whole            Allocate the full context's KV at startup (required on HIP)\n"
#else
                          "  --kv-grow             Grow physical KV with actual usage (default, if supported)\n"
                          "  --kv-whole            Allocate the full context's KV at startup\n"
#endif
                          "  --stage-mib N          Prompt Expert streaming buffer (default 256 MiB)");
                return 0;
            } else if (arg == "--model" || arg == "--native") opt.model = next();
            else if (arg == "--serve") server = true;
            else if (arg == "--tokens") ids_text = next();
            else if (arg == "--force") force_text = next();
            else if (arg == "--dump-full-logits") dump = next();
            else if (arg == "--n" || arg == "--max-new") n_gen = bounded_integer(next(), "n", 1, 131072);
            else if (arg == "--max-context") ctx = next();
            else if (arg == "--chunk") chunk = next();
            else if (arg == "--expert-vram") {
                const auto s = next();
                if (s == "auto") opt.expert_vram_gib = -1;
                else {
                    size_t used = 0;
                    opt.expert_vram_gib = std::stod(s, &used);
                    if (used != s.size() || !std::isfinite(opt.expert_vram_gib) || opt.expert_vram_gib < 0)
                        throw std::invalid_argument("invalid --expert-vram");
                }
            } else if (arg == "--vram-reserve-mib") opt.vram_reserve_mib = bounded_integer(next(), "reserve", 0, 1048576);
            else if (arg == "--stage-mib") opt.stage_mib = bounded_integer(next(), "stage", 1, 65536);
            else if (arg == "--cpu-threads" || arg == "--threads") opt.threads = bounded_integer(next(), "threads", 0, 1024);
            else if (arg == "--no-graph") opt.graphs = false;
            else if (arg == "--no-gpu-experts") opt.gpu_experts = false;
            else if (arg == "--kv-grow") opt.kv_grow = true;
            else if (arg == "--kv-whole" || arg == "--no-kv-grow") opt.kv_grow = false;
            else if (arg == "--kv-type") {
                kv = next();
                if (kv != "auto" && kv != "fp16" && kv != "int8" && kv != "q8_0" && kv != "fp8" && kv != "fp8-mma")
                    throw std::invalid_argument("LFM KV supports auto, fp16, int8 or fp8");
            } else if (arg == "--mtp") {
                bounded_integer(next(), "MTP (this checkpoint has no MTP head)", 0, 0);
            } else throw std::invalid_argument("unknown option: " + arg);
        }
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
        if (opt.kv_grow)
            throw std::invalid_argument("--kv-grow is not supported on the AMD HIP backend; use --kv-whole (issue #1)");
#endif
        if (opt.model.empty()) throw std::invalid_argument("--model is required");
        if (!server && ids_text.empty()) throw std::invalid_argument("--serve or --tokens is required");
        if (server && (!ids_text.empty() || !force_text.empty() || !dump.empty()))
            throw std::invalid_argument("--tokens, --force and --dump-full-logits cannot be combined with --serve");
        if (opt.threads < 0 || opt.threads > 1024 || opt.vram_reserve_mib < 0 || opt.vram_reserve_mib > 1048576 ||
            opt.stage_mib < 1 || opt.stage_mib > 65536 || n_gen < 1)
            throw std::invalid_argument("option outside supported range");
        if (ctx == "auto") {
            size_t free = 0, total = 0;
            if (strata::lfm2::resident_mem_info(&free, &total) != cudaSuccess) throw std::runtime_error("cannot query CUDA VRAM");
            // Context follows the card's capacity class, capped by this model's
            // 128K limit. Expert placement still uses the actual free VRAM.
            opt.max_context = strata::context_policy::for_total_vram(total, 131072);
            std::fprintf(stderr, "tinystrata-lfm2: automatic context: %.2f GiB total GPU VRAM -> %d tokens (%.2f GiB free)\n",
                         total / 1073741824.0, opt.max_context, free / 1073741824.0);
        } else opt.max_context = bounded_integer(ctx, "context", 16, 131072);
        if (opt.max_context < 16 || opt.max_context > 131072) throw std::invalid_argument("context must be 16..131072");
        if (chunk != "auto") opt.prompt_batch = bounded_integer(chunk, "chunk", 1, 4096);
        if (opt.prompt_batch < 1 || opt.prompt_batch > 4096) throw std::invalid_argument("chunk must be 1..4096");
        if (kv == "auto")
            opt.kv_format = strata::lfm2::KvFmt::kAuto;
        else if (kv == "fp8" || kv == "fp8-mma")
            opt.kv_format = strata::lfm2::KvFmt::kFp8;
        else
            opt.kv_format = kv == "fp16" ? strata::lfm2::KvFmt::kFp16 : strata::lfm2::KvFmt::kQ8_0;
        if (opt.stage_read_mib > 0)
            std::fprintf(stderr, "tinystrata-lfm2: full-layer stage enabled for eligible long reads "
                                 "(cap %d MiB); borrowed VRAM returns before decode\n",
                         opt.stage_read_mib);
        auto e = Engine::open(opt);
        if (server) return serve(*e, opt);
        const auto ids = parse_ids(ids_text), forced = parse_ids(force_text);
        check_ids(ids, *e);
        for (int id : forced) if (id >= e->n_vocab()) throw std::invalid_argument("forced token outside vocabulary");
        if (ids.size() > (size_t) e->max_context() || n_gen > e->max_context() - (int64_t) ids.size())
            throw std::invalid_argument("prompt + n exceeds context");
        std::atomic<bool> stop{false};
        const auto p0 = Clock::now();
        prompt(*e, ids.data(), (int) ids.size(), stop, p0, false, opt.stage_read_mib > 0);
        const double pp = elapsed(p0);
        std::ofstream out;
        if (!dump.empty()) {
            out.open(dump, std::ios::binary);
            if (!out) throw std::runtime_error("cannot open logits output");
        }
        const auto eos = stop_tokens(*e);
        const auto d0 = Clock::now();
        int produced = 0;
        int evaluated = 0;
        for (int i = 0; i < n_gen; ++i) {
            if (out.is_open()) {
                out.write((const char*) e->logits(), (std::streamsize) e->n_vocab() * sizeof(float));
                if (!out) throw std::runtime_error("cannot write logits output");
            }
            const int token = strata::lfm2::most_likely(e->logits(), e->n_vocab());
            std::printf("%s%d", i ? "," : "", token);
            ++produced;
            if ((is_stop(token, eos) && forced.empty()) || i + 1 == n_gen) break;
            const int32_t feed = (size_t) i < forced.size() ? forced[(size_t) i] : token;
            e->eval(&feed, 1, true);
            ++evaluated;
        }
        std::puts("");
        const double tg = elapsed(d0);
        const auto& s = e->stats();
        std::fprintf(stderr, "lfm2: prompt=%zu pp_ms=%.2f generated=%d decode_evaluated=%d prompt_logits_first=1 "
                     "tg_ms=%.2f tg=%.2f tok/s "
                     "cache=%d/%d gpu_experts=%lld cpu_experts=%lld staged=%lld\n", ids.size(), pp, produced, evaluated, tg,
                     produced * 1000.0 / std::max(tg, 0.001), e->cached_experts(), e->total_experts(),
                     (long long) s.gpu_experts, (long long) s.cpu_experts, (long long) s.batch_staged);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "tinystrata-lfm2: %s\n", ex.what());
        return 2;
    }
}
