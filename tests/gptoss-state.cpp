// Model-backed GPT-OSS state/window parity. Not a benchmark; run each backend/KV/graph/cache arm separately.
// Direct-link with tinystrata_gemma4. The engine owns all GPU work; this test needs no GPU headers.
#include "engine.hpp" // select the matching frozen source's src/gemma4 include directory when direct-linking
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using strata::gemma4::Engine;
using strata::gemma4::Options;
using strata::gemma4::KvFmt;
static unsigned checks = 0;
static void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static std::vector<int32_t> sequence(int n, int offset = 0) {
    std::vector<int32_t> result(n);
    for (int i = 0; i < n; ++i) result[i] = 32 + (i * 37 + i / 7 + offset) % 512;
    return result;
}
static void prefix(Engine& e, const std::vector<int32_t>& ids) {
    e.reset();
    for (size_t at = 0; at < ids.size();) {
        const int n = int(std::min<size_t>(64, ids.size() - at));
        e.eval(ids.data() + at, n, false);
        at += n;
    }
}
static void equal(const std::vector<float>& expected, const float* got, const char* stage, int pos, int kept) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(got[i]) ||
            std::memcmp(expected.data() + i, got + i, sizeof(float))) {
            std::fprintf(stderr, "FAIL %s pos=%d kept=%d logit=%zu expected=%.9g got=%.9g\n",
                         stage, pos, kept, i, expected[i], got[i]);
            throw std::runtime_error("state/window logits differ");
        }
    }
    ++checks;
}
static void one_position(Engine& e, int pos) {
    const auto input = sequence(pos);
    const auto continuation = sequence(16, 113);
    prefix(e, input);
    std::vector<uint8_t> state(e.state_bytes()), rows(size_t(pos) * e.row_bytes());
    e.save_state(state.data());
    e.save_rows(0, pos, rows.data());
    std::vector<std::vector<float>> expected;
    for (const int32_t token : continuation) {
        e.eval(&token, 1, true);
        expected.emplace_back(e.logits(), e.logits() + e.n_vocab());
    }
    const auto restore = [&] {
        e.load_rows(rows.data(), 0, pos, 0, pos);
        e.load_state(state.data(), pos);
        need(e.pos() == pos, "restored position differs");
    };

    // Change every full-attention row and wrap the sliding rings before restoring the saved sequence.
    prefix(e, sequence(pos + 17, 271));
    restore();
    for (int i = 0; i < 4; ++i) {
        e.eval(continuation.data() + i, 1, true);
        equal(expected[size_t(i)], e.logits(), "save/load after unrelated sequence", pos, i + 1);
    }

    for (const int kept : {1, 4, 8}) {
        restore();
        int32_t proposed[8];
        for (int i = 0; i < 8; ++i)
            proposed[i] = i < kept ? continuation[size_t(i)] : 800 + i;
        e.window(proposed, 8);
        // Only the accepted prefix is required to agree; rejected future rows are deliberately different.
        for (int i = 0; i < kept; ++i)
            equal(expected[size_t(i)], e.logits_of(i), "window accepted prefix", pos, kept);
        e.keep(kept);
        need(e.pos() == pos + kept, "keep position differs");
        for (int i = kept; i < 16; ++i) {
            e.eval(continuation.data() + i, 1, true);
            equal(expected[size_t(i)], e.logits(), "continuation after rejected suffix", pos, kept);
        }
    }
    std::printf("PASS state/window pos=%d checks=%u\n", pos, checks);
    std::fflush(stdout);
}
int main(int argc, char** argv) {
    try {
        need(argc >= 2, "usage: gptoss_state_test MODEL [--graphs 0|1] [--expert-gib N] [--kv int8|fp16] [--position N]");
        Options o;
        o.model = argv[1];
        o.allow_gpt_oss_hip = true; // this is the explicit numerical-validation executable
        o.max_context = 768;
        o.prompt_batch = 64; // GPT-OSS SWA=128, hence ring=192; all boundaries below exercise real wrapping
        o.window = 8;
        o.expert_swaps = 0;
        o.kv_grow = false;
        o.kv_format = KvFmt::kQ8_0;
        o.draft = false;
        int position = -1;
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            need(i + 1 < argc, "missing option value");
            const std::string value = argv[++i];
            if (key == "--graphs") { need(value == "0" || value == "1", "graphs must be 0 or 1"); o.graphs = value == "1"; }
            else if (key == "--expert-gib") o.expert_vram_gib = std::stod(value);
            else if (key == "--kv") {
                need(value == "int8" || value == "fp16", "kv must be int8 or fp16");
                o.kv_format = value == "fp16" ? KvFmt::kFp16 : KvFmt::kQ8_0;
            } else if (key == "--position") position = std::stoi(value);
            else throw std::runtime_error("unknown option: " + key);
        }
        need(std::getenv("STRATA_FREEZE_EXPERTS") != nullptr,
             "set STRATA_FREEZE_EXPERTS=1 for fixed placement and fixed whole-window arithmetic");
        need(position < 0 || (position > 0 && position <= 700), "position must be 1..700");
        auto e = Engine::open(o);
        need(e->n_vocab() > 1024 && e->max_window() == 8 && !e->drafts(), "unexpected model/window/draft geometry");
        need(e->meta_u("gpt-oss.attention.sliding_window", -1) == 128, "this test requires GPT-OSS sliding window 128");
        if (position >= 0) one_position(*e, position);
        else for (const int p : {127, 128, 191, 192, 193, 384}) one_position(*e, p);
        std::printf("PASS GPT-OSS state/window checks=%u graph=%d cached=%d/%d\n", checks, int(o.graphs),
                    e->cached_experts(), e->total_experts());
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "gptoss_state_test: %s\n", ex.what());
        return 1;
    }
}
