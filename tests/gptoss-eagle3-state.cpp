// Model-backed EAGLE-3/target state regression. Run graph/cache/KV arms
// separately. The executable itself launches no services.
#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using strata::gemma4::Engine;
using strata::gemma4::Options;
using strata::gemma4::KvFmt;
static unsigned checks = 0, proposed = 0;
static void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
    ++checks;
}
static std::vector<int32_t> sequence(int count, int offset = 0) {
    std::vector<int32_t> result(size_t(count), 0);
    for (int i = 0; i < count; ++i) result[size_t(i)] = 32 + (i * 37 + i / 7 + offset) % 512;
    return result;
}
static void exact(const std::vector<float>& expected, const float* actual,
                  const char* stage, int pos, int kept) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
            std::memcmp(expected.data() + i, actual + i, sizeof(float))) {
            std::fprintf(stderr, "FAIL %s pos=%d kept=%d logit=%zu expected=%.9g actual=%.9g\n",
                         stage, pos, kept, i, expected[i], actual[i]);
            throw std::runtime_error("target full-vocabulary logits changed");
        }
    }
    ++checks;
}
static void prefix(Engine& engine, const std::vector<int32_t>& ids) {
    engine.reset();
    need(engine.pos() == 0 && engine.external_draft_pos() == 0, "reset did not reset both positions");
    for (size_t at = 0; at < ids.size();) {
        const int count = int(std::min<size_t>(64, ids.size() - at));
        engine.eval(ids.data() + at, count, at + size_t(count) == ids.size());
        at += size_t(count);
        need(engine.pos() == int64_t(at) && engine.external_draft_pos() == int64_t(at),
             "prompt did not append exactly its target features");
    }
}
static std::vector<int32_t> immutable_propose(Engine& engine, int32_t anchor,
                                             const float* target_logits, const char* stage,
                                             int pos, int kept, int row = -1) {
    const std::vector<float> before(target_logits, target_logits + engine.n_vocab());
    const int64_t target_pos = engine.pos(), draft_pos = engine.external_draft_pos();
    std::vector<int32_t> samples;
    if (engine.device_sampling_enabled()) {
        for (int i = 0; i < 32; ++i) {
            int32_t token = -1;
            need(engine.sample_device(row, 1., (double(i) + .314159) / 32., &token),
                 "device sample unavailable before propose");
            samples.push_back(token);
        }
    }
    std::vector<int32_t> ids(2, -1);
    const int count = engine.external_draft_propose(anchor, 2, ids.data());
    need(count == 2, "EAGLE-3 did not supply the complete requested proposal prefix");
    need(engine.pos() == target_pos && engine.external_draft_pos() == draft_pos,
         "propose advanced a committed target/draft position");
    for (int32_t id : ids) need(id >= 0 && id < engine.n_vocab(), "EAGLE-3 proposed an invalid token");
    // The pointer refers to a target-owned host copy, and accessors must also
    // keep the same current GPU/host logit identity after the drafter's work.
    exact(before, target_logits, stage, pos, kept);
    for (size_t i = 0; i < samples.size(); ++i) {
        int32_t token = -1;
        need(engine.sample_device(row, 1., (double(i) + .314159) / 32., &token) && token == samples[i],
             "propose changed target GPU categorical draws");
    }
    proposed += unsigned(count);
    return ids;
}
static void one_position(Engine& engine, int position) {
    const auto input = sequence(position);
    const auto continuation = sequence(16, 113);
    prefix(engine, input);
    immutable_propose(engine, continuation[0], engine.logits(), "prompt propose immutability", position, 0);

    // Only the target's public state is saved. A later external restoration
    // must invalidate, rather than silently reuse unrelated EAGLE-3 KV.
    std::vector<uint8_t> state(engine.state_bytes());
    std::vector<uint8_t> rows(size_t(position) * engine.row_bytes());
    engine.save_state(state.data());
    engine.save_rows(0, position, rows.data());

    std::vector<std::vector<float>> expected;
    std::map<int, std::vector<int32_t>> expected_draft;
    for (int i = 0; i < 16; ++i) {
        engine.eval(continuation.data() + i, 1, true);
        expected.emplace_back(engine.logits(), engine.logits() + engine.n_vocab());
        need(engine.external_draft_pos() == engine.pos(), "serial eval did not append its target feature");
        if (i + 1 == 1 || i + 1 == 2 || i + 1 == 3)
            expected_draft.emplace(i + 1, immutable_propose(engine, continuation[size_t(i + 1)],
                engine.logits(), "serial propose immutability", position, i + 1));
    }

    for (int kept : {1, 2, 3}) {
        prefix(engine, input);
        int32_t window[3];
        for (int i = 0; i < 3; ++i) window[i] = i < kept ? continuation[size_t(i)] : 800 + i;
        const int64_t draft_before = engine.external_draft_pos();
        engine.window(window, 3);
        need(engine.pos() == position && engine.external_draft_pos() == draft_before,
             "tentative window committed target or draft state");
        for (int i = 0; i < kept; ++i)
            exact(expected[size_t(i)], engine.logits_of(i), "window accepted causal prefix", position, kept);
        engine.keep(kept);
        need(engine.pos() == position + kept && engine.external_draft_pos() == engine.pos(),
             "keep committed more than the accepted prefix");
        const auto next = immutable_propose(engine, continuation[size_t(kept)], engine.logits_of(kept - 1),
                                            "kept prefix propose immutability", position, kept, kept - 1);
        need(next == expected_draft.at(kept), "rejected suffix contaminated the next EAGLE-3 proposal");
        for (int i = kept; i < 16; ++i) {
            engine.eval(continuation.data() + i, 1, true);
            exact(expected[size_t(i)], engine.logits(), "continuation after rejected draft suffix", position, kept);
            need(engine.external_draft_pos() == engine.pos(), "continuation draft position diverged");
        }
    }

    prefix(engine, sequence(position + 17, 271));
    engine.load_rows(rows.data(), 0, position, 0, position);
    engine.load_state(state.data(), position);
    int32_t guesses[2] = {};
    need(engine.pos() == position && engine.external_draft_propose(continuation[0], 2, guesses) == 0,
         "external restoration used stale draft context");
    for (int i = 0; i < 4; ++i) {
        engine.eval(continuation.data() + i, 1, true);
        exact(expected[size_t(i)], engine.logits(), "restored target with invalid draft context", position, i + 1);
        need(engine.external_draft_propose(continuation[size_t(i + 1)], 2, guesses) == 0,
             "invalid draft context became valid without a full rebuild");
    }
    prefix(engine, input);
    immutable_propose(engine, continuation[0], engine.logits(), "reset/full-prefix rebuild", position, 0);

    std::printf("PASS EAGLE-3 state position=%d checks=%u proposals=%u\n", position, checks, proposed);
    std::fflush(stdout);
}
int main(int argc, char** argv) {
    try {
        need(argc >= 3, "usage: gptoss_eagle3_state_test MODEL EAGLE3 [--graphs 0|1 --expert-gib N --kv int8|fp16 --context N --position N]");
        need(std::getenv("STRATA_FREEZE_EXPERTS") != nullptr, "set STRATA_FREEZE_EXPERTS=1 for target byte parity");
        Options options;
        options.model = argv[1]; options.eagle_model = argv[2]; options.eagle_tokens = 2;
        options.max_context = 4096; options.prompt_batch = 64; options.window = 3;
        options.expert_swaps = 0; options.kv_grow = false; options.kv_format = KvFmt::kQ8_0;
        options.draft = false;
        int position = 193;
        for (int i = 3; i < argc; ++i) {
            const std::string key = argv[i];
            need(i + 1 < argc, "missing option value");
            const std::string value = argv[++i];
            if (key == "--graphs") { need(value == "0" || value == "1", "graphs must be0or1"); options.graphs = value == "1"; }
            else if (key == "--expert-gib") options.expert_vram_gib = std::stod(value);
            else if (key == "--context") options.max_context = std::stoi(value);
            else if (key == "--position") position = std::stoi(value);
            else if (key == "--kv") {
                need(value == "int8" || value == "fp16", "kv must beint8orfp16");
                options.kv_format = value == "fp16" ? KvFmt::kFp16 : KvFmt::kQ8_0;
            } else throw std::runtime_error("unknown option: " + key);
        }
        need(position > 0 && position <= options.max_context - 32, "position leaves insufficient context for tests");
        auto engine = Engine::open(options);
        need(engine->external_draft_enabled() && !engine->drafts() && engine->max_window() == 3,
             "unexpected model/drafter/window geometry");
        need(std::string(engine->model_family()) == "gpt-oss", "this fixture requires GPT-OSS");
        one_position(*engine, position);
        std::printf("PASS EAGLE-3 target/draft state checks=%u proposals=%u graph=%d cache=%d/%d context=%d position=%d\n",
                     checks, proposed, int(options.graphs), engine->cached_experts(), engine->total_experts(),
                     engine->max_context(), position);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "gptoss_eagle3_state_test: %s\n", error.what());
        return 1;
    }
}
