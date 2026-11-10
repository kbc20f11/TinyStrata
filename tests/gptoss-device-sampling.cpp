// Model-backed regression for GPT-OSS device sampling and lazy host logits.
// No GPU headers or mutable-engine test hooks: the Engine owns all device work.
#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using strata::gemma4::Engine;
using strata::gemma4::KvFmt;
using strata::gemma4::Options;
static unsigned checks = 0;
static unsigned samples = 0;
static void need(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
    ++checks;
}
static std::vector<int32_t> sequence(int n, int offset = 0) {
    std::vector<int32_t> ids((size_t) n);
    for (int i = 0; i < n; ++i) ids[(size_t) i] = 32 + (i * 37 + i / 7 + offset) % 512;
    return ids;
}
static void equal(const std::vector<float>& expected, const float* got, const char* stage) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(got[i]) ||
            std::memcmp(expected.data() + i, got + i, sizeof(float))) {
            std::fprintf(stderr, "FAIL %s logit=%zu expected=%.9g got=%.9g\n", stage, i, expected[i], got[i]);
            throw std::runtime_error("device-sampling logits differ");
        }
    }
    ++checks;
}
static std::vector<float> latest(Engine& e) {
    const float* p = e.logits();
    return {p, p + e.n_vocab()};
}
static void copies(const Engine& e, uint64_t expected, const char* stage) {
    if (e.logits_host_rows() != expected) {
        std::fprintf(stderr, "FAIL %s host_rows expected=%llu got=%llu\n", stage,
                     (unsigned long long) expected, (unsigned long long) e.logits_host_rows());
        throw std::runtime_error("unexpected vocabulary D2H copy count");
    }
    ++checks;
}
// Independent CPU categorical reference. No ranking, cutoff or omitted vocabulary mass.
static int reference(const std::vector<float>& logits, double temperature, double uniform) {
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size());
    double total = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("reference contains nonfinite logits");
        weights[i] = std::exp((double(logits[i]) - maximum) / temperature);
        total += weights[i];
    }
    need(total > 0 && std::isfinite(total), "invalid reference mass");
    const double target = std::min(uniform * total, std::nextafter(total, 0.0));
    double cumulative = 0;
    int last = -1;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (weights[i] > 0) {
            last = (int) i;
            if (target < cumulative) return (int) i;
        }
    }
    return last;
}
static void draw(Engine& e, int output_row, const std::vector<float>& expected, const char* stage) {
    const uint64_t before = e.logits_host_rows();
    for (int arm = 0; arm < 2; ++arm) {
        const double temperature = arm ? 0.7 : 1.0;
        const double uniform = arm ? 0.891732105984 : 0.193817264905;
        int32_t token = -17;
        need(e.sample_device(output_row, temperature, uniform, &token), "device sampler unexpectedly fell back");
        const int wanted = reference(expected, temperature, uniform);
        if (token != wanted) {
            std::fprintf(stderr, "FAIL %s row=%d T=%.3f U=%.12f expected=%d got=%d\n",
                         stage, output_row, temperature, uniform, wanted, token);
            throw std::runtime_error("device sampler used another output row");
        }
        ++checks;
        ++samples;
    }
    copies(e, before, stage);
}
template<class Exception, class Fn>
static void rejected(Fn&& fn, const char* why) {
    bool caught = false;
    try { fn(); } catch (const Exception&) { caught = true; }
    need(caught, why);
}
struct Snapshot {
    int64_t position;
    std::vector<uint8_t> state, rows;
    explicit Snapshot(Engine& e) : position(e.pos()), state(e.state_bytes()),
        rows((size_t) position * e.row_bytes()) {
        e.save_state(state.data());
        e.save_rows(0, position, rows.data());
    }
    void restore(Engine& e) const {
        e.load_rows(rows.data(), 0, position, 0, position);
        e.load_state(state.data(), position);
        need(e.pos() == position, "restored position differs");
    }
};
static void prefix(Engine& e, const std::vector<int32_t>& ids) {
    e.reset();
    const uint64_t before = e.logits_host_rows();
    for (size_t at = 0; at < ids.size();) {
        const int n = (int) std::min<size_t>(64, ids.size() - at);
        e.eval(ids.data() + at, n, false);
        at += (size_t) n;
    }
    copies(e, before, "prefix without logits");
}
static void regression(Engine& e, int position) {
    int32_t sentinel = -17;
    rejected<std::logic_error>([&] { e.sample_device(-1, 1, 0.2, &sentinel); }, "initial latest row accepted");
    rejected<std::logic_error>([&] { e.sample_device(0, 1, 0.2, &sentinel); }, "initial window row accepted");
    need(sentinel == -17, "rejected sample changed output token");

    prefix(e, sequence(position));
    const int32_t first = 611, no_head = 612;
    uint64_t count = e.logits_host_rows();
    e.eval(&first, 1, true);
    copies(e, count, "eval defers host copy");
    // Sampling must work before a host accessor has ever materialized this output.
    int32_t initial_token = -1;
    need(e.sample_device(-1, 1.0, 0.193817264905, &initial_token), "initial device sample fell back");
    copies(e, count, "initial device sample");
    const auto initial = latest(e);
    copies(e, ++count, "first latest materialization");
    need(initial_token == reference(initial, 1.0, 0.193817264905), "initial device/CPU draw differs");
    equal(initial, e.logits(), "repeat latest accessor");
    copies(e, count, "repeat latest does not recopy");
    draw(e, -1, initial, "materialized latest remains sampleable");

    e.eval(&no_head, 1, false);
    draw(e, -1, initial, "eval false preserves last output");
    equal(initial, e.logits(), "eval false preserves host output");
    copies(e, count, "eval false keeps copy readiness");
    const Snapshot base(e);
    const auto continuation = sequence(8, 113);
    std::vector<std::vector<float>> expected;
    for (int i = 0; i < 8; ++i) {
        e.eval(continuation.data() + i, 1, true);
        copies(e, count, "oracle eval remains deferred");
        expected.push_back(latest(e));
        copies(e, ++count, "oracle output copied once");
    }
    const auto previous_latest = expected.back();
    base.restore(e);
    equal(previous_latest, e.logits(), "restore preserves last-produced output");
    draw(e, -1, previous_latest, "restore preserves last-produced device output");
    copies(e, count, "restore does not reset output copy readiness");

    e.window(continuation.data(), 8);
    copies(e, count, "window defers all eight rows");
    // eval and window own separate buffers. Computing one must neither replace nor dirty the other.
    draw(e, -1, previous_latest, "window preserves latest device buffer");
    equal(previous_latest, e.logits(), "window preserves latest host buffer");
    copies(e, count, "window leaves latest copy ready");
    for (int i = 0; i < 8; ++i) draw(e, i, expected[(size_t) i], "window before keep");
    equal(expected[0], e.logits_of(0), "first window row materialization");
    copies(e, ++count, "window row zero copied once");
    equal(expected[0], e.logits_of(0), "repeat window row accessor");
    copies(e, count, "repeat window row does not recopy");
    equal(expected[3], e.logits_of(3), "nonadjacent window row materialization");
    copies(e, ++count, "only requested nonadjacent row copied");
    e.keep(4);
    need(e.pos() == base.position + 4, "keep position differs");
    for (int i = 0; i < 8; ++i) {
        draw(e, i, expected[(size_t) i], "window after partial keep");
        equal(expected[(size_t) i], e.logits_of(i), "lazy window row after partial keep");
        if (i != 0 && i != 3) ++count;
        copies(e, count, "each window row copied at most once");
    }
    e.eval(continuation.data() + 4, 1, true);
    copies(e, count, "new eval only dirties latest buffer");
    draw(e, -1, expected[4], "new eval latest row");
    equal(expected[4], e.logits(), "new eval latest materialization");
    copies(e, ++count, "new eval copied once");
    for (int i : {0, 7}) {
        draw(e, i, expected[(size_t) i], "eval preserves old window rows");
        equal(expected[(size_t) i], e.logits_of(i), "eval preserves materialized window row");
    }
    const auto unread = sequence(2, 281);
    need(e.read(unread.data(), (int) unread.size()), "short read was cancelled");
    equal(expected[4], e.logits(), "read without head preserves latest output");
    draw(e, -1, expected[4], "read without head preserves latest device buffer");
    copies(e, count, "read without head does not copy logits");

    e.bind(1);
    prefix(e, sequence(17, 271));
    equal(expected[4], e.logits(), "bind/reset preserves global latest output");
    equal(expected[7], e.logits_of(7), "bind/reset preserves global window output");
    copies(e, count, "sequence switch preserves copy readiness");
    const int32_t other = 719;
    e.eval(&other, 1, true);
    copies(e, count, "second sequence eval remains deferred");
    const auto second_latest = latest(e);
    copies(e, ++count, "second sequence output copied once");
    draw(e, -1, second_latest, "second sequence output");
    e.bind(0);
    base.restore(e);
    equal(second_latest, e.logits(), "bind/restore leaves last output global");
    equal(expected[7], e.logits_of(7), "state restore preserves window scratch");
    draw(e, -1, second_latest, "bind/restore device output");
    copies(e, count, "state restore preserves copy readiness");
    e.eval(continuation.data(), 1, true);
    equal(expected[0], e.logits(), "restored ring/state continuation");
    copies(e, ++count, "restored continuation copied once");

    // Build separate one-token oracles, restore each sequence, then feed both through step().
    const int seqs[2] = {0, 1};
    const int32_t tokens[2] = {401, 402};
    std::vector<std::vector<float>> step_expected;
    int64_t before_pos[2];
    for (int s = 0; s < 2; ++s) {
        e.bind(s);
        const Snapshot before(e);
        before_pos[s] = e.pos();
        e.eval(tokens + s, 1, true);
        step_expected.push_back(latest(e));
        copies(e, ++count, "step oracle copied once");
        before.restore(e);
    }
    e.bind(0);
    e.step(seqs, tokens, 2);
    need(e.bound() == 0, "step changed the bound sequence");
    copies(e, count, "step defers both output rows");
    equal(step_expected[1], e.logits(), "step leaves latest eval buffer independent");
    for (int i : {1, 0}) {
        e.bind(i);
        need(e.pos() == before_pos[i] + 1, "step sequence position differs");
        draw(e, i, step_expected[(size_t) i], "step row survives bind");
        equal(step_expected[(size_t) i], e.logits_of(i), "step row materialization");
        copies(e, ++count, "step row copied once across bind");
        equal(step_expected[(size_t) i], e.logits_of(i), "repeat step row");
        copies(e, count, "repeat step row does not recopy");
    }
    e.bind(0);
    draw(e, -1, step_expected[1], "sequence hold preserves latest eval output");
    copies(e, count, "sequence hold preserves latest readiness");

    const auto invalid = [&](int r, double t, double u, int32_t* out) { e.sample_device(r, t, u, out); };
    sentinel = -17;
    rejected<std::invalid_argument>([&] { invalid(-2, 1, 0.2, &sentinel); }, "row below -1 accepted");
    rejected<std::logic_error>([&] { invalid(2, 1, 0.2, &sentinel); }, "row past last step output accepted");
    rejected<std::invalid_argument>([&] { invalid(-1, 1, 0.2, nullptr); }, "null token pointer accepted");
    for (double t : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()})
        rejected<std::invalid_argument>([&] { invalid(-1, t, 0.2, &sentinel); }, "invalid temperature accepted");
    for (double u : {-0.1, 1.0, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()})
        rejected<std::invalid_argument>([&] { invalid(-1, 1, u, &sentinel); }, "invalid uniform accepted");
    rejected<std::logic_error>([&] { (void) e.logits_of(-1); }, "negative host row accepted");
    rejected<std::logic_error>([&] { (void) e.logits_of(2); }, "host row past last step output accepted");
    need(sentinel == -17, "invalid sampler call changed output token");
    copies(e, count, "invalid calls never materialize logits");
    // Nonfinite device logits are injected and rejected by the model-free gptoss_sampling_test;
    // production Engine logits remain read-only here, and every materialized model row is checked finite.
}
int main(int argc, char** argv) {
    try {
        need(argc >= 2, "usage: gptoss_device_sampling_test MODEL [--graphs 0|1] [--expert-gib N] [--kv int8|fp16] [--position N]");
        const char* mode = std::getenv("TINYSTRATA_GPTOSS_GPU_SAMPLE");
        need(!mode || std::strcmp(mode, "1") == 0, "leave TINYSTRATA_GPTOSS_GPU_SAMPLE unset or set it to 1");
        need(std::getenv("STRATA_FREEZE_EXPERTS") != nullptr, "set STRATA_FREEZE_EXPERTS=1");
        Options o;
        o.model = argv[1];
        o.max_context = 768;
        o.prompt_batch = 64;
        o.window = 8;
        o.batch = 1;
        o.expert_swaps = 0;
        o.kv_grow = false;
        o.kv_format = KvFmt::kQ8_0;
        o.draft = false;
        int position = 193;
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            need(i + 1 < argc, "missing option value");
            const std::string value = argv[++i];
            if (key == "--graphs") {
                need(value == "0" || value == "1", "graphs must be 0 or 1");
                o.graphs = value == "1";
            } else if (key == "--expert-gib") o.expert_vram_gib = std::stod(value);
            else if (key == "--kv") {
                need(value == "int8" || value == "fp16", "kv must be int8 or fp16");
                o.kv_format = value == "int8" ? KvFmt::kQ8_0 : KvFmt::kFp16;
            } else if (key == "--position") position = std::stoi(value);
            else throw std::runtime_error("unknown option: " + key);
        }
        need(position > 0 && position <= 700, "position must be 1..700");
        auto e = Engine::open(o);
        need(e->device_sampling_requested() && e->device_sampling_enabled(), "device sampling is unavailable");
        need(e->n_vocab() > 1024 && e->max_window() == 8 && e->sequences() == 2 && !e->drafts(),
             "unexpected model/window/sequence geometry");
        need(e->meta_u("gpt-oss.attention.sliding_window", -1) == 128, "expected GPT-OSS sliding window 128");
        regression(*e, position);
        std::printf("PASS GPT-OSS device sampling checks=%u samples=%u host_rows=%llu position=%d graph=%d cached=%d/%d\n",
                    checks, samples, (unsigned long long) e->logits_host_rows(), position, int(o.graphs),
                    e->cached_experts(), e->total_experts());
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "gptoss_device_sampling_test: %s\n", ex.what());
        return 1;
    }
}
