// CPU-only independent checks for the opt-in Clover D0 contract.
#include "strata/intel/clover_causal.hpp"
#include "strata/intel/sampling_policy.hpp"
#include "strata/core/family_layout.hpp"
#include <array>
#include <cstdio>
#include <limits>
#include <map>

namespace C = strata::intel::clover_causal;
namespace P = strata::intel::sampling_policy;
namespace D = strata::qwen35::clover_decode;
int checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& f) {
    bool caught = false;
    try { f(); } catch (const std::runtime_error&) { caught = true; }
    require(caught, "unsupported Clover contract was accepted");
}

void sampler(const std::vector<int>& response) {
    std::vector<float> logits(248320, -10.f);
    logits[1] = 1.1f; logits[2] = 1.4f; logits[3] = -.3f;
    logits[D::kEndOfText] = -2; logits[D::kEndOfTurn] = -2; logits[D::kThinkEnd] = -2;
    auto reference = logits;
    P::Sampling clover;
    clover.clover = true; clover.temperature = .7f; clover.top_k = 64;
    clover.top_p = .95f; clover.min_p = .01f;
    clover.frequency = .1f; clover.presence = .9f; clover.repeat = 1.05f;
    clover.last_n = 0; // Clover deliberately ignores Qwen's history-window switch.
    std::map<int, int> counts;
    for (int token : response) ++counts[token];
    for (auto [token, seen] : counts) {
        float& value = reference[token];
        value -= float(seen) * .1f;
        value -= .9f;
        value = value < 0 ? value * 1.05f : value / 1.05f;
    }
    const auto ramp = [&](int start, int window) {
        const float position = float(response.size() + 1);
        return std::clamp((position - float(start)) / float(window), 0.f, 1.f) * 10.f;
    };
    reference[D::kEndOfText] += ramp(8192, 8192);
    reference[D::kEndOfTurn] += ramp(8192, 8192);
    if (!counts.contains(D::kThinkEnd)) reference[D::kThinkEnd] += ramp(4096, 4096);
    const auto actual = P::sampling_distribution(logits.data(), int(logits.size()), clover, response);
    auto plain = clover; plain.clover = false; plain.repeat = 1; plain.frequency = plain.presence = 0;
    const auto expected = P::sampling_distribution(reference.data(), int(reference.size()), plain, {});
    require(actual.tokens == expected.tokens, "Clover target support differs from independent scalar policy");
    for (size_t i = 0; i < actual.probabilities.size(); ++i)
        require(std::abs(actual.probabilities[i] - expected.probabilities[i]) <=
                    2e-6 * std::max(1., expected.probabilities[i]),
                "Clover frequency/presence/repetition/ramp order differs from FP32 reference");
    std::mt19937_64 a(123), b(123);
    for (int draw = 0; draw < 64; ++draw)
        require(actual.draw(a) == expected.draw(b), "Clover seeded sampler changed its RNG policy");
    clover.temperature = 0;
    const int greedy = int(std::max_element(reference.begin(), reference.end()) - reference.begin());
    require(P::sample(logits.data(), int(logits.size()), clover, response, a) == greedy,
            "Clover greedy selection omitted response controls");
}

int main() { try {
    std::array<float, 5> a = {-3, -1, 0, 1, 3};
    C::convert_a_log(a);
    for (size_t i = 0; i < a.size(); ++i)
        require(std::abs(double(a[i]) + std::exp(double(int(i) == 0 ? -3 : int(i) == 1 ? -1 : int(i) == 2 ? 0 : int(i) == 3 ? 1 : 3)))
                < 2e-6, "Clover raw A_log was not converted to negative exponential");
    require(a[2] == -1, "Clover A_log zero is a decay of -1, not zero");
    rejects([] { C::causal_a(std::numeric_limits<float>::infinity()); });
    rejects([] { C::causal_a(std::numeric_limits<float>::quiet_NaN()); });
    rejects([] { C::causal_a(1000); });
    std::array<float, 2> invalid = {0, std::numeric_limits<float>::infinity()};
    rejects([&] { C::convert_a_log(invalid); });
    require(invalid[0] == 0, "invalid A_log partially modified the weight tensor");
    C::validate_launch(true, 0, 0, false, false);
    rejects([] { C::validate_launch(false, 0, 0, false, false); });
    rejects([] { C::validate_launch(true, 1, 0, false, false); });
    rejects([] { C::validate_launch(true, 0, 2048, false, false); });
    rejects([] { C::validate_launch(true, 0, 0, true, false); });
    rejects([] { C::validate_launch(true, 0, 0, false, true); });
    rejects([] { C::validate_launch(true, 0, 0, false, false, "self-spec"); });
    rejects([] { C::validate_launch(true, 0, 0, false, false, "diffusion"); });
    rejects([] { C::validate_launch(true, 0, 0, false, false, "causal", 3); });
    require(C::terminal(248044) && C::terminal(248046) && !C::terminal(248069), "Clover stopping IDs differ");
    const std::vector<int> prompt = {1, 2, 2, 3};
    require(C::request_history(true, prompt).empty(), "prompt leaked into Clover response penalties");
    require(C::request_history(false, prompt) == prompt, "Qwen/Tiel prompt penalties changed");
    sampler({}); sampler(prompt);
    sampler(std::vector<int>(4096, 5)); sampler(std::vector<int>(9000, 5));
    auto closed_think = std::vector<int>(9000, 5); closed_think[0] = D::kThinkEnd; sampler(closed_think);
    strata::core::ModelGeometry g;
    g.family = strata::ModelFamily::CloverMoe; g.n_layers = 40; g.n_embd = 2048;
    g.n_vocab = 248320; g.n_head = 16; g.n_head_kv = 2; g.head_dim = 256;
    g.n_expert = 256; g.n_expert_used = 8; g.n_ff = g.n_ff_shexp = 512;
    g.ssm_d_conv = 4; g.ssm_state_size = 128; g.ssm_k_heads = 16; g.ssm_v_heads = 32;
    g.ssm_value_dim = 4096; g.ssm_conv_channels = 8192; g.full_attention_interval = 4;
    std::map<int, int> types;
    for (const auto& t : strata::core::global_tensors(g)) ++types[C::tensor_type(t.name)];
    for (int layer = 0; layer < 40; ++layer)
        for (const auto& t : strata::core::layer_tensors(g, layer))
            ++types[C::tensor_type("blk." + std::to_string(layer) + "." + t.name)];
    require(types == std::map<int, int>{{0,421},{8,312},{18,80},{21,40}}, "pinned Clover layout/type counts differ");
    require(C::tensor_type("blk.0.nextn.eh_proj.weight") == -1, "Qwen MTP tensor admitted as Clover D0");
    std::printf("PASS %d Clover D0 CPU checks; no Intel device/model was opened\n", checks);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "Clover D0 fixture failed: %s\n", e.what()); return 1; } }
