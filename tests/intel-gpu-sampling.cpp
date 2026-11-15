#include "strata/intel/gpu_sampling_candidate.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata::intel;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void same(const speculative_distribution::Distribution& expected, const speculative_distribution::Distribution& actual) {
    require(expected.tokens == actual.tokens, "Target token ordering changed");
    require(expected.probabilities.size() == actual.probabilities.size(), "Target probability length changed");
    require(!std::memcmp(expected.probabilities.data(), actual.probabilities.data(),
                        expected.probabilities.size() * sizeof(double)), "Target probability bits changed");
    std::mt19937_64 left(2026), right(2026);
    for (int draw = 0; draw < 128; ++draw) require(expected.draw(left) == actual.draw(right), "Host RNG output changed");
}
int main() {
    try {
        sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        std::mt19937_64 random(20261115);
        int checks = 0;
        for (int vocabulary : {1, 15, 17, 1023, 4097, 248320}) {
            GpuSamplingCandidate candidate(queue, vocabulary);
            std::vector<float> logits(size_t(vocabulary), 0);
            for (float& value : logits) value = float(int64_t(random() % 16000001) - 8000000) / 1000000.f + 0.00000013f;
            std::vector<int> history;
            for (int index = 0; index < 137; ++index) history.push_back(int(random() % uint64_t(vocabulary)));
            const int raw_leader = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            for (int index = 0; index < 4; ++index) history.push_back(raw_leader);
            for (int variant = 0; variant < 7; ++variant) {
                sampling_policy::Sampling policy;
                policy.temperature = variant == 0 ? 0 : variant == 1 ? .7f : 1.f;
                policy.top_k = variant == 2 ? 0 : variant == 3 ? 1 : 20;
                policy.top_p = variant == 5 ? 1.f : .95f; policy.min_p = variant == 4 ? .1f : 0;
                policy.last_n = variant == 1 ? -1 : 64;
                policy.repeat = variant == 4 ? 1.1f : 1.f;
                policy.presence = variant == 6 ? -.7f : 1.5f;
                policy.frequency = variant == 6 ? -.13f : variant == 4 ? .1f : 0;
                same(sampling_policy::sampling_distribution(logits.data(), vocabulary, policy, history),
                     candidate.target(logits.data(), policy, history)); ++checks;
            }
            const int support = std::min(vocabulary, 2048);
            require(candidate.support(logits.data(), support, history) ==
                    make_draft_vocabulary(logits.data(), vocabulary, support, history), "Support IDs changed"); ++checks;
            // All ties and +/-zero force the safe original target ordering.
            for (float constant : {1.f, -1.f, 0.f, -0.f}) {
                std::fill(logits.begin(), logits.end(), constant);
                sampling_policy::Sampling policy; policy.temperature = 1; policy.top_p = .95f;
                same(sampling_policy::sampling_distribution(logits.data(), vocabulary, policy, history),
                     candidate.target(logits.data(), policy, history));
                require(candidate.support(logits.data(), support, history) ==
                        make_draft_vocabulary(logits.data(), vocabulary, support, history), "Tie support IDs changed");
                checks += 2;
            }
            logits[0] = std::numeric_limits<float>::infinity();
            bool rejected = false;
            try { candidate.support(logits.data(), support, history); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Nonfinite support accepted"); ++checks;
            // Exact rejection/residual remains host-side. Matching target
            // distributions must consume the same RNG for accepted/rejected q.
            logits[0] = 3.125f;
            sampling_policy::Sampling stochastic; stochastic.temperature = 1.f;
            auto cpu_target = sampling_policy::sampling_distribution(logits.data(), vocabulary, stochastic, {});
            auto gpu_target = candidate.target(logits.data(), stochastic, {});
            auto proposal = cpu_target;
            if (proposal.tokens.size() > 1) proposal.probabilities[0] *= .01;
            std::mt19937_64 cpu_random(42), gpu_random(42), proposal_random(123);
            for (int draw = 0; draw < 128; ++draw) {
                const int token = proposal.draw(proposal_random);
                require(speculative_distribution::verify(cpu_target, proposal, token, cpu_random) ==
                        speculative_distribution::verify(gpu_target, proposal, token, gpu_random), "Exact rejection RNG changed");
            }
            ++checks;
            candidate.print();
        }
        queue.wait_and_throw();
        std::printf("{\"passed\":true,\"checks\":%d,\"full_vocabulary\":248320,\"max_support\":2048,\"rng_exact\":true,\"probability_bits_exact\":true}\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GPU sampler test failed: %s\n", error.what()); return 1;
    }
}
