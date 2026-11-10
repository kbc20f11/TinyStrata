// Model-free exact p/q rejection and independent CPU probability-law regression.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include "strata/gemma4/sampling.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace G = strata::gemma4;
#define CK(x) do { const auto e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)
static unsigned checks = 0;
static void need(bool condition, const char* why) {
    if (!condition) throw std::runtime_error(why);
    ++checks;
}
struct Device {
    uint8_t* base = nullptr;
    size_t size;
    cudaStream_t stream;
    Device(size_t bytes, cudaStream_t s) : size(bytes), stream(s) {
        CK(cudaMalloc(reinterpret_cast<void**>(&base), size + 128));
        CK(cudaMemsetAsync(base, 0xA7, size + 128, s));
    }
    ~Device() { if (base) (void) cudaFree(base); }
    template<class T> T* data() const { return reinterpret_cast<T*>(base + 64); }
    void guards() const {
        uint8_t a[64], b[64];
        CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(a, base, 64, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b, base + 64 + size, 64, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 64; ++i) need(a[i] == 0xA7 && b[i] == 0xA7, "device red zone overwritten");
    }
};
struct Outcome { G::FullVocabResult result; uint32_t accepted; };
struct Verifier {
    int vocab;
    cudaStream_t stream;
    Device logits, q, workspace, result, accepted;
    Verifier(int n, cudaStream_t s) : vocab(n), stream(s), logits(size_t(n) * sizeof(float), s),
        q(size_t(n) * sizeof(double), s), workspace(G::full_vocab_sample_workspace_bytes(n), s),
        result(sizeof(G::FullVocabResult), s), accepted(sizeof(uint32_t), s) {}
    void put(const std::vector<float>& p, const std::vector<double>& draft) {
        need(p.size() == size_t(vocab) && draft.size() == size_t(vocab), "wrong input dimensions");
        CK(cudaMemcpyAsync(logits.data<void>(), p.data(), logits.size, cudaMemcpyHostToDevice, stream));
        CK(cudaMemcpyAsync(q.data<void>(), draft.data(), q.size, cudaMemcpyHostToDevice, stream));
        CK(cudaStreamSynchronize(stream));
    }
    void launch(int token, double temperature, double accept_u, double residual_u,
                G::FullVocabResult* output = nullptr, uint32_t* flag = nullptr) {
        G::verify_full_vocab(logits.data<float>(), q.data<double>(), token, vocab, temperature,
            accept_u, residual_u, workspace.data<void>(), workspace.size,
            output ? output : result.data<G::FullVocabResult>(), flag ? flag : accepted.data<uint32_t>(), stream);
    }
    Outcome get() {
        Outcome value{};
        CK(cudaMemcpyAsync(&value.result, result.data<void>(), sizeof(value.result), cudaMemcpyDeviceToHost, stream));
        CK(cudaMemcpyAsync(&value.accepted, accepted.data<void>(), sizeof(value.accepted), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        return value;
    }
    Outcome verify(int token, double temperature, double accept_u, double residual_u, bool graph = false) {
        cudaGraph_t plan = nullptr; cudaGraphExec_t exec = nullptr;
        if (graph) { CK(cudaStreamSynchronize(stream)); CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); }
        launch(token, temperature, accept_u, residual_u);
        if (graph) {
            CK(cudaStreamEndCapture(stream, &plan)); CK(cudaGraphInstantiate(&exec, plan, nullptr, nullptr, 0));
            CK(cudaGraphLaunch(exec, stream));
        }
        const Outcome result = get();
        if (graph) { CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(plan)); }
        return result;
    }
    std::vector<double> copy_sample(double temperature) {
        G::sample_full_vocab(logits.data<float>(), vocab, temperature, .371, workspace.data<void>(), workspace.size,
                              result.data<G::FullVocabResult>(), stream);
        G::copy_full_vocab_probabilities(workspace.data<void>(), workspace.size,
            result.data<G::FullVocabResult>(), vocab, q.data<double>(), stream);
        std::vector<double> host(size_t(vocab), 0.0);
        CK(cudaMemcpyAsync(host.data(), q.data<void>(), q.size, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        return host;
    }
    void guards() { logits.guards(); q.guards(); workspace.guards(); result.guards(); accepted.guards(); }
};
static std::vector<double> distribution(const std::vector<float>& logits, double temperature) {
    const double max = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) result[i] = std::exp((double(logits[i]) - max) / temperature);
    const double total = std::accumulate(result.begin(), result.end(), 0.0);
    for (double& p : result) p /= total;
    return result;
}
static int categorical(const std::vector<double>& weights, double uniform) {
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double threshold = std::min(uniform * total, std::nextafter(total, 0.0));
    double sum = 0.0;
    int last = -1;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i];
        if (weights[i] > 0) { last = int(i); if (threshold < sum) return int(i); }
    }
    return last;
}
static void independent_oracle(cudaStream_t stream) {
    const std::vector<float> logits{1.5f, -.5f, .25f, -6.f};
    const std::vector<double> q{.1, .5, .2, .2};
    Verifier v(4, stream); v.put(logits, q);
    for (double temp : {.25, 1., 3.5}) {
        const auto p = distribution(logits, temp);
        std::vector<double> residual(p.size());
        for (size_t i = 0; i < p.size(); ++i) residual[i] = std::max(p[i] - q[i], 0.0);
        for (int token = 0; token < 4; ++token)
            for (double a : {0., .2, .8, std::nextafter(1., 0.)})
                for (double r : {0., .25, .8, std::nextafter(1., 0.)}) {
                    const bool accept = a < std::min(1., p[size_t(token)] / q[size_t(token)]);
                    const int expected = accept ? token : categorical(residual, r);
                    for (bool graph : {false, true}) {
                        const auto got = v.verify(token, temp, a, r, graph);
                        need(got.result.status == G::kSampleOk && got.accepted == uint32_t(accept), "CPU/GPU p/q acceptance differs");
                        need(got.result.token == expected, "CPU/GPU residual CDF differs");
                        const double prob = accept ? p[size_t(token)] : residual[size_t(expected)] /
                            std::accumulate(residual.begin(), residual.end(), 0.0);
                        need(std::abs(got.result.probability - prob) < 1e-12, "CPU/GPU selected probability differs");
                    }
                }
    }
    v.guards();
}
static void boundaries(cudaStream_t stream) {
    for (int n : {1, 2, 31, 32, 257, 2047, 2048, 2049, 201088}) {
        Verifier v(n, stream);
        std::vector<float> logits(size_t(n), 0.f);
        for (int i = 0; i < n; ++i) logits[size_t(i)] = float((i * 17) % 31 - 15) * .125f;
        v.put(logits, std::vector<double>(size_t(n), 1. / n));
        const auto copied = v.copy_sample(1.);
        const auto cpu = distribution(logits, 1.);
        for (int i = 0; i < n; ++i) need(std::abs(copied[size_t(i)] - cpu[size_t(i)]) < 1e-13,
                                        "normalized draft probability copy differs");
        need(std::abs(std::accumulate(copied.begin(), copied.end(), 0.) - 1.) < 1e-10, "copied q is not normalized");
        for (int token : {0, n - 1})
            for (double a : {0., .5, std::nextafter(1., 0.)})
                for (bool graph : {false, true}) {
                    const auto got = v.verify(token, 1., a, .5, graph);
                    need(got.result.status == G::kSampleOk && got.accepted == 1 && got.result.token == token,
                         "q=p must accept every sampled token");
                }
        v.guards();
    }
    Verifier v(3, stream);
    v.put({0.f, -10000.f, -10000.f}, {0., .5, .5});
    for (bool graph : {false, true}) {
        for (double a : {0., std::nextafter(1., 0.)}) {
            const auto r = v.verify(1, 1., a, .5, graph);
            need(r.result.status == G::kSampleOk && r.accepted == 0 && r.result.token == 0,
                 "disjoint-support rejection did not return target support");
        }
    }
    // Capture once, change both device inputs, and replay the same graph. No
    // cached acceptance flag or probability from its earlier run may survive.
    cudaGraph_t plan = nullptr; cudaGraphExec_t exec = nullptr;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); v.launch(1, 1., .5, .5);
    CK(cudaStreamEndCapture(stream, &plan)); CK(cudaGraphInstantiate(&exec, plan, nullptr, nullptr, 0));
    CK(cudaGraphLaunch(exec, stream)); need(v.get().accepted == 0, "graph first rejection failed");
    v.put({-10000.f, 0.f, -10000.f}, {0., 1., 0.});
    CK(cudaGraphLaunch(exec, stream)); const auto next = v.get();
    need(next.result.status == G::kSampleOk && next.accepted == 1 && next.result.token == 1,
         "changed-input graph reused stale residual/acceptance");
    CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(plan));
    v.guards();
}
static void invalid_inputs(cudaStream_t stream) {
    Verifier v(3, stream);
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    for (const auto& q : std::vector<std::vector<double>>{{.2,.2,.2}, {0.,0.,0.}, {-.1,.6,.5},
            {1.1,0.,0.}, {nan,.5,.5}, {inf,0.,0.}, {0.,.5,.5}}) {
        v.put({0.f,1.f,2.f}, q);
        const auto r = v.verify(0, 1., .5, .5);
        need(r.result.status == G::kSampleInvalidDraft && r.result.token == -1, "invalid q accepted");
    }
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        v.put({0.f,bad,2.f}, {.2,.3,.5});
        const auto r = v.verify(1, 1., .5, .5);
        need(r.result.status == G::kSampleNonFinite && r.result.token == -1, "nonfinite target accepted");
        const auto q = v.copy_sample(1.);
        need(std::all_of(q.begin(), q.end(), [](double x) { return std::isnan(x); }), "invalid sample copied as valid q");
    }
    v.put({0.f,1.f,2.f}, {.2,.3,.5});
    for (double temp : {0., -1., inf, nan}) {
        bool refused = false; try { v.verify(1, temp, .5, .5); } catch (const std::invalid_argument&) { refused = true; }
        need(refused, "invalid speculative temperature accepted");
    }
    for (double u : {-1., 1., inf, nan}) {
        bool a = false, r = false;
        try { v.verify(1, 1., u, .5); } catch (const std::invalid_argument&) { a = true; }
        try { v.verify(1, 1., .5, u); } catch (const std::invalid_argument&) { r = true; }
        need(a && r, "invalid speculative uniform accepted");
    }
    for (int token : {-1,3}) {
        bool refused = false; try { v.verify(token, 1., .5, .5); } catch (const std::invalid_argument&) { refused = true; }
        need(refused, "out-of-vocabulary draft accepted");
    }
    bool refused = false;
    try { G::verify_full_vocab(v.logits.data<float>(), v.q.data<double>(), 1, 3, 1., .5, .5,
        v.workspace.data<void>(), v.workspace.size - 1, v.result.data<G::FullVocabResult>(), v.accepted.data<uint32_t>(), stream); }
    catch (const std::invalid_argument&) { refused = true; }
    need(refused, "undersized verifier workspace accepted");
    refused = false;
    try { G::copy_full_vocab_probabilities(v.workspace.data<void>(), v.workspace.size,
        v.result.data<G::FullVocabResult>(), 3, v.workspace.data<double>(), stream); }
    catch (const std::invalid_argument&) { refused = true; }
    need(refused, "aliased probability copy accepted");
    v.guards();
}
static void probability_law(cudaStream_t stream) {
    const std::vector<float> logits{std::log(.1f),std::log(.2f),std::log(.3f),std::log(.4f)};
    const std::vector<double> q{.25,.25,.25,.25};
    const auto p = distribution(logits, 1.);
    Verifier v(4, stream); v.put(logits, q);
    constexpr int draws = 8192;
    Device results(size_t(draws) * sizeof(G::FullVocabResult), stream), accepted(size_t(draws) * sizeof(uint32_t), stream);
    std::mt19937_64 rng(0xDF1A5A);
    const auto uniform = [&] { return std::generate_canonical<double,53>(rng); };
    for (int i = 0; i < draws; ++i) {
        const int token = categorical(q, uniform());
        const double a = uniform(), r = uniform();
        v.launch(token, 1., a, r, results.data<G::FullVocabResult>() + i, accepted.data<uint32_t>() + i);
    }
    std::vector<G::FullVocabResult> values(size_t(draws), G::FullVocabResult{});
    std::vector<uint32_t> flags(size_t(draws), 0);
    CK(cudaMemcpyAsync(values.data(), results.data<void>(), results.size, cudaMemcpyDeviceToHost, stream));
    CK(cudaMemcpyAsync(flags.data(), accepted.data<void>(), accepted.size, cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    int counts[4] = {}, accepted_count = 0;
    for (int i = 0; i < draws; ++i) {
        need(values[size_t(i)].status == G::kSampleOk && values[size_t(i)].token >= 0 && values[size_t(i)].token < 4,
             "probability-law verifier failed");
        ++counts[values[size_t(i)].token]; accepted_count += int(flags[size_t(i)]);
    }
    double expected_acceptance = 0.;
    for (int i = 0; i < 4; ++i) {
        expected_acceptance += std::min(p[size_t(i)], q[size_t(i)]);
        const double error = std::abs(double(counts[i]) / draws - p[size_t(i)]);
        need(error < 6. * std::sqrt(p[size_t(i)] * (1.-p[size_t(i)]) / draws), "speculative mixture does not follow target distribution");
    }
    need(std::abs(double(accepted_count)/draws - expected_acceptance) <
        6. * std::sqrt(expected_acceptance * (1.-expected_acceptance) / draws), "empirical p/q acceptance law differs");
    std::printf("probability-law draws=%d target_counts=%d/%d/%d/%d accepted=%d expected_acceptance=%.9f\n",
                draws, counts[0],counts[1],counts[2],counts[3],accepted_count,expected_acceptance);
    results.guards(); accepted.guards(); v.guards();
}
int main() {
    try {
        int devices = 0; CK(cudaGetDeviceCount(&devices)); if (!devices) return 77;
        cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        independent_oracle(stream); boundaries(stream); invalid_inputs(stream); probability_law(stream);
        CK(cudaStreamDestroy(stream));
        std::printf("PASS exact speculative sampling checks=%u CPU oracle, p=q, zeros, disjoint, graph, invalids, histogram and guards\n", checks);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"gptoss_speculative_sampling_test: %s\n",error.what()); return 1; }
}
#endif
