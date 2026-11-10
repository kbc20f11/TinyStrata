// Model-free sparse EAGLE vocabulary and exact rejection integration.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include "strata/gemma4/eagle3.hpp"
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
#include <utility>
#include <vector>

namespace G = strata::gemma4;
#define CK(x) do { const auto e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)
static uint64_t checks = 0;
static uint64_t verified = 0;
static void need(bool okay, const char* what) { if (!okay) throw std::runtime_error(what); ++checks; }

struct Buffer {
    uint8_t* base = nullptr;
    size_t bytes;
    cudaStream_t stream;
    Buffer(size_t n, cudaStream_t s) : bytes(n), stream(s) {
        CK(cudaMalloc(reinterpret_cast<void**>(&base), n + 128));
        CK(cudaMemsetAsync(base, 0xA7, n + 128, stream));
    }
    ~Buffer() { if (base) (void) cudaFree(base); }
    template<class T> T* data() const { return reinterpret_cast<T*>(base + 64); }
    void put(const void* src) { CK(cudaMemcpyAsync(base + 64, src, bytes, cudaMemcpyHostToDevice, stream)); }
    template<class T> std::vector<T> get() const {
        std::vector<T> out(bytes / sizeof(T));
        CK(cudaMemcpyAsync(out.data(), base + 64, bytes, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        return out;
    }
    void guards() const {
        CK(cudaStreamSynchronize(stream));
        uint8_t first[64], last[64];
        CK(cudaMemcpy(first, base, 64, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(last, base + 64 + bytes, 64, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 64; ++i) need(first[i] == 0xA7 && last[i] == 0xA7, "buffer red zone overwritten");
    }
};

static std::vector<double> distribution(const std::vector<float>& logits, double temperature) {
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = std::exp((double(logits[i]) - maximum) / temperature);
    const double total = std::accumulate(result.begin(), result.end(), 0.);
    for (double& q : result) q /= total;
    return result;
}

static int categorical(const std::vector<double>& weights, double uniform) {
    const double total = std::accumulate(weights.begin(), weights.end(), 0.);
    const double threshold = std::min(uniform * total, std::nextafter(total, 0.));
    double sum = 0.; int last = -1;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i];
        if (weights[i] > 0.) { last = int(i); if (threshold < sum) return int(i); }
    }
    return last;
}

static std::vector<int32_t> mapping(int draft, int target, unsigned seed) {
    std::vector<int32_t> ids(static_cast<size_t>(target));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 random(seed);
    std::shuffle(ids.begin(), ids.end(), random);
    ids.resize(size_t(draft));
    return ids;
}

static void check_scatter(const std::vector<double>& q, const std::vector<int32_t>& ids,
                          const std::vector<double>& got, bool normalized = true) {
    std::vector<double> expected(got.size(), 0.);
    for (size_t i = 0; i < q.size(); ++i) expected[size_t(ids[i])] = q[i];
    for (size_t i = 0; i < expected.size(); ++i)
        need(std::memcmp(&expected[i], &got[i], sizeof(double)) == 0, "sparse mapping changed bits or failed to clear outside support");
    if (normalized)
        need(std::abs(std::accumulate(got.begin(), got.end(), 0.) - 1.) < 1e-10, "expanded q does not sum to one");
}

static void direct_scatter(cudaStream_t stream) {
    for (const auto dims : {std::pair<int,int>{1,9}, {7,31}, {64000,201088}}) {
        const int draft = dims.first, target = dims.second;
        Buffer input(size_t(draft) * sizeof(double), stream), map(size_t(draft) * sizeof(int32_t), stream),
               output(size_t(target) * sizeof(double), stream);
        std::vector<double> q(size_t(draft), 1. / double(draft));
        if (draft == 7) q = {-.0, .125, .25, .625, std::numeric_limits<double>::denorm_min(), 0., 0.};
        auto ids = mapping(draft, target, 1971 + unsigned(draft));
        input.put(q.data()); map.put(ids.data());
        G::eagle3_scatter_probabilities(input.data<double>(), map.data<int32_t>(), output.data<double>(), draft, target, stream);
        check_scatter(q, ids, output.get<double>());
        const auto unchanged = input.get<double>();
        need(std::memcmp(q.data(), unchanged.data(), input.bytes) == 0, "scatter overwrote reduced q");
        // Reuse exactly the same output, changing both values and sparse support.
        std::reverse(q.begin(), q.end());
        ids = mapping(draft, target, 499 + unsigned(draft));
        input.put(q.data()); map.put(ids.data());
        G::eagle3_scatter_probabilities(input.data<double>(), map.data<int32_t>(), output.data<double>(), draft, target, stream);
        check_scatter(q, ids, output.get<double>());
        input.guards(); map.guards(); output.guards();
    }
}

struct Outcome { G::FullVocabResult result; uint32_t accepted; };
struct Pipeline {
    int draft, target;
    cudaStream_t stream;
    Buffer logits, target_logits, map, reduced, full, workspace, sample_results, result, accepted;
    std::vector<float> h_logits, h_target;
    std::vector<int32_t> ids;
    const double temperatures[2] = {.65, 1.7};
    const double uniforms[2] = {.137, .813};
    Pipeline(int d, int t, cudaStream_t s) : draft(d), target(t), stream(s),
        logits(size_t(2) * d * sizeof(float),s), target_logits(size_t(t) * sizeof(float),s),
        map(size_t(d) * sizeof(int32_t),s), reduced(size_t(2) * d * sizeof(double),s),
        full(size_t(2) * t * sizeof(double),s), workspace(G::full_vocab_sample_workspace_bytes(t),s),
        sample_results(2 * sizeof(G::FullVocabResult),s), result(sizeof(G::FullVocabResult),s),
        accepted(sizeof(uint32_t),s), h_logits(size_t(2) * d), h_target(size_t(t)) {}
    void fill(int phase) {
        for (int r = 0; r < 2; ++r)
            for (int i = 0; i < draft; ++i)
                h_logits[size_t(r) * draft + i] = float(((i * 17 + phase * 31 + r * 7) % 97) - 48) * .0625f;
        for (int i = 0; i < target; ++i) h_target[size_t(i)] = float(((i * 29 + phase * 13) % 73) - 36) * .03125f;
        ids = mapping(draft, target, 0xEAC1E + unsigned(phase));
        logits.put(h_logits.data()); target_logits.put(h_target.data()); map.put(ids.data());
        CK(cudaStreamSynchronize(stream));
    }
    void launch_row(int row) {
        G::sample_full_vocab(logits.data<float>() + size_t(row) * draft, draft, temperatures[row], uniforms[row],
            workspace.data<void>(), workspace.bytes, sample_results.data<G::FullVocabResult>() + row, stream);
        G::copy_full_vocab_probabilities(workspace.data<void>(), workspace.bytes,
            sample_results.data<G::FullVocabResult>() + row, draft, reduced.data<double>() + size_t(row) * draft, stream);
        G::eagle3_scatter_probabilities(reduced.data<double>() + size_t(row) * draft, map.data<int32_t>(),
            full.data<double>() + size_t(row) * target, draft, target, stream);
    }
    void launch() { launch_row(0); launch_row(1); }
    std::vector<double> check_rows() {
        const auto q = reduced.get<double>(), expanded = full.get<double>();
        const auto samples = sample_results.get<G::FullVocabResult>();
        for (int row = 0; row < 2; ++row) {
            const std::vector<double> qr(q.begin() + size_t(row) * draft, q.begin() + size_t(row + 1) * draft);
            const std::vector<double> out(expanded.begin() + size_t(row) * target, expanded.begin() + size_t(row + 1) * target);
            check_scatter(qr, ids, out);
            const std::vector<float> raw(h_logits.begin() + size_t(row) * draft, h_logits.begin() + size_t(row + 1) * draft);
            const auto expected = distribution(raw, temperatures[row]);
            for (int i = 0; i < draft; ++i) need(std::abs(qr[size_t(i)] - expected[size_t(i)]) < 1e-13,
                                                "reduced GPU normalization differs from CPU");
            need(samples[size_t(row)].status == G::kSampleOk && samples[size_t(row)].token == categorical(expected, uniforms[row]),
                 "reduced sample differs from independent CPU CDF");
        }
        return expanded;
    }
    Outcome verify(int row, int token, double a, double r) {
        G::verify_full_vocab(target_logits.data<float>(), full.data<double>() + size_t(row) * target,
            token, target, temperatures[row], a, r, workspace.data<void>(), workspace.bytes,
            result.data<G::FullVocabResult>(), accepted.data<uint32_t>(), stream);
        return {result.get<G::FullVocabResult>()[0], accepted.get<uint32_t>()[0]};
    }
    void guards() {
        logits.guards(); target_logits.guards(); map.guards(); reduced.guards(); full.guards();
        workspace.guards(); sample_results.guards(); result.guards(); accepted.guards();
    }
};

static void check_outcome(const Outcome& got, const std::vector<double>& p, const std::vector<double>& q,
                           int token, double a, double r) {
    const bool accept = a < std::min(1., p[size_t(token)] / q[size_t(token)]);
    std::vector<double> residual(p.size());
    for (size_t i = 0; i < p.size(); ++i) residual[i] = std::max(p[i] - q[i], 0.);
    const int expected = accept ? token : categorical(residual, r);
    const double prob = accept ? p[size_t(token)] : residual[size_t(expected)] /
        std::accumulate(residual.begin(), residual.end(), 0.);
    need(got.result.status == G::kSampleOk && got.accepted == uint32_t(accept), "sparse p/q accept law differs from CPU");
    need(got.result.token == expected, "sparse p/q residual CDF differs from CPU");
    need(std::abs(got.result.probability - prob) < 1e-12, "sparse p/q selected probability differs from CPU");
    ++verified;
}

static void large_pipeline_graph(cudaStream_t stream) {
    Pipeline f(64000, 201088, stream);
    f.fill(0);
    // Row1's temporary sampler workspace and scatter must not overwrite row0.
    f.launch_row(0);
    const auto first = f.full.get<double>();
    f.launch_row(1);
    const auto both = f.check_rows();
    need(std::memcmp(first.data(), both.data(), size_t(f.target) * sizeof(double)) == 0, "second proposal overwrote first q row");
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); f.launch();
    CK(cudaStreamEndCapture(stream, &graph)); CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int phase : {0, 1}) {
        f.fill(phase); // Same captured addresses, new draft logits and d2t support.
        CK(cudaGraphLaunch(exec, stream));
        const auto q = f.check_rows();
        const auto draws = f.sample_results.get<G::FullVocabResult>();
        for (int row = 0; row < 2; ++row) {
            const auto p = distribution(f.h_target, f.temperatures[row]);
            const std::vector<double> qr(q.begin() + size_t(row) * f.target, q.begin() + size_t(row + 1) * f.target);
            const int token = f.ids[size_t(draws[size_t(row)].token)];
            for (double a : {0., .13, .81, std::nextafter(1., 0.)})
                for (double r : {0., .23, .79, std::nextafter(1., 0.)})
                    check_outcome(f.verify(row, token, a, r), p, qr, token, a, r);
        }
        const auto after = f.full.get<double>();
        need(std::memcmp(q.data(), after.data(), f.full.bytes) == 0, "target verifier overwrote saved sparse proposal rows");
    }
    CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
    f.guards();
}

static void sparse_probability_law(cudaStream_t stream) {
    constexpr int target = 9, draft = 3;
    Pipeline f(draft, target, stream); f.fill(0);
    f.ids = {8, 0, 4}; f.map.put(f.ids.data());
    const std::vector<double> q_reduced{.5,.3,.2};
    CK(cudaMemcpyAsync(f.reduced.data<double>(), q_reduced.data(), draft * sizeof(double), cudaMemcpyHostToDevice, stream));
    G::eagle3_scatter_probabilities(f.reduced.data<double>(), f.map.data<int32_t>(), f.full.data<double>(), draft, target, stream);
    const auto all = f.full.get<double>();
    const std::vector<double> q(all.begin(), all.begin() + target);
    check_scatter(q_reduced, f.ids, q);
    std::mt19937_64 rng(0xEAC1E320BULL);
    const auto uniform = [&] { return std::generate_canonical<double,53>(rng); };
    for (int row = 0; row < 2; ++row) {
        // Only row0 is used here; vary the target law while retaining sparse q.
        const double temperature = f.temperatures[0];
        for (int i = 0; i < target; ++i)
            f.h_target[size_t(i)] = float(std::log(double(i + row + 1))) * float(temperature);
        f.target_logits.put(f.h_target.data());
        const auto p = distribution(f.h_target, temperature);
        double accept_mass = 0.;
        std::vector<double> residual(target), mixture(target);
        for (int i = 0; i < target; ++i) { accept_mass += std::min(p[size_t(i)], q[size_t(i)]); residual[size_t(i)] = std::max(p[size_t(i)] - q[size_t(i)], 0.); }
        const double remainder = std::accumulate(residual.begin(), residual.end(), 0.);
        for (int i = 0; i < target; ++i) {
            mixture[size_t(i)] = std::min(p[size_t(i)], q[size_t(i)]) + (1. - accept_mass) * residual[size_t(i)] / remainder;
            need(std::abs(mixture[size_t(i)] - p[size_t(i)]) < 1e-14, "analytic sparse rejection mixture is not p");
        }
        constexpr int draws = 192;
        int counts[target] = {}, accepted_count = 0;
        for (int i = 0; i < draws; ++i) {
            const int token = categorical(q, uniform());
            const double a = uniform(), r = uniform();
            const auto got = f.verify(0, token, a, r);
            check_outcome(got, p, q, token, a, r);
            ++counts[got.result.token]; accepted_count += int(got.accepted);
        }
        for (int i = 0; i < target; ++i)
            need(std::abs(double(counts[i]) / draws - p[size_t(i)]) <
                 6. * std::sqrt(p[size_t(i)] * (1. - p[size_t(i)]) / draws), "sparse sample mixture histogram differs from p");
        need(accepted_count > 0 && accepted_count < draws, "probability-law fixture did not exercise both acceptance and rejection");
        std::printf("sparse-law draws=%d accepted=%d expected_acceptance=%.9f\n", draws, accepted_count, accept_mass);
    }
    f.guards();
}

static void invalid_arguments(cudaStream_t stream) {
    Buffer q(3*sizeof(double),stream), map(3*sizeof(int32_t),stream), full(9*sizeof(double),stream);
    const auto refused = [&](const double* input, const int32_t* ids, double* out, int d, int t) {
        bool error = false;
        try { G::eagle3_scatter_probabilities(input,ids,out,d,t,stream); }
        catch (const std::invalid_argument&) { error = true; }
        need(error, "invalid sparse expansion arguments were accepted");
    };
    refused(nullptr,map.data<int32_t>(),full.data<double>(),3,9);
    refused(q.data<double>(),nullptr,full.data<double>(),3,9);
    refused(q.data<double>(),map.data<int32_t>(),nullptr,3,9);
    refused(q.data<double>(),map.data<int32_t>(),q.data<double>(),3,9);
    refused(q.data<double>(),map.data<int32_t>(),full.data<double>(),0,9);
    refused(q.data<double>(),map.data<int32_t>(),full.data<double>(),3,2);
    q.guards(); map.guards(); full.guards();
}

int main() {
    try {
        int devices = 0; CK(cudaGetDeviceCount(&devices)); if (!devices) return 77;
        cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        direct_scatter(stream); large_pipeline_graph(stream); sparse_probability_law(stream); invalid_arguments(stream);
        CK(cudaStreamDestroy(stream));
        std::printf("PASS EAGLE sparse q law checks=%llu verified=%llu target=201088 draft=64000 double-bits/zero-support/two-rows/changed-input-graph/CPU-pq/guards\n",
                    (unsigned long long)checks,(unsigned long long)verified);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"gptoss_eagle3_sampling_test: %s\n",error.what()); return 1; }
}
#endif
