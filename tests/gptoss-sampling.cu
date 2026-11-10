// Model-free boundaries plus optional saved-logit/same-uniform categorical comparison.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include "strata/gemma4/sampling.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace G = strata::gemma4;
#define CK(x) do { const auto e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)
static unsigned checks = 0;
static void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); ++checks; }
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
struct Sampler {
    int vocab;
    cudaStream_t stream;
    Device logits, workspace, result;
    Sampler(int n, cudaStream_t s) : vocab(n), stream(s), logits(size_t(n) * 4, s),
        workspace(G::full_vocab_sample_workspace_bytes(n), s), result(sizeof(G::FullVocabResult), s) {}
    void put(const std::vector<float>& values) {
        need(values.size() == size_t(vocab), "logit length differs");
        CK(cudaMemcpyAsync(logits.data<void>(), values.data(), values.size() * 4, cudaMemcpyHostToDevice, stream));
    }
    G::FullVocabResult draw(double temperature, double uniform, bool graph = false) {
        if (graph) CK(cudaStreamSynchronize(stream));
        const auto launch = [&] { G::sample_full_vocab(logits.data<float>(), vocab, temperature, uniform,
            workspace.data<void>(), workspace.size, result.data<G::FullVocabResult>(), stream); };
        cudaGraph_t g = nullptr; cudaGraphExec_t exec = nullptr;
        if (graph) CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch();
        if (graph) {
            CK(cudaStreamEndCapture(stream, &g));
            CK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
            CK(cudaGraphLaunch(exec, stream));
        }
        G::FullVocabResult host{};
        CK(cudaMemcpyAsync(&host, result.data<void>(), sizeof(host), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        if (graph) { CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(g)); }
        return host;
    }
    void guards() const { logits.guards(); workspace.guards(); result.guards(); }
};
static void boundaries(cudaStream_t stream) {
    for (const int n : {1, 2, 31, 32, 255, 256, 257, 2047, 2048, 2049, 201088}) {
        Sampler sampler(n, stream);
        std::vector<float> values(n, 1234.5f);
        sampler.put(values);
        for (const bool graph : {false, true}) {
            for (const double u : {0.0, 0.25, 0.5, std::nextafter(1.0, 0.0)}) {
                const auto r = sampler.draw(1.0, u, graph);
                const int token = std::min(n - 1, int(u * n));
                need(r.status == G::kSampleOk && r.token == token, "uniform vocabulary boundary differs");
                need(r.total == double(n), "equal logits lost probability mass");
                need(std::abs(r.probability - 1.0 / n) <= 1e-15, "equal-logit probability differs");
            }
        }
        sampler.guards();
    }
    {
        Sampler sampler(2049, stream);
        std::vector<float> values(2049, -1.f); values.back() = 1.f;
        sampler.put(values);
        for (const double t : {std::numeric_limits<double>::denorm_min(), 1e-300, 1e-10})
            for (const double u : {0.0, std::nextafter(1.0, 0.0)}) {
                const auto r = sampler.draw(t, u);
                need(r.status == G::kSampleOk && r.token == 2048 && r.total == 1.0, "tiny-temperature maximum differs");
            }
        sampler.guards();
    }
    {
        Sampler sampler(4, stream);
        for (const float tail : {-700.f, -744.f}) {
            sampler.put({-1000.f, -800.f, tail, 0.f});
            const auto r = sampler.draw(1.0, 0.0);
            need(r.status == G::kSampleOk && r.token == 2 && r.probability > 0.0, "a finite small tail was cut off");
        }
        for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()}) {
            sampler.put({0.f, 1.f, bad, 2.f});
            const auto invalid = sampler.draw(1.0, 0.5);
            need(invalid.status == G::kSampleNonFinite && invalid.token == -1, "nonfinite logits were accepted");
        }
        sampler.put({0.f, 1.f, 2.f, 3.f});
        for (const double t : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            bool refused = false; try { sampler.draw(t, 0.5); } catch (const std::invalid_argument&) { refused = true; }
            need(refused, "invalid temperature accepted");
        }
        for (const double u : {-1.0, 1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            bool refused = false; try { sampler.draw(1.0, u); } catch (const std::invalid_argument&) { refused = true; }
            need(refused, "invalid uniform accepted");
        }
        bool refused = false;
        try { G::sample_full_vocab(sampler.logits.data<float>(), 4, 1.0, 0.5, sampler.workspace.data<void>(),
                                  G::full_vocab_sample_workspace_bytes(4) - 1, sampler.result.data<G::FullVocabResult>(), stream); }
        catch (const std::invalid_argument&) { refused = true; }
        need(refused, "undersized workspace accepted");
        sampler.guards();
    }
}

struct Interval { double lower = 0, upper = 0; };
static Interval interval(const std::vector<double>& p, int token) {
    double sum = 0.0;
    for (int i = 0; i < token; ++i) sum += p[size_t(i)];
    return {sum, sum + p[size_t(token)]};
}
static double margin(double u, const Interval& x) { return std::min(std::abs(u - x.lower), std::abs(u - x.upper)); }

static void real_logits(const std::string& file, int rows, int vocab, uint64_t seed, double temperature,
                        int draws, const std::string& output, cudaStream_t stream) {
    std::ifstream input(file, std::ios::binary);
    need(bool(input), "cannot open saved logits");
    std::ofstream jsonl;
    if (!output.empty()) { jsonl.open(output, std::ios::binary); need(bool(jsonl), "cannot open JSONL output"); }
    Sampler sampler(vocab, stream);
    std::mt19937_64 rng(seed);
    std::vector<float> logits(vocab);
    unsigned mismatches = 0, outside_margin = 0;
    for (int row = 0; row < rows; ++row) {
        input.read(reinterpret_cast<char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
        need(input.gcount() == std::streamsize(logits.size() * sizeof(float)), "saved logits ended early");
        double maximum = -std::numeric_limits<double>::infinity();
        for (float x : logits) { need(std::isfinite(x), "saved logits contain nonfinite values"); maximum = std::max(maximum, double(x)); }
        std::vector<double> weights(vocab);
        for (int i = 0; i < vocab; ++i) weights[size_t(i)] = std::exp((double(logits[size_t(i)]) - maximum) / temperature);
        std::discrete_distribution<size_t> cpu(weights.begin(), weights.end());
        const auto probabilities = cpu.probabilities();
        sampler.put(logits);
        for (int draw = 0; draw < draws; ++draw) {
            auto cpu_rng = rng;
            // Exactly the STL uniform primitive used by discrete_distribution on the native toolchain.
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
            const int cpu_token = int(cpu(cpu_rng));
            const bool rng_equal = rng == cpu_rng;
            need(rng_equal, "CPU categorical and supplied uniform consumed different RNG states");
            const auto gpu = sampler.draw(temperature, u);
            need(gpu.status == G::kSampleOk && gpu.token >= 0 && gpu.token < vocab, "GPU categorical failed");
            const auto a = interval(probabilities, cpu_token), b = interval(probabilities, gpu.token);
            if (cpu_token != gpu.token) {
                ++mismatches;
                if (std::min(margin(u, a), margin(u, b)) > 1e-12) ++outside_margin;
            }
            if (jsonl) jsonl << std::setprecision(17) << "{\"row\":" << row << ",\"draw\":" << draw
                << ",\"temperature\":" << temperature << ",\"uniform\":" << u
                << ",\"cpu_token\":" << cpu_token << ",\"gpu_token\":" << gpu.token << ",\"status\":" << gpu.status
                << ",\"cpu_cdf_lower\":" << a.lower << ",\"cpu_cdf_upper\":" << a.upper
                << ",\"gpu_cdf_lower\":" << b.lower << ",\"gpu_cdf_upper\":" << b.upper
                << ",\"gpu_total\":" << gpu.total << ",\"gpu_probability\":" << gpu.probability
                << ",\"rng_state_equal\":" << (rng_equal ? "true" : "false") << "}\n";
        }
    }
    sampler.guards();
    std::printf("real logits rows=%d draws=%d mismatches=%u outside_1e-12_boundary=%u\n", rows, draws, mismatches, outside_margin);
    need(outside_margin == 0, "CPU/GPU token difference away from a CDF boundary");
}

int main(int argc, char** argv) {
    try {
        std::string file, output;
        int rows = 0, vocab = 201088, draws = 1;
        uint64_t seed = 1234;
        double temperature = 1.0;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (++i == argc) throw std::invalid_argument("missing option value");
            const std::string value = argv[i];
            if (key == "--logits") file = value;
            else if (key == "--rows") rows = std::stoi(value);
            else if (key == "--vocab") vocab = std::stoi(value);
            else if (key == "--seed") seed = std::stoull(value);
            else if (key == "--draws") draws = std::stoi(value);
            else if (key == "--temperature") temperature = std::stod(value);
            else if (key == "--output") output = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        need(vocab > 0 && draws > 0 && temperature > 0 && std::isfinite(temperature), "invalid test dimensions/temperature");
        int devices = 0; CK(cudaGetDeviceCount(&devices)); if (!devices) return 77;
        cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        boundaries(stream);
        if (!file.empty()) { need(rows > 0, "--rows must be positive with --logits"); real_logits(file, rows, vocab, seed, temperature, draws, output, stream); }
        CK(cudaStreamDestroy(stream));
        std::printf("PASS full-vocabulary sampler checks=%u, full mass, endpoints, invalids, graph and guards\n", checks);
        return 0;
    } catch (const std::exception& ex) { std::fprintf(stderr, "gptoss_sampling_test: %s\n", ex.what()); return 1; }
}
#endif
