// Native CPU expert scheduling parity: no GPU, model downloads or services.
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace C = strata::kernels::cpu;

namespace {
void environment(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct alignas(64) Activation {
    unsigned char bytes[C::kNativeActBytes];
};

struct Fixture {
    C::NativeFmt format;
    int width;
    std::vector<unsigned char> weights;
    std::vector<Activation> activations;
    std::vector<C::ExpertJobMulti> jobs;
    std::vector<float> actual, expected;

    Fixture(int gu_type, int down_type, int experts, int max_tokens, unsigned seed = 12345)
        : width(max_tokens), activations((size_t) experts * max_tokens), jobs(experts),
          actual((size_t) experts * max_tokens * 2048, -123.375f), expected(actual) {
        std::string error;
        require(C::native_fmt(gu_type, down_type, 2048, 512, format, error), error.c_str());
        weights.resize(format.bytes);
        std::mt19937 random(seed);
        std::normal_distribution<float> normal(0.f, .018f);
        std::vector<float> input(2048 * 512), importance(2048, 1.f);
        for (int projection = 0; projection < 3; ++projection) {
            for (float& value : input) value = normal(random);
            const bool down = projection == 2;
            const size_t offset = projection == 0 ? 0 : down ? format.down_off : format.up_off;
            ggml_quantize_chunk((ggml_type) (down ? down_type : gu_type), input.data(), weights.data() + offset,
                                0, down ? 2048 : 512, down ? 512 : 2048, importance.data());
        }
        input.resize(2048);
        for (int expert = 0; expert < experts; ++expert) {
            auto& job = jobs[expert];
            job.blob = weights.data();
            job.nt = 1 + expert % max_tokens;
            for (int token = 0; token < job.nt; ++token) {
                for (float& value : input) value = normal(random) * 30.f;
                C::native_quant_act(format, input.data(), activations[expert * max_tokens + token].bytes);
                job.nact[token] = activations[expert * max_tokens + token].bytes;
                job.out[token] = actual.data() + ((size_t) expert * max_tokens + token) * 2048;
            }
        }
        std::vector<float> hidden((size_t) max_tokens * 512);
        std::vector<unsigned char> quantized((size_t) max_tokens * C::kNativeHBytes);
        for (int expert = 0; expert < experts; ++expert) {
            const auto& job = jobs[expert];
            float* ff[C::MAXT];
            float* out[C::MAXT];
            const void* hq[C::MAXT];
            for (int token = 0; token < job.nt; ++token) {
                ff[token] = hidden.data() + token * 512;
                hq[token] = quantized.data() + token * C::kNativeHBytes;
                out[token] = expected.data() + ((size_t) expert * max_tokens + token) * 2048;
            }
            C::native_gu_rows(format, job.blob, job.nact, job.nt, ff, 0, 512);
            for (int token = 0; token < job.nt; ++token)
                C::native_quant_h(format, ff[token], quantized.data() + token * C::kNativeHBytes);
            C::native_down_rows(format, job.blob, hq, job.nt, out, 0, 2048);
        }
    }

    void reset() { std::fill(actual.begin(), actual.end(), -123.375f); }

    void check(int experts) const {
        for (int expert = 0; expert < experts; ++expert) {
            const size_t start = (size_t) expert * width * 2048;
            require(std::memcmp(actual.data() + start, expected.data() + start,
                                (size_t) jobs[expert].nt * 2048 * sizeof(float)) == 0,
                    "pool output differs from serial native expert rows");
            for (int token = jobs[expert].nt; token < width; ++token)
                for (int row = 0; row < 2048; ++row)
                    require(actual[start + token * 2048 + row] == -123.375f,
                            "pool wrote outside an expert's token count");
        }
        // Poison every call, then also inspect jobs not submitted: a late old-batch writer must not be hidden
        // by the previous call's already-correct output or by checking only the requested prefix.
        for (size_t i = (size_t) experts * width * 2048; i < actual.size(); ++i)
            require(actual[i] == -123.375f, "pool wrote an unsubmitted expert");
    }
};

int exercise(Fixture& fixture, int workers, bool host_works, int tasks, const char* spin) {
    int checks = 0;
    for (int park : {1, 0, 2}) for (bool fused : {false, true}) {
        environment("STRATA_POOL_FUSED", fused ? "1" : "0");
        environment("STRATA_POOL_PARK_BARRIER", std::to_string(park));
        environment("STRATA_POOL_SPIN_US", spin);
        C::ExpertPool pool(workers, false, host_works, C::PoolAffinity::All, tasks);
        require(pool.park_barrier_mode() == park && pool.park_barrier() == (park != 0), "pool selected the wrong completion policy");
        fixture.reset();
        pool.run_split_multi_native(fixture.format, nullptr, 0);
        pool.run_split_multi_native(fixture.format, nullptr, -1);
        fixture.check(0);
        require(pool.completed_phases() == 0, "empty submissions published a phase");
        for (int iteration = 0; iteration < 32; ++iteration) {
            if (iteration % 8 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(25));
            const int count = iteration % 2 ? (int) fixture.jobs.size() : 1 + iteration % fixture.jobs.size();
            fixture.reset();
            const uint64_t before = pool.completed_phases();
            pool.run_split_multi_native(fixture.format, fixture.jobs.data(), count);
            require(pool.completed_phases() > before, "native batch did not complete a phase");
            fixture.check(count);
            ++checks;
        }
    }
    return checks;
}

int lifecycle_stress(Fixture& first, Fixture& second) {
    int checks = 0;
    for (int park : {1, 0, 2}) for (bool fused : {false, true}) for (bool host : {false, true}) {
        environment("STRATA_POOL_PARK_BARRIER", std::to_string(park));
        environment("STRATA_POOL_FUSED", fused ? "1" : "0");
        for (int cycle = 0; cycle < 8; ++cycle) {
            environment("STRATA_POOL_SPIN_US", cycle % 2 ? "0" : "20000");
            // Start with work immediately: some newly created workers can wake only after a host-drained batch.
            // Destroy immediately after the final checked batch: no implicit re-park wait is added by the test.
            C::ExpertPool pool(3, false, host, C::PoolAffinity::All, cycle % 2 ? 1 : 0);
            require(pool.park_barrier_mode() == park && pool.park_barrier() == (park != 0), "lifecycle pool selected the wrong completion policy");
            require(pool.completed_phases() == 0, "new pool already has completed phases");
            for (int iteration = 0; iteration < 6; ++iteration) {
                Fixture& fixture = iteration % 2 ? second : first;
                const int count = iteration % 3 == 0 ? 1 : iteration % 3 == 1 ? 3 : (int) fixture.jobs.size();
                fixture.reset();
                const uint64_t before = pool.completed_phases();
                {
                    // A fresh descriptor array dies as soon as run returns, exercising its promised lifetime.
                    auto jobs = fixture.jobs;
                    pool.run_split_multi_native(fixture.format, jobs.data(), count);
                }
                require(pool.completed_phases() > before, "lifecycle batch did not complete a phase");
                fixture.check(count);
                ++checks;
                const uint64_t completed = pool.completed_phases();
                pool.run_split_multi_native(fixture.format, nullptr, 0);
                pool.run_split_multi_native(fixture.format, nullptr, -1);
                require(pool.completed_phases() == completed, "empty submissions changed the phase count");
                if (iteration == 2 && cycle % 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
    return checks;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const int mt = argc > 1 ? std::atoi(argv[1]) : 2;
        require(mt == 1 || mt == 2, "expected native expert token policy 1 or 2");
        environment("STRATA_IQ_MT_MIN", std::to_string(mt));
        int checks = 0;
        // The routed-expert format pairs of the pinned Qwen and Clover files,
        // plus the former Q4_0 Clover format.
        for (auto types : {std::pair{22, 21}, {22, 23}, {21, 23}, {10, 11}, {18, 21}, {2, 2}}) {
            Fixture fixture(types.first, types.second, 8, 8);
            checks += exercise(fixture, 3, true, 17, "0");
        }
        Fixture batches(18, 21, 97, 1);
        checks += exercise(batches, 3, true, 0, "20000");
        Fixture auto_grain(22, 21, 8, 8);
        checks += exercise(auto_grain, 3, true, 0, "0");
        Fixture one_worker(22, 21, 7, 4);
        checks += exercise(one_worker, 1, false, 0, "0");
        Fixture alternate(18, 21, 7, 2, 98765);
        checks += lifecycle_stress(auto_grain, alternate);
        std::printf("PASS native expert pool: %d batches, MT%d, serial/legacy/fused and park/epoch bitwise parity\n", checks, mt);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native expert pool: %s\n", error.what());
        return 1;
    }
}
