#include "strata/intel/npu.hpp"
#include "strata/intel/device.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

template<class T> T read(std::ifstream& f) {
    T value{};
    if (!f.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("Truncated fixture");
    return value;
}
void bytes(std::ifstream& f, void* target, size_t size) {
    if (!f.read(static_cast<char*>(target), std::streamsize(size))) throw std::runtime_error("Truncated fixture payload");
}
template<class T> struct Buffer {
    sycl::queue& q; T* data;
    Buffer(sycl::queue& queue, size_t count) : q(queue), data(sycl::malloc_host<T>(count, queue)) {
        if (!data) throw std::runtime_error("USM fixture allocation failed");
    }
    ~Buffer() { sycl::free(data, q); }
};
int main(int argc, char** argv) {
    try {
        strata::paths::Utf8Arguments args(argc, argv); argc = args.argc(); argv = args.argv();
        if (argc != 2) throw std::runtime_error("Usage: intel_npu_projection_test quantization-reference.bin");
        std::ifstream fixture(strata::paths::native_path(argv[1]), std::ios::binary);
        sycl::queue q(strata::intel::select_device(0), sycl::property::queue::in_order{});
        strata::intel::NpuProjection npu(q);
        const auto cases = read<uint32_t>(fixture);
        if (cases == 0 || cases > 100) throw std::runtime_error("Invalid fixture count");
        constexpr int tokens = 128;
        for (uint32_t c = 0; c < cases; ++c) {
            const int type = read<int32_t>(fixture), rows = read<int32_t>(fixture), width = read<int32_t>(fixture);
            const auto size = read<uint64_t>(fixture);
            if (rows < 1 || rows > 100 || width < 1 || width > 65536 || size > 16 * 1024 * 1024) throw std::runtime_error("Invalid fixture geometry");
            Buffer<uint8_t> raw(q, size);
            std::vector<float> original_x(width), weights(size_t(rows) * width);
            std::vector<double> original_dot(rows);
            bytes(fixture, raw.data, size); bytes(fixture, original_x.data(), width * sizeof(float));
            bytes(fixture, weights.data(), weights.size() * sizeof(float)); bytes(fixture, original_dot.data(), rows * sizeof(double));
            Buffer<float> x(q, size_t(tokens) * width), y(q, size_t(tokens) * rows);
            for (int t = 0; t < tokens; ++t) for (int col = 0; col < width; ++col)
                x.data[size_t(t) * width + col] = original_x[col] * float(t + 1) / tokens + .01f * float((t + col) % 9 - 4);
            npu.run(type, raw.data, x.data, y.data, rows, width, tokens).wait_and_throw();
            double maximum = 0;
            for (int t = 0; t < tokens; ++t) for (int row = 0; row < rows; ++row) {
                double expected = 0;
                for (int col = 0; col < width; ++col) expected += double(float(sycl::half(weights[size_t(row) * width + col]))) * float(sycl::half(x.data[size_t(t) * width + col]));
                const float actual = y.data[size_t(t) * rows + row];
                const double error = std::abs(actual - expected);
                // The NPU MatMul rounds its result to FP16 before Convert(F32).
                // Keep that rounding explicit, with the original small FP32
                // accumulation tolerance; a F32 output tensor does not imply
                // an internally F32 result.
                const double output_rounding = std::abs(double(float(sycl::half(expected))) - expected);
                if (!std::isfinite(actual) || error > output_rounding + 5e-5 * (1 + std::abs(expected))) {
                    std::fprintf(stderr, "npu_mismatch case=%u type=%d token=%d row=%d actual=%.12g expected=%.12g abs_error=%.12g half_expected=%.12g\n",
                        c, type, t, row, actual, expected, error, float(sycl::half(expected)));
                    throw std::runtime_error("NPU projection disagrees with FP16-multiplicand FP64 reference");
                }
                maximum = std::max(maximum, error);
            }
            std::printf("npu_case=%u type=%d rows=%d width=%d tokens=%d max_abs_error=%.9g result=passed\n", c, type, rows, width, tokens, maximum);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NPU projection check failed: %s\n", error.what());
        return 1;
    }
}
