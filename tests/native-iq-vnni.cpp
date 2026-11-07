// Link the production CPU library and a baseline iq_avx512.cpp object compiled
// with iq512_rows/iq512_gu_rows renamed to their _reference counterparts.
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/cpu_features.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
void iq512_rows_reference(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
void iq512_gu_rows_reference(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int);
}
namespace C = strata::kernels::cpu;

struct Q8K { float d; int8_t q[256]; int16_t sums[16]; };
static_assert(sizeof(Q8K) == 292, "stored Q8_K layout");
static uint32_t rng = 0x92D30A61u;
static uint32_t random_word() { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static uint32_t bits(float value) { uint32_t x; std::memcpy(&x, &value, 4); return x; }

int main() {
    if (!C::cpu_avx512_ok()) { std::puts("SKIP: AVX512 VNNI/VBMI CPU required"); return 77; }
    constexpr int rows = 9, first = 1, last = 8;
    unsigned cases = 0;
    uint64_t hash = 1469598103934665603ull;
    for (int type : {16, 17, 18, 21, 22}) {
        const size_t block_bytes = type == 16 ? 66 : type == 17 ? 74 : type == 18 ? 98 : type == 21 ? 110 : 82;
        for (int n : {256, 768, 2560, 4096}) for (int nt = 1; nt <= 8; ++nt) {
            const int blocks = n / 256;
            const size_t row_bytes = size_t(blocks) * block_bytes, up = rows * row_bytes;
            std::vector<uint8_t> weights(2 * up);
            for (auto& b : weights) b = uint8_t(random_word());
            for (int r = 0; r < 2 * rows; ++r) for (int b = 0; b < blocks; ++b) {
                const uint16_t d = uint16_t(0x1800 + ((r + b) % 6) * 0x400 + ((r + b) & 1) * 0x8000);
                std::memcpy(weights.data() + size_t(r) * row_bytes + b * block_bytes, &d, 2);
            }
            std::vector<Q8K> q(size_t(nt) * blocks);
            for (size_t b = 0; b < q.size(); ++b) {
                q[b].d = std::ldexp(1.0f, -10 + int(b % 4));
                for (int i = 0; i < 256; ++i) q[b].q[i] = int8_t(random_word() & 255);
                for (int i = 0; i < 16; ++i) {
                    int sum = 0;
                    for (int j = 0; j < 16; ++j) sum += q[b].q[16 * i + j];
                    q[b].sums[i] = int16_t(sum);
                }
            }
            const void* act[8]{};
            float* got_rows[8]{};
            float* expected_rows[8]{};
            std::vector<float> got(size_t(nt) * rows), expected(got.size());
            for (int t = 0; t < nt; ++t) {
                act[t] = q.data() + size_t(t) * blocks;
                got_rows[t] = got.data() + size_t(t) * rows;
                expected_rows[t] = expected.data() + size_t(t) * rows;
            }
            for (bool gate_up : {false, true}) {
                std::memset(got.data(), 0xA7, got.size() * sizeof(float));
                std::memset(expected.data(), 0xA7, expected.size() * sizeof(float));
                if (gate_up) {
                    C::iq512_gu_rows_reference(type, weights.data(), row_bytes, up, n, act, nt, expected_rows, first, last);
                    C::iq512_gu_rows(type, weights.data(), row_bytes, up, n, act, nt, got_rows, first, last);
                } else {
                    C::iq512_rows_reference(type, weights.data(), row_bytes, n, act, nt, expected_rows, first, last);
                    C::iq512_rows(type, weights.data(), row_bytes, n, act, nt, got_rows, first, last);
                }
                for (int t = 0; t < nt; ++t) for (int r = 0; r < rows; ++r) {
                    const size_t i = size_t(t) * rows + r;
                    const bool written = r >= first && r < last;
                    if (bits(got[i]) != bits(expected[i]) || (!written && bits(got[i]) != 0xA7A7A7A7u) ||
                        (written && !std::isfinite(got[i]))) {
                        std::fprintf(stderr, "FAIL type=%d n=%d nt=%d gu=%d token=%d row=%d got=%08x expected=%08x\n",
                                     type, n, nt, int(gate_up), t, r, bits(got[i]), bits(expected[i]));
                        return 1;
                    }
                    hash = (hash ^ bits(got[i])) * 1099511628211ull;
                }
                ++cases;
            }
        }
    }
    std::printf("PASS IQ VNNI cases=%u bitwise baseline parity hash=%016llx\n", cases, (unsigned long long)hash);
}
