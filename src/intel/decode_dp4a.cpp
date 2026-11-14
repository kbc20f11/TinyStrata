// GGUF integer interpretation follows GGML and the Gufo packed-word/sub-block
// approach (MIT; NOTICE, third_party/ggml/LICENSE, licenses/gufo.txt).
// Explicit SIMD dispatch, Q8 input quantization and IQ2_S support are TinyStrata's.
#include "strata/intel/decode_dp4a.hpp"
#include "strata/intel/quant.hpp"
#ifdef TINYSTRATA_INTEL_INT8_XMX
#include "strata/intel/q8_xmx.hpp"
#endif
#include <sycl/ext/intel/esimd.hpp>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

namespace strata::intel {
namespace {
namespace es = sycl::ext::intel::esimd;
constexpr int lanes = 16;
using U = es::simd<uint32_t, lanes>;
using I = es::simd<int32_t, lanes>;
using F = es::simd<float, lanes>;
template<class T, int N, class Offset>
SYCL_ESIMD_FUNCTION ESIMD_INLINE es::simd<T, N> cached(const T* pointer, es::simd<Offset, N> offsets) {
    return es::gather<T, N>(pointer, offsets, es::properties{
        es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>});
}
SYCL_ESIMD_FUNCTION ESIMD_INLINE U words(const uint8_t* base, U offsets) {
    es::simd<uint32_t, lanes * 2> halves;
    halves.template select<lanes, 2>(0) = offsets;
    halves.template select<lanes, 2>(1) = offsets + 2;
    auto data = cached<uint16_t, lanes * 2>(reinterpret_cast<const uint16_t*>(base), halves);
    return data.template bit_cast_view<uint32_t>().read();
}
SYCL_ESIMD_FUNCTION ESIMD_INLINE U bytes(const uint8_t* base, U offsets) {
    return U(cached<uint8_t, lanes>(base, offsets));
}
SYCL_ESIMD_FUNCTION ESIMD_INLINE float half_scalar(const uint8_t* p) {
    return float(sycl::bit_cast<sycl::half>(*reinterpret_cast<const uint16_t*>(p)));
}
SYCL_ESIMD_FUNCTION ESIMD_INLINE U signs(U magnitudes, U bits) {
    U mask = (bits & 1) * 0xffu | ((bits >> 1) & 1) * 0xff00u |
        ((bits >> 2) & 1) * 0xff0000u | ((bits >> 3) & 1) * 0xff000000u;
    // IQ codebooks contain positive nonzero bytes, so +1 cannot carry.
    return (magnitudes ^ mask) + (mask & 0x01010101u);
}
SYCL_ESIMD_FUNCTION ESIMD_INLINE U duplicate_octets(es::simd<uint8_t, 8> values) {
    es::simd<uint8_t, 16> repeated;
    repeated.template select<8, 2>(0) = values;
    repeated.template select<8, 2>(1) = values;
    return U(repeated);
}
template<int Type> constexpr size_t row_bytes(size_t width) {
    if constexpr (Type == 8) return width / 32 * 34;
    if constexpr (Type == 10) return width / 256 * 84;
    if constexpr (Type == 11) return width / 256 * 110;
    if constexpr (Type == 14) return width / 256 * 210;
    if constexpr (Type == 18) return width / 256 * 98;
    if constexpr (Type == 21) return width / 256 * 110;
    if constexpr (Type == 22) return width / 256 * 82;
    if constexpr (Type == 23) return width / 256 * 136;
}
template<int Type> SYCL_ESIMD_FUNCTION ESIMD_INLINE void decode(const uint8_t* w, U col,
    const uint32_t* iq2, const uint32_t* iq3, const int8_t* iq4,
    const uint32_t* iq3xxs, const uint8_t* iq_signs, U& packed, F& scale) {
    if constexpr (Type == 11) {
        const U group = col >> 4;
        const U low_scale = (bytes(w, 96 + (group & 7)) >> ((group >> 3) * 4)) & 15;
        const U high_scale = (bytes(w, 104 + (group & 3)) >> ((group >> 2) * 2)) & 3;
        const U low = (words(w, 32 + (col >> 7) * 32 + (col & 31)) >> (((col & 127) >> 5) * 2)) & 0x03030303u;
        const U high = ((words(w, col & 31) >> (col >> 5)) & 0x01010101u) * 4;
        const U value = low | high;
        packed = ((value | 0x80808080u) - 0x04040404u) ^ 0x80808080u;
        scale = F(I(low_scale | (high_scale << 4)) - 32) * half_scalar(w + 108);
    }
    if constexpr (Type == 14) {
        U segment = col >> 7, part = (col & 127) >> 5, within = col & 31;
        U low = (words(w, segment * 64 + (part & 1) * 32 + within) >> ((part >> 1) * 4)) & 0x0f0f0f0fu;
        U high = (words(w, 128 + segment * 32 + within) >> (part * 2)) & 0x03030303u;
        U value = low | (high << 4);
        packed = ((value | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        scale = F(cached<int8_t, lanes>(reinterpret_cast<const int8_t*>(w), 192 + (col >> 4))) * half_scalar(w + 208);
    }
    if constexpr (Type == 18) {
        const U group = col >> 5, octet = (col & 31) >> 3;
        // The 98-byte block's auxiliary DWORDs start at byte 66. words()
        // reads two aligned uint16 halves, including the final word at 94.
        const U aux = words(w, 66 + group * 4);
        const U code = bytes(w, 2 + group * 8 + ((col & 31) >> 2));
        const U grid = cached<uint32_t, lanes>(iq3xxs, code * 4);
        const U sign = bytes(iq_signs, (aux >> (octet * 7)) & 127);
        packed = signs(grid, (sign >> (col & 7)) & 15);
        scale = (half_scalar(w) * (F(aux >> 28) + .5f)) * .5f;
    }
    if constexpr (Type == 22) {
        U group = col >> 5, octet = (col & 31) >> 3;
        U code = bytes(w, 2 + group * 4 + octet) | (((bytes(w, 66 + group) >> (octet * 2)) & 3) << 8);
        U grid = cached<uint32_t, lanes>(iq2, (code * 2 + ((col & 7) >> 2)) * 4);
        U sign = (bytes(w, 34 + group * 4 + octet) >> (col & 7)) & 15;
        packed = signs(grid, sign);
        U scales = (bytes(w, 74 + group) >> ((col & 16) >> 2)) & 15;
        scale = (F(scales) + .5f) * (half_scalar(w) * .25f);
    }
    if constexpr (Type == 21) {
        U group = col >> 5, nibble = (col & 31) >> 2;
        U code = bytes(w, 2 + group * 8 + nibble) | (((bytes(w, 66 + group) >> nibble) & 1) << 8);
        U grid = cached<uint32_t, lanes>(iq3, code * 4);
        U sign = (bytes(w, 74 + group * 4 + ((col & 31) >> 3)) >> (col & 7)) & 15;
        packed = signs(grid, sign);
        U scales = (bytes(w, 106 + (group >> 1)) >> ((group & 1) * 4)) & 15;
        scale = F(1 + 2 * scales) * half_scalar(w);
    }
    if constexpr (Type == 23) {
        U group = col >> 5;
        U codes = (words(w, 8 + group * 16 + (col & 15)) >> ((col & 16) >> 2)) & 0x0f0f0f0fu;
        es::simd<uint32_t, lanes * 4> indices(codes.template bit_cast_view<uint8_t>().read());
        auto values = cached<int8_t, lanes * 4>(iq4, indices);
        packed = values.template bit_cast_view<uint32_t>().read();
        U low = (bytes(w, 4 + (group >> 1)) >> ((group & 1) * 4)) & 15;
        const uint32_t high = *reinterpret_cast<const uint16_t*>(w + 2);
        I scales = I(low | (((U(high) >> (2 * group)) & 3) << 4)) - 32;
        scale = F(scales) * half_scalar(w);
    }
    if constexpr (Type == 8) {
        U block = col >> 5;
        packed = words(w, block * 34 + 2 + (col & 31));
        auto bits = cached<uint16_t, lanes>(reinterpret_cast<const uint16_t*>(w), block * 34);
        scale = F(bits.template bit_cast_view<sycl::half>().read());
    }
}
template<int Tile> SYCL_ESIMD_FUNCTION ESIMD_INLINE void accumulate(F (&sum)[Tile], U packed, F dw,
    const int32_t* x, const float* dx, size_t base, int width, int tokens, const int (&input_rows)[Tile]) {
#pragma unroll
    for (int token = 0; token < Tile; ++token) if (token < tokens) {
        const size_t input_row = size_t(input_rows[token]);
        I activation = es::block_load<int32_t, lanes>(x + (input_row * width + base) / 4,
            es::properties{es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>});
        // ESIMD takes the accumulator first (unlike CUDA's dp4a order).
        I dot = es::dp4a<int32_t>(I(0), I(packed), activation);
        U columns(0, 4);
        F d = cached<float, lanes>(dx, (columns + uint32_t(input_row * width + base)) / 32 * sizeof(float));
        sum[token] += F(dot) * dw * d;
    }
}
template<int Tile> SYCL_ESIMD_FUNCTION ESIMD_INLINE void accumulate_affine(F (&sum)[Tile], U packed, F dw, F minimum,
    const int32_t* x, const float* dx, size_t base, int width, int tokens, const int (&input_rows)[Tile]) {
#pragma unroll
    for (int token = 0; token < Tile; ++token) if (token < tokens) {
        const size_t input_row = size_t(input_rows[token]);
        I activation = es::block_load<int32_t, lanes>(x + (input_row * width + base) / 4,
            es::properties{es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>});
        I dot = es::dp4a<int32_t>(I(0), I(packed), activation);
        I total = es::dp4a<int32_t>(I(0), I(0x01010101), activation);
        U columns(0, 4);
        F d = cached<float, lanes>(dx, (columns + uint32_t(input_row * width + base)) / 32 * sizeof(float));
        sum[token] += (F(dot) * dw - F(total) * minimum) * d;
    }
}
template<int Type, int Tile, bool Grouped = false> sycl::event product(sycl::queue& q, const uint8_t* weights, const int8_t* input,
    const float* scales, const int32_t* ids, int picks, int tokens, float* output, int rows, int width, bool separate,
    const uint32_t* iq2, const uint32_t* iq3, const int8_t* iq4,
    const uint32_t* iq3xxs, const uint8_t* iq_signs, ExpertPlan plan = {}) {
    const int bound = Grouped ? std::min(picks * tokens, (picks * tokens + Tile - 1) / Tile + plan.experts) : (ids ? picks * tokens : (tokens + Tile - 1) / Tile);
    const size_t stride = row_bytes<Type>(size_t(width)), entries = size_t(rows) * bound;
    return q.parallel_for(sycl::nd_range<1>(((entries + 15) / 16) * 16, 16),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const size_t entry = it.get_global_linear_id();
        if (entry >= entries) return;
        const size_t row = entry % rows, pick = entry / rows;
        if constexpr (Grouped) if (pick >= size_t(plan.tiles[0])) return;
        const size_t expert = Grouped ? size_t(plan.owners[pick]) : ids ? size_t(ids[pick]) : 0;
        const uint8_t* w = weights + (expert * rows + row) * stride;
        const int32_t* x = reinterpret_cast<const int32_t*>(input);
        const float* dx = scales;
        F sum[Tile];
        int input_rows[Tile], output_slots[Tile];
        const int first_token = !Grouped && !ids ? int(pick) * Tile : 0;
#pragma unroll
        for (int t = 0; t < Tile; ++t) { sum[t] = 0.f; input_rows[t] = first_token + t; output_slots[t] = 0; }
        const int remaining = tokens - first_token;
        int tile_tokens = ids ? 1 : remaining < Tile ? remaining : Tile;
        if constexpr (Grouped) {
            const int first = plan.offsets[pick];
            const int count = plan.counts[expert] - first;
            tile_tokens = count < Tile ? count : Tile;
#pragma unroll
            for (int t = 0; t < Tile; ++t) if (t < tile_tokens) {
                const int slot = plan.slots[expert * plan.tokens + first + t];
                output_slots[t] = slot; input_rows[t] = separate ? slot : slot / picks;
            }
        } else if (ids) input_rows[0] = int(separate ? pick : pick / picks);
        U columns(0, 4);
        if constexpr (Type == 10 || Type == 11) {
            for (size_t base = 0; base < size_t(width); base += 64) {
                const uint8_t* block = w + (base / 256) * (Type == 10 ? 84 : 110);
                U col = columns + uint32_t(base % 256);
                if constexpr (Type == 10) {
                    U packed = (words(block, 16 + (col >> 7) * 32 + (col & 31)) >> (((col & 127) >> 5) * 2)) & 0x03030303u;
                    U scales = bytes(block, col >> 4);
                    F dw = F(scales & 15) * half_scalar(block + 80);
                    F minimum = F(scales >> 4) * half_scalar(block + 82);
                    accumulate_affine<Tile>(sum, packed, dw, minimum, x, dx, base, width, tile_tokens, input_rows);
                } else {
                    U packed; F dw;
                    decode<11>(block, col, iq2, iq3, iq4, iq3xxs, iq_signs, packed, dw);
                    accumulate<Tile>(sum, packed, dw, x, dx, base, width, tile_tokens, input_rows);
                }
            }
        } else if constexpr (Type == 14) {
            // One Q6_K block covers four 64-coefficient dot products. Load its
            // low/high bit planes once, then reuse the words from registers.
            es::simd<uint32_t, 64> low_offsets(0, 2);
            es::simd<uint32_t, 32> high_offsets(128, 2);
            es::simd<uint32_t, 16> scale_offsets(192, 1);
            es::simd<uint16_t, 16> scale_indices(0, 1);
            scale_indices /= 4;
            for (size_t base = 0; base < size_t(width); base += 256) {
                const uint8_t* block = w + (base / 256) * 210;
                auto low_halves = cached<uint16_t, 64>(reinterpret_cast<const uint16_t*>(block), low_offsets);
                auto high_halves = cached<uint16_t, 32>(reinterpret_cast<const uint16_t*>(block), high_offsets);
                auto low_words = low_halves.template bit_cast_view<uint32_t>().read();
                auto high_words = high_halves.template bit_cast_view<uint32_t>().read();
                auto block_scales = cached<int8_t, 16>(reinterpret_cast<const int8_t*>(block), scale_offsets);
                const float delta = half_scalar(block + 208);
                for (int part = 0; part < 4; ++part) {
                    U low = (low_words.template select<16, 1>((part / 2) * 16).read() >> ((part & 1) * 4)) & 0x0f0f0f0fu;
                    U high;
                    high.template select<8, 1>(0) = high_words.template select<8, 1>((part / 2) * 8);
                    high.template select<8, 1>(8) = high_words.template select<8, 1>((part / 2) * 8);
                    U shift = U(0, 1) / 8 * 2 + uint32_t((part & 1) * 4);
                    U value = low | (((high >> shift) & 0x03030303u) << 4);
                    U packed = ((value | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
                    F dw = F(block_scales.iselect<16>(scale_indices + uint16_t(part * 4))) * delta;
                    accumulate<Tile>(sum, packed, dw, x, dx, base + part * 64, width, tile_tokens, input_rows);
                }
            }
        } else if constexpr (Type == 21 || Type == 22) {
            // Direct constant regions avoid the indirect register lookups of
            // the rejected metadata-prefetch experiment. Each code/sign byte
            // supplies two adjacent four-coefficient dot products.
            U lane(0, 1);
            es::simd<uint32_t, 32> offsets(2, 2);
            for (size_t base = 0; base < size_t(width); base += 256) {
                const uint8_t* block = w + (base / 256) * row_bytes<Type>(256);
                auto halves = cached<uint16_t, 32>(reinterpret_cast<const uint16_t*>(block), offsets);
                auto codes = halves.template bit_cast_view<uint8_t>().read();
                es::simd<uint8_t, 32> sign_bytes;
                es::simd<uint16_t, 8> meta;
                es::simd<uint16_t, 4> high;
                es::simd<uint8_t, 4> scales;
                if constexpr (Type == 22) {
                    meta = cached<uint16_t, 8>(reinterpret_cast<const uint16_t*>(block), es::simd<uint32_t, 8>(66, 2));
                } else {
                    auto sign_halves = cached<uint16_t, 16>(reinterpret_cast<const uint16_t*>(block), es::simd<uint32_t, 16>(74, 2));
                    sign_bytes = sign_halves.template bit_cast_view<uint8_t>();
                    high = cached<uint16_t, 4>(reinterpret_cast<const uint16_t*>(block), es::simd<uint32_t, 4>(66, 2));
                    auto scale_halves = cached<uint16_t, 2>(reinterpret_cast<const uint16_t*>(block), es::simd<uint32_t, 2>(106, 2));
                    scales = scale_halves.template bit_cast_view<uint8_t>();
                }
                const float delta = half_scalar(block);
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    U packed;
                    F dw;
                    if constexpr (Type == 22) {
                        U code = duplicate_octets(codes.template select<8, 1>(part * 8).read());
                        const uint32_t high_word = meta[part], scale_word = meta[4 + part];
                        code |= ((U(high_word) >> ((lane / 8) * 8 + ((lane & 7) / 2) * 2)) & 3) << 8;
                        U grid = cached<uint32_t, 16>(iq2, (code * 2 + (lane & 1)) * 4);
                        U sign = (duplicate_octets(codes.template select<8, 1>(32 + part * 8).read()) >> ((lane & 1) * 4)) & 15;
                        packed = signs(grid, sign);
                        U scale = (U(scale_word) >> ((lane / 8) * 8 + ((lane & 7) / 4) * 4)) & 15;
                        dw = (F(scale) + .5f) * (delta * .25f);
                    } else {
                        const uint32_t high_word = high[part], scale_byte = scales[part];
                        U code = U(codes.template select<16, 1>(part * 16).read());
                        code |= ((U(high_word) >> ((lane / 8) * 8 + (lane & 7))) & 1) << 8;
                        U grid = cached<uint32_t, 16>(iq3, code * 4);
                        U sign = (duplicate_octets(sign_bytes.template select<8, 1>(part * 8).read()) >> ((lane & 1) * 4)) & 15;
                        packed = signs(grid, sign);
                        U scale = (U(scale_byte) >> ((lane / 8) * 4)) & 15;
                        dw = F(1 + 2 * scale) * delta;
                    }
                    accumulate<Tile>(sum, packed, dw, x, dx, base + part * 64, width, tile_tokens, input_rows);
                }
            }
        } else {
        for (size_t base = 0; base < size_t(width); base += 64) {
            U col = columns + uint32_t(base % 256), packed;
            F dw;
            decode<Type>(w + (base / 256) * row_bytes<Type>(256), col, iq2, iq3, iq4, iq3xxs, iq_signs, packed, dw);
            accumulate<Tile>(sum, packed, dw, x, dx, base, width, tile_tokens, input_rows);
        }
        }
#pragma unroll
        for (int t = 0; t < Tile; ++t) if (t < tile_tokens)
            output[Grouped ? size_t(output_slots[t]) * rows + row : ids ? entry : size_t(first_token + t) * rows + row] = es::reduce<float>(sum[t], std::plus<>());
    });
}
}
sycl::event quantize_decode_input(sycl::queue& q, const float* input, int8_t* values,
    float* scales, int width, int rows, int32_t* error) {
    if (!q.has_property<sycl::property::queue::in_order>() || width < 1 || width % 32 || rows < 1 || rows > 8192)
        throw std::invalid_argument("Invalid Q8 decode activation geometry");
    return q.parallel_for(sycl::nd_range<1>(size_t(rows) * width, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t index = it.get_global_linear_id(), block = index / 32;
        const float value = input[index];
        const bool bad = sycl::any_of_group(it.get_sub_group(), !sycl::isfinite(value));
        const float maximum = sycl::reduce_over_group(it.get_sub_group(), sycl::fabs(value), sycl::maximum<float>());
        const float scale = bad ? 0.f : maximum / 127.f;
        const float code = !bad && maximum > 0 ? sycl::rint(value / maximum * 127.f) : 0.f;
        values[index] = int8_t(sycl::clamp(code, -127.f, 127.f));
        if (it.get_local_linear_id() == 0) {
            scales[block] = scale;
            if (error && bad) sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space>(error[0]).store(1);
        }
    });
}
struct DecodeDp4a::Impl {
    sycl::queue& q;
    bool xmx;
    int8_t *input = nullptr, *iq4 = nullptr;
    float *scales = nullptr, *dense_partials = nullptr;
    int32_t* error = nullptr;
    uint32_t *iq2 = nullptr, *iq3 = nullptr, *iq3xxs = nullptr;
    uint8_t* iq_signs = nullptr;
    sycl::event quantization;
    explicit Impl(sycl::queue& queue, bool windows, bool dense_split_k) : q(queue), xmx(windows) {
#ifndef TINYSTRATA_INTEL_INT8_XMX
        if (xmx || dense_split_k) throw std::runtime_error("INT8 XMX requires TINYSTRATA_INTEL_INT8_XMX=ON");
#endif
        try {
            input = sycl::malloc_device<int8_t>(512 * 8192, q); scales = sycl::malloc_device<float>(512 * 8192 / 32, q);
            error = sycl::malloc_shared<int32_t>(1, q);
            if (dense_split_k) dense_partials = sycl::malloc_device<float>(4 * 8 * 8192, q);
            iq2 = sycl::malloc_device<uint32_t>(2048, q); iq3 = sycl::malloc_device<uint32_t>(512, q); iq4 = sycl::malloc_device<int8_t>(16, q);
            iq3xxs = sycl::malloc_device<uint32_t>(256, q); iq_signs = sycl::malloc_device<uint8_t>(128, q);
            if (!input || !scales || !error || !iq2 || !iq3 || !iq4 || !iq3xxs || !iq_signs || (dense_split_k && !dense_partials)) throw std::runtime_error("Intel DP4A scratch allocation failed");
            q.memset(error, 0, sizeof(int32_t)); q.memcpy(iq2, iq2s_grid, sizeof(iq2s_grid));
            q.memcpy(iq3, iq3s_grid, sizeof(iq3s_grid)); q.memcpy(iq4, kvalues_iq4nl, sizeof(kvalues_iq4nl));
            q.memcpy(iq3xxs, iq3xxs_grid, sizeof(iq3xxs_grid)); q.memcpy(iq_signs, ksigns_iq2xs, sizeof(ksigns_iq2xs));
        } catch (...) { release(); throw; }
        std::fprintf(stderr, "Intel decode_dp4a=ESIMD Q8_activations=1 unchanged_GGUF_weights=1\n");
        if (xmx) std::fprintf(stderr, "Intel verify_xmx=INT8_DPAS unchanged_GGUF_weights=1\n");
    }
    void release() noexcept {
        try { q.wait_and_throw(); } catch (...) {}
        for (void* p : {static_cast<void*>(input), static_cast<void*>(scales), static_cast<void*>(dense_partials), static_cast<void*>(error), static_cast<void*>(iq2), static_cast<void*>(iq3), static_cast<void*>(iq4), static_cast<void*>(iq3xxs), static_cast<void*>(iq_signs)})
            if (p) try { sycl::free(p, q); } catch (...) {}
    }
    ~Impl() { release(); }
};
DecodeDp4a::DecodeDp4a(sycl::queue& q, bool windows, bool dense_split_k) : impl_(std::make_unique<Impl>(q, windows, dense_split_k)) {}
DecodeDp4a::~DecodeDp4a() = default;
bool DecodeDp4a::valid() const noexcept { return impl_->error[0] == 0; }
sycl::event DecodeDp4a::quantization_event() const { return impl_->quantization; }
sycl::event DecodeDp4a::dense(int type, const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens) {
    return selected(type, weights, nullptr, 1, input, output, rows, width, false, tokens);
}
sycl::event DecodeDp4a::dense_interleaved_q6(const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens, bool split_k, sycl::event* compute_stage) {
#ifdef TINYSTRATA_INTEL_INT8_XMX
    if (rows < 1 || rows % 16 || width < 256 || width > 8192 || width % 256 || tokens < 1 || tokens > 8)
        throw std::invalid_argument("Invalid interleaved Q6 matrix shape");
    auto& p = *impl_;
    if (split_k && (!p.dense_partials || rows > 8192)) throw std::invalid_argument("Split-K Q6 exceeds configured bounded scratch");
    p.quantization = quantize_decode_input(p.q, input, p.input, p.scales, width, tokens, p.error);
    if (split_k) return q6_interleaved_splitk_product(p.q, weights, p.input, p.scales, output, rows, width, tokens, p.dense_partials, compute_stage);
    return q6_interleaved_product(p.q, weights, p.input, p.scales, output, rows, width, tokens);
#else
    throw std::runtime_error("Interleaved Q6 XMX requires TINYSTRATA_INTEL_INT8_XMX=ON");
#endif
}
sycl::event DecodeDp4a::selected_vocabulary_q6(const uint8_t* weights, const float* input, float* output,
    int vocabulary, int width, const int32_t* ids, int count) {
#ifdef TINYSTRATA_INTEL_INT8_XMX
    if (vocabulary < 16 || vocabulary % 16 || width != 2048 || count < 1 || count > 4096 || count > vocabulary || !ids)
        throw std::invalid_argument("Invalid selected draft vocabulary geometry");
    auto& p = *impl_;
    p.quantization = quantize_decode_input(p.q, input, p.input, p.scales, width, 1, p.error);
    return q6_interleaved_selected_product(p.q, weights, p.input, p.scales, ids, output, vocabulary, width, count);
#else
    throw std::runtime_error("Selected draft vocabulary requires TINYSTRATA_INTEL_INT8_XMX=ON");
#endif
}
sycl::event DecodeDp4a::selected(int type, const uint8_t* weights, const int32_t* ids, int picks,
    const float* input, float* output, int rows, int width, bool separate, int tokens) {
    auto& p = *impl_;
    if (rows < 1 || width < 64 || width > 8192 || width % 64 || (type != 8 && width % 256) || picks < 1 || picks > 8 || tokens < 1 || tokens > 8 || (!ids && (picks != 1 || separate)))
        throw std::invalid_argument("Invalid DP4A matrix shape");
    p.quantization = quantize_decode_input(p.q, input, p.input, p.scales, width, separate ? picks * tokens : tokens, p.error);
#define PRODUCT(T,Tile) product<T,Tile>(p.q, weights, p.input, p.scales, ids, picks, tokens, output, rows, width, separate, p.iq2, p.iq3, p.iq4, p.iq3xxs, p.iq_signs)
#define RUN(T) ((!ids && tokens > 4) ? PRODUCT(T,8) : (!ids && tokens > 1) ? PRODUCT(T,4) : PRODUCT(T,1))
    switch (type) { case 8: return RUN(8); case 10: return RUN(10); case 11: return RUN(11); case 14: return RUN(14); case 18: return RUN(18); case 21: return RUN(21); case 22: return RUN(22); case 23: return RUN(23);
        default: throw std::invalid_argument("Unsupported DP4A weight type"); }
#undef RUN
#undef PRODUCT
}
sycl::event DecodeDp4a::grouped(int type, const uint8_t* weights, ExpertPlan plan,
    const float* input, float* output, int rows, int width, bool separate, bool interleaved) {
    auto& p = *impl_;
    bool xmx_shape = false;
#ifdef TINYSTRATA_INTEL_INT8_XMX
    xmx_shape = p.xmx && q8_xmx_supports(type);
#endif
    if (rows < 1 || width < 64 || width > 8192 || width % 64 || (type != 8 && width % 256) ||
        plan.picks < 1 || plan.picks > 8 || plan.tokens < 1 || plan.tokens > 1024 || plan.experts < 1 || plan.experts > 256 ||
        (plan.tile_width != 2 && plan.tile_width != 4 && !(plan.tile_width == 1 && xmx_shape)) || (interleaved && (!xmx_shape || rows % 16 || plan.tile_width == 2)) ||
        size_t(plan.tokens) * (separate ? plan.picks : 1) * width > 512 * 8192)
        throw std::invalid_argument("Invalid grouped DP4A matrix shape");
    p.quantization = quantize_decode_input(p.q, input, p.input, p.scales, width, separate ? plan.picks * plan.tokens : plan.tokens, p.error);
#ifdef TINYSTRATA_INTEL_INT8_XMX
    if (xmx_shape && (plan.tile_width == 1 || plan.tile_width == 4))
        return q8_xmx_product(p.q, type, weights, p.input, p.scales, output, rows, width, plan, separate, p.iq2, p.iq3, interleaved, p.iq3xxs, p.iq_signs);
#endif
#define GROUP_PRODUCT(T,Tile) product<T,Tile,true>(p.q, weights, p.input, p.scales, nullptr, plan.picks, plan.tokens, output, rows, width, separate, p.iq2, p.iq3, p.iq4, p.iq3xxs, p.iq_signs, plan)
#define RUN(T) (plan.tile_width == 2 ? GROUP_PRODUCT(T,2) : GROUP_PRODUCT(T,4))
    switch (type) { case 8: return RUN(8); case 10: return RUN(10); case 11: return RUN(11); case 14: return RUN(14); case 18: return RUN(18); case 21: return RUN(21); case 22: return RUN(22); case 23: return RUN(23);
        default: throw std::invalid_argument("Unsupported grouped DP4A weight type"); }
#undef RUN
#undef GROUP_PRODUCT
}
} // namespace strata::intel
