// Qwen3.6 operation semantics follow TinyStrata's q36_engine.cu, originally
// adapted from Niko1221/Strata (MIT; see LICENSE and NOTICE).
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/quant.hpp"
#include <stdexcept>

namespace strata::intel {
namespace {
void require_in_order(const sycl::queue& queue) {
    if (!queue.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("Qwen SYCL operations require an in-order queue");
}
template<int Type> constexpr size_t row_bytes(size_t width) {
    if constexpr (Type == 0) return width * 4;
    if constexpr (Type == 1 || Type == 30) return width * 2;
    if constexpr (Type == 8) return width / 32 * 34;
    if constexpr (Type == 14) return width / 256 * 210;
    if constexpr (Type == 10) return width / 256 * 84;
    if constexpr (Type == 11 || Type == 21) return width / 256 * 110;
    if constexpr (Type == 22) return width / 256 * 82;
    if constexpr (Type == 23) return width / 256 * 136;
}
// One 16-lane subgroup owns a row. Each lane decodes eight adjacent
// elements, allowing the compiler to reuse IQ codebook entries and scales.
// Independent accumulators avoid a long FMA dependency chain.
template<int Type, int Tile> sycl::event packed_product(sycl::queue& q, const uint8_t* w, const float* input, float* output,
    int rows, int width, int tokens, const int32_t* ids = nullptr, int picks = 1, bool separate_inputs = false) {
    constexpr size_t group = 128, subgroup = 16, rows_per_group = group / subgroup, pack = 8;
    const size_t stride = row_bytes<Type>(size_t(width));
    const size_t entries = ids ? size_t(tokens) * picks * rows : size_t((tokens + Tile - 1) / Tile) * rows;
    return q.parallel_for(sycl::nd_range<1>(((entries + rows_per_group - 1) / rows_per_group) * group, group),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg = it.get_sub_group();
        const size_t entry = it.get_group_linear_id() * rows_per_group + sg.get_group_linear_id();
        if (entry >= entries) return; // Uniform within this subgroup.
        const size_t row = entry % rows, lane = sg.get_local_linear_id(), pick = entry / rows;
        const size_t first = ids ? pick / picks : pick * Tile;
        const uint8_t* weight = w + (ids ? size_t(ids[pick]) * rows + row : row) * stride;
        float acc[Tile][pack] = {};
        for (size_t base = lane * pack; base < size_t(width); base += subgroup * pack) {
#pragma unroll
            for (size_t k = 0; k < pack; ++k) {
                const float value = base + k < size_t(width) ? quant::value<Type>(weight, base + k) : 0.f;
#pragma unroll
                for (size_t t = 0; t < Tile; ++t) {
                    if (first + t < size_t(tokens) && base + k < size_t(width)) {
                        const size_t offset = ids && separate_inputs ? pick * width : (first + t) * width;
                        acc[t][k] = sycl::fma(value, input[offset + base + k], acc[t][k]);
                    }
                }
            }
        }
#pragma unroll
        for (size_t t = 0; t < Tile; ++t) {
            float sum = acc[t][0];
#pragma unroll
            for (size_t k = 1; k < pack; ++k) sum += acc[t][k];
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            if (lane == 0 && first + t < size_t(tokens)) output[ids ? entry : (first + t) * rows + row] = sum;
        }
    });
}
template<int Type> sycl::event matvec(sycl::queue& q, const uint8_t* w, const float* x, float* y, int rows, int width) {
    return packed_product<Type, 1>(q, w, x, y, rows, width, 1);
}
template<int Type> sycl::event matmul(sycl::queue& q, const uint8_t* w, const float* x, float* y, int rows, int width, int tokens) {
    return packed_product<Type, 4>(q, w, x, y, rows, width, tokens);
}
template<int Type> sycl::event unpack(sycl::queue& q, const uint8_t* w, float* output, size_t elements) {
    return q.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> index) { output[index[0]] = quant::value<Type>(w, index[0]); });
}
template<int Type> sycl::event unpack_half(sycl::queue& q, const uint8_t* w, sycl::half* output, size_t elements) {
    return q.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> index) { output[index] = sycl::half(quant::value<Type>(w, index[0])); });
}
template<int Type> sycl::event embedding(sycl::queue& q, const uint8_t* w, const int32_t* ids, float* output, int width, int tokens) {
    const size_t stride = row_bytes<Type>(size_t(width));
    return q.parallel_for(sycl::range<1>(size_t(tokens) * width), [=](sycl::id<1> id) {
        const size_t token = id[0] / width, col = id[0] % width;
        output[id] = quant::value<Type>(w + size_t(ids[token]) * stride, col);
    });
}
template<int Type> sycl::event selected(sycl::queue& q, const uint8_t* w, const int32_t* ids,
    int picks, const float* input, float* output, int rows, int width, bool separate_inputs, int tokens) {
    return packed_product<Type, 1>(q, w, input, output, rows, width, tokens, ids, picks, separate_inputs);
}
void validate_quantized_shape(int type, size_t width) {
    const size_t block = type == 0 || type == 1 || type == 30 ? 1 : type == 8 ? 32 : 256;
    if (!width || width % block) throw std::invalid_argument("Invalid quantized row width");
}
template<int Type> sycl::event grouped_product(sycl::queue& q, const uint8_t* w, ExpertPlan plan,
    const float* input, float* output, int rows, int width, bool separate_inputs) {
    constexpr size_t group = 128, subgroup = 16, rows_per_group = group / subgroup, pack = 8, tile = 4;
    const size_t stride = row_bytes<Type>(size_t(width));
    const size_t capacity = (size_t(plan.tokens) * plan.picks + tile - 1) / tile + plan.experts;
    return q.parallel_for(sycl::nd_range<1>(capacity * ((rows + rows_per_group - 1) / rows_per_group) * group, group),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg = it.get_sub_group();
        const size_t row_groups = (rows + rows_per_group - 1) / rows_per_group;
        const size_t index = it.get_group_linear_id() / row_groups;
        const size_t row = (it.get_group_linear_id() % row_groups) * rows_per_group + sg.get_group_linear_id();
        if (index >= size_t(plan.tiles[0]) || row >= size_t(rows)) return;
        const size_t expert = plan.owners[index], first = plan.offsets[index], lane = sg.get_local_linear_id();
        const uint8_t* weight = w + (expert * rows + row) * stride;
        int entries[tile];
#pragma unroll
        for (size_t t = 0; t < tile; ++t)
            entries[t] = first + t < size_t(plan.counts[expert]) ? plan.slots[expert * plan.tokens + first + t] : -1;
        float acc[tile][pack] = {};
        for (size_t base = lane * pack; base < size_t(width); base += subgroup * pack) {
#pragma unroll
            for (size_t k = 0; k < pack; ++k) {
                const float value = base + k < size_t(width) ? quant::value<Type>(weight, base + k) : 0.f;
#pragma unroll
                for (size_t t = 0; t < tile; ++t) if (entries[t] >= 0 && base + k < size_t(width)) {
                    const size_t source = separate_inputs ? size_t(entries[t]) : size_t(entries[t]) / plan.picks;
                    acc[t][k] = sycl::fma(value, input[source * width + base + k], acc[t][k]);
                }
            }
        }
#pragma unroll
        for (size_t t = 0; t < tile; ++t) {
            float sum = acc[t][0];
#pragma unroll
            for (size_t k = 1; k < pack; ++k) sum += acc[t][k];
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            if (lane == 0 && entries[t] >= 0) output[size_t(entries[t]) * rows + row] = sum;
        }
    });
}
}
#define TINYSTRATA_INTEL_DISPATCH(call) \
    switch (type) { \
        case 0: return call(0); case 1: return call(1); case 30: return call(30); \
        case 8: return call(8); case 10: return call(10); case 11: return call(11); \
        case 14: return call(14); case 21: return call(21); case 22: return call(22); case 23: return call(23); \
        default: throw std::invalid_argument("GGUF type is not implemented by the initial Intel kernels"); }
sycl::event quantized_matvec(sycl::queue& q, int type, const uint8_t* w, const float* x, float* y, int rows, int width) {
    require_in_order(q);
    if (rows <= 0 || width <= 0) throw std::invalid_argument("Invalid matrix shape");
    validate_quantized_shape(type, size_t(width));
#define RUN(T) matvec<T>(q, w, x, y, rows, width)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event quantized_matmul(sycl::queue& q, int type, const uint8_t* w, const float* x, float* y, int rows, int width, int tokens) {
    if (tokens == 1) return quantized_matvec(q, type, w, x, y, rows, width);
    require_in_order(q);
    if (rows <= 0 || width <= 0 || tokens <= 0 || tokens > 1024) throw std::invalid_argument("Invalid batched matrix shape");
    validate_quantized_shape(type, size_t(width));
#define RUN(T) matmul<T>(q, w, x, y, rows, width, tokens)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event dequantize(sycl::queue& q, int type, const uint8_t* w, float* output, size_t elements) {
    require_in_order(q);
    validate_quantized_shape(type, elements);
#define RUN(T) unpack<T>(q, w, output, elements)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event selected_matvec(sycl::queue& q, int type, const uint8_t* w, const int32_t* ids,
    int picks, const float* input, float* output, int rows, int width, bool separate_inputs, int tokens) {
    require_in_order(q);
    if (rows <= 0 || width <= 0 || picks <= 0 || picks > 512 || tokens <= 0 || tokens > 1024) throw std::invalid_argument("Invalid selected-expert shape");
    validate_quantized_shape(type, size_t(width));
#define RUN(T) selected<T>(q, w, ids, picks, input, output, rows, width, separate_inputs, tokens)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event plan_experts(sycl::queue& q, const int32_t* ids, ExpertPlan plan) {
    require_in_order(q);
    if (plan.experts < 1 || plan.tokens < 1 || plan.picks < 1 || plan.picks > plan.experts || (plan.tile_width != 1 && plan.tile_width != 2 && plan.tile_width != 4 && plan.tile_width != 8))
        throw std::invalid_argument("Invalid expert plan geometry");
    // Each expert owns its list, preserving the token/pick order. IDs come from
    // top-k without duplicates per token, hence at most tokens entries/expert.
    q.parallel_for(sycl::range<1>(size_t(plan.experts)), [=](sycl::id<1> id) {
        const size_t expert = id[0];
        int count = 0;
        for (int entry = 0; entry < plan.tokens * plan.picks; ++entry)
            if (ids[entry] == int(expert)) plan.slots[expert * plan.tokens + count++] = entry;
        plan.counts[expert] = count;
    });
    return q.single_task([=] {
        int tile = 0;
        for (int expert = 0; expert < plan.experts; ++expert) for (int offset = 0; offset < plan.counts[expert]; offset += plan.tile_width) {
            plan.owners[tile] = expert;
            plan.offsets[tile++] = offset;
        }
        plan.tiles[0] = tile;
    });
}
sycl::event selected_matmul(sycl::queue& q, int type, const uint8_t* w, ExpertPlan plan,
    const float* input, float* output, int rows, int width, bool separate_inputs) {
    require_in_order(q);
    if (rows <= 0 || width <= 0 || plan.tokens <= 0 || plan.tokens > 1024 || plan.picks < 1 || plan.tile_width != 4)
        throw std::invalid_argument("Invalid grouped expert shape");
    validate_quantized_shape(type, size_t(width));
#define RUN(T) grouped_product<T>(q, w, plan, input, output, rows, width, separate_inputs)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event embedding_lookup(sycl::queue& q, int type, const uint8_t* w, const int32_t* ids, float* output, int width, int tokens) {
    require_in_order(q);
    if (width <= 0 || tokens <= 0 || tokens > 1024) throw std::invalid_argument("Invalid embedding shape");
    validate_quantized_shape(type, size_t(width));
#define RUN(T) embedding<T>(q, w, ids, output, width, tokens)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event dequantize_half(sycl::queue& q, int type, const uint8_t* w, sycl::half* output, size_t elements) {
    require_in_order(q);
    validate_quantized_shape(type, elements);
#define RUN(T) unpack_half<T>(q, w, output, elements)
    TINYSTRATA_INTEL_DISPATCH(RUN)
#undef RUN
}
sycl::event convert_half(sycl::queue& q, const float* input, sycl::half* output, size_t elements) {
    require_in_order(q);
    return q.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) { output[id] = sycl::half(input[id]); });
}
#undef TINYSTRATA_INTEL_DISPATCH
sycl::event rms_norm(sycl::queue& q, const float* x, const float* w, float* y,
                     int rows, int width, float epsilon) {
    require_in_order(q);
    if (rows <= 0 || width <= 0 || !(epsilon > 0)) throw std::invalid_argument("Invalid RMS dimensions/epsilon");
    constexpr size_t group = 128;
    return q.parallel_for(sycl::nd_range<1>(size_t(rows) * group, group), [=](sycl::nd_item<1> it) {
        const size_t row = it.get_group_linear_id(), lane = it.get_local_linear_id();
        float sum = 0;
        for (size_t col = lane; col < size_t(width); col += group) {
            const float value = x[row * width + col];
            sum = sycl::fma(value, value, sum);
        }
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float scale = sycl::rsqrt(sum / float(width) + epsilon);
        for (size_t col = lane; col < size_t(width); col += group)
            y[row * width + col] = x[row * width + col] * scale * w[col];
    });
}
sycl::event q8_matvec(sycl::queue& q, const Q8Block* w, const float* x, float* y, int rows, int width) {
    require_in_order(q);
    if (rows <= 0 || width <= 0 || width % 32) throw std::invalid_argument("Invalid Q8_0 matrix dimensions");
    constexpr size_t group = 128;
    return q.parallel_for(sycl::nd_range<1>(size_t(rows) * group, group), [=](sycl::nd_item<1> it) {
        const size_t row = it.get_group_linear_id(), lane = it.get_local_linear_id();
        float sum = 0;
        for (size_t col = lane; col < size_t(width); col += group) {
            const Q8Block& block = w[row * (width / 32) + col / 32];
            sum = sycl::fma(float(block.scale) * float(block.values[col % 32]), x[col], sum);
        }
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        if (lane == 0) y[row] = sum;
    });
}
sycl::event gdn_recurrence(sycl::queue& q, float* state, const float* qkv, const float* gate,
                          const float* beta, float* output, int hk, int hv, int tokens) {
    require_in_order(q);
    if (hk <= 0 || hv <= 0 || hv % hk || tokens <= 0) throw std::invalid_argument("Invalid GDN geometry");
    // One work item owns a complete state column. No host polling, fixed warp
    // size, cross-work-group synchronization, or duplicate recurrent store.
    constexpr size_t width = 128;
    const size_t channels = (2 * size_t(hk) + hv) * width;
    return q.parallel_for(sycl::range<1>(size_t(hv) * width), [=](sycl::id<1> id) {
        const size_t head = id[0] / width, col = id[0] % width, qhead = head % hk;
        float column[128];
        for (size_t row = 0; row < width; ++row) column[row] = state[(row * hv + head) * width + col];
        for (int token = 0; token < tokens; ++token) {
            const float* input = qkv + size_t(token) * channels;
            const float* query = input + qhead * width;
            const float* key = input + (size_t(hk) + qhead) * width;
            const float value = input[2 * size_t(hk) * width + head * width + col];
            const float decay = sycl::exp(gate[size_t(token) * hv + head]);
            float prediction = 0;
            for (size_t row = 0; row < width; ++row) prediction = sycl::fma(column[row], key[row], prediction);
            const float delta = (value - decay * prediction) * beta[size_t(token) * hv + head];
            float result = 0;
            for (size_t row = 0; row < width; ++row) {
                column[row] = sycl::fma(decay, column[row], key[row] * delta);
                result = sycl::fma(column[row], query[row], result);
            }
            output[(size_t(token) * hv + head) * width + col] = result * sycl::rsqrt(float(width));
        }
        for (size_t row = 0; row < width; ++row) state[(row * hv + head) * width + col] = column[row];
    });
}
sycl::event gdn_recurrence_tiled(sycl::queue& q, float* state, const float* qkv, const float* gate,
                                const float* beta, float* output, int hk, int hv, int tokens) {
    require_in_order(q);
    if (hk <= 0 || hv <= 0 || hv % hk || tokens <= 0) throw std::invalid_argument("Invalid GDN geometry");
    // Four row partitions cooperate on 32 contiguous value columns. Keeping
    // 32 rather than 128 state elements per work item reduces register spills.
    constexpr size_t width = 128, columns = 32, partitions = 4, rows = width / partitions;
    const size_t channels = (2 * size_t(hk) + hv) * width;
    return q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> partial(columns * partitions, handler);
        handler.parallel_for(sycl::nd_range<1>(size_t(hv) * width * partitions, columns * partitions), [=](sycl::nd_item<1> it) {
            const size_t group = it.get_group_linear_id(), head = group / (width / columns);
            const size_t lane = it.get_local_linear_id(), col_lane = lane % columns, partition = lane / columns;
            const size_t col = (group % (width / columns)) * columns + col_lane, first_row = partition * rows, qhead = head % hk;
            float column[rows];
            for (size_t r = 0; r < rows; ++r) column[r] = state[((first_row + r) * hv + head) * width + col];
            for (int token = 0; token < tokens; ++token) {
                const float* input = qkv + size_t(token) * channels;
                const float* query = input + qhead * width;
                const float* key = input + (size_t(hk) + qhead) * width;
                const float value = input[2 * size_t(hk) * width + head * width + col];
                const float decay = sycl::exp(gate[size_t(token) * hv + head]);
                float prediction = 0;
                for (size_t r = 0; r < rows; ++r) prediction = sycl::fma(column[r], key[first_row + r], prediction);
                partial[lane] = prediction;
                sycl::group_barrier(it.get_group());
                prediction = partial[col_lane];
                for (size_t p = 1; p < partitions; ++p) prediction += partial[p * columns + col_lane];
                const float delta = (value - decay * prediction) * beta[size_t(token) * hv + head];
                float result = 0;
                for (size_t r = 0; r < rows; ++r) {
                    column[r] = sycl::fma(decay, column[r], key[first_row + r] * delta);
                    result = sycl::fma(column[r], query[first_row + r], result);
                }
                // Every partition must finish reading the previous partials.
                sycl::group_barrier(it.get_group());
                partial[lane] = result;
                sycl::group_barrier(it.get_group());
                if (partition == 0) {
                    result = partial[col_lane];
                    for (size_t p = 1; p < partitions; ++p) result += partial[p * columns + col_lane];
                    output[(size_t(token) * hv + head) * width + col] = result * sycl::rsqrt(float(width));
                }
                sycl::group_barrier(it.get_group());
            }
            for (size_t r = 0; r < rows; ++r) state[((first_row + r) * hv + head) * width + col] = column[r];
        });
    });
}
} // namespace strata::intel
