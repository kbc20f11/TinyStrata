#include "strata/intel/qwen_ops.hpp"
#include <stdexcept>

namespace strata::intel {
sycl::event causal_attention(sycl::queue& q, const float* queries, const float* gates,
                             const float* keys, const float* values, float* output,
                             int heads, int kv_heads, int context, int position, int tokens) {
    if (!q.has_property<sycl::property::queue::in_order>() || heads <= 0 || kv_heads <= 0 || heads % kv_heads ||
        position < 0 || tokens < 1 || position + tokens > context)
        throw std::invalid_argument("Invalid causal attention geometry/queue");
    constexpr size_t dimension = 256, query_tile = 8, key_tile = 16, group = query_tile * key_tile;
    constexpr size_t local_bytes = (query_tile * dimension + dimension * (key_tile + 1) + key_tile * dimension) * sizeof(float);
    if (q.get_device().get_info<sycl::info::device::local_mem_size>() < local_bytes)
        throw std::runtime_error("Intel tiled attention needs 41 KiB of local memory");
    return q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> query(query_tile * dimension, handler);
        // Odd pitch avoids bank conflicts both when loading and contracting K.
        sycl::local_accessor<float, 1> key(dimension * (key_tile + 1), handler);
        sycl::local_accessor<float, 1> value(key_tile * dimension, handler);
        handler.parallel_for(sycl::nd_range<1>(size_t((tokens + query_tile - 1) / query_tile) * heads * group, group),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const size_t head = it.get_group_linear_id() % heads, kv_head = head / (heads / kv_heads);
            const size_t first_query = (it.get_group_linear_id() / heads) * query_tile;
            const size_t local = it.get_local_linear_id(), query_row = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
            const size_t token = first_query + query_row;
            for (size_t i = local; i < query_tile * dimension; i += group) {
                const size_t t = first_query + i / dimension;
                query[i] = t < size_t(tokens) ? queries[(t * heads + head) * dimension + i % dimension] : 0.f;
            }
            float maximum = -INFINITY, total = 0, acc[dimension / key_tile] = {};
            const size_t end = position + sycl::min(size_t(tokens), first_query + query_tile);
            for (size_t begin = 0; begin < end; begin += key_tile) {
                for (size_t i = local; i < key_tile * dimension; i += group) {
                    const size_t cell = begin + i / dimension, col = i % dimension;
                    const bool valid = cell < end;
                    const size_t source = (kv_head * context + cell) * dimension + col;
                    key[col * (key_tile + 1) + i / dimension] = valid ? keys[source] : 0.f;
                    value[i] = valid ? values[source] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                const bool visible = token < size_t(tokens) && begin + lane <= size_t(position) + token;
                float score = 0;
                if (visible) for (size_t col = 0; col < dimension; ++col)
                    score = sycl::fma(query[query_row * dimension + col], key[col * (key_tile + 1) + lane], score);
                score = visible ? score / 16.f : -INFINITY;
                const float block_max = sycl::reduce_over_group(sg, score, sycl::maximum<float>());
                const float next_max = sycl::fmax(maximum, block_max);
                const float correction = total > 0 ? sycl::exp(maximum - next_max) : 0.f;
                const float probability = visible ? sycl::exp(score - next_max) : 0.f;
                total = total * correction + sycl::reduce_over_group(sg, probability, sycl::plus<float>());
#pragma unroll
                for (size_t col = 0; col < dimension / key_tile; ++col) acc[col] *= correction;
                for (size_t cell = 0; cell < key_tile; ++cell) {
                    const float p = sycl::select_from_group(sg, probability, cell);
#pragma unroll
                    for (size_t col = 0; col < dimension / key_tile; ++col)
                        acc[col] = sycl::fma(p, value[cell * dimension + col * key_tile + lane], acc[col]);
                }
                maximum = next_max;
                sycl::group_barrier(it.get_group());
            }
            if (token < size_t(tokens)) {
#pragma unroll
                for (size_t col = 0; col < dimension / key_tile; ++col) {
                    const size_t index = (token * heads + head) * dimension + col * key_tile + lane;
                    output[index] = acc[col] / total / (1.f + sycl::exp(-gates[index]));
                }
            }
        });
    });
}
sycl::event decode_attention(sycl::queue& q, const float* queries, const float* gates, const float* keys, const float* values,
    float* scores, float* partial, float* output, int heads, int kv_heads, int context, const int32_t* position, int extent, sycl::event* stages) {
    if (!q.has_property<sycl::property::queue::in_order>() || heads < 1 || kv_heads < 1 || heads % kv_heads || extent < 1 || extent > context)
        throw std::invalid_argument("Invalid decode attention geometry/queue");
    constexpr size_t group = 128, subgroup = 16, per_group = group / subgroup, block = 256;
    const size_t entries = size_t(heads) * extent, splits = (context + block - 1) / block;
    // Adjacent subgroups share a KV head/cell, and each subgroup reads a
    // contiguous 16-column slice. This avoids scalar 256-term row walks.
    auto qk = q.parallel_for(sycl::nd_range<1>(((entries + per_group - 1) / per_group) * group, group),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg = it.get_sub_group();
        const size_t entry = it.get_group_linear_id() * per_group + sg.get_group_linear_id();
        const size_t head = entry % heads, cell = entry / heads, lane = sg.get_local_linear_id();
        if (cell >= size_t(extent) || cell > size_t(position[0])) return;
        const size_t kv = head / (heads / kv_heads);
        float sum = 0;
        for (size_t c = lane; c < 256; c += subgroup)
            sum = sycl::fma(queries[head * 256 + c], keys[(kv * context + cell) * 256 + c], sum);
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        if (lane == 0) scores[head * context + cell] = sum / 16.f;
    });
    auto softmax = q.parallel_for(sycl::nd_range<1>(size_t(heads) * group, group), [=](sycl::nd_item<1> it) {
        const size_t head = it.get_group_linear_id(), lane = it.get_local_linear_id(), end = position[0];
        float maximum = -INFINITY;
        for (size_t cell = lane; cell <= end; cell += group) maximum = sycl::fmax(maximum, scores[head * context + cell]);
        maximum = sycl::reduce_over_group(it.get_group(), maximum, sycl::maximum<float>());
        float sum = 0;
        for (size_t cell = lane; cell <= end; cell += group) sum += sycl::exp(scores[head * context + cell] - maximum);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        for (size_t cell = lane; cell <= end; cell += group) scores[head * context + cell] = sycl::exp(scores[head * context + cell] - maximum) / sum;
    });
    // Split the V contraction into bounded 256-cell pieces; long contexts
    // retain enough independent work instead of one serial walk per column.
    auto pv = q.parallel_for(sycl::nd_range<1>(size_t((extent + block - 1) / block) * heads * group, group), [=](sycl::nd_item<1> it) {
        const size_t head = it.get_group_linear_id() % heads, split = it.get_group_linear_id() / heads;
        const size_t lane = it.get_local_linear_id(), kv = head / (heads / kv_heads);
        const size_t begin = split * block, end = sycl::min(begin + block, size_t(position[0] + 1));
        float a = 0, b = 0;
        for (size_t cell = begin; cell < end; ++cell) {
            const float weight = scores[head * context + cell];
            const size_t base = (kv * context + cell) * 256;
            a = sycl::fma(weight, values[base + lane], a);
            b = sycl::fma(weight, values[base + lane + 128], b);
        }
        const size_t target = (head * splits + split) * 256 + lane;
        partial[target] = a; partial[target + 128] = b;
    });
    if (stages) { stages[0] = qk; stages[1] = softmax; stages[2] = pv; }
    return q.parallel_for(sycl::range<1>(size_t(heads) * 256), [=](sycl::id<1> id) {
        const size_t head = id[0] / 256, col = id[0] % 256;
        const size_t used = (size_t(position[0] + 1) + block - 1) / block;
        float sum = 0;
        for (size_t split = 0; split < used; ++split) sum += partial[(head * splits + split) * 256 + col];
        output[id] = sum / (1.f + sycl::exp(-gates[id]));
    });
}
} // namespace strata::intel
