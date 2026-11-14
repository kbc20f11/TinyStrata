#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/quant.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <stdexcept>

namespace strata::intel {
namespace {
namespace mx = sycl::ext::oneapi::experimental::matrix;
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
template<int Type, bool Grouped> sycl::event product(sycl::queue& q, const uint8_t* w, const float* x, float* y,
    int rows, int width, int tokens, ExpertPlan plan, bool separate_inputs) {
    constexpr size_t m = 8, n = 16, k = 16, subgroup = 16;
    const size_t stride = row_bytes<Type>(size_t(width)), row_tiles = (rows + n - 1) / n;
    const size_t tiles = Grouped ? (size_t(tokens) * plan.picks + m - 1) / m + plan.experts : (tokens + m - 1) / m;
    return q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<sycl::half, 1> a(m * k, handler), b(k * n, handler);
        sycl::local_accessor<float, 1> c(m * n, handler);
        handler.parallel_for(sycl::nd_range<1>(tiles * row_tiles * subgroup, subgroup),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const size_t tile = it.get_group_linear_id() / row_tiles, first_row = (it.get_group_linear_id() % row_tiles) * n;
            const size_t lane = it.get_local_linear_id();
            if constexpr (Grouped) if (tile >= size_t(plan.tiles[0])) return;
            const size_t expert = Grouped ? size_t(plan.owners[tile]) : 0;
            const size_t first = Grouped ? size_t(plan.offsets[tile]) : tile * m;
            int entries[m];
#pragma unroll
            for (size_t t = 0; t < m; ++t) {
                if constexpr (Grouped) entries[t] = first + t < size_t(plan.counts[expert]) ? plan.slots[expert * tokens + first + t] : -1;
                else entries[t] = first + t < size_t(tokens) ? int(first + t) : -1;
            }
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, m, k, mx::layout::row_major> ja;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, k, n, mx::layout::row_major> jb;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, m, n> jc;
            mx::joint_matrix_fill(sg, jc, 0.f);
            for (size_t base = 0; base < size_t(width); base += k) {
#pragma unroll
                for (size_t i = lane; i < m * k; i += subgroup) {
                    const size_t t = i / k, col = base + i % k;
                    const size_t source = Grouped && !separate_inputs ? size_t(entries[t]) / plan.picks : size_t(entries[t]);
                    a[i] = entries[t] >= 0 ? sycl::half(x[source * width + col]) : sycl::half(0);
                }
#pragma unroll
                for (size_t i = lane; i < k * n; i += subgroup) {
                    const size_t row = first_row + i % n, col = base + i / n;
                    b[i] = row < size_t(rows) ? sycl::half(quant::value<Type>(w + (expert * rows + row) * stride, col)) : sycl::half(0);
                }
                sycl::group_barrier(it.get_group());
                mx::joint_matrix_load(sg, ja, a.template get_multi_ptr<sycl::access::decorated::yes>(), k);
                mx::joint_matrix_load(sg, jb, b.template get_multi_ptr<sycl::access::decorated::yes>(), n);
                mx::joint_matrix_mad(sg, jc, ja, jb, jc);
                sycl::group_barrier(it.get_group());
            }
            mx::joint_matrix_store(sg, jc, c.template get_multi_ptr<sycl::access::decorated::yes>(), n, mx::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (size_t i = lane; i < m * n; i += subgroup) {
                const size_t t = i / n, row = first_row + i % n;
                if (entries[t] >= 0 && row < size_t(rows)) y[size_t(entries[t]) * rows + row] = c[i];
            }
        });
    });
}
void validate(const sycl::queue& q, int rows, int width, int tokens) {
    if (!q.has_property<sycl::property::queue::in_order>() || rows < 1 || width < 1 || width % 256 || tokens < 1 || tokens > 1024)
        throw std::invalid_argument("Invalid XMX matrix shape/queue");
}
}
#define XMX_DISPATCH(G) switch (type) { \
    case 0: return product<0, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 1: return product<1, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 30: return product<30, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 8: return product<8, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 10: return product<10, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 11: return product<11, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 14: return product<14, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 21: return product<21, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 22: return product<22, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    case 23: return product<23, G>(q,w,input,output,rows,width,tokens,plan,separate); \
    default: throw std::invalid_argument("Unsupported XMX weight type"); }
sycl::event quantized_matmul_xmx(sycl::queue& q, int type, const uint8_t* w, const float* input, float* output,
    int rows, int width, int tokens) {
    validate(q, rows, width, tokens);
    ExpertPlan plan{};
    const bool separate = false;
    XMX_DISPATCH(false)
}
sycl::event selected_matmul_xmx(sycl::queue& q, int type, const uint8_t* w, ExpertPlan plan,
    const float* input, float* output, int rows, int width, bool separate) {
    const int tokens = plan.tokens;
    validate(q, rows, width, tokens);
    if (plan.tile_width != 8) throw std::invalid_argument("XMX requires eight-row expert tiles");
    XMX_DISPATCH(true)
}
#undef XMX_DISPATCH
} // namespace strata::intel
