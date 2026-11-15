// Model semantics adapted from src/lfm2/lfm2moe.cpp and kernels.cu.
// Their PolyStrata attribution is retained in LICENSE and NOTICE.
#include "strata/intel/lfm_ops.hpp"
#include <cmath>
#include <stdexcept>

namespace strata::intel {
namespace {
void ordered(const sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("LFM operations require an in-order queue");
}
template<bool Selected> sycl::event q4_fp32_product(sycl::queue& q, const uint8_t* weights,
    const int32_t* ids, const float* input, float* output, int rows, int width, bool separate) {
    ordered(q);
    if (!weights || !input || !output || reinterpret_cast<uintptr_t>(weights)%16 || rows < 1 || width < 256 || width % 256 || (Selected && !ids))
        throw std::invalid_argument("Invalid LFM FP32 Q4_K matrix shape/buffer");
    constexpr size_t subgroup=16, pack=8, group=128, rows_per_group=group/subgroup;
    constexpr size_t picks=Selected?4:1;
    const size_t entries=size_t(rows)*picks, stride=size_t(width/256)*144;
    return q.parallel_for(sycl::nd_range<1>(((entries+rows_per_group-1)/rows_per_group)*group,group),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg=it.get_sub_group();
        const size_t entry=it.get_group_linear_id()*rows_per_group+sg.get_group_linear_id();
        if(entry>=entries)return; // Uniform subgroup return, including row17 tails.
        const size_t row=entry%rows,pick=entry/rows,lane=sg.get_local_linear_id();
        const uint8_t* weight=weights+(Selected?size_t(ids[pick])*rows+row:row)*stride;
        const float* x=input+(Selected&&separate?pick*width:0);
        float acc[pack]={};
        for(size_t block=0;block<size_t(width/256);++block){
            const uint8_t* raw=weight+block*144;
            float delta=0.f,minimum=0.f;
            if(lane==0){
                const auto* h=reinterpret_cast<const uint16_t*>(raw);
                delta=float(sycl::bit_cast<sycl::half>(h[0]));
                minimum=float(sycl::bit_cast<sycl::half>(h[1]));
            }
            delta=sycl::select_from_group(sg,delta,0);
            minimum=sycl::select_from_group(sg,minimum,0);
#pragma unroll
            for(size_t half=0;half<2;++half){
                const size_t scale_group=half*4+lane/4;
                float scaled=0.f,correction=0.f;
                if(lane%4==0){
                    const uint8_t* s=raw+4;
                    const int sc=scale_group<4?s[scale_group]&63:
                        (s[scale_group+4]&15)|((s[scale_group-4]>>6)<<4);
                    const int mn=scale_group<4?s[scale_group+4]&63:
                        (s[scale_group+4]>>4)|((s[scale_group]>>6)<<4);
                    scaled=delta*float(sc);correction=minimum*float(mn);
                }
                scaled=sycl::select_from_group(sg,scaled,lane&~size_t(3));
                correction=sycl::select_from_group(sg,correction,lane&~size_t(3));
                uint32_t low=0,high=0;
                if((lane&4)==0){
                    // All Q4_K block/row starts are 16-byte aligned. This is
                    // an aligned8-byte payload, read once for both nibbles.
                    const auto* words=reinterpret_cast<const uint32_t*>(raw+16+half*64+(lane/8)*32+(lane%4)*8);
                    low=words[0];high=words[1];
                }
                low=sycl::select_from_group(sg,low,lane&~size_t(4));
                high=sycl::select_from_group(sg,high,lane&~size_t(4));
                const size_t base=block*256+half*128+lane*pack;
                const unsigned nibble=unsigned((scale_group&1)*4);
#pragma unroll
                for(size_t k=0;k<pack;++k){
                    const uint32_t word=k<4?low:high;
                    const int code=int((word>>(unsigned(k%4)*8+nibble))&15);
                    // Same FP32 nesting and FMA/accumulator order as the
                    // scalar baseline: ((delta*scale)*code)-(dmin*minimum).
                    const float value=scaled*float(code)-correction;
                    acc[k]=sycl::fma(value,x[base+k],acc[k]);
                }
            }
        }
        float sum=acc[0];
#pragma unroll
        for(size_t k=1;k<pack;++k)sum+=acc[k];
        sum=sycl::reduce_over_group(sg,sum,sycl::plus<float>());
        if(lane==0)output[entry]=sum;
    });
}
}
sycl::event lfm_q4_fp32_matvec(sycl::queue& q,const uint8_t* weights,const float* input,float* output,int rows,int width){
    return q4_fp32_product<false>(q,weights,nullptr,input,output,rows,width,false);
}
sycl::event lfm_q4_fp32_selected(sycl::queue& q,const uint8_t* weights,const int32_t* ids,
    const float* input,float* output,int rows,int width,bool separate){
    return q4_fp32_product<true>(q,weights,ids,input,output,rows,width,separate);
}
sycl::event lfm_shortconv(sycl::queue& q, const float* bcx, float* state,
                         const float* weights, float* output, int channels, int tokens) {
    ordered(q);
    if (!bcx || !state || !weights || !output || channels <= 0 || tokens <= 0)
        throw std::invalid_argument("Invalid LFM short-convolution shape/buffer");
    // Every output reads the old state; update it only in the following
    // dispatch. Parallel token evaluation avoids a serial T-step channel loop.
    q.parallel_for(sycl::range<1>(size_t(tokens) * channels), [=](sycl::id<1> ix) {
        const size_t at = ix[0];
        const int c = int(at % channels), t = int(at / channels);
        const auto bx = [&](int row) {
            if (row < 0) return state[size_t(c) * 2 + 2 + row];
            const size_t base = size_t(row) * channels * 3;
            return bcx[base + c] * bcx[base + channels * 2 + c];
        };
        const float* w = weights + size_t(c) * 3;
        output[at] = bcx[size_t(t) * channels * 3 + channels + c] *
            (bx(t - 2) * w[0] + bx(t - 1) * w[1] + bx(t) * w[2]);
    });
    return q.parallel_for(sycl::range<1>(size_t(channels)), [=](sycl::id<1> ix) {
        const size_t c = ix[0];
        const auto bx = [&](int row) {
            if (row < 0) return state[c * 2 + 2 + row];
            const size_t base = size_t(row) * channels * 3;
            return bcx[base + c] * bcx[base + channels * 2 + c];
        };
        // Read both before modifying state: a single-token call keeps old s1.
        const float a = bx(tokens - 2), b = bx(tokens - 1);
        state[c * 2] = a;
        state[c * 2 + 1] = b;
    });
}
sycl::event lfm_route(sycl::queue& q, const float* logits, const float* bias,
                     int32_t* ids, float* weights, int tokens, int experts,
                     int picks, bool normalize, float scale) {
    ordered(q);
    if (!logits || !bias || !ids || !weights || tokens <= 0 || experts <= 0 ||
        experts > 64 || picks <= 0 || picks > experts || !std::isfinite(scale))
        throw std::invalid_argument("Invalid LFM sigmoid-routing shape/buffer");
    return q.parallel_for(sycl::range<1>(size_t(tokens)), [=](sycl::id<1> ix) {
        const size_t row = ix[0];
        float probability[64], rank[64];
        for (int e = 0; e < experts; ++e) {
            probability[e] = 1.f / (1.f + sycl::exp(-logits[row * experts + e]));
            rank[e] = probability[e] + bias[e];
        }
        float total = 0.f;
        for (int k = 0; k < picks; ++k) {
            int best = 0;
            for (int e = 1; e < experts; ++e)
                if (rank[e] > rank[best]) best = e;
            rank[best] = -INFINITY;
            ids[row * picks + k] = best;
            weights[row * picks + k] = probability[best];
            total += probability[best];
        }
        const float divisor = normalize ? sycl::fmax(total, 6.103515625e-5f) : 1.f;
        for (int k = 0; k < picks; ++k) {
            float w = weights[row * picks + k] / divisor;
            if (scale != 0.f) w *= scale;
            weights[row * picks + k] = w;
        }
    });
}
}
