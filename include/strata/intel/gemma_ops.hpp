#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <memory>

namespace strata::intel {
struct GemmaKv {
    int8_t *key=nullptr,*value=nullptr;
    sycl::half *key_scale=nullptr,*value_scale=nullptr;
    int capacity=0,heads=0,dim=0,window=0;
};
sycl::event gemma_head_norm_rope(sycl::queue&,const float* input,const float* weights,float* output,
    int heads,int dim,int rotation,int position,int tokens,float epsilon,float base,const float* frequencies=nullptr);
sycl::event gemma_store_q8(sycl::queue&,const float* key,const float* value,GemmaKv,int position,int tokens,int32_t* error);
sycl::event gemma_geglu(sycl::queue&,const float* gate,const float* up,float* output,size_t count);
sycl::event gemma_geglu_packed(sycl::queue&,const float* gate_up,float* output,int width,int rows);
sycl::event gemma_route(sycl::queue&,const float* logits,const float* expert_scales,int32_t* ids,float* weights,
    int experts,int picks,int tokens,int embedding_width);
class GemmaAttention {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    // Only the explicit Gemma caller enables PV; all other callers keep false.
    GemmaAttention(sycl::queue&,int context,int tokens,int heads,int max_dim,size_t max_cache_elements,bool prefill_mkl,bool split_pv=false);
    ~GemmaAttention();
    sycl::event run(const float* queries,GemmaKv,float* output,int heads,int position,int tokens,float scale=1.f);
};
}
