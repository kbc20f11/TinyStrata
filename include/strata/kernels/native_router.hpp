#pragma once
#include <cstdint>
namespace strata::kernels {
// Qwen3.6: logits [n_tok, 256], top-8 IDs and normalized weights.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
}
