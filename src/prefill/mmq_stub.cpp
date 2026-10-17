#include "strata/prefill/moe_mmq.hpp"
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
bool fits(int, int64_t) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*, float*) {}
bool fp4_activations(int) { return false; }
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*, const void*, void*, size_t) {}
bool gather_native_group(const GatherGroup&, size_t, size_t, size_t, size_t, void*, size_t, void*, size_t, void*, size_t,
                         void*, size_t) { return false; }
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void swiglu_scaled(const float*, float*, int64_t, int64_t, const int32_t*, int, const float*, int64_t, void*) {}
void down_row_scales(float*, const int32_t*, int, const float*, int64_t, void*) {}
void iota(int32_t*, int64_t, void*) {}
}  // namespace strata::prefill::mmq
