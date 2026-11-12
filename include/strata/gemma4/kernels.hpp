// Gemma-only adaptation from VecSzn/PolyStrata, include/strata/poly/kernels.hpp
// Revision 26e4cb5df7996edfb5ff549a63b82d8581e1e658 (MIT).
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::gemma4 {

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream);

void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows);

void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream);

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride = 0);

int sum_threads(int n);

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream);

void store_q8(const float* x, int8_t* codes, uint16_t* scales, int n, int64_t pos, int rows, void* stream);
void store_q8_at(const float* x, int8_t* codes, uint16_t* scales, int n, const long long* at, int rows, int ring, void* stream);

void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream);

void* blas_create(void* stream);
void blas_destroy(void* blas);

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream);

void geglu(const float* gate, const float* up, float* out, int n, void* stream);
void softcap(float* x, size_t n, float cap, void* stream);

void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream);

void route_rows(const float* logits, int n_expert, int n_used, int rows, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int32_t* ids, float* w, void* stream);

void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream);

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);

void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream);

struct ScaledRows {
    const float* x[32];
    float* acc[32];
    float w[32];
};
void add_scaled(const ScaledRows& rows, int count, int n, void* stream);

void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq = nullptr,
                    const int32_t* axes = nullptr, int sections = 1, const long long* at = nullptr);

struct Kvq8 {
    const int8_t* codes = nullptr;
    const uint16_t* scales = nullptr;
};

void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
            int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0, size_t score_floats = 0,
            Kvq8 kq8 = {}, Kvq8 vq8 = {});

size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim);

void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window = 0, int ring = 0, bool whole = false, Kvq8 kq8 = {},
                  Kvq8 vq8 = {}, uint16_t* kwork = nullptr, uint16_t* vwork = nullptr, size_t work_halves = 0);

struct Likeliest {
    int32_t token;
    float prob;
};
void likeliest(const float* logits, int n, Likeliest* out, void* stream);

void shortconv(const float* bcx, float* state, const float* w, float* out, int channels, int rows, void* stream,
               float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);

}
