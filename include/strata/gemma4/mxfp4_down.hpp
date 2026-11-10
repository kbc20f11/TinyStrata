#pragma once
#include "strata/gemma4/matvec.hpp"
#include <cstdint>
namespace strata::kernels::mxfp4_down {
// Experimental HIP/gfx1200 cached-Expert down path; Engine gates the device once at load.
// Exact original GGUF bytes and Q8_1 bytes; only 2880x2880, type39, count1..32.
// Matrix bases must be 4-byte aligned, Q8_1 bases 4-byte aligned, y float-aligned.
// IDs table entries must meet the same alignment contract; IDs must be valid.
bool direct(int type,const void* w,const void* x,float* y,int n_in,int n_out,int ncols,void* stream);
bool slots(int type,const strata::kernels::MmvqSlots& p,int count,int n_in,int n_out,void* stream);
bool ids(int type,const strata::kernels::MmvqIds& p,int count,int n_in,int n_out,void* stream);
// Unit probe only: all256 original scale bit patterns and all16 packed E2M1 int8 codes.
void probe(uint32_t* scale_bits,uint32_t* packed_codes,void* stream);
}
