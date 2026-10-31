// Clover INT8 attention: adjacent queries share KV reads/dequantization.
// Complete chunks avoid per-query bounds tests in the ordered QK/PV loops.
// Adapted from TinyStrata attention.cuh / Strata MIT attention kernels.
#pragma once
#include "strata/qwen35/attention.cuh"
#include <stdexcept>

namespace strata::qwen35::clover_attention {
namespace A = strata::qwen35::attention;
constexpr int HD=256, CELLS=256, THREADS=256;

// Queries share only KV loads/dequantization. Each query keeps its own ordered
// 256-dimension dot, identical 256-lane max/sum tree, and ordered V accumulation.
// Partial row layout and the existing merge kernel are unchanged.
template <int Queries>
__global__ void part(const float* q,A::KvCache cache,const int32_t* pos0,int t0,int valid_queries,
    int H,int HKV,int ctx,float scale,int n_chunk,float* pacc,float* pm,float* pl,int visible_end=0) {
    static_assert(Queries==2 || Queries==4);
    __shared__ float qs[Queries][HD],sc[Queries][CELLS],reduce[Queries][THREADS];
    const int chunk=blockIdx.x, h=blockIdx.y%H;
    const int first=(blockIdx.y/H)*Queries,kvh=h/(H/HKV),j=threadIdx.x;
    const int p0=chunk*CELLS,p=p0+j;
    int npos[Queries],maximum_position=0,minimum_live_position=0x7fffffff;
#pragma unroll
    for(int r=0;r<Queries;++r) {
        const int t=t0+first+r;
        npos[r]=first+r<valid_queries ? (visible_end>0 ? visible_end : *pos0+t+1) : 0;
        maximum_position=max(maximum_position,npos[r]);
        if(first+r<valid_queries)minimum_live_position=min(minimum_live_position,npos[r]);
    }
    if(p0>=maximum_position)return; // The maximum, never the first query's bound.
    // Uniform over the CTA. Ignore padded queries when finding the live
    // minimum; their zero Q/P rows can perform harmless unused arithmetic.
    const bool full_chunk=p0+CELLS<=minimum_live_position;
#pragma unroll
    for(int r=0;r<Queries;++r)
        qs[r][j]=first+r<valid_queries ? q[((size_t)(t0+first+r)*H+h)*HD+j] : 0.f;
    __syncthreads();
    float dots[Queries]={};
    if(full_chunk) {
        const size_t offset=((size_t)kvh*ctx+p)*HD;
        for(int i=0;i<HD;++i) {
            const float key=A::kv_read<2>(cache.key,cache.key_scale,offset+i,HD,cache.key_scale16);
#pragma unroll
            for(int r=0;r<Queries;++r)dots[r]+=qs[r][i]*key;
        }
    } else if(p<maximum_position) {
        const size_t offset=((size_t)kvh*ctx+p)*HD;
        for(int i=0;i<HD;++i) {
            const float key=A::kv_read<2>(cache.key,cache.key_scale,offset+i,HD,cache.key_scale16);
#pragma unroll
            for(int r=0;r<Queries;++r)
                if(p<npos[r])dots[r]+=qs[r][i]*key;
        }
    }
#pragma unroll
    for(int r=0;r<Queries;++r) {
        dots[r]=p<npos[r] ? dots[r]*scale : -INFINITY;
        reduce[r][j]=dots[r];
    }
    __syncthreads();
    for(int step=THREADS/2;step>0;step>>=1) {
        if(j<step) {
#pragma unroll
            for(int r=0;r<Queries;++r)reduce[r][j]=fmaxf(reduce[r][j],reduce[r][j+step]);
        }
        __syncthreads();
    }
    float maximum[Queries];
#pragma unroll
    for(int r=0;r<Queries;++r)maximum[r]=reduce[r][0];
    __syncthreads(); // Same barrier after the old block_max return value read.
#pragma unroll
    for(int r=0;r<Queries;++r) {
        const float weight=p<npos[r] ? expf(dots[r]-maximum[r]) : 0.f;
        sc[r][j]=weight;
        reduce[r][j]=weight;
    }
    __syncthreads();
    for(int step=THREADS/2;step>0;step>>=1) {
        if(j<step) {
#pragma unroll
            for(int r=0;r<Queries;++r)reduce[r][j]+=reduce[r][j+step];
        }
        __syncthreads();
    }
    float denominator[Queries];
#pragma unroll
    for(int r=0;r<Queries;++r)denominator[r]=reduce[r][0];
    __syncthreads(); // Same barrier after the old block_sum return value read.
    float accumulators[Queries]={};
    const int maximum_cells=min(CELLS,maximum_position-p0);
    const size_t offset=((size_t)kvh*ctx+p0)*HD+j;
    if(full_chunk) {
        for(int k=0;k<CELLS;++k) {
            const float value=A::kv_read<2>(cache.value,cache.value_scale,offset+(size_t)k*HD,HD,cache.value_scale16);
#pragma unroll
            for(int r=0;r<Queries;++r)accumulators[r]+=sc[r][k]*value;
        }
    } else {
    for(int k=0;k<maximum_cells;++k) {
        const float value=A::kv_read<2>(cache.value,cache.value_scale,offset+(size_t)k*HD,HD,cache.value_scale16);
#pragma unroll
        for(int r=0;r<Queries;++r)
            if(k<npos[r]-p0)accumulators[r]+=sc[r][k]*value;
    }
    }
#pragma unroll
    for(int r=0;r<Queries;++r) {
        if(first+r>=valid_queries || p0>=npos[r])continue;
        const size_t at=(size_t)((first+r)*H+h)*n_chunk+chunk;
        pacc[at*HD+j]=accumulators[r];
        if(j==0) {pm[at]=maximum[r];pl[at]=denominator[r];}
    }
}

inline cudaError_t launch(const float* q,A::KvCache cache,const int32_t* pos0,int t0,int rows,
    int H,int HKV,int ctx,float scale,int n_chunk,int grid_chunks,float* pacc,float* pm,float* pl,
    cudaStream_t stream,int visible_end=0,int queries=0) {
    if(!q || !cache.key || !cache.value || !cache.key_scale16 || !cache.value_scale16 || !pos0 ||
       !pacc || !pm || !pl || t0<0 || rows<2 || rows>4 || H<=0 || HKV<=0 || H!=8*HKV || ctx<1 ||
       n_chunk<(ctx+255)/256 || grid_chunks<1 || grid_chunks>n_chunk || visible_end<0 || visible_end>ctx ||
       (queries!=0 && queries!=2 && queries!=4))
        throw std::invalid_argument("Invalid Clover INT8 query-shared shape");
    if(queries==0)queries=rows==2?2:4;
    if(queries==2)part<2><<<dim3(grid_chunks,((rows+1)/2)*H),THREADS,0,stream>>>(q,cache,pos0,t0,rows,H,HKV,ctx,scale,n_chunk,pacc,pm,pl,visible_end);
    else part<4><<<dim3(grid_chunks,((rows+3)/4)*H),THREADS,0,stream>>>(q,cache,pos0,t0,rows,H,HKV,ctx,scale,n_chunk,pacc,pm,pl,visible_end);
    return cudaGetLastError();
}
} // namespace strata::qwen35::clover_attention
