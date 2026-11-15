#pragma once
#include "strata/intel/memory_policy.hpp"
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <string>

namespace strata::intel::context_policy {
constexpr int full_limit=262144, lfm_limit=131072, auto_quantum=2048;
inline int model_limit(double trained,int policy){
    if(!std::isfinite(trained)||trained<1||trained>std::numeric_limits<int>::max()||std::floor(trained)!=trained)
        throw std::invalid_argument("Intel GGUF requires a valid context_length");
    return std::min(int(trained),policy);
}
inline int parse(std::string_view value,int limit){
    if(value=="auto")return 0;
    if(value.empty())throw std::invalid_argument("Intel context must be auto or a positive integer");
    if(limit<1)throw std::invalid_argument("Invalid Intel context policy limit");
    uint64_t n=0;for(char ch:value){if(ch<'0'||ch>'9'||n>uint64_t(limit)/10)throw std::invalid_argument("Intel context exceeds model policy limit");n=n*10+uint64_t(ch-'0');if(n>uint64_t(limit))throw std::invalid_argument("Intel context exceeds model policy limit");}
    if(!n||n>uint64_t(limit))throw std::invalid_argument("Intel context exceeds model policy limit");return int(n);
}
inline uint64_t add(uint64_t a,uint64_t b){if(b>UINT64_MAX-a)throw std::overflow_error("Intel context byte sum overflow");return a+b;}
inline uint64_t mul(uint64_t a,uint64_t b){if(a&&b>UINT64_MAX/a)throw std::overflow_error("Intel context byte product overflow");return a*b;}
inline uint64_t round256(uint64_t n){return mul(add(n,255)/256,256);}

// Counts allocations, not logical tensor sizes: device includes shared USM's
// possible GPU backing; host_visible is that same shared allocation's RAM
// admission. UMA charges it once. Ordinary snapshots are host_only.
struct Footprint {
    uint64_t device=0,host_visible=0,host_only=0,largest=0,cache=0;
    uint64_t owned_device=0,owned_host_visible=0;
    void allocation(uint64_t count,uint64_t bytes,bool shared=false,uint64_t copies=1,bool kv=false){
        const auto single=mul(count,bytes),total=mul(single,copies);
        device=add(device,total);if(shared)host_visible=add(host_visible,total);
        largest=std::max(largest,single);if(kv)cache=add(cache,total);
    }
    void host(uint64_t bytes){host_only=add(host_only,bytes);}
    void own(){owned_device=device;owned_host_visible=host_visible;}
    uint64_t uma_total()const{return add(device,host_only);}
};
inline void gemm(Footprint& p,uint64_t context,uint64_t tokens){
    p.allocation(8192ull*2048,2);p.allocation(1024ull*8192,2);
    p.allocation(1024ull*2048,4);p.allocation(mul(context,tokens),4);
}
inline void attention(Footprint& p,uint64_t context,uint64_t tokens,uint64_t heads,uint64_t dim,uint64_t expansion,bool mkl){
    p.allocation(mul(context,std::max(tokens,mul(8,heads))),4);
    if(mkl){p.allocation(expansion,4,false,2);p.allocation(mul(tokens,dim),4);}
}
inline void q8_cache(Footprint& p,uint64_t cells,bool shared=false,uint64_t copies=1){
    p.allocation(cells,1,shared,mul(copies,2),true);
    p.allocation(cells/32,2,shared,mul(copies,2),true);
}

struct GemmaSpec {
    uint64_t chunk=512,width=2816,vocabulary=262144,heads=16,full_dim=512,swa_dim=256;
    uint64_t dense_ff=2112,experts=128,picks=8,expert_ff=704,window=1024;
    bool mkl=true,pv_split=false;
};
inline Footprint gemma(uint64_t c,const GemmaSpec& s={}){
    Footprint p;const auto n=s.chunk;
    const auto sliding=std::min(c,add(s.window,n));
    q8_cache(p,mul(mul(sliding,8),s.swa_dim),false,25);
    q8_cache(p,mul(mul(c,2),s.full_dim),false,5);
    p.allocation(mul(n,s.width),4,false,8);
    p.allocation(mul(mul(n,s.heads),s.full_dim),4,false,3);
    p.allocation(mul(mul(n,8),s.swa_dim),4,false,4);
    p.allocation(mul(n,s.dense_ff),4,false,3);
    p.allocation(mul(mul(n,s.picks),mul(2,s.expert_ff)),4); // packed gate/up
    p.allocation(mul(mul(n,s.picks),s.expert_ff),4);
    p.allocation(mul(mul(n,s.picks),s.width),4);
    p.allocation(mul(n,s.experts),4);p.allocation(mul(n,s.picks),4);p.allocation(mul(n,s.picks),4);
    p.allocation(n,4,true);p.allocation(1,4,true);p.allocation(s.vocabulary,4,true);
    p.allocation(s.experts,4);p.allocation(mul(n,s.experts),4);
    const auto tiles=add(add(mul(n,s.picks),3)/4,s.experts);
    p.allocation(tiles,4,false,2);p.allocation(1,4);
    p.own();
    const auto expansion=std::max(mul(mul(sliding,8),s.swa_dim),mul(mul(c,2),s.full_dim));
    attention(p,c,n,s.heads,s.full_dim,expansion,s.mkl);
    if(s.pv_split)p.allocation(mul(mul(s.heads,add(c,255)/256),s.full_dim),4);
    if(s.mkl)gemm(p,1,n);
    p.host(16ull<<20); // bounded sampler/history/metadata work; not another OS floor
    return p;
}
struct LfmSpec {uint64_t chunk=512,width=2048,vocabulary=128000,heads=32,kv_heads=8,dim=64,dense_ff=11776,experts=64,picks=4,expert_ff=1536;bool mkl=true;};
inline Footprint lfm(uint64_t c,const LfmSpec& s={}){
    Footprint p;const auto n=s.chunk,cells=mul(mul(c,s.kv_heads),s.dim);
    q8_cache(p,cells,false,10);p.allocation(mul(2,s.width),4,false,60); // live and rollback convolution,30 layers
    p.allocation(mul(n,s.width),4,false,5);
    p.allocation(mul(mul(n,3),s.width),4); // BCX is one allocation
    p.allocation(mul(mul(n,s.heads),s.dim),4,false,3);
    p.allocation(mul(mul(n,s.kv_heads),s.dim),4,false,3);
    p.allocation(mul(n,std::max(s.dense_ff,mul(s.picks,s.expert_ff))),4,false,3);
    p.allocation(mul(mul(n,s.picks),s.width),4);p.allocation(mul(n,s.experts),4);
    p.allocation(mul(n,s.picks),4,false,2);p.allocation(s.width,4,false,2);
    p.allocation(n,4,true);p.allocation(1,4,true);p.allocation(s.vocabulary,4,true);p.allocation(s.experts,4,true);
    p.allocation(mul(n,s.experts),4);const auto tiles=add(add(mul(n,s.picks),3)/4,s.experts);
    p.allocation(tiles,4,false,2);p.allocation(1,4);p.own();
    attention(p,c,n,s.heads,s.dim,cells,s.mkl);if(s.mkl)gemm(p,1,n);
    p.host(16ull<<20);return p;
}
struct QwenSpec {
    uint64_t chunk=128,vocabulary=248320,draft_vocabulary=0,verify_capacity=8;
    bool int8=false,mtp=false,mkl=false,dp4a=false,dense_split=false,support=false;
    bool device_state=false,turn_checkpoint=true,graph=true,verify_graph=false,npu=false,npu_split=false;
};
inline Footprint qwen(uint64_t c,const QwenSpec& s={}){
    Footprint p;const auto n=s.chunk;const bool shared=!s.device_state;
    const auto a=[&](uint64_t count,uint64_t size=4,uint64_t copies=1){p.allocation(count,size,shared,copies);};
    const auto control=[&](uint64_t count){p.allocation(count,4,true);};
    const auto cells=mul(mul(2,c),256),layers=10+uint64_t(s.mtp);
    if(s.int8)q8_cache(p,cells,shared,layers);else p.allocation(cells,4,shared,2*layers,true);
    const uint64_t recurrent=30ull*(32*128*128+64*128*3)*4;
    a(32*128*128,4,60);a(64*128*3,4,60);
    a(mul(n,2048),4,3);a(mul(n,8192));a(mul(n,4096));a(mul(n,32),4,2);
    a(mul(n,8192));a(mul(n,4096),4,2);
    a(mul(n,256));a(mul(n,8));a(mul(n,8));a(mul(n,8*512),4,3);a(mul(n,8*2048));
    a(mul(n,512),4,3);a(mul(n,2048));a(n);
    a(mul(n,16*256),4,2);a(mul(n,2*256),4,2);
    const auto rows=mul(16,s.int8?std::min(n,s.verify_capacity):1);
    a(mul(rows,c));a(mul(rows,round256(c)));a(mul(n,16*256));
    control(s.vocabulary);control(256);a(mul(256,n));
    const auto tiles=add(add(mul(n,8),3)/4,256);a(tiles,4,2);a(1);
    control(n);control(n);a(2048);
    if(s.int8){control(1);a(cells,4,2);}
    if(s.mtp){
        a(mul(n,4096));a(mul(n,2048),4,4);a(2048,4,2);
        control(s.draft_vocabulary?s.draft_vocabulary:s.vocabulary);
        if(s.draft_vocabulary)control(s.draft_vocabulary);
        control(mul(s.verify_capacity,s.vocabulary));a(mul(s.verify_capacity,2048),4,2);
        a(mul(s.verify_capacity,8192),4,60);a(mul(s.verify_capacity,32),4,60);
    }
    p.own();
    if(s.support){control(s.vocabulary);control(65536);control(2);control(8192);control(s.vocabulary);control(s.vocabulary);}
    if(s.mkl)gemm(p,c,n);
    if(s.dp4a){p.allocation(512*8192,1);p.allocation(512*8192/32,4);p.allocation(1,4,true);
        p.allocation(2048,4);p.allocation(512,4);p.allocation(16,1);p.allocation(256,4);p.allocation(128,1);
        if(s.dense_split)p.allocation(4*8*8192,4);}
    if(s.turn_checkpoint)p.host(add(add(recurrent,2048ull*4*(s.mtp?2:1)),mul(c,4)));
    p.host(16ull<<20);
    // Bounded opaque graph/NPU runtime reservations retain the old safety
    // margins, distinct from exact USM arrays and the one4GiB OS headroom.
    if(s.graph)p.device=add(p.device,64ull<<20);
    if(s.verify_graph)p.device=add(p.device,128ull<<20);
    if(s.npu||s.npu_split){p.allocation(8192*2048,2,true);p.allocation(1024*8192,2,true);p.allocation(1024*8192,4,true);p.host(96ull<<20);}
    if(s.npu_split){p.allocation(64ull*1024*1024,2,true);p.allocation(1024*8192,4);}
    return p;
}
inline bool fits(const MemoryBudget& memory,uint64_t max_allocation,const Footprint& p){
    return p.largest<=max_allocation&&execution_allocation_fits(memory,p.device,p.host_visible,p.host_only);
}
inline std::string rejection(const char* why,int requested,const Footprint& p,const MemoryBudget& m,uint64_t maximum){
    return std::string(why)+" requested="+std::to_string(requested)+" device_bytes="+std::to_string(p.device)+
        " device_budget="+std::to_string(allocation_budget(m))+" host_visible_bytes="+std::to_string(p.host_visible)+
        " host_only_bytes="+std::to_string(p.host_only)+" host_budget="+std::to_string(system_allocation_budget(m))+
        " nonlocal_budget="+std::to_string(m.wddm_nonlocal_known?m.wddm_nonlocal_available:0)+
        " largest_allocation="+std::to_string(p.largest)+" maximum_allocation="+std::to_string(maximum);
}
struct Selection {int tokens=0;bool automatic=false;Footprint bytes;};
template<class Estimate>Selection select(int requested,int ceiling,const MemoryBudget& memory,uint64_t max_allocation,Estimate estimate){
    if(requested<0||ceiling<1||ceiling>full_limit||requested>ceiling)throw std::invalid_argument("Intel context exceeds model policy limit");
    if(requested){auto p=estimate(uint64_t(requested));if(!fits(memory,max_allocation,p))throw std::runtime_error(rejection("Explicit Intel context does not fit; request was not reduced.",requested,p,memory,max_allocation));return {requested,false,p};}
    int lo=0,hi=ceiling/auto_quantum;
    while(lo<hi){const int mid=lo+(hi-lo+1)/2;const auto p=estimate(uint64_t(mid)*auto_quantum);if(fits(memory,max_allocation,p))lo=mid;else hi=mid-1;}
    if(!lo)throw std::runtime_error(rejection("Intel auto context cannot fit2048 tokens after weights and the existing4GiB OS headroom.",auto_quantum,estimate(auto_quantum),memory,max_allocation));
    const int chosen=lo*auto_quantum;return {chosen,true,estimate(uint64_t(chosen))};
}
inline void report(const Selection& s,int limit,const MemoryBudget& m,uint64_t maximum){
    std::fprintf(stderr,"Intel context_auto=%d max_context=%d model_limit=%d quantum=%d context_budget_device=%llu context_budget_host=%llu context_budget_nonlocal=%llu context_budget_wddm_known=%d context_budget_max_allocation=%llu context_bytes_device=%llu context_bytes_host_visible=%llu context_bytes_host_only=%llu context_bytes_cache=%llu context_bytes_largest_allocation=%llu os_headroom_bytes=%llu\n",int(s.automatic),s.tokens,limit,auto_quantum,(unsigned long long)allocation_budget(m),(unsigned long long)system_allocation_budget(m),(unsigned long long)(m.wddm_nonlocal_known?m.wddm_nonlocal_available:0),int(m.wddm_known),(unsigned long long)maximum,(unsigned long long)s.bytes.device,(unsigned long long)s.bytes.host_visible,(unsigned long long)s.bytes.host_only,(unsigned long long)s.bytes.cache,(unsigned long long)s.bytes.largest,(unsigned long long)(4*GiB));
    std::printf("INFO context_auto=%d\nINFO context_budget_device=%llu\nINFO context_bytes_device=%llu\nINFO context_bytes_host_visible=%llu\nINFO context_bytes_host_only=%llu\nINFO max_context=%d\n",int(s.automatic),(unsigned long long)allocation_budget(m),(unsigned long long)s.bytes.device,(unsigned long long)s.bytes.host_visible,(unsigned long long)s.bytes.host_only,s.tokens);std::fflush(stdout);
}
} // namespace strata::intel::context_policy
