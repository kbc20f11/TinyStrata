#include "strata/intel/context_policy.hpp"
#include <cstdio>
#include <stdexcept>
namespace cp=strata::intel::context_policy;
namespace si=strata::intel;
namespace v=strata::intel::qwen_vision;
static int checks=0;
static void check(bool x){++checks;if(!x)throw std::runtime_error("Clover workspace ledger check "+std::to_string(checks));}
int main()try{
    for(bool clover:{false,true})for(bool vision:{false,true})for(int mtp:{0,1,3})
        check(v::clover_workspace_aliases(clover,vision,mtp)==(clover&&vision&&mtp==0));
    for(uint64_t chunk:{1,7,128,512,1024})for(uint64_t context:{2048,4096,131072})for(bool device:{false,true}){
        cp::QwenSpec original;original.chunk=chunk;original.int8=true;original.vision=true;original.device_state=device;
        auto alias=original;alias.clover_vision_aliases=true;
        const auto a=cp::qwen(context,original),b=cp::qwen(context,alias);
        // Four independently counted dead sources: expert hidden, shared
        // hidden, GDN normalized output, and causal convolution output.
        const uint64_t saved=(8*512+512+32*128+(2*16+32)*128)*chunk*sizeof(float);
        check(a.owned_device-b.owned_device==saved);check(a.device-b.device==saved);
        check(a.owned_host_visible-b.owned_host_visible==(device?0:saved));
        check(a.host_visible-b.host_visible==(device?0:saved));
        check(a.cache==b.cache&&a.host_only==b.host_only&&a.largest==b.largest);
        check(saved==(chunk==512?33ull<<20:chunk*67584));
    }
    for(bool vision:{false,true})for(bool mtp:{false,true})if(!vision||mtp){
        cp::QwenSpec bad;bad.clover_vision_aliases=true;bad.vision=vision;bad.mtp=mtp;
        bool rejected=false;try{(void)cp::qwen(4096,bad);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    }
    cp::QwenSpec original;original.chunk=512;original.vision=true;original.int8=true;
    original.device_state=true;original.mkl=true;original.dp4a=true;original.dense_split=true;
    auto alias=original;alias.clover_vision_aliases=true;
    const auto initial=cp::qwen(2048,original);
    si::MemoryBudget memory;memory.integrated=true;memory.wddm_known=true;
    memory.wddm_available=initial.device+si::uma_wddm_headroom+(8ull<<20);
    memory.available_physical=memory.available_commit=initial.uma_total()+4*si::GiB+si::GiB;
    const auto a=cp::select(0,131072,memory,16*si::GiB,[&](uint64_t c){return cp::qwen(c,original);});
    const auto b=cp::select(0,131072,memory,16*si::GiB,[&](uint64_t c){return cp::qwen(c,alias);});
    check(a.tokens==2048&&b.tokens==4096);
    check(memory.wddm_available-b.bytes.device>=si::uma_wddm_headroom);
    check(memory.available_physical-b.bytes.uma_total()>=4*si::GiB);
    check(cp::qwen(4096,original).device-initial.device==36986880);
    std::printf("Clover image workspace CPU checks=%d PASS saved_chunk512=%llu baseline_context=%d aliased_context=%d unchanged_WDDM_reserve=%llu\n",
        checks,(unsigned long long)(initial.device-cp::qwen(2048,alias).device),a.tokens,b.tokens,(unsigned long long)si::uma_wddm_headroom);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
