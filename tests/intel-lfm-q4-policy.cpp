// CPU-only integration policy/ledger gate; no queue/model/device selection.
#include "strata/intel/lfm_q4_policy.hpp"
#include "strata/intel/lfm_ops.hpp"
#include "strata/intel/decode_dp4a.hpp"
#include "strata/intel/context_policy.hpp"
#include <cstdio>
#include <limits>
namespace si=strata::intel;
namespace cp=si::context_policy;
namespace{int checks=0;void need(bool value){++checks;if(!value)throw std::runtime_error("LFM Q4 integration policy check "+std::to_string(checks));}}
int main(){try{
    for(int type:{8,10,11,14,18,21,22,23})need(si::DecodeDp4a::supports(type)&&si::DecodeDp4a::supports(type,true));
    need(!si::DecodeDp4a::supports(12)&&!si::DecodeDp4a::supports(12,false)&&si::DecodeDp4a::supports(12,true));
    for(int type:{-1,0,2,99})need(!si::DecodeDp4a::supports(type)&&!si::DecodeDp4a::supports(type,true));
    need(!si::lfm_q4_candidate_flag(nullptr,"test"));need(!si::lfm_q4_candidate_flag("0","test"));need(si::lfm_q4_candidate_flag("1","test"));
    for(const char* bad:{"","2","01","true","1 ","-1"}){bool rejected=false;try{si::lfm_q4_candidate_flag(bad,"test");}catch(const std::invalid_argument&){rejected=true;}need(rejected);}
    for(bool fp32:{false,true})for(bool decode:{false,true}){bool rejected=false;try{si::lfm_q4_validate_modes(fp32,decode);}catch(const std::invalid_argument&){rejected=true;}need(rejected==(fp32&&decode));}
    for(int type:{0,2,8,10,11,12,14,18,21,22})for(int tokens:{1,2,8,15,16,17,63,104,128,129,512}){
        const bool fast=si::lfm_q4_fp32_eligible(type,tokens,true),prefill=si::lfm_q4_prefill_dp4a_shape(type,tokens);
        need(fast==(type==12&&tokens==1));need(prefill==(type==12&&tokens>=16&&tokens<=128));need(!(fast&&prefill));
        need(!si::lfm_q4_fp32_eligible(type,tokens,false));
    }
    for(uint64_t chunk:{1,128,512})for(bool mkl:{false,true})for(bool decode:{false,true})for(bool prefill:{false,true}){
        cp::LfmSpec spec;spec.chunk=chunk;spec.mkl=mkl;const auto base=cp::lfm(32768,spec);spec.q4_dp4a=decode||prefill;const auto candidate=cp::lfm(32768,spec);const uint64_t expected=decode||prefill?4730004:0;
        need(candidate.device-base.device==expected);need(candidate.host_visible-base.host_visible==(expected?4:0));need(candidate.uma_total()-base.uma_total()==expected);
        need(candidate.owned_device==base.owned_device&&candidate.owned_host_visible==base.owned_host_visible);
        need(candidate.largest==(expected?std::max<uint64_t>(base.largest,4194304):base.largest));
    }
    std::printf("PASS LFM Q4 integration CPU %d checks: FP32 T1/prefill16..128 disjoint; strict flags/conflicts; one auxiliary ledger; no queue/model\n",checks);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"LFM Q4 policy: %s\n",error.what());return 1;}}
