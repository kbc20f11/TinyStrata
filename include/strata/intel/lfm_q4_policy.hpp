#pragma once
#include <stdexcept>
#include <string>

namespace strata::intel {
inline bool lfm_q4_candidate_flag(const char* value,const char* name){
    if(!value)return false;
    if(value[0]=='0'&&value[1]=='\0')return false;
    if(value[0]=='1'&&value[1]=='\0')return true;
    throw std::invalid_argument(std::string(name)+" requires0 or1");
}
constexpr bool lfm_q4_prefill_dp4a_shape(int type,int tokens)noexcept{
    return type==12&&tokens>=16&&tokens<=128;
}
inline void lfm_q4_validate_modes(bool fp32,bool decode_dp4a){
    if(fp32&&decode_dp4a)throw std::invalid_argument("LFM FP32 decode and Q8 decode candidates are mutually exclusive; explicitly disable FP32 for the Q8 decode experiment");
}
}
