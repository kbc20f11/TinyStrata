#pragma once
#include <bit>
#include <cstdint>

namespace strata::intel::fp8 {
// E4M3FN: bias7, finite maximum448, signed zero, and the two signed NaN
// codes0x7f/0xff. This is a software bit codec, not an FP8 matrix instruction.
inline constexpr float maximum=448.f;
inline float decode(uint8_t code) {
    const uint32_t sign=uint32_t(code&0x80)<<24;
    const uint32_t magnitude=code&0x7f,exponent=magnitude>>3,mantissa=magnitude&7;
    if(!exponent){
        const float value=float(mantissa)*(1.f/512.f);
        return code&0x80 ? -value : value;
    }
    if(magnitude==0x7f)return std::bit_cast<float>(sign|0x7fc00000u);
    return std::bit_cast<float>(sign|((exponent+120u)<<23)|(mantissa<<20));
}
// Saturating round-to-nearest-even. NaNs stay NaN; infinities saturate. KV
// storage separately rejects nonfinite inputs and zeroes the whole32-cell block.
inline uint8_t encode(float value) {
    const uint32_t bits=std::bit_cast<uint32_t>(value),absolute=bits&0x7fffffffu;
    const uint8_t sign=uint8_t((bits>>24)&0x80u);
    if(absolute>0x7f800000u)return uint8_t(sign|0x7f);
    if(absolute>=0x43e00000u)return uint8_t(sign|0x7e); //448, including saturation
    if(absolute<0x3c800000u){ // subnormal FP8 output, below2^-6
        const uint32_t exponent=absolute>>23;
        if(exponent<117)return sign; // below the halfway point2^-10
        const uint32_t mantissa=(absolute&0x7fffffu)|0x800000u,shift=141u-exponent;
        uint32_t rounded=mantissa>>shift;
        const uint32_t remainder=mantissa&((1u<<shift)-1),halfway=1u<<(shift-1);
        if(remainder>halfway||(remainder==halfway&&(rounded&1)))++rounded;
        return uint8_t(sign|rounded);
    }
    // Keep three significand bits; carry naturally updates the exponent.
    const uint32_t rounded=absolute+0x7ffffu+((absolute>>20)&1u);
    return uint8_t(sign|uint8_t((rounded>>20)-960u));
}
}
