#include "strata/intel/fp8_codec.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
namespace f8=strata::intel::fp8;
namespace {
uint64_t checks=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("FP8 codec check "+std::to_string(checks));}
// Independent numeric reference, deliberately without the production bit codec.
double value(unsigned code){
    const unsigned magnitude=code&127,e=magnitude/8,m=magnitude%8;
    double x=e?std::ldexp(1.+double(m)/8,int(e)-7):std::ldexp(double(m),-9);
    if(magnitude==127)x=std::numeric_limits<double>::quiet_NaN();
    return code&128?-x:x;
}
uint8_t reference(float x){
    const bool negative=std::signbit(x);const double absolute=std::abs(double(x));
    if(std::isnan(x))return uint8_t((negative?128:0)|127);
    if(absolute>=448)return uint8_t((negative?128:0)|126);
    unsigned selected=0;double distance=absolute;
    for(unsigned code=1;code<=126;++code){
        const double d=std::abs(value(code)-absolute);
        if(d<distance||(d==distance&&!(code&1))){selected=code;distance=d;}
    }
    return uint8_t((negative?128:0)|selected);
}
void compare(float x){check(f8::encode(x)==reference(x));}
}
int main()try{
    for(unsigned code=0;code<256;++code){
        const float actual=f8::decode(uint8_t(code));const double wanted=value(code);
        check(std::isnan(wanted)?std::isnan(actual):double(actual)==wanted);
        if(!std::isnan(wanted)){check(f8::encode(actual)==code);check(std::signbit(actual)==bool(code&128));}
    }
    for(unsigned code=0;code<126;++code){
        const float midpoint=float((value(code)+value(code+1))/2);
        for(float x:{std::nextafter(midpoint,-INFINITY),midpoint,std::nextafter(midpoint,INFINITY)}){compare(x);compare(-x);}
    }
    for(float x:{0.f,-0.f,0x1p-149f,-0x1p-149f,0x1p-24f,0x1p-10f,0x1p-9f,0x1p-6f,
        240.f,256.f,432.f,448.f,464.f,1e30f,INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()})compare(x);
    uint32_t state=0x12345678;
    for(int i=0;i<100000;++i){state=1664525u*state+1013904223u;compare(std::bit_cast<float>(state));}
    std::printf("FP8 E4M3FN codec PASS checks=%llu all256codes midpoint_RNE subnormal saturation random_float32 software_codec=1\n",(unsigned long long)checks);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
