// Independent Q8-input/Q4_K-affine references. --gpu is explicit; CPU modes
// never create a queue/model. Input precision differs from the FP32 baseline.
#include "strata/intel/decode_dp4a.hpp"
#include "strata/intel/context_policy.hpp"
#include "strata/intel/device.hpp"
#include "strata/intel/lfm_q4_policy.hpp"
#ifdef TINYSTRATA_INTEL_INT8_XMX
#include "strata/intel/q8_xmx.hpp"
#endif
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class T>T read(std::ifstream& f){T value{};need(bool(f.read(reinterpret_cast<char*>(&value),sizeof value)),"Truncated reference");return value;}
template<class T>std::vector<T> array(std::ifstream& f,size_t count){need(count<=64*1024*1024/sizeof(T),"Oversized reference array");std::vector<T> value(count);need(bool(f.read(reinterpret_cast<char*>(value.data()),std::streamsize(count*sizeof(T)))),"Truncated reference array");return value;}
constexpr int extracted[4]={0,63,17,42};
struct Run {int tokens,separate,route,inputs;std::vector<float> x,scales;std::vector<int8_t> codes;std::vector<int32_t> ids;std::vector<double> dots,original;};
struct Case {int rows,width,experts;std::string name;std::vector<uint8_t> raw;std::vector<float> values;std::vector<Run> runs;};
std::vector<Case> load(const char* path){
    std::ifstream f(strata::paths::native_path(path),std::ios::binary);need(bool(f),"Missing independent reference");
    need(read<uint64_t>(f)==0x32513444504d464cULL,"Wrong prefill Q8/Q4 reference magic");const auto count=read<uint32_t>(f);need(count==1,"Use one bounded expert tensor per file");
    std::vector<Case> result;
    for(uint32_t n=0;n<count;++n){Case c;c.rows=int(read<uint32_t>(f));c.width=int(read<uint32_t>(f));c.experts=int(read<uint32_t>(f));
        const auto runs=read<uint32_t>(f);const auto bytes=read<uint64_t>(f);const auto names=read<uint32_t>(f);
        need(c.rows==17&&c.width>=256&&c.width<=8192&&c.width%256==0&&c.experts==64&&runs==24&&names>0&&names<=160,"Invalid prefill case shape");
        need(bytes==uint64_t(c.experts)*c.rows*(c.width/256)*144,"Invalid Q4_K row extent");
        const auto name=array<char>(f,names);c.name.assign(name.begin(),name.end());c.raw=array<uint8_t>(f,size_t(bytes));c.values=array<float>(f,size_t(c.experts==1?1:4)*c.rows*c.width);
        for(uint32_t i=0;i<runs;++i){Run r;r.tokens=int(read<uint32_t>(f));r.separate=int(read<uint32_t>(f));r.route=int(read<uint32_t>(f));r.inputs=int(read<uint32_t>(f));
            need((r.tokens==17||r.tokens==63||r.tokens==104||r.tokens==128)&&r.separate>=0&&r.separate<=1&&r.route>=0&&r.route<3&&r.inputs==r.tokens*(r.separate?4:1),"Invalid prefill input/route shape");
            const size_t elements=size_t(r.inputs)*c.width,outputs=size_t(r.tokens)*(c.experts==1?1:4)*c.rows;
            r.x=array<float>(f,elements);r.codes=array<int8_t>(f,elements);r.scales=array<float>(f,elements/32);r.ids=array<int32_t>(f,c.experts==1?0:size_t(r.tokens)*4);
            r.dots=array<double>(f,outputs);r.original=array<double>(f,outputs);c.runs.push_back(std::move(r));
        }result.push_back(std::move(c));
    }need(f.peek()==std::char_traits<char>::eof(),"Trailing reference data");return result;
}
double half(const uint8_t* p){
    const unsigned bits=unsigned(p[0])|(unsigned(p[1])<<8),exponent=(bits>>10)&31,mantissa=bits&1023;
    need(exponent!=31,"Nonfinite reference Q4 weight");return (bits&32768?-1.:1.)*std::ldexp(double(exponent?1024+mantissa:mantissa),exponent?int(exponent)-25:-24);
}
void scales(const uint8_t* p,int group,int& scale,int& minimum){
    const uint8_t* s=p+4;
    scale=group<4?s[group]&63:(s[group+4]&15)|((s[group-4]>>6)<<4);
    minimum=group<4?s[group+4]&63:(s[group+4]>>4)|((s[group]>>6)<<4);
}
int code(const uint8_t* block,int local){return (block[16+(local/64)*32+local%32]>>((local/32%2)*4))&15;}
uint32_t ulp(float a,float b){
    need(std::isfinite(a)&&std::isfinite(b)&&a>=0&&b>=0,"Invalid nonnegative Q8 scale");
    const auto left=std::bit_cast<uint32_t>(a),right=std::bit_cast<uint32_t>(b);return left>right?left-right:right-left;
}
std::vector<double> affine_reference(const Case& c,const Run& r,const int8_t* codes,const float* dx,std::vector<double>* magnitude=nullptr,std::vector<double>* scale_sum=nullptr){
    const size_t stride=size_t(c.width/256)*144;std::vector<double> result(r.dots.size());
    if(magnitude)magnitude->assign(r.dots.size(),0.);if(scale_sum)scale_sum->assign(r.dots.size(),0.);
    for(int token=0;token<r.tokens;++token)for(int pick=0;pick<(c.experts==1?1:4);++pick){
        const int slot=token*4+pick,expert=c.experts==1?0:r.ids[slot],input=r.separate?slot:token;
        need(c.experts==1||std::find(std::begin(extracted),std::end(extracted),expert)!=std::end(extracted),"Unexpected source expert ID");
        for(int row=0;row<c.rows;++row){const auto* w=c.raw.data()+(size_t(expert)*c.rows+row)*stride;double total=0,pre_absolute=0,dx_sum=0;
            for(int first=0;first<c.width;first+=32){const auto* block=w+(first/256)*144;int sc,mn;scales(block,(first%256)/32,sc,mn);int dot=0,sum=0;
                const double dw=half(block)*sc,minimum=half(block+2)*mn,d=dx[(size_t(input)*c.width+first)/32];
                need(double(float(dw))==dw&&double(float(minimum))==minimum,"FP16 times6bit affine factors must be exact FP32");
                for(int k=0;k<32;k+=4){int dot4=0,sum4=0;
                    for(int j=0;j<4;++j){const int x=codes[size_t(input)*c.width+first+k+j];dot4+=code(block,first%256+k+j)*x;sum4+=x;}
                    need(std::abs(dot4)<=7620&&std::abs(sum4)<=508,"DP4A integer bounds differ");dot+=dot4;sum+=sum4;
                    pre_absolute+=(std::abs(double(dot4)*dw)+std::abs(double(sum4)*minimum))*std::abs(d);dx_sum+=std::abs(d);
                }
                total+=(half(block)*sc*dot-half(block+2)*mn*sum)*dx[(size_t(input)*c.width+first)/32];
            }
            const size_t at=(size_t(token)*(c.experts==1?1:4)+pick)*c.rows+row;result[at]=total;
            if(magnitude)(*magnitude)[at]=pre_absolute;if(scale_sum)(*scale_sum)[at]=dx_sum;
        }
    }return result;
}
double upward(double value){return std::nextafter(value,std::numeric_limits<double>::infinity());}
double gamma(int operations,int bits){const double nu=std::ldexp(double(operations),-bits);return upward(nu/(1.-nu));}
double roundoff_bound(int width,double magnitude,double dx_sum){
    // dot4/sum4 and FP16*6bit factors are exact. Each real contribution passes
    // through <=3 rounded mul/sub/mul operations, <=W/64 lane additions, and
    // the SDK's balanced16-lane reduce_single (4 additions). FMA is covered
    // by this noncontracted upper bound. Magnitude is measured BEFORE affine
    // subtraction, not from the possibly cancelling dequantized result.
    const int terms=width/4;const double gf=gamma(width/64+7,24),gp=gamma(terms+4,53),gr=gamma(width/32+4,53);
    const double denominator=std::nextafter(1.-gp,0.);
    const double positive_upper=upward(magnitude/denominator),dx_upper=upward(dx_sum/denominator);
    const double rounding=upward(upward(gf+gr)*positive_upper);
    // Explicit absolute allowance also covers a flushed/subnormal intermediate:
    //3 pre-scale rounding sites times|dx|, scaled term+accumulator,15 tree adds.
    const double tiny_count=upward(upward(3.*dx_upper)+double(2*terms+15));
    const double tiny=upward(upward(tiny_count*std::numeric_limits<float>::min())*upward(1.+gf));
    return upward(rounding+tiny);
}
float ordered_fp32(const Case& c,const Run& r,const int8_t* codes,const float* dx,size_t output,int mode){
    const int row=int(output%c.rows),slot=int(output/c.rows),token=slot/4,pick=slot%4,expert=r.ids[token*4+pick],input=r.separate?slot:token;
    const auto* w=c.raw.data()+(size_t(expert)*c.rows+row)*size_t(c.width/256)*144;float lanes[16]={};
    for(int base=0;base<c.width;base+=64)for(int lane=0;lane<16;++lane){const int col=base+lane*4;const auto* block=w+(col/256)*144;int sc,mn;scales(block,col%256/32,sc,mn);int dot=0,sum=0;
        for(int j=0;j<4;++j){const int x=codes[size_t(input)*c.width+col+j];dot+=code(block,(col+j)%256)*x;sum+=x;}
        const float dw=float(half(block)*sc),minimum=float(half(block+2)*mn),d=dx[(size_t(input)*c.width+col)/32];
        volatile float a=float(dot)*dw,b=float(sum)*minimum;float difference;
        if(mode%3==0){volatile float sub=a-b;difference=sub;}else if(mode%3==1)difference=std::fma(float(dot),dw,-b);else difference=std::fma(-float(sum),minimum,a);
        if(mode>=3)lanes[lane]=std::fma(difference,d,lanes[lane]);else{volatile float term=difference*d,next=lanes[lane]+term;lanes[lane]=next;}
    }
    for(int n:{8,4,2,1})for(int i=0;i<n;++i){volatile float next=lanes[i]+lanes[i+n];lanes[i]=next;}return lanes[0];
}
void division_proof(const std::vector<Case>& cases){
    size_t groups=0,differences=0,shown=0;uint32_t maximum_ulp=0;
    for(const auto& c:cases)for(const auto& r:c.runs)for(size_t first=0;first<r.x.size();first+=32){
        float maximum=0;for(int k=0;k<32;++k)maximum=std::max(maximum,std::abs(r.x[first+k]));
        volatile float denominator=127.f;const float divided=maximum/denominator,constant=maximum/127.f,reciprocal=maximum*(1.f/127.f),oracle=r.scales[first/32];
        need(ulp(divided,oracle)==0,"Volatile FP32 divide differs from independent NumPy");
        const auto distance=ulp(constant,oracle);maximum_ulp=std::max(maximum_ulp,distance);++groups;
        if(distance){++differences;if(shown++<8)std::printf("DIVISION %s block%zu maximum%.9g fdiv%.9g[%08x] constant%.9g[%08x] reciprocal%.9g[%08x] numpy%.9g[%08x] const_ulp%u\n",c.name.c_str(),first/32,maximum,divided,std::bit_cast<uint32_t>(divided),constant,std::bit_cast<uint32_t>(constant),reciprocal,std::bit_cast<uint32_t>(reciprocal),oracle,std::bit_cast<uint32_t>(oracle),distance);}
    }
    std::printf("PASS CPU division proof groups%zu constant_vs_numpy_differences%zu maximum_ulp%u; independent fdiv remains exact; GPU arithmetic not inferred\n",groups,differences,maximum_ulp);
}
void policy(){
    using namespace strata::intel;using namespace context_policy;
    need(!lfm_q4_candidate_flag(nullptr,"test")&&!lfm_q4_candidate_flag("0","test")&&lfm_q4_candidate_flag("1","test"),"Strict candidate defaults differ");
    for(const char* invalid:{"","true","01","2","1 ","-1"}){bool rejected=false;try{lfm_q4_candidate_flag(invalid,"test");}catch(const std::invalid_argument&){rejected=true;}need(rejected,"Invalid candidate value accepted");}
    for(int type:{0,2,8,10,11,12,14,18,21})for(int tokens:{1,2,8,15,16,17,63,104,128,129,512})
        need(lfm_q4_prefill_dp4a_shape(type,tokens)==(type==12&&tokens>=16&&tokens<=128),"Prefill shape boundary differs");
    need(!DecodeDp4a::supports(12)&&DecodeDp4a::supports(12,true),"Q4 DP4A requires explicit model opt-in");
#ifdef TINYSTRATA_INTEL_INT8_XMX
    need(!q8_xmx_supports(12),"Q4_K INT8 XMX must remain unsupported");
#endif
    for(int chunk:{1,128,512})for(bool mkl:{false,true}){
        LfmSpec s;s.chunk=chunk;s.mkl=mkl;const auto baseline=lfm(32768,s);s.q4_dp4a=true;const auto enabled=lfm(32768,s);
        need(enabled.device-baseline.device==4730004&&enabled.host_visible-baseline.host_visible==4&&enabled.uma_total()-baseline.uma_total()==4730004,"DP4A exact one-pool auxiliary ledger differs");
        need(enabled.owned_device==baseline.owned_device&&enabled.owned_host_visible==baseline.owned_host_visible,"DP4A auxiliary must follow p.own()");
        need(enabled.largest==std::max<uint64_t>(baseline.largest,4194304),"DP4A max allocation ledger differs");
    }
}
void cpu(const Case& c){
    const size_t stride=size_t(c.width/256)*144;
    for(int slot=0;slot<(c.experts==1?1:4);++slot)for(int row=0;row<c.rows;++row){
        const uint8_t* w=c.raw.data()+(size_t(c.experts==1?0:extracted[slot])*c.rows+row)*stride;
        for(int col=0;col<c.width;++col){const uint8_t* block=w+(col/256)*144;int sc,mn;scales(block,(col%256)/32,sc,mn);
            const double actual=half(block)*sc*code(block,col%256)-half(block+2)*mn,expected=c.values[(size_t(slot)*c.rows+row)*c.width+col];
            need(std::abs(actual-expected)<=2e-6*(1+std::abs(expected)),"Independent Q4 nibble/scale/min differs from NumPy");
        }
    }
    for(const auto& r:c.runs){
        for(size_t first=0;first<r.x.size();first+=32){float maximum=0;for(int i=0;i<32;++i){need(std::isfinite(r.x[first+i]),"Nonfinite reference input");maximum=std::max(maximum,std::abs(r.x[first+i]));}
            // A volatile denominator prevents CPU /127 constant-reciprocal
            // folding: the independent NumPy oracle uses rounded FP32 divide.
            volatile float denominator=127.f;const float scale=maximum/denominator;
            if(std::bit_cast<uint32_t>(scale)!=std::bit_cast<uint32_t>(r.scales[first/32])){
                std::fprintf(stderr,"CPU Q8 scale block%zu max%.9g actual%.9g expected%.9g bits%08x/%08x\n",first/32,maximum,scale,r.scales[first/32],std::bit_cast<uint32_t>(scale),std::bit_cast<uint32_t>(r.scales[first/32]));
                throw std::runtime_error("CPU Q8 scale differs from NumPy");}
            for(int i=0;i<32;++i){volatile float ratio=maximum>0?r.x[first+i]/maximum:0.f;volatile float scaled=ratio*127.f;
                need(int(std::nearbyint(scaled))==r.codes[first+i],"CPU Q8 ties-to-even code differs from NumPy");}
        }
        const auto reference=affine_reference(c,r,r.codes.data(),r.scales.data());
        for(size_t i=0;i<reference.size();++i)need(std::isfinite(reference[i])&&std::abs(reference[i]-r.dots[i])<=5e-6*(1+std::abs(r.dots[i])),"Independent Q4 affine CPU sum differs from FP64 oracle");
    }
}
template<class T>struct Buffer {
    sycl::queue& q;T* base;T* p;size_t size;T sentinel;
    Buffer(sycl::queue& queue,size_t n,T s):q(queue),base(sycl::malloc_shared<T>(n+32,queue)),p(base?base+16:nullptr),size(n),sentinel(s){need(base!=nullptr,"Numerical USM allocation failed");std::fill_n(base,n+32,sentinel);}
    ~Buffer(){try{q.wait();sycl::free(base,q);}catch(...){}}
    void guard()const{for(size_t i=0;i<16;++i)need(base[i]==sentinel&&p[size+i]==sentinel,"Q8/Q4 input/output/weight outer canary changed");}
};
void gpu(sycl::queue& q,strata::intel::DecodeDp4a& dp,const Case& c){
    using namespace strata::intel;
    Buffer<uint8_t> raw(q,c.raw.size(),0xa5);std::copy(c.raw.begin(),c.raw.end(),raw.p);
    int comparisons=0;double max_error=0,max_precision_change=0,max_scale_effect=0,max_bound_ratio=0,max_single_difference=0;size_t single_comparisons=0,single_bit_differences=0;uint32_t max_scale_ulp=0;
    for(const auto& r:c.runs){
        Buffer<float> x(q,r.x.size(),123456.f),sc(q,r.scales.size()+16,123456.f),out(q,r.dots.size()+64,123456.f);
        Buffer<int8_t> codes(q,r.codes.size()+64,-111);Buffer<int32_t> ids(q,std::max<size_t>(r.ids.size(),1)+16,-777),error(q,1,-999);
        double shape_error=0,shape_ratio=0,shape_min_bound=std::numeric_limits<double>::infinity(),shape_max_bound=0,shape_original_difference=0,shape_scale_effect=0,shape_single_difference=0;
        float shape_min_scale=std::numeric_limits<float>::infinity(),shape_max_scale=0;uint32_t shape_scale_ulp=0;size_t shape_single_comparisons=0,shape_single_bit_differences=0;
        std::copy(r.x.begin(),r.x.end(),x.p);std::copy(r.ids.begin(),r.ids.end(),ids.p);error.p[0]=0;
        quantize_decode_input(q,x.p,codes.p,sc.p,c.width,r.inputs,error.p).wait_and_throw();
        need(error.p[0]==0,"Finite Q8 input marked invalid");
        need(std::memcmp(codes.p,r.codes.data(),r.codes.size())==0,"GPU Q8 code differs from independent NumPy");
        // Device /127 may be contracted to FP32 reciprocal multiplication.
        // Codes are exact. Scales admit <=1ULP versus rounded NumPy divide;
        // zero stays exact. Weight math uses actual independently checked Q8.
        for(size_t i=0;i<r.scales.size();++i){const auto distance=ulp(sc.p[i],r.scales[i]);max_scale_ulp=std::max(max_scale_ulp,distance);shape_scale_ulp=std::max(shape_scale_ulp,distance);shape_min_scale=std::min(shape_min_scale,sc.p[i]);shape_max_scale=std::max(shape_max_scale,sc.p[i]);
            need(distance<=(r.scales[i]==0?0u:1u),"GPU Q8 scale exceeds documented 1ULP independent divide bound");}
        std::vector<double> magnitudes,dx_sums;const auto actual_reference=affine_reference(c,r,codes.p,sc.p,&magnitudes,&dx_sums);
        for(size_t i=r.codes.size();i<codes.size;++i)need(codes.p[i]==-111,"Q8 quantizer overwrote unused code tail");
        for(size_t i=r.scales.size();i<sc.size;++i)need(sc.p[i]==123456.f,"Q8 quantizer overwrote unused scale tail");
        auto check=[&]{
            need(dp.valid(),"Finite affine input set sticky error");
            for(size_t i=0;i<r.dots.size();++i){const double expected=actual_reference[i],e=std::abs(double(out.p[i])-expected),bound=roundoff_bound(c.width,magnitudes[i],dx_sums[i]);max_error=std::max(max_error,e);max_precision_change=std::max(max_precision_change,std::abs(expected-r.original[i]));max_scale_effect=std::max(max_scale_effect,std::abs(expected-r.dots[i]));max_bound_ratio=std::max(max_bound_ratio,e/bound);
                shape_error=std::max(shape_error,e);shape_ratio=std::max(shape_ratio,e/bound);shape_min_bound=std::min(shape_min_bound,bound);shape_max_bound=std::max(shape_max_bound,bound);shape_original_difference=std::max(shape_original_difference,std::abs(expected-r.original[i]));shape_scale_effect=std::max(shape_scale_effect,std::abs(expected-r.dots[i]));
                if(!std::isfinite(out.p[i])||!std::isfinite(bound)||e>bound){std::fprintf(stderr,"%s T%d separate%d route%d idx%zu actual%.9g expected%.17g error%.9g theory_bound%.9g pre_subtraction_abs%.17g gamma_n%d\n",c.name.c_str(),r.tokens,r.separate,r.route,i,double(out.p[i]),expected,e,bound,magnitudes[i],c.width/64+7);throw std::runtime_error("Q4 DP4A exceeds independently derived affine FP32 rounding bound");}}
            for(size_t i=r.dots.size();i<out.size;++i)need(out.p[i]==123456.f,"DP4A overwrote inactive output tail");
            need(std::memcmp(x.p,r.x.data(),r.x.size()*sizeof(float))==0&&std::memcmp(raw.p,c.raw.data(),c.raw.size())==0,"Read-only FP32 input/GGUF bytes changed");
            need(std::equal(r.ids.begin(),r.ids.end(),ids.p),"Selected expert IDs changed");
            for(size_t i=r.ids.size();i<ids.size;++i)need(ids.p[i]==-777,"Unused expert-ID tail changed");
            raw.guard();x.guard();codes.guard();sc.guard();out.guard();ids.guard();error.guard();++comparisons;
        };
        Buffer<int32_t> counts(q,64,-777),slots(q,size_t(64)*r.tokens,-777),owners(q,size_t(4)*r.tokens+64,-777),offsets(q,size_t(4)*r.tokens+64,-777),tiles(q,1,-777);
        ExpertPlan plan{counts.p,slots.p,owners.p,offsets.p,tiles.p,64,r.tokens,4,4};
        plan_experts(q,ids.p,plan);dp.grouped(12,raw.p,plan,x.p,out.p,c.rows,c.width,r.separate!=0).wait_and_throw();check();
        const bool original_failure=c.name=="synthetic-all-groups-64"&&r.tokens==104&&r.separate==1&&r.route==2;
        if(original_failure){const size_t at=4444;std::printf("DIAG original_failure output4444 slot261 token65 pick1 expert%d row7 input261 grouped%.17g[%08x] actual_q8_fp64%.17g pre_subtraction_abs%.17g gamma_n%d bound%.17g\n",r.ids[261],double(out.p[at]),std::bit_cast<uint32_t>(out.p[at]),actual_reference[at],magnitudes[at],c.width/64+7,roundoff_bound(c.width,magnitudes[at],dx_sums[at]));
            for(int mode=0;mode<6;++mode){const float value=ordered_fp32(c,r,codes.p,sc.p,at,mode);std::printf("DIAG actual_q8_cpu_order mode%d value%.17g bits%08x matches_grouped%d\n",mode,double(value),std::bit_cast<uint32_t>(value),int(std::bit_cast<uint32_t>(value)==std::bit_cast<uint32_t>(out.p[at])));}
            for(int group=0;group<c.width/32;++group)std::printf("DIAG actual_q8_scale group%d bits%08x value%.17g\n",group,std::bit_cast<uint32_t>(sc.p[size_t(261)*c.width/32+group]),double(sc.p[size_t(261)*c.width/32+group]));
        }
        // Compare unchanged single-token selected() against grouped() at the
        // first/middle/last tokens, plus the retained failing token65. Bounds
        // are applied independently to each, and every bit difference reported.
        Buffer<float> single(q,4*c.rows+64,123456.f);std::vector<int> selected_tokens={0,r.tokens/2,r.tokens-1};if(original_failure)selected_tokens.push_back(65);
        for(int token:selected_tokens){std::fill_n(single.p,single.size,123456.f);
            dp.selected(12,raw.p,ids.p+token*4,4,x.p+size_t(token)*(r.separate?4:1)*c.width,single.p,c.rows,c.width,r.separate!=0,1).wait_and_throw();need(dp.valid(),"Single-token reference input set error");
            for(int i=0;i<4*c.rows;++i){const size_t at=size_t(token)*4*c.rows+i;const double expected=actual_reference[at],bound=roundoff_bound(c.width,magnitudes[at],dx_sums[at]),difference=std::abs(double(single.p[i])-out.p[at]);
                need(std::isfinite(single.p[i])&&std::abs(double(single.p[i])-expected)<=bound&&difference<=2*bound,"Single-token/grouped affine routing or rounding differs");
                max_single_difference=std::max(max_single_difference,difference);single_bit_differences+=std::bit_cast<uint32_t>(single.p[i])!=std::bit_cast<uint32_t>(out.p[at]);++single_comparisons;
                shape_single_difference=std::max(shape_single_difference,difference);shape_single_bit_differences+=std::bit_cast<uint32_t>(single.p[i])!=std::bit_cast<uint32_t>(out.p[at]);++shape_single_comparisons;
                if(original_failure&&token==65&&i==24)std::printf("DIAG original_failure_single value%.17g[%08x] grouped%.17g[%08x]\n",double(single.p[i]),std::bit_cast<uint32_t>(single.p[i]),double(out.p[at]),std::bit_cast<uint32_t>(out.p[at]));
            }
            for(size_t i=4*c.rows;i<single.size;++i)need(single.p[i]==123456.f,"Single-token reference overwrote inactive output");single.guard();
        }
        size_t used_tiles=0;
        for(int expert=0;expert<64;++expert){std::vector<int> expected;
            for(size_t entry=0;entry<r.ids.size();++entry)if(r.ids[entry]==expert)expected.push_back(int(entry));
            need(counts.p[expert]==int(expected.size()),"Expert plan count differs");
            for(int at=0;at<r.tokens;++at)need(slots.p[size_t(expert)*r.tokens+at]==(at<int(expected.size())?expected[at]:-777),"Expert plan slot/unused canary differs");
            for(int offset=0;offset<int(expected.size());offset+=4){need(owners.p[used_tiles]==expert&&offsets.p[used_tiles]==offset,"Expert plan owner/offset differs");++used_tiles;}
        }
        need(tiles.p[0]==int(used_tiles),"Expert plan tile count differs");
        for(size_t i=used_tiles;i<owners.size;++i)need(owners.p[i]==-777&&offsets.p[i]==-777,"Expert plan inactive owner/offset canary changed");
        counts.guard();slots.guard();owners.guard();offsets.guard();tiles.guard();
        std::printf("{\"shape\":true,\"tensor\":\"%s\",\"width\":%d,\"tokens\":%d,\"separate\":%d,\"route\":%d,\"tile_width\":4,\"q8_codes_exact\":true,\"maximum_scale_ulp\":%u,\"minimum_actual_scale\":%.12g,\"maximum_actual_scale\":%.12g,\"gamma_n\":%d,\"gamma_fp32\":%.17g,\"minimum_theory_bound\":%.12g,\"maximum_theory_bound\":%.12g,\"maximum_actual_q8_fp64_error\":%.12g,\"maximum_error_over_theory_bound\":%.12g,\"maximum_q8_vs_original_fp32_difference\":%.12g,\"maximum_scale_reference_effect\":%.12g,\"single_comparisons\":%zu,\"single_bit_differences\":%zu,\"maximum_single_difference\":%.12g,\"readonly_canaries\":true}\n",c.name.c_str(),c.width,r.tokens,r.separate,r.route,shape_scale_ulp,double(shape_min_scale),double(shape_max_scale),c.width/64+7,gamma(c.width/64+7,24),shape_min_bound,shape_max_bound,shape_error,shape_ratio,shape_original_difference,shape_scale_effect,shape_single_comparisons,shape_single_bit_differences,shape_single_difference);std::fflush(stdout);
    }
    std::printf("PASS GPU Q8-affine %s runs%zu comparisons%d max_fp64_error%.9g max_q8_vs_fp32_dot_change%.9g max_scale_ulp%u max_scale_reference_effect%.9g max_theory_bound_ratio%.9g single_comparisons%zu single_bit_differences%zu max_single_difference%.9g exact_q8_codes all-canary\n",c.name.c_str(),c.runs.size(),comparisons,max_error,max_precision_change,max_scale_ulp,max_scale_effect,max_bound_ratio,single_comparisons,single_bit_differences,max_single_difference);
}
void sticky(sycl::queue& q,strata::intel::DecodeDp4a& dp,const Case& c){
    using namespace strata::intel;constexpr int tokens=17,picks=4;const size_t inputs=size_t(tokens)*picks*c.width,outputs=size_t(tokens)*picks*c.rows;
    Buffer<uint8_t> raw(q,c.raw.size(),0xa5);std::copy(c.raw.begin(),c.raw.end(),raw.p);Buffer<float> x(q,inputs,123456.f),y(q,outputs+64,123456.f),sc(q,inputs/32+16,123456.f);
    Buffer<int8_t> codes(q,inputs+64,-111);Buffer<int32_t> ids(q,tokens*picks,-777),error(q,1,-777),counts(q,64,-777),slots(q,64*tokens,-777),owners(q,tokens*picks+64,-777),offsets(q,tokens*picks+64,-777),tiles(q,1,-777);
    for(int i=0;i<tokens*picks;++i)ids.p[i]=extracted[i%4];std::fill_n(x.p,inputs,0.f);x.p[inputs-1]=std::numeric_limits<float>::quiet_NaN();error.p[0]=0;
    quantize_decode_input(q,x.p,codes.p,sc.p,c.width,tokens*picks,error.p).wait_and_throw();need(error.p[0]==1,"Separate last Q8 group did not set error");error.guard();
    ExpertPlan plan{counts.p,slots.p,owners.p,offsets.p,tiles.p,64,tokens,picks,4};plan_experts(q,ids.p,plan);
    dp.grouped(12,raw.p,plan,x.p,y.p,c.rows,c.width,true).wait_and_throw();need(!dp.valid(),"Nonfinite grouped input did not set sticky error");
    x.p[inputs-1]=0;dp.grouped(12,raw.p,plan,x.p,y.p,c.rows,c.width,true).wait_and_throw();need(!dp.valid(),"Finite retry must not silently clear sticky input error");
    for(size_t i=0;i<outputs;++i)need(y.p[i]==0.f,"Zeroed nonfinite/finite-retry Q8 output differs");
    for(size_t i=outputs;i<y.size;++i)need(y.p[i]==123456.f,"Grouped error/retry overwrote unused output");
    for(size_t i=inputs;i<codes.size;++i)need(codes.p[i]==-111,"Nonfinite quantizer overwrote unused code tail");
    for(size_t i=inputs/32;i<sc.size;++i)need(sc.p[i]==123456.f,"Nonfinite quantizer overwrote unused scale tail");
    raw.guard();x.guard();y.guard();codes.guard();sc.guard();ids.guard();counts.guard();slots.guard();owners.guard();offsets.guard();tiles.guard();std::puts("PASS sticky nonfinite/finite-retry error gate");
}
}
int main(int argc,char** argv){try{
    strata::paths::Utf8Arguments utf8(argc,argv);argc=utf8.argc();argv=utf8.argv();need(std::fesetround(FE_TONEAREST)==0,"CPU nearest-even rounding unavailable");policy();
    if(argc==2&&std::string(argv[1])=="--policy-only"){std::puts("PASS Q4 DP4A policy/4730004-byte aux; no queue/model");return 0;}
    need(argc==3&&(std::string(argv[1])=="--cpu"||std::string(argv[1])=="--gpu"||std::string(argv[1])=="--division-proof"),"Usage: --policy-only | --division-proof FIXTURE | --cpu FIXTURE | --gpu FIXTURE (explicit GPU)");
    const auto cases=load(argv[2]);if(std::string(argv[1])=="--division-proof"){division_proof(cases);return 0;}for(const auto& c:cases)cpu(c);
    if(std::string(argv[1])=="--gpu"){sycl::queue q(strata::intel::select_device(0),sycl::property::queue::in_order{});strata::intel::DecodeDp4a dp(q,false,false);q.wait_and_throw();need(dp.valid(),"Initial error not cleared");for(const auto& c:cases)gpu(q,dp,c);sticky(q,dp,cases.front());}
    std::printf("PASS %zu independent Q8/Q4-affine cases (%s); FP32 input precision is changed; no model engine\n",cases.size(),argv[1]+2);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"LFM Q4 DP4A: %s\n",e.what());return 1;}}
