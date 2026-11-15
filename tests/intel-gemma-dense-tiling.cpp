// Model-free independent test of the bounded oneMKL output-row tiling needed
// by Gemma's 2816 -> 8192 projection. GPU execution is explicit, never CTest.
#include "strata/intel/device.hpp"
#include "strata/intel/prefill_gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
template<class T>struct Buffer {
    sycl::queue& q;T* p;size_t n;
    Buffer(sycl::queue& queue,size_t count,bool shared=true):q(queue),p(shared?sycl::malloc_shared<T>(count,queue):sycl::malloc_device<T>(count,queue)),n(count){if(!p)throw std::runtime_error("Dense tiling test allocation failed");}
    ~Buffer(){sycl::free(p,q);}
};
void run(sycl::queue& q,int rows,int width,int tokens){
    const size_t stride=size_t(width/32)*18;
    Buffer<uint8_t> weights(q,size_t(rows)*stride);
    Buffer<float> input(q,size_t(tokens)*width),output(q,size_t(tokens)*rows+37);
    for(int row=0;row<rows;++row)for(int b=0;b<width/32;++b){
        auto* block=weights.p+size_t(row)*stride+b*18;
        const sycl::half d=sycl::half(float((row+b)%17-8)/128.f);const uint16_t bits=sycl::bit_cast<uint16_t>(d);
        std::memcpy(block,&bits,2);
        for(int j=0;j<16;++j)block[j+2]=uint8_t(((row+b*7+j)%16)|(((row*3+j*5+b)%16)<<4));
    }
    for(size_t i=0;i<input.n;++i)input.p[i]=float(std::sin(double(i)*.0131)+.3*std::cos(double(i)*.071));
    std::fill_n(output.p,size_t(tokens)*rows,NAN);std::fill(output.p+size_t(tokens)*rows,output.p+output.n,123456.f);
    strata::intel::PrefillGemm gemm(q,1,tokens);
    gemm.dense(2,weights.p,input.p,output.p,rows,width,tokens).wait_and_throw();
    for(size_t i=0;i<size_t(tokens)*rows;++i)if(!std::isfinite(output.p[i]))throw std::runtime_error("Unwritten/nonfinite tiled GEMM output");
    for(size_t i=size_t(tokens)*rows;i<output.n;++i)if(output.p[i]!=123456.f)throw std::runtime_error("Tiled GEMM output canary changed");
    const int split=int((8192*2048)/width);
    std::set<int> samples{0,1,rows/2,rows-1};
    for(int row:{split-1,split,split+1})if(row>=0&&row<rows)samples.insert(row);
    for(int i=0;i<37;++i)samples.insert((i*197+13)%rows);
    double maximum=0;
    for(int token=0;token<tokens;++token)for(int row:samples){
        double expected=0;
        for(int col=0;col<width;++col){
            const uint8_t* block=weights.p+size_t(row)*stride+(col/32)*18;
            uint16_t bits;std::memcpy(&bits,block,2);const float scale=float(sycl::bit_cast<sycl::half>(bits));
            const int j=col%32,code=int((block[2+j%16]>>(j>=16?4:0))&15)-8;
            // Independent byte decode, then exactly the two FP16 operands
            // contracted by the declared GEMM, accumulated here in FP64.
            const float w=float(sycl::half(scale*float(code)));
            const float x=float(sycl::half(input.p[size_t(token)*width+col]));
            expected+=double(w)*double(x);
        }
        const double actual=output.p[size_t(token)*rows+row],error=std::abs(actual-expected);
        maximum=std::max(maximum,error);
        if(error>2e-4*(1+std::abs(expected))){std::fprintf(stderr,"rowtile mismatch T%d R%d K%d token%d row%d actual%.9g expected%.12g\n",tokens,rows,width,token,row,actual,expected);throw std::runtime_error("Tiled dense GEMM differs from independent reference");}
    }
    std::printf("PASS Gemma dense tiling rows=%d width=%d tokens=%d row_tile=%d sampled_values=%zu max_abs=%.9g canary=passed\n",rows,width,tokens,split,samples.size()*tokens,maximum);
}
void selected_usm(sycl::queue& q,bool device){
    constexpr int experts=7,picks=2,tokens=17,width=704,rows=17;
    constexpr int control_guard=0x3f4a5b6c;constexpr size_t float_guard=16,tail=37;
    const size_t stride=size_t(width/32)*18,expert_bytes=stride*rows,weight_bytes=expert_bytes*experts;
    const size_t input_values=size_t(tokens)*picks*width,output_values=size_t(tokens)*picks*rows;
    const size_t tile_capacity=(tokens*picks+3)/4+experts;
    Buffer<uint8_t> weights(q,16+weight_bytes+17,!device);
    Buffer<float> input(q,float_guard+input_values+tail,!device),output(q,float_guard+output_values+tail,!device);
    Buffer<int32_t> ids(q,tokens*picks+2,!device),counts(q,experts+2,!device),slots(q,experts*tokens+2,!device);
    Buffer<int32_t> owners(q,tile_capacity+2,!device),offsets(q,tile_capacity+2,!device),tiles(q,3,!device);
    std::vector<uint8_t> raw(weights.n,0xa7);
    for(int expert=0;expert<experts;++expert)for(int row=0;row<rows;++row)for(int block=0;block<width/32;++block){
        uint8_t* b=raw.data()+16+size_t(expert)*expert_bytes+row*stride+block*18;
        const sycl::half d=sycl::half(float((expert+row+block)%17-8)/128.f);const uint16_t bits=sycl::bit_cast<uint16_t>(d);std::memcpy(b,&bits,2);
        for(int j=0;j<16;++j)b[j+2]=uint8_t(((expert+row+block*7+j)%16)|(((expert*3+row+j*5+block)%16)<<4));
    }
    std::vector<float> hx(input.n,123456.f),hy(output.n,123456.f);
    for(size_t i=0;i<input_values;++i)hx[float_guard+i]=float(std::sin(double(i)*.0131)+.3*std::cos(double(i)*.071));
    for(size_t i=0;i<output_values;++i)hy[float_guard+i]=NAN;
    std::vector<int32_t> hi(ids.n,control_guard);
    for(int token=0;token<tokens;++token)for(int pick=0;pick<picks;++pick)hi[1+token*picks+pick]=(token+2*pick)%experts;
    q.memcpy(weights.p,raw.data(),raw.size());q.memcpy(input.p,hx.data(),hx.size()*sizeof(float));q.memcpy(ids.p,hi.data(),hi.size()*sizeof(int32_t));
    auto init_control=[&](Buffer<int32_t>& b){std::vector<int32_t> h(b.n,control_guard);q.memcpy(b.p,h.data(),h.size()*sizeof(int32_t)).wait_and_throw();};
    init_control(counts);init_control(slots);init_control(owners);init_control(offsets);init_control(tiles);
    strata::intel::ExpertPlan plan{counts.p+1,slots.p+1,owners.p+1,offsets.p+1,tiles.p+1,experts,tokens,picks};
    strata::intel::plan_experts(q,ids.p+1,plan).wait_and_throw();
    strata::intel::PrefillGemm gemm(q,1,tokens);
    for(bool separate:{false,true}){
        q.memcpy(output.p,hy.data(),hy.size()*sizeof(float));
        gemm.selected(2,weights.p+16,expert_bytes,plan,input.p+float_guard,output.p+float_guard,rows,width,separate).wait_and_throw();
        std::vector<float> got(output.n);q.memcpy(got.data(),output.p,got.size()*sizeof(float)).wait_and_throw();
        for(size_t i=0;i<float_guard;++i)if(got[i]!=123456.f)throw std::runtime_error("Selected USM output prefix changed");
        for(size_t i=float_guard+output_values;i<got.size();++i)if(got[i]!=123456.f)throw std::runtime_error("Selected USM output suffix changed");
        double maximum=0;
        for(int slot=0;slot<tokens*picks;++slot)for(int row=0;row<rows;++row){
            const int expert=hi[1+slot];double expected=0;
            for(int col=0;col<width;++col){
                const uint8_t* b=raw.data()+16+size_t(expert)*expert_bytes+row*stride+(col/32)*18;
                uint16_t bits;std::memcpy(&bits,b,2);const float d=float(sycl::bit_cast<sycl::half>(bits));
                const int code=int((b[2+col%16]>>(col%32>=16?4:0))&15)-8;
                const float w=float(sycl::half(d*code));const int src=separate?slot:slot/picks;
                expected+=double(w)*float(sycl::half(hx[float_guard+size_t(src)*width+col]));
            }
            const float actual=got[float_guard+size_t(slot)*rows+row];const double error=std::abs(actual-expected);maximum=std::max(maximum,error);
            if(!std::isfinite(actual)||error>2e-4*(1+std::abs(expected)))throw std::runtime_error("Selected device/shared-USM FP64 mismatch");
        }
        std::printf("PASS Gemma selected all_%s_USM %s values=%zu max_abs=%.9g output_guards=passed\n",device?"device":"shared",separate?"down":"gate",output_values,maximum);
    }
    for(auto* b:{&ids,&counts,&slots,&owners,&offsets,&tiles}){
        std::vector<int32_t> h(b->n);q.memcpy(h.data(),b->p,h.size()*sizeof(int32_t)).wait_and_throw();
        if(h.front()!=control_guard||h.back()!=control_guard)throw std::runtime_error("Selected USM control canary changed");
        if(b==&counts){int sum=0;for(int i=1;i<=experts;++i){if(h[i]<0||h[i]>tokens)throw std::runtime_error("Invalid selected count bytes");sum+=h[i];}if(sum!=tokens*picks)throw std::runtime_error("Selected count sum differs");}
    }
    std::vector<uint8_t> after_weights(weights.n);std::vector<float> after_input(input.n);
    q.memcpy(after_weights.data(),weights.p,after_weights.size());q.memcpy(after_input.data(),input.p,after_input.size()*sizeof(float)).wait_and_throw();
    if(after_weights!=raw||after_input!=hx)throw std::runtime_error("Selected USM read-only input/weight changed");
}
}
int main(){
    try{sycl::queue q(strata::intel::select_device(0),sycl::property::queue::in_order{});
        run(q,8192,2816,17);run(q,2816,8192,17);run(q,2112,704,17);selected_usm(q,false);selected_usm(q,true);std::puts("PASS 3 Gemma dense tiling shapes and 4 shared/device-USM expert comparisons");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Gemma dense tiling: %s\n",e.what());return 1;}
}
