#include "strata/intel/device.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/decode_dp4a.hpp"
#include "strata/intel/kv_q8.hpp"
#include "strata/intel/q6_interleave.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include "strata/intel/prefill_gemm.hpp"
#endif
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fstream>

namespace {
template<class T> struct Buffer {
    sycl::queue& q;
    T* data;
    size_t size;
    Buffer(sycl::queue& queue, size_t count) : q(queue), data(sycl::malloc_shared<T>(count, queue)), size(count) {
        if (!data) throw std::runtime_error("USM test allocation failed");
    }
    ~Buffer() { sycl::free(data, q); }
    Buffer(const Buffer&) = delete;
    T& operator[](size_t index) { return data[index]; }
};
void compare(const char* name, const float* actual, const std::vector<double>& expected, double tolerance) {
    double maximum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double difference = std::abs(double(actual[i]) - expected[i]);
        if (!std::isfinite(actual[i]) || difference > tolerance * (1 + std::abs(expected[i]))) {
            std::fprintf(stderr, "%s index=%zu actual=%.9g expected=%.12g difference=%.9g\n", name, i, actual[i], expected[i], difference);
            throw std::runtime_error("Intel numerical reference mismatch");
        }
        maximum = std::max(maximum, difference);
    }
    std::printf("%s values=%zu max_abs_error=%.9g result=passed\n", name, expected.size(), maximum);
}
constexpr float output_canary = 123456.f;
void prepare_output(Buffer<float>& output, size_t active) {
    if (active > output.size) throw std::runtime_error("Numerical output exceeds its allocation");
    std::fill_n(output.data, active, NAN);
    std::fill(output.data + active, output.data + output.size, output_canary);
}
void check_output_tail(const Buffer<float>& output, size_t active) {
    for (size_t i = active; i < output.size; ++i)
        if (output.data[i] != output_canary) throw std::runtime_error("Kernel overwrote inactive numerical output");
}
void norm_test(sycl::queue& q) {
    constexpr int width = 2048, rows = 3;
    Buffer<float> x(q, width * rows), w(q, width), y(q, width * rows);
    for (int c = 0; c < width; ++c) {
        w[c] = 0.9f + float(c % 17) / 100;
        x[c] = 0;
        x[width + c] = (c % 2 ? -0.01f : 0.01f);
        x[2 * width + c] = float(std::sin(c * .17));
    }
    std::vector<double> expected(width * rows);
    const float epsilon = 1e-6f;
    for (int row = 0; row < rows; ++row) {
        double square_sum = 0;
        for (int col = 0; col < width; ++col) square_sum += double(x[row * width + col]) * x[row * width + col];
        for (int col = 0; col < width; ++col)
            expected[row * width + col] = x[row * width + col] * w[col] / std::sqrt(square_sum / width + epsilon);
    }
    strata::intel::rms_norm(q, x.data, w.data, y.data, rows, width, epsilon).wait_and_throw();
    compare("rms_fp64", y.data, expected, 2e-6);
}
void activation_test(sycl::queue& q) {
    constexpr int width = 256, rows = 8192, count = width * rows;
    Buffer<float> x(q, count), scales(q, count / 32);
    Buffer<int8_t> values(q, count);
    Buffer<int32_t> error(q, 1);
    error[0] = 0;
    for (int i = 0; i < count; ++i) x[i] = i < 32 ? 0.f : i < 64 ? float((i % 32) - 16) * 1e-38f : float(std::sin(i * .17) * (1 + i % 7));
    strata::intel::quantize_decode_input(q, x.data, values.data, scales.data, width, rows, error.data).wait_and_throw();
    if (error[0]) throw std::runtime_error("Finite activation incorrectly rejected");
    double maximum = 0;
    for (int b = 0; b < count / 32; ++b) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(x[b * 32 + j]));
        if (std::abs(scales[b] - amax / 127.f) > 2e-7f * amax) throw std::runtime_error("Q8 activation scale mismatch");
        for (int j = 0; j < 32; ++j) {
            const size_t i = b * 32 + j;
            const double difference = std::abs(double(x[i]) - double(scales[b]) * values[i]);
            if (values[i] < -127 || difference > double(scales[b]) * .5001 + double(amax) * 1e-6 + 1e-44)
                throw std::runtime_error("Q8 activation exceeds half-step error bound");
            maximum = std::max(maximum, difference);
        }
    }
    std::printf("q8_activation_error_bound values=%d max_abs_error=%.9g result=passed\n", count, maximum);
    x[count - 7] = INFINITY;
    strata::intel::quantize_decode_input(q, x.data, values.data, scales.data, width, rows, error.data).wait_and_throw();
    if (error[0] != 1) throw std::runtime_error("Non-finite activation was silently accepted");
}
void q8_kv_test(sycl::queue& q) {
    constexpr int context=19,kv=2,heads=16,width=256,count=context*kv*width;
    Buffer<float> key(q,count),value(q,count),expanded_k(q,count),expanded_v(q,count);
    Buffer<int8_t> codes_k(q,count),codes_v(q,count);
    Buffer<sycl::half> scales_k(q,count/32),scales_v(q,count/32);
    Buffer<int32_t> position(q,1),error(q,1);
    strata::intel::Q8Kv cache{codes_k.data,codes_v.data,scales_k.data,scales_v.data,context,kv};
    for(int i=0;i<count;++i){key[i]=i<32?0.f:float(std::sin(i*.17)*(1+(i/32)%7));value[i]=float(std::cos(i*.23)*(1+(i/32)%5));}
    position[0]=0;error[0]=0;
    strata::intel::store_q8_kv(q,key.data,value.data,cache,position.data,context,error.data).wait_and_throw();
    if(error[0])throw std::runtime_error("Finite Q8 KV input rejected");
    auto check_row=[&](int cell,int head,int source){
        for(int b=0;b<8;++b)for(int kind=0;kind<2;++kind){
            float maximum=0;const float* input=kind?value.data:key.data;const int8_t* codes=kind?codes_v.data:codes_k.data;
            const sycl::half* scales=kind?scales_v.data:scales_k.data;
            for(int j=0;j<32;++j)maximum=std::max(maximum,std::abs(input[size_t(source)*width+b*32+j]));
            const float d=maximum/127.f;const size_t target=(size_t(head)*context+cell)*width+b*32;
            if(float(scales[target/32])!=float(sycl::half(d)))throw std::runtime_error("Q8 KV scale/layout mismatch");
            for(int j=0;j<32;++j){
                const double difference=std::abs(double(input[size_t(source)*width+b*32+j])-double(codes[target+j])*float(scales[target/32]));
                const double bound=double(d)*.5001+std::abs(double(d)-float(scales[target/32]))*127+double(maximum)*1e-6+1e-44;
                if(codes[target+j]<-127||difference>bound)throw std::runtime_error("Q8 KV exceeds independent quantization bound");
            }
        }
    };
    for(int t=0;t<context;++t)for(int h=0;h<kv;++h)check_row(t,h,t*kv+h);
    std::vector<int8_t> old_k(codes_k.data,codes_k.data+count),old_v(codes_v.data,codes_v.data+count);
    std::vector<sycl::half> old_ks(scales_k.data,scales_k.data+count/32),old_vs(scales_v.data,scales_v.data+count/32);
    for(int i=0;i<3*kv*width;++i){key[i]=float(std::cos(i*.37)*3);value[i]=float(std::sin(i*.29)*5);}
    position[0]=5;
    strata::intel::store_q8_kv(q,key.data,value.data,cache,position.data,3,error.data).wait_and_throw();
    for(int t=0;t<context;++t)for(int h=0;h<kv;++h){
        if(t>=5&&t<8)check_row(t,h,(t-5)*kv+h);
        else for(int c=0;c<width;++c){
            const size_t i=(size_t(h)*context+t)*width+c;
            if(codes_k[i]!=old_k[i]||codes_v[i]!=old_v[i]||scales_k[i/32]!=old_ks[i/32]||scales_v[i/32]!=old_vs[i/32])
                throw std::runtime_error("Q8 KV partial store changed another cell/head");
        }
    }
    std::vector<double> expected_k(count),expected_v(count);
    for(int i=0;i<count;++i){expected_k[i]=double(codes_k[i])*float(scales_k[i/32]);expected_v[i]=double(codes_v[i])*float(scales_v[i/32]);}
    strata::intel::dequantize_q8_kv(q,cache,expanded_k.data,expanded_v.data,context).wait_and_throw();
    compare("q8_kv_dequant_key",expanded_k.data,expected_k,1e-7);compare("q8_kv_dequant_value",expanded_v.data,expected_v,1e-7);
    Buffer<float> query(q,heads*width),gate(q,heads*width),output(q,heads*width),scores(q,heads*context),partial(q,heads*width);
    for(int i=0;i<heads*width;++i){query[i]=float(std::sin(i*.13));gate[i]=float(std::cos(i*.07));}
    for(int p:{0,7,18}){
        position[0]=p;std::vector<double> expected(heads*width,0);
        for(int h=0;h<heads;++h){
            const int kh=h/(heads/kv);std::vector<double> logits(p+1,0);
            for(int t=0;t<=p;++t)for(int c=0;c<width;++c)logits[t]+=double(query[h*width+c])*expected_k[(kh*context+t)*width+c]/16.;
            const double maximum=*std::max_element(logits.begin(),logits.end());double sum=0;
            for(auto& x:logits)sum+=x=std::exp(x-maximum);
            for(int c=0;c<width;++c){
                double result=0;for(int t=0;t<=p;++t)result+=logits[t]/sum*expected_v[(kh*context+t)*width+c];
                expected[h*width+c]=result/(1+std::exp(-double(gate[h*width+c])));
            }
        }
        strata::intel::decode_attention_q8(q,query.data,gate.data,cache,scores.data,partial.data,output.data,heads,position.data,context).wait_and_throw();
        compare(("q8_kv_attention_fp64_position_"+std::to_string(p)).c_str(),output.data,expected,2e-5);
    }
    key[35]=INFINITY;position[0]=0;
    strata::intel::store_q8_kv(q,key.data,value.data,cache,position.data,1,error.data).wait_and_throw();
    if(!error[0])throw std::runtime_error("Non-finite Q8 KV input silently accepted");
    std::puts("q8_kv_quantization_bound_partial_store_nonfinite passed");
}
void q8_attention_batch_test(sycl::queue& q) {
    constexpr int context=1031,heads=16,kv=2,width=256,capacity=8,splits=(context+255)/256;
    const size_t cache_size=size_t(kv)*context*width;
    Buffer<int8_t> keys(q,cache_size),values(q,cache_size);
    Buffer<sycl::half> key_scales(q,cache_size/32),value_scales(q,cache_size/32);
    Buffer<float> query(q,capacity*heads*width),gate(q,capacity*heads*width),output(q,capacity*heads*width);
    Buffer<float> scores(q,capacity*heads*context),partial(q,capacity*heads*splits*width);
    Buffer<int32_t> positions(q,capacity);
    const int cells[capacity]={0,255,256,511,512,768,1023,1030};
    for(size_t i=0;i<cache_size;++i){keys[i]=int8_t(int((i*17+3)%255)-127);values[i]=int8_t(int((i*29+11)%255)-127);}
    for(size_t i=0;i<cache_size/32;++i){key_scales[i]=sycl::half(float(i%11+1)/1024);value_scales[i]=sycl::half(float(i%7+1)/256);}
    for(int i=0;i<capacity*heads*width;++i){query[i]=float(std::sin(i*.13));gate[i]=float(std::cos(i*.07));}
    for(int i=0;i<capacity;++i)positions[i]=cells[i];
    const strata::intel::Q8Kv cache{keys.data,values.data,key_scales.data,value_scales.data,context,kv};
    for(int tokens:{1,3,4,7,8}){
        std::fill(scores.data,scores.data+capacity*heads*context,NAN);
        std::fill(partial.data,partial.data+capacity*heads*splits*width,NAN);
        std::vector<double> expected(size_t(tokens)*heads*width,0.);
        for(int token=0;token<tokens;++token)for(int head=0;head<heads;++head){
            const size_t row=size_t(token)*heads+head;const int kh=head/(heads/kv),end=positions[token];
            std::vector<double> probabilities(size_t(end)+1,0.);
            for(int cell=0;cell<=end;++cell)for(int col=0;col<width;++col){
                const size_t i=(size_t(kh)*context+cell)*width+col;
                probabilities[cell]+=double(query[row*width+col])*double(keys[i])*float(key_scales[i/32])/16.;
            }
            const double maximum=*std::max_element(probabilities.begin(),probabilities.end());double total=0.;
            for(auto& value:probabilities)total+=value=std::exp(value-maximum);
            for(int col=0;col<width;++col){
                double sum=0.;
                for(int cell=0;cell<=end;++cell){
                    const size_t i=(size_t(kh)*context+cell)*width+col;
                    sum+=probabilities[cell]/total*double(values[i])*float(value_scales[i/32]);
                }
                expected[row*width+col]=sum/(1.+std::exp(-double(gate[row*width+col])));
            }
        }
        strata::intel::decode_attention_q8(q,query.data,gate.data,cache,scores.data,partial.data,output.data,
            heads,positions.data,context,nullptr,tokens).wait_and_throw();
        compare(("q8_attention_batch_fp64_"+std::to_string(tokens)).c_str(),output.data,expected,2e-5);
    }
}
void q8_attention_long_reuse_test(sycl::queue& q) {
    constexpr int context=32768,heads=16,kv=2,width=256,capacity=4,splits=(context+255)/256;
    const size_t cells=size_t(kv)*context*width;
    Buffer<int8_t> keys(q,cells),values(q,cells);
    Buffer<sycl::half> ks(q,cells/32),vs(q,cells/32);
    Buffer<float> query(q,capacity*heads*width),gate(q,capacity*heads*width);
    Buffer<float> old_y(q,capacity*heads*width+64),new_y(q,old_y.size);
    Buffer<float> old_scores(q,capacity*heads*context+64),new_scores(q,old_scores.size);
    Buffer<float> old_partial(q,capacity*heads*splits*width+64),new_partial(q,old_partial.size);
    Buffer<int32_t> positions(q,capacity);
    for(size_t i=0;i<cells;++i){keys[i]=int8_t(int((i*17+3)%255)-127);values[i]=int8_t(int((i*29+11)%255)-127);}
    for(size_t i=0;i<cells/32;++i){ks[i]=sycl::half(float(i%11+1)/1024);vs[i]=sycl::half(float(i%7+1)/256);}
    for(size_t i=0;i<query.size;++i){query[i]=float(std::sin(i*.13));gate[i]=float(std::cos(i*.07));}
    const strata::intel::Q8Kv cache{keys.data,values.data,ks.data,vs.data,context,kv};
    for(int visible:{16384,16416,16512})for(int tokens:{1,2,3,4}) {
        for(int t=0;t<capacity;++t)positions[t]=t<tokens?visible-tokens+t:-1;
        const size_t active=size_t(tokens)*heads*width;
        prepare_output(old_y,active);prepare_output(new_y,active);
        for(auto* buffer:{&old_scores,&new_scores,&old_partial,&new_partial})std::fill_n(buffer->data,buffer->size,output_canary);
        // An omitted/zero hint is the exact old path, also used by recorded
        // graphs. The positive hint is supplied from CPU-known positions.
        strata::intel::decode_attention_q8(q,query.data,gate.data,cache,old_scores.data,old_partial.data,
            old_y.data,heads,positions.data,visible,nullptr,tokens).wait_and_throw();
        strata::intel::decode_attention_q8(q,query.data,gate.data,cache,new_scores.data,new_partial.data,
            new_y.data,heads,positions.data,visible,nullptr,tokens,visible).wait_and_throw();
        for(const auto pair:{std::pair{&old_y,&new_y},std::pair{&old_scores,&new_scores},std::pair{&old_partial,&new_partial}})
            if(std::memcmp(pair.first->data,pair.second->data,pair.first->size*sizeof(float)))
                throw std::runtime_error("Long Q8 reuse differs from the original output/scores/partial bits");
        check_output_tail(old_y,active);check_output_tail(new_y,active);
        for(size_t row=0;row<size_t(capacity)*heads;++row)for(int cell=0;cell<context;++cell)
            if((row>=size_t(tokens)*heads || cell>positions[row/heads]) && new_scores[row*context+cell]!=output_canary)
                throw std::runtime_error("Long Q8 reuse wrote a future/inactive score");
        for(size_t row=0;row<size_t(capacity)*heads;++row)for(int split=0;split<splits;++split)
            if(row>=size_t(tokens)*heads || split>=(visible+255)/256)for(int col=0;col<width;++col)
                if(new_partial[(row*splits+split)*width+col]!=output_canary)
                    throw std::runtime_error("Long Q8 reuse wrote an inactive partial");
        for(size_t i=size_t(capacity)*heads*context;i<new_scores.size;++i)
            if(new_scores[i]!=output_canary)throw std::runtime_error("Long Q8 scores tail changed");
        for(size_t i=size_t(capacity)*heads*splits*width;i<new_partial.size;++i)
            if(new_partial[i]!=output_canary)throw std::runtime_error("Long Q8 partial tail changed");
        if(tokens>=3) {
            std::vector<double> expected(active,0.);
            for(int token=0;token<tokens;++token)for(int head=0;head<heads;++head) {
                const size_t row=size_t(token)*heads+head;const int kh=head/(heads/kv),end=positions[token];
                std::vector<double> probabilities(size_t(end)+1,0.);
                for(int cell=0;cell<=end;++cell)for(int col=0;col<width;++col) {
                    const size_t i=(size_t(kh)*context+cell)*width+col;
                    probabilities[cell]+=double(query[row*width+col])*double(keys[i])*float(ks[i/32])/16.;
                }
                const double maximum=*std::max_element(probabilities.begin(),probabilities.end());double total=0;
                for(auto& probability:probabilities)total+=probability=std::exp(probability-maximum);
                for(int col=0;col<width;++col) {
                    double sum=0;
                    for(int cell=0;cell<=end;++cell) {
                        const size_t i=(size_t(kh)*context+cell)*width+col;
                        sum+=probabilities[cell]/total*double(values[i])*float(vs[i/32]);
                    }
                    expected[row*width+col]=sum/(1.+std::exp(-double(gate[row*width+col])));
                }
            }
            compare(("q8_long_reuse_fp64_"+std::to_string(visible)+"_T"+std::to_string(tokens)).c_str(),new_y.data,expected,2e-5);
        }
        std::printf("q8_long_reuse_bit_and_all_inactive_canaries visible=%d tokens=%d passed\n",visible,tokens);
    }
    bool refused=false;
    try{strata::intel::decode_attention_q8(q,query.data,gate.data,cache,new_scores.data,new_partial.data,
        new_y.data,heads,positions.data,16384,nullptr,4,16385);}
    catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("Q8 reuse accepted actual visibility beyond its extent");
}
#ifdef TINYSTRATA_INTEL_INT8_XMX
void splitk_capacity_test(sycl::queue& q) {
    constexpr int rows=8192,width=256,tokens=8,block_bytes=210;
    Buffer<uint8_t> original(q,size_t(rows)*block_bytes),packed(q,strata::intel::q6_interleaved_bytes(rows,width));
    // Every coefficient is exactly 0.125: low=8, high=2, signed code=8,
    // scale=1, delta=1/64. This reference does not use the GPU decoder.
    const uint16_t delta=sycl::bit_cast<uint16_t>(sycl::half(1.f/64.f));
    for(int row=0;row<rows;++row){
        auto* block=original.data+size_t(row)*block_bytes;
        std::fill_n(block,128,uint8_t(0x88));std::fill_n(block+128,64,uint8_t(0xaa));
        std::fill_n(block+192,16,uint8_t(1));block[208]=uint8_t(delta);block[209]=uint8_t(delta>>8);
    }
    strata::intel::interleave_q6(original.data,packed.data,rows,width);
    Buffer<float> input(q,tokens*width),scales(q,tokens*width/32),output(q,size_t(tokens)*rows+16);
    Buffer<int8_t> codes(q,tokens*width);
    for(int i=0;i<tokens*width;++i)input[i]=float(std::sin(i*.13)*.5);
    strata::intel::quantize_decode_input(q,input.data,codes.data,scales.data,width,tokens).wait_and_throw();
    std::vector<double> expected(size_t(tokens)*rows);
    for(int token=0;token<tokens;++token){
        double dot=0.;for(int col=0;col<width;++col)dot+=.125*double(codes[token*width+col])*scales[token*(width/32)+col/32];
        std::fill_n(expected.begin()+size_t(token)*rows,rows,dot);
    }
    std::fill_n(output.data+size_t(tokens)*rows,16,12345.f);
    strata::intel::DecodeDp4a product(q,true,true);
    product.dense_interleaved_q6(packed.data,input.data,output.data,rows,width,tokens,true).wait_and_throw();
    compare("splitk_exact_1mib_capacity_fp64",output.data,expected,5e-5);
    for(int i=0;i<16;++i)if(output[size_t(tokens)*rows+i]!=12345.f)throw std::runtime_error("Split-K overwrote the output tail");
    bool refused=false;
    try{product.dense_interleaved_q6(packed.data,input.data,output.data,rows+16,width,tokens,true);}
    catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("Oversized split-K scratch was not refused");
}
#endif
#ifdef TINYSTRATA_INTEL_INT8_XMX
void selected_vocabulary_capacity_test(sycl::queue& q) {
    constexpr int rows=4112,width=2048,count=4096,blocks=width/256;
    Buffer<uint8_t> original(q,size_t(rows)*blocks*210),packed(q,strata::intel::q6_interleaved_bytes(rows,width));
    const uint16_t delta=sycl::bit_cast<uint16_t>(sycl::half(1.f/64.f));
    for(int row=0;row<rows;++row)for(int block=0;block<blocks;++block){
        auto* bytes=original.data+(size_t(row)*blocks+block)*210;
        std::fill_n(bytes,128,uint8_t(0x88));std::fill_n(bytes+128,64,uint8_t(0xaa));
        std::fill_n(bytes+192,16,uint8_t(1));bytes[208]=uint8_t(delta);bytes[209]=uint8_t(delta>>8);
    }
    strata::intel::interleave_q6(original.data,packed.data,rows,width);
    Buffer<float> input(q,width),scales(q,width/32),output(q,count+16);
    Buffer<int8_t> codes(q,width);Buffer<int32_t> ids(q,count);
    for(int col=0;col<width;++col)input[col]=float(std::sin(col*.13)*.5);
    for(int row=0;row<count;++row)ids[row]=row+1==count?rows-1:row;
    std::fill_n(output.data,count+16,12345.f);
    strata::intel::quantize_decode_input(q,input.data,codes.data,scales.data,width,1).wait_and_throw();
    double dot=0.;for(int col=0;col<width;++col)dot+=.125*double(codes[col])*scales[col/32];
    strata::intel::DecodeDp4a product(q,true);
    product.selected_vocabulary_q6(packed.data,input.data,output.data,rows,width,ids.data,count).wait_and_throw();
    compare("selected_vocabulary_4096_capacity_fp64",output.data,std::vector<double>(count,dot),5e-5);
    for(int row=count;row<count+16;++row)if(output[row]!=12345.f)throw std::runtime_error("Selected vocabulary overwrote the output tail");
    bool refused=false;
    try{product.selected_vocabulary_q6(packed.data,input.data,output.data,rows,width,ids.data,count+1);}
    catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("Oversized selected vocabulary was not refused");
}
#endif
void matvec_test(sycl::queue& q) {
    constexpr int width = 2048, rows = 257;
    Buffer<strata::intel::Q8Block> w(q, size_t(rows) * width / 32);
    Buffer<float> x(q, width), y(q, rows);
    for (int col = 0; col < width; ++col) x[col] = float(std::sin(col * .123));
    for (size_t block = 0; block < size_t(rows) * width / 32; ++block) {
        w[block].scale = sycl::half(float(block % 7 + 1) / 128);
        for (int value = 0; value < 32; ++value) w[block].values[value] = int8_t(int((block * 13 + value * 17) % 255) - 127);
    }
    std::vector<double> expected(rows, 0);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < width; ++col) {
            const auto& b = w[size_t(row) * width / 32 + col / 32];
            expected[row] += double(float(b.scale)) * b.values[col % 32] * x[col];
        }
    strata::intel::q8_matvec(q, w.data, x.data, y.data, rows, width).wait_and_throw();
    compare("q8_0_matvec_fp64", y.data, expected, 2e-5);
}
void attention_test(sycl::queue& q) {
    constexpr int heads = 16, kv_heads = 2, context = 37, tokens = 7, width = 256;
    Buffer<float> query(q, tokens * heads * width), gate(q, tokens * heads * width), output(q, tokens * heads * width);
    Buffer<float> key(q, kv_heads * context * width), value(q, kv_heads * context * width);
    Buffer<float> decode_scores(q, heads * context), decode_partial(q, heads * ((context + 255) / 256) * width);
    Buffer<int32_t> positions(q, tokens);
#ifdef TINYSTRATA_INTEL_MKL
    strata::intel::PrefillGemm gemm(q, context, 64);
#endif
    for (size_t i = 0; i < size_t(tokens) * heads * width; ++i) { query[i] = float(std::sin(i * .13)); gate[i] = float(std::cos(i * .07)); }
    for (size_t i = 0; i < size_t(kv_heads) * context * width; ++i) { key[i] = float(std::sin(i * .17)); value[i] = float(std::cos(i * .23)); }
    for (int position : {0, 13, 30}) {
        for (size_t i = 0; i < size_t(kv_heads) * context * width; ++i) { key[i] = float(std::sin(i * .17)); value[i] = float(std::cos(i * .23)); }
        std::vector<double> expected(tokens * heads * width, 0);
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
            const int kv = h / (heads / kv_heads), visible = position + t + 1;
            std::vector<double> scores(visible, 0);
            for (int cell = 0; cell < visible; ++cell) for (int c = 0; c < width; ++c)
                scores[cell] += double(query[(t * heads + h) * width + c]) * key[(kv * context + cell) * width + c] / 16.;
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (auto& score : scores) sum += score = std::exp(score - maximum);
            for (int c = 0; c < width; ++c) {
                double result = 0;
                for (int cell = 0; cell < visible; ++cell) result += scores[cell] / sum * value[(kv * context + cell) * width + c];
                expected[(t * heads + h) * width + c] = result / (1 + std::exp(-double(gate[(t * heads + h) * width + c])));
            }
        }
        strata::intel::causal_attention(q, query.data, gate.data, key.data, value.data, output.data,
            heads, kv_heads, context, position, tokens).wait_and_throw();
        compare(("causal_attention_fp64_position_" + std::to_string(position)).c_str(), output.data, expected, 2e-5);
        for (int t = 0; t < tokens; ++t) positions[t] = position + t;
        for (int t = 0; t < 4; ++t) {
            const size_t offset = size_t(t) * heads * width;
            strata::intel::decode_attention(q, query.data + offset, gate.data + offset, key.data, value.data,
                decode_scores.data, decode_partial.data, output.data + offset, heads, kv_heads, context,
                positions.data + t, position + t + 1);
        }
        q.wait_and_throw();
        compare(("batched_decode_attention_fp64_position_" + std::to_string(position)).c_str(), output.data,
            std::vector<double>(expected.begin(), expected.begin() + 4 * heads * width), 2e-5);
#ifdef TINYSTRATA_INTEL_MKL
        gemm.attention(query.data, gate.data, key.data, value.data, output.data,
            heads, kv_heads, context, position, tokens).wait_and_throw();
        compare(("mkl_attention_fp64_position_" + std::to_string(position)).c_str(), output.data, expected, 2e-5);
        for (int count : {1, 4, 7}) {
            for (size_t i = 0; i < size_t(kv_heads) * context * width; ++i) {
                const bool future = (i / width) % context >= size_t(position + count);
                key[i] = future ? 0.f : float(std::sin(i * .17));
                value[i] = future ? 0.f : float(std::cos(i * .23));
            }
            gemm.attention_small(query.data, gate.data, key.data, value.data, output.data,
                heads, kv_heads, context, position, count).wait_and_throw();
            compare(("mkl_grouped_attention_fp64_position_" + std::to_string(position) + "_tokens_" + std::to_string(count)).c_str(), output.data,
                std::vector<double>(expected.begin(), expected.begin() + count * heads * width), 2e-5);
        }
#endif
    }
}
void decode_attention_test(sycl::queue& q) {
    constexpr int heads = 16, kv_heads = 2, context = 1025, width = 256;
    Buffer<float> query(q, heads * width), gate(q, heads * width), output(q, heads * width), scores(q, heads * context);
    Buffer<float> key(q, kv_heads * context * width), value(q, kv_heads * context * width);
    Buffer<float> partial(q, heads * ((context + 255) / 256) * width);
    Buffer<int32_t> position(q, 1);
    for (int i = 0; i < heads * width; ++i) { query[i] = float(std::sin(i * .13)); gate[i] = float(std::cos(i * .07)); }
    for (int i = 0; i < kv_heads * context * width; ++i) { key[i] = float(std::sin(i * .17)); value[i] = float(std::cos(i * .23)); }
    for (int p : {0, 256, 1024}) {
        position[0] = p;
        std::vector<double> expected(heads * width, 0);
        for (int h = 0; h < heads; ++h) {
            const int kv = h / (heads / kv_heads);
            std::vector<double> weights(p + 1, 0);
            for (int cell = 0; cell <= p; ++cell) for (int c = 0; c < width; ++c)
                weights[cell] += double(query[h * width + c]) * key[(kv * context + cell) * width + c] / 16.;
            const double maximum = *std::max_element(weights.begin(), weights.end());
            double sum = 0;
            for (auto& x : weights) sum += x = std::exp(x - maximum);
            for (int c = 0; c < width; ++c) {
                double result = 0;
                for (int cell = 0; cell <= p; ++cell) result += weights[cell] / sum * value[(kv * context + cell) * width + c];
                expected[h * width + c] = result / (1 + std::exp(-double(gate[h * width + c])));
            }
        }
        // Exercise graph-like padded extents, including a partial final block.
        strata::intel::decode_attention(q, query.data, gate.data, key.data, value.data, scores.data, partial.data,
            output.data, heads, kv_heads, context, position.data, context).wait_and_throw();
        compare(("decode_attention_fp64_position_" + std::to_string(p)).c_str(), output.data, expected, 2e-5);
    }
}
void recurrent_test(sycl::queue& q, int tokens) {
    constexpr int hk = 16, hv = 32, width = 128;
    constexpr size_t channels = (2 * hk + hv) * width, out_width = hv * width;
    constexpr size_t state_size = width * hv * width;
    Buffer<float> state(q, state_size), input(q, tokens * channels), gate(q, tokens * hv), beta(q, tokens * hv);
    Buffer<float> output(q, tokens * out_width);
    std::vector<float> initial(state_size);
    std::vector<double> reference(state_size), expected(tokens * out_width);
    for (size_t i = 0; i < state_size; ++i) reference[i] = initial[i] = state[i] = float(std::sin(i * .17) * .01);
    for (size_t i = 0; i < tokens * channels; ++i) input[i] = float(std::sin(i * .13) / std::sqrt(128.));
    for (int i = 0; i < tokens * hv; ++i) { gate[i] = -.01f * (1 + i % 13); beta[i] = .1f + .01f * (i % 17); }
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < hv; ++head) {
            const size_t q_head = head % hk;
            const size_t base = size_t(token) * channels;
            const double decay = std::exp(double(gate[token * hv + head]));
            for (int col = 0; col < width; ++col) {
                double prediction = 0;
                for (int row = 0; row < width; ++row)
                    prediction += reference[(size_t(row) * hv + head) * width + col] * input[base + (hk + q_head) * width + row];
                const double delta = (input[base + 2 * hk * width + head * width + col] - decay * prediction) * beta[token * hv + head];
                double value = 0;
                for (int row = 0; row < width; ++row) {
                    auto& element = reference[(size_t(row) * hv + head) * width + col];
                    element = decay * element + input[base + (hk + q_head) * width + row] * delta;
                    value += element * input[base + q_head * width + row];
                }
                expected[size_t(token) * out_width + head * width + col] = value / std::sqrt(128.);
            }
        }
    }
    strata::intel::gdn_recurrence(q, state.data, input.data, gate.data, beta.data, output.data, hk, hv, tokens).wait_and_throw();
    compare("gdn_outputs_fp64", output.data, expected, 2e-6);
    compare("gdn_state_fp64", state.data, reference, 2e-6);
    // A split prompt must preserve state and use the correct token offset.
    q.memcpy(state.data, initial.data(), state_size * sizeof(float)).wait_and_throw();
    strata::intel::gdn_recurrence(q, state.data, input.data, gate.data, beta.data, output.data, hk, hv, 2);
    strata::intel::gdn_recurrence(q, state.data, input.data + 2 * channels, gate.data + 2 * hv,
        beta.data + 2 * hv, output.data + 2 * out_width, hk, hv, tokens - 2).wait_and_throw();
    compare("gdn_split_outputs_fp64", output.data, expected, 2e-6);
    compare("gdn_split_state_fp64", state.data, reference, 2e-6);
    // Restore the original state as conversation reset would do.
    q.memcpy(state.data, initial.data(), state_size * sizeof(float)).wait_and_throw();
    strata::intel::gdn_recurrence(q, state.data, input.data, gate.data, beta.data, output.data, hk, hv, tokens).wait_and_throw();
    compare("gdn_restored_outputs_fp64", output.data, expected, 2e-6);
    q.memcpy(state.data, initial.data(), state_size * sizeof(float)).wait_and_throw();
    strata::intel::gdn_recurrence_tiled(q, state.data, input.data, gate.data, beta.data, output.data, hk, hv, tokens).wait_and_throw();
    compare("gdn_tiled_outputs_fp64", output.data, expected, 2e-6);
    compare("gdn_tiled_state_fp64", state.data, reference, 2e-6);
    q.memcpy(state.data, initial.data(), state_size * sizeof(float)).wait_and_throw();
    strata::intel::gdn_recurrence_tiled(q, state.data, input.data, gate.data, beta.data, output.data, hk, hv, 2);
    strata::intel::gdn_recurrence_tiled(q, state.data, input.data + 2 * channels, gate.data + 2 * hv,
        beta.data + 2 * hv, output.data + 2 * out_width, hk, hv, tokens - 2).wait_and_throw();
    compare("gdn_tiled_split_outputs_fp64", output.data, expected, 2e-6);
    compare("gdn_tiled_split_state_fp64", state.data, reference, 2e-6);
}
template<class T> T read(std::ifstream& file) {
    T value{};
    if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("Truncated quantization fixture");
    return value;
}
void read_bytes(std::ifstream& file, void* destination, size_t bytes) {
    if (!file.read(static_cast<char*>(destination), std::streamsize(bytes))) throw std::runtime_error("Truncated quantization fixture data");
}
// Actual IQ3_XXS rows are supplied by gguf's independent NumPy decoder. Rotate
// those unchanged compressed rows across eleven experts so every request still
// selects all eight distinct experts, with varied/clustered/common route plans.
void type18_routing_test(sycl::queue& q, const uint8_t* raw, size_t bytes, int rows, int width,
    const std::vector<float>& expected_weights, const std::string& label, strata::intel::DecodeDp4a& dp4a
#ifdef TINYSTRATA_INTEL_INT8_XMX
    , strata::intel::DecodeDp4a& int8_xmx
#endif
#ifdef TINYSTRATA_INTEL_MKL
    , strata::intel::PrefillGemm& gemm
#endif
    ) {
    constexpr int tokens = 8, picks = 8, experts = 11;
    const int padded = (rows + 15) / 16 * 16;
    const size_t stride = bytes / rows, packed_bytes = strata::intel::quant_interleaved_bytes(padded, width, 98);
    if (bytes != size_t(rows) * (width / 256) * 98) throw std::runtime_error("Invalid Type18 fixture stride");
    Buffer<uint8_t> expert_raw(q, experts * bytes), expanded(q, size_t(experts) * padded * stride), packed(q, experts * packed_bytes);
    std::fill_n(expanded.data, expanded.size, uint8_t(0));
    for (int e = 0; e < experts; ++e) for (int row = 0; row < rows; ++row) {
        const auto source = raw + size_t((row + e) % rows) * stride;
        std::copy_n(source, stride, expert_raw.data + (size_t(e) * rows + row) * stride);
        std::copy_n(source, stride, expanded.data + (size_t(e) * padded + row) * stride);
    }
    strata::intel::interleave_quant(expanded.data, packed.data, experts * padded, width, 98);
    for (size_t byte = 0; byte < expanded.size; ++byte)
        if (expanded[byte] != packed[strata::intel::quant_interleaved_offset(byte, width, 98)])
            throw std::runtime_error("Type18 routed interleave changed an original byte");
    for (int row = 0; row < experts * padded; ++row) for (int block = 0; block < width / 256; ++block)
        for (int byte = 98; byte < 100; ++byte) {
            const size_t physical = ((((size_t(row / 16) * (width / 256) + block) * 25 + byte / 4) * 16 + row % 16) * 4 + byte % 4);
            if (packed[physical] != 0) throw std::runtime_error("Type18 interleave padding is not zero");
        }
    Buffer<int32_t> ids(q, tokens * picks), counts(q, experts), slots(q, experts * tokens),
        owners(q, tokens * picks + experts), offsets(q, tokens * picks + experts), tiles(q, 1);
    Buffer<float> input(q, size_t(tokens) * picks * width), output(q, size_t(tokens) * picks * padded + 64);
    Buffer<int8_t> codes(q, input.size); Buffer<float> scales(q, input.size / 32);
    for (size_t i = 0; i < input.size; ++i) input[i] = float(std::sin(double(i) * .127) + double(int(i % 19) - 9) * .013);
    for (int route = 0; route < 3; ++route) for (int count = 1; count <= tokens; ++count) {
        for (int t = 0; t < count; ++t) for (int p = 0; p < picks; ++p)
            ids[t * picks + p] = route == 0 ? (t + 2 * p) % experts : route == 1 ? (p + t / 4) % experts : (p + (t & 1) * 3) % experts;
        strata::intel::ExpertPlan plan{counts.data, slots.data, owners.data, offsets.data, tiles.data, experts, count, picks};
        for (bool separate : {false, true}) {
            const int inputs = separate ? count * picks : count;
            strata::intel::quantize_decode_input(q, input.data, codes.data, scales.data, width, inputs).wait_and_throw();
            const size_t active = size_t(count) * picks * rows, packed_active = size_t(count) * picks * padded;
            std::vector<double> full(active, 0), q8(active, 0), half(active, 0), packed_q8(packed_active, 0), packed_half(packed_active, 0);
            for (int slot = 0; slot < count * picks; ++slot) for (int row = 0; row < rows; ++row) {
                const int source = separate ? slot : slot / picks, weight_row = (row + ids[slot]) % rows;
                const size_t destination = size_t(slot) * rows + row;
                for (int column = 0; column < width; ++column) {
                    const float w = expected_weights[size_t(weight_row) * width + column], x = input[size_t(source) * width + column];
                    const float qx = float(codes[size_t(source) * width + column]) * scales[size_t(source) * (width / 32) + column / 32];
                    full[destination] += double(w) * x;
                    q8[destination] += double(w) * qx;
                    half[destination] += double(float(sycl::half(w))) * float(sycl::half(x));
                }
                packed_q8[size_t(slot) * padded + row] = q8[destination];
                packed_half[size_t(slot) * padded + row] = half[destination];
            }
            const auto name = label + "_route_" + std::to_string(route) + "_T" + std::to_string(count) + (separate ? "_down" : "_gate");
            const auto check = [&](const std::string& suffix, const std::vector<double>& reference) {
                compare((name + suffix).c_str(), output.data, reference, 5e-5);
                check_output_tail(output, reference.size());
            };
            prepare_output(output, active);
            strata::intel::selected_matvec(q, 18, expert_raw.data, ids.data, picks, input.data, output.data, rows, width, separate, count).wait_and_throw();
            check("_scalar_fp64", full);
            prepare_output(output, active);
            dp4a.selected(18, expert_raw.data, ids.data, picks, input.data, output.data, rows, width, separate, count).wait_and_throw();
            check("_selected_q8_fp64", q8);
            for (int tile : {2, 4}) {
                plan.tile_width = tile; strata::intel::plan_experts(q, ids.data, plan);
                prepare_output(output, active);
                dp4a.grouped(18, expert_raw.data, plan, input.data, output.data, rows, width, separate).wait_and_throw();
                check("_grouped_q8_fp64_tile" + std::to_string(tile), q8);
            }
#ifdef TINYSTRATA_INTEL_INT8_XMX
            for (int tile : {1, 4}) {
                plan.tile_width = tile; strata::intel::plan_experts(q, ids.data, plan);
                prepare_output(output, active);
                int8_xmx.grouped(18, expert_raw.data, plan, input.data, output.data, rows, width, separate).wait_and_throw();
                check("_original_xmx_q8_fp64_tile" + std::to_string(tile), q8);
                prepare_output(output, packed_active);
                int8_xmx.grouped(18, packed.data, plan, input.data, output.data, padded, width, separate, true).wait_and_throw();
                check("_packed_xmx_q8_fp64_tile" + std::to_string(tile), packed_q8);
            }
#endif
            plan.tile_width = 8; strata::intel::plan_experts(q, ids.data, plan);
            prepare_output(output, active);
            strata::intel::selected_matmul_xmx(q, 18, expert_raw.data, plan, input.data, output.data, rows, width, separate).wait_and_throw();
            check("_half_xmx_fp64", half);
#ifdef TINYSTRATA_INTEL_MKL
            prepare_output(output, active);
            gemm.selected(18, expert_raw.data, bytes, plan, input.data, output.data, rows, width, separate).wait_and_throw();
            check("_original_mkl_half_fp64", half);
            prepare_output(output, packed_active);
            gemm.selected(18, packed.data, packed_bytes, plan, input.data, output.data, padded, width, separate, true).wait_and_throw();
            check("_restore_mkl_half_fp64", packed_half);
#endif
        }
    }
    if (!dp4a.valid()) throw std::runtime_error("Type18 routing rejected finite inputs");
#ifdef TINYSTRATA_INTEL_INT8_XMX
    if (!int8_xmx.valid()) throw std::runtime_error("Type18 XMX routing rejected finite inputs");
#endif
}
void quant_reference(sycl::queue& q, const char* path) {
    strata::intel::DecodeDp4a dp4a(q);
#ifdef TINYSTRATA_INTEL_INT8_XMX
    strata::intel::DecodeDp4a int8_xmx(q, true, true);
#endif
#ifdef TINYSTRATA_INTEL_MKL
    strata::intel::PrefillGemm gemm(q);
#endif
    std::ifstream file(strata::paths::native_path(path), std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open quantization reference fixture");
    const auto count = read<uint32_t>(file);
    if (!count || count > 100) throw std::runtime_error("Invalid fixture count");
    for (uint32_t index = 0; index < count; ++index) {
        const int type = read<int32_t>(file), rows = read<int32_t>(file), width = read<int32_t>(file);
        const uint64_t bytes = read<uint64_t>(file);
        if (rows <= 0 || rows > 100 || width <= 0 || width > 65536 || !bytes || bytes > 16 * 1024 * 1024)
            throw std::runtime_error("Invalid reference fixture dimensions");
        Buffer<uint8_t> raw(q, size_t(bytes));
        Buffer<float> x(q, size_t(width)), decoded(q, size_t(rows) * width), result(q, size_t(rows));
        std::vector<float> expected_weights(size_t(rows) * width);
        std::vector<double> expected_dot(static_cast<size_t>(rows));
        read_bytes(file, raw.data, size_t(bytes));
        read_bytes(file, x.data, size_t(width) * sizeof(float));
        read_bytes(file, expected_weights.data(), expected_weights.size() * sizeof(float));
        read_bytes(file, expected_dot.data(), expected_dot.size() * sizeof(double));
        strata::intel::dequantize(q, type, raw.data, decoded.data, size_t(rows) * width).wait_and_throw();
        const std::string label = "gguf_type_" + std::to_string(type) + "_case_" + std::to_string(index);
        compare((label + "_numpy_weights").c_str(), decoded.data,
                std::vector<double>(expected_weights.begin(), expected_weights.end()), 2e-6);
        if (type == 18) {
            if (std::memcmp(decoded.data, expected_weights.data(), expected_weights.size() * sizeof(float)))
                throw std::runtime_error("IQ3_XXS dequantization is not bit-exact to gguf NumPy");
            type18_routing_test(q, raw.data, size_t(bytes), rows, width, expected_weights, label, dp4a
#ifdef TINYSTRATA_INTEL_INT8_XMX
                , int8_xmx
#endif
#ifdef TINYSTRATA_INTEL_MKL
                , gemm
#endif
            );
        }
        strata::intel::quantized_matvec(q, type, raw.data, x.data, result.data, rows, width).wait_and_throw();
        compare((label + "_numpy_fp64_dot").c_str(), result.data, expected_dot, 5e-5);
        if (strata::intel::DecodeDp4a::supports(type) && width % 64 == 0) {
            Buffer<int8_t> codes(q, width);
            Buffer<float> scales(q, width / 32);
            strata::intel::quantize_decode_input(q, x.data, codes.data, scales.data, width, 1).wait_and_throw();
            std::vector<double> reference(rows, 0);
            for (int r = 0; r < rows; ++r) for (int c = 0; c < width; ++c)
                reference[r] += double(expected_weights[size_t(r) * width + c]) * (float(codes[c]) * scales[c / 32]);
            dp4a.dense(type, raw.data, x.data, result.data, rows, width).wait_and_throw();
            compare((label + "_dp4a_q8_fp64_dot").c_str(), result.data, reference, 5e-5);
            if (!dp4a.valid()) throw std::runtime_error("DP4A incorrectly rejected a finite input");
        }
        // Distinct inputs exercise partial decode tiles and a short-prefill expert tile.
        constexpr int tokens = 131, picks = 2, experts = 3;
        Buffer<float> batch_x(q, size_t(tokens) * width), batch_y(q, size_t(tokens) * rows);
        std::vector<double> batch_reference(size_t(tokens) * rows, 0);
        for (int t = 0; t < tokens; ++t) for (int c = 0; c < width; ++c)
            batch_x.data[size_t(t) * width + c] = x.data[c] * float(t + 1) / 9.f + float((c + t) % 11 - 5) * .013f;
        for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) for (int c = 0; c < width; ++c)
            batch_reference[size_t(t) * rows + r] += double(expected_weights[size_t(r) * width + c]) * batch_x.data[size_t(t) * width + c];
        strata::intel::quantized_matmul(q, type, raw.data, batch_x.data, batch_y.data, rows, width, tokens).wait_and_throw();
        compare((label + "_batch_numpy_fp64_dot").c_str(), batch_y.data, batch_reference, 5e-5);
        if (strata::intel::DecodeDp4a::supports(type) && width % 64 == 0) {
            Buffer<int8_t> codes(q, 8 * width); Buffer<float> scales(q, 8 * width / 32);
            for (int count : {1, 2, 3, 4, 5, 6, 7, 8}) {
                strata::intel::quantize_decode_input(q, batch_x.data, codes.data, scales.data, width, count).wait_and_throw();
                std::vector<double> reference(size_t(count) * rows, 0);
                for (int t = 0; t < count; ++t) for (int r = 0; r < rows; ++r) for (int c = 0; c < width; ++c)
                    reference[size_t(t) * rows + r] += double(expected_weights[size_t(r) * width + c]) *
                        (float(codes[size_t(t) * width + c]) * scales[size_t(t) * (width / 32) + c / 32]);
                dp4a.dense(type, raw.data, batch_x.data, batch_y.data, rows, width, count).wait_and_throw();
                compare((label + "_dp4a_batch_" + std::to_string(count)).c_str(), batch_y.data, reference, 5e-5);
#ifdef TINYSTRATA_INTEL_INT8_XMX
                if (type == 14) {
                    const int padded_rows = (rows + 15) / 16 * 16;
                    const size_t row_bytes = size_t(bytes) / rows;
                    Buffer<uint8_t> source(q, size_t(padded_rows) * row_bytes);
                    Buffer<uint8_t> packed(q, strata::intel::q6_interleaved_bytes(padded_rows, width));
                    Buffer<float> packed_y(q, size_t(padded_rows) * count);
                    for (int row = 0; row < padded_rows; ++row)
                        std::copy_n(raw.data + size_t(row % rows) * row_bytes, row_bytes, source.data + size_t(row) * row_bytes);
                    strata::intel::interleave_q6(source.data, packed.data, padded_rows, width);
                    for (size_t byte = 0; byte < size_t(padded_rows) * row_bytes; ++byte)
                        if (source[byte] != packed[strata::intel::q6_interleaved_offset(byte, width)])
                            throw std::runtime_error("Q6 interleave changed an original quantization byte");
                    int8_xmx.dense_interleaved_q6(packed.data, batch_x.data, packed_y.data, padded_rows, width, count).wait_and_throw();
                    std::vector<double> packed_reference(size_t(count) * padded_rows);
                    for (int token = 0; token < count; ++token) for (int row = 0; row < padded_rows; ++row)
                        packed_reference[size_t(token) * padded_rows + row] = reference[size_t(token) * rows + row % rows];
                    compare((label + "_interleaved_q6_numpy_fp64_" + std::to_string(count)).c_str(), packed_y.data, packed_reference, 5e-5);
                    int8_xmx.dense_interleaved_q6(packed.data, batch_x.data, packed_y.data, padded_rows, width, count, true).wait_and_throw();
                    compare((label + "_splitk_q6_numpy_fp64_" + std::to_string(count)).c_str(), packed_y.data, packed_reference, 5e-5);
                    if(width==2048 && count==1)for(int selected_count:{1,15,17}){
                        if(selected_count>padded_rows)continue;
                        Buffer<int32_t> ids(q,selected_count);Buffer<float> selected_y(q,selected_count+16);
                        for(int row=0;row<selected_count;++row)ids[row]=row+1==selected_count?padded_rows-1:row;
                        std::fill_n(selected_y.data,selected_count+16,12345.f);
                        int8_xmx.selected_vocabulary_q6(packed.data,batch_x.data,selected_y.data,padded_rows,width,ids.data,selected_count).wait_and_throw();
                        std::vector<double> expected(selected_count);
                        for(int row=0;row<selected_count;++row)expected[row]=packed_reference[ids[row]];
                        compare((label+"_selected_q6_numpy_fp64_"+std::to_string(selected_count)).c_str(),selected_y.data,expected,5e-5);
                        for(int row=selected_count;row<selected_count+16;++row)if(selected_y[row]!=12345.f)throw std::runtime_error("Selected Q6 overwrote the output tail");
                    }
#ifdef TINYSTRATA_INTEL_MKL
                    if (count == 8) {
                        Buffer<float> prefill_y(q, size_t(tokens) * padded_rows);
                        std::vector<double> prefill_reference(size_t(tokens) * padded_rows, 0.);
                        for (int token = 0; token < tokens; ++token) for (int row = 0; row < padded_rows; ++row)
                            for (int column = 0; column < width; ++column)
                                prefill_reference[size_t(token) * padded_rows + row] += double(float(sycl::half(expected_weights[size_t(row % rows) * width + column]))) *
                                    float(sycl::half(batch_x.data[size_t(token) * width + column]));
                        gemm.dense_interleaved_q6(packed.data, batch_x.data, prefill_y.data, padded_rows, width, tokens).wait_and_throw();
                        compare((label + "_interleaved_q6_mkl_half_fp64").c_str(), prefill_y.data, prefill_reference, 5e-5);
                    }
#endif
                }
#endif
            }
        }
        if (width % 256 == 0) {
            std::vector<double> half_reference(size_t(tokens) * rows, 0);
            for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) for (int c = 0; c < width; ++c)
                half_reference[size_t(t) * rows + r] += double(float(sycl::half(expected_weights[size_t(r) * width + c]))) *
                    float(sycl::half(batch_x.data[size_t(t) * width + c]));
            strata::intel::quantized_matmul_xmx(q, type, raw.data, batch_x.data, batch_y.data, rows, width, tokens).wait_and_throw();
            compare((label + "_xmx_half_fp64_dot").c_str(), batch_y.data, half_reference, 5e-5);
#ifdef TINYSTRATA_INTEL_MKL
            gemm.dense(type, raw.data, batch_x.data, batch_y.data, rows, width, tokens).wait_and_throw();
            compare((label + "_mkl_half_fp64_dot").c_str(), batch_y.data, half_reference, 5e-5);
#endif
        }
        Buffer<uint8_t> expert_raw(q, size_t(bytes) * experts);
        Buffer<int32_t> ids(q, tokens * picks);
        Buffer<int32_t> counts(q, experts), slots(q, experts * tokens), owners(q, tokens * picks + experts), offsets(q, tokens * picks + experts), tiles(q, 1);
        strata::intel::ExpertPlan plan{counts.data, slots.data, owners.data, offsets.data, tiles.data, experts, tokens, picks};
        Buffer<float> expert_x(q, size_t(tokens) * picks * width), expert_y(q, size_t(tokens) * picks * rows + 64);
        const size_t stride = size_t(bytes) / rows;
        for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; ++r)
            std::copy_n(raw.data + size_t((r + e) % rows) * stride, stride, expert_raw.data + (size_t(e) * rows + r) * stride);
        for (int t = 0; t < tokens; ++t) for (int p = 0; p < picks; ++p) {
            ids.data[t * picks + p] = (t + 2 * p) % experts;
            for (int c = 0; c < width; ++c) expert_x.data[(size_t(t) * picks + p) * width + c] = batch_x.data[size_t(t) * width + c] + p * .021f;
        }
        for (bool separate : {false, true}) {
            std::vector<double> reference(size_t(tokens) * picks * rows, 0);
            for (int t = 0; t < tokens; ++t) for (int p = 0; p < picks; ++p) for (int r = 0; r < rows; ++r) {
                const int source_row = (r + ids.data[t * picks + p]) % rows;
                for (int c = 0; c < width; ++c)
                    reference[(size_t(t) * picks + p) * rows + r] += double(expected_weights[size_t(source_row) * width + c]) *
                        (separate ? expert_x.data[(size_t(t) * picks + p) * width + c] : batch_x.data[size_t(t) * width + c]);
            }
            strata::intel::selected_matvec(q, type, expert_raw.data, ids.data, picks, separate ? expert_x.data : batch_x.data,
                expert_y.data, rows, width, separate, tokens).wait_and_throw();
            compare((label + (separate ? "_batch_selected_down" : "_batch_selected_gate")).c_str(), expert_y.data, reference, 5e-5);
            strata::intel::selected_matvec(q, type, expert_raw.data, ids.data, picks, separate ? expert_x.data : batch_x.data,
                expert_y.data, rows, width, separate, 1).wait_and_throw();
            compare((label + (separate ? "_decode_selected_down" : "_decode_selected_gate")).c_str(), expert_y.data,
                std::vector<double>(reference.begin(), reference.begin() + picks * rows), 5e-5);
            if (strata::intel::DecodeDp4a::supports(type) && width % 64 == 0) {
                Buffer<int8_t> codes(q, tokens * picks * width);
                Buffer<float> scales(q, tokens * picks * width / 32);
                strata::intel::quantize_decode_input(q, separate ? expert_x.data : batch_x.data, codes.data, scales.data,
                    width, separate ? picks : 1).wait_and_throw();
                std::vector<double> q8_reference(size_t(picks) * rows, 0);
                for (int p = 0; p < picks; ++p) for (int r = 0; r < rows; ++r) {
                    const int source_row = (r + ids.data[p]) % rows, source = separate ? p : 0;
                    for (int c = 0; c < width; ++c)
                        q8_reference[size_t(p) * rows + r] += double(expected_weights[size_t(source_row) * width + c]) *
                            (float(codes[size_t(source) * width + c]) * scales[size_t(source) * (width / 32) + c / 32]);
                }
                dp4a.selected(type, expert_raw.data, ids.data, picks, separate ? expert_x.data : batch_x.data,
                    expert_y.data, rows, width, separate).wait_and_throw();
                compare((label + (separate ? "_dp4a_down_fp64" : "_dp4a_gate_fp64")).c_str(), expert_y.data, q8_reference, 5e-5);
                // The measured IQ packed-gather ranges are 2..8 and 32..128.
                // Their adjacent 31/129 controls retain the previous path.
                const std::vector<int> counts_to_test = (type == 21 || type == 22) ?
                    std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 17, 31, 32, 64, 84, 128, 129} :
                    std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 17, 129};
                for (int count : counts_to_test) {
                    strata::intel::quantize_decode_input(q, separate ? expert_x.data : batch_x.data, codes.data, scales.data,
                        width, separate ? picks * count : count).wait_and_throw();
                    std::vector<double> reference(size_t(count) * picks * rows, 0);
                    for (int t = 0; t < count; ++t) for (int p = 0; p < picks; ++p) for (int r = 0; r < rows; ++r) {
                        const int source_row = (r + ids.data[t * picks + p]) % rows, source = separate ? t * picks + p : t;
                        for (int c = 0; c < width; ++c)
                            reference[(size_t(t) * picks + p) * rows + r] += double(expected_weights[size_t(source_row) * width + c]) *
                                (float(codes[size_t(source) * width + c]) * scales[size_t(source) * (width / 32) + c / 32]);
                    }
                    if (count <= 8) {
                    dp4a.selected(type, expert_raw.data, ids.data, picks, separate ? expert_x.data : batch_x.data,
                        expert_y.data, rows, width, separate, count).wait_and_throw();
                    compare((label + (separate ? "_dp4a_batch_down_" : "_dp4a_batch_gate_") + std::to_string(count)).c_str(), expert_y.data, reference, 5e-5);
                    }
                    plan.tokens = count;
                    for (int tile : {2, 4}) {
                        plan.tile_width = tile;
                        strata::intel::plan_experts(q, ids.data, plan);
                        prepare_output(expert_y, reference.size());
                        dp4a.grouped(type, expert_raw.data, plan, separate ? expert_x.data : batch_x.data,
                            expert_y.data, rows, width, separate).wait_and_throw();
                        compare((label + (separate ? "_dp4a_grouped_down_" : "_dp4a_grouped_gate_") + std::to_string(count) + "_tile_" + std::to_string(tile)).c_str(), expert_y.data, reference, 5e-5);
                        check_output_tail(expert_y, reference.size());
#ifdef TINYSTRATA_INTEL_INT8_XMX
                        if (tile == 4 && (type == 14 || type == 21 || type == 22)) {
                            prepare_output(expert_y, reference.size());
                            int8_xmx.grouped(type, expert_raw.data, plan, separate ? expert_x.data : batch_x.data,
                                expert_y.data, rows, width, separate).wait_and_throw();
                            compare((label + (separate ? "_int8_xmx_down_" : "_int8_xmx_gate_") + std::to_string(count)).c_str(), expert_y.data, reference, 5e-5);
                            check_output_tail(expert_y, reference.size());
                            const int padded_rows = (rows + 15) / 16 * 16;
                            const int block_bytes = type == 14 ? 210 : type == 21 ? 110 : 82;
                            const size_t row_bytes = size_t(bytes) / rows;
                            Buffer<uint8_t> expanded(q, size_t(experts) * padded_rows * row_bytes);
                            Buffer<uint8_t> interleaved(q, strata::intel::quant_interleaved_bytes(experts * padded_rows, width, block_bytes));
                            Buffer<float> packed_y(q, size_t(count) * picks * padded_rows + 64);
                            for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < padded_rows; ++row)
                                std::copy_n(expert_raw.data + (size_t(expert) * rows + row % rows) * row_bytes,
                                    row_bytes, expanded.data + (size_t(expert) * padded_rows + row) * row_bytes);
                            strata::intel::interleave_quant(expanded.data, interleaved.data, experts * padded_rows, width, block_bytes);
                            for (size_t byte = 0; byte < size_t(experts) * padded_rows * row_bytes; ++byte)
                                if (expanded[byte] != interleaved[strata::intel::quant_interleaved_offset(byte, width, block_bytes)])
                                    throw std::runtime_error("Expert interleave changed a quantization byte");
                            std::vector<double> packed_reference(size_t(count) * picks * padded_rows);
                            for (int slot = 0; slot < count * picks; ++slot) for (int row = 0; row < padded_rows; ++row)
                                packed_reference[size_t(slot) * padded_rows + row] = reference[size_t(slot) * rows + row % rows];
                            for (int packed_tile : {1, 4}) {
                                plan.tile_width = packed_tile;
                                strata::intel::plan_experts(q, ids.data, plan);
                                prepare_output(packed_y, packed_reference.size());
                                int8_xmx.grouped(type, interleaved.data, plan, separate ? expert_x.data : batch_x.data,
                                    packed_y.data, padded_rows, width, separate, true).wait_and_throw();
                                compare((label + (separate ? "_interleaved_down_" : "_interleaved_gate_") + std::to_string(count) + "_tile_" + std::to_string(packed_tile)).c_str(), packed_y.data, packed_reference, 5e-5);
                                check_output_tail(packed_y, packed_reference.size());
                            }
                        }
#endif
                    }
                    plan.tile_width = 4;
                    plan.tokens = tokens;
                }
                if (!dp4a.valid()) throw std::runtime_error("DP4A incorrectly rejected finite expert inputs");
            }
            strata::intel::plan_experts(q, ids.data, plan);
            strata::intel::selected_matmul(q, type, expert_raw.data, plan, separate ? expert_x.data : batch_x.data,
                expert_y.data, rows, width, separate).wait_and_throw();
            compare((label + (separate ? "_grouped_down" : "_grouped_gate")).c_str(), expert_y.data, reference, 5e-5);
            if (width % 256 == 0) {
                std::vector<double> half_reference(size_t(tokens) * picks * rows, 0);
                for (int t = 0; t < tokens; ++t) for (int p = 0; p < picks; ++p) for (int r = 0; r < rows; ++r) {
                    const int source_row = (r + ids.data[t * picks + p]) % rows;
                    for (int c = 0; c < width; ++c)
                        half_reference[(size_t(t) * picks + p) * rows + r] += double(float(sycl::half(expected_weights[size_t(source_row) * width + c]))) *
                            float(sycl::half(separate ? expert_x.data[(size_t(t) * picks + p) * width + c] : batch_x.data[size_t(t) * width + c]));
                }
                plan.tile_width = 8;
                strata::intel::plan_experts(q, ids.data, plan);
                strata::intel::selected_matmul_xmx(q, type, expert_raw.data, plan, separate ? expert_x.data : batch_x.data,
                    expert_y.data, rows, width, separate).wait_and_throw();
                compare((label + (separate ? "_xmx_grouped_down" : "_xmx_grouped_gate")).c_str(), expert_y.data, half_reference, 5e-5);
#ifdef TINYSTRATA_INTEL_MKL
                gemm.selected(type, expert_raw.data, size_t(bytes), plan, separate ? expert_x.data : batch_x.data,
                    expert_y.data, rows, width, separate).wait_and_throw();
                compare((label + (separate ? "_mkl_grouped_down" : "_mkl_grouped_gate")).c_str(), expert_y.data, half_reference, 5e-5);
                if(type==14||type==21||type==22){
                    const int padded_rows=(rows+15)/16*16,block_bytes=type==14?210:type==21?110:82;
                    const size_t row_bytes=size_t(bytes)/rows,packed_bytes=strata::intel::quant_interleaved_bytes(padded_rows,width,block_bytes);
                    Buffer<uint8_t> expanded(q,size_t(experts)*padded_rows*row_bytes),packed(q,size_t(experts)*packed_bytes);
                    Buffer<float> packed_y(q,size_t(tokens)*picks*padded_rows);
                    for(int expert=0;expert<experts;++expert)for(int row=0;row<padded_rows;++row)
                        std::copy_n(expert_raw.data+(size_t(expert)*rows+row%rows)*row_bytes,row_bytes,
                            expanded.data+(size_t(expert)*padded_rows+row)*row_bytes);
                    strata::intel::interleave_quant(expanded.data,packed.data,experts*padded_rows,width,block_bytes);
                    std::vector<double> expected(size_t(tokens)*picks*padded_rows);
                    for(int slot=0;slot<tokens*picks;++slot)for(int row=0;row<padded_rows;++row)
                        expected[size_t(slot)*padded_rows+row]=half_reference[size_t(slot)*rows+row%rows];
                    gemm.selected(type,packed.data,packed_bytes,plan,separate?expert_x.data:batch_x.data,
                        packed_y.data,padded_rows,width,separate,true).wait_and_throw();
                    compare((label+(separate?"_mkl_interleaved_down":"_mkl_interleaved_gate")).c_str(),packed_y.data,expected,5e-5);
                }
#endif
                plan.tile_width = 4;
            }
        }
    }
    if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing fixture data");
}
}
int main(int argc, char** argv) {
    try {
        strata::paths::Utf8Arguments arguments(argc, argv);
        argc = arguments.argc(); argv = arguments.argv();
        if (argc > 2) throw std::runtime_error("Usage: intel_qwen_ops_test [quantization-reference.bin]");
        const auto device = strata::intel::select_device(0);
        sycl::queue queue(device, [](sycl::exception_list exceptions) {
            for (const auto& error : exceptions) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});
        std::printf("device=%s\n", device.get_info<sycl::info::device::name>().c_str());
        norm_test(queue);
        activation_test(queue);
        q8_kv_test(queue);
        q8_attention_batch_test(queue);
        q8_attention_long_reuse_test(queue);
#ifdef TINYSTRATA_INTEL_INT8_XMX
        splitk_capacity_test(queue);
        selected_vocabulary_capacity_test(queue);
#endif
        matvec_test(queue);
        attention_test(queue);
        decode_attention_test(queue);
        for (int tokens : {7, 129, 1024}) recurrent_test(queue, tokens);
        if (argc == 2) quant_reference(queue, argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Qwen SYCL numerical checks failed: %s\n", error.what());
        return 1;
    }
}
