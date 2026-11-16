#include "strata/intel/device.hpp"
#include "strata/intel/npu.hpp"
#include "strata/intel/prefill_gemm.hpp"
#include "strata/intel/q6_interleave.hpp"
#include "strata/intel/quant.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

template<class T>struct Buffer{
    sycl::queue& q;T* data;
    Buffer(sycl::queue& q,size_t n):q(q),data(sycl::malloc_shared<T>(n,q)){if(!data)throw std::bad_alloc();}
    ~Buffer(){try{q.wait_and_throw();}catch(...){}sycl::free(data,q);}
};
static void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());return x.at(x.size()/2);}
static uint64_t hash(const void* p,size_t n){const auto* b=static_cast<const uint8_t*>(p);uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ull;return h;}
int main(int argc,char** argv){try{
    strata::paths::Utf8Arguments args(argc,argv);argc=args.argc();argv=args.argv();
    if(argc!=2)throw std::runtime_error("Usage: intel_npu_two_request_test real-qkv.bin");
    using namespace strata::intel;
    constexpr int rows=8192,width=2048,capacity=1024,npu_rows=1024;
    constexpr size_t raw_bytes=size_t(rows)*8*210,output_elements=size_t(capacity)*rows;
    sycl::queue q(select_device(0),sycl::property::queue::in_order{});
    Buffer<uint8_t> compressed(q,raw_bytes+q6_interleaved_bytes(rows,width));
    auto* raw=compressed.data;auto* packed=raw+raw_bytes;
    std::ifstream file(strata::paths::native_path(argv[1]),std::ios::binary);int actual_rows=0,actual_width=0;
    file.read(reinterpret_cast<char*>(&actual_rows),4);file.read(reinterpret_cast<char*>(&actual_width),4);
    check(actual_rows==rows&&actual_width==width,"Unexpected real QKV fixture shape");
    file.read(reinterpret_cast<char*>(raw),raw_bytes);check(bool(file)&&file.peek()==std::char_traits<char>::eof(),"QKV fixture length mismatch");
    interleave_q6(raw,packed,rows,width);
    Buffer<float> x0(q,size_t(capacity)*width),x1(q,size_t(capacity)*width);
    Buffer<float> ref0(q,output_elements+32),ref1(q,output_elements+32),out0(q,output_elements+32),out1(q,output_elements+32);
    for(size_t i=0;i<size_t(capacity)*width;++i){x0.data[i]=std::sin(float(i%1987)*.013f)*.3f;x1.data[i]=std::cos(float(i%1831)*.021f)*.2f;}
    const auto weight_hash=hash(compressed.data,raw_bytes+q6_interleaved_bytes(rows,width));
    const auto input0_hash=hash(x0.data,size_t(capacity)*width*4),input1_hash=hash(x1.data,size_t(capacity)*width*4);
    PrefillGemm gpu(q,1,capacity);auto owner=std::make_unique<NpuProjection>(q,2);auto& npu=*owner;
    check(npu.memory_usage().request_slots==2&&npu.memory_usage().host_usm_bytes==(128ull<<20),"Two independent bank allocation ledger mismatch");
    check(npu.memory_usage().opaque_runtime_reservation_bytes==(192ull<<20),"Two-request opaque margin mismatch");
    bool invalid_bank=false;try{npu.prepare_split_input(2,x0.data,width,128);}catch(const std::invalid_argument&){invalid_bank=true;}
    check(invalid_bank,"Out-of-range bank accepted");
    const float poison=12345.f;int cases=0;
    for(int tokens:{256,512,1024})for(bool packed_layout:{true,false})for(bool remote:{true,false}){
        NpuSplitOptions options;options.interleaved=packed_layout;options.remote=remote;
        const auto* weights=packed_layout?packed:raw;
        auto run=[&](bool paired){
            const auto started=std::chrono::steady_clock::now();
            npu.prepare_split_input(0,x0.data,width,tokens);
            npu.prepare_split_input(1,x1.data,width,tokens);
            sycl::event suffix;
            check(npu.begin_split(gpu,14,weights,x0.data,paired?out0.data:ref0.data,rows,width,tokens,npu_rows,options,suffix,0),"Bank zero admission failed");
            if(!paired)npu.finish_split(0);
            check(npu.begin_split(gpu,14,weights,x1.data,paired?out1.data:ref1.data,rows,width,tokens,npu_rows,options,suffix,1),"Bank one admission failed");
            if(paired){
                check(npu.pending_split_count()==2&&npu.split_pending(0)&&npu.split_pending(1),"Two independent requests were not pending");
                bool busy=false;try{npu.prepare_split_input(0,x1.data,width,tokens);}catch(const std::logic_error&){busy=true;}
                check(busy&&npu.pending_split_count()==2,"Pending bank input was rebound");
                bool third=false;try{npu.begin_split(gpu,14,weights,x0.data,out0.data,rows,width,tokens,npu_rows,options,suffix,0);}catch(const std::logic_error&){third=true;}
                check(third&&npu.pending_split_count()==2,"A third request changed a pending bank");
                npu.finish_split(0);
            }
            npu.finish_split(1).wait_and_throw();
            check(npu.pending_split_count()==0,"Join did not retire both requests");
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        };
        std::vector<double> serial,paired;
        for(int round=0;round<10;++round){
            for(float* p:{ref0.data,ref1.data,out0.data,out1.data}){std::fill(p,p+size_t(tokens)*rows,NAN);std::fill(p+size_t(tokens)*rows,p+output_elements+32,poison);}
            double a,b;if(round%2){a=run(false);b=run(true);}else{b=run(true);a=run(false);}
            if(round>=3){serial.push_back(a);paired.push_back(b);}
            check(std::memcmp(ref0.data,out0.data,size_t(tokens)*rows*4)==0&&std::memcmp(ref1.data,out1.data,size_t(tokens)*rows*4)==0,"Paired outputs differ from serialized requests");
            for(size_t i=0;i<size_t(tokens)*rows;++i)check(std::isfinite(out0.data[i])&&std::isfinite(out1.data[i]),"Nonfinite/unwritten two-request output");
            for(size_t i=size_t(tokens)*rows;i<output_elements+32;++i)check(ref0.data[i]==poison&&ref1.data[i]==poison&&out0.data[i]==poison&&out1.data[i]==poison,"Unused/canary output changed");
        }
        double maximum_error=0;
        for(int bank:{0,1})for(int token:{0,tokens-1})for(int row:{0,npu_rows-1,npu_rows,rows-1}){
            const float* input=bank?x1.data:x0.data;const float* output=bank?out1.data:out0.data;
            double expected=0;for(int col=0;col<width;++col)expected+=double(float(sycl::half(input[size_t(token)*width+col])))*float(sycl::half(quant::value<14>(raw+size_t(row)*1680,col)));
            const double error=std::abs(double(output[size_t(token)*rows+row])-expected);
            const double output_rounding=row<npu_rows?std::abs(double(float(sycl::half(expected)))-expected):0;
            check(error<=output_rounding+1e-3*(1+std::abs(expected)),"Two-request FP64 boundary mismatch");maximum_error=std::max(maximum_error,error);
        }
        ++cases;std::printf("{\"case\":%d,\"tokens\":%d,\"rows\":%d,\"npu_rows\":%d,\"packed\":%s,\"remote\":%s,\"serial_two_slots_ms\":%.6f,\"paired_two_slots_ms\":%.6f,\"normal_queue_profiling\":false,\"warm_rounds\":3,\"measured_rounds\":7,\"max_FP64_abs\":%.9g,\"both_pending\":true,\"serial_bit_equal\":true,\"allfinite\":true,\"unused_canary\":true}\n",cases,tokens,rows,npu_rows,packed_layout?"true":"false",remote?"true":"false",median(serial),median(paired),maximum_error);std::fflush(stdout);
    }
    // Recycle bank zero while bank one is still pending. begin() must retire
    // zero's earlier GPU merge before the NPU overwrites that output bank.
    NpuSplitOptions options;options.remote=true;options.interleaved=true;sycl::event suffix;
    npu.prepare_split_input(0,x0.data,width,128);npu.prepare_split_input(1,x1.data,width,128);
    check(npu.begin_split(gpu,14,packed,x0.data,out0.data,rows,width,128,npu_rows,options,suffix,0),"Recycle zero failed");
    check(npu.begin_split(gpu,14,packed,x1.data,out1.data,rows,width,128,npu_rows,options,suffix,1),"Recycle one failed");
    npu.finish_split(0);npu.prepare_split_input(0,x0.data,width,128);
    check(npu.begin_split(gpu,14,packed,x0.data,out0.data,rows,width,128,npu_rows,options,suffix,0),"Retired zero could not be reused");
    check(npu.pending_split_count()==2,"Bank zero reuse disturbed bank one");npu.finish_split(0);npu.finish_split(1).wait_and_throw();
    check(npu.split_counters().max_pending_requests==2,"Two-request concurrency counter mismatch");
    // Independent request bindings must also survive a different graph shape
    // in each bank, rather than treating a previous logical extent as capacity.
    for(bool paired:{false,true}){
        for(unsigned bank=0;bank<2;++bank){
            const int tokens=bank?1005:128;float* output=paired?(bank?out1.data:out0.data):(bank?ref1.data:ref0.data);
            std::fill(output,output+size_t(tokens)*rows,NAN);std::fill(output+size_t(tokens)*rows,output+output_elements+32,poison);
            npu.prepare_split_input(bank,bank?x1.data:x0.data,width,tokens);
        }
        for(unsigned bank=0;bank<2;++bank){const int tokens=bank?1005:128;
            check(npu.begin_split(gpu,14,packed,bank?x1.data:x0.data,paired?(bank?out1.data:out0.data):(bank?ref1.data:ref0.data),rows,width,tokens,npu_rows,options,suffix,bank),"Changed-shape admission failed");
            if(!paired)npu.finish_split(bank).wait_and_throw();
        }
        if(paired){check(npu.pending_split_count()==2,"Changed shapes did not use independent requests");npu.finish_split(0);npu.finish_split(1).wait_and_throw();}
    }
    check(std::memcmp(ref0.data,out0.data,size_t(128)*rows*4)==0&&std::memcmp(ref1.data,out1.data,size_t(1005)*rows*4)==0,"Changed-shape output bits differ");
    for(size_t i=size_t(128)*rows;i<output_elements+32;++i)check(out0.data[i]==poison&&ref0.data[i]==poison,"Short shape canary changed");
    for(size_t i=size_t(1005)*rows;i<output_elements+32;++i)check(out1.data[i]==poison&&ref1.data[i]==poison,"Long shape canary changed");
    owner.reset();
    std::fill(out0.data,out0.data+output_elements+32,poison);std::fill(out1.data,out1.data+output_elements+32,poison);
    {
        NpuProjection abandoned(q,2);
        abandoned.prepare_split_input(0,x0.data,width,128);abandoned.prepare_split_input(1,x1.data,width,128);
        check(abandoned.begin_split(gpu,14,packed,x0.data,out0.data,rows,width,128,npu_rows,options,suffix,0),"Abandoned zero admission failed");
        check(abandoned.begin_split(gpu,14,packed,x1.data,out1.data,rows,width,128,npu_rows,options,suffix,1),"Abandoned one admission failed");
        check(abandoned.pending_split_count()==2,"Abandoned scope did not own both requests");
    }
    q.wait_and_throw();
    for(int token=0;token<128;++token)for(int row=0;row<npu_rows;++row)check(out0.data[size_t(token)*rows+row]==poison&&out1.data[size_t(token)*rows+row]==poison,"Abandoned request prefix was merged");
    for(size_t i=size_t(128)*rows;i<output_elements+32;++i)check(out0.data[i]==poison&&out1.data[i]==poison,"Abandoned request canary changed");
    {
        // Preserve the legacy full operator's previous-output-copy retirement
        // on an out-of-order queue; split admission itself requires in-order.
        sycl::queue unordered(q.get_context(),q.get_device());NpuProjection legacy(unordered);
        legacy.run(14,raw,x0.data,ref0.data,64,width,128).wait_and_throw();
        legacy.run(14,raw,x1.data,ref1.data,64,width,128).wait_and_throw();
        const auto first=legacy.run(14,raw,x0.data,out0.data,64,width,128);
        const auto second=legacy.run(14,raw,x1.data,out1.data,64,width,128);
        auto first_copy=first;first_copy.wait_and_throw();auto second_copy=second;second_copy.wait_and_throw();
        check(std::memcmp(ref0.data,out0.data,size_t(128)*64*4)==0&&std::memcmp(ref1.data,out1.data,size_t(128)*64*4)==0,"Out-of-order full operator output-copy race");
    }
    check(hash(compressed.data,raw_bytes+q6_interleaved_bytes(rows,width))==weight_hash&&hash(x0.data,size_t(capacity)*width*4)==input0_hash&&hash(x1.data,size_t(capacity)*width*4)==input1_hash,"Fixture inputs/weights mutated");
    std::printf("{\"passed\":true,\"cases\":%d,\"request_slots\":2,\"max_pending_requests\":2,\"independent_content_inputs\":true,\"recycle_with_other_pending\":true,\"changed_shapes\":true,\"abandoned_both_drained\":true,\"out_of_order_full_retirement\":true,\"readonly\":true,\"hardware_parallel_execution_claim\":false}\n",cases);
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"NPU two-request fixture: %s\n",error.what());return 1;}}
