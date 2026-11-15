#include "strata/intel/device.hpp"
#include "strata/intel/npu.hpp"
#include "strata/intel/prefill_gemm.hpp"
#include "strata/intel/q6_interleave.hpp"
#include "strata/intel/quant.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

template<class T> struct Buffer {
    sycl::queue& q;T* ptr;
    Buffer(sycl::queue& queue,size_t count):q(queue),ptr(sycl::malloc_shared<T>(count,q)){if(!ptr)throw std::runtime_error("Async fixture allocation failed");}
    ~Buffer(){try{q.wait_and_throw();}catch(...){}sycl::free(ptr,q);}
};
static uint64_t hash(const void* data,size_t bytes){auto p=static_cast<const uint8_t*>(data);uint64_t h=1469598103934665603ull;for(size_t i=0;i<bytes;++i)h=(h^p[i])*1099511628211ull;return h;}
static void load(const char* name,uint8_t* data,int rows){
    std::ifstream f(strata::paths::native_path(name),std::ios::binary);int r=0,w=0;f.read(reinterpret_cast<char*>(&r),4);f.read(reinterpret_cast<char*>(&w),4);
    if(r!=rows||w!=2048||!f.read(reinterpret_cast<char*>(data),size_t(rows)*1680))throw std::runtime_error("Invalid real Q6 async fixture");
}
static void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(int argc,char** argv){
    try {
        strata::paths::Utf8Arguments args(argc,argv);argc=args.argc();argv=args.argv();
        if(argc!=3)throw std::runtime_error("Usage: intel_npu_async_test real-qkv.bin real-gate.bin");
        constexpr int width=2048,rows=8192,gate_rows=4096,max_tokens=1024;
        constexpr size_t qbytes=size_t(rows)*1680,gbytes=size_t(gate_rows)*1680;
        sycl::queue q(strata::intel::select_device(0),sycl::property::queue::in_order{});
        const size_t packed_bytes=strata::intel::q6_interleaved_bytes(rows,width);
        Buffer<uint8_t> arena(q,qbytes+gbytes+packed_bytes);auto raw=arena.ptr,gate=raw+qbytes,packed=gate+gbytes;
        load(argv[1],raw,rows);load(argv[2],gate,gate_rows);strata::intel::interleave_q6(raw,packed,rows,width);
        Buffer<float> x(q,size_t(max_tokens)*width),reference(q,size_t(max_tokens)*rows+32),actual(q,size_t(max_tokens)*rows+32);
        Buffer<float> gate_reference(q,size_t(max_tokens)*gate_rows+32),gate_actual(q,size_t(max_tokens)*gate_rows+32);
        for(size_t i=0;i<size_t(max_tokens)*width;++i)x.ptr[i]=std::sin(float(i%1987)*.013f)*.3f;
        const auto weight_hash=hash(arena.ptr,qbytes+gbytes+packed_bytes),input_hash=hash(x.ptr,size_t(max_tokens)*width*sizeof(float));
        strata::intel::PrefillGemm gpu(q,1,max_tokens);auto npu=std::make_unique<strata::intel::NpuProjection>(q);
        int cases=0;const float poison=12345.f;
        bool empty_rejected=false;try{npu->finish_split();}catch(const std::logic_error&){empty_rejected=true;}check(empty_rejected,"Empty finish must reject");
        // Start remote-first1024, matching the real model admission order.
        for(int tokens:{1024,128})for(int npu_rows:{1024,512})for(bool interleaved:{true,false})for(bool remote:{true,false}){
            std::fill(reference.ptr,reference.ptr+size_t(max_tokens)*rows+32,poison);std::fill(actual.ptr,actual.ptr+size_t(max_tokens)*rows+32,poison);
            std::fill(gate_reference.ptr,gate_reference.ptr+size_t(max_tokens)*gate_rows+32,poison);std::fill(gate_actual.ptr,gate_actual.ptr+size_t(max_tokens)*gate_rows+32,poison);
            strata::intel::NpuSplitOptions options;options.remote=remote;options.interleaved=interleaved;
            sycl::event completion;check(npu->run_split(gpu,14,interleaved?packed:raw,x.ptr,reference.ptr,rows,width,tokens,npu_rows,options,completion),"Sync split rejected");completion.wait_and_throw();
            gpu.dense(14,gate,x.ptr,gate_reference.ptr,gate_rows,width,tokens).wait_and_throw();
            sycl::event suffix;check(npu->begin_split(gpu,14,interleaved?packed:raw,x.ptr,actual.ptr,rows,width,tokens,npu_rows,options,suffix),"Async split rejected");check(npu->split_pending(),"Begin did not retain ticket");
            bool second_rejected=false,full_rejected=false;
            try{npu->begin_split(gpu,14,raw,x.ptr,actual.ptr,rows,width,tokens,256,options,suffix);}catch(const std::logic_error&){second_rejected=true;}
            try{npu->run(14,raw,x.ptr,actual.ptr,rows,width,tokens);}catch(const std::logic_error&){full_rejected=true;}
            check(second_rejected&&full_rejected&&npu->split_pending(),"Pending guard mutated ticket");
            // Reuse the SAME GPU GEMM scratch while QKV's NPU request runs.
            gpu.dense(14,gate,x.ptr,gate_actual.ptr,gate_rows,width,tokens);
            completion=npu->finish_split();check(!npu->split_pending(),"Join retained ticket");completion.wait_and_throw();
            check(std::memcmp(reference.ptr,actual.ptr,size_t(tokens)*rows*sizeof(float))==0,"Async QKV differs from same-operator sync");
            check(std::memcmp(gate_reference.ptr,gate_actual.ptr,size_t(tokens)*gate_rows*sizeof(float))==0,"Intervening GPU gate scratch corrupted");
            for(size_t i=0;i<size_t(tokens)*rows;++i)check(std::isfinite(actual.ptr[i]),"Nonfinite async QKV output");
            for(size_t i=0;i<size_t(tokens)*gate_rows;++i)check(std::isfinite(gate_actual.ptr[i]),"Nonfinite intervening gate output");
            for(size_t i=size_t(tokens)*rows;i<size_t(max_tokens)*rows+32;++i)check(actual.ptr[i]==poison&&reference.ptr[i]==poison,"QKV unused/canary write");
            for(size_t i=size_t(tokens)*gate_rows;i<size_t(max_tokens)*gate_rows+32;++i)check(gate_actual.ptr[i]==poison&&gate_reference.ptr[i]==poison,"Gate unused/canary write");
            for(int token:{0,tokens-1})for(int row:{0,npu_rows-1,npu_rows,rows-1}){
                double expected=0;for(int col=0;col<width;++col)expected+=double(float(sycl::half(x.ptr[size_t(token)*width+col])))*float(sycl::half(strata::intel::quant::value<14>(raw+size_t(row)*1680,col)));
                const double rounding=row<npu_rows?std::abs(double(float(sycl::half(expected)))-expected):0;
                check(std::isfinite(actual.ptr[size_t(token)*rows+row])&&std::abs(double(actual.ptr[size_t(token)*rows+row])-expected)<=rounding+1e-3*(1+std::abs(expected)),"Async FP64 boundary mismatch");
            }
            ++cases;std::printf("{\"case\":%d,\"tokens\":%d,\"npu_rows\":%d,\"remote\":%s,\"packed\":%s,\"sync_bit_equal\":true,\"gate_bit_equal\":true,\"FP64\":true,\"unused_canary\":true,\"pending_guards\":true}\n",cases,tokens,npu_rows,remote?"true":"false",interleaved?"true":"false");std::fflush(stdout);
        }
        npu.reset();std::fill(actual.ptr,actual.ptr+size_t(max_tokens)*rows+32,poison);
        {
            strata::intel::NpuProjection retiring(q);strata::intel::NpuSplitOptions options;options.remote=true;
            sycl::event suffix;check(retiring.begin_split(gpu,14,raw,x.ptr,actual.ptr,rows,width,128,1024,options,suffix),"Destructor drain admission failed");
            gpu.dense(14,gate,x.ptr,gate_actual.ptr,gate_rows,width,128);
            // Intentionally abandon the request: destructor drains NPU+GPU,
            // but never merges into a destination whose caller abandoned it.
        }
        q.wait_and_throw();
        for(int t=0;t<128;++t)for(int row=0;row<1024;++row)check(actual.ptr[size_t(t)*rows+row]==poison,"Destructor unexpectedly merged prefix");
        check(hash(arena.ptr,qbytes+gbytes+packed_bytes)==weight_hash&&hash(x.ptr,size_t(max_tokens)*width*sizeof(float))==input_hash,"Async fixture inputs mutated");
        std::printf("{\"passed\":true,\"cases\":%d,\"destructor_drain\":true,\"read_only\":true}\n",cases);
    }catch(const std::exception& error){std::fprintf(stderr,"NPU async fixture: %s\n",error.what());return 1;}
}
