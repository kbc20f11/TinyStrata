#include "strata/intel/device.hpp"
#include "strata/intel/npu.hpp"
#include "strata/intel/prefill_gemm.hpp"
#include "strata/intel/quant.hpp"
#include "strata/intel/q6_interleave.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
double median(std::vector<double> v) { std::sort(v.begin(),v.end()); return v[v.size()/2]; }
int main(int argc, char** argv) {
    try {
        strata::paths::Utf8Arguments arguments(argc, argv); argc=arguments.argc(); argv=arguments.argv();
        if (argc != 2) throw std::runtime_error("Usage: intel_npu_split_test projection-fixture.bin");
        std::ifstream f(strata::paths::native_path(argv[1]),std::ios::binary);
        int rows=0,width=0;
        f.read(reinterpret_cast<char*>(&rows),4); f.read(reinterpret_cast<char*>(&width),4);
        if (!f || width!=2048 || (rows!=4096 && rows!=8192)) throw std::runtime_error("Invalid fixture");
        sycl::queue q(strata::intel::select_device(0), sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
        const size_t bytes=size_t(rows)*(width/256)*210, count=1024ull*rows;
        auto w=sycl::malloc_host<uint8_t>(bytes,q);
        auto x=sycl::malloc_shared<float>(1024ull*width,q);
        auto a=sycl::malloc_shared<float>(count+32,q), b=sycl::malloc_shared<float>(count+32,q);
        if (!w||!x||!a||!b) throw std::runtime_error("Allocation failed");
        f.read(reinterpret_cast<char*>(w),bytes);
        if (!f) throw std::runtime_error("Truncated quantized tensor");
        auto packed=sycl::malloc_host<uint8_t>(strata::intel::q6_interleaved_bytes(rows,width),q);
        if(!packed) throw std::runtime_error("Packed allocation failed");
        strata::intel::interleave_q6(w,packed,rows,width);
        for (size_t i=0;i<1024ull*width;++i) x[i]=std::sin(float(i%1987)*.013f)*.3f;
        strata::intel::PrefillGemm gpu(q,1,1024);
        strata::intel::NpuProjection npu(q);
        for (bool interleaved : {false,true}) for (int tokens : {128,257,1024}) {
            auto dense=[&](){ return interleaved?gpu.dense_interleaved_q6(packed,x,a,rows,width,tokens):gpu.dense(14,w,x,a,rows,width,tokens); };
            for (size_t i=size_t(tokens)*rows;i<size_t(tokens)*rows+32;++i) a[i]=b[i]=12345.f;
            dense().wait_and_throw();
            for (bool remote : {false,true}) {
                std::vector<double> full,serial,parallel;
                double maxerr=0; bool finite=true;
                for (int r=0;r<8;++r) {
                    auto started=Clock::now(); dense().wait_and_throw();
                    if(r>=3) full.push_back(milliseconds(started));
                    for (bool overlap : {false,true}) {
                        sycl::event end; started=Clock::now();
                        if (!npu.run_split(gpu,14,interleaved?packed:w,x,b,rows,width,tokens,1024,overlap,remote,end,interleaved)) throw std::runtime_error("Split rejected fixture");
                        end.wait_and_throw();
                        if(r>=3) (overlap?parallel:serial).push_back(milliseconds(started));
                    }
                }
                for (size_t i=0;i<size_t(tokens)*rows;++i) {
                    const double error=std::abs(double(b[i])-a[i]); maxerr=std::max(maxerr,error);
                    finite=finite&&std::isfinite(b[i]);
                    const double rounding=std::abs(double(float(sycl::half(a[i])))-a[i]);
                    if(!std::isfinite(b[i]) || error > (i%rows<1024?rounding+1e-3*(1+std::abs(a[i])):2e-4*(1+std::abs(a[i]))))
                        throw std::runtime_error("Split output differs beyond FP16 rounding plus tolerance");
                }
                for(size_t i=size_t(tokens)*rows;i<size_t(tokens)*rows+32;++i)
                    if(a[i]!=12345.f||b[i]!=12345.f) throw std::runtime_error("Output canary overwritten");
                // Independent FP64 dot samples, with precisely the same FP16
                // multiplicands. Include both sides of the output partition.
                for(int token : {0,tokens-1}) for(int row : {0,1023,1024,rows-1}) {
                    double expected=0;
                    for(int col=0;col<width;++col) expected+=double(float(sycl::half(x[size_t(token)*width+col]))) *
                        float(sycl::half(strata::intel::quant::value<14>(w+size_t(row)*(width/256)*210,col)));
                    const double error=std::abs(double(b[size_t(token)*rows+row])-expected);
                    const double round=std::abs(double(float(sycl::half(expected)))-expected);
                    if(error>(row<1024?round:0)+1e-3*(1+std::abs(expected))) throw std::runtime_error("FP64 sample mismatch");
                }
                std::printf("{\"rows\":%d,\"tokens\":%d,\"npu_rows\":1024,\"remote\":%s,\"interleaved\":%s,\"gpu_ms\":%.6f,\"serial_ms\":%.6f,\"parallel_ms\":%.6f,\"max_abs\":%.8g,\"finite\":true,\"canary\":true,\"fp64\":true}\n",
                    rows,tokens,remote?"true":"false",interleaved?"true":"false",median(full),median(serial),median(parallel),maxerr);
                std::fflush(stdout);
            }
        }
        q.wait_and_throw();
        sycl::free(packed,q);sycl::free(b,q);sycl::free(a,q);sycl::free(x,q);sycl::free(w,q);
    } catch(const std::exception& e) { std::fprintf(stderr,"NPU split test: %s\n",e.what()); return 1; }
}
