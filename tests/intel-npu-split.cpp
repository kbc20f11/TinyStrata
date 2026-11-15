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
#include <cstring>

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
double median(std::vector<double> v) { std::sort(v.begin(),v.end()); return v[v.size()/2]; }
int main(int argc, char** argv) {
    try {
        strata::paths::Utf8Arguments arguments(argc, argv); argc=arguments.argc(); argv=arguments.argv();
        const bool sweep=argc==3 && std::strcmp(argv[2],"--sweep")==0;
        const bool probe=argc==3 && std::strcmp(argv[2],"--probe")==0;
        if (argc != 2 && !sweep && !probe) throw std::runtime_error("Usage: intel_npu_split_test projection-fixture.bin [--sweep|--probe]");
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
        const auto tokens_to_test=probe?std::vector<int>{128}:sweep?std::vector<int>{128,256,512,1024}:std::vector<int>{128,257,1024};
        const auto rows_to_test=probe?std::vector<int>{rows/16,rows/8}:sweep?std::vector<int>{rows/16,rows/8,rows/4,rows/2}:std::vector<int>{1024};
        for (bool interleaved : (probe?std::vector<bool>{false}:std::vector<bool>{false,true})) for (int tokens : tokens_to_test) for(int npu_rows:rows_to_test) for(bool reuse:((sweep||probe)?std::vector<bool>{false,true}:std::vector<bool>{true})) {
            auto dense=[&](){ return interleaved?gpu.dense_interleaved_q6(packed,x,a,rows,width,tokens):gpu.dense(14,w,x,a,rows,width,tokens); };
            for (size_t i=size_t(tokens)*rows;i<size_t(tokens)*rows+32;++i) a[i]=b[i]=12345.f;
            dense().wait_and_throw();
            for (bool remote : {false,true}) {
                std::vector<double> full,serial,parallel;
                std::vector<strata::intel::NpuSplitTiming> stages;
                double maxerr=0; bool finite=true;
                for (int r=0;r<8;++r) {
                    auto started=Clock::now(); dense().wait_and_throw();
                    if(r>=3) full.push_back(milliseconds(started));
                    for (bool overlap : {false,true}) {
                        sycl::event end; started=Clock::now();
                        strata::intel::NpuSplitOptions options;
                        options.parallel=overlap;options.remote=remote;options.interleaved=interleaved;options.reuse_tensors=reuse;options.profile_stages=true;
                        if (!npu.run_split(gpu,14,interleaved?packed:w,x,b,rows,width,tokens,npu_rows,options,end)) throw std::runtime_error("Split rejected fixture");
                        end.wait_and_throw();
                        if(r>=3) (overlap?parallel:serial).push_back(milliseconds(started));
                        const auto timing=npu.last_split_timing();
                        if(!timing.valid || timing.npu_rows!=npu_rows || timing.gpu_remainder_ms<0 || timing.input_convert_gpu_ms<0)throw std::runtime_error("Missing completed stage profile");
                        if(r>=3 && overlap)stages.push_back(timing);
                        if(r==0) for(size_t i=0;i<size_t(tokens)*rows;++i) {
                            const double error=std::abs(double(b[i])-a[i]);
                            const double rounding=std::abs(double(float(sycl::half(a[i])))-a[i]);
                            if(!std::isfinite(b[i]) || error>(i%rows<size_t(npu_rows)?rounding+1e-3*(1+std::abs(a[i])):2e-4*(1+std::abs(a[i])))) {
                                const int row=int(i%rows),token=int(i/rows);double expected=0,alias512=0;
                                for(int col=0;col<width;++col) {
                                    const double xv=float(sycl::half(x[size_t(token)*width+col]));
                                    expected+=xv*float(sycl::half(strata::intel::quant::value<14>(w+size_t(row)*(width/256)*210,col)));
                                    alias512+=xv*float(sycl::half(strata::intel::quant::value<14>(w+size_t(row%512)*(width/256)*210,col)));
                                }
                                std::fprintf(stderr,"NPU mismatch index=%zu token=%d row=%d npu_rows=%d remote=%d parallel=%d reuse=%d got=%.12g gpu=%.12g fp64=%.12g alias_first512_fp64=%.12g delta=%.12g\n",i,token,row,npu_rows,int(remote),int(overlap),int(reuse),double(b[i]),double(a[i]),expected,alias512,error);
                                throw std::runtime_error("Serial/parallel output mismatch");
                            }
                        }
                    }
                }
                for (size_t i=0;i<size_t(tokens)*rows;++i) {
                    const double error=std::abs(double(b[i])-a[i]); maxerr=std::max(maxerr,error);
                    finite=finite&&std::isfinite(b[i]);
                    const double rounding=std::abs(double(float(sycl::half(a[i])))-a[i]);
                    if(!std::isfinite(b[i]) || error > (i%rows<size_t(npu_rows)?rounding+1e-3*(1+std::abs(a[i])):2e-4*(1+std::abs(a[i]))))
                        throw std::runtime_error("Split output differs beyond FP16 rounding plus tolerance");
                }
                for(size_t i=size_t(tokens)*rows;i<size_t(tokens)*rows+32;++i)
                    if(a[i]!=12345.f||b[i]!=12345.f) throw std::runtime_error("Output canary overwritten");
                // Independent FP64 dot samples, with precisely the same FP16
                // multiplicands. Include both sides of the output partition.
                for(int token : {0,tokens-1}) for(int row : {0,npu_rows-1,npu_rows,rows-1}) {
                    double expected=0;
                    for(int col=0;col<width;++col) expected+=double(float(sycl::half(x[size_t(token)*width+col]))) *
                        float(sycl::half(strata::intel::quant::value<14>(w+size_t(row)*(width/256)*210,col)));
                    const double error=std::abs(double(b[size_t(token)*rows+row])-expected);
                    const double round=std::abs(double(float(sycl::half(expected)))-expected);
                    if(error>(row<npu_rows?round:0)+1e-3*(1+std::abs(expected))) throw std::runtime_error("FP64 sample mismatch");
                }
                const auto stage_median=[&](double strata::intel::NpuSplitTiming::* member){std::vector<double> values;for(const auto& t:stages)values.push_back(t.*member);return median(values);};
                using T=strata::intel::NpuSplitTiming;
                const auto memory=npu.memory_usage();const auto coverage=npu.split_counters();
                std::printf("{\"rows\":%d,\"tokens\":%d,\"npu_rows\":%d,\"remote\":%s,\"interleaved\":%s,\"reuse_tensors\":%s,\"profiling_fences\":true,\"gpu_ms\":%.6f,\"serial_ms\":%.6f,\"parallel_ms\":%.6f,\"max_abs\":%.8g,\"finite\":true,\"canary\":true,\"fp64\":true,",
                    rows,tokens,npu_rows,remote?"true":"false",interleaved?"true":"false",reuse?"true":"false",median(full),median(serial),median(parallel),maxerr);
                std::printf("\"stages_ms\":{\"compile\":%.6f,\"prior_gpu_drain\":%.6f,\"weight_submit\":%.6f,\"input_submit\":%.6f,\"prepare_wait\":%.6f,\"bind\":%.6f,\"npu_submit\":%.6f,\"npu_wait\":%.6f,\"npu_host_span\":%.6f,\"gpu_submit\":%.6f,\"input_kernel\":%.6f,\"weight_kernel\":%.6f,\"gpu_remainder\":%.6f,\"gpu_compute_event\":%.6f,\"gpu_gather\":%.6f,\"npu_merge\":%.6f},",
                    stage_median(&T::compile_ms),stage_median(&T::prior_gpu_drain_ms),stage_median(&T::weight_prepare_ms),stage_median(&T::input_prepare_ms),stage_median(&T::prepare_wait_ms),stage_median(&T::tensor_bind_ms),stage_median(&T::npu_submit_ms),stage_median(&T::npu_wait_ms),stage_median(&T::npu_host_span_ms),stage_median(&T::gpu_submit_ms),stage_median(&T::input_convert_gpu_ms),stage_median(&T::weight_decode_gpu_ms),stage_median(&T::gpu_remainder_ms),stage_median(&T::gpu_compute_event_ms),stage_median(&T::gpu_gather_ms),stage_median(&T::npu_merge_ms));
                std::printf("\"cache_bytes\":%zu,\"host_usm_bytes\":%zu,\"device_usm_bytes\":%zu,\"opaque_reservation_bytes\":%zu,\"graph_count\":%zu,\"tensor_bindings\":%zu,\"eligible\":%zu,\"served\":%zu,\"cache_capacity_skips\":%zu,\"shape_capacity_skips\":%zu}\n",memory.fp16_cache_used_bytes,memory.host_usm_bytes,memory.device_usm_bytes,memory.opaque_runtime_reservation_bytes,memory.compiled_graphs,memory.tensor_bindings,coverage.eligible,coverage.served,coverage.cache_capacity_skips,coverage.graph_capacity_skips);
                std::fflush(stdout);
            }
        }
        q.wait_and_throw();
        sycl::free(packed,q);sycl::free(b,q);sycl::free(a,q);sycl::free(x,q);sycl::free(w,q);
    } catch(const std::exception& e) { std::fprintf(stderr,"NPU split test: %s\n",e.what()); return 1; }
}
