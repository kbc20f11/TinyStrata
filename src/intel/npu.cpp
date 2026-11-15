#include "strata/intel/npu.hpp"
#include "strata/intel/qwen_ops.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include "strata/intel/prefill_gemm.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#endif
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>

namespace strata::intel {
struct NpuProjection::Impl {
    sycl::queue& q;
    ov::Core core;
    sycl::half *weights = nullptr, *input = nullptr;
    float* output = nullptr;
    static constexpr size_t weight_elements = 8192 * 2048, activation_elements = 1024 * 8192;
    struct Graph {
        ov::CompiledModel compiled; ov::InferRequest request;
        ov::Tensor host_input, host_output, remote_input, remote_output;
        std::map<const sycl::half*, ov::Tensor> host_weights, remote_weights;
        const sycl::half* bound_weights = nullptr;
        bool bound_remote = false, binding_valid = false;
    };
    std::map<std::tuple<int,int,int>, Graph> graphs;
    Graph* pending_graph = nullptr;
    float* pending_destination = nullptr;
    int pending_rows = 0, pending_npu_rows = 0, pending_tokens = 0;
    bool pending_async = false;
    std::chrono::steady_clock::time_point pending_call_start, pending_npu_begin, pending_begin_end;
    static constexpr size_t graph_limit = 16;
    NpuSplitTiming timing;
    NpuSplitCounters counters;
    sycl::event timing_weight, timing_input;
    sycl::event timing_gpu_begin, timing_gpu_compute, timing_merge;
    size_t calls = 0;
    double wall_ms = 0;
    static constexpr size_t split_cache_elements = 64 * 1024 * 1024;
    size_t split_cache_used = 0, split_calls = 0, split_hits = 0;
    double split_compile_ms = 0, split_prepare_ms = 0, split_wait_ms = 0;
    ov::Tensor split_remote_input, split_remote_output;
    size_t split_gpu_done_npu_pending = 0;
    std::map<std::tuple<const uint8_t*, int, int, int>, sycl::half*> split_cache;
    explicit Impl(sycl::queue& queue) : q(queue) {
        // Query only the explicitly requested NPU. Enumerating every OV device
        // also initialized its OpenCL GPU plugin and loaded an SDK ICD.
        try { (void)core.get_property("NPU",ov::device::full_name); }
        catch(const std::exception&) { throw std::runtime_error("--prefill-npu requires an available Intel NPU; CPU fallback is disabled"); }
        try {
            weights = sycl::malloc_host<sycl::half>(weight_elements, q);
            input = sycl::aligned_alloc_host<sycl::half>(4096, activation_elements, q);
            output = sycl::aligned_alloc_host<float>(4096, activation_elements, q);
            if (!weights || !input || !output) throw std::runtime_error("Intel NPU projection scratch allocation failed");
        } catch (...) {
            if (weights) sycl::free(weights, q);
            if (input) sycl::free(input, q);
            if (output) sycl::free(output, q);
            throw;
        }
        std::fprintf(stderr, "Intel prefill_npu=enabled device=%s scratch_bytes=%zu runtime_weights=1 FP16_multiplicands=1\n",
            core.get_property("NPU", ov::device::full_name).c_str(), weight_elements * sizeof(sycl::half) + activation_elements * (sizeof(sycl::half) + sizeof(float)));
    }
    ~Impl() {
        if(pending_graph) {
            try {if(pending_async)pending_graph->request.wait();}
            catch(const std::exception& error){std::fprintf(stderr,"NPU pending drain: %s\n",error.what());}
            pending_graph=nullptr;
        }
        try { q.wait_and_throw(); } catch (...) {}
        const size_t shapes = graphs.size();
        graphs.clear();
        split_remote_input = {}; split_remote_output = {};
        // Requests/tensor views above retire before the independently aligned
        // slice owners. CPU_VA never receives an interior cache-arena pointer.
        for(const auto& entry:split_cache) {
            try {sycl::free(entry.second,q);} catch(const std::exception& error){std::fprintf(stderr,"NPU slice release: %s\n",error.what());}
        }
        for (void* buffer : {static_cast<void*>(output), static_cast<void*>(input), static_cast<void*>(weights)}) {
            try { sycl::free(buffer, q); } catch (const std::exception& error) { std::fprintf(stderr, "NPU scratch release: %s\n", error.what()); }
        }
        std::fprintf(stderr, "Intel NPU projection calls=%zu wall_ms=%.3f shapes=%zu\n", calls, wall_ms, shapes);
        if (split_calls) std::fprintf(stderr, "Intel NPU split calls=%zu cache_hits=%zu cache_bytes=%zu compile_ms=%.3f prepare_ms=%.3f wait_ms=%.3f gpu_done_npu_pending=%zu\n",
            split_calls, split_hits, split_cache_used * sizeof(sycl::half), split_compile_ms, split_prepare_ms, split_wait_ms, split_gpu_done_npu_pending);
        if(counters.eligible) std::fprintf(stderr,"Intel NPU coverage eligible=%zu served=%zu cache_capacity_skips=%zu graph_capacity_skips=%zu first_fit_no_eviction=1\n",
            counters.eligible,counters.served,counters.cache_capacity_skips,counters.graph_capacity_skips);
    }
    Graph& graph(int rows, int width, int tokens) {
        const auto key = std::make_tuple(rows, width, tokens);
        if (auto found = graphs.find(key); found != graphs.end()) return found->second;
        if (graphs.size() >= graph_limit) throw std::runtime_error("NPU compiled shape cache is full");
        const ov::Shape input_shape{size_t(tokens), size_t(width)}, weight_shape{size_t(rows), size_t(width)}, output_shape{size_t(tokens), size_t(rows)};
        auto x = std::make_shared<ov::opset13::Parameter>(ov::element::f16, input_shape);
        auto w = std::make_shared<ov::opset13::Parameter>(ov::element::f16, weight_shape);
        auto dot = std::make_shared<ov::opset13::MatMul>(x, w, false, true);
        auto result = std::make_shared<ov::opset13::Convert>(dot, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{x,w}, "TinyStrata runtime-weight projection");
        Graph entry;
        entry.compiled = core.compile_model(model, "NPU", {{"PERFORMANCE_HINT", "LATENCY"}});
        entry.request = entry.compiled.create_infer_request();
        entry.host_input = ov::Tensor(ov::element::f16, input_shape, input);
        entry.host_output = ov::Tensor(ov::element::f32, output_shape, output);
        return graphs.emplace(key, std::move(entry)).first->second;
    }
};
NpuProjection::NpuProjection(sycl::queue& q) : impl_(std::make_unique<Impl>(q)) {}
NpuProjection::~NpuProjection() = default;
sycl::event NpuProjection::run(int type, const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens) {
    auto& p = *impl_;
    if(p.pending_graph)throw std::logic_error("NPU split must be joined before another projection");
    if (rows < 1 || width < 1 || tokens < 1 || size_t(rows) * width > p.weight_elements ||
        size_t(tokens) * width > p.activation_elements || size_t(tokens) * rows > p.activation_elements)
        throw std::invalid_argument("Intel NPU projection exceeds bounded scratch");
    const auto start = std::chrono::steady_clock::now();
    auto& graph = p.graph(rows, width, tokens);
    auto weight_ready = dequantize_half(p.q, type, weights, p.weights, size_t(rows) * width);
    auto input_ready = convert_half(p.q, input, p.input, size_t(tokens) * width);
    weight_ready.wait_and_throw(); input_ready.wait_and_throw();
    // The full operator also permits out-of-order callers. Preserve its old
    // retirement fence before infer can overwrite the shared output buffer.
    if(!p.q.has_property<sycl::property::queue::in_order>()) p.q.wait_and_throw();
    graph.request.set_input_tensor(0, graph.host_input);
    graph.request.set_input_tensor(1, ov::Tensor(ov::element::f16, {size_t(rows),size_t(width)}, p.weights));
    graph.request.set_output_tensor(graph.host_output);
    graph.binding_valid = false;
    graph.request.infer();
    const auto event = p.q.memcpy(output, p.output, size_t(tokens) * rows * sizeof(float));
    ++p.calls;
    p.wall_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
    return event;
}
#ifdef TINYSTRATA_INTEL_MKL
bool NpuProjection::run_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
    float* output, int rows, int width, int tokens, int npu_rows, bool parallel, bool remote, sycl::event& completion, bool interleaved) {
    NpuSplitOptions options;
    options.parallel=parallel; options.remote=remote; options.interleaved=interleaved;
    return run_split(gpu,type,weights,input,output,rows,width,tokens,npu_rows,options,completion);
}
bool NpuProjection::run_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
    float* output, int rows, int width, int tokens, int npu_rows, const NpuSplitOptions& options, sycl::event& completion) {
    sycl::event GPUcompletion;
    if(!begin_split(gpu,type,weights,input,output,rows,width,tokens,npu_rows,options,GPUcompletion))return false;
    completion=finish_split();return true;
}
bool NpuProjection::begin_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
    float* output, int rows, int width, int tokens, int npu_rows, const NpuSplitOptions& options, sycl::event& GPUcompletion) {
    auto& p = *impl_;
    if(p.pending_graph)throw std::logic_error("Only one NPU split may be pending");
    p.timing = {};
    const bool parallel=options.parallel, remote=options.remote, interleaved=options.interleaved;
    const auto call_start=std::chrono::steady_clock::now();
    const auto elapsed=[](auto start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
    // The experiment is bounded to original Q6_K projection rows. Other
    // types/layouts keep the existing GPU path rather than guessing strides.
    if (type != 14 || width != 2048 || tokens < 128 || tokens > 1024 ||
        npu_rows < 256 || npu_rows % 256 || npu_rows >= rows ||
        size_t(rows) * tokens > p.activation_elements ||
        !p.q.has_property<sycl::property::queue::in_order>()) return false;
    ++p.counters.eligible;
    const auto graph_key=std::make_tuple(npu_rows,width,tokens);
    if (!p.graphs.count(graph_key) && p.graphs.size()>=p.graph_limit) {++p.counters.graph_capacity_skips;return false;}
    const auto key = std::make_tuple(weights, type + (interleaved ? 1000 : 0), npu_rows, width);
    const size_t elements = size_t(npu_rows) * width;
    auto found = p.split_cache.find(key);
    if (found == p.split_cache.end() && p.split_cache_used + elements > p.split_cache_elements) {++p.counters.cache_capacity_skips;return false;}
    auto started = std::chrono::steady_clock::now();
    auto& graph = p.graph(npu_rows, width, tokens);
    p.timing.compile_ms=elapsed(started); p.split_compile_ms += p.timing.compile_ms;
    p.timing.rows=rows; p.timing.npu_rows=npu_rows; p.timing.tokens=tokens;
    p.timing.cache_hit=found!=p.split_cache.end();
    p.timing.profiled=options.profile_stages && p.q.has_property<sycl::property::queue::enable_profiling>();
    if(p.timing.profiled) {
        // Explicit diagnostic fence separates predecessor drain from our
        // preparation. It is not present on the normal candidate path.
        const auto before=std::chrono::steady_clock::now();
        auto prior=p.q.ext_oneapi_submit_barrier();prior.wait_and_throw();
        p.timing.prior_gpu_drain_ms=elapsed(before);
    }
    started = std::chrono::steady_clock::now();
    sycl::half* prepared = nullptr;
    struct SliceDeleter {
        sycl::queue* queue;
        void operator()(sycl::half* ptr) const noexcept {
            if(!ptr)return;
            try {queue->wait_and_throw();} catch(...) {}
            try {sycl::free(ptr,*queue);} catch(...) {}
        }
    };
    std::unique_ptr<sycl::half,SliceDeleter> pending_slice(nullptr,SliceDeleter{&p.q});
    sycl::event weight_ready;
    if (found == p.split_cache.end()) {
        // The observed NPU CPU_VA import aliases interior pointers to the
        // registered base. Each cached slab instead owns an aligned BASE;
        // the total first-fit capacity remains the same 128MiB, no extra copy.
        prepared=sycl::aligned_alloc_host<sycl::half>(4096,elements,p.q);
        if(!prepared)throw std::runtime_error("NPU cached slice allocation failed");
        pending_slice.reset(prepared);
        if (!interleaved) weight_ready=dequantize_half(p.q, type, weights, prepared, elements);
        else weight_ready=p.q.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
            const size_t row = id[0] / width, col = id[0] % width;
            const size_t block = col / 256, local = col % 256;
            const size_t start = (((row / 16) * (width / 256) + block) * 53 * 16 + row % 16) * 4;
            const auto byte = [=](size_t at) { return weights[start + (at / 4) * 64 + at % 4]; };
            const size_t segment = local / 128, part = (local % 128) / 32, lane = local % 32;
            const int low = (byte(segment * 64 + (part % 2) * 32 + lane) >> (part >= 2 ? 4 : 0)) & 15;
            const int high = (byte(128 + segment * 32 + lane) >> (2 * part)) & 3;
            const int scale = int8_t(byte(192 + segment * 8 + part * 2 + lane / 16));
            const uint16_t bits = uint16_t(byte(208)) | (uint16_t(byte(209)) << 8);
            prepared[id[0]] = sycl::half(float(sycl::bit_cast<sycl::half>(bits)) * float(scale) * float((low | high << 4) - 32));
        });
    } else { prepared = found->second; ++p.split_hits; }
    p.timing.weight_prepare_ms=elapsed(started);
    started=std::chrono::steady_clock::now();
    auto input_ready=convert_half(p.q, input, p.input, size_t(tokens) * width);
    p.timing_input=input_ready;if(found==p.split_cache.end())p.timing_weight=weight_ready;
    p.timing.input_prepare_ms=elapsed(started);
    started=std::chrono::steady_clock::now();
    // Wait only for these buffers. The old global queue wait also drained
    // unrelated work, and did not identify conversion versus NPU waiting.
    if(found==p.split_cache.end()) weight_ready.wait_and_throw();
    input_ready.wait_and_throw();
    p.timing.prepare_wait_ms=elapsed(started);
    if (found == p.split_cache.end()) {
        p.split_cache.emplace(key, prepared);
        p.split_cache_used += elements;
        pending_slice.release();
    }
    const ov::Shape xshape{size_t(tokens), size_t(width)}, wshape{size_t(npu_rows), size_t(width)}, yshape{size_t(tokens), size_t(npu_rows)};
    started=std::chrono::steady_clock::now();
    const bool reuse=options.reuse_tensors && graph.binding_valid && graph.bound_remote==remote && graph.bound_weights==prepared;
    p.timing.tensor_reused=reuse;
    if (remote) {
        auto context = graph.compiled.get_context().as<ov::intel_npu::level_zero::ZeroContext>();
        const auto kind = ov::intel_npu::MemType::CPU_VA;
        const auto import_tensor=[&](const char* stage,ov::element::Type type,const ov::Shape& shape,void* ptr)->ov::Tensor {
            try {return context.create_tensor(type,shape,ptr,kind);}
            catch(const std::exception& error) {
                void *gpu_base=nullptr,*npu_base=nullptr;size_t gpu_span=0,npu_span=0;
                const auto gpu_context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(p.q.get_context());
                const auto gpu_result=zeMemGetAddressRange(gpu_context,ptr,&gpu_base,&gpu_span);
                const auto npu_result=zeMemGetAddressRange(static_cast<ze_context_handle_t>(context.get()),ptr,&npu_base,&npu_span);
                std::fprintf(stderr,"NPU remote import stage=%s ptr=%p requested_bytes=%zu gpu_ze=%x gpu_base=%p gpu_span=%zu npu_ze=%x npu_base=%p npu_span=%zu rows=%d tokens=%d\n",stage,ptr,ov::shape_size(shape)*type.size(),unsigned(gpu_result),gpu_base,gpu_span,unsigned(npu_result),npu_base,npu_span,npu_rows,tokens);
                throw std::runtime_error(std::string("NPU remote import ")+stage+": "+error.what());
            }
        };
        // Import each COMPLETE allocation once. Keeping a small first-shape
        // import alive otherwise prevents a later larger graph from wrapping
        // the same address range. All requests are destroyed before the owners.
        if (!p.split_remote_input) {
            auto input_region = import_tensor("input_full",ov::element::f16,{p.activation_elements},p.input);
            auto output_region = import_tensor("output_full",ov::element::f32,{p.activation_elements},p.output);
            p.split_remote_input = std::move(input_region);
            p.split_remote_output = std::move(output_region);
        }
        if(!reuse) {
            if(!options.reuse_tensors || !graph.remote_input) {
                graph.remote_input=import_tensor("input_view",ov::element::f16,xshape,p.input);
                graph.remote_output=import_tensor("output_view",ov::element::f32,yshape,p.output);
            }
            auto tensor=graph.remote_weights.find(prepared);
            if(!options.reuse_tensors || tensor==graph.remote_weights.end())
                tensor=graph.remote_weights.insert_or_assign(prepared,import_tensor("weights_view",ov::element::f16,wshape,prepared)).first;
            graph.request.set_input_tensor(0,graph.remote_input);
            graph.request.set_input_tensor(1,tensor->second);
            graph.request.set_output_tensor(graph.remote_output);
        }
    } else if(!reuse) {
        auto tensor=graph.host_weights.find(prepared);
        if(!options.reuse_tensors || tensor==graph.host_weights.end())
            tensor=graph.host_weights.insert_or_assign(prepared,ov::Tensor(ov::element::f16,wshape,prepared)).first;
        graph.request.set_input_tensor(0,graph.host_input);
        graph.request.set_input_tensor(1,tensor->second);
        graph.request.set_output_tensor(graph.host_output);
    }
    graph.bound_weights=prepared; graph.bound_remote=remote; graph.binding_valid=true;
    p.timing.tensor_bind_ms=elapsed(started);
    p.split_prepare_ms+=p.timing.weight_prepare_ms+p.timing.input_prepare_ms+p.timing.prepare_wait_ms+p.timing.tensor_bind_ms;
    // Independent buffers: GPU computes every remaining row while NPU is in
    // flight. The request is joined only after useful GPU work is submitted.
    const int gpu_rows = rows - npu_rows;
    const size_t stride = size_t(width / 256) * (interleaved ? 212 : 210);
    const auto npu_begin=std::chrono::steady_clock::now();
    try {
        started=std::chrono::steady_clock::now();
        if (parallel) graph.request.start_async();
        else graph.request.infer();
        p.timing.npu_submit_ms=elapsed(started);
        p.pending_graph=&graph;p.pending_destination=output;
        p.pending_rows=rows;p.pending_npu_rows=npu_rows;p.pending_tokens=tokens;p.pending_async=parallel;
        p.pending_call_start=call_start;p.pending_npu_begin=npu_begin;
        if(p.timing.profiled) p.timing_gpu_begin=p.q.ext_oneapi_submit_barrier();
        started=std::chrono::steady_clock::now();
        auto gpu_compute=interleaved?gpu.dense_interleaved_q6(weights + size_t(npu_rows) * stride, input, output+npu_rows, gpu_rows, width, tokens,rows):
            gpu.dense(type, weights + size_t(npu_rows) * stride, input, output+npu_rows, gpu_rows, width, tokens,rows);
        p.timing.gpu_submit_ms=elapsed(started);p.timing_gpu_compute=gpu_compute;
        // GEMM writes the final strided remainder directly: no 32MiB staging
        // allocation or gather kernel. Prefix rows remain exclusively NPU's.
        GPUcompletion=gpu_compute;
        p.pending_begin_end=std::chrono::steady_clock::now();
        p.timing.begin_host_ms=elapsed(call_start);
        return true;
    } catch (...) {
        if (parallel) { try { graph.request.wait(); } catch (...) {} }
        try {p.q.wait_and_throw();}catch(...){}
        p.pending_graph=nullptr;p.pending_destination=nullptr;
        throw;
    }
}
sycl::event NpuProjection::finish_split() {
    auto& p=*impl_;
    if(!p.pending_graph)throw std::logic_error("No NPU split is pending");
    auto& graph=*p.pending_graph;
    const auto elapsed=[](auto start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
    const bool parallel=p.pending_async;
    p.timing.deferred_gap_ms=elapsed(p.pending_begin_end);
    try {
        // The runtime status query can block; diagnostics measure it explicitly
        // and normal execution never makes this optional observation.
        if(parallel && p.timing.profiled) {
            const auto query=std::chrono::steady_clock::now();
            if(p.timing_gpu_compute.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete &&
               !graph.request.wait_for(std::chrono::milliseconds(0)))++p.split_gpu_done_npu_pending;
            p.timing.gpu_status_query_ms=elapsed(query);
        }
        const auto wait_begin=std::chrono::steady_clock::now();
        if(parallel)graph.request.wait();
        p.timing.npu_wait_ms=parallel?elapsed(wait_begin):0;
        p.timing.npu_host_span_ms=elapsed(p.pending_npu_begin);p.split_wait_ms+=p.timing.npu_wait_ms;
        const float* npu_result=p.output;float* output=p.pending_destination;
        const int rows=p.pending_rows,npu_rows=p.pending_npu_rows,tokens=p.pending_tokens;
        auto completion=p.q.submit([&](sycl::handler& handler){handler.depends_on(p.timing_gpu_compute);handler.parallel_for(sycl::range<1>(size_t(tokens)*npu_rows),[=](sycl::id<1> id){
            const size_t token=id[0]/npu_rows,row=id[0]%npu_rows;
            output[token*rows+row]=npu_result[id[0]];
        });});
        p.pending_graph=nullptr;p.pending_destination=nullptr;
        p.timing_merge=completion;p.timing.valid=true;p.timing.gpu_gather_ms=0;p.timing.host_ms=elapsed(p.pending_call_start);
        ++p.split_calls;++p.counters.served;if(p.timing.cache_hit)++p.counters.cache_hits;
        return completion;
    } catch(...) {
        if(parallel){try{graph.request.wait();}catch(...){}}
        try{p.q.wait_and_throw();}catch(...){}
        p.pending_graph=nullptr;p.pending_destination=nullptr;p.timing.valid=false;
        throw;
    }
}
bool NpuProjection::split_pending() const noexcept {return impl_->pending_graph!=nullptr;}
NpuSplitTiming NpuProjection::last_split_timing() const {
    const auto& p=*impl_;auto result=p.timing;
    if(!result.valid || !result.profiled || p.timing_merge.get_info<sycl::info::event::command_execution_status>()!=sycl::info::event_command_status::complete) return result;
    const auto begin=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_start>();};
    const auto end=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_end>();};
    result.gpu_remainder_ms=double(end(p.timing_gpu_compute)-end(p.timing_gpu_begin))*1e-6;
    result.gpu_compute_event_ms=double(end(p.timing_gpu_compute)-begin(p.timing_gpu_compute))*1e-6;
    result.weight_decode_gpu_ms=result.cache_hit?0:double(end(p.timing_weight)-begin(p.timing_weight))*1e-6;
    result.input_convert_gpu_ms=double(end(p.timing_input)-begin(p.timing_input))*1e-6;
    result.gpu_gather_ms=0; // Eliminated operation, not an unavailable timer.
    result.npu_merge_ms=double(end(p.timing_merge)-begin(p.timing_merge))*1e-6;
    return result;
}
NpuSplitCounters NpuProjection::split_counters() const {return impl_->counters;}
#endif
NpuMemoryUsage NpuProjection::memory_usage() const {
    const auto& p=*impl_;NpuMemoryUsage result;
    result.host_usm_bytes=p.weight_elements*sizeof(sycl::half)+p.activation_elements*(sizeof(sycl::half)+sizeof(float));
    if(p.counters.eligible){result.fp16_cache_capacity_bytes=p.split_cache_elements*sizeof(sycl::half);result.host_usm_bytes+=p.split_cache_used*sizeof(sycl::half);}
    result.fp16_cache_used_bytes=p.split_cache_used*sizeof(sycl::half);result.compiled_graphs=p.graphs.size();
    for(const auto& item:p.graphs) result.tensor_bindings+=2+item.second.host_weights.size()+item.second.remote_weights.size()+(item.second.remote_input?2:0);
    return result;
}
} // namespace strata::intel
