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
#include <vector>

namespace strata::intel {
struct NpuProjection::Impl {
    sycl::queue& q;
    ov::Core core;
    sycl::half* weights = nullptr;
    static constexpr size_t weight_elements = 8192 * 2048, activation_elements = 1024 * 8192;
    struct Binding {
        ov::InferRequest request;
        ov::Tensor host_input, host_output, remote_input, remote_output;
        std::map<const sycl::half*, ov::Tensor> host_weights, remote_weights;
        const sycl::half* bound_weights = nullptr;
        bool bound_remote = false, binding_valid = false;
    };
    struct Graph {
        ov::CompiledModel compiled;
        std::vector<Binding> bindings;
    };
    std::map<std::tuple<int,int,int>, Graph> graphs;
    struct Slot {
        sycl::half* input = nullptr;
        float* output = nullptr;
        Graph* pending_graph = nullptr;
        float* pending_destination = nullptr;
        int pending_rows = 0, pending_npu_rows = 0, pending_tokens = 0;
        bool pending_async = false;
        std::chrono::steady_clock::time_point pending_call_start, pending_npu_begin, pending_begin_end;
        NpuSplitTiming timing;
        sycl::event timing_weight, timing_input, timing_gpu_begin, timing_gpu_compute, timing_merge;
        ov::Tensor remote_input_region, remote_output_region;
        const float* prepared_source = nullptr;
        int prepared_width = 0, prepared_tokens = 0;
        sycl::event prepared_ready, consumer;
        bool prepared = false, consumer_valid = false;
    };
    std::vector<Slot> slots;
    static constexpr size_t graph_limit = 16;
    NpuSplitCounters counters;
    size_t calls = 0;
    double wall_ms = 0;
    static constexpr size_t split_cache_elements = 64 * 1024 * 1024;
    size_t split_cache_used = 0, split_calls = 0, split_hits = 0;
    double split_compile_ms = 0, split_prepare_ms = 0, split_wait_ms = 0;
    size_t split_gpu_done_npu_pending = 0;
    std::map<std::tuple<const uint8_t*, int, int, int>, sycl::half*> split_cache;
    Slot& slot(unsigned bank) {
        if(bank>=slots.size())throw std::invalid_argument("NPU bank exceeds admitted request slots");
        return slots[bank];
    }
    unsigned pending_count() const noexcept {
        unsigned count=0;for(const auto& slot:slots)count+=slot.pending_graph!=nullptr;return count;
    }
    explicit Impl(sycl::queue& queue,unsigned request_slots) : q(queue) {
        if(request_slots<1||request_slots>2)throw std::invalid_argument("Intel NPU request slots must be 1 or 2");
        slots.resize(request_slots);
        // Query only the explicitly requested NPU. Enumerating every OV device
        // also initialized its OpenCL GPU plugin and loaded an SDK ICD.
        try { (void)core.get_property("NPU",ov::device::full_name); }
        catch(const std::exception&) { throw std::runtime_error("--prefill-npu requires an available Intel NPU; CPU fallback is disabled"); }
        try {
            weights = sycl::malloc_host<sycl::half>(weight_elements, q);
            if(!weights)throw std::runtime_error("Intel NPU weight scratch allocation failed");
            for(auto& slot:slots){
                slot.input = sycl::aligned_alloc_host<sycl::half>(4096, activation_elements, q);
                slot.output = sycl::aligned_alloc_host<float>(4096, activation_elements, q);
                if(!slot.input||!slot.output)throw std::runtime_error("Intel NPU request-bank allocation failed");
            }
        } catch (...) {
            if (weights) sycl::free(weights, q);
            for(auto& slot:slots){if(slot.input)sycl::free(slot.input,q);if(slot.output)sycl::free(slot.output,q);}
            throw;
        }
        std::fprintf(stderr, "Intel prefill_npu=enabled device=%s scratch_bytes=%zu request_slots=%u runtime_weights=1 FP16_multiplicands=1\n",
            core.get_property("NPU", ov::device::full_name).c_str(), weight_elements * sizeof(sycl::half) + slots.size()*activation_elements * (sizeof(sycl::half) + sizeof(float)),request_slots);
    }
    ~Impl() {
        for(size_t bank=0;bank<slots.size();++bank)if(slots[bank].pending_graph) {
            auto& slot=slots[bank];
            try {if(slot.pending_async)slot.pending_graph->bindings[bank].request.wait();}
            catch(const std::exception& error){std::fprintf(stderr,"NPU pending drain: %s\n",error.what());}
            slot.pending_graph=nullptr;
        }
        try { q.wait_and_throw(); } catch (...) {}
        const size_t shapes = graphs.size();
        graphs.clear();
        for(auto& slot:slots){slot.remote_input_region={};slot.remote_output_region={};}
        // Requests/tensor views above retire before the independently aligned
        // slice owners. CPU_VA never receives an interior cache-arena pointer.
        for(const auto& entry:split_cache) {
            try {sycl::free(entry.second,q);} catch(const std::exception& error){std::fprintf(stderr,"NPU slice release: %s\n",error.what());}
        }
        for(auto& slot:slots)for(void* buffer:{static_cast<void*>(slot.output),static_cast<void*>(slot.input)}) {
            try {sycl::free(buffer,q);}catch(const std::exception& error){std::fprintf(stderr,"NPU bank release: %s\n",error.what());}
        }
        for (void* buffer : {static_cast<void*>(weights)}) {
            try { sycl::free(buffer, q); } catch (const std::exception& error) { std::fprintf(stderr, "NPU scratch release: %s\n", error.what()); }
        }
        std::fprintf(stderr, "Intel NPU projection calls=%zu wall_ms=%.3f shapes=%zu\n", calls, wall_ms, shapes);
        if (split_calls) std::fprintf(stderr, "Intel NPU split calls=%zu cache_hits=%zu cache_bytes=%zu compile_ms=%.3f prepare_ms=%.3f wait_ms=%.3f gpu_done_npu_pending=%zu\n",
            split_calls, split_hits, split_cache_used * sizeof(sycl::half), split_compile_ms, split_prepare_ms, split_wait_ms, split_gpu_done_npu_pending);
        if(counters.eligible) std::fprintf(stderr,"Intel NPU coverage eligible=%zu served=%zu cache_capacity_skips=%zu graph_capacity_skips=%zu first_fit_no_eviction=1 request_slots=%zu max_pending_requests=%zu\n",
            counters.eligible,counters.served,counters.cache_capacity_skips,counters.graph_capacity_skips,slots.size(),counters.max_pending_requests);
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
        entry.bindings.resize(slots.size());
        for(size_t bank=0;bank<slots.size();++bank){
            auto& binding=entry.bindings[bank];
            binding.request=entry.compiled.create_infer_request();
            binding.host_input=ov::Tensor(ov::element::f16,input_shape,slots[bank].input);
            binding.host_output=ov::Tensor(ov::element::f32,output_shape,slots[bank].output);
        }
        return graphs.emplace(key, std::move(entry)).first->second;
    }
};
NpuProjection::NpuProjection(sycl::queue& q,unsigned request_slots) : impl_(std::make_unique<Impl>(q,request_slots)) {}
NpuProjection::~NpuProjection() = default;
sycl::event NpuProjection::run(int type, const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens) {
    auto& p = *impl_;
    auto& s=p.slots[0];
    if(p.pending_count())throw std::logic_error("NPU split must be joined before another projection");
    if(!p.q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("Intel NPU full projection requires an in-order queue");
    if (rows < 1 || width < 1 || tokens < 1 || size_t(rows) * width > p.weight_elements ||
        size_t(tokens) * width > p.activation_elements || size_t(tokens) * rows > p.activation_elements)
        throw std::invalid_argument("Intel NPU projection exceeds bounded scratch");
    const auto start = std::chrono::steady_clock::now();
    auto& graph = p.graph(rows, width, tokens);
    auto& binding=graph.bindings[0];s.prepared=false;
    auto weight_ready = dequantize_half(p.q, type, weights, p.weights, size_t(rows) * width);
    auto input_ready = convert_half(p.q, input, s.input, size_t(tokens) * width);
    weight_ready.wait_and_throw(); input_ready.wait_and_throw();
    // On the supported in-order queue these readiness events also retire the
    // previous output copy before inference overwrites this shared bank.
    binding.request.set_input_tensor(0, binding.host_input);
    binding.request.set_input_tensor(1, ov::Tensor(ov::element::f16, {size_t(rows),size_t(width)}, p.weights));
    binding.request.set_output_tensor(binding.host_output);
    binding.binding_valid = false;
    binding.request.infer();
    const auto event = p.q.memcpy(output, s.output, size_t(tokens) * rows * sizeof(float));
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
sycl::event NpuProjection::prepare_split_input(unsigned bank,const float* input,int width,int tokens) {
    auto& p=*impl_;auto& s=p.slot(bank);
    if(s.pending_graph)throw std::logic_error("Cannot prepare a pending NPU request bank");
    if(!input||width!=2048||tokens<128||tokens>1024||!p.q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("Invalid prepared NPU input shape");
    const auto ready=convert_half(p.q,input,s.input,size_t(tokens)*width);
    s.prepared_source=input;s.prepared_width=width;s.prepared_tokens=tokens;
    s.prepared_ready=ready;s.prepared=true;
    return ready;
}
bool NpuProjection::begin_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
    float* output, int rows, int width, int tokens, int npu_rows, const NpuSplitOptions& options, sycl::event& GPUcompletion,unsigned bank) {
    auto& p = *impl_;
    auto& s=p.slot(bank);
    if(s.pending_graph)throw std::logic_error("NPU request bank is still pending");
    s.timing = {};
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
    auto& binding=graph.bindings[bank];
    s.timing.compile_ms=elapsed(started); p.split_compile_ms += s.timing.compile_ms;
    s.timing.rows=rows; s.timing.npu_rows=npu_rows; s.timing.tokens=tokens;
    s.timing.cache_hit=found!=p.split_cache.end();
    s.timing.profiled=options.profile_stages && p.q.has_property<sycl::property::queue::enable_profiling>();
    if(s.timing.profiled) {
        // Explicit diagnostic fence separates predecessor drain from our
        // preparation. It is not present on the normal candidate path.
        const auto before=std::chrono::steady_clock::now();
        auto prior=p.q.ext_oneapi_submit_barrier();prior.wait_and_throw();
        s.timing.prior_gpu_drain_ms=elapsed(before);
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
    s.timing.weight_prepare_ms=elapsed(started);
    started=std::chrono::steady_clock::now();
    if(s.prepared && (s.prepared_source!=input || s.prepared_width!=width || s.prepared_tokens!=tokens))
        throw std::invalid_argument("Prepared NPU bank input does not match projection");
    // The prior inference's GPU consumer must retire before this independent
    // output slot can be written again. No other bank is joined here.
    if(s.consumer_valid){s.consumer.wait_and_throw();s.consumer_valid=false;}
    auto input_ready=s.prepared?s.prepared_ready:convert_half(p.q,input,s.input,size_t(tokens)*width);
    s.prepared=false;
    s.timing_input=input_ready;if(found==p.split_cache.end())s.timing_weight=weight_ready;
    s.timing.input_prepare_ms=elapsed(started);
    started=std::chrono::steady_clock::now();
    // Wait only for these buffers. The old global queue wait also drained
    // unrelated work, and did not identify conversion versus NPU waiting.
    if(found==p.split_cache.end()) weight_ready.wait_and_throw();
    input_ready.wait_and_throw();
    s.timing.prepare_wait_ms=elapsed(started);
    if (found == p.split_cache.end()) {
        p.split_cache.emplace(key, prepared);
        p.split_cache_used += elements;
        pending_slice.release();
    }
    const ov::Shape xshape{size_t(tokens), size_t(width)}, wshape{size_t(npu_rows), size_t(width)}, yshape{size_t(tokens), size_t(npu_rows)};
    started=std::chrono::steady_clock::now();
    const bool reuse=options.reuse_tensors && binding.binding_valid && binding.bound_remote==remote && binding.bound_weights==prepared;
    s.timing.tensor_reused=reuse;
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
        if (!s.remote_input_region) {
            auto input_region = import_tensor("input_full",ov::element::f16,{p.activation_elements},s.input);
            auto output_region = import_tensor("output_full",ov::element::f32,{p.activation_elements},s.output);
            s.remote_input_region = std::move(input_region);
            s.remote_output_region = std::move(output_region);
        }
        if(!reuse) {
            if(!options.reuse_tensors || !binding.remote_input) {
                binding.remote_input=import_tensor("input_view",ov::element::f16,xshape,s.input);
                binding.remote_output=import_tensor("output_view",ov::element::f32,yshape,s.output);
            }
            auto tensor=binding.remote_weights.find(prepared);
            if(!options.reuse_tensors || tensor==binding.remote_weights.end())
                tensor=binding.remote_weights.insert_or_assign(prepared,import_tensor("weights_view",ov::element::f16,wshape,prepared)).first;
            binding.request.set_input_tensor(0,binding.remote_input);
            binding.request.set_input_tensor(1,tensor->second);
            binding.request.set_output_tensor(binding.remote_output);
        }
    } else if(!reuse) {
        auto tensor=binding.host_weights.find(prepared);
        if(!options.reuse_tensors || tensor==binding.host_weights.end())
            tensor=binding.host_weights.insert_or_assign(prepared,ov::Tensor(ov::element::f16,wshape,prepared)).first;
        binding.request.set_input_tensor(0,binding.host_input);
        binding.request.set_input_tensor(1,tensor->second);
        binding.request.set_output_tensor(binding.host_output);
    }
    binding.bound_weights=prepared; binding.bound_remote=remote; binding.binding_valid=true;
    s.timing.tensor_bind_ms=elapsed(started);
    p.split_prepare_ms+=s.timing.weight_prepare_ms+s.timing.input_prepare_ms+s.timing.prepare_wait_ms+s.timing.tensor_bind_ms;
    // Independent buffers: GPU computes every remaining row while NPU is in
    // flight. The request is joined only after useful GPU work is submitted.
    const int gpu_rows = rows - npu_rows;
    const size_t stride = size_t(width / 256) * (interleaved ? 212 : 210);
    const auto npu_begin=std::chrono::steady_clock::now();
    try {
        started=std::chrono::steady_clock::now();
        if (parallel) binding.request.start_async();
        else binding.request.infer();
        s.timing.npu_submit_ms=elapsed(started);
        s.pending_graph=&graph;s.pending_destination=output;
        s.pending_rows=rows;s.pending_npu_rows=npu_rows;s.pending_tokens=tokens;s.pending_async=parallel;
        s.pending_call_start=call_start;s.pending_npu_begin=npu_begin;
        p.counters.max_pending_requests=std::max(p.counters.max_pending_requests,size_t(p.pending_count()));
        if(s.timing.profiled) s.timing_gpu_begin=p.q.ext_oneapi_submit_barrier();
        started=std::chrono::steady_clock::now();
        auto gpu_compute=interleaved?gpu.dense_interleaved_q6(weights + size_t(npu_rows) * stride, input, output+npu_rows, gpu_rows, width, tokens,rows):
            gpu.dense(type, weights + size_t(npu_rows) * stride, input, output+npu_rows, gpu_rows, width, tokens,rows);
        s.timing.gpu_submit_ms=elapsed(started);s.timing_gpu_compute=gpu_compute;
        // GEMM writes the final strided remainder directly: no 32MiB staging
        // allocation or gather kernel. Prefix rows remain exclusively NPU's.
        GPUcompletion=gpu_compute;
        s.pending_begin_end=std::chrono::steady_clock::now();
        s.timing.begin_host_ms=elapsed(call_start);
        return true;
    } catch (...) {
        if (parallel) { try { binding.request.wait(); } catch (...) {} }
        try {p.q.wait_and_throw();}catch(...){}
        s.pending_graph=nullptr;s.pending_destination=nullptr;
        throw;
    }
}
sycl::event NpuProjection::finish_split(unsigned bank) {
    auto& p=*impl_;
    auto& s=p.slot(bank);
    if(!s.pending_graph)throw std::logic_error("No NPU split is pending");
    auto& graph=*s.pending_graph;
    auto& binding=graph.bindings[bank];
    const auto elapsed=[](auto start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
    const bool parallel=s.pending_async;
    s.timing.deferred_gap_ms=elapsed(s.pending_begin_end);
    try {
        // The runtime status query can block; diagnostics measure it explicitly
        // and normal execution never makes this optional observation.
        if(parallel && s.timing.profiled) {
            const auto query=std::chrono::steady_clock::now();
            if(s.timing_gpu_compute.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete &&
               !binding.request.wait_for(std::chrono::milliseconds(0)))++p.split_gpu_done_npu_pending;
            s.timing.gpu_status_query_ms=elapsed(query);
        }
        const auto wait_begin=std::chrono::steady_clock::now();
        if(parallel)binding.request.wait();
        s.timing.npu_wait_ms=parallel?elapsed(wait_begin):0;
        s.timing.npu_host_span_ms=elapsed(s.pending_npu_begin);p.split_wait_ms+=s.timing.npu_wait_ms;
        const float* npu_result=s.output;float* output=s.pending_destination;
        const int rows=s.pending_rows,npu_rows=s.pending_npu_rows,tokens=s.pending_tokens;
        auto completion=p.q.submit([&](sycl::handler& handler){handler.depends_on(s.timing_gpu_compute);handler.parallel_for(sycl::range<1>(size_t(tokens)*npu_rows),[=](sycl::id<1> id){
            const size_t token=id[0]/npu_rows,row=id[0]%npu_rows;
            output[token*rows+row]=npu_result[id[0]];
        });});
        s.pending_graph=nullptr;s.pending_destination=nullptr;
        s.consumer=completion;s.consumer_valid=true;
        s.timing_merge=completion;s.timing.valid=true;s.timing.gpu_gather_ms=0;s.timing.host_ms=elapsed(s.pending_call_start);
        ++p.split_calls;++p.counters.served;if(s.timing.cache_hit)++p.counters.cache_hits;
        return completion;
    } catch(...) {
        if(parallel){try{binding.request.wait();}catch(...){}}
        try{p.q.wait_and_throw();}catch(...){}
        s.pending_graph=nullptr;s.pending_destination=nullptr;s.timing.valid=false;
        throw;
    }
}
bool NpuProjection::split_pending(unsigned bank) const noexcept {return bank<impl_->slots.size()&&impl_->slots[bank].pending_graph!=nullptr;}
unsigned NpuProjection::pending_split_count() const noexcept {return impl_->pending_count();}
NpuSplitTiming NpuProjection::last_split_timing(unsigned bank) const {
    const auto& s=impl_->slot(bank);auto result=s.timing;
    if(!result.valid || !result.profiled || s.timing_merge.get_info<sycl::info::event::command_execution_status>()!=sycl::info::event_command_status::complete) return result;
    const auto begin=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_start>();};
    const auto end=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_end>();};
    result.gpu_remainder_ms=double(end(s.timing_gpu_compute)-end(s.timing_gpu_begin))*1e-6;
    result.gpu_compute_event_ms=double(end(s.timing_gpu_compute)-begin(s.timing_gpu_compute))*1e-6;
    result.weight_decode_gpu_ms=result.cache_hit?0:double(end(s.timing_weight)-begin(s.timing_weight))*1e-6;
    result.input_convert_gpu_ms=double(end(s.timing_input)-begin(s.timing_input))*1e-6;
    result.gpu_gather_ms=0; // Eliminated operation, not an unavailable timer.
    result.npu_merge_ms=double(end(s.timing_merge)-begin(s.timing_merge))*1e-6;
    return result;
}
NpuSplitCounters NpuProjection::split_counters() const {return impl_->counters;}
#endif
NpuMemoryUsage NpuProjection::memory_usage() const {
    const auto& p=*impl_;NpuMemoryUsage result;
    result.host_usm_bytes=p.weight_elements*sizeof(sycl::half)+p.slots.size()*p.activation_elements*(sizeof(sycl::half)+sizeof(float));
    result.request_slots=p.slots.size();result.infer_requests=p.graphs.size()*p.slots.size();
    result.opaque_runtime_reservation_bytes=p.slots.size()*(96ull<<20);
    if(p.counters.eligible){result.fp16_cache_capacity_bytes=p.split_cache_elements*sizeof(sycl::half);result.host_usm_bytes+=p.split_cache_used*sizeof(sycl::half);}
    result.fp16_cache_used_bytes=p.split_cache_used*sizeof(sycl::half);result.compiled_graphs=p.graphs.size();
    for(const auto& item:p.graphs)for(const auto& binding:item.second.bindings)
        result.tensor_bindings+=2+binding.host_weights.size()+binding.remote_weights.size()+(binding.remote_input?2:0);
    return result;
}
} // namespace strata::intel
