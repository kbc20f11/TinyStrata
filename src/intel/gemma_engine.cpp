// Native Gemma4 target-only causal text engine. Model formulas are the existing
// TinyStrata Gemma engine's formulas (MIT; LICENSE/NOTICE), not Qwen aliases.
#include "strata/intel/gemma_model.hpp"
#include "strata/intel/context_policy.hpp"
#include "strata/intel/gemma_ops.hpp"
#include "strata/intel/gemma_turn_snapshot.hpp"
#include "strata/intel/weight_store.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/execution_request.hpp"
#include "strata/intel/sampling_policy.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include "strata/intel/prefill_gemm.hpp"
#endif
#include "strata/common/command_inbox.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace strata::intel {
namespace {
class GemmaEngine {
    sycl::queue queue_;
    DeviceInfo device_;
    std::unique_ptr<WeightStore> weights_;
    GemmaGeometry geometry_;
    std::vector<GemmaKv> cache_;
    std::vector<void*> allocations_;
    context_policy::Footprint actual_allocations_;
    std::unique_ptr<GemmaAttention> attention_;
    std::unique_ptr<GemmaTurnSnapshot> turn_;
#ifdef TINYSTRATA_INTEL_MKL
    std::unique_ptr<PrefillGemm> gemm_;
#endif
    int context_,chunk_,position_=0;
    bool logits_valid_=false,pv_split_=false;
    float *residual_=nullptr,*norm_=nullptr,*query_raw_=nullptr,*key_raw_=nullptr,*value_raw_=nullptr;
    float *query_=nullptr,*key_=nullptr,*value_=nullptr,*attended_=nullptr,*attention_out_=nullptr;
    float *dense_gate_=nullptr,*dense_up_=nullptr,*dense_hidden_=nullptr,*dense_out_=nullptr,*dense_norm_=nullptr;
    float *router_input_=nullptr,*router_=nullptr,*expert_gu_=nullptr,*expert_hidden_=nullptr,*expert_down_=nullptr;
    float *expert_sum_=nullptr,*expert_norm_=nullptr,*weights_selected_=nullptr,*logits_=nullptr;
    int32_t *ids_=nullptr,*input_ids_=nullptr,*quant_error_=nullptr;
    int32_t *counts_=nullptr,*slots_=nullptr,*owners_=nullptr,*offsets_=nullptr,*tiles_=nullptr;
    std::vector<int> stops_;
    template<class T>T* allocate(size_t count,bool shared=false){
        if(!count||count>std::numeric_limits<size_t>::max()/sizeof(T))throw std::runtime_error("Gemma allocation overflow");
        auto* p=shared?sycl::malloc_shared<T>(count,queue_):sycl::malloc_device<T>(count,queue_);
        if(!p)throw std::runtime_error("Gemma USM allocation failed");allocations_.push_back(p);
        actual_allocations_.allocation(count,sizeof(T),shared);return p;
    }
    const ResidentTensor& tensor(const std::string& name) const {
        const auto it=weights_->tensors().find(name);if(it==weights_->tensors().end())throw std::runtime_error("Missing Gemma tensor "+name);return it->second;
    }
    const float* f32(const std::string& name) const {
        const auto& t=tensor(name);if(t.info.type!=0)throw std::runtime_error("Expected F32 Gemma tensor "+name);return reinterpret_cast<const float*>(t.data);
    }
    void matrix(const std::string& name,const float* input,float* output,int tokens){
        const auto& t=tensor(name);if(t.info.shape.size()!=2)throw std::runtime_error("Gemma dense matrix rank differs");
        const int rows=int(t.info.shape[1]),width=int(t.info.shape[0]);
#ifdef TINYSTRATA_INTEL_MKL
        if(gemm_&&tokens>=16&&t.info.type!=0){gemm_->dense(int(t.info.type),t.data,input,output,rows,width,tokens);return;}
#endif
        quantized_matmul(queue_,int(t.info.type),t.data,input,output,rows,width,tokens);
    }
    void expert_matrix(const ResidentTensor& t,ExpertPlan plan,const float* input,float* output,int rows,int width,bool separate,int tokens){
#ifdef TINYSTRATA_INTEL_MKL
        if(gemm_&&tokens>=16){gemm_->selected(int(t.info.type),t.data,t.bytes/geometry_.experts,plan,input,output,rows,width,separate);return;}
#endif
        if(tokens>=16)selected_matmul(queue_,int(t.info.type),t.data,plan,input,output,rows,width,separate);
        else selected_matvec(queue_,int(t.info.type),t.data,ids_,geometry_.picks,input,output,rows,width,separate,tokens);
    }
    void add(const float* input,float* destination,size_t count){queue_.parallel_for(sycl::range<1>(count),[=](sycl::id<1> i){destination[i]+=input[i];});}
    void release() noexcept {
        try{queue_.wait_and_throw();}catch(...){}
        turn_.reset();attention_.reset();
#ifdef TINYSTRATA_INTEL_MKL
        gemm_.reset();
#endif
        for(auto* p:allocations_)try{sycl::free(p,queue_);}catch(...){}allocations_.clear();weights_.reset();
    }
public:
    GemmaEngine(const std::string& path,unsigned device,int context,int chunk,WeightMemory memory,bool prefill_mkl)
        :queue_(select_device(device),sycl::property::queue::in_order{}),device_(inspect_device(queue_.get_device())),context_(context),chunk_(chunk){
        if(context<0||context>context_policy::full_limit||chunk<1||chunk>512)throw std::runtime_error("Gemma context must be auto or1..262144; chunk1..512 required");
        bool measured_runtime=false;
#ifdef TINYSTRATA_INTEL_AOT_TARGET
        measured_runtime=std::string(TINYSTRATA_INTEL_AOT_TARGET)=="intel_gpu_ptl_h";
#endif
        pv_split_=measured_runtime&&device_.vendor_id==0x8086&&device_.device_id==0xB082&&device_.integrated&&device_.ptl_h_aot;
        if(const char* value=std::getenv("TINYSTRATA_INTEL_GEMMA_SPLIT_PV")){
            const std::string mode(value);
            if(!mode.empty()&&mode!="0"&&mode!="1")throw std::runtime_error("Gemma PV candidate accepts only0/1");
            pv_split_=mode=="1";
        }
        try{
            ExecutionRequest loading;
            weights_=WeightStore::load_validated(queue_,path,memory,[&](const strata::GgufModel& m){geometry_=GemmaGeometry::validate(m);});
            const auto& g=geometry_;const size_t n=size_t(chunk_);
            context_policy::GemmaSpec context_shape;
            context_shape.chunk=chunk_;context_shape.mkl=prefill_mkl;context_shape.pv_split=pv_split_;
            const auto remaining=memory_budget(device_);
            const auto chosen=context_policy::select(context,g.model_context,remaining,device_.max_allocation,
                [&](uint64_t tokens){return context_policy::gemma(tokens,context_shape);});
            context_=chosen.tokens;
            size_t cache_elements=0;
            for(int l=0;l<g.layers;++l)cache_elements=std::max(cache_elements,g.capacity(l,context_,chunk_)*g.kv_heads[l]*g.dim(l));
            cache_.resize(g.layers);
            for(int l=0;l<g.layers;++l){auto& c=cache_[l];c.capacity=int(g.capacity(l,context_,chunk_));c.heads=g.kv_heads[l];c.dim=g.dim(l);c.window=g.sliding[l]?g.window:0;
                const size_t cells=size_t(c.capacity)*c.heads*c.dim;c.key=allocate<int8_t>(cells);c.value=allocate<int8_t>(cells);c.key_scale=allocate<sycl::half>(cells/32);c.value_scale=allocate<sycl::half>(cells/32);}
            turn_=std::make_unique<GemmaTurnSnapshot>(cache_);
            if(turn_->bytes()!=gemma_prefix::snapshot_bytes(std::min(size_t(context_),size_t(g.window+chunk_)),8,g.swa_dim,25))
                throw std::logic_error("Gemma turn snapshot differs from admitted host footprint");
            actual_allocations_.host(turn_->bytes());
            std::printf("INFO gemma_turn_checkpoint=1 turn_snapshot_host_bytes=%zu\n",turn_->bytes());std::fflush(stdout);
            residual_=allocate<float>(n*g.width);norm_=allocate<float>(n*g.width);
            query_raw_=allocate<float>(n*g.heads*g.full_dim);query_=allocate<float>(n*g.heads*g.full_dim);
            key_raw_=allocate<float>(n*8*g.swa_dim);value_raw_=allocate<float>(n*8*g.swa_dim);key_=allocate<float>(n*8*g.swa_dim);value_=allocate<float>(n*8*g.swa_dim);
            attended_=allocate<float>(n*g.heads*g.full_dim);attention_out_=allocate<float>(n*g.width);
            dense_gate_=allocate<float>(n*g.dense_ff);dense_up_=allocate<float>(n*g.dense_ff);dense_hidden_=allocate<float>(n*g.dense_ff);
            dense_out_=allocate<float>(n*g.width);dense_norm_=allocate<float>(n*g.width);router_input_=allocate<float>(n*g.width);router_=allocate<float>(n*g.experts);
            expert_gu_=allocate<float>(n*g.picks*2*g.expert_ff);expert_hidden_=allocate<float>(n*g.picks*g.expert_ff);expert_down_=allocate<float>(n*g.picks*g.width);
            expert_sum_=allocate<float>(n*g.width);expert_norm_=allocate<float>(n*g.width);weights_selected_=allocate<float>(n*g.picks);ids_=allocate<int32_t>(n*g.picks);
            input_ids_=allocate<int32_t>(n,true);quant_error_=allocate<int32_t>(1,true);logits_=allocate<float>(g.vocabulary,true);
            counts_=allocate<int32_t>(g.experts);slots_=allocate<int32_t>(n*g.experts);
            const size_t max_tiles=(n*g.picks+3)/4+g.experts;owners_=allocate<int32_t>(max_tiles);offsets_=allocate<int32_t>(max_tiles);tiles_=allocate<int32_t>(1);
            if(actual_allocations_.device!=chosen.bytes.owned_device||actual_allocations_.host_visible!=chosen.bytes.owned_host_visible||actual_allocations_.largest>chosen.bytes.largest||actual_allocations_.host_only!=chosen.bytes.host_only-(16ull<<20))
                throw std::logic_error("Gemma context planner differs from actual engine allocations");
            attention_=std::make_unique<GemmaAttention>(queue_,context_,chunk_,g.heads,g.full_dim,cache_elements,prefill_mkl,pv_split_);
            std::fprintf(stderr,"Intel Gemma split_pv=%d decode_only=1 min_visible=1024 key_tile=256 scratch_bytes=%llu\n",int(pv_split_),
                (unsigned long long)(pv_split_?uint64_t(g.heads)*((context_+255)/256)*g.full_dim*sizeof(float):0));
            std::printf("INFO gemma_pv_split=%d\n",int(pv_split_));std::fflush(stdout);
#ifdef TINYSTRATA_INTEL_MKL
            if(prefill_mkl)gemm_=std::make_unique<PrefillGemm>(queue_,1,chunk_);
#endif
            stops_=g.terminal_tokens;
            std::fprintf(stderr,"Intel Gemma terminal_ids=");for(int id:stops_)std::fprintf(stderr,"%d,",id);std::fprintf(stderr," full_vocabulary_lookup=1\n");
            std::fprintf(stderr,"Intel Gemma4 target_only=1 layers=%d width=%d SWA_window=%d full_dim=%d swa_dim=%d experts=%d picks=%d KV=int8 cache_bytes=%llu single_weight_store=1\n",g.layers,g.width,g.window,g.full_dim,g.swa_dim,g.experts,g.picks,(unsigned long long)g.cache_bytes(context_,chunk_));
            context_policy::report(chosen,g.model_context,remaining,device_.max_allocation);
        }catch(...){release();throw;}
    }
    ~GemmaEngine(){release();}
    const DeviceInfo& device_info() const{return device_;}
    int context() const{return context_;}int chunk() const{return chunk_;}int consumed() const{return position_;}int vocabulary() const{return geometry_.vocabulary;}
    bool terminal(int token) const{return std::find(stops_.begin(),stops_.end(),token)!=stops_.end();}
    bool logits_valid()const{return logits_valid_;}
    bool turn_valid()const{return turn_->valid();}
    void invalidate_turn(){turn_->invalidate();}
    void reset(){queue_.wait_and_throw();position_=0;logits_valid_=false;invalidate_turn();}
    bool save_turn(size_t logical,const std::atomic<bool>* cancelled){return turn_->save(queue_,logical,size_t(position_),size_t(chunk_),cancelled);}
    void restore_turn(){if(turn_->position()>size_t(position_))throw std::logic_error("Gemma turn is ahead of resident KV");position_=int(turn_->restore(queue_));logits_valid_=false;}
    void rewind_one(){
        if(position_<1)throw std::logic_error("Gemma cannot replay before position0");
        for(const auto& c:cache_)if(c.window&&!gemma_prefix::ring_retains(size_t(position_-1),size_t(position_),c.capacity,c.window))
            throw std::logic_error("Gemma replay no longer retains the required window");
        queue_.wait_and_throw();--position_;logits_valid_=false;
    }
    const float* logits(){if(!logits_valid_)throw std::runtime_error("Gemma logits are not available");return logits_;}
    const float* step_many(const int* tokens,int count,bool final=true,const std::atomic<bool>* cancelled=nullptr){
        if(count<1||count>chunk_||position_+count>context_)throw std::runtime_error("Gemma step exceeds context/chunk");
        for(int i=0;i<count;++i)if(tokens[i]<0||tokens[i]>=vocabulary())throw std::runtime_error("Gemma token outside vocabulary");
        if(cancelled&&cancelled->load())return nullptr;
        const auto& g=geometry_;const int width=g.width,picks=g.picks,ff=g.expert_ff,heads=g.heads;const size_t all=size_t(count)*width;
        std::copy_n(tokens,count,input_ids_);*quant_error_=0;logits_valid_=false;
        const auto& table=tensor("token_embd.weight");embedding_lookup(queue_,int(table.info.type),table.data,input_ids_,residual_,width,count);
        float* residual=residual_;const float embed=std::sqrt(float(width));queue_.parallel_for(sycl::range<1>(all),[=](sycl::id<1> i){residual[i]*=embed;});
        for(int l=0;l<g.layers;++l){
            if(cancelled&&cancelled->load()){queue_.wait_and_throw();reset();return nullptr;}
            const std::string p="blk."+std::to_string(l)+".";const int dim=g.dim(l),kv=g.kv_heads[l];
            rms_norm(queue_,residual_,f32(p+"attn_norm.weight"),norm_,count,width,g.epsilon);
            matrix(p+"attn_q.weight",norm_,query_raw_,count);matrix(p+"attn_k.weight",norm_,key_raw_,count);
            const float* raw_value=key_raw_;
            if(g.sliding[l]){matrix(p+"attn_v.weight",norm_,value_raw_,count);raw_value=value_raw_;}
            const float* frequency=g.sliding[l]?nullptr:f32("rope_freqs.weight");
            gemma_head_norm_rope(queue_,query_raw_,f32(p+"attn_q_norm.weight"),query_,heads,dim,g.rotation(l),position_,count,g.epsilon,g.base(l),frequency);
            gemma_head_norm_rope(queue_,key_raw_,f32(p+"attn_k_norm.weight"),key_,kv,dim,g.rotation(l),position_,count,g.epsilon,g.base(l),frequency);
            gemma_head_norm_rope(queue_,raw_value,nullptr,value_,kv,dim,0,position_,count,g.epsilon,1.f);
            gemma_store_q8(queue_,key_,value_,cache_[l],position_,count,quant_error_);
            attention_->run(query_,cache_[l],attended_,heads,position_,count);matrix(p+"attn_output.weight",attended_,attention_out_,count);
            rms_norm(queue_,attention_out_,f32(p+"post_attention_norm.weight"),norm_,count,width,g.epsilon);add(norm_,residual_,all);
            // Router, dense MLP and routed FFN read differently normalized
            // copies of this same post-attention residual.
            rms_norm(queue_,residual_,f32(p+"ffn_gate_inp.scale"),router_input_,count,width,g.epsilon);
            matrix(p+"ffn_gate_inp.weight",router_input_,router_,count);
            gemma_route(queue_,router_,f32(p+"ffn_down_exps.scale"),ids_,weights_selected_,g.experts,picks,count,width);
            rms_norm(queue_,residual_,f32(p+"ffn_norm.weight"),norm_,count,width,g.epsilon);
            matrix(p+"ffn_gate.weight",norm_,dense_gate_,count);matrix(p+"ffn_up.weight",norm_,dense_up_,count);
            gemma_geglu(queue_,dense_gate_,dense_up_,dense_hidden_,size_t(count)*g.dense_ff);matrix(p+"ffn_down.weight",dense_hidden_,dense_out_,count);
            rms_norm(queue_,dense_out_,f32(p+"post_ffw_norm_1.weight"),dense_norm_,count,width,g.epsilon);
            rms_norm(queue_,residual_,f32(p+"pre_ffw_norm_2.weight"),norm_,count,width,g.epsilon);
            ExpertPlan plan{counts_,slots_,owners_,offsets_,tiles_,g.experts,count,picks};if(count>=16)plan_experts(queue_,ids_,plan);
            expert_matrix(tensor(p+"ffn_gate_up_exps.weight"),plan,norm_,expert_gu_,2*ff,width,false,count);
            gemma_geglu_packed(queue_,expert_gu_,expert_hidden_,ff,count*picks);
            expert_matrix(tensor(p+"ffn_down_exps.weight"),plan,expert_hidden_,expert_down_,width,ff,true,count);
            float *sum=expert_sum_;const float *down=expert_down_,*weight=weights_selected_;
            queue_.parallel_for(sycl::range<1>(all),[=](sycl::id<1> i){const size_t token=i[0]/width,col=i[0]%width;float v=0;for(int k=0;k<picks;++k)v+=weight[token*picks+k]*down[(token*picks+k)*width+col];sum[i]=v;});
            rms_norm(queue_,expert_sum_,f32(p+"post_ffw_norm_2.weight"),expert_norm_,count,width,g.epsilon);add(dense_norm_,expert_norm_,all);
            rms_norm(queue_,expert_norm_,f32(p+"post_ffw_norm.weight"),norm_,count,width,g.epsilon);add(norm_,residual_,all);
            const float* scale=f32(p+"layer_output_scale.weight");queue_.parallel_for(sycl::range<1>(all),[=](sycl::id<1> i){residual[i]*=scale[0];});
        }
        if(final){
            rms_norm(queue_,residual_+size_t(count-1)*width,f32("output_norm.weight"),norm_,1,width,g.epsilon);
            quantized_matvec(queue_,int(table.info.type),table.data,norm_,logits_,g.vocabulary,width);
            float* output=logits_;const float cap=g.softcap;queue_.parallel_for(sycl::range<1>(size_t(g.vocabulary)),[=](sycl::id<1> i){output[i]=cap*sycl::tanh(output[i]/cap);});
        }
        queue_.wait_and_throw();if(*quant_error_)throw std::runtime_error("Nonfinite/overflow Gemma INT8 KV input");
        if(cancelled&&cancelled->load()){reset();return nullptr;}
        position_+=count;logits_valid_=final;return final?logits_:residual_;
    }
};

std::vector<int> parse_tokens(std::string text){
    std::replace(text.begin(),text.end(),',',' ');std::istringstream stream(text);std::vector<int> ids;int64_t id;
    while(stream>>id){if(id<0||id>std::numeric_limits<int>::max())throw std::runtime_error("Invalid token ID");ids.push_back(int(id));}
    if(!stream.eof()||ids.empty())throw std::runtime_error("Expected nonempty token ID list");return ids;
}
int integer(const std::string& text,int maximum){
    if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Expected nonnegative integer");
    const uint64_t v=std::stoull(text);if(v>uint64_t(maximum))throw std::runtime_error("Integer outside supported range");return int(v);
}
void option(sampling_policy::Sampling& s,const std::string& text){
    const auto at=text.find('=');const std::string k=text.substr(0,at),v=text.substr(at+1);size_t used=0;
    if(k=="seed"){if(v.empty()||v.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Invalid seed");s.seed=std::stoull(v);return;}
    const float n=std::stof(v,&used);if(used!=v.size()||!std::isfinite(n))throw std::runtime_error("Invalid sampler field");
    if(k=="temperature"&&n>=0)s.temperature=n;
    else if(k=="top_p"&&n>0&&n<=1)s.top_p=n;
    else if(k=="min_p"&&n>=0&&n<=1)s.min_p=n;
    else if(k=="top_k")s.top_k=integer(v,64);
    else if(k=="penalty_repeat"&&n>0)s.repeat=n;
    else if(k=="penalty_freq")s.frequency=n;
    else if(k=="penalty_present")s.presence=n;
    else if(k=="penalty_last_n"&&n>=-1&&n<=32768&&n==std::floor(n))s.last_n=int(n);
    else if((k=="mtp_auto"||k=="lookup_draft")&&n==0){}
    else throw std::runtime_error("Unsupported Gemma request option: "+text);
}
int serve(GemmaEngine& e){
    const auto& d=e.device_info();
    std::printf("INFO backend=sycl experimental=1 model=gemma-4-26b-a4b-it family=gemma4 kv=int8 mtp=0 vision=0 memory_model=%s single_weight_store=1 expert_gpu_percent=100 expert_cpu_percent=0 device_vendor=%u device_id=%u device_integrated=%d device_architecture=%s adapter_luid=%s adapter_node_mask=%u kv_used_tokens=0\nREADY %d stop\n",d.integrated?"uma":"discrete",d.vendor_id,d.device_id,int(d.integrated),d.architecture.c_str(),d.adapter_luid.empty()?"unknown":d.adapter_luid.c_str(),d.adapter_node_mask,e.context());std::fflush(stdout);
    strata::common::CommandInbox inbox;std::string line;std::vector<int> live,saved;
    int reported_position=0;
    const auto report_position=[&]{if(reported_position!=e.consumed()){reported_position=e.consumed();std::printf("INFO kv_used_tokens=%d\n",reported_position);std::fflush(stdout);}};
    while(inbox.next(line)){
        if(line=="QUIT")return 0;if(line.empty())continue;
        std::vector<int> prompt;int count=0;sampling_policy::Sampling sampling;
        bool checkpoint=true,seen_checkpoint=false,seen_turn=false;size_t turn_at=0;
        try{
            if(line.rfind("GEN ",0)!=0)throw std::runtime_error("Gemma target-only route supports text GEN only");
            std::istringstream in(line.substr(4));std::string field,ids;
            if(!(in>>field))throw std::runtime_error("Missing output count");count=integer(field,e.context());
            while(in>>field){if(ids.empty()&&field.find('=')!=std::string::npos){
                if(field.rfind("ckpt=",0)==0){if(seen_checkpoint)throw std::runtime_error("Duplicate Gemma ckpt option");checkpoint=integer(field.substr(5),1)!=0;seen_checkpoint=true;}
                else if(field.rfind("turn_at=",0)==0){if(seen_turn)throw std::runtime_error("Duplicate Gemma turn_at option");turn_at=size_t(integer(field.substr(8),e.context()));seen_turn=true;}
                else option(sampling,field);
            }else{if(!ids.empty())ids+=' ';ids+=field;}}
            prompt=parse_tokens(ids);if(count<1||prompt.size()+size_t(count)>size_t(e.context()))throw std::runtime_error("Gemma request exceeds context");
            for(int id:prompt)if(id>=e.vocabulary())throw std::runtime_error("Gemma token outside vocabulary");
            turn_at=gemma_prefix::turn_boundary(turn_at,prompt.size());
        }catch(const std::exception& error){std::printf("ERROR %s\n",error.what());std::fflush(stdout);continue;}
        ExecutionRequest active;const auto start=std::chrono::steady_clock::now();
        if(live.size()!=size_t(e.consumed()))throw std::logic_error("Gemma committed history/position mismatch");
        const auto reuse=gemma_prefix::choose(live,prompt,saved,size_t(e.consumed()),e.logits_valid(),e.turn_valid(),checkpoint);
        if(reuse.source==gemma_prefix::Source::reset){e.reset();live.clear();saved.clear();}
        else if(reuse.source==gemma_prefix::Source::saved_turn){e.restore_turn();live=saved;}
        else if(reuse.source==gemma_prefix::Source::replay_last){e.rewind_one();live.resize(reuse.tokens);}
        const size_t reused=reuse.tokens;
        report_position();std::printf("RESUME %zu\n",reused);std::fflush(stdout);
        size_t consumed=reused;const float* logits=nullptr;
        for(;consumed<prompt.size()&&!inbox.cancelled().load();){const int n=int(std::min(size_t(e.chunk()),prompt.size()-consumed));
            const size_t before=consumed;
            logits=e.step_many(prompt.data()+consumed,n,consumed+n==prompt.size(),&inbox.cancelled());if(!logits)break;consumed+=n;
            live.insert(live.end(),prompt.begin()+before,prompt.begin()+consumed);
            // Keep the existing cold-prefill chunks. Save their complete SWA
            // rings but attach only the requested logical boundary. Causal
            // masking ignores later rows, and every fresh suffix rewrites them.
            if(checkpoint&&before<turn_at&&turn_at<=consumed){
                saved.clear();
                if(e.save_turn(turn_at,&inbox.cancelled()))saved.assign(prompt.begin(),prompt.begin()+turn_at);
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            report_position();std::printf("PP %zu %zu %.3f\n",consumed,prompt.size(),ms);std::fflush(stdout);}
        const auto decode_start=std::chrono::steady_clock::now();const double prefill=std::chrono::duration<double,std::milli>(decode_start-start).count();
        std::vector<int> history(prompt);std::mt19937_64 random(sampling.seed?sampling.seed:std::random_device{}());int produced=0;const char* finish="length";
        if(inbox.cancelled().load())finish="cancel";
        else for(int n=0;n<count;++n){
            if(inbox.cancelled().load()){finish="cancel";break;}
            if(n==0)logits=e.logits();
            const int token=sampling_policy::sample(logits,e.vocabulary(),sampling,history,random);++produced;history.push_back(token);
            std::printf("T %d\n",token);std::fflush(stdout);if(e.terminal(token)){finish="stop";break;}
            if(n+1<count){logits=e.step_many(&token,1,true,&inbox.cancelled());if(!logits){finish="cancel";break;}live.push_back(token);report_position();}
        }
        if(inbox.cancelled().load()){e.reset();live.clear();saved.clear();}
        report_position();
        const double decode=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_start).count();
        std::printf("DONE %d %zu %.3f %.3f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n",produced,prompt.size(),prefill,decode,finish,reused,consumed-reused);std::fflush(stdout);
    }
    return 0;
}
}
}

int main(int argc,char** argv){
    try{
        strata::paths::Utf8Arguments arguments(argc,argv);argc=arguments.argc();argv=arguments.argv();
        if(strata::intel::print_device_report_if_requested(argc,argv))return 0;
        std::string model,tokens,token_file,dump;unsigned device=0;int context=0,chunk=512;bool server=false;
#ifdef TINYSTRATA_INTEL_MKL
        bool mkl=true;
#else
        bool mkl=false;
#endif
        auto memory=strata::intel::WeightMemory::Host;
        for(int i=1;i<argc;++i){const std::string option=argv[i];
            if(option=="--help"){std::puts("tinystrata-intel-gemma4 --model TARGET.gguf [--serve | --tokens IDS | --tokens-file FILE] [--dump FILE] [--max-context auto|N] [--chunk N] [--kv-type int8] [--mtp 0] [--prefill-mkl | --no-prefill-mkl] [--device N] [--weight-memory host|shared|device]\nExperimental target-only causal text path. Gemma INT8 KV is the default; assistant/speculation/FP8/images are not implemented by this executable.");return 0;}
            if(option=="--capabilities"){
#ifdef TINYSTRATA_INTEL_MKL
                constexpr const char* mkl="true";
#else
                constexpr const char* mkl="false";
#endif
#ifdef TINYSTRATA_INTEL_INT8_XMX
                constexpr const char* int8_xmx="true";
#else
                constexpr const char* int8_xmx="false";
#endif
#ifdef TINYSTRATA_INTEL_AOT_TARGET
                constexpr const char* aot=TINYSTRATA_INTEL_AOT_TARGET;
#else
                constexpr const char* aot="";
#endif
                std::printf("{\"backend\":\"sycl\",\"experimental\":true,\"model_families\":[],"
                    "\"experimental_models\":[\"gemma-4-26b-a4b-it\"],\"experimental_profiles\":{\"b390-gemma4\":1,\"gemma4-target-only\":1},"
                    "\"experimental_model_contracts\":{\"gemma-4-26b-a4b-it\":{\"version\":1,\"architecture\":\"gemma4\",\"kv_types\":[\"int8\"],\"mtp\":false,\"experimental_mtp_max_depth\":0,\"images\":false,\"thinking\":\"on_off\",\"mode\":\"target_only\",\"context_max\":262144}},"
                    "\"kv_types\":[\"int8\"],\"images\":false,\"mtp\":false,\"experimental_mtp_max_depth\":0,\"serve\":true,\"capabilities_schema\":1,\"context_auto\":true,\"max_context\":262144,\"target_only\":true,"
                    "\"build\":{\"mkl\":%s,\"int8_xmx\":%s,\"aot_target\":\"%s\"},"
                    "\"weight_types\":{\"scalar\":[0,2,14],\"dp4a\":[],\"int8_xmx\":[]}}\n",mkl,int8_xmx,aot);return 0;}
            if(option=="--serve"){server=true;continue;}if(option=="--prefill-mkl"){mkl=true;continue;}if(option=="--no-prefill-mkl"){mkl=false;continue;}
            if(i+1>=argc)throw std::runtime_error("Missing value for "+option);const std::string value=argv[++i];
            if(option=="--model")model=value;else if(option=="--tokens")tokens=value;else if(option=="--tokens-file")token_file=value;else if(option=="--dump")dump=value;
            else if(option=="--device")device=unsigned(strata::intel::integer(value,1024));
            else if(option=="--max-context")context=strata::intel::context_policy::parse(value,strata::intel::context_policy::full_limit);else if(option=="--chunk")chunk=strata::intel::integer(value,512);
            else if(option=="--kv-type"){if(value!="int8"&&value!="auto")throw std::runtime_error("Gemma target-only SYCL currently validates INT8 KV only");}
            else if(option=="--mtp"){if(value!="0")throw std::runtime_error("Gemma assistant verification is separate; use --mtp 0 for this target-only milestone");}
            else if(option=="--weight-memory"){if(value=="host")memory=strata::intel::WeightMemory::Host;else if(value=="shared")memory=strata::intel::WeightMemory::Shared;else if(value=="device")memory=strata::intel::WeightMemory::Device;else throw std::runtime_error("Invalid weight memory kind");}
            else throw std::runtime_error("Unknown Gemma option "+option);
        }
        if(model.empty())throw std::runtime_error("--model is required");if(server&&(!tokens.empty()||!token_file.empty()||!dump.empty()))throw std::runtime_error("CLI token options cannot accompany --serve");
        strata::intel::GemmaEngine engine(model,device,context,chunk,memory,mkl);
        if(server)return strata::intel::serve(engine);
        if(!token_file.empty()){std::ifstream f(strata::paths::native_path(token_file));if(!f)throw std::runtime_error("Cannot read token file");std::ostringstream s;s<<f.rdbuf();tokens=s.str();}
        const auto ids=strata::intel::parse_tokens(tokens);strata::intel::ExecutionRequest active;
        for(size_t first=0;first<ids.size();){const int n=int(std::min(size_t(engine.chunk()),ids.size()-first));engine.step_many(ids.data()+first,n,first+n==ids.size());first+=n;}
        if(!dump.empty()){std::ofstream f(strata::paths::native_path(dump),std::ios::binary);f.write(reinterpret_cast<const char*>(engine.logits()),size_t(engine.vocabulary())*sizeof(float));if(!f)throw std::runtime_error("Cannot write logits dump");}
        const float* logits=engine.logits();const int best=int(std::max_element(logits,logits+engine.vocabulary())-logits);std::printf("LOGITS tokens=%zu vocabulary=%d argmax=%d value=%.9g\n",ids.size(),engine.vocabulary(),best,logits[best]);return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"Intel Gemma4: %s\n",error.what());return 1;}
}
