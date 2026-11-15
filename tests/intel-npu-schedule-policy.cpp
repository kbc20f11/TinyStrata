#include "strata/intel/npu_schedule_policy.hpp"
#include <iostream>
#include <limits>
#include <string>

using namespace strata::intel::npu_policy;
namespace {
std::size_t checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
template<class F> void rejects(F&& fn, const char* label) {
    bool rejected = false;
    try { fn(); } catch (const std::logic_error&) { rejected = true; }
    check(rejected, label);
}
void shares() {
    SharePolicy off;
    check(!off.select({Projection::qkv,14,6144,2048,1024}).offloaded(), "Default must stay GPU-only");
    SharePolicy policy({{Projection::qkv,128,511,256}, {Projection::qkv,512,1024,1024},
                        {Projection::gate,128,255,0}, {Projection::gate,256,1024,512}});
    for (std::size_t tokens : {127u,128u,255u,256u,511u,512u,1024u,1025u}) {
        for (std::size_t rows : {255u,256u,257u,1024u,1025u,6145u}) {
            for (auto projection : {Projection::qkv,Projection::gate,Projection::attention_output}) {
                const auto p = policy.select({projection,14,rows,2048,tokens});
                check(p.npu_rows+p.gpu_rows==rows,"Every output row retained");
                check(!p.npu_rows || (p.npu_rows%256==0 && p.npu_rows<rows),"Aligned proper partition");
                if (tokens<128 || tokens>1024 || projection==Projection::attention_output)
                    check(!p.offloaded(),"Unsupported length/projection falls back");
                if (p.offloaded()) check(p.prepared_weight_bytes==p.npu_rows*2048*2,"Exact FP16 slab bytes");
            }
        }
    }
    check(policy.select({Projection::qkv,14,6144,2048,511}).npu_rows==256,"Short QKV explicit rule");
    check(policy.select({Projection::qkv,14,6144,2048,512}).npu_rows==1024,"Long QKV explicit rule");
    check(policy.select({Projection::gate,14,6144,2048,512}).npu_rows==512,"Gate has its own share");
    check(!policy.select({Projection::qkv,2,6144,2048,512}).offloaded(),"Unvalidated weight type");
    check(!policy.select({Projection::qkv,14,6144,4096,512}).offloaded(),"Unvalidated width");
    rejects([]{SharePolicy p({{Projection::qkv,128,512,256},{Projection::qkv,512,1024,512}});},"Overlapping rules rejected");
    rejects([]{SharePolicy p({{Projection::qkv,128,1024,257}});},"Unaligned rule rejected");
    rejects([]{SharePolicy p({{Projection::qkv,1024,128,256}});},"Inverted token interval rejected");
    SplitLimits limit;limit.max_prepared_weight_bytes=1048575;
    check(!SharePolicy({{Projection::qkv,128,1024,256}},limit).select({Projection::qkv,14,6144,2048,512}).offloaded(),"Slab cap before allocation");
    limit.max_prepared_weight_bytes=1048576;
    check(SharePolicy({{Projection::qkv,128,1024,256}},limit).select({Projection::qkv,14,6144,2048,512}).offloaded(),"Exact slab cap accepted");
}
void pairs() {
    const PairResources ready{true,true,true,true,false};
    for (std::size_t chunk : {128u,512u,1024u}) {
        for (std::size_t remaining=0;remaining<=4*chunk+1;++remaining) {
            auto p=choose_pair(17,remaining,chunk,17+remaining,ready);
            check(p.first_tokens==std::min(chunk,remaining),"Original first chunk shape");
            check(p.paired()==(remaining>=2*chunk),"Only original full chunks paired");
            if(p.paired()) check(p.second_tokens==chunk,"Original second chunk shape");
        }
    }
    for(unsigned missing=0;missing<5;++missing) {
        auto r=ready;
        if(missing==0)r.separate_GPU_banks=false;
        if(missing==1)r.separate_NPU_request_buffers=false;
        if(missing==2)r.extra_memory_admitted=false;
        if(missing==3)r.ordered_state_dependencies=false;
        if(missing==4)r.capture_or_vision=true;
        check(!choose_pair(0,2048,1024,2048,r).paired(),"Pair needs every resource proof");
    }
    check(!choose_pair(0,2048,1024,2048,ready,1024).paired(),"Exact checkpoint not crossed");
    check(choose_pair(0,2048,1024,2048,ready,2048).paired(),"Boundary after pair retained");
    const auto max=std::numeric_limits<std::size_t>::max();
    check(choose_pair(max-2048,2048,1024,max,ready).paired(),"High-position subtraction bounds");
    rejects([&]{choose_pair(max-100,101,128,max,ready);},"Context addition overflow refused");
    rejects([&]{choose_pair(0,1024,0,1024,ready);},"Zero chunk refused");
}
void ordering_and_cancel() {
    for(std::size_t layers : {1u,2u,10u,40u}) {
        PairSchedule schedule(layers,73);
        std::array<std::size_t,2> committed{};
        std::vector<std::uint64_t> recurrence(layers);
        for(std::size_t l=0;l<layers;++l)recurrence[l]=l+1;
        std::vector<Action> expected{{0,0,Stage::linear_submit,73},{1,0,Stage::linear_submit,73}};
        for(std::size_t l=0;l<layers;++l) {
            for(unsigned bank=0;bank<2;++bank) {
                expected.push_back({bank,l,Stage::state_apply,73});
                expected.push_back({bank,l,Stage::output_submit,73});
            }
            for(unsigned bank=0;bank<2;++bank) {
                expected.push_back({bank,l,Stage::finish_layer,73});
                if(l+1<layers)expected.push_back({bank,l+1,Stage::linear_submit,73});
            }
        }
        std::size_t actions=0;
        while(auto a=schedule.next()) {
            check(*a==expected.at(actions),"Yield only after independent NPU submissions");
            if(a->stage==Stage::linear_submit || a->stage==Stage::output_submit) {
                schedule.submitted(a->bank,a->epoch);
                rejects([&]{schedule.submitted(a->bank,a->epoch);},"No live tensor/request rebind");
            }
            if(a->stage==Stage::state_apply) {
                check(a->bank==0 || committed[0]==a->layer+1,"First chunk state precedes second");
                committed[a->bank]=a->layer+1;
                // The serial reference consumes both chunks in token order;
                // this shared noncommutative state catches bank reversal.
                for(unsigned token : (a->bank ? std::array<unsigned,2>{5,7} : std::array<unsigned,2>{2,3}))
                    recurrence[a->layer]=recurrence[a->layer]*11+token;
                if(a->bank==1) {
                    std::uint64_t serial=a->layer+1;
                    for(unsigned token : {2u,3u,5u,7u})serial=serial*11+token;
                    check(recurrence[a->layer]==serial,"Shared recurrence matches serial token order");
                }
            }
            schedule.advance(*a);++actions;
            if(a->stage==Stage::linear_submit || a->stage==Stage::output_submit)
                schedule.retire(a->bank,a->epoch);
        }
        check(actions==layers*8 && schedule.publishable(),"Publish only complete pair");
        rejects([&]{schedule.advance({0,0,Stage::linear_submit,72});},"Stale completion rejected");
    }
    for(unsigned stop=0;stop<16;++stop) {
        PairSchedule schedule(2,101);
        for(unsigned step=0;step<stop;++step) {
            auto a=*schedule.next();schedule.advance(a);
        }
        const auto a=*schedule.next();
        if(a.stage==Stage::linear_submit || a.stage==Stage::output_submit)
            schedule.submitted(a.bank,a.epoch);
        schedule.cancel();check(!schedule.next() && !schedule.publishable(),"Cancelled pair never publishes");
        rejects([&]{schedule.advance(a);},"No work after cancel");
        if(a.stage==Stage::linear_submit || a.stage==Stage::output_submit) {
            check(!schedule.drained(),"Submitted buffers retained during cancellation");
            schedule.retire(a.bank,a.epoch);
        }
        check(schedule.drained(),"Drain before rollback/buffer reuse");
    }
    PairSchedule pending(1,8);auto first=*pending.next();pending.submitted(0,8);pending.advance(first);
    pending.advance(*pending.next());
    rejects([&]{pending.advance(*pending.next());},"State waits for NPU result retirement");
    pending.retire(0,8);pending.advance(*pending.next());
    rejects([&]{pending.retire(0,8);},"No double retirement");
}
}
int main() {
    try { shares();pairs();ordering_and_cancel();std::cout<<"NPU CPU schedule policy: "<<checks<<" checks passed\n";return 0; }
    catch(const std::exception& error) {std::cerr<<"NPU CPU schedule policy failed: "<<error.what()<<'\n';return 1;}
}
