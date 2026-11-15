#include "strata/intel/lfm_turn_checkpoint.hpp"
#include "strata/intel/context_policy.hpp"
#include <bit>
#include <cstring>
#include <iostream>
#include <numeric>

namespace lt=strata::intel::lfm_turn;
namespace cp=strata::intel::context_policy;
size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("LFM turn checkpoint CPU regression");}
template<class F>void rejects(F run){bool rejected=false;try{run();}catch(const std::exception&){rejected=true;}require(rejected);}
int main(){
    lt::Options options;require(options.checkpoint&&options.turn_at==0);
    require(!options.parse("temperature","0.1"));require(options.parse("turn_at","3"));require(options.parse("ckpt","1"));options.validate(8,16);
    rejects([&]{options.parse("ckpt","0");});rejects([&]{options.parse("turn_at","4");});
    for(auto text:{"","-1","+1","1.0","nan","184467440737095516160"}){lt::Options value;rejects([&]{value.parse("turn_at",text);});}
    for(auto text:{"","-1","2","true","1.0","01"}){lt::Options value;rejects([&]{value.parse("ckpt",text);});}
    rejects([&]{options.validate(2,16);});rejects([&]{options.validate(8,7);});rejects([&]{options.validate(0,16);});
    for(size_t capacity:{size_t(0),size_t(131073),SIZE_MAX})rejects([&]{lt::Snapshot invalid(capacity);});
    require(lt::state_bytes==499712);for(uint64_t c:{1,32768,131072}){
        lt::Snapshot snapshot(c);require(snapshot.bytes()==lt::host_bytes(c));
        const auto footprint=cp::lfm(c);require(footprint.host_only==(16ull<<20)+snapshot.bytes());
        require(footprint.host_visible==footprint.owned_host_visible); // no ordinary snapshot in USM/WDDM
    }
    rejects([]{(void)lt::host_bytes(UINT64_MAX);});
    const std::vector<int> live{1,2,3,4,5},saved{1,2,3};
    const auto choose=[&](std::vector<int> old,std::vector<int> prompt,std::vector<int> prefix,bool on,lt::Source source,size_t count){
        const auto result=lt::choose(old,prompt,prefix,on);require(result.source==source&&result.tokens==count);
    };
    choose(live,{1,2,3,4,5,6},saved,true,lt::Source::live,5);
    choose(live,live,saved,true,lt::Source::live,5);
    choose(live,{1,2,3,8},saved,true,lt::Source::saved_turn,3);
    choose(live,saved,saved,true,lt::Source::saved_turn,3); // fresh0
    choose(live,{1,2},saved,true,lt::Source::reset,0);
    choose(live,{1,9,3,8},saved,true,lt::Source::reset,0);
    choose({1,9,3,4,5},{1,2,3,8},saved,true,lt::Source::reset,0); // resident KV no longer matches
    choose(live,live,saved,false,lt::Source::reset,0);
    choose({},saved,{},true,lt::Source::live,0);
    for(size_t prompt:{size_t(104),size_t(512),size_t(513),size_t(4085),size_t(16370)}){
        lt::Options cold;cold.parse("ckpt","0");cold.turn_at=prompt-5;
        for(size_t i=0;i<prompt;){const size_t end=lt::chunk_end(i,prompt,512,cold);require(end==std::min(i+512,prompt));i=end;}
        lt::Options warm;warm.turn_at=prompt-5;bool reached=false;
        for(size_t i=0;i<prompt;){const size_t end=lt::chunk_end(i,prompt,512,warm);require(end>i&&end-i<=512);reached|=end==warm.turn_at;i=end;}require(reached);
    }
    rejects([&]{lt::chunk_end(0,8,0,options);});rejects([&]{lt::chunk_end(8,8,512,options);});
    // Bit-copy all30 convolution states and final hidden, including NaN bits,
    // signed zero and subnormal payloads. Canaries surround each mock device pool.
    std::vector<float> source(lt::state_values+2),target(source.size(),-999.f);
    for(size_t i=0;i<source.size();++i)source[i]=std::bit_cast<float>(uint32_t(i*2654435761u));
    const auto original=source;lt::Snapshot snapshot(64);std::atomic<bool> cancelled=false;
    auto copy_out=[&](std::span<float> host){require(host.size()==lt::state_values);std::memcpy(host.data(),source.data()+1,host.size_bytes());};
    auto copy_in=[&](std::span<const float> host){std::memcpy(target.data()+1,host.data(),host.size_bytes());};
    require(snapshot.save(saved,saved.size(),copy_out,&cancelled));require(snapshot.position()==3);
    require(snapshot.restore(live,std::vector<int>{1,2,3,8},copy_in)==3);
    require(std::memcmp(target.data()+1,source.data()+1,lt::state_bytes)==0);require(target.front()==-999.f&&target.back()==-999.f);
    require(std::memcmp(source.data(),original.data(),source.size()*4)==0);
    // Per-step rollback storage is independent of the saved turn.
    std::fill(source.begin()+1,source.end()-1,42.f);std::fill(target.begin()+1,target.end()-1,99.f);
    snapshot.restore(live,saved,copy_in);require(std::memcmp(target.data()+1,original.data()+1,lt::state_bytes)==0);
    cancelled=true;bool copied=false;require(!snapshot.save(saved,3,[&](auto){copied=true;},&cancelled));require(!copied&&snapshot.position()==3);
    cancelled=false;require(!snapshot.save(saved,3,[&](auto host){copy_out(host);cancelled=true;},&cancelled));require(snapshot.position()==0);
    rejects([&]{snapshot.restore(live,saved,copy_in);});cancelled=false;
    rejects([&]{snapshot.save(saved,3,[](auto){throw std::runtime_error("incomplete device copy");});});require(snapshot.position()==0);
    rejects([&]{snapshot.save(saved,2,copy_out);});require(snapshot.save(saved,3,copy_out));
    snapshot.invalidate();require(snapshot.position()==0);
    // Recurrent replay oracle: a branch restore must be bit-identical to a cold
    // sequential calculation, while speculative/partial state rolls back.
    auto advance=[](std::vector<float>& state,int id){for(size_t i=0;i<state.size();++i)state[i]=state[i]*.5f+float(id)+float(i%7)*.125f;};
    std::vector<float> recurrent(lt::state_values),cold(recurrent.size()),rollback;
    lt::Snapshot turn(64);std::vector<int> committed;
    for(int id:saved){advance(recurrent,id);committed.push_back(id);}
    turn.save(committed,committed.size(),[&](auto host){std::copy(recurrent.begin(),recurrent.end(),host.begin());});
    for(int id:{4,5}){advance(recurrent,id);committed.push_back(id);}
    turn.restore(committed,std::vector<int>{1,2,3,8},[&](auto host){std::copy(host.begin(),host.end(),recurrent.begin());});
    committed.resize(3);advance(recurrent,8);committed.push_back(8);
    for(int id:committed)advance(cold,id);require(std::memcmp(cold.data(),recurrent.data(),lt::state_bytes)==0);
    rollback=recurrent;advance(recurrent,99);recurrent=rollback; // cancelled in-flight step, no committed ID appended
    turn.restore(committed,saved,[&](auto host){std::copy(host.begin(),host.end(),recurrent.begin());});
    std::fill(cold.begin(),cold.end(),0);for(int id:saved)advance(cold,id);require(std::memcmp(cold.data(),recurrent.data(),lt::state_bytes)==0);
    require(committed.size()==4); // emitted-but-uncommitted terminal/cancel tokens never extend the saved prefix
    std::cout<<"LFM turn checkpoint CPU checks="<<checks<<" state_bytes="<<lt::state_bytes<<" max_host_bytes="<<lt::host_bytes(131072)<<" passed\n";
}
