// One fixed-placement worker: serial full-vocabulary reference versus grouped
// verify windows. Run env0/env1 separately and compare the word hash or dump.
#include "engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

using strata::gemma4::Engine;
using strata::gemma4::Options;
using strata::gemma4::KvFmt;
using Clock=std::chrono::steady_clock;
static unsigned checks=0;
static uint64_t hash=1469598103934665603ULL,logit_bytes=0;
static std::ofstream dump;
static void need(bool yes,const char* message){if(!yes)throw std::runtime_error(message);++checks;}
static std::vector<int32_t> sequence(int n,int offset=0) {
    std::vector<int32_t> ids(size_t(n),0);
    for(int i=0;i<n;++i)ids[size_t(i)]=32+(i*37+i/7+offset)%512;
    return ids;
}
static void equal(const std::vector<float>& expected,const float* actual,const char* stage,int position,int width,int kept) {
    for(size_t i=0;i<expected.size();++i) {
        uint32_t word=0;
        if(!std::isfinite(expected[i])||!std::isfinite(actual[i])||std::memcmp(expected.data()+i,actual+i,sizeof(float))) {
            std::fprintf(stderr,"FAIL %s pos=%d width=%d kept=%d logit=%zu expected=%.9g actual=%.9g\n",
                         stage,position,width,kept,i,expected[i],actual[i]);
            throw std::runtime_error("serial/window full-vocabulary logits differ");
        }
        std::memcpy(&word,actual+i,sizeof(word));hash^=word;hash*=1099511628211ULL;
    }
    if(dump.is_open()) {
        dump.write(reinterpret_cast<const char*>(actual),std::streamsize(expected.size()*sizeof(float)));
        if(!dump)throw std::runtime_error("cannot write full-vocabulary dump");
    }
    logit_bytes+=expected.size()*sizeof(float);++checks;
}
static void prefix(Engine& e,const std::vector<int32_t>& ids) {
    e.reset();
    for(size_t at=0;at<ids.size();) {
        const int n=int(std::min<size_t>(64,ids.size()-at));
        e.eval(ids.data()+at,n,false);at+=size_t(n);
    }
    need(e.pos()==int64_t(ids.size()),"prefix position differs");
}
static void state_cases(Engine& e,int position) {
    const auto input=sequence(position);
    // Both phases revisit the same captured window/position keys with changed
    // tokens and routing. A grouped map must be rebuilt on every graph replay.
    for(int phase:{113,271}) {
        const auto continuation=sequence(12,phase);
        prefix(e,input);
        std::vector<std::vector<float>> expected;
        for(int32_t token:continuation) {
            e.eval(&token,1,true);
            expected.emplace_back(e.logits(),e.logits()+e.n_vocab());
        }
        for(int width:{2,3,4}) {
            std::vector<int> keeps{1,width};if(width==4)keeps.insert(keeps.begin()+1,2);
            for(int kept:keeps) {
                prefix(e,input);
                int32_t window[4];
                for(int i=0;i<width;++i)window[i]=i<kept?continuation[size_t(i)]:800+phase+i;
                e.window(window,width);
                for(int i=0;i<kept;++i)equal(expected[size_t(i)],e.logits_of(i),"accepted prefix",position,width,kept);
                e.keep(kept);
                need(e.pos()==position+kept,"keep retained a rejected suffix");
                for(int i=kept;i<int(continuation.size());++i) {
                    e.eval(continuation.data()+i,1,true);
                    equal(expected[size_t(i)],e.logits(),"continuation after rejected suffix",position,width,kept);
                }
            }
        }
    }
}
static void time_windows(Engine& e,int position,int width,int rounds) {
    constexpr int warmup=4;
    const auto input=sequence(position);
    const auto ids=sequence((warmup+rounds)*width,419+width);
    need(position+int(ids.size())<=e.max_context(),"timing stream exceeds context");
    prefix(e,input);
    std::vector<std::vector<float>> expected;
    for(size_t i=0;i<ids.size();++i) {
        e.eval(ids.data()+i,1,true);
        if(i>=ids.size()-size_t(width))expected.emplace_back(e.logits(),e.logits()+e.n_vocab());
    }
    prefix(e,input);
    for(int round=0;round<warmup;++round) {
        e.window(ids.data()+round*width,width);e.keep(width);(void)e.took_record();
    }
    const auto before=e.stats();
    const auto start=Clock::now();
    bool recorded=false;
    for(int round=warmup;round<warmup+rounds;++round) {
        e.window(ids.data()+round*width,width);e.keep(width);
        recorded=e.took_record()||recorded;
    }
    const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    const auto after=e.stats();
    // Reading/copying/hashing logits is outside the timed region. This is a
    // target-only verification cost, not an end-to-end speculative tok/s claim.
    for(int i=0;i<width;++i)equal(expected[size_t(i)],e.logits_of(i),"timed stream final window",position,width,width);
    std::printf("TIMING width=%d rounds=%d processed=%d milliseconds=%.6f mean_block_ms=%.6f recorded=%d "
                "gpu_experts=%lld cpu_experts=%lld batch_cached=%lld batch_staged=%lld\n",width,rounds,rounds*width,ms,ms/rounds,
                int(recorded),(long long)(after.gpu_experts-before.gpu_experts),(long long)(after.cpu_experts-before.cpu_experts),
                (long long)(after.batch_cached-before.batch_cached),(long long)(after.batch_staged-before.batch_staged));
}
int main(int argc,char**argv) {
    try {
        need(argc>=2,"usage: gptoss_grouped_state_test MODEL [--graphs 0|1 --expert-gib N --kv int8|fp16 --context N --position N --timed-rounds N --expected-seen N --logits FILE]");
        need(std::getenv("STRATA_FREEZE_EXPERTS")!=nullptr,"set STRATA_FREEZE_EXPERTS=1 for fixed placement and exact target arithmetic");
        Options options;options.model=argv[1];options.max_context=4096;options.prompt_batch=64;options.window=4;
        options.expert_swaps=0;options.kv_grow=false;options.kv_format=KvFmt::kQ8_0;options.draft=false;
        int position=193,rounds=16,expected_seen=-1;
        std::string dump_path;
        for(int i=2;i<argc;++i) {
            const std::string key=argv[i];need(i+1<argc,"missing option value");const std::string value=argv[++i];
            if(key=="--graphs"){need(value=="0"||value=="1","graphs must be0or1");options.graphs=value=="1";}
            else if(key=="--expert-gib")options.expert_vram_gib=std::stod(value);
            else if(key=="--context")options.max_context=std::stoi(value);
            else if(key=="--position")position=std::stoi(value);
            else if(key=="--timed-rounds")rounds=std::stoi(value);
            else if(key=="--expected-seen")expected_seen=std::stoi(value);
            else if(key=="--logits")dump_path=value;
            else if(key=="--kv"){
                need(value=="int8"||value=="fp16","kv must beint8orfp16");options.kv_format=value=="fp16"?KvFmt::kFp16:KvFmt::kQ8_0;
            }else throw std::runtime_error("unknown option: "+key);
        }
        need(position>0&&position<=options.max_context-32&&rounds>=0&&rounds<=256,"invalid position/timing rounds");
        if(!dump_path.empty()){dump.open(dump_path,std::ios::binary|std::ios::trunc);need(bool(dump),"cannot open logits dump");}
        auto engine=Engine::open(options);
        need(std::string(engine->model_family())=="gpt-oss"&&engine->max_window()==4&&!engine->drafts(),"unexpected model/window/draft geometry");
        need(!engine->external_draft_enabled(),"this target-only fixture must not load an auxiliary drafter");
        if(expected_seen<0)expected_seen=engine->grouped_mxfp4_enabled()?
            (engine->cached_experts()==engine->total_experts()?1:engine->cached_experts()>0?2:0):0;
        state_cases(*engine,position);
        if(rounds){time_windows(*engine,position,2,rounds);time_windows(*engine,position,3,rounds);}
        need((engine->grouped_mxfp4_seen()&uint32_t(expected_seen))==uint32_t(expected_seen),"required grouped kernel path not seen");
        if(!engine->grouped_mxfp4_enabled())need(engine->grouped_mxfp4_seen()==0,"disabled grouped path was used");
        const auto stats=engine->stats();
        if(dump.is_open()){dump.close();need(bool(dump),"cannot finish logits dump");}
        std::printf("PASS grouped-state checks=%u logit_bytes=%llu fnv64_words=%016llx graph=%d cache=%d/%d context=%d position=%d "
                    "grouped=%d seen=%u tokens=%lld gpu_experts=%lld cpu_experts=%lld batch_cached=%lld batch_staged=%lld swapped=%lld\n",
                    checks,(unsigned long long)logit_bytes,(unsigned long long)hash,int(options.graphs),engine->cached_experts(),engine->total_experts(),
                    engine->max_context(),position,int(engine->grouped_mxfp4_enabled()),engine->grouped_mxfp4_seen(),(long long)stats.tokens,
                    (long long)stats.gpu_experts,(long long)stats.cpu_experts,(long long)stats.batch_cached,(long long)stats.batch_staged,
                    (long long)stats.swapped_experts);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"gptoss_grouped_state_test: %s\n",error.what());return 1;}
}
