#include "strata/intel/gemma_vision.hpp"
#include "strata/intel/context_policy.hpp"
#include <array>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace gv=strata::intel::gemma_vision;
namespace cp=strata::intel::context_policy;
namespace {
int checks=0;
void check(bool x){if(!x)throw std::runtime_error("Gemma vision policy check "+std::to_string(checks));++checks;}
void write(const std::filesystem::path& p,int rows,int nx,int ny,int width,float value=1.f,bool truncate=false){
    std::ofstream f(p,std::ios::binary);const int32_t h[]={0x31455653,rows,nx,ny,width};f.write(reinterpret_cast<const char*>(h),sizeof h);
    if(rows>0&&rows<=1120&&width>0&&width<16){std::vector<float> v(size_t(rows)*width,value);f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*4-(truncate?1:0)));}
}
}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Pass an owned fixture directory");
    const auto dir=strata::paths::native_path(argv[1]);std::filesystem::create_directories(dir);
    const auto file=dir/std::filesystem::path(L"Gemma image \u753b\u50cf.sve");const auto path=strata::paths::utf8_path(file);
    std::vector<int> ids={7,42,42,8};gv::Input input;std::string error;
    write(file,2,2,1,4);check(gv::load(path,ids,4,4096,42,input,error));check(input.images.size()==1&&input.images[0].cell==1&&input.images[0].rows==2&&input.embeddings.size()==8);
    check(gv::whole_prefix(input,1)&&!gv::whole_prefix(input,2)&&gv::whole_prefix(input,3));
    check(gv::next_count(input,0,4,512)==1&&gv::next_count(input,1,3,512)==2&&gv::next_count(input,3,1,512)==1);
    bool rejected=false;try{(void)gv::next_count(input,2,2,512);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    auto changed=input;changed.embeddings[0]=2;check(gv::same_prefix(input,changed,1,4)&&!gv::same_prefix(input,changed,3,4));
    check(!gv::same_prefix(input,input,2,4));gv::Input none;check(gv::same_prefix(input,none,1,4)&&!gv::same_prefix(input,none,3,4));
    auto future=input;future.images.push_back({4,1,8});future.embeddings.insert(future.embeddings.end(),4,3.f);
    check(gv::same_prefix(input,future,3,4)&&!gv::same_prefix(input,future,5,4));
    // Every malformed record is rejected without replacing prior validated data.
    const auto retained=input.embeddings;
    for(const auto& shape:std::array<std::array<int,4>,7>{{{0,0,1,4},{2,1,2,4},{2,2,1,3},{1121,1121,1,4},{-1,-1,1,4},{3,3,1,4},{2,2,1,0}}}){
        write(file,shape[0],shape[1],shape[2],shape[3]);check(!gv::load(path,ids,4,4096,42,input,error));check(input.embeddings==retained);
    }
    write(file,2,2,1,4,std::numeric_limits<float>::infinity());check(!gv::load(path,ids,4,4096,42,input,error));
    write(file,2,2,1,4,1.f,true);check(!gv::load(path,ids,4,4096,42,input,error));
    write(file,2,2,1,4);check(!gv::load(path,{7,42,42},4,4096,42,input,error));check(!gv::load(path,ids,4,3,42,input,error));
    std::vector<int> large(1122,42);large.front()=7;large.back()=8;write(file,1120,1120,1,4);
    check(gv::load(path,large,4,4096,42,input,error));check(gv::next_count(input,1,1121,512)==1120);
    check(!gv::whole_prefix(input,513)&&gv::whole_prefix(input,1121));
    check(gv::workspace(512,32768,false)==512&&gv::workspace(512,32768,true)==1120&&gv::workspace(512,128,true)==512);
    check(gv::host_reservation(262144,2816)==512ull*1024*1024);
    check(gv::host_reservation(4096,2816)==2ull*4096*2816*4);
    cp::GemmaSpec text,vision;vision.vision=true;
    for(uint64_t context:{2048ull,4096ull,32768ull}){
        const auto a=cp::gemma(context,text),b=cp::gemma(context,vision);
        const uint64_t before=std::min(context,1536ull),after=std::min(context,2144ull);
        const uint64_t snapshot_delta=(after-before)*8*256*2*34/32*25;
        check(b.host_only-a.host_only==gv::host_reservation(context,2816)+snapshot_delta);
        check(b.device>a.device&&b.owned_device>a.owned_device);
    }
    strata::intel::MemoryBudget memory;memory.integrated=true;memory.wddm_known=true;memory.wddm_available=4*strata::intel::GiB;
    memory.available_physical=memory.available_commit=12*strata::intel::GiB;
    const auto a=cp::select(0,262144,memory,8*strata::intel::GiB,[&](uint64_t c){return cp::gemma(c,text);});
    const auto b=cp::select(0,262144,memory,8*strata::intel::GiB,[&](uint64_t c){return cp::gemma(c,vision);});
    check(a.automatic&&b.automatic&&a.tokens>b.tokens&&b.tokens>8192);
    check(memory.wddm_available-b.bytes.device>=strata::intel::uma_wddm_headroom);
    std::filesystem::remove(file);
    std::printf("Gemma vision CPU policy PASS checks=%d text_auto=%llu vision_auto=%llu\n",checks,(unsigned long long)a.tokens,(unsigned long long)b.tokens);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
