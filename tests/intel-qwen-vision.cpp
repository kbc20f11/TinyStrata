#include "strata/intel/qwen_vision.hpp"
#include "strata/intel/context_policy.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace v = strata::intel::qwen_vision;
namespace cp = strata::intel::context_policy;
static int checks = 0;
static void check(bool value) { if (!value) throw std::runtime_error("Qwen vision CPU check " + std::to_string(checks)); ++checks; }
static std::string utf8(const std::filesystem::path& p) { const auto s=p.u8string(); return {s.begin(),s.end()}; }
struct Temp {
    std::filesystem::path parent=std::filesystem::temp_directory_path(), path;
    Temp() { path=parent/("intel-qwen-vision-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        if(!std::filesystem::create_directory(path))throw std::runtime_error("Cannot create owned test directory"); }
    ~Temp() { if(path.parent_path()==parent && path.filename().string().rfind("intel-qwen-vision-test-",0)==0) {
        std::error_code error;std::filesystem::remove_all(path,error); } }
};
static void record(std::ofstream& f,int rows,int nx,int ny,int width,float first=1.f) {
    const int32_t h[]={0x31455653,rows,nx,ny,width};f.write(reinterpret_cast<const char*>(h),sizeof h);
    for(int i=0;i<rows*width;++i){const float x=first+float(i)/16;f.write(reinterpret_cast<const char*>(&x),4);}
}
int main() try {
    Temp temp;const auto file=temp.path/std::filesystem::path(std::u8string(u8"image \u753b\u50cf.sve"));
    constexpr int pad=77,width=8,context=20;
    const std::vector<int> ids={10,pad,pad,pad,pad,11,12,pad,pad,13};
    {std::ofstream f(file,std::ios::binary);record(f,4,2,2,width);record(f,2,2,1,width,9);}
    v::Input original;std::string error;
    check(v::load(utf8(file),ids,width,context,pad,original,error));check(original.images.size()==2);
    check(original.embeddings.size()==6*width && original.positions.size()==context*3);
    const int32_t expected[][3]={{0,0,0},{1,1,1},{1,1,2},{1,2,1},{1,2,2},{3,3,3},{4,4,4},{5,5,5},{5,5,6},{7,7,7},{8,8,8}};
    for(size_t i=0;i<std::size(expected);++i)for(int j=0;j<3;++j)check(original.positions[3*i+j]==expected[i][j]);
    check(v::canonical::whole_prefix(original,1));check(!v::canonical::whole_prefix(original,2));
    check(v::canonical::whole_prefix(original,5));check(!v::canonical::whole_prefix(original,8));check(v::canonical::whole_prefix(original,9));
    check(v::overlaps(original,1,1));check(v::overlaps(original,0,2));check(!v::overlaps(original,5,2));check(!v::overlaps(original,9,1));
    auto next=original;check(v::canonical::same_prefix(original,next,ids.size(),width));
    next.embeddings[4*width]+=1;
    check(v::canonical::same_prefix(original,next,7,width));check(!v::canonical::same_prefix(original,next,9,width));
    auto prompt=ids;prompt.push_back(14);std::vector<int> saved(ids.begin(),ids.begin()+7);
    check(v::choose(ids,prompt,saved,saved.size(),true,original,original,width).source==strata::intel::turn_checkpoint::Source::live);
    auto reuse=v::choose(ids,prompt,saved,saved.size(),true,original,next,width);
    check(reuse.source==strata::intel::turn_checkpoint::Source::saved_turn && reuse.tokens==7);
    check(v::choose(ids,prompt,saved,saved.size(),false,original,next,width).source==strata::intel::turn_checkpoint::Source::reset);
    next.embeddings[0]+=1;check(v::choose(ids,prompt,saved,saved.size(),true,original,next,width).tokens==0);
    next=original;next.images[0].nx=4;next.images[0].ny=1;
    check(!v::canonical::same_prefix(original,next,5,width));
    const v::Input text;
    check(v::canonical::same_prefix(original,text,1,width));check(!v::canonical::same_prefix(original,text,5,width));
    check(v::canonical::same_prefix(text,original,1,width));
    auto changed_position=original;changed_position.positions[0]=1;
    check(!v::canonical::same_prefix(original,changed_position,1,width));
    check(v::saved_prefix(ids,prompt,saved,7,true));check(!v::saved_prefix(ids,prompt,saved,8,true));
    auto output=original;
    const auto reject=[&](const std::vector<int>& tokens,int w=8){error.clear();check(!v::load(utf8(file),tokens,w,context,pad,output,error));
        check(!error.empty());check(output.embeddings==original.embeddings && output.positions==original.positions);};
    reject({10,11});reject({pad,pad,10});reject({10,pad,pad,pad,pad,pad,pad});reject(ids,16);
    {std::ofstream f(file,std::ios::binary);record(f,4,4,2,width);record(f,2,2,1,width);}
    reject(ids);
    {std::ofstream f(file,std::ios::binary);const int32_t h[]={0x31455653,4,2,2,width};f.write(reinterpret_cast<const char*>(h),sizeof h);}
    reject(ids);
    {std::ofstream f(file,std::ios::binary);record(f,4,2,2,width);record(f,2,2,1,width,std::numeric_limits<float>::quiet_NaN());}
    reject(ids);
    {std::ofstream f(file,std::ios::binary);const int32_t h[]={0x31455653,INT32_MAX,1,INT32_MAX,width};f.write(reinterpret_cast<const char*>(h),sizeof h);}
    reject(ids);
    std::vector<int> cap(size_t(v::max_embedding_bytes/sizeof(float)/2048),pad);cap.push_back(10);
    error.clear();check(!v::load(utf8(temp.path/"absent"),cap,2048,262144,pad,output,error));check(error.find("cannot open")!=std::string::npos);
    cap.insert(cap.begin(),pad);error.clear();check(!v::load(utf8(temp.path/"absent"),cap,2048,262144,pad,output,error));check(error.find("256 MiB")!=std::string::npos);
    // Independently charge the actual M-RoPE USM and old/new host payload/maps.
    for(bool device:{false,true})for(uint64_t c:{1,2048,32768,262144}) {
        cp::QwenSpec off;off.device_state=device;auto on=off;on.vision=true;
        const auto a=cp::qwen(c,off),b=cp::qwen(c,on);
        check(b.device-a.device==c*3*4);check(b.owned_device-a.owned_device==c*3*4);
        check(b.host_visible-a.host_visible==(device?0:c*3*4));
        check(b.host_only-a.host_only==2*v::max_embedding_bytes+c*3*4*2);check(a.cache==b.cache);
    }
    using strata::intel::GiB;using strata::intel::MemoryBudget;
    cp::QwenSpec shape;shape.chunk=1024;shape.int8=true;shape.mtp=true;shape.mkl=true;shape.dp4a=true;shape.device_state=true;
    MemoryBudget before{12*GiB,12*GiB,4*GiB,true,true},after=before;after.available_physical-=GiB;after.available_commit-=GiB;after.wddm_available-=GiB;
    auto text_context=cp::select(0,262144,before,16*GiB,[&](uint64_t c){return cp::qwen(c,shape);});shape.vision=true;
    auto image_context=cp::select(0,262144,after,16*GiB,[&](uint64_t c){return cp::qwen(c,shape);});
    check(image_context.tokens>=2048 && image_context.tokens<text_context.tokens);
    check(after.wddm_available-image_context.bytes.device>=strata::intel::uma_wddm_headroom);
    std::printf("Intel Qwen vision CPU checks=%d PASS\n",checks);return 0;
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
