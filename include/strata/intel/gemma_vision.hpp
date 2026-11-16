#pragma once
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::intel::gemma_vision {
inline constexpr int max_image_rows = 1120;
inline constexpr size_t max_images = 64;
inline constexpr uint64_t payload_limit = 256ull << 20;
struct Image {size_t cell=0,rows=0,offset=0;};
// The old validated payload remains alive while a new request is checked.
// After exact prefix comparison the live/saved views share the new owner.
inline uint64_t host_reservation(uint64_t context,uint64_t width) {
    const uint64_t rows = width && context <= UINT64_MAX / width / 4 ? context * width * 4 : UINT64_MAX;
    const uint64_t payload = std::min(payload_limit,rows);
    const uint64_t metadata = std::min(uint64_t(max_images),width ? payload / width / sizeof(float) : 0) * sizeof(Image);
    return 2 * (payload + metadata);
}
inline uint64_t workspace(uint64_t chunk,uint64_t context,bool vision) {
    return vision ? std::max(chunk,std::min(context,uint64_t(max_image_rows))) : chunk;
}
struct Input {std::vector<float> embeddings;std::vector<Image> images;};
inline const Image* image_at(const Input& input,size_t cell) {
    for(const auto& im:input.images)if(cell>=im.cell&&cell-im.cell<im.rows)return &im;
    return nullptr;
}
inline bool whole_prefix(const Input& input,size_t end) {
    for(const auto& im:input.images)if(im.cell<end&&end-im.cell<im.rows)return false;
    return true;
}
inline bool same_prefix(const Input& old,const Input& next,size_t end,size_t width) {
    if(!width||!whole_prefix(old,end)||!whole_prefix(next,end))return false;
    size_t a=0,b=0;
    while(a<old.images.size()&&old.images[a].cell<end) {
        if(b>=next.images.size())return false;
        const auto& x=old.images[a++];const auto& y=next.images[b++];
        if(x.cell!=y.cell||x.rows!=y.rows||!x.rows||x.rows>SIZE_MAX/width)return false;
        const size_t count=x.rows*width;
        if(x.offset>old.embeddings.size()||count>old.embeddings.size()-x.offset||
           y.offset>next.embeddings.size()||count>next.embeddings.size()-y.offset||
           std::memcmp(old.embeddings.data()+x.offset,next.embeddings.data()+y.offset,count*sizeof(float)))return false;
    }
    return b==next.images.size()||next.images[b].cell>=end;
}
// A whole picture is one attention transaction: only its SWA layers see future
// patches. Text chunks stop at the first picture; no checkpoint splits one.
inline size_t next_count(const Input& input,size_t at,size_t remaining,size_t chunk) {
    size_t count=std::min(remaining,chunk);
    for(const auto& im:input.images) {
        if(at>=im.cell+im.rows)continue;
        if(at<im.cell)return std::min(count,im.cell-at);
        if(at!=im.cell||im.rows>remaining)throw std::invalid_argument("Gemma continuation splits a picture");
        return im.rows;
    }
    return count;
}
// SVE1 uses one row of F32 projector output per image-pad token. Gemma's
// projector geometry is one-dimensional; Qwen's spatial M-RoPE is not used.
inline bool load(const std::string& path,const std::vector<int>& ids,int width,int context,int pad,
                 Input& result,std::string& error) {
    Input next;
    const size_t pads=size_t(std::count(ids.begin(),ids.end(),pad));
    if(width<1||context<1||pad<0||ids.empty()||ids.size()>size_t(context)||!pads||
       pads>payload_limit/sizeof(float)/size_t(width)) {error="Gemma image prompt exceeds its bounded payload/context";return false;}
    std::ifstream file(strata::paths::native_path(path),std::ios::binary);
    if(!file){error="Cannot open Gemma image embeddings file";return false;}
    // Reserve once from the validated complete prompt. Incremental vector
    // growth can temporarily retain a third payload beyond the host ledger.
    next.embeddings.reserve(pads*size_t(width));
    next.images.reserve(std::min(max_images,pads));
    size_t cursor=0,remaining=pads;
    while(file.peek()!=std::char_traits<char>::eof()) {
        int32_t header[5]{};file.read(reinterpret_cast<char*>(header),sizeof header);
        if(!file||header[0]!=0x31455653||header[1]<1||header[1]>max_image_rows||header[2]!=header[1]||header[3]!=1||
           header[4]!=width||size_t(header[1])>remaining||next.images.size()>=max_images){error="Invalid Gemma SVE1 image geometry";return false;}
        while(cursor<ids.size()&&ids[cursor]!=pad)++cursor;
        const size_t rows=size_t(header[1]);
        if(cursor>ids.size()||rows>ids.size()-cursor||!std::all_of(ids.begin()+cursor,ids.begin()+cursor+rows,[&](int id){return id==pad;})){
            error="Gemma image rows do not match image-pad tokens";return false;}
        const size_t offset=next.embeddings.size(),count=rows*size_t(width);
        next.embeddings.resize(offset+count);file.read(reinterpret_cast<char*>(next.embeddings.data()+offset),std::streamsize(count*sizeof(float)));
        if(!file){error="Short Gemma image embeddings file";return false;}
        for(size_t i=offset;i<next.embeddings.size();++i)if(!std::isfinite(next.embeddings[i])){error="Nonfinite Gemma image embedding";return false;}
        next.images.push_back({cursor,rows,offset});cursor+=rows;remaining-=rows;
    }
    if(file.bad()||remaining||next.images.empty()||ids.back()==pad){error="Gemma image records do not cover a complete prompt";return false;}
    result=std::move(next);return true;
}
} // namespace strata::intel::gemma_vision
