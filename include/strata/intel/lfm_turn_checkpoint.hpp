#pragma once
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strata::intel::lfm_turn {
constexpr size_t width=2048, convolution_layers=30, convolution_values=convolution_layers*2*width;
constexpr size_t state_values=convolution_values+width, state_bytes=state_values*sizeof(float);
static_assert(state_bytes==499712);
inline uint64_t host_bytes(uint64_t context){
    if(context>(UINT64_MAX-state_bytes)/sizeof(int))throw std::overflow_error("LFM turn checkpoint size overflow");
    return state_bytes+context*sizeof(int);
}
struct Options {
    bool checkpoint=true,seen_checkpoint=false,seen_turn=false;
    size_t turn_at=0;
    bool parse(std::string_view key,std::string_view value){
        if(key!="ckpt"&&key!="turn_at")return false;
        if(key=="ckpt"){
            if(seen_checkpoint||(value!="0"&&value!="1"))throw std::invalid_argument("LFM ckpt requires one0 or1 value");
            seen_checkpoint=true;checkpoint=value=="1";return true;
        }
        if(seen_turn||value.empty())throw std::invalid_argument("LFM turn_at requires one nonnegative integer");
        size_t parsed=0;const auto result=std::from_chars(value.data(),value.data()+value.size(),parsed);
        if(result.ec!=std::errc{}||result.ptr!=value.data()+value.size())throw std::invalid_argument("Invalid LFM turn_at");
        seen_turn=true;turn_at=parsed;return true;
    }
    void validate(size_t prompt,size_t context)const{
        if(!prompt||prompt>context||turn_at>prompt)throw std::invalid_argument("LFM turn boundary exceeds prompt/context");
    }
};
enum class Source { reset, live, saved_turn };
struct Reuse {Source source=Source::reset;size_t tokens=0;};
inline bool prefix(std::span<const int> part,std::span<const int> whole){
    return part.size()<=whole.size()&&std::equal(part.begin(),part.end(),whole.begin());
}
inline Reuse choose(std::span<const int> live,std::span<const int> prompt,
                    std::span<const int> saved,bool checkpoint){
    // A canonical ckpt0 request is a cold replay, including the old chunk shape.
    if(!checkpoint)return {};
    if(prefix(live,prompt))return {Source::live,live.size()};
    // Full attention KV is not copied. Both resident and incoming committed
    // prefixes must still match every ID covered by the recurrent snapshot.
    if(!saved.empty()&&prefix(saved,live)&&prefix(saved,prompt))return {Source::saved_turn,saved.size()};
    return {};
}
inline size_t chunk_end(size_t index,size_t prompt,size_t chunk,const Options& options){
    if(!chunk||index>=prompt)throw std::invalid_argument("Invalid LFM prefill chunk boundary");
    size_t end=index+std::min(chunk,prompt-index);
    if(options.checkpoint&&index<options.turn_at&&options.turn_at<end)end=options.turn_at;
    return end;
}
class Snapshot {
    std::vector<float> state_;
    std::vector<int> tokens_;
    size_t position_=0;
    static size_t capacity(size_t context){
        if(!context||context>131072)throw std::invalid_argument("Invalid LFM turn checkpoint capacity");
        return context;
    }
public:
    explicit Snapshot(size_t context):state_(state_values),tokens_(capacity(context)){}
    void invalidate()noexcept{position_=0;}
    size_t position()const noexcept{return position_;}
    uint64_t bytes()const noexcept{return state_bytes+tokens_.size()*sizeof(int);}
    std::span<const int> prefix()const noexcept{return {tokens_.data(),position_};}
    template<class Copy> bool save(std::span<const int> live,size_t position,Copy copy,const std::atomic<bool>* cancelled=nullptr){
        if(live.empty()||live.size()!=position||position>tokens_.size())throw std::invalid_argument("LFM turn snapshot position/history mismatch");
        if(cancelled&&cancelled->load())return false;
        invalidate();
        // The callback completes all device-to-ordinary-host copies before
        // returning. Never publish a partially copied or cancelled snapshot.
        copy(std::span<float>(state_));
        if(cancelled&&cancelled->load())return false;
        std::copy(live.begin(),live.end(),tokens_.begin());position_=position;return true;
    }
    template<class Copy> size_t restore(std::span<const int> live,std::span<const int> prompt,Copy copy)const{
        const auto selected=choose(live,prompt,prefix(),true);
        if(selected.source!=Source::saved_turn)throw std::invalid_argument("LFM turn snapshot is not an exact resident prefix");
        copy(std::span<const float>(state_));return position_;
    }
};
} // namespace strata::intel::lfm_turn
