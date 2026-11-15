#pragma once
#include "strata/intel/gemma_ops.hpp"
#include "strata/intel/gemma_prefix.hpp"
#include <atomic>
#include <cstddef>
#include <vector>

namespace strata::intel {
// One ordinary host allocation owns only sliding-window KV codes/scales.
// Full-attention KV stays in the existing resident arena and is trusted only
// while the caller's exact saved prefix remains a prefix of committed history.
class GemmaTurnSnapshot {
    struct Part {void* device;size_t offset,bytes;};
    std::vector<GemmaKv> sliding_;
    std::vector<Part> parts_;
    std::vector<uint8_t> data_;
    size_t position_=0;
    bool valid_=false;
public:
    explicit GemmaTurnSnapshot(const std::vector<GemmaKv>& caches){
        size_t total=0;
        for(const auto& c:caches)if(c.window){
            const auto n=gemma_prefix::snapshot_bytes(c.capacity,c.heads,c.dim);
            if(n>SIZE_MAX-total)throw std::overflow_error("Gemma turn snapshot size overflow");
            sliding_.push_back(c);
            const size_t cells=size_t(c.capacity)*c.heads*c.dim;
            const auto part=[&](void* p,size_t bytes){if(!p)throw std::invalid_argument("Null Gemma turn cache");parts_.push_back({p,total,bytes});total+=bytes;};
            part(c.key,cells);part(c.value,cells);part(c.key_scale,cells/32*sizeof(sycl::half));part(c.value_scale,cells/32*sizeof(sycl::half));
        }
        data_.resize(total);
    }
    size_t bytes()const{return data_.size();}
    bool valid()const{return valid_;}
    size_t position()const{return valid_?position_:0;}
    void invalidate(){valid_=false;position_=0;}
    bool save(sycl::queue& q,size_t logical,size_t written,size_t chunk,const std::atomic<bool>* cancelled=nullptr){
        invalidate();
        if(!logical||logical>written||written-logical>chunk)throw std::invalid_argument("Invalid Gemma post-chunk turn boundary");
        for(const auto& c:sliding_)if(!gemma_prefix::ring_retains(logical,written,c.capacity,c.window))
            throw std::invalid_argument("Gemma turn snapshot no longer retains its required window");
        for(const auto& p:parts_)q.memcpy(data_.data()+p.offset,p.device,p.bytes);
        q.wait_and_throw();
        if(cancelled&&cancelled->load())return false;
        position_=logical;valid_=true;return true;
    }
    size_t restore(sycl::queue& q){
        if(!valid_)throw std::logic_error("No completed Gemma turn snapshot");
        for(const auto& p:parts_)q.memcpy(p.device,data_.data()+p.offset,p.bytes);
        q.wait_and_throw();return position_;
    }
};
}
