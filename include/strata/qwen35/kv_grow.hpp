#pragma once
// CUDA/HIP virtual-memory KV growth, following Strata's core/vmm.hpp and vmm.cpp
// (Niko1221 and the Strata contributors, MIT; see LICENSE and NOTICE).
// Virtual addresses and logical head strides never move. Only physical pages
// behind each head's used prefix are committed; expert arenas may return pages.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::qwen35::kv_grow {
struct Api {
    bool ok = false;
    size_t gran = 0;
    int device = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
#if !defined(STRATA_USE_HIP) && !defined(__HIPCC__)
    template<class F> static bool resolve(const char* name, F& f) {
        cudaDriverEntryPointQueryResult q{};
        void* p = nullptr;
#if CUDART_VERSION >= 12050
        const auto e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
        const auto e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
        if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || !p) {
            (void) cudaGetLastError();
            return false;
        }
        f = reinterpret_cast<F>(p);
        return true;
    }
#endif
    Api() {
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
        // HIP exposes VMM through its runtime API. The CUDA-shaped wrappers in
        // hip_compat/cuda.h preserve Range's stable addresses and page accounting.
        // Older drivers can report VMM unavailable; Engine then allocates the
        // whole logical context before sizing its Expert cache.
        if (hipGetDevice(&device) != hipSuccess) { (void) hipGetLastError(); return; }
        int supports = 0;
        if (cuDeviceGetAttribute(&supports, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device) != CUDA_SUCCESS || !supports) {
            (void) hipGetLastError();
            return;
        }
        reserve = &cuMemAddressReserve;
        free_va = &cuMemAddressFree;
        create = &cuMemCreate;
        release = &cuMemRelease;
        map = &cuMemMap;
        unmap = &cuMemUnmap;
        access = &cuMemSetAccess;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        ok = cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && gran != 0;
#if defined(_WIN32)
        if (ok) {
            // Small Windows HIP handles can pass API checks but fail GPU
            // access (#1). Use coarse backing while respecting the minimum.
            constexpr size_t chunk = size_t(4) << 20;
            gran = ((std::max(gran, chunk) + gran - 1) / gran) * gran;
        }
#endif
        if (!ok) (void) hipGetLastError();
#else
        decltype(&cuCtxGetDevice) current_device = nullptr;
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        // The model has already initialized the current runtime context. Driver
        // allocations must use its CUdevice, rather than a runtime ordinal.
        if (!resolve("cuCtxGetDevice", current_device) || current_device(&device) != CUDA_SUCCESS ||
            !resolve("cuDeviceGetAttribute", attr) ||
            !resolve("cuMemGetAllocationGranularity", granularity) || !resolve("cuMemAddressReserve", reserve) ||
            !resolve("cuMemAddressFree", free_va) || !resolve("cuMemCreate", create) ||
            !resolve("cuMemRelease", release) || !resolve("cuMemMap", map) ||
            !resolve("cuMemUnmap", unmap) || !resolve("cuMemSetAccess", access)) return;
        int supports = 0;
        if (attr(&supports, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device) != CUDA_SUCCESS || !supports)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        ok = granularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && gran != 0;
#endif
    }
};
inline const Api& api() { static const Api a; return a; }

// Strata's program/generate.cpp kvg_ensure/kvg_trim transfers live VMM chunks
// between Expert and KV addresses through kvg.spare (Niko1221 and contributors,
// MIT). Keeping the handle alive also avoids Windows HIP release/recreate
// accounting: a page can serve KV without requiring newly reported free VRAM.
// The Engine owns this pool, outlives every attached Range, and serializes all
// mutations after draining the streams which could still use a returned page.
class PagePool {
    std::vector<CUmemGenericAllocationHandle> spare_;
public:
    PagePool() = default;
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;
    [[noreturn]] static void failed(const char* operation) {
        std::fprintf(stderr, "tinystrata: HIP VMM page ownership failed during %s; stopping before another GPU access\n",
                     operation);
        std::abort();
    }
    ~PagePool() {
        // Attached Ranges have already been destroyed. These handles have no
        // mappings or pending readers, and each is released exactly once here.
        for (auto h : spare_)
            if (api().release(h) != CUDA_SUCCESS) failed("releasing a spare page");
    }
    size_t bytes() const { return spare_.size() * api().gran; }
    size_t fresh_bytes(size_t mapped_need) const {
        const size_t reusable = bytes();
        return mapped_need > reusable ? mapped_need - reusable : 0;
    }
    // Reserve before unmapping anything, so a host allocation failure cannot
    // strand a live physical handle between its Range and the pool.
    bool prepare_return(size_t count) {
        try { spare_.reserve(spare_.size() + count); }
        catch (...) { return false; }
        return true;
    }
    CUmemGenericAllocationHandle take() {
        if (spare_.empty()) return 0;
        const auto h = spare_.back();
        spare_.pop_back();
        return h;
    }
    void give(CUmemGenericAllocationHandle h) {
        // A trim reserved its returns; a failed borrowed map restores the slot
        // that take() just removed. No allocating push is allowed after unmap.
        if (!h || spare_.size() == spare_.capacity()) failed("returning an unowned page");
        spare_.push_back(h);
    }
};

class Range {
    CUdeviceptr base_ = 0;
    std::vector<CUmemGenericAllocationHandle> pages_;
    PagePool* pool_ = nullptr;
public:
    explicit Range(PagePool* pool = nullptr) : pool_(pool) {}
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
    ~Range() {
        if (!base_) return;
        // A partially initialized range may still have queued zeroing/copies.
        // This also covers local ranges destroyed before Engine cleanup.
        const auto drained = cudaDeviceSynchronize();
        if (pool_ && drained != cudaSuccess) PagePool::failed("draining a Range destructor");
        const auto& a = api();
        for (size_t i = 0; i < pages_.size(); ++i) if (pages_[i]) {
            const auto unmap = a.unmap(base_ + i * a.gran, a.gran);
            if (pool_ && unmap != CUDA_SUCCESS) PagePool::failed("unmapping a Range destructor");
            const auto release = a.release(pages_[i]);
            if (pool_ && release != CUDA_SUCCESS) PagePool::failed("releasing a Range destructor");
        }
        const auto free_va = a.free_va(base_, pages_.size() * a.gran);
        if (pool_ && free_va != CUDA_SUCCESS) PagePool::failed("freeing a Range address");
    }
    bool reserve(size_t bytes) {
        const auto& a = api();
        if (!a.ok || base_) return false;
        const size_t n = (bytes + a.gran - 1) / a.gran;
        if (!n || a.reserve(&base_, n * a.gran, 0, 0, 0) != CUDA_SUCCESS) return false;
        pages_.resize(n, 0);
        return true;
    }
    void* base() const { return reinterpret_cast<void*>(base_); }
    size_t mapped() const {
        size_t n = 0;
        for (auto h : pages_) n += h != 0;
        return n * api().gran;
    }
    // Prefixes can share a page at a head boundary when ctx is not aligned.
    // A union mask avoids double mapping or unmapping such boundary pages.
    std::vector<uint8_t> mask(size_t rows, size_t heads, size_t stride, size_t row_bytes) const {
        const size_t g = api().gran;
        std::vector<uint8_t> out(pages_.size(), 0);
        if (!rows) return out;
        for (size_t h = 0; h < heads; ++h) {
            const size_t lo = h * stride / g;
            const size_t hi = (h * stride + rows * row_bytes + g - 1) / g;
            for (size_t i = lo; i < hi && i < out.size(); ++i) out[i] = 1;
        }
        return out;
    }
    size_t need(const std::vector<uint8_t>& want) const {
        size_t n = 0;
        for (size_t i = 0; i < pages_.size(); ++i) n += want[i] && !pages_[i];
        return n * api().gran;
    }
    bool map(const std::vector<uint8_t>& want, cudaStream_t stream) {
        const auto& a = api();
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.device;
        CUmemAccessDesc access{};
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        for (size_t i = 0; i < pages_.size(); ++i) if (want[i] && !pages_[i]) {
            CUmemGenericAllocationHandle h = pool_ ? pool_->take() : 0;
            const bool borrowed = h != 0;
            if (!h && a.create(&h, a.gran, &prop, 0) != CUDA_SUCCESS) return false;
            const auto give_back = [&] {
                if (borrowed) pool_->give(h);
                else if (a.release(h) != CUDA_SUCCESS && pool_) PagePool::failed("rolling back a fresh page");
            };
            const CUdeviceptr p = base_ + i * a.gran;
            if (a.map(p, a.gran, 0, h, 0) != CUDA_SUCCESS) { give_back(); return false; }
            if (a.access(p, a.gran, &access, 1) != CUDA_SUCCESS) {
                const auto unmap = a.unmap(p, a.gran);
                // Do not put a possibly still mapped handle back into the pool.
                if (pool_ && unmap != CUDA_SUCCESS) PagePool::failed("rolling back page access");
                give_back();
                return false;
            }
            pages_[i] = h;
            if (cudaMemsetAsync(reinterpret_cast<void*>(p), 0, a.gran, stream) != cudaSuccess) {
                if (pool_) {
                    // A borrowed page contains an old Expert/KV payload. Never
                    // expose it as a newly zeroed KV prefix after a failed fill.
                    if (cudaDeviceSynchronize() != cudaSuccess) PagePool::failed("draining a failed page fill");
                    if (a.unmap(p, a.gran) != CUDA_SUCCESS) PagePool::failed("rolling back a page fill");
                    pages_[i] = 0;
                    give_back();
                }
                return false;
            }
        }
        return true;
    }
    // The caller must have drained every stream that could access these pages.
    bool trim(const std::vector<uint8_t>& want) {
        const auto& a = api();
        if (pool_) {
            size_t count = 0;
            for (size_t i = 0; i < pages_.size(); ++i) count += !want[i] && pages_[i];
            if (!pool_->prepare_return(count)) return false;
        }
        for (size_t i = 0; i < pages_.size(); ++i) if (!want[i] && pages_[i]) {
            if (a.unmap(base_ + i * a.gran, a.gran) != CUDA_SUCCESS) return false;
            if (pool_) pool_->give(pages_[i]);
            else a.release(pages_[i]);
            pages_[i] = 0;
        }
        return true;
    }
    std::vector<uint8_t> prefix(size_t bytes) const { return mask(bytes, 1, 0, 1); }
};
} // namespace strata::qwen35::kv_grow
