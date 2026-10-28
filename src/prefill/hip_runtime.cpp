// AMD-only runtime support adapted from Niko1221/Strata's src/core/device.cu
// at cbfd5b926b794b356195d3cb600191b0c3567540 (MIT).
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
// The free VRAM figure on Windows HIP (include/strata/hip_compat/cuda_runtime.h maps cudaMemGetInfo here).
//
// hipMemGetInfo (ROCclr's PAL backend, Device::globalFreeMemory) is the card's size minus this process's own
// allocations: it asks Windows for this process's usage only, never for what the desktop and other programs hold.
// So on a card that also drives the desktop the engine counted ~930 MiB that was not there, the expert cache filled
// the card, and WDDM moved memory out to system RAM: decode at 30 tok/s instead of 41 (RX 6800, HIP SDK 7.2).
// Windows itself gives each process a video memory budget (DXGI QueryVideoMemoryInfo) that does account for the
// others.  Measured on that card: the budget sits 0.8 GiB below the card's size while this process is small and
// 1.8 GiB below once it holds 11 GiB; allocations past it still succeed, and are what Windows moves out.  The
// free figure here is hipMemGetInfo's lowered by what the budget withholds (the card's size minus the budget).
// Subtracting from HIP's own figure, rather than taking the budget minus DXGI's usage, keeps memory the HIP runtime
// has freed and holds in its cache counted as free, as hipMemGetInfo counts it.  dxgi.dll is loaded at run time, so
// nothing new is linked; when it or the card's adapter cannot be found, hipMemGetInfo's figure stands.
// That per-process budget is not a device-wide free counter. A separate Vision
// worker can still be invisible to it. The final clamp below uses DXCore's
// adapter-wide dedicated committed/resident memory (PDH on older Windows).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include "strata/hip_compat/dedicated_vram.hpp"

namespace strata::hip_compat {
namespace {
struct DeviceAdapter {
    LUID luid{};
    IDXGIAdapter3* budget = nullptr;
};

DeviceAdapter device_adapter(int device) {
    constexpr int kMaxDevices = 16;
    static std::mutex mu;
    static DeviceAdapter adapters[kMaxDevices] = {};
    static bool tried[kMaxDevices] = {};
    if (device < 0 || device >= kMaxDevices) return {};
    std::lock_guard<std::mutex> lock(mu);
    if (tried[device]) return adapters[device];
    tried[device] = true;
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device) != hipSuccess) {
        (void) hipGetLastError();
        return {};
    }
    LUID luid{};
    std::memcpy(&luid, p.luid, sizeof luid);
    adapters[device].luid = luid;
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    static HMODULE dxgi = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = dxgi ? (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory4* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory4), (void**) &factory))) return adapters[device];
    IDXGIAdapter3* adapter = nullptr;
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**) &adapter))) adapter = nullptr;
    factory->Release();
    adapters[device].budget = adapter;
    return adapters[device];
}
}  // namespace

// #873 / #654: on some Windows setups the first hipMemGetInfo of a process fails with hipErrorInvalidValue because the ROCm
// runtime has not finished initialising against the Adrenalin WDDM driver, and the first kernel launch after that
// crashes (0xC0000005 in amdhip64_7.dll).  A hipFree(0) before the first query initialises the context, and one
// failed query gets one retry after 250 ms.  Both happen once per process, before the DXGI budget below; any other
// error, and a failure of the retry, are returned as they are (a real failure is never retried).
static hipError_t first_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    static std::atomic<bool> warmed{false}, retried{false};
    if (!warmed.exchange(true)) (void) hipFree(nullptr);
    hipError_t e = hipMemGetInfo(free_bytes, total_bytes);
    if (e == hipErrorInvalidValue && !retried.exchange(true)) {
        (void) hipGetLastError();
        std::fprintf(stderr, "tinystrata: hipMemGetInfo failed (%s); retrying once after 250 ms\n", hipGetErrorString(e));
        Sleep(250);
        e = hipMemGetInfo(free_bytes, total_bytes);
        std::fprintf(stderr, "tinystrata: hipMemGetInfo retry: %s\n", e == hipSuccess ? "ok" : hipGetErrorString(e));
    }
    return e;
}

hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return hipMemGetInfo(free_bytes, total_bytes);
    const hipError_t e = first_mem_get_info(free_bytes, total_bytes);
    static const bool off = [] {
        const char* v = std::getenv("STRATA_WDDM_BUDGET");
        return v != nullptr && std::atoi(v) == 0;
    }();
    if (e != hipSuccess || off || free_bytes == nullptr || total_bytes == nullptr) return e;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        (void) hipGetLastError();
        return e;
    }
    const DeviceAdapter adapter = device_adapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    if (adapter.budget && SUCCEEDED(adapter.budget->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
        local.Budget > 0 && local.Budget < *total_bytes) {
        const size_t withheld = *total_bytes - (size_t) local.Budget;
        static std::atomic<bool> said_budget{false};
        if (!said_budget.exchange(true)) {
            std::fprintf(stderr, "tinystrata: Windows budgets %llu of this card's %llu MiB for this process; free VRAM is "
                                 "counted within that (STRATA_WDDM_BUDGET=0: off)\n",
                         (unsigned long long) (local.Budget >> 20), (unsigned long long) (*total_bytes >> 20));
        }
        *free_bytes = *free_bytes > withheld ? *free_bytes - withheld : 0;
    }
    // A per-process budget alone can still include VRAM held by another process,
    // notably the GPU Vision worker. Clamp placement to this same adapter's
    // device-wide dedicated capacity; count evicted allocations as occupied too.
    DedicatedVramInfo physical{};
    if (query_dedicated_vram(&adapter.luid, sizeof adapter.luid, physical, *total_bytes)) {
        const size_t estimate = *free_bytes;
        *free_bytes = std::min(estimate, static_cast<size_t>(physical.free_bytes));
        static std::atomic<bool> said_physical{false};
        if (!said_physical.exchange(true)) {
            std::fprintf(stderr, "tinystrata: device-wide VRAM (%s): %llu MiB free of %llu MiB; HIP/WDDM estimate "
                                 "%llu MiB, using %llu MiB (includes Vision and other processes)\n",
                         physical.source, (unsigned long long) (physical.free_bytes >> 20),
                         (unsigned long long) (physical.total >> 20), (unsigned long long) (estimate >> 20),
                         (unsigned long long) (*free_bytes >> 20));
        }
    } else {
        static std::atomic<bool> said_fallback{false};
        if (!said_fallback.exchange(true))
            std::fprintf(stderr, "tinystrata: device-wide VRAM query unavailable; using the HIP/WDDM estimate\n");
    }
    return e;
}
}  // namespace strata::hip_compat
#endif
