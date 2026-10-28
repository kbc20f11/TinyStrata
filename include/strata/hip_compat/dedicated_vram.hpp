// Device-wide dedicated-memory accounting for Windows HIP placement.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <cwchar>
#include <mutex>
#include <string>
#include <vector>
#if !defined(STRATA_DISABLE_DXCORE_VRAM) && __has_include(<dxcore_interface.h>)
#define STRATA_HAS_DXCORE_VRAM 1
#include <dxcore_interface.h>
#else
#define STRATA_HAS_DXCORE_VRAM 0
#endif
#else
#define STRATA_HAS_DXCORE_VRAM 0
#endif

namespace strata::hip_compat {
struct DedicatedVramInfo {
    std::uint64_t total = 0;
    std::uint64_t committed = 0;
    std::uint64_t resident = 0;
    std::uint64_t free_bytes = 0;
    const char* source = nullptr;
};

// Count committed dedicated allocations even if Windows has already evicted
// some of them. They still need dedicated memory when their owners use them.
// Shared host-pinned Expert storage is a separate memory type and is excluded.
inline std::uint64_t dedicated_free_bytes(std::uint64_t total, std::uint64_t committed,
                                          std::uint64_t resident) {
    const std::uint64_t used = std::max(committed, resident);
    return used < total ? total - used : 0;
}

#if defined(_WIN32)
namespace dedicated_vram_detail {
class PdhState {
public:
    std::mutex mutex;
    HMODULE module = nullptr;
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
    decltype(&PdhOpenQueryW) open = nullptr;
    decltype(&PdhAddEnglishCounterW) add = nullptr;
    decltype(&PdhCollectQueryData) collect = nullptr;
    decltype(&PdhGetFormattedCounterArrayW) values = nullptr;
    decltype(&PdhCloseQuery) close = nullptr;

    PdhState() {
        module = LoadLibraryExW(L"pdh.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) return;
        open = reinterpret_cast<decltype(open)>(GetProcAddress(module, "PdhOpenQueryW"));
        add = reinterpret_cast<decltype(add)>(GetProcAddress(module, "PdhAddEnglishCounterW"));
        collect = reinterpret_cast<decltype(collect)>(GetProcAddress(module, "PdhCollectQueryData"));
        values = reinterpret_cast<decltype(values)>(GetProcAddress(module, "PdhGetFormattedCounterArrayW"));
        close = reinterpret_cast<decltype(close)>(GetProcAddress(module, "PdhCloseQuery"));
        if (!open || !add || !collect || !values || !close) return;
        if (open(nullptr, 0, &query) != ERROR_SUCCESS) { query = nullptr; return; }
        // An English path works on localized Windows installations too.
        if (add(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &counter) != ERROR_SUCCESS) {
            close(query); query = nullptr; counter = nullptr;
        }
    }
    ~PdhState() {
        if (query && close) close(query);
        if (module) FreeLibrary(module);
    }
    PdhState(const PdhState&) = delete;
    PdhState& operator=(const PdhState&) = delete;

    bool used(const LUID& luid, std::uint64_t& bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!counter || collect(query) != ERROR_SUCCESS) return false;
        DWORD size = 0, count = 0;
        if (values(counter, PDH_FMT_LARGE, &size, &count, nullptr) != PDH_MORE_DATA || size == 0 ||
            size > 1024 * 1024) return false;
        // 64-bit elements give the PDH records their required alignment.
        std::vector<std::uint64_t> buffer((size + 7) / 8);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (values(counter, PDH_FMT_LARGE, &size, &count, items) != ERROR_SUCCESS) return false;
        wchar_t prefix[80]{};
        std::swprintf(prefix, sizeof prefix / sizeof *prefix, L"luid_0x%08lx_0x%08lx_phys_",
                      static_cast<unsigned long>(luid.HighPart), static_cast<unsigned long>(luid.LowPart));
        const std::wstring wanted(prefix);
        unsigned matching = 0;
        for (DWORD i = 0; i < count; ++i) {
            if (!items[i].szName) continue;
            const std::wstring name(items[i].szName);
            if (name.size() < wanted.size() ||
                CompareStringOrdinal(name.data(), static_cast<int>(wanted.size()),
                                     wanted.data(), static_cast<int>(wanted.size()), TRUE) != CSTR_EQUAL) continue;
            // Refuse linked multi-node adapters rather than budget node 0 for
            // a HIP device whose logical LUID does not identify the node.
            if (name.substr(wanted.size()) != L"0") return false;
            const auto& value = items[i].FmtValue;
            if ((value.CStatus != PDH_CSTATUS_VALID_DATA && value.CStatus != PDH_CSTATUS_NEW_DATA) ||
                value.largeValue < 0) return false;
            bytes = static_cast<std::uint64_t>(value.largeValue);
            ++matching;
        }
        return matching == 1;
    }
};

inline PdhState& pdh_state() {
    static PdhState instance;
    return instance;
}
} // namespace dedicated_vram_detail
#endif

#if STRATA_HAS_DXCORE_VRAM
namespace dedicated_vram_detail {
// These public DXCore states/properties and their fixed-width wire layouts were
// added to newer SDK headers without changing IDXCoreAdapter's QueryState ABI.
// Keep source compatible with older Windows SDKs and ask the OS whether each
// query is supported; an older runtime simply keeps the previous HIP estimate.
constexpr auto kUsage = static_cast<DXCoreAdapterState>(2); // AdapterMemoryUsageBytes
constexpr auto kDedicatedTotal = static_cast<DXCoreAdapterProperty>(7); // DedicatedAdapterMemory
constexpr auto kPhysicalCount = static_cast<DXCoreAdapterProperty>(15); // PhysicalAdapterCount
struct MemoryQueryInput { std::uint32_t physical_adapter_index, memory_type; };
struct MemoryUsage { std::uint64_t committed, resident; };
static_assert(sizeof(MemoryQueryInput) == 8 && sizeof(MemoryUsage) == 16);

struct Adapter {
    LUID luid{};
    IDXCoreAdapter* object = nullptr;
    std::uint64_t total = 0;
    bool ambiguous_nodes = false;
};

class State {
public:
    std::mutex mutex;
    HMODULE module = nullptr;
    IDXCoreAdapterFactory* factory = nullptr;
    std::vector<Adapter> adapters;

    State() {
        // Use only the OS copy; never search the model directory or PATH.
        module = LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) return;
        using Create = HRESULT (WINAPI*)(REFIID, void**);
        const auto create = reinterpret_cast<Create>(GetProcAddress(module, "DXCoreCreateAdapterFactory"));
        if (!create || FAILED(create(__uuidof(IDXCoreAdapterFactory), reinterpret_cast<void**>(&factory))))
            factory = nullptr;
    }
    ~State() {
        for (Adapter& a : adapters) if (a.object) a.object->Release();
        if (factory) factory->Release();
        if (module) FreeLibrary(module);
    }
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    Adapter& find(const LUID& luid) {
        for (Adapter& a : adapters)
            if (a.luid.LowPart == luid.LowPart && a.luid.HighPart == luid.HighPart) return a;
        adapters.push_back({luid, nullptr, 0});
        Adapter& a = adapters.back();
        if (!factory || FAILED(factory->GetAdapterByLuid(luid, __uuidof(IDXCoreAdapter),
                                                         reinterpret_cast<void**>(&a.object))) || !a.object)
            return a;
        std::uint32_t physical_count = 0;
        const bool known_count = a.object->IsPropertySupported(kPhysicalCount) &&
            SUCCEEDED(a.object->GetProperty(kPhysicalCount, sizeof physical_count, &physical_count));
        a.ambiguous_nodes = known_count && physical_count != 1;
        // HIP's LUID identifies the logical adapter. Without an unambiguous
        // physical-node mapping, using node 0 could budget the wrong GPU in LDA.
        if (!a.object->IsQueryStateSupported(kUsage) || !known_count || physical_count != 1 ||
            !a.object->IsPropertySupported(kDedicatedTotal) ||
            FAILED(a.object->GetProperty(kDedicatedTotal, sizeof a.total, &a.total)) || a.total == 0) {
            a.object->Release();
            a.object = nullptr;
            a.total = 0;
        }
        return a;
    }
};

inline State& state() {
    static State instance;
    return instance;
}
} // namespace dedicated_vram_detail
#endif

// LUID bytes come from the selected HIP adapter, not its enumeration ordinal.
// False means unavailable/unsupported: the caller retains its old estimate.
inline bool query_dedicated_vram(const void* luid_bytes, std::size_t luid_size, DedicatedVramInfo& info,
                                 std::uint64_t fallback_total = 0) {
    info = {};
#if defined(_WIN32)
    if (!luid_bytes || luid_size != sizeof(LUID)) return false;
    LUID luid{};
    std::memcpy(&luid, luid_bytes, sizeof luid);
    if (luid.LowPart == 0 && luid.HighPart == 0) return false;
#if STRATA_HAS_DXCORE_VRAM
    {
        auto& state = dedicated_vram_detail::state();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto& adapter = state.find(luid);
        if (adapter.ambiguous_nodes) return false; // PDH must not override a known multi-node refusal.
        if (adapter.object && adapter.object->IsValid()) {
            const dedicated_vram_detail::MemoryQueryInput input{0, 0}; // physical node 0, Dedicated
            dedicated_vram_detail::MemoryUsage usage{};
            if (SUCCEEDED(adapter.object->QueryState(dedicated_vram_detail::kUsage,
                                                    sizeof input, &input, sizeof usage, &usage))) {
                info.total = adapter.total;
                info.committed = usage.committed;
                info.resident = usage.resident;
                info.free_bytes = dedicated_free_bytes(info.total, info.committed, info.resident);
                info.source = "DXCore";
                return true;
            }
        }
    }
#endif
    // Windows 10 drivers may not support the newer DXCore usage state. The
    // adapter-wide counter is an instant gauge (one collection suffices), not
    // the per-process counter whose values can drift on older Windows builds.
    std::uint64_t used = 0;
    if (fallback_total && dedicated_vram_detail::pdh_state().used(luid, used)) {
        info.total = fallback_total;
        info.committed = info.resident = used;
        info.free_bytes = dedicated_free_bytes(info.total, used, used);
        info.source = "PDH";
        return true;
    }
    return false;
#else
    (void) luid_bytes;
    (void) luid_size;
    (void) fallback_total;
    return false;
#endif
}
} // namespace strata::hip_compat
