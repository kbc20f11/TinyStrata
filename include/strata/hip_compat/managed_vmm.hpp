#pragma once
// Small Windows HIP VMM backing requires one verified app-local runtime set.
// CUDA and other platforms keep the driver's reported granularity unchanged.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace strata::hip_vmm {
inline constexpr size_t coarse_bytes = size_t(4) << 20;
inline constexpr int minimum_runtime = 71700000;
inline constexpr const char* proof_name = ".tinystrata-hip-vmm.txt";

struct CoreFile { std::string name, sha256; };
struct Proof { std::array<CoreFile, 3> files; };

inline bool versioned_name(std::string_view name, std::string_view base) {
    if (name == std::string(base) + ".dll") return true;
    if (!name.starts_with(base) || !name.ends_with(".dll") || name.size() <= base.size() + 5 ||
        name[base.size()] != '_') return false;
    for (size_t i = base.size() + 1; i < name.size() - 4; ++i)
        if (name[i] < '0' || name[i] > '9') return false;
    return true;
}

// This is an installer-produced fixed ASCII format, not a second JSON parser.
// Exact records, ordering and hashes also match the Python-verified manifest.
inline bool parse_proof(std::string_view text, Proof& result) {
    if (text.empty() || text.size() > 1024) return false;
    std::array<std::string_view, 4> lines{};
    size_t at = 0;
    for (auto& line : lines) {
        if (at >= text.size()) return false;
        const size_t end = text.find('\n', at);
        line = text.substr(at, (end == text.npos ? text.size() : end) - at);
        if (line.ends_with('\r')) line.remove_suffix(1);
        at = end == text.npos ? text.size() : end + 1;
    }
    if (at != text.size() || lines[0] != "TINYSTRATA_HIP_VMM_V1") return false;
    Proof parsed;
    int hip = 0, comgr = 0, kpack = 0;
    for (size_t i = 0; i < parsed.files.size(); ++i) {
        const auto line = lines[i + 1];
        const size_t space = line.find(' ');
        if (space == line.npos || space == 0 || line.size() - space - 1 != 64) return false;
        auto& file = parsed.files[i];
        file.name = line.substr(0, space);
        file.sha256 = line.substr(space + 1);
        if (i && parsed.files[i - 1].name >= file.name) return false;
        if (versioned_name(file.name, "amdhip64")) ++hip;
        else if (versioned_name(file.name, "amd_comgr")) ++comgr;
        else if (file.name == "rocm_kpack.dll") ++kpack;
        else return false;
        for (char c : file.sha256)
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    if (hip != 1 || comgr != 1 || kpack != 1) return false;
    result = std::move(parsed);
    return true;
}

struct Evidence {
    bool proof_valid = false;
    bool files_verified = false;
    bool actual_module_verified = false;
    int runtime_version = 0;
    int compiled_version = 0;
};
inline bool eligible(const Evidence& e) {
    return e.proof_valid && e.files_verified && e.actual_module_verified &&
        e.runtime_version >= minimum_runtime && e.runtime_version == e.compiled_version;
}
inline size_t select_quantum(size_t driver_minimum, const Evidence& e, bool force_coarse = false) {
    if (!driver_minimum) return 0;
    const size_t target = eligible(e) && !force_coarse ? driver_minimum : std::max(driver_minimum, coarse_bytes);
    const size_t remainder = target % driver_minimum;
    return remainder ? target + (driver_minimum - remainder) : target;
}
} // namespace strata::hip_vmm

#if defined(_WIN32) && (defined(STRATA_USE_HIP) || defined(__HIPCC__) || defined(STRATA_HIP_VMM_CPU_TEST))
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>
#endif
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace strata::hip_vmm {
namespace detail {
// HIP's declarations need not carry dllimport. Taking &hipRuntimeGetVersion
// can therefore name an EXE jump thunk. Read the loader-resolved IAT slot of
// that exact import instead; no instruction decoding or extra HIP API call.
inline void* imported_function(HMODULE executable, const char* library, const char* symbol) {
    if (!executable || sizeof(void*) != 8) return nullptr;
    const auto* base = reinterpret_cast<const unsigned char*>(executable);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0 || dos->e_lfanew > (1 << 20)) return nullptr;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    const size_t size = nt->OptionalHeader.SizeOfImage;
    const auto inside = [&](size_t rva, size_t bytes) { return rva < size && bytes <= size - rva; };
    const auto ascii = [&](size_t rva) -> const char* {
        if (!inside(rva, 1)) return nullptr;
        const auto* p = reinterpret_cast<const char*>(base + rva);
        return std::memchr(p, 0, std::min<size_t>(size - rva, 256)) ? p : nullptr;
    };
    const auto directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!directory.VirtualAddress || !inside(directory.VirtualAddress, directory.Size)) return nullptr;
    const size_t count = std::min<size_t>(directory.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR), 4096);
    for (size_t i = 0; i < count; ++i) {
        const auto* descriptor = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress) + i;
        if (!descriptor->Name) break;
        const char* dll = ascii(descriptor->Name);
        if (!dll || _stricmp(dll, library) != 0) continue;
        if (!descriptor->OriginalFirstThunk || !descriptor->FirstThunk) return nullptr;
        for (size_t j = 0; j < 65536; ++j) {
            const size_t offset = j * sizeof(IMAGE_THUNK_DATA64);
            if (!inside(descriptor->OriginalFirstThunk, offset + sizeof(IMAGE_THUNK_DATA64)) ||
                !inside(descriptor->FirstThunk, offset + sizeof(IMAGE_THUNK_DATA64))) return nullptr;
            const auto entry = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + descriptor->OriginalFirstThunk)[j];
            if (!entry.u1.AddressOfData) break;
            if (IMAGE_SNAP_BY_ORDINAL64(entry.u1.Ordinal) || !inside(entry.u1.AddressOfData, sizeof(WORD) + 1)) continue;
            const char* name = ascii(entry.u1.AddressOfData + sizeof(WORD));
            if (name && std::strcmp(name, symbol) == 0) {
                const auto address = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + descriptor->FirstThunk)[j].u1.Function;
                return reinterpret_cast<void*>(static_cast<uintptr_t>(address));
            }
        }
        return nullptr;
    }
    return nullptr;
}
inline std::wstring module_path(HMODULE module) {
    std::wstring path(32768, L'\0');
    const DWORD n = GetModuleFileNameW(module, path.data(), (DWORD) path.size());
    if (!n || n >= path.size()) return {};
    path.resize(n);
    return path;
}
inline std::wstring final_path(const std::filesystem::path& path) {
    HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    std::wstring canonical(32768, L'\0');
    const DWORD n = GetFinalPathNameByHandleW(file, canonical.data(), (DWORD) canonical.size(),
        FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(file);
    if (!n || n >= canonical.size()) return {};
    canonical.resize(n);
    return canonical;
}
inline bool same_path(const std::filesystem::path& a, const std::filesystem::path& b) {
    const auto x = final_path(a), y = final_path(b);
    return !x.empty() && !y.empty() && CompareStringOrdinal(x.c_str(), (int) x.size(), y.c_str(), (int) y.size(), TRUE) == CSTR_EQUAL;
}

// Dynamically resolve the OS hashing API; no new link-time library, PATH lookup
// or vendor runtime is introduced. Only the existing loaded HIP is inspected.
struct HashApi {
    HMODULE module = LoadLibraryExW(L"bcrypt.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    decltype(&BCryptOpenAlgorithmProvider) open = nullptr;
    decltype(&BCryptGetProperty) property = nullptr;
    decltype(&BCryptCreateHash) create = nullptr;
    decltype(&BCryptHashData) update = nullptr;
    decltype(&BCryptFinishHash) finish = nullptr;
    decltype(&BCryptDestroyHash) destroy = nullptr;
    decltype(&BCryptCloseAlgorithmProvider) close = nullptr;
    HashApi() {
        if (!module) return;
        open = reinterpret_cast<decltype(open)>(GetProcAddress(module, "BCryptOpenAlgorithmProvider"));
        property = reinterpret_cast<decltype(property)>(GetProcAddress(module, "BCryptGetProperty"));
        create = reinterpret_cast<decltype(create)>(GetProcAddress(module, "BCryptCreateHash"));
        update = reinterpret_cast<decltype(update)>(GetProcAddress(module, "BCryptHashData"));
        finish = reinterpret_cast<decltype(finish)>(GetProcAddress(module, "BCryptFinishHash"));
        destroy = reinterpret_cast<decltype(destroy)>(GetProcAddress(module, "BCryptDestroyHash"));
        close = reinterpret_cast<decltype(close)>(GetProcAddress(module, "BCryptCloseAlgorithmProvider"));
    }
    ~HashApi() { if (module) FreeLibrary(module); }
    bool ready() const { return open && property && create && update && finish && destroy && close; }
};
inline std::string file_hash(const std::filesystem::path& path) {
    static const HashApi api;
    if (!api.ready()) return {};
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    if (api.open(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) return {};
    struct Close { const HashApi& api; BCRYPT_ALG_HANDLE h; ~Close() { api.close(h, 0); } } close{api, algorithm};
    DWORD object_bytes = 0, got = 0;
    if (api.property(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes), &got, 0) < 0 ||
        got != sizeof(object_bytes) || !object_bytes || object_bytes > (1u << 20)) return {};
    std::vector<unsigned char> object(object_bytes);
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (api.create(algorithm, &hash, object.data(), object_bytes, nullptr, 0, 0) < 0) return {};
    struct Destroy { const HashApi& api; BCRYPT_HASH_HANDLE h; ~Destroy() { api.destroy(h); } } destroy{api, hash};
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::array<unsigned char, 65536> data{};
    while (file) {
        file.read(reinterpret_cast<char*>(data.data()), data.size());
        const auto count = file.gcount();
        if (count && api.update(hash, data.data(), (ULONG) count, 0) < 0) return {};
    }
    if (!file.eof()) return {};
    std::array<unsigned char, 32> bytes{};
    if (api.finish(hash, bytes.data(), (ULONG) bytes.size(), 0) < 0) return {};
    constexpr char digits[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        result[2 * i] = digits[bytes[i] >> 4];
        result[2 * i + 1] = digits[bytes[i] & 15];
    }
    return result;
}
inline bool verify_files(const std::filesystem::path& directory, const Proof& proof) {
    for (const auto& item : proof.files) {
        const auto path = directory / item.name;
        const auto canonical = final_path(path);
        if (canonical.empty() || !same_path(std::filesystem::path(canonical).parent_path(), directory) ||
            file_hash(path) != item.sha256) return false;
    }
    return true;
}
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
struct Verification { Evidence evidence; const char* reason = "unverified"; };
inline Verification inspect_runtime() {
    Verification v;
    v.evidence.compiled_version = HIP_VERSION;
    if (hipRuntimeGetVersion(&v.evidence.runtime_version) != hipSuccess) { v.reason = "runtime_query_failed"; return v; }
    if (v.evidence.runtime_version < minimum_runtime) { v.reason = "old_runtime"; return v; }
    if (v.evidence.runtime_version != HIP_VERSION) { v.reason = "runtime_compile_mismatch"; return v; }
    try {
        const auto exe = module_path(nullptr);
        if (exe.empty()) { v.reason = "executable_path_failed"; return v; }
        const auto directory = std::filesystem::path(exe).parent_path();
        std::ifstream file(directory / proof_name, std::ios::binary | std::ios::ate);
        if (!file) { v.reason = "no_managed_proof"; return v; }
        const auto size = file.tellg();
        if (size <= 0 || size > 1024) { v.reason = "invalid_managed_proof"; return v; }
        std::string text((size_t) size, '\0');
        file.seekg(0);
        if (!file.read(text.data(), text.size())) { v.reason = "invalid_managed_proof"; return v; }
        Proof proof;
        if (!parse_proof(text, proof)) { v.reason = "invalid_managed_proof"; return v; }
        v.evidence.proof_valid = true;
        if (!verify_files(directory, proof)) { v.reason = "core_hash_mismatch"; return v; }
        v.evidence.files_verified = true;
        std::filesystem::path hip_path;
        std::string hip_name;
        for (const auto& item : proof.files) {
            const auto expected = directory / item.name;
            if (versioned_name(item.name, "amdhip64")) { hip_path = expected; hip_name = item.name; }
            const std::wstring wide(item.name.begin(), item.name.end());
            if (const HMODULE loaded = GetModuleHandleW(wide.c_str())) {
                const auto actual = module_path(loaded);
                if (actual.empty() || !same_path(actual, expected)) { v.reason = "loaded_core_path_mismatch"; return v; }
            }
        }
        const void* target = imported_function(GetModuleHandleW(nullptr), hip_name.c_str(), "hipRuntimeGetVersion");
        if (!target) { v.reason = "runtime_import_failed"; return v; }
        HMODULE actual_hip = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(target), &actual_hip)) { v.reason = "runtime_module_failed"; return v; }
        const auto actual_path = module_path(actual_hip);
        if (actual_path.empty() || !same_path(actual_path, hip_path)) { v.reason = "runtime_module_mismatch"; return v; }
        v.evidence.actual_module_verified = true;
        v.reason = "verified_managed";
    } catch (...) { v.reason = "verification_failed"; }
    return v;
}
#endif
} // namespace detail

#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
inline size_t quantum(size_t driver_minimum, const char* family) {
    static const detail::Verification checked = detail::inspect_runtime();
    wchar_t escape[16]{};
    const DWORD n = GetEnvironmentVariableW(L"TINYSTRATA_HIP_VMM_COARSE", escape, (DWORD) (sizeof(escape) / sizeof(escape[0])));
    // Only coarse recovery is expressible. Unknown values also stay coarse.
    const bool coarse = n != 0;
    const size_t actual = select_quantum(driver_minimum, checked.evidence, coarse);
    std::printf("INFO hip_vmm_family=%s hip_vmm_verified=%d hip_vmm_driver_min_kib=%zu hip_vmm_quantum_kib=%zu "
                "hip_runtime_version=%d hip_compiled_version=%d hip_vmm_reason=%s\n",
        family, eligible(checked.evidence) ? 1 : 0, driver_minimum >> 10, actual >> 10,
        checked.evidence.runtime_version, checked.evidence.compiled_version, coarse ? "coarse_recovery" : checked.reason);
    std::fflush(stdout);
    return actual;
}
#endif
} // namespace strata::hip_vmm
#endif
