#if defined(_WIN32)
#define STRATA_HIP_VMM_CPU_TEST 1
#endif
#include "strata/hip_compat/managed_vmm.hpp"
#include <cstdio>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
std::string proof_text() {
    return "TINYSTRATA_HIP_VMM_V1\namd_comgr.dll " + abc + "\namdhip64_7.dll " + abc + "\nrocm_kpack.dll " + abc + "\n";
}
void parser_tests() {
    using namespace strata::hip_vmm;
    Proof p;
    const auto text = proof_text();
    check(parse_proof(text, p), "valid proof rejected");
    check(p.files[1].name == "amdhip64_7.dll" && p.files[1].sha256 == abc, "proof fields changed");
    check(parse_proof(std::string_view(text).substr(0, text.size() - 1), p), "final newline became mandatory");
    std::string crlf;
    for (char c : text) { if (c == '\n') crlf += '\r'; crlf += c; }
    check(parse_proof(crlf, p), "Windows CRLF proof rejected");
    check(!parse_proof("\xef\xbb\xbf" + text, p), "non-ASCII BOM accepted");
    check(!parse_proof(text + "\n", p), "extra blank record accepted");
    check(!parse_proof(text + "extra\n", p), "extra record accepted");
    check(!parse_proof(std::string(1100, 'x'), p), "oversized proof accepted");
    for (const auto& replacement : {"../amd_comgr.dll", "AMD_COMGR.DLL", "amd_comgr_foo.dll", "amd_comgr_.dll", "amdhip64_7.dll"}) {
        auto invalid = text;
        invalid.replace(invalid.find("amd_comgr.dll"), std::string("amd_comgr.dll").size(), replacement);
        check(!parse_proof(invalid, p), "invalid/duplicate core basename accepted");
    }
    auto invalid = text;
    invalid[invalid.find(abc)] = 'G';
    check(!parse_proof(invalid, p), "invalid hash alphabet accepted");
    invalid = text;
    invalid.insert(invalid.find(abc), " ");
    check(!parse_proof(invalid, p), "ambiguous record spacing accepted");
    invalid = text;
    invalid.pop_back(); invalid.push_back('\0');
    check(!parse_proof(invalid, p), "NUL accepted");
    check(versioned_name("amd_comgr_3.dll", "amd_comgr"), "versioned COMGR name rejected");
    check(versioned_name("amdhip64.dll", "amdhip64"), "unversioned HIP name rejected");
}
void policy_tests() {
    using namespace strata::hip_vmm;
    const Evidence good{true, true, true, 71726391, 71726391};
    check(eligible(good), "complete new-runtime evidence rejected");
    for (size_t gran : {size_t(64) << 10, size_t(256) << 10, size_t(3) << 20, size_t(4) << 20, size_t(8) << 20}) {
        check(select_quantum(gran, good) == gran, "verified runtime did not retain reported minimum");
        const size_t coarse = ((std::max(gran, coarse_bytes) / gran) + (std::max(gran, coarse_bytes) % gran != 0)) * gran;
        check(select_quantum(gran, good, true) == coarse, "coarse recovery ignored driver alignment");
        for (int missing = 0; missing < 5; ++missing) {
            auto bad = good;
            if (missing == 0) bad.proof_valid = false;
            if (missing == 1) bad.files_verified = false;
            if (missing == 2) bad.actual_module_verified = false;
            if (missing == 3) bad.runtime_version = 71726390;
            if (missing == 4) bad.runtime_version = bad.compiled_version = 71260602;
            check(!eligible(bad) && select_quantum(gran, bad) == coarse, "incomplete/old/mixed runtime obtained small backing");
        }
    }
    check(select_quantum(0, good) == 0, "unavailable driver granularity fabricated");
    check(eligible(Evidence{true, true, true, 71800001, 71800001}), "matching newer SDK unnecessarily hardcoded out");
}
#if defined(_WIN32)
void windows_io_tests() {
    namespace fs = std::filesystem;
    using namespace strata::hip_vmm;
    const auto bound = detail::imported_function(GetModuleHandleW(nullptr), "kernel32.dll", "GetCurrentProcessId");
    check(bound == reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetCurrentProcessId")), "actual IAT binding differs from loader export");
    check(!detail::imported_function(GetModuleHandleW(nullptr), "missing-library.dll", "GetCurrentProcessId"), "missing import library fabricated");
    check(!detail::imported_function(GetModuleHandleW(nullptr), "kernel32.dll", "missing_symbol"), "missing import symbol fabricated");
    const fs::path directory = fs::temp_directory_path() /
        (L"TinyStrata-HIP-\u691c\u8a3c-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64()));
    check(fs::create_directory(directory), "private CPU fixture directory was not fresh");
    struct Cleanup {
        fs::path directory;
        ~Cleanup() {
            std::error_code e;
            for (const char* name : {"amd_comgr.dll", "amdhip64_7.dll", "rocm_kpack.dll", "empty", "million-a"}) fs::remove(directory / name, e);
            fs::remove(directory / "other/amdhip64_7.dll", e);
            fs::remove(directory / "other", e);
            fs::remove(directory, e);
        }
    } cleanup{directory};
    Proof p;
    check(parse_proof(proof_text(), p), "IO fixture proof invalid");
    for (const auto& item : p.files) { std::ofstream f(directory / item.name, std::ios::binary); f << "abc"; }
    check(detail::verify_files(directory, p), "valid Unicode-path app-local file set rejected");
    { std::ofstream f(directory / "amd_comgr.dll", std::ios::binary); f << "abd"; }
    check(!detail::verify_files(directory, p), "tampered COMGR bytes accepted");
    { std::ofstream f(directory / "amd_comgr.dll", std::ios::binary); f << "abc"; }
    fs::remove(directory / "rocm_kpack.dll");
    check(!detail::verify_files(directory, p), "missing Kpack accepted");
    { std::ofstream f(directory / "rocm_kpack.dll", std::ios::binary); f << "abc"; }
    { std::ofstream f(directory / "empty", std::ios::binary); }
    check(detail::file_hash(directory / "empty") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA256 vector mismatch");
    { std::ofstream f(directory / "million-a", std::ios::binary); f << std::string(1000000, 'a'); }
    check(detail::file_hash(directory / "million-a") == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "multi-block NIST SHA256 vector mismatch");
    fs::create_directory(directory / "other");
    { std::ofstream f(directory / "other/amdhip64_7.dll", std::ios::binary); f << "abc"; }
    check(!detail::same_path(directory / "amdhip64_7.dll", directory / "other/amdhip64_7.dll"), "different loaded-module directory accepted");
    check(detail::same_path(directory / "amdhip64_7.dll", directory / "other/../amdhip64_7.dll"), "equivalent normalized path rejected");
}
#endif
}
int main() {
    try {
        parser_tests(); policy_tests();
#if defined(_WIN32)
        windows_io_tests();
#endif
        std::printf("PASS HIP VMM policy: %d CPU checks; no HIP initialization or GPU operations\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL HIP VMM policy: %s\n", error.what());
        return 1;
    }
}
