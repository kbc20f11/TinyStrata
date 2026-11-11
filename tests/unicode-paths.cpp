// CPU-only consumer of independent GGUF fixtures created by Python.
#include "strata/artifact/gguf_reader.hpp"
#ifndef STRATA_UNICODE_PATH_LEGACY_PROBE
#include "strata/common/utf8_path.hpp"
#endif
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void need(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void mapped(const std::string& path) {
    const strata::GgufFile file(path);
    need(file.version() == 3, "wrong GGUF version");
    const auto* name = file.get("general.name");
    need(name && name->s == "independent unicode fixture", "wrong independent metadata");
    const auto* tensor = file.find("weight");
    need(tensor && tensor->shape == std::vector<uint64_t>{4} && tensor->type == 0, "wrong tensor directory");
    const float expected[4] = {1.f, -2.f, 3.5f, 0.f};
    need(std::memcmp(file.tensor_data(*tensor), expected, sizeof expected) == 0, "wrong mapped payload");
}
template<class F> void invalid(F&& call) {
    bool rejected = false;
    try { call(); }
    catch (const std::invalid_argument&) { rejected = true; }
    need(rejected, "invalid path was not rejected");
}
}

int main(int argc, char** argv) {
    try {
        need(argc == 2, "one ASCII operation is required");
        const std::string mode = argv[1];
        std::string path;
        std::getline(std::cin, path); // explicitly UTF-8 protocol, independent of main's ANSI argv
        if (mode == "mapped") mapped(path);
        else if (mode == "missing") {
            bool rejected = false;
            try { const strata::GgufFile file(path); }
            catch (const std::runtime_error& error) {
                const std::string text = error.what();
                rejected = text.find("cannot open") != std::string::npos && text.find(path) != std::string::npos;
            }
            need(rejected, "missing-file error lost its UTF-8 path");
        } else if (mode == "split") {
            const auto parts = strata::gguf_split_paths(path);
            need(parts.size() == 3, "expected three independently written shards");
            for (size_t i = 0; i < parts.size(); ++i) {
                const std::string ending = "-0000" + std::to_string(i + 1) + "-of-00003.gguf";
                need(parts[i].size() >= ending.size() && parts[i].compare(parts[i].size() - ending.size(), ending.size(), ending) == 0,
                     "wrong shard order or spelling");
                mapped(parts[i]);
            }
        } else if (mode == "split-missing") {
            bool rejected = false;
            try { (void) strata::gguf_split_paths(path); }
            catch (const std::runtime_error& error) {
                const std::string text = error.what();
                std::string missing = path;
                const size_t at = missing.rfind("-of-");
                need(at != std::string::npos, "invalid test shard name");
                missing.replace(at - 5, 5, "00002");
                rejected = text.find(missing) != std::string::npos && text.find("shard 2 of 3") != std::string::npos;
            }
            need(rejected, "missing-shard error lost its UTF-8 filename/index");
        } else if (mode == "fopen") {
#ifdef STRATA_UNICODE_PATH_LEGACY_PROBE
            std::FILE* file = std::fopen(path.c_str(), "rb");
#else
            std::FILE* file = strata::paths::fopen(path.c_str(), "rb");
#endif
            need(file != nullptr, "fopen did not open the UTF-8 fixture path");
            char bytes[4]{};
            const size_t got = std::fread(bytes, 1, sizeof bytes, file);
            std::fclose(file);
            need(got == 4 && std::memcmp(bytes, "GGUF", 4) == 0, "wrong file content");
        }
#ifndef STRATA_UNICODE_PATH_LEGACY_PROBE
        else if (mode == "roundtrip") {
            need(strata::paths::utf8_path(strata::paths::native_path(path)) == path, "path roundtrip changed bytes");
        } else if (mode == "write") {
            std::FILE* file = strata::paths::fopen(path.c_str(), "wb");
            need(file != nullptr, "cannot write UTF-8 output path");
            const char payload[] = "independent output\n";
            const size_t wrote = std::fwrite(payload, 1, sizeof payload - 1, file);
            const int closed = std::fclose(file);
            need(wrote == sizeof payload - 1 && closed == 0, "incomplete output");
        } else if (mode == "env") {
            const auto value = strata::paths::getenv("TINYSTRATA_UTF8_TEST_PATH");
            need(value && *value == path, "environment path lost Unicode");
        } else if (mode == "env-unset") {
            need(!strata::paths::getenv("TINYSTRATA_UTF8_TEST_PATH"), "unset environment value was not absent");
        } else if (mode == "invalid") {
            invalid([] { (void) strata::paths::native_path(std::string("x\0y", 3)); });
#if defined(_WIN32)
            for (const char* value : {"\xff", "\xc0\xaf", "\xed\xa0\x80", "\xe6\x97"})
                invalid([&] { (void) strata::paths::native_path(value); });
            invalid([] { (void) strata::paths::wide_to_utf8(std::wstring(1, wchar_t(0xd800))); });
            invalid([] { (void) strata::paths::wide_to_utf8(std::wstring(L"x\0y", 3)); });
            need(strata::paths::utf8_to_wide("").empty() && strata::paths::wide_to_utf8(L"").empty(), "empty conversion failed");
#endif
            need(strata::paths::fopen(nullptr, "rb") == nullptr && strata::paths::fopen("", "rb") == nullptr,
                 "invalid fopen input did not return ordinary failure");
        }
#endif
        else throw std::invalid_argument("unknown operation " + mode);
        std::puts("OK");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
