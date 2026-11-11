// Native interfaces exchange UTF-8. Windows file APIs receive UTF-16 explicitly,
// independently of the machine's ANSI code page; POSIX filenames retain bytes.
#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::paths {

inline void check_no_nul(std::string_view value) {
    if (value.find('\0') != std::string_view::npos)
        throw std::invalid_argument("a UTF-8 path cannot contain NUL");
}

#if defined(_WIN32)
inline std::wstring utf8_to_wide(std::string_view value) {
    check_no_nul(value);
    if (value.empty()) return {};
    if (value.size() > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("UTF-8 path is too long");
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), int(value.size()), nullptr, 0);
    if (!count) throw std::invalid_argument("path is not valid UTF-8 (Windows error " + std::to_string(GetLastError()) + ")");
    std::wstring result(size_t(count), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), int(value.size()), result.data(), count) != count)
        throw std::invalid_argument("UTF-8 path conversion failed (Windows error " + std::to_string(GetLastError()) + ")");
    return result;
}

inline std::string wide_to_utf8(std::wstring_view value) {
    if (value.find(L'\0') != std::wstring_view::npos)
        throw std::invalid_argument("a native path cannot contain NUL");
    if (value.empty()) return {};
    if (value.size() > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native path is too long");
    const int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), int(value.size()), nullptr, 0, nullptr, nullptr);
    if (!count) throw std::invalid_argument("path is not valid UTF-16 (Windows error " + std::to_string(GetLastError()) + ")");
    std::string result(size_t(count), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), int(value.size()), result.data(), count, nullptr, nullptr) != count)
        throw std::invalid_argument("native path conversion failed (Windows error " + std::to_string(GetLastError()) + ")");
    return result;
}
#endif

inline std::filesystem::path native_path(std::string_view value) {
    check_no_nul(value);
#if defined(_WIN32)
    return std::filesystem::path(utf8_to_wide(value));
#else
    return std::filesystem::path(value);
#endif
}

inline std::string utf8_path(const std::filesystem::path& value) {
#if defined(_WIN32)
    return wide_to_utf8(value.native());
#else
    return value.native();
#endif
}

inline std::FILE* fopen(const char* path, const char* mode) {
    // Avoid the Windows CRT invalid-parameter handler for empty paths, retaining
    // fopen's ordinary null/error contract for a missing or empty filename.
    if (!path || !mode || !*path || !*mode) { errno = EINVAL; return nullptr; }
#if defined(_WIN32)
    return _wfopen(utf8_to_wide(path).c_str(), utf8_to_wide(mode).c_str());
#else
    return std::fopen(path, mode);
#endif
}

// getenv's narrow Windows result uses the ANSI code page too. Preserve an
// unset variable separately from one explicitly set to an empty value.
inline std::optional<std::string> getenv(const char* name) {
    if (!name || !*name || std::string_view(name).find('=') != std::string_view::npos)
        throw std::invalid_argument("invalid environment variable name");
#if defined(_WIN32)
    const std::wstring key = utf8_to_wide(name);
    DWORD size = 0;
    for (;;) {
        SetLastError(ERROR_SUCCESS);
        const DWORD needed = GetEnvironmentVariableW(key.c_str(), nullptr, 0);
        if (!needed) {
            const DWORD error = GetLastError();
            if (error == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
            if (error == ERROR_SUCCESS) return std::string();
            throw std::runtime_error("cannot read environment variable (Windows error " + std::to_string(error) + ")");
        }
        size = needed;
        std::wstring value(size, L'\0');
        SetLastError(ERROR_SUCCESS);
        const DWORD got = GetEnvironmentVariableW(key.c_str(), value.data(), size);
        if (!got) {
            const DWORD error = GetLastError();
            if (error == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
            if (error == ERROR_SUCCESS) return std::string();
            throw std::runtime_error("cannot read environment variable (Windows error " + std::to_string(error) + ")");
        }
        if (got >= size) continue; // changed between the size and value queries
        value.resize(got);
        return wide_to_utf8(value);
    }
#else
    const char* value = std::getenv(name);
    return value ? std::optional<std::string>(value) : std::nullopt;
#endif
}

} // namespace strata::paths
