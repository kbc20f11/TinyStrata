#pragma once

#include "strata/common/utf8_path.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <shellapi.h>
#endif

namespace strata::paths {

// Windows' narrow main() arguments use the active ANSI code page. Recover the
// original Unicode command line before giving paths to UTF-8 native libraries.
// Storage outlives the rebound argc/argv; this does not change process locale,
// console encoding, environment variables, or POSIX argument bytes.
class Utf8Arguments {
public:
    Utf8Arguments(int argc, char** argv) : argc_(argc), argv_(argv) {
#if defined(_WIN32)
        int count = 0;
        auto* wide = CommandLineToArgvW(GetCommandLineW(), &count);
        if (!wide) throw std::runtime_error("cannot read Unicode command line");
        struct Release {
            wchar_t** values;
            ~Release() { LocalFree(values); }
        } release{wide};
        strings_.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) strings_.push_back(wide_to_utf8(wide[i]));
        pointers_.reserve(strings_.size() + 1);
        for (auto& value : strings_) pointers_.push_back(value.data());
        pointers_.push_back(nullptr);
        argc_ = count;
        argv_ = pointers_.data();
#endif
    }
    Utf8Arguments(const Utf8Arguments&) = delete;
    Utf8Arguments& operator=(const Utf8Arguments&) = delete;
    int argc() const { return argc_; }
    char** argv() const { return argv_; }
private:
    int argc_;
    char** argv_;
    std::vector<std::string> strings_;
    std::vector<char*> pointers_;
};

} // namespace strata::paths
