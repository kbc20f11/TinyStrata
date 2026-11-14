#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdexcept>
#include <string>

namespace strata::intel {
// Modern Standby has separate system and execution requests. Hold both during
// loading/inference; do not request DisplayRequired, so the panel may turn off.
// Clear them when the scenario ends, including cancellation or an exception.
class ExecutionRequest {
    HANDLE handle_ = nullptr;
public:
    ExecutionRequest() {
        REASON_CONTEXT reason{};
        reason.Version = POWER_REQUEST_CONTEXT_VERSION;
        reason.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING;
        reason.Reason.SimpleReasonString = const_cast<wchar_t*>(L"TinyStrata Intel model loading or inference");
        handle_ = PowerCreateRequest(&reason);
        if (!handle_ || handle_ == INVALID_HANDLE_VALUE) {
            handle_ = nullptr;
            throw std::runtime_error("Cannot create Intel execution power request: " + std::to_string(GetLastError()));
        }
        if (!PowerSetRequest(handle_, PowerRequestSystemRequired)) {
            const DWORD error = GetLastError();
            CloseHandle(handle_); handle_ = nullptr;
            throw std::runtime_error("Cannot prevent idle sleep during Intel inference: " + std::to_string(error));
        }
        if (!PowerSetRequest(handle_, PowerRequestExecutionRequired)) {
            const DWORD error = GetLastError();
            PowerClearRequest(handle_, PowerRequestSystemRequired);
            CloseHandle(handle_); handle_ = nullptr;
            throw std::runtime_error("Cannot retain Intel process execution: " + std::to_string(error));
        }
    }
    ExecutionRequest(const ExecutionRequest&) = delete;
    ~ExecutionRequest() {
        if (handle_) {
            PowerClearRequest(handle_, PowerRequestExecutionRequired);
            PowerClearRequest(handle_, PowerRequestSystemRequired);
            CloseHandle(handle_);
        }
    }
};
} // namespace strata::intel
