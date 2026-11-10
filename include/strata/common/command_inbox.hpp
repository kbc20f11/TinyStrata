// Owned stdin reader for the native line protocol. No model or GPU dependencies.
// The stop/join lifetime follows Basalt 4039751f (MIT); pipe/console handling and
// the queued cancellation epochs preserve TinyStrata's serving protocol.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <cerrno>
#include <poll.h>
#include <unistd.h>
#endif

namespace strata::common {

class CommandInbox {
public:
    explicit CommandInbox(int fd = 0) : fd_(fd), reader_([this] { read_loop(); }) {
#if defined(_WIN32)
        // Use a real Windows thread handle rather than assuming the C++
        // library's native_handle_type is HANDLE (e.g. MinGW pthread builds).
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return initialized_ || reader_done_.load(); });
        if (error_) {
            const auto error = error_;
            lock.unlock();
            stop();
            std::rethrow_exception(error);
        }
#endif
    }

    CommandInbox(const CommandInbox&) = delete;
    CommandInbox& operator=(const CommandInbox&) = delete;
    ~CommandInbox() { stop(); }

    // STOP after a queued GEN must remain visible when that GEN is taken;
    // a GEN queued after STOP belongs to the newer epoch and starts normally.
    bool next(std::string& text) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return shutdown_.load() || eof_ || !lines_.empty(); });
        if (shutdown_.load()) return false;
        if (lines_.empty()) {
            if (error_) std::rethrow_exception(error_);
            return false;
        }
        auto command = std::move(lines_.front());
        lines_.pop_front();
        text = std::move(command.text);
        cancelled_.store(command.epoch != cancel_epoch_);
        return true;
    }

    std::atomic<bool>& cancelled() { return cancelled_; }

#if defined(STRATA_COMMAND_INBOX_TEST_HOOKS)
    static void fail_next_initialization_for_test() { fail_initialization_.store(true); }
#endif

    bool input_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return eof_;
    }

    // Does not close/change the caller's stdin descriptor. Pipes are polled
    // without blocking. A Windows cooked console read may block after becoming
    // ready; cancellation is retried until done to cover the check/read race.
    void stop() noexcept {
        shutdown_.store(true);
        changed_.notify_all();
        if (!reader_.joinable()) return;
#if defined(_WIN32)
        while (!reader_done_.load()) {
            if (thread_handle_) (void) CancelSynchronousIo(thread_handle_);
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait_for(lock, std::chrono::milliseconds(10), [this] { return reader_done_.load(); });
        }
#endif
        reader_.join();
#if defined(_WIN32)
        if (thread_handle_) { CloseHandle(thread_handle_); thread_handle_ = nullptr; }
#endif
    }

private:
    struct Command { std::string text; uint64_t epoch; };
    int fd_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Command> lines_;
    uint64_t cancel_epoch_ = 0;
    bool eof_ = false;
    std::exception_ptr error_;
    std::atomic<bool> shutdown_{false}, cancelled_{false}, reader_done_{false};
#if defined(STRATA_COMMAND_INBOX_TEST_HOOKS)
    static inline std::atomic<bool> fail_initialization_{false};
#endif
#if defined(_WIN32)
    HANDLE thread_handle_ = nullptr;
    bool initialized_ = false;
#endif
    // Last member: all state is initialized before the worker starts and the
    // destructor joins it before any of that state is destroyed.
    std::thread reader_;

    bool publish(std::string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_.load()) return false;
        if (line == "STOP") {
            ++cancel_epoch_;
            cancelled_.store(true);
            return true;
        }
        const bool keep_reading = line != "QUIT";
        lines_.push_back({std::move(line), cancel_epoch_});
        changed_.notify_all();
        return keep_reading;
    }

    // Positive bytes, zero for EOF, -1 for an explicit owner stop.
    int read_some(char* bytes, int capacity) {
#if defined(_WIN32)
        const auto raw = _get_osfhandle(fd_);
        if (raw == -1 || raw == -2) return 0;
        const HANDLE input = reinterpret_cast<HANDLE>(raw);
        const bool pipe = GetFileType(input) == FILE_TYPE_PIPE;
        while (!shutdown_.load()) {
            DWORD want = static_cast<DWORD>(capacity);
            if (pipe) {
                DWORD available = 0;
                if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr)) {
                    const DWORD code = GetLastError();
                    if (code == ERROR_BROKEN_PIPE || code == ERROR_NO_DATA || code == ERROR_HANDLE_EOF) return 0;
                    throw std::runtime_error("native stdin pipe wait failed (" + std::to_string(code) + ")");
                }
                if (!available) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    changed_.wait_for(lock, std::chrono::milliseconds(10), [this] { return shutdown_.load(); });
                    continue;
                }
                want = std::min(want, available);
            }
            if (shutdown_.load()) return -1;
            DWORD got = 0;
            const BOOL ok = ReadFile(input, bytes, want, &got, nullptr);
            if (ok || (GetLastError() == ERROR_MORE_DATA && got)) return static_cast<int>(got);
            const DWORD code = GetLastError();
            if (shutdown_.load() || code == ERROR_OPERATION_ABORTED) return -1;
            if (code == ERROR_BROKEN_PIPE || code == ERROR_HANDLE_EOF || code == ERROR_NO_DATA) return 0;
            throw std::runtime_error("native stdin read failed (" + std::to_string(code) + ")");
        }
        return -1;
#else
        while (!shutdown_.load()) {
            pollfd descriptor{fd_, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 20);
            if (ready < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("native stdin poll failed (" + std::to_string(errno) + ")");
            }
            if (!ready) continue;
            if (shutdown_.load()) return -1;
            const auto got = ::read(fd_, bytes, static_cast<size_t>(capacity));
            if (got >= 0) return static_cast<int>(got);
            if (errno == EINTR || errno == EAGAIN) continue;
            throw std::runtime_error("native stdin read failed (" + std::to_string(errno) + ")");
        }
        return -1;
#endif
    }

    void read_loop() noexcept {
        try {
#if defined(STRATA_COMMAND_INBOX_TEST_HOOKS)
            if (fail_initialization_.exchange(false)) throw std::runtime_error("injected reader initialization failure");
#endif
#if defined(_WIN32)
            HANDLE handle = nullptr;
            const BOOL ok = DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                                             &handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!ok) error_ = std::make_exception_ptr(std::runtime_error("cannot own native stdin reader handle"));
                thread_handle_ = handle;
                initialized_ = true;
                changed_.notify_all();
            }
            if (!ok) throw std::runtime_error("cannot own native stdin reader handle");
#endif
            std::string line;
            char chunk[4096];
            bool keep_reading = true;
            while (keep_reading && !shutdown_.load()) {
                const int count = read_some(chunk, sizeof chunk);
                if (count <= 0) {
                    if (count == 0 && !line.empty() && !shutdown_.load()) publish(std::move(line));
                    break;
                }
                const char* at = chunk;
                const char* end = chunk + count;
                while (at < end) {
                    const char* nl = static_cast<const char*>(std::memchr(at, '\n', static_cast<size_t>(end - at)));
                    const char* until = nl ? nl : end;
                    line.append(at, static_cast<size_t>(until - at));
                    if (!nl) break;
                    keep_reading = publish(std::move(line));
                    line.clear();
                    if (!keep_reading) break;
                    at = nl + 1;
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            eof_ = true;
            reader_done_.store(true);
            changed_.notify_all();
        }
    }
};

} // namespace strata::common
