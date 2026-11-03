// Immutable temporary conversation storage; PolyStrata/Strata contributors (MIT).
// Adapted from PolyStrata 6f279e9f4b82ce9c20a27df02b72601e690ce4ea.
#include "strata/common/conversation_spill.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <limits>
#include <new>
#include <string>
#include <cwchar>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace strata::conversation {
Budget::Budget(size_t limit, size_t disk_limit, size_t metadata)
    : limit_(limit), disk_limit_(std::min(disk_limit, limit)) {
    stats_.reserved = std::min(metadata, limit_);
    stats_.peak = stats_.reserved;
}
bool Budget::acquire(size_t bytes, bool disk) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t used = stats_.heap + stats_.disk + stats_.reserved;
    if (bytes > limit_ - used || (disk && bytes > disk_limit_ - stats_.disk)) return false;
    (disk ? stats_.disk : stats_.heap) += bytes;
    stats_.peak = std::max(stats_.peak, used + bytes);
    return true;
}
void Budget::release(size_t bytes, bool disk) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& value = disk ? stats_.disk : stats_.heap;
    assert(value >= bytes); value -= bytes;
}
MemoryStats Budget::stats() const { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
size_t Budget::available() const {
    const auto s = stats(); return limit_ - s.heap - s.disk - s.reserved;
}
void Budget::spill_failed() { std::lock_guard<std::mutex> lock(mutex_); ++stats_.spill_failures; }

namespace {
intptr_t temporary_file(size_t bytes) {
    constexpr uint64_t floor = uint64_t(64) << 20;
#ifdef _WIN32
    wchar_t dir[MAX_PATH + 1], name[MAX_PATH + 80];
    const DWORD length = GetTempPathW(MAX_PATH, dir);
    if (!length || length >= MAX_PATH) return -1;
    ULARGE_INTEGER room{};
    if (!GetDiskFreeSpaceExW(dir, &room, nullptr, nullptr) || bytes > room.QuadPart || room.QuadPart - bytes < floor)
        return -1;
    static std::atomic<unsigned> count{0};
    for (int attempt = 0; attempt < 16; ++attempt) {
        swprintf(name, sizeof name / sizeof *name, L"%lstinystrata-chat-%lu-%u.tmp", dir, GetCurrentProcessId(), count++);
        const HANDLE file = CreateFileW(name, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
            FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (file != INVALID_HANDLE_VALUE) return reinterpret_cast<intptr_t>(file);
        if (GetLastError() != ERROR_FILE_EXISTS) return -1;
    }
    return -1;
#else
    const char* dir = std::getenv("TMPDIR"); if (!dir || !*dir) dir = "/tmp";
    struct statvfs space{};
    if (statvfs(dir, &space) != 0) return -1;
    const uint64_t room = uint64_t(space.f_bavail) * space.f_frsize;
    if (bytes > room || room - bytes < floor) return -1;
    std::string path = std::string(dir) + "/tinystrata-chat-XXXXXX";
    std::vector<char> name(path.begin(), path.end()); name.push_back(0);
    const int file = mkstemp(name.data());
    if (file >= 0) { unlink(name.data()); fcntl(file, F_SETFD, FD_CLOEXEC); }
    return file;
#endif
}
void close_file(intptr_t file) {
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(file));
#else
    close(static_cast<int>(file));
#endif
}
bool write_all(intptr_t file, const uint8_t* data, size_t bytes) {
    for (size_t at = 0; at < bytes;) {
#ifdef _WIN32
        DWORD written = 0;
        if (!WriteFile(reinterpret_cast<HANDLE>(file), data + at, static_cast<DWORD>(std::min(bytes - at, size_t(1) << 30)), &written, nullptr) || !written) return false;
#else
        const ssize_t written = pwrite(static_cast<int>(file), data + at, bytes - at, static_cast<off_t>(at));
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
#endif
        at += static_cast<size_t>(written);
    }
    return true;
}
} // namespace
Blob::Blob(std::shared_ptr<Budget> budget, size_t bytes) : budget_(std::move(budget)), bytes_(bytes) {}
std::shared_ptr<Blob> Blob::make(const std::shared_ptr<Budget>& budget, size_t bytes) {
    if (!bytes || !budget->acquire(bytes, false)) return nullptr;
    Blob* raw = nullptr;
    try { raw = new Blob(budget, bytes); }
    catch (const std::bad_alloc&) { budget->release(bytes, false); return nullptr; }
    try {
        // shared_ptr deletes raw itself if its control-block allocation fails.
        auto out = std::shared_ptr<Blob>(raw);
        out->heap_.reset(new uint8_t[bytes]);
        return out;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
Blob::~Blob() {
    unmap();
    if (file_ != -1) { close_file(file_); budget_->release(bytes_, true); }
    else budget_->release(bytes_, false);
}
uint8_t* Blob::data() const {
    if (heap_) return heap_.get();
    if (view_ || file_ == -1) return view_;
#ifdef _WIN32
    const HANDLE map = CreateFileMappingW(reinterpret_cast<HANDLE>(file_), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map) return nullptr;
    view_ = static_cast<uint8_t*>(MapViewOfFile(map, FILE_MAP_READ, 0, 0, bytes_));
    if (!view_) CloseHandle(map); else mapping_ = map;
#else
    void* map = mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, static_cast<int>(file_), 0);
    if (map != MAP_FAILED) view_ = static_cast<uint8_t*>(map);
#endif
    return view_;
}
void Blob::unmap() const {
    if (!view_) return;
#ifdef _WIN32
    UnmapViewOfFile(view_); CloseHandle(static_cast<HANDLE>(mapping_)); mapping_ = nullptr;
#else
    munmap(view_, bytes_);
#endif
    view_ = nullptr;
}
bool Blob::spill() {
    unmap();
    if (file_ != -1) return true;
    if (!heap_ || !budget_->acquire(bytes_, true)) return false;
    intptr_t file = -1;
    try { file = temporary_file(bytes_); }
    catch (...) {
        // POSIX path/vector construction may allocate after the disk overlap
        // reservation. Preserve the original heap blob on any such failure.
        budget_->release(bytes_, true); budget_->spill_failed(); return false;
    }
    if (file == -1 || !write_all(file, heap_.get(), bytes_)) {
        if (file != -1) close_file(file);
        budget_->release(bytes_, true); budget_->spill_failed(); return false;
    }
    file_ = file; heap_.reset(); budget_->release(bytes_, false);
    return true;
}
} // namespace strata::conversation
