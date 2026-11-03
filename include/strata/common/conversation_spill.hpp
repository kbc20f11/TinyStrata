// Bounded immutable conversation blobs, adapting PolyStrata Spill (MIT),
// 6f279e9f4b82ce9c20a27df02b72601e690ce4ea. See NOTICE.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace strata::conversation {
struct MemoryStats { size_t heap = 0, disk = 0, reserved = 0, peak = 0, spill_failures = 0; };
class Budget {
public:
    Budget(size_t limit, size_t disk_limit, size_t metadata);
    bool acquire(size_t bytes, bool disk);
    void release(size_t bytes, bool disk);
    MemoryStats stats() const;
    size_t limit() const { return limit_; }
    size_t available() const;
    void spill_failed();
private:
    size_t limit_, disk_limit_;
    mutable std::mutex mutex_;
    MemoryStats stats_;
};

class Blob {
public:
    static std::shared_ptr<Blob> make(const std::shared_ptr<Budget>& budget, size_t bytes);
    ~Blob();
    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;
    uint8_t* data() const;
    size_t bytes() const { return bytes_; }
    // Synchronous between requests: reserve the temporary heap+file overlap,
    // then publish only a complete private file. Failure leaves heap intact.
    bool spill();
    void unmap() const;
private:
    Blob(std::shared_ptr<Budget> budget, size_t bytes);
    std::shared_ptr<Budget> budget_;
    size_t bytes_ = 0;
    std::unique_ptr<uint8_t[]> heap_;
    intptr_t file_ = -1;
    mutable uint8_t* view_ = nullptr;
    mutable void* mapping_ = nullptr;
};
} // namespace strata::conversation
