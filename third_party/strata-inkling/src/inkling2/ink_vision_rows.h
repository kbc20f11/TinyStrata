// Independent CPU Vision rows, with a portable alternative to OpenMP.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace ink2 {

class VisionRows {
public:
    // Zero preserves the build's existing OpenMP or serial path.
    explicit VisionRows(int threads) : use_openmp_(threads == 0) {
        for (int worker = 1; worker < threads; ++worker) {
            try {
                workers_.emplace_back([this, worker](std::stop_token stop) { wait_for_rows(stop, worker); });
            } catch (const std::system_error&) {
                break;
            }
        }
    }

    int thread_count() const noexcept { return static_cast<int>(workers_.size()) + 1; }

    template<class Fn>
    void run(int rows, Fn fn) {
        if (use_openmp_) {
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < rows; ++row) fn(row);
            return;
        }

        const int threads = thread_count();
        if (threads == 1 || rows < threads) {
            for (int row = 0; row < rows; ++row) fn(row);
            return;
        }

        {
            std::lock_guard lock(mutex_);
            run_range_ = [&fn, rows, threads](int worker) {
                const int first = static_cast<int>(static_cast<int64_t>(rows) * worker / threads);
                const int end = static_cast<int>(static_cast<int64_t>(rows) * (worker + 1) / threads);
                for (int row = first; row < end; ++row) fn(row);
            };
            error_ = nullptr;
            pending_ = threads - 1;
            ++generation_;
        }
        work_.notify_all();
        run_range(0);

        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return pending_ == 0; });
        run_range_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    void run_range(int worker) noexcept {
        try {
            run_range_(worker);
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }

    void wait_for_rows(std::stop_token stop, int worker) {
        unsigned seen = 0;
        std::unique_lock lock(mutex_);
        while (work_.wait(lock, stop, [this, &seen] { return seen != generation_; })) {
            seen = generation_;
            lock.unlock();
            run_range(worker);
            lock.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }

    bool use_openmp_;
    std::mutex mutex_;
    std::condition_variable_any work_;
    std::condition_variable done_;
    std::function<void(int)> run_range_;
    std::exception_ptr error_;
    unsigned generation_ = 0;
    int pending_ = 0;
    // Destroy and join the workers before destroying the state they use.
    std::vector<std::jthread> workers_;
};

}  // namespace ink2
