// Persistent launch worker for Qwen3.6 CPU/GPU graphs.
// Adapted from Strata's HostGraphLaunch (MIT), which keeps an instrumented,
// potentially blocking cudaGraphLaunch outside the CPU doorbell service thread.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
#pragma once
#include <cuda_runtime.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace strata::qwen35::hybrid_graph {
class Launcher {
public:
    explicit Launcher(int device) : worker_([this, device] { run(device); }) {}
    ~Launcher() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_all();
        worker_.join();
    }
    Launcher(const Launcher&) = delete;
    Launcher& operator=(const Launcher&) = delete;
    void start(cudaGraphExec_t graph, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        graph_ = graph;
        stream_ = stream;
        error_.store(cudaSuccess, std::memory_order_relaxed);
        pending_.store(true, std::memory_order_release);
        submitted_ = true;
        cv_.notify_all();
    }
    bool pending() const { return pending_.load(std::memory_order_acquire); }
    cudaError_t status() const { return error_.load(std::memory_order_acquire); }
    cudaError_t finish() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !pending(); });
        return status();
    }
private:
    void run(int device) {
        const cudaError_t device_error = cudaSetDevice(device);
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return submitted_ || stop_; });
            if (stop_) return;
            const auto graph = graph_;
            const auto stream = stream_;
            submitted_ = false;
            lock.unlock();
            cudaError_t error = device_error == cudaSuccess ? cudaGraphLaunch(graph, stream) : device_error;
            // Flush WDDM submission once before the CPU starts polling mapped
            // notifications. No CUDA call is needed for ordinary layer handoffs.
            if (error == cudaSuccess) {
                const auto q = cudaStreamQuery(stream);
                if (q != cudaSuccess && q != cudaErrorNotReady) error = q;
            }
            lock.lock();
            error_.store(error, std::memory_order_release);
            pending_.store(false, std::memory_order_release);
            cv_.notify_all();
        }
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> pending_{false};
    std::atomic<cudaError_t> error_{cudaSuccess};
    bool submitted_ = false, stop_ = false;
    cudaGraphExec_t graph_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::thread worker_;
};
} // namespace strata::qwen35::hybrid_graph
