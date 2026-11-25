// Inkling 2 Mini Vision on CUDA. The ViT graph follows the native CPU and HIP
// encoders (MIT, Copyright (c) 2026 Strata contributors).
#pragma once

#include "ink_cuda.cuh"
#include "ink_cuda_vision_ops.cuh"
#include "ink_vision.h"

#include <array>
#include <initializer_list>
#include <memory>

namespace ink2 {

static __global__ void store_vision_attention(const float* tile, float* rows, int count, int heads, int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int head_width = count * head_dim;
    if (i >= heads * head_width) return;
    const int head = i / head_width;
    const int row = (i % head_width) / head_dim;
    rows[(row * heads + head) * head_dim + i % head_dim] = tile[i];
}

class CudaVision {
public:
    ~CudaVision() {
        if (device_ >= 0) (void)cudaSetDevice(device_);
        if (blas_) (void)cublasDestroy(blas_);
        if (stream_) (void)cudaStreamDestroy(stream_);
    }

    bool load(const std::string& path, int device, std::string& err) {
        device_ = device;
        const char* reuse_workspace = std::getenv("STRATA_INK_VISION_WORKSPACE_REUSE");
        reuse_attention_workspace_ = !reuse_workspace || std::strcmp(reuse_workspace, "0") != 0;
        VisionShardInfo shard;
        if (!vision_probe(path, shard, err)) return false;
        cfg_ = shard.c;
        if (cfg_.hidden != 1024 || cfg_.heads != 16 || cfg_.head_dim != 64 ||
            cfg_.inter != 4096 || cfg_.patch != 20 || cfg_.merge != 2 ||
            cfg_.layers != 23 || cfg_.posemb_grid != 32 ||
            cfg_.proj_hidden != 4096 || cfg_.out_hidden != 2304) {
            err = "unsupported CUDA Vision tensor geometry";
            return false;
        }
        if (!check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream", err) ||
            !check(cublasCreate(&blas_), "create cuBLAS handle", err) ||
            !check(cublasSetStream(blas_, stream_), "set cuBLAS stream", err) ||
            !check(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH), "set cuBLAS FP32 mode", err) ||
            !allocate(blas_work_, kBlasWorkspaceBytes, err) ||
            !check(cublasSetWorkspace(blas_, blas_work_.get(), kBlasWorkspaceBytes), "set cuBLAS workspace", err)) {
            return false;
        }

        strata::artifact::StrataFileReader file;
        if (!shard.is_safetensors && !file.open(path, err)) return false;
        std::vector<uint8_t> slice;
        auto tensor = [&](const std::string& name, std::initializer_list<int64_t> shape,
                          Buffer* device_data, int* linear_format, std::vector<float>* host_data) {
            const VitTensor* t = shard.find(name);
            if (!t || t->shape.size() != shape.size() ||
                !std::equal(shape.begin(), shape.end(), t->shape.begin()) ||
                (t->fmt != F_BF16 && t->fmt != F_F16 && t->fmt != F_F32)) {
                err = "missing or incompatible Vision tensor " + name;
                return false;
            }
            size_t count = 1;
            for (int64_t extent : shape) count *= static_cast<size_t>(extent);
            if (t->bytes != count * (t->fmt == F_F32 ? 4 : 2)) {
                err = "unexpected Vision tensor size for " + name;
                return false;
            }
            slice.resize(static_cast<size_t>(t->bytes));
            if (shard.is_safetensors) {
                if (!detail::read_slice_from_file(t->shard_path, t->off, slice.data(), slice.size(), err)) return false;
            } else {
                if (t->off > shard.sec1_len || t->bytes > shard.sec1_len - t->off) {
                    err = "Vision tensor extends beyond its data section: " + name;
                    return false;
                }
                if (!file.read_at(shard.sec1_off + t->off, slice.data(), slice.size(), err)) return false;
            }
            const auto* bytes = slice.data();
            if (linear_format) {
                *linear_format = t->fmt;
                if (t->fmt != F_F32) max_expand_ = std::max(max_expand_, count);
                weight_bytes_ += static_cast<size_t>(t->bytes);
                return upload(*device_data, bytes, static_cast<size_t>(t->bytes), err);
            }
            std::vector<float> unpacked;
            auto& values = host_data ? *host_data : unpacked;
            values.resize(count);
            if (t->fmt == F_F32) {
                std::memcpy(values.data(), bytes, count * sizeof(float));
            } else {
                for (size_t i = 0; i < count; ++i) {
                    uint16_t bits;
                    std::memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
                    values[i] = t->fmt == F_BF16 ? bf16_to_f32(bits) : fp16_to_f32(bits);
                }
            }
            if (host_data) return true;
            weight_bytes_ += count * sizeof(float);
            return upload(*device_data, values.data(), count * sizeof(float), err);
        };
        auto vector = [&](const std::string& name, Buffer& target, int dim) {
            return tensor(name, {dim}, &target, nullptr, nullptr);
        };
        auto norm = [&](const std::string& name, Norm& n) {
            return vector(name + ".weight", n.weight, cfg_.hidden) &&
                   vector(name + ".bias", n.bias, cfg_.hidden);
        };
        auto linear = [&](const std::string& name, Linear& l, int input, int output, bool bias) {
            l.input = input;
            l.output = output;
            return tensor(name + ".weight", {output, input}, &l.weight, &l.format, nullptr) &&
                   (!bias || vector(name + ".bias", l.bias, output));
        };

        const int d = cfg_.hidden, grid = cfg_.posemb_grid;
        const std::string embeddings = "model.visual.backbone.embeddings.";
        const auto* positions = shard.find(embeddings + "position_embedding");
        if (!positions) { err = "missing Vision position embedding"; return false; }
        const bool positions_ok = positions->shape.size() == 3
            ? tensor(embeddings + "position_embedding", {grid, grid, d}, nullptr, nullptr, &position_)
            : tensor(embeddings + "position_embedding", {grid * grid, d}, nullptr, nullptr, &position_);
        if (!positions_ok ||
            !tensor(embeddings + "class_embedding", {d}, nullptr, nullptr, &cls_) ||
            !tensor(embeddings + "cls_position_embedding", {d}, nullptr, nullptr, &cls_position_) ||
            !linear(embeddings + "patch_embedding", patch_, 3 * cfg_.patch * cfg_.patch, d, false) ||
            !norm("model.visual.backbone.ln_pre", pre_norm_)) return false;
        for (int i = 0; i < cfg_.layers; ++i) {
            Layer& layer = layers_[i];
            const std::string name = "model.visual.backbone.layers." + std::to_string(i);
            if (!norm(name + ".ln_1", layer.ln1) || !norm(name + ".ln_2", layer.ln2) ||
                !linear(name + ".self_attn.q_proj", layer.q, d, d, true) ||
                !linear(name + ".self_attn.k_proj", layer.k, d, d, true) ||
                !linear(name + ".self_attn.v_proj", layer.v, d, d, true) ||
                !linear(name + ".self_attn.out_proj", layer.out, d, d, true) ||
                !linear(name + ".mlp.c_fc", layer.fc, d, cfg_.inter, true) ||
                !linear(name + ".mlp.c_proj", layer.project, cfg_.inter, d, true) ||
                !vector(name + ".ls_1.gamma", layer.scale1, d) ||
                !vector(name + ".ls_2.gamma", layer.scale2, d)) return false;
        }
        if (!linear("model.visual.proj_in", projector_in_, d * cfg_.merge * cfg_.merge, cfg_.proj_hidden, false) ||
            !linear("model.visual.proj_out", projector_out_, cfg_.proj_hidden, cfg_.out_hidden, false) ||
            (max_expand_ && !allocate(expanded_, max_expand_ * sizeof(float), err)) ||
            !allocate(work_, workspace(kResidentImageTokens).size * sizeof(float), err)) return false;

        // Initialize cuBLAS and touch the workspace before KV/Expert placement.
        Work first = workspace(1, ptr(work_));
        if (!check(cudaMemsetAsync(work_.get(), 0, workspace(kResidentImageTokens).size * sizeof(float), stream_),
                   "clear Vision workspace", err) ||
            !multiply(patch_, first.patch, first.current + cfg_.hidden, 1, err) ||
            !check(cudaStreamSynchronize(stream_), "warm up GPU Vision", err)) return false;
        return true;
    }

    size_t weight_bytes() const { return weight_bytes_; }
    size_t workspace_bytes() const {
        return (workspace(kResidentImageTokens).size + max_expand_) * sizeof(float) + kBlasWorkspaceBytes;
    }

    bool encode(const ImageRec& img, std::vector<float>& out, std::string& err) {
        using namespace cuda_vision_ops;
        if (!check(cudaSetDevice(device_), "select device", err)) return false;
        const int n = img.num_lm_tokens();
        Buffer extra;
        float* base = ptr(work_);
        if (n > kResidentImageTokens) {
            size_t free = 0, total = 0;
            const size_t bytes = workspace(n).size * sizeof(float);
            if (!check(strata::ink::ink_cuda_mem_info(&free, &total, device_), "query image workspace budget", err)) return false;
            if (bytes > free || free - bytes < (64ull << 20)) {
                err = "image workspace exceeds free device memory";
                return false;
            }
            if (!allocate(extra, bytes, err)) return false;
            base = ptr(extra);
        }
        const auto input = prepare_vision_input(img, cfg_, cls_.data(), cls_position_.data(), position_.data());
        Work w = workspace(n, base);
        const int d = cfg_.hidden, hd = cfg_.head_dim, h = cfg_.heads;
        const int s = n * cfg_.merge * cfg_.merge + 1;
        const size_t elements = static_cast<size_t>(s) * d;
        auto copy = [&](float* dest, const std::vector<float>& src) {
            return check(cudaMemcpyAsync(dest, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
                         "upload Vision input", err);
        };
        // All queued transfers complete before input/extra workspaces are released on any path.
        bool ok = copy(w.patch, input.patches) && copy(w.position, input.position) &&
                  copy(w.rope_cos, input.rope_cos) && copy(w.rope_sin, input.rope_sin) &&
                  check(cudaMemsetAsync(w.current, 0, d * sizeof(float), stream_), "clear CLS row", err) &&
                  multiply(patch_, w.patch, w.current + d, s - 1, err);
        if (ok) {
            add_position<<<blocks(elements), kThreads, 0, stream_>>>(w.current, w.position, elements);
            normalise(pre_norm_, w.current, w.current, s);
            for (const Layer& l : layers_) {
                normalise(l.ln1, w.current, w.norm, s);
                ok = multiply(l.q, w.norm, w.q, s, err) &&
                     multiply(l.k, w.norm, w.k, s, err) &&
                     multiply(l.v, w.norm, w.v, s, err);
                if (!ok) break;
                rope_2d<<<dim3(h, s), hd / 2, 0, stream_>>>(w.q, w.k, w.rope_cos, w.rope_sin, h, hd);
                ok = attend(w, s, err) && multiply(l.out, w.attended, w.projected, s, err);
                if (!ok) break;
                scaled_residual<<<blocks(elements), kThreads, 0, stream_>>>(w.current, w.projected, ptr(l.scale1), elements, d);
                normalise(l.ln2, w.current, w.norm, s);
                ok = multiply(l.fc, w.norm, w.mlp, s, err);
                if (!ok) break;
                const size_t middle = static_cast<size_t>(s) * cfg_.inter;
                gelu<<<blocks(middle), kThreads, 0, stream_>>>(w.mlp, middle);
                ok = multiply(l.project, w.mlp, w.projected, s, err);
                if (!ok) break;
                scaled_residual<<<blocks(elements), kThreads, 0, stream_>>>(w.current, w.projected, ptr(l.scale2), elements, d);
            }
        }
        if (ok && (ok = multiply(projector_in_, w.current + d, w.mlp, n, err))) {
            const size_t middle = static_cast<size_t>(n) * cfg_.proj_hidden;
            gelu<<<blocks(middle), kThreads, 0, stream_>>>(w.mlp, middle);
            ok = multiply(projector_out_, w.mlp, w.projected, n, err);
        }
        if (ok) {
            out.resize(static_cast<size_t>(n) * cfg_.out_hidden);
            ok = check(cudaGetLastError(), "launch CUDA Vision kernels", err) &&
                 check(cudaMemcpyAsync(out.data(), w.projected, out.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_), "download Vision embeddings", err);
        }
        const auto sync = cudaStreamSynchronize(stream_);
        return ok && check(sync, "finish CUDA Vision", err);
    }

private:
    struct FreeDevice {
        void operator()(void* p) const { (void)cudaFree(p); }
    };
    using Buffer = std::unique_ptr<void, FreeDevice>;

    struct Norm { Buffer weight, bias; };
    struct Linear {
        Buffer weight, bias;
        int format = F_F32;
        int input = 0, output = 0;
    };
    struct Layer {
        Norm ln1, ln2;
        Linear q, k, v, out, fc, project;
        Buffer scale1, scale2;
    };
    struct Work {
        float *patch, *position, *rope_cos, *rope_sin;
        float *current, *norm, *q, *k, *v, *attended, *projected, *mlp, *scores, *attended_tile;
        size_t size;
    };

    static constexpr int kThreads = 256;
    static constexpr int kQueryTile = 64;
    // Approximately 0.9 megapixels after padding to 40px merged cells.
    static constexpr int kResidentImageTokens = 576;
    static constexpr size_t kBlasWorkspaceBytes = 4ull << 20;

    static int blocks(size_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }
    static float* ptr(const Buffer& b) { return static_cast<float*>(b.get()); }

    static bool check(cudaError_t status, const char* what, std::string& err) {
        if (status == cudaSuccess) return true;
        err = std::string(what) + ": " + cudaGetErrorString(status);
        (void)cudaGetLastError();
        return false;
    }
    static bool check(cublasStatus_t status, const char* what, std::string& err) {
        if (status == CUBLAS_STATUS_SUCCESS) return true;
        err = std::string(what) + ": " + cublasGetStatusString(status);
        return false;
    }
    static bool allocate(Buffer& b, size_t bytes, std::string& err) {
        void* p = nullptr;
        if (!check(cudaMalloc(&p, bytes), "allocate CUDA Vision memory", err)) return false;
        b.reset(p);
        return true;
    }
    static bool upload(Buffer& b, const void* data, size_t bytes, std::string& err) {
        return allocate(b, bytes, err) &&
               check(cudaMemcpy(b.get(), data, bytes, cudaMemcpyHostToDevice), "upload Vision weights", err);
    }

    Work workspace(int tokens, float* base = nullptr) const {
        const size_t n = static_cast<size_t>(tokens);
        const size_t patches = n * cfg_.merge * cfg_.merge;
        const size_t s = patches + 1, d = cfg_.hidden;
        size_t offset = 0;
        auto take = [&](size_t count) {
            // 256-byte alignment for cuBLAS submatrices.
            offset = (offset + 63) & ~size_t(63);
            float* p = base ? base + offset : nullptr;
            offset += count;
            return p;
        };
        Work w{};
        w.rope_cos = take(s * cfg_.head_dim);
        w.rope_sin = take(s * cfg_.head_dim);
        w.current = take(s * d);
        w.norm = take(s * d);
        w.q = take(s * d);
        w.k = take(s * d);
        w.v = take(s * d);
        w.attended = take(s * d);
        w.projected = take(std::max(s * d, n * cfg_.out_hidden));
        // As in the existing Clover image workspace, share the memory of
        // phases that finish on the same stream before the next one starts.
        // Q, K, V, and the attention result are dead before the MLP begins.
        w.mlp = reuse_attention_workspace_ ? w.q : take(std::max(s * cfg_.inter, n * cfg_.proj_hidden));
        // Patch pixels and absolute position embeddings are consumed before
        // the first transformer layer uses the MLP and normalisation buffers.
        w.patch = w.mlp;
        w.position = w.norm;
        w.scores = take(s * std::min(s, static_cast<size_t>(kQueryTile)) * cfg_.heads);
        w.attended_tile = take(std::min(s, static_cast<size_t>(kQueryTile)) * d);
        w.size = offset;
        return w;
    }

    void normalise(const Norm& n, const float* input, float* out, int rows) {
        cuda_vision_ops::layer_norm<<<rows, kThreads, 0, stream_>>>(
            input, ptr(n.weight), ptr(n.bias), out, cfg_.hidden, cfg_.ln_eps);
    }

    bool multiply(const Linear& l, const float* input, float* out, int rows, std::string& err) {
        using namespace cuda_vision_ops;
        const float* weight = ptr(l.weight);
        if (l.format != F_F32) {
            // Widen the stored values exactly. Keep FP32 activations and avoid
            // TF32/BF16 input rounding in the projection or backbone.
            const size_t count = static_cast<size_t>(l.input) * l.output;
            const auto* bits = static_cast<const uint16_t*>(l.weight.get());
            if (l.format == F_BF16) unpack_bf16<<<blocks(count), kThreads, 0, stream_>>>(bits, ptr(expanded_), count);
            else unpack_f16<<<blocks(count), kThreads, 0, stream_>>>(bits, ptr(expanded_), count);
            weight = ptr(expanded_);
        }
        const float one = 1.0f, zero = 0.0f;
        if (!check(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, l.output, rows, l.input, &one,
                               weight, l.input, input, l.input, &zero, out, l.output), "Vision linear", err)) {
            return false;
        }
        if (l.bias) {
            const size_t count = static_cast<size_t>(rows) * l.output;
            add_bias<<<blocks(count), kThreads, 0, stream_>>>(out, ptr(l.bias), count, l.output);
        }
        return true;
    }

    bool attend(const Work& w, int rows, std::string& err) {
        const int hd = cfg_.head_dim, d = cfg_.hidden, heads = cfg_.heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd)), one = 1.0f, zero = 0.0f;
        // Head-contiguous columns are interleaved per token. Score tiles keep
        // every key for a stable, noncausal softmax without a full S x S buffer.
        for (int start = 0; start < rows; start += kQueryTile) {
            const int count = std::min(kQueryTile, rows - start);
            const long long score_stride = static_cast<long long>(rows) * count;
            const size_t from = static_cast<size_t>(start) * d;
            if (!check(cublasSgemmStridedBatched(blas_, CUBLAS_OP_T, CUBLAS_OP_N, rows, count, hd, &scale,
                                               w.k, d, hd, w.q + from, d, hd, &zero,
                                               w.scores, rows, score_stride, heads), "Vision attention scores", err)) return false;
            cuda_vision_ops::softmax<<<dim3(count, heads), kThreads, 0, stream_>>>(w.scores, rows, score_stride, count);
            if (!check(cublasSgemmStridedBatched(blas_, CUBLAS_OP_N, CUBLAS_OP_N, hd, count, rows, &one,
                                               w.v, d, hd, w.scores, rows, score_stride, &zero,
                                               w.attended_tile, hd, static_cast<long long>(hd) * count, heads),
                       "Vision attention output", err)) return false;
            store_vision_attention<<<blocks(count * d), kThreads, 0, stream_>>>(
                w.attended_tile, w.attended + from, count, heads, hd);
        }
        return true;
    }

    int device_ = -1;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    bool reuse_attention_workspace_ = false;
    VitCfg cfg_;
    std::vector<float> cls_, cls_position_, position_;
    Linear patch_, projector_in_, projector_out_;
    Norm pre_norm_;
    std::array<Layer, 23> layers_;
    Buffer work_, expanded_, blas_work_;
    size_t max_expand_ = 0, weight_bytes_ = 0;
};

}  // namespace ink2
