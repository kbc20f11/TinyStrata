// Private adapter appended to the pinned llama.cpp clip.cpp at configure time.
// The image encoder is serialized by TinyStrata: call only after an encode has
// completed and its output has been copied to the mtmd-owned host vector.
extern "C" size_t tinystrata_clip_workspace_bytes(struct clip_ctx * ctx) {
    if (ctx == nullptr || !ctx->sched) {
        return 0;
    }
    size_t bytes = 0;
    for (const auto & entry : ctx->mem_compute) {
        const ggml_backend_dev_t device = entry.first;
        if (device != nullptr &&
            (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
             ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
            bytes += entry.second;
        }
    }
    return bytes;
}

extern "C" size_t tinystrata_clip_release_workspace(struct clip_ctx * ctx) {
    const size_t bytes = tinystrata_clip_workspace_bytes(ctx);
    if (bytes == 0) {
        return 0;
    }
    // Do not leave an in-flight compute or tensor copy using the allocator.
    // Keep the CUDA backend, its warmed kernel pool and model weights alive.
    for (ggml_backend_t backend : ctx->backend_ptrs) {
        ggml_backend_synchronize(backend);
    }

    ggml_backend_sched_ptr replacement(ggml_backend_sched_new(
        ctx->backend_ptrs.data(), ctx->backend_buft.data(),
        static_cast<int>(ctx->backend_ptrs.size()), ctx->max_nodes, false, true));
    if (!replacement) {
        return 0;
    }
    if (ctx->tinystrata_cb_eval != nullptr) {
        ggml_backend_sched_set_eval_callback(replacement.get(),
                                             ctx->tinystrata_cb_eval,
                                             ctx->tinystrata_cb_eval_user_data);
    }

    // A scheduler owns its compute allocator, but not the supplied backends.
    // Swapping then destroying the old scheduler frees only the graph buffers.
    // clip_encode already warms/reserves a new graph when is_allocated is false.
    ctx->sched.swap(replacement);
    ctx->is_allocated = false;
    ctx->mem_compute.clear();
    replacement.reset();
    return bytes;
}
