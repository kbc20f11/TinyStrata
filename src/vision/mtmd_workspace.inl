// Private adapter appended to the pinned llama.cpp mtmd.cpp at configure time.
// Only the vision context is trimmed; model weights and host output embeddings
// remain owned by the same mtmd context throughout its lifetime.
extern "C" size_t tinystrata_clip_workspace_bytes(struct clip_ctx * ctx);
extern "C" size_t tinystrata_clip_release_workspace(struct clip_ctx * ctx);

extern "C" size_t tinystrata_mtmd_workspace_bytes(struct mtmd_context * ctx) {
    return ctx != nullptr ? tinystrata_clip_workspace_bytes(ctx->ctx_v) : 0;
}

extern "C" size_t tinystrata_mtmd_release_workspace(struct mtmd_context * ctx) {
    return ctx != nullptr ? tinystrata_clip_release_workspace(ctx->ctx_v) : 0;
}
