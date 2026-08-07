#!/usr/bin/env python3
from pathlib import Path

path = Path("ggml/src/ggml-cuda/tiered.cu")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one source match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
'''// Identifies tiered buffers by init_tensor's function pointer rather than a
// ggml_backend_buft_name() string compare, since this runs for every src of
// every graph node, every token.
static tiered_buffer_context * buffer_context(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    size_t view_offset = 0;
    tensor = tiered_view_base(tensor, &view_offset);
    GGML_UNUSED(view_offset);
    ggml_backend_buffer_t buffer = tensor ? tensor->buffer : nullptr;
    if (!buffer || !buffer->context || buffer->iface.init_tensor != tiered_buffer_init_tensor) {
        return nullptr;
    }
    return static_cast<tiered_buffer_context *>(buffer->context);
}

// tensor->extra caches the resolved tensor_state, set in
// tiered_buffer_init_tensor, so the graph-compute hot path is a field read
// instead of an unordered_map lookup.
static tensor_state * state_for(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    size_t view_offset = 0;
    const ggml_tensor * base = tiered_view_base(tensor, &view_offset);
    if (!buffer_context(tensor)) {
        return nullptr;
    }
    return base ? static_cast<tensor_state *>(base->extra) : nullptr;
}
''',
'''// Identifies tiered buffers by init_tensor's function pointer rather than a
// ggml_backend_buft_name() string compare, since this runs for every src of
// every graph node, every token.
static tiered_buffer_context * buffer_context_base(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = tensor ? tensor->buffer : nullptr;
    if (!buffer || !buffer->context || buffer->iface.init_tensor != tiered_buffer_init_tensor) {
        return nullptr;
    }
    return static_cast<tiered_buffer_context *>(buffer->context);
}

static tiered_buffer_context * buffer_context(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    size_t view_offset = 0;
    const ggml_tensor * base = tiered_view_base(tensor, &view_offset);
    GGML_UNUSED(view_offset);
    return buffer_context_base(base);
}

// tensor->extra caches the resolved tensor_state, set in
// tiered_buffer_init_tensor, so the graph-compute hot path is a field read
// instead of an unordered_map lookup. Resolve a view chain only once here;
// buffer_context(tensor) used to walk the same chain a second time.
static tensor_state * state_for(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    size_t view_offset = 0;
    const ggml_tensor * base = tiered_view_base(tensor, &view_offset);
    GGML_UNUSED(view_offset);
    if (!buffer_context_base(base)) {
        return nullptr;
    }
    return base ? static_cast<tensor_state *>(base->extra) : nullptr;
}
''',
"single view-base traversal",
)

replace_once(
'''        const std::vector<int32_t> & ranked_ids,
        const std::vector<int32_t> & sorted_ids,
        size_t ids_per_row,
''',
'''        const std::vector<int32_t> & ranked_ids,
        size_t ids_per_row,
''',
"remove unused sorted ids argument",
)

replace_once(
'''    ctx->host_ids_scratch.assign(ranked_ids.begin(), ranked_ids.end());
    std::vector<int32_t> & host_ids = ctx->host_ids_scratch;
    std::sort(host_ids.begin(), host_ids.end());
    host_ids.erase(std::unique(host_ids.begin(), host_ids.end()), host_ids.end());

    set_device(ctx->device);

    // Decode selects a fixed number of experts, so it only needs room for those
    // rather than the whole stack. Prompt batches can touch every expert and
    // keep the original layout.
    const bool packed = (n_rows == 1) && tensor_offset == 0 && !tensor->view_src && tensor->ne[3] == 1;
''',
'''    // Decode selects a fixed number of experts and consumes them in router
    // order. It never needs the sorted/unique id list used by batched staging,
    // so avoid the copy + sort + unique on the per-token path.
    const bool packed = (n_rows == 1) && tensor_offset == 0 && !tensor->view_src && tensor->ne[3] == 1;
    std::vector<int32_t> & host_ids = ctx->host_ids_scratch;
    if (!packed) {
        host_ids.assign(ranked_ids.begin(), ranked_ids.end());
        std::sort(host_ids.begin(), host_ids.end());
        host_ids.erase(std::unique(host_ids.begin(), host_ids.end()), host_ids.end());
    }

    set_device(ctx->device);
''',
"skip decode id sorting",
)

replace_once(
'''    if (!stage_cached_experts(ctx, tensor, state, tensor_offset, ranked_ids, host_ids,
                              ids_per_row, n_rows, direct_ids, direct_stride)) {
''',
'''    if (!stage_cached_experts(ctx, tensor, state, tensor_offset, ranked_ids,
                              ids_per_row, n_rows, direct_ids, direct_stride)) {
''',
"update cache call",
)

path.write_text(text, encoding="utf-8")
print(f"applied tiered decode hot-path optimizations: {path}")
