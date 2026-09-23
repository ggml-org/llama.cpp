#pragma once

#include "ggml-backend.h"
#include "ggml-kv-stream-partial.h"
#include "ggml-kv-stream.h"

// Two two-split partials, a normalized staging output, and a device validation flag.
struct ggml_kv_stream_block_layout {
    ggml_kv_stream_partial_layout partial;
    size_t second_offset = 0, value_offset = 0, value_bytes = 0, status_offset = 0, bytes = 0;
};
GGML_API ggml_kv_stream_partial_result ggml_kv_stream_block_layout_make(
        size_t rows, size_t width, ggml_kv_stream_block_layout & output);

struct ggml_kv_stream_query_tile {
    size_t queries = 0, first_row = 0, rows = 0;
};
// Bound each launch to 256 queries; row offsets address the full-batch accumulator.
GGML_API bool ggml_kv_stream_query_tile_make(
        size_t queries, size_t heads, size_t first, ggml_kv_stream_query_tile & output);

// Optional registry extension "ggml_backend_kv_stream_partial_ops". Partial/conversion calls complete before returning;
// version 6 resume/span calls enqueue work and require the caller's stream/event lifetime fence.
// Getter may return null when disabled. Call outside active capture; CUDA execution errors follow backend error handling.
// All tensor/workspace buffers belong to the backend; workspace must not overlap inputs or public output.
struct ggml_kv_stream_resume_plan;
struct ggml_kv_stream_partial_ops {
    uint32_t version;
    bool (*supports)(ggml_backend_t backend, const ggml_tensor * attention);
    // All query tiles finish before return; encoded K/V or conversion planes must remain valid until then.
    bool (*partial)(ggml_backend_t backend, const ggml_tensor * attention, ggml_backend_buffer_t workspace, bool second);
    // Validate on device and publish output only on success; failure preserves the public output bytes.
    bool (*merge)(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t workspace);
    // Version 2: fold both exports into the first (unnormalized) plane; scratch is unspecified on failure.
    bool (*fold)(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t workspace);
    // Set one export to empty, including both splits; never touches public output.
    bool (*clear)(ggml_backend_t backend, ggml_tensor * output, ggml_backend_buffer_t workspace, bool second);
    // Version 3: query the compiled native path, actual online writer, and F16 converter before planning storage.
    ggml_kv_stream_capabilities (*capabilities)(ggml_backend_t backend, int32_t key, int32_t value);
    bool (*supports_conversion)(ggml_backend_t backend, const ggml_tensor * source, const ggml_tensor * destination);
    // Caller supplies a disjoint, bounded F16 plane; this operation does not allocate device scratch.
    bool (*convert)(ggml_backend_t backend, const ggml_tensor * source, ggml_tensor * destination);
    // Version 4: ordinary attention, requiring the caller's native FA output allocation including backend extras.
    bool (*direct)(ggml_backend_t backend, const ggml_tensor * attention) = nullptr;
    // Version 6: plan a TG1/TG2 native vector evaluation, then consume ordered logical spans.
    bool (*resume_plan)(ggml_backend_t backend, int32_t key, int32_t value, uint32_t heads, uint32_t kv_heads,
            uint32_t queries, size_t tokens, ggml_kv_stream_resume_plan & output) = nullptr;
    // No allocation; scratch is caller-owned and retains per-thread state until the final span publishes output.
    bool (*resume)(ggml_backend_t backend, const ggml_tensor * attention, ggml_backend_buffer_t workspace,
            const ggml_kv_stream_resume_plan & plan, size_t tokens, size_t first, bool last) = nullptr;
    // Consume a retained stage-7 span plan without gathering or resetting attention state.
    bool (*spans)(ggml_backend_t backend, const ggml_tensor * attention,
            ggml_kv_stream_span_plan_t spans, ggml_backend_buffer_t workspace) = nullptr;
    // Version 7: exact bounded workspace for either vector or F16 MMA span execution.
    bool (*spans_workspace)(ggml_backend_t backend, const ggml_tensor * attention,
            ggml_kv_stream_span_plan_t spans, size_t & bytes) = nullptr;
    // Version 8: invoke the exact tile-local MMA dequantizer for qualification and reusable staging.
    bool (*convert_mma_rows)(ggml_backend_t backend, const ggml_tensor * source,
            ggml_tensor * destination) = nullptr;
    // Version 9: reserve the largest TG3/TG4 MMA span workspace for a given layout.
    bool (*mma_workspace)(ggml_backend_t backend, int32_t key, int32_t value, uint32_t heads,
            uint32_t kv_heads, size_t tokens, size_t spans, size_t & bytes) = nullptr;
};
using ggml_kv_stream_partial_ops_get = const ggml_kv_stream_partial_ops * (*)();

struct ggml_kv_stream_resume_plan {
    uint32_t heads = 0, queries = 0, splits = 0, values_per_thread = 0;
    size_t tokens = 0, state_bytes = 0, partial_offset = 0, meta_offset = 0, bytes = 0;
};
// Bound the native per-thread state and final partials without a context-sized KV allocation.
GGML_API bool ggml_kv_stream_resume_layout_make(
        uint32_t heads, uint32_t queries, uint32_t splits, uint32_t values_per_thread,
        ggml_kv_stream_resume_plan & output);
