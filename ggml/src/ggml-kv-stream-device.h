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

// Optional registry extension "ggml_backend_kv_stream_partial_ops". All calls complete before returning.
// Getter may return null when disabled. Call outside active capture; CUDA execution errors follow backend error handling.
// All tensor/workspace buffers belong to the backend; workspace must not overlap inputs or public output.
struct ggml_kv_stream_partial_ops {
    uint32_t version;
    bool (*supports)(ggml_backend_t backend, const ggml_tensor * attention);
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
};
using ggml_kv_stream_partial_ops_get = const ggml_kv_stream_partial_ops * (*)();
