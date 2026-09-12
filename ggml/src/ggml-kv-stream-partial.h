#pragma once

#include "ggml.h"

#include <limits>
#include <vector>

// Natural-exponential coordinates: numerator=sum(exp(score-max_logit)*V), normalizer=sum(exp(score-max_logit)).
// Matches the two-float CUDA metadata record without depending on CUDA headers.
struct alignas(8) ggml_kv_stream_partial_meta {
    float max_logit = -std::numeric_limits<float>::infinity();
    float normalizer = 0;
};
static_assert(sizeof(float) == 4 && sizeof(ggml_kv_stream_partial_meta) == 8 &&
              offsetof(ggml_kv_stream_partial_meta, normalizer) == 4, "partial metadata must be two packed FP32 values");

struct ggml_kv_stream_partial_layout {
    size_t rows = 0, parts = 0, width = 0;
    size_t elements = 0, entries = 0;
    size_t numerator_bytes = 0, meta_offset = 0, meta_bytes = 0, bytes = 0;
};

// Host-readable packed planes: numerator[(row*parts+part)*width+channel], meta[row*parts+part].
// For ordinary GGML attention, row=query*query_heads+head. Capacities may exceed the required prefix.
struct ggml_kv_stream_partial_view {
    size_t rows = 0, parts = 0, width = 0;
    const float * numerator = nullptr;
    size_t numerator_count = 0;
    const ggml_kv_stream_partial_meta * meta = nullptr;
    size_t meta_count = 0;
};
struct ggml_kv_stream_partial_batch {
    size_t rows = 0, parts = 0, width = 0;
    std::vector<float> numerator;
    std::vector<ggml_kv_stream_partial_meta> meta;
    ggml_kv_stream_partial_view view() const noexcept {
        return {rows, parts, width, numerator.data(), numerator.size(), meta.data(), meta.size()};
    }
};
struct ggml_kv_stream_partial_value {
    size_t rows = 0, width = 0;
    std::vector<float> value;
    // Empty/all-masked rows produce zeros, explicitly distinguished from a valid zero-valued result.
    std::vector<uint8_t> empty;
};
enum class ggml_kv_stream_partial_status { success, invalid_shape, invalid_buffer, invalid_partial, overflow, allocation_failed };
struct ggml_kv_stream_partial_result {
    ggml_kv_stream_partial_status status = ggml_kv_stream_partial_status::success;
    size_t input = SIZE_MAX, row = SIZE_MAX, part = SIZE_MAX;
};

// Checked scratch layout; alignment must be a power of two and at least alignof(meta).
GGML_API ggml_kv_stream_partial_result ggml_kv_stream_partial_layout_make(
        size_t rows, size_t parts, size_t width, size_t alignment, ggml_kv_stream_partial_layout & output);

// CPU reference for the producer/merge contract, not a GPU kernel or hot-path implementation.
// Inputs share rows/width but may have different part counts. Output has one unnormalized part per row.
// A zero normalizer requires a zero numerator; finite or -infinity empty maxima are accepted and canonicalized.
// Nonempty metadata/numerators must be finite. Failure leaves output unchanged; input/output aliases are allowed.
GGML_API ggml_kv_stream_partial_result ggml_kv_stream_partial_merge(
        const std::vector<ggml_kv_stream_partial_view> & inputs, ggml_kv_stream_partial_batch & output);

// Normalize a merged (parts==1) result once. Empty rows return zero values and empty=1, never 0/0.
// FP64 intermediates are checked before FP32 publication; underflow is allowed, overflow is rejected.
GGML_API ggml_kv_stream_partial_result ggml_kv_stream_partial_normalize(
        const ggml_kv_stream_partial_view & input, ggml_kv_stream_partial_value & output);
