#include "ggml-kv-stream-partial.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <utility>

using status = ggml_kv_stream_partial_status;

// Check products before forming byte counts or flattened indices.
static bool multiply(size_t a, size_t b, size_t & output) {
    if (b && a > SIZE_MAX/b) return false;
    output = a*b;
    return true;
}

// A packed producer buffer contains a numerator plane followed by aligned two-float metadata.
ggml_kv_stream_partial_result ggml_kv_stream_partial_layout_make(
        size_t rows, size_t parts, size_t width, size_t alignment, ggml_kv_stream_partial_layout & output) {
    if (!rows || !parts || !width || alignment < alignof(ggml_kv_stream_partial_meta) || (alignment & (alignment-1)))
        return {status::invalid_shape};
    ggml_kv_stream_partial_layout next;
    next.rows = rows; next.parts = parts; next.width = width;
    if (!multiply(rows, parts, next.entries) || !multiply(next.entries, width, next.elements) ||
            !multiply(next.elements, sizeof(float), next.numerator_bytes) ||
            !multiply(next.entries, sizeof(ggml_kv_stream_partial_meta), next.meta_bytes) ||
            next.numerator_bytes > SIZE_MAX - (alignment-1)) return {status::overflow};
    next.meta_offset = (next.numerator_bytes + alignment-1) & ~(alignment-1);
    if (next.meta_bytes > SIZE_MAX - next.meta_offset) return {status::overflow};
    next.bytes = next.meta_offset + next.meta_bytes;
    output = next;
    return {};
}

// Validate only the referenced prefix; extra capacity can hold unrelated rows or padding.
static bool readable(const void * data, size_t bytes, size_t alignment) {
    const auto address = reinterpret_cast<uintptr_t>(data);
    return data && address % alignment == 0 && bytes <= UINTPTR_MAX - address;
}

// Zero mass is an identity only when the producer also cleared the whole numerator.
static ggml_kv_stream_partial_result validate(const ggml_kv_stream_partial_view & view, size_t input,
        ggml_kv_stream_partial_layout & layout) {
    auto result = ggml_kv_stream_partial_layout_make(view.rows, view.parts, view.width, alignof(ggml_kv_stream_partial_meta), layout);
    if (result.status != status::success) { result.input = input; return result; }
    if (view.numerator_count < layout.elements || view.meta_count < layout.entries ||
            !readable(view.numerator, layout.numerator_bytes, alignof(float)) ||
            !readable(view.meta, layout.meta_bytes, alignof(ggml_kv_stream_partial_meta))) return {status::invalid_buffer, input};
    for (size_t row = 0; row < view.rows; ++row) for (size_t part = 0; part < view.parts; ++part) {
        const size_t index = row*view.parts + part;
        const auto & meta = view.meta[index];
        if (!std::isfinite(meta.normalizer) || meta.normalizer < 0 || std::isnan(meta.max_logit) ||
                meta.max_logit == std::numeric_limits<float>::infinity() ||
                (meta.normalizer > 0 && !std::isfinite(meta.max_logit))) return {status::invalid_partial, input, row, part};
        for (size_t d = 0; d < view.width; ++d) {
            const float value = view.numerator[index*view.width + d];
            if (!std::isfinite(value) || (meta.normalizer == 0 && value != 0)) return {status::invalid_partial, input, row, part};
        }
    }
    return {};
}

// Do not publish an infinite FP32 accumulator even when a normalized quotient could still be finite.
static bool fits_float(double value) {
    return std::isfinite(value) && std::abs(value) <= double(std::numeric_limits<float>::max());
}

// Use common maximum rescaling; inputs can include a previously merged accumulator for streamed updates.
ggml_kv_stream_partial_result ggml_kv_stream_partial_merge(
        const std::vector<ggml_kv_stream_partial_view> & inputs, ggml_kv_stream_partial_batch & output) {
    if (inputs.empty()) return {status::invalid_shape};
    const size_t rows = inputs.front().rows, width = inputs.front().width;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rows != rows || inputs[i].width != width) return {status::invalid_shape, i};
        ggml_kv_stream_partial_layout layout;
        const auto result = validate(inputs[i], i, layout);
        if (result.status != status::success) return result;
    }
    ggml_kv_stream_partial_layout layout;
    const auto result = ggml_kv_stream_partial_layout_make(rows, 1, width, alignof(ggml_kv_stream_partial_meta), layout);
    if (result.status != status::success) return result;
    try {
        ggml_kv_stream_partial_batch next;
        next.rows = rows; next.parts = 1; next.width = width;
        std::vector<double> numerator;
        if (layout.elements > next.numerator.max_size() || rows > next.meta.max_size() || width > numerator.max_size())
            return {status::overflow};
        next.numerator.assign(layout.elements, 0);
        next.meta.resize(rows);
        numerator.resize(width);
        for (size_t row = 0; row < rows; ++row) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (const auto & view : inputs) for (size_t part = 0; part < view.parts; ++part) {
                const auto & meta = view.meta[row*view.parts + part];
                if (meta.normalizer > 0) maximum = std::max(maximum, meta.max_logit);
            }
            // Do not evaluate -infinity - -infinity for an all-empty row.
            if (!std::isfinite(maximum)) continue;
            double denominator = 0;
            std::fill(numerator.begin(), numerator.end(), 0);
            for (const auto & view : inputs) for (size_t part = 0; part < view.parts; ++part) {
                const size_t index = row*view.parts + part;
                const auto & meta = view.meta[index];
                if (meta.normalizer == 0) continue;
                const double scale = std::exp(double(meta.max_logit) - double(maximum));
                denominator += scale*double(meta.normalizer);
                for (size_t d = 0; d < width; ++d) numerator[d] += scale*double(view.numerator[index*width + d]);
            }
            if (!fits_float(denominator) || denominator <= 0) return {status::overflow, SIZE_MAX, row};
            next.meta[row] = {maximum, float(denominator)};
            for (size_t d = 0; d < width; ++d) {
                if (!fits_float(numerator[d])) return {status::overflow, SIZE_MAX, row};
                next.numerator[row*width+d] = float(numerator[d]);
            }
        }
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed};
    }
}

// Normalize once after all disjoint KV contributions have been accumulated.
ggml_kv_stream_partial_result ggml_kv_stream_partial_normalize(
        const ggml_kv_stream_partial_view & input, ggml_kv_stream_partial_value & output) {
    if (input.parts != 1) return {status::invalid_shape, 0};
    ggml_kv_stream_partial_layout layout;
    const auto result = validate(input, 0, layout);
    if (result.status != status::success) return result;
    try {
        ggml_kv_stream_partial_value next;
        next.rows = input.rows; next.width = input.width;
        if (layout.elements > next.value.max_size() || input.rows > next.empty.max_size()) return {status::overflow};
        next.value.assign(layout.elements, 0);
        next.empty.assign(input.rows, 0);
        for (size_t row = 0; row < input.rows; ++row) {
            const double denominator = input.meta[row].normalizer;
            if (denominator == 0) { next.empty[row] = 1; continue; }
            for (size_t d = 0; d < input.width; ++d) {
                const double value = double(input.numerator[row*input.width+d])/denominator;
                if (!fits_float(value)) return {status::overflow, 0, row};
                next.value[row*input.width+d] = float(value);
            }
        }
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed};
    }
}
