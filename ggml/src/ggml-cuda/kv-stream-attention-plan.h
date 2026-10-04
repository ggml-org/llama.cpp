#pragma once

#include "ggml.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

enum class ggml_cuda_kv_stream_kernel_family : uint8_t { none, vector, tile, mma };
enum class ggml_cuda_kv_stream_execution_style : uint8_t { native, spanned, resumable };

// Serial attention geometry; live layout and hardware checks stay in the backend adapter.
struct ggml_cuda_kv_stream_attention_metadata {
    int32_t type_k = GGML_TYPE_COUNT, type_v = GGML_TYPE_COUNT;
    int64_t head_dim_k = 0, head_dim_v = 0;
    int64_t query_heads = 0, kv_heads = 0, queries = 0;
    int64_t active_tokens = 0, padded_tokens = 0;
};

// Backend evidence for the selected family, not inferred from a device CC threshold.
struct ggml_cuda_kv_stream_plan_support {
    bool stock_compiled = false, streamed_compiled = false;
    bool required_features = false, geometry = false, launch_resources = false;
};

struct ggml_cuda_kv_stream_workspace_requirements {
    // Output extras and caller-owned scratch are known; native backend-pool temporaries may not be.
    bool known = false;
    // Extras are adjacent to the output; scratch is a separate global-memory region.
    size_t output_extra_bytes = 0, scratch_bytes = 0, scratch_alignment = 128;
    // Per-block shared memory affects launch admission, not the global-memory grant.
    size_t shared_bytes = 0;
    bool backend_scratch_known = false;
    bool shared_bytes_known = false;
};

struct ggml_cuda_kv_stream_attention_plan {
    ggml_cuda_kv_stream_kernel_family family = ggml_cuda_kv_stream_kernel_family::none;
    ggml_cuda_kv_stream_execution_style style = ggml_cuda_kv_stream_execution_style::native;
    ggml_cuda_kv_stream_attention_metadata metadata;
    ggml_cuda_kv_stream_workspace_requirements requirements;
    size_t output_allocation_bytes = 0;
};

enum class ggml_cuda_kv_stream_plan_status : uint8_t {
    success, invalid_metadata, stock_unavailable, missing_stock_code,
    missing_streamed_implementation, unsupported_device_features, unsupported_geometry,
    unsupported_launch_resources, missing_requirements, invalid_requirements, overflow,
};

// Check geometry and derive the FP32 output payload before any backend selector inspects it.
static inline ggml_cuda_kv_stream_plan_status ggml_cuda_kv_stream_attention_metadata_validate(
        const ggml_cuda_kv_stream_attention_metadata & metadata, size_t & output_bytes) noexcept {
    using status = ggml_cuda_kv_stream_plan_status;
    if (metadata.type_k < 0 || metadata.type_k >= GGML_TYPE_COUNT ||
            metadata.type_v < 0 || metadata.type_v >= GGML_TYPE_COUNT ||
            metadata.head_dim_k <= 0 || metadata.head_dim_v <= 0 ||
            metadata.query_heads <= 0 || metadata.kv_heads <= 0 ||
            metadata.query_heads % metadata.kv_heads || metadata.queries <= 0 ||
            metadata.active_tokens < metadata.queries || metadata.padded_tokens < metadata.active_tokens)
        return status::invalid_metadata;
    const auto key_block = ggml_blck_size(ggml_type(metadata.type_k));
    const auto value_block = ggml_blck_size(ggml_type(metadata.type_v));
    if (key_block <= 0 || value_block <= 0 || metadata.head_dim_k % key_block || metadata.head_dim_v % value_block)
        return status::invalid_metadata;
    size_t bytes = sizeof(float);
    for (int64_t extent : {metadata.head_dim_v, metadata.query_heads, metadata.queries}) {
        if (uint64_t(extent) > std::numeric_limits<size_t>::max() / bytes) return status::overflow;
        bytes *= size_t(extent);
    }
    output_bytes = bytes;
    return status::success;
}

// Validate a metadata-only descriptor supplied by the backend; never select a family or allocate storage.
// Live tensor/stride/mask validation remains the caller's responsibility. Failure preserves output.
static inline ggml_cuda_kv_stream_plan_status ggml_cuda_kv_stream_attention_plan_make(
        const ggml_cuda_kv_stream_attention_metadata & metadata,
        ggml_cuda_kv_stream_kernel_family stock,
        ggml_cuda_kv_stream_execution_style execution,
        const ggml_cuda_kv_stream_plan_support & support,
        const ggml_cuda_kv_stream_workspace_requirements & requirements,
        ggml_cuda_kv_stream_attention_plan & output) noexcept {
    using status = ggml_cuda_kv_stream_plan_status;
    using family = ggml_cuda_kv_stream_kernel_family;
    using style = ggml_cuda_kv_stream_execution_style;
    if (stock > family::mma || execution > style::resumable) return status::invalid_metadata;
    size_t bytes = 0;
    const auto valid = ggml_cuda_kv_stream_attention_metadata_validate(metadata, bytes);
    if (valid != status::success) return valid;
    if (stock == family::none) return status::stock_unavailable;
    if (!support.stock_compiled) return status::missing_stock_code;
    if (execution != style::native && !support.streamed_compiled) return status::missing_streamed_implementation;
    if (!support.required_features) return status::unsupported_device_features;
    if (!support.geometry) return status::unsupported_geometry;
    if (!support.launch_resources) return status::unsupported_launch_resources;
    if (!requirements.known) return status::missing_requirements;
    if (!requirements.scratch_alignment || (requirements.scratch_alignment & (requirements.scratch_alignment - 1)) ||
            (execution != style::native && requirements.output_extra_bytes)) return status::invalid_requirements;

    if (requirements.output_extra_bytes > std::numeric_limits<size_t>::max() - bytes) return status::overflow;
    bytes += requirements.output_extra_bytes;
    if (requirements.scratch_bytes > std::numeric_limits<size_t>::max() - bytes) return status::overflow;

    ggml_cuda_kv_stream_attention_plan next;
    next.family = stock;
    next.style = execution;
    next.metadata = metadata;
    next.requirements = requirements;
    next.output_allocation_bytes = bytes;
    output = next;
    return status::success;
}
