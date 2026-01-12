# Heterogeneous Tensor Layout Architecture Design

**Date:** 2026-01-07
**Status:** Draft
**Goal:** Fix OOM, clean architecture, per-tensor optimal layouts

## Problem Statement

The current SYCL backend has multiple layout conversion paths (AoS→SOA, SOA→COALESCED, AoS→XMX_TILED) with scattered decision points and potential memory duplication. When XMX tiled conversion is enabled, both the original buffer and tiled buffer may exist simultaneously, causing OOM errors.

## Design Goals

1. **Fix OOM** - Ensure only one canonical buffer per tensor
2. **Clean architecture** - Single source of truth for layout state
3. **Heterogeneous layouts** - Each tensor type gets its optimal layout based on usage

## Architecture

### 1. Layout Descriptor Struct

Replace `optimized_feature.reorder_` and `xmx_mxfp4_tiled[device]` with unified struct:

```cpp
enum class layout_mode : uint8_t {
    AOS = 0,        // Original Array-of-Structures (mmap'd data)
    SOA,            // Structure-of-Arrays
    COALESCED,      // Warp-coalesced tile layout
    XMX_TILED,      // XMX matrix engine tiled layout
};

struct tensor_layout_info {
    layout_mode mode = layout_mode::AOS;
    void* data_ptr = nullptr;      // Canonical data pointer
    size_t size = 0;               // Buffer size in bytes
    bool owns_memory = false;      // True = we allocated, must free
    int device_id = -1;            // GPU device ID

    // Quantization metadata (avoids recomputation)
    ggml_type qtype = GGML_TYPE_F32;
    int64_t n_elements = 0;
    int64_t n_experts = 1;         // For MoE tensors

    // XMX-specific (only valid when mode == XMX_TILED)
    struct {
        int64_t tile_n = 0;
        int64_t tile_k = 0;
        int64_t n_tile_groups = 0;
    } xmx_info;

    void release(sycl::queue& q) {
        if (owns_memory && data_ptr) {
            sycl::free(data_ptr, q);
            data_ptr = nullptr;
        }
    }
};
```

### 2. Layout Selection Policy

Lookup table mapping `{qtype, usage} → optimal layout` with env var overrides:

```cpp
enum class tensor_usage : uint8_t {
    UNKNOWN = 0,
    ATTENTION_WEIGHT,    // Q, K, V, O projections
    FFN_WEIGHT,          // feed-forward non-MoE
    MOE_EXPERT_WEIGHT,   // MoE expert gate/up/down
    MOE_GATE,            // MoE routing gate
    EMBEDDING,           // token embeddings
    NORM,                // RMS/LayerNorm weights
};

struct layout_policy {
    static layout_mode get_optimal(ggml_type qtype, tensor_usage usage) {
        static const std::map<std::pair<ggml_type, tensor_usage>, layout_mode> table = {
            // MoE experts: XMX tiled for GEMM performance
            {{GGML_TYPE_MXFP4, tensor_usage::MOE_EXPERT_WEIGHT}, layout_mode::XMX_TILED},
            {{GGML_TYPE_Q8_0,  tensor_usage::MOE_EXPERT_WEIGHT}, layout_mode::XMX_TILED},

            // Attention weights: COALESCED for decode (batch=1 MMVQ)
            {{GGML_TYPE_Q4_0, tensor_usage::ATTENTION_WEIGHT}, layout_mode::COALESCED},
            {{GGML_TYPE_Q8_0, tensor_usage::ATTENTION_WEIGHT}, layout_mode::COALESCED},
            {{GGML_TYPE_Q6_K, tensor_usage::ATTENTION_WEIGHT}, layout_mode::COALESCED},

            // FFN weights: COALESCED
            {{GGML_TYPE_Q4_0, tensor_usage::FFN_WEIGHT}, layout_mode::COALESCED},
        };

        auto it = table.find({qtype, usage});
        if (it != table.end()) return it->second;

        return layout_mode::SOA;  // Safe default
    }

    // Env var override: GGML_SYCL_LAYOUT_<TYPE>_<USAGE>=<mode>
    static layout_mode get_with_override(ggml_type qtype, tensor_usage usage);
};
```

### 3. Two-Phase Lifecycle

**Phase 1: Upload** (in `set_tensor()`)
- Upload tensor as AoS to GPU
- Initialize layout descriptor with `{mode=AOS, data_ptr, owns_memory=true}`
- No conversion yet

**Phase 2: Finalize** (explicit `finalize_layouts()` call before inference)
- Iterate all weight tensors
- Infer usage from tensor name
- Convert each to optimal layout
- Free original buffer after conversion

```cpp
void ggml_sycl_finalize_layouts(ggml_backend_t backend) {
    for (auto& tensor : all_weight_tensors) {
        tensor_usage usage = infer_usage_from_name(tensor->name);
        layout_mode target = layout_policy::get_with_override(tensor->type, usage);

        if (extra->layout.mode != target) {
            convert_tensor_layout(tensor, target);
        }
    }
}
```

### 4. Layout Conversion with Memory Deduplication

```cpp
void convert_tensor_layout(ggml_tensor* tensor, layout_mode target) {
    auto& layout = extra->layout;
    sycl::queue& q = *get_stream_for_device(layout.device_id);

    void* new_data = nullptr;
    size_t new_size = compute_layout_size(tensor, target);
    new_data = sycl::malloc_device(new_size, q);

    // Convert based on target
    switch (target) {
        case layout_mode::SOA:
            convert_aos_to_soa(q, layout.data_ptr, new_data, tensor);
            break;
        case layout_mode::COALESCED:
            convert_aos_to_coalesced(q, layout.data_ptr, new_data, tensor);
            break;
        case layout_mode::XMX_TILED:
            convert_aos_to_xmx_tiled(q, layout.data_ptr, new_data, tensor);
            break;
    }

    q.wait();

    // FREE THE ORIGINAL - fixes OOM
    if (layout.owns_memory && layout.data_ptr) {
        sycl::free(layout.data_ptr, q);
    }

    // Update descriptor
    layout.data_ptr = new_data;
    layout.size = new_size;
    layout.mode = target;
    layout.owns_memory = true;
}
```

### 5. Kernel Dispatch

Single switch on `layout.mode`:

```cpp
void ggml_sycl_mul_mat(ggml_tensor* dst) {
    const auto& layout = src0_extra->layout;

    switch (layout.mode) {
        case layout_mode::AOS:
            mul_mat_vec_q_aos(layout.data_ptr, ...);
            break;
        case layout_mode::SOA:
            mul_mat_vec_q_soa(layout.data_ptr, ...);
            break;
        case layout_mode::COALESCED:
            mul_mat_vec_q_coalesced(layout.data_ptr, ...);
            break;
        case layout_mode::XMX_TILED:
            mul_mat_xmx_tiled(layout.data_ptr, layout.xmx_info, ...);
            break;
    }
}
```

### 6. Usage Inference from Tensor Names

```cpp
tensor_usage infer_usage_from_name(const char* name) {
    if (strstr(name, "ffn_gate_exps") || strstr(name, "ffn_up_exps") ||
        strstr(name, "ffn_down_exps"))
        return tensor_usage::MOE_EXPERT_WEIGHT;

    if (strstr(name, "ffn_gate_inp"))
        return tensor_usage::MOE_GATE;

    if (strstr(name, "attn_q") || strstr(name, "attn_k") ||
        strstr(name, "attn_v") || strstr(name, "attn_output"))
        return tensor_usage::ATTENTION_WEIGHT;

    if (strstr(name, "ffn_gate.") || strstr(name, "ffn_up.") ||
        strstr(name, "ffn_down."))
        return tensor_usage::FFN_WEIGHT;

    if (strstr(name, "token_embd") || strstr(name, "output.weight"))
        return tensor_usage::EMBEDDING;

    if (strstr(name, "_norm"))
        return tensor_usage::NORM;

    return tensor_usage::UNKNOWN;
}
```

## Migration Strategy

Incremental migration (not big-bang):

1. Add `tensor_layout_info` to `ggml_tensor_extra_gpu` alongside existing fields
2. Update `set_tensor()` to populate layout descriptor
3. Add `finalize_layouts()` call before graph recording
4. Update one kernel family at a time to use `layout.data_ptr`
5. Once all kernels migrated, remove deprecated fields:
   - `extra->optimized_feature`
   - `extra->xmx_mxfp4_tiled[device]`
   - `extra->data_device[i]`

## Files to Modify

- `ggml/src/ggml-sycl/common.hpp` - Add `tensor_layout_info`, `layout_policy`
- `ggml/src/ggml-sycl/ggml-sycl.cpp` - Update `set_tensor()`, add `finalize_layouts()`
- `ggml/src/ggml-sycl/mmvq.cpp` - Update kernel dispatch
- `ggml/src/ggml-sycl/mmq.cpp` - Update kernel dispatch
- `ggml/src/ggml-sycl/convert.cpp` - Consolidate conversion functions
- `ggml/src/ggml-sycl/common.cpp` - Update cleanup

## Benefits

1. **Fixes OOM** - No duplicate buffers, one canonical data pointer per tensor
2. **Clean architecture** - Single source of truth for layout state
3. **Heterogeneous layouts** - Each tensor gets optimal layout based on type + usage
4. **Extensible** - Easy to add new layout types
5. **Tunable** - Env var overrides for experimentation
