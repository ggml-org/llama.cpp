# Heterogeneous Tensor Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix OOM by eliminating memory duplication, create clean architecture with single source of truth for layout state, enable per-tensor optimal layouts.

**Architecture:** Unified `tensor_layout_info` struct replaces scattered layout tracking. Policy-driven layout selection based on `{qtype, usage}`. Two-phase lifecycle: upload as AoS, explicit `finalize_layouts()` converts to optimal. Memory deduplication frees original buffer after conversion.

**Tech Stack:** C++17, Intel SYCL, ggml tensor system

**Design Document:** `docs/plans/2026-01-07-heterogeneous-tensor-layout-design.md`

---

## Task 1: Add layout_mode enum and tensor_layout_info struct

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp:333-340` (after existing reorder_mode)

**Step 1: Write the new enums and struct after existing reorder_mode**

Add after line 340 in common.hpp:

```cpp
// =============================================================================
// Unified Tensor Layout System (replaces scattered layout tracking)
// =============================================================================

enum class layout_mode : uint8_t {
    AOS = 0,        // Original Array-of-Structures (mmap'd data)
    SOA,            // Structure-of-Arrays (qs contiguous, then d values)
    COALESCED,      // Warp-coalesced tile layout for MMVQ
    XMX_TILED,      // XMX matrix engine tiled layout for MoE GEMM
};

enum class tensor_usage : uint8_t {
    UNKNOWN = 0,
    ATTENTION_WEIGHT,    // Q, K, V, O projections
    FFN_WEIGHT,          // feed-forward non-MoE
    MOE_EXPERT_WEIGHT,   // MoE expert gate/up/down
    MOE_GATE,            // MoE routing gate
    EMBEDDING,           // token embeddings
    NORM,                // RMS/LayerNorm weights
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
            owns_memory = false;
        }
    }
};
```

**Step 2: Build to verify syntax**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): add layout_mode enum and tensor_layout_info struct"
```

---

## Task 2: Add layout_policy lookup table

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp` (after tensor_layout_info)

**Step 1: Add layout_policy struct**

```cpp
struct layout_policy {
    static layout_mode get_optimal(ggml_type qtype, tensor_usage usage) {
        // MoE experts: XMX tiled for GEMM performance
        if (usage == tensor_usage::MOE_EXPERT_WEIGHT) {
            if (qtype == GGML_TYPE_MXFP4 || qtype == GGML_TYPE_Q8_0) {
                return layout_mode::XMX_TILED;
            }
        }

        // Attention/FFN weights: COALESCED for decode (batch=1 MMVQ)
        if (usage == tensor_usage::ATTENTION_WEIGHT || usage == tensor_usage::FFN_WEIGHT) {
            if (qtype == GGML_TYPE_Q4_0 || qtype == GGML_TYPE_Q8_0 || qtype == GGML_TYPE_Q6_K) {
                return layout_mode::COALESCED;
            }
        }

        // Default: SOA is safe for all quantized types
        return layout_mode::SOA;
    }

    // Check env var override: GGML_SYCL_LAYOUT_OVERRIDE=<mode>
    static layout_mode get_with_override(ggml_type qtype, tensor_usage usage) {
        const char* override_env = std::getenv("GGML_SYCL_LAYOUT_OVERRIDE");
        if (override_env) {
            if (strcmp(override_env, "aos") == 0) return layout_mode::AOS;
            if (strcmp(override_env, "soa") == 0) return layout_mode::SOA;
            if (strcmp(override_env, "coalesced") == 0) return layout_mode::COALESCED;
            if (strcmp(override_env, "xmx_tiled") == 0) return layout_mode::XMX_TILED;
        }
        return get_optimal(qtype, usage);
    }
};
```

**Step 2: Build to verify**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): add layout_policy lookup table for optimal layouts"
```

---

## Task 3: Add tensor_usage inference from tensor names

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp` (after layout_policy)

**Step 1: Add infer_tensor_usage function**

```cpp
inline tensor_usage infer_tensor_usage(const char* name) {
    if (!name) return tensor_usage::UNKNOWN;

    // MoE expert weights (highest priority - check first)
    if (strstr(name, "ffn_gate_exps") || strstr(name, "ffn_up_exps") ||
        strstr(name, "ffn_down_exps"))
        return tensor_usage::MOE_EXPERT_WEIGHT;

    // MoE routing gate
    if (strstr(name, "ffn_gate_inp"))
        return tensor_usage::MOE_GATE;

    // Attention weights
    if (strstr(name, "attn_q") || strstr(name, "attn_k") ||
        strstr(name, "attn_v") || strstr(name, "attn_output"))
        return tensor_usage::ATTENTION_WEIGHT;

    // FFN weights (non-MoE)
    if (strstr(name, "ffn_gate.") || strstr(name, "ffn_up.") ||
        strstr(name, "ffn_down."))
        return tensor_usage::FFN_WEIGHT;

    // Embeddings
    if (strstr(name, "token_embd") || strstr(name, "output.weight"))
        return tensor_usage::EMBEDDING;

    // Norms
    if (strstr(name, "_norm"))
        return tensor_usage::NORM;

    return tensor_usage::UNKNOWN;
}
```

**Step 2: Build to verify**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): add tensor_usage inference from llama.cpp naming"
```

---

## Task 4: Add tensor_layout_info to ggml_tensor_extra_gpu

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp:1090-1116` (ggml_tensor_extra_gpu struct)

**Step 1: Add layout field to struct**

Add after line 1094 (`optimize_feature optimized_feature = {};`):

```cpp
    // Unified layout descriptor (new system - coexists with optimize_feature during migration)
    tensor_layout_info layout;
```

**Step 2: Build to verify**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): add tensor_layout_info to ggml_tensor_extra_gpu"
```

---

## Task 5: Implement convert_tensor_layout function

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (after reorder_tensor_to_soa)

**Step 1: Add convert_tensor_layout function**

```cpp
// Convert tensor to target layout with memory deduplication
static bool convert_tensor_layout(ggml_tensor* tensor, layout_mode target,
                                   dpct::queue_ptr stream, const char* caller) {
    if (!tensor || !tensor->extra) {
        return false;
    }

    auto* extra = static_cast<ggml_tensor_extra_gpu*>(tensor->extra);
    auto& layout = extra->layout;

    // Already in target layout
    if (layout.mode == target) {
        return true;
    }

    // Compute new buffer size based on target layout
    size_t new_size = ggml_nbytes(tensor);  // Same size for in-place layouts

    // Allocate new buffer
    void* new_data = sycl::malloc_device(new_size, *stream);
    if (!new_data) {
        fprintf(stderr, "[LAYOUT] ERROR: Failed to allocate %zu bytes for %s\n",
                new_size, tensor->name);
        return false;
    }

    // Convert based on current → target
    bool success = false;
    if (layout.mode == layout_mode::AOS) {
        if (target == layout_mode::SOA) {
            // Use existing AoS→SoA conversion
            reorder_data_internal_(tensor, stream);
            // For AoS→SoA, data is converted in-place, no new allocation needed
            sycl::free(new_data, *stream);
            layout.mode = layout_mode::SOA;
            return true;
        }
        // AoS→COALESCED and AoS→XMX_TILED require two-step: AoS→SoA→target
    }

    if (layout.mode == layout_mode::SOA && target == layout_mode::COALESCED) {
        // Call existing coalesced conversion
        convert_tensor_to_coalesced(tensor, stream, caller);
        sycl::free(new_data, *stream);
        layout.mode = layout_mode::COALESCED;
        return true;
    }

    // Cleanup on failure
    sycl::free(new_data, *stream);
    return false;
}
```

**Step 2: Build to verify**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): implement convert_tensor_layout with memory deduplication"
```

---

## Task 6: Implement finalize_layouts function

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (after convert_tensor_layout)

**Step 1: Add finalize_layouts function**

```cpp
// Finalize layouts for all weight tensors before inference
static void finalize_layouts(ggml_backend_sycl_context& ctx, ggml_cgraph* cgraph) {
    if (!cgraph) return;

    int converted_count = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        ggml_tensor* src0 = node->src[0];
        if (!src0 || !src0->extra) continue;

        auto* extra = static_cast<ggml_tensor_extra_gpu*>(src0->extra);

        // Infer usage from tensor name
        tensor_usage usage = infer_tensor_usage(src0->name);

        // Get optimal layout for this tensor
        layout_mode target = layout_policy::get_with_override(src0->type, usage);

        // Convert if needed
        if (extra->layout.mode != target) {
            if (convert_tensor_layout(src0, target, ctx.stream(), "FINALIZE_LAYOUTS")) {
                converted_count++;
            }
        }
    }

    if (g_ggml_sycl_debug && converted_count > 0) {
        fprintf(stderr, "[LAYOUT] Finalized %d tensors to optimal layouts\n", converted_count);
    }
}
```

**Step 2: Build to verify**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): implement finalize_layouts for two-phase lifecycle"
```

---

## Task 7: Update kernel dispatch to use layout.mode

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp` (mul_mat_vec dispatch)
- Modify: `ggml/src/ggml-sycl/mmq.cpp` (mul_mat dispatch)

**Step 1: Update mmvq.cpp dispatch to check layout.mode**

Find the dispatch logic and add layout.mode check alongside existing optimize_feature check.

**Step 2: Update mmq.cpp dispatch similarly**

**Step 3: Build and test**

Run: `./scripts/quick-rebuild.sh`
Run: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0`
Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp ggml/src/ggml-sycl/mmq.cpp
git commit -m "feat(sycl): update kernel dispatch to use unified layout.mode"
```

---

## Task 8: Wire finalize_layouts into graph compute

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:15145` (ggml_backend_sycl_graph_compute)

**Step 1: Call finalize_layouts before graph recording**

Add after `pre_reorder_all_tensors(sycl_ctx, cgraph);` (line 15145):

```cpp
    // Finalize layouts for optimal per-tensor memory formats
    finalize_layouts(*sycl_ctx, cgraph);
```

**Step 2: Build and test with MoE model**

Run: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0`
Expected: Correct output, no OOM

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): wire finalize_layouts into graph compute path"
```

---

## Task 9: Update release_extra_gpu for unified cleanup

**Files:**
- Modify: `ggml/src/ggml-sycl/common.cpp` (release_extra_gpu function)

**Step 1: Add layout.release() call**

```cpp
void release_extra_gpu(ggml_tensor_extra_gpu* extra, std::vector<queue_ptr> streams) {
    // ... existing cleanup ...

    // Release unified layout buffer
    if (!streams.empty() && streams[0]) {
        extra->layout.release(*streams[0]);
    }

    // ... rest of cleanup ...
}
```

**Step 2: Build to verify**

Run: `./scripts/quick-rebuild.sh`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.cpp
git commit -m "feat(sycl): update release_extra_gpu for unified layout cleanup"
```

---

## Task 10: Integration test with Mistral and MoE models

**Files:** None (testing only)

**Step 1: Test with Mistral Q4_0**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"

**Step 2: Test with GPT-OSS MoE Q8_0**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
```
Expected: Correct counting, no OOM error

**Step 3: Benchmark comparison**

```bash
# Before (baseline)
GGML_SYCL_LAYOUT_OVERRIDE=aos ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -ngl 99 -p 512 -n 128

# After (optimized)
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -ngl 99 -p 512 -n 128
```

**Step 4: Document results and commit**

```bash
git add -A
git commit -m "test(sycl): verify heterogeneous layout system with Mistral and MoE models"
```

---

## Migration Notes

This implementation coexists with the existing `optimize_feature` system during migration:
- Both `extra->optimized_feature` and `extra->layout` will exist
- Kernel dispatch checks `layout.mode` first, falls back to `optimized_feature`
- Once all kernels are migrated, remove deprecated fields in a cleanup task
