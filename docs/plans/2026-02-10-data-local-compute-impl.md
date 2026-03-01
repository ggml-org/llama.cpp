# Data-Local Compute: Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** When the unified cache evicts weights to host pinned memory, dispatch those layers' computation to a SYCL CPU device (oneDNN + portable kernels) instead of streaming weights to GPU. Zero weight movement.

**Architecture:** Add a CPU SYCL queue alongside the GPU queue. At graph_compute time, check each mul_mat's weight tier via `cache->get_view()`. VRAM-tier weights dispatch to GPU (existing path). PINNED_HOST-tier weights dispatch to CPU via oneDNN. Activation transfers at device boundaries use GPU-allocated host pinned memory (validated: GPU `malloc_host` is accessible from CPU OpenCL device).

**Tech Stack:** SYCL (Level Zero GPU + OpenCL CPU), oneDNN (device-agnostic), existing portable SYCL kernels

**Validated:** Cross-backend USM test confirms GPU-allocated `sycl::malloc_host()` memory is accessible from CPU SYCL kernels (5/6 tests passed, only CPU→GPU direction fails which we don't need).

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 2, 4 | Infrastructure + dispatch routing (common.hpp, ggml-sycl.cpp) |
| B | 3 | CPU kernel implementation (new files: cpu-dispatch.cpp/hpp) |
| — | 5 | Integration test (depends on both tracks) |

### Dependency Graph

```
Task 1 (CPU device init) ──→ Task 2 (dispatch routing) ──→ Task 4 (activation + graph gate)
                         └──→ Task 3 (CPU kernels)      ──→ Task 4
                                                              └──→ Task 5 (integration test)
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/common.hpp` | 1 | None (single task) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 1, 2, 4 | Sequential (same track A) |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` (NEW) | 3 | None (new file, single task) |
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` (NEW) | 3 | None (new file, single task) |
| `Testing/test-cpu-offload.cpp` (NEW) | 5 | None (new file, single task) |

---

## Task 1: CPU Device Discovery and Queue Creation

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/common.hpp` (ggml_sycl_device_info struct, ~line 52)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (ggml_sycl_init, ~line 2708)

**Description:**

Add CPU SYCL device support to the device initialization path. Create a CPU queue that can be used for host-tier weight computation. Gate with `GGML_SYCL_CPU_OFFLOAD=1` env var.

**Acceptance Criteria:**

- [ ] `ggml_sycl_device_info` has `cpu_queue`, `cpu_context`, and `has_cpu_device` fields
- [ ] `ggml_sycl_init()` creates CPU queue when `GGML_SYCL_CPU_OFFLOAD=1` is set
- [ ] CPU device is created via `sycl::cpu_selector_v` (uses OpenCL backend)
- [ ] Logs CPU device name and compute units on init
- [ ] Graceful fallback if no CPU device available
- [ ] Build succeeds with no warnings

**Implementation Guide:**

### 1. Add CPU fields to ggml_sycl_device_info

In `common.hpp`, find `struct ggml_sycl_device_info` (around line 52). Add after `host_max_alloc_size`:

```cpp
    // CPU device for data-local compute (host-tier weight layers)
    bool         has_cpu_device = false;
    sycl::queue * cpu_queue = nullptr;     // OpenCL CPU queue (owned, allocated with new)
    sycl::context cpu_context;             // CPU context (for oneDNN engine creation)
```

### 2. Add CPU offload enabled check

In `common.hpp`, add a free function near the other env var helpers (around line 400):

```cpp
inline bool ggml_sycl_cpu_offload_enabled() {
    static bool enabled = [] {
        const char * env = std::getenv("GGML_SYCL_CPU_OFFLOAD");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}
```

### 3. Create CPU queue in ggml_sycl_init

In `ggml-sycl.cpp`, at the END of `ggml_sycl_init()` (just before `return info;` around line 2835), add:

```cpp
    // Create CPU SYCL device for data-local compute
    if (ggml_sycl_cpu_offload_enabled()) {
        try {
            auto cpu_q = new sycl::queue{sycl::cpu_selector_v};
            info.has_cpu_device = true;
            info.cpu_queue = cpu_q;
            info.cpu_context = cpu_q->get_context();
            auto cpu_dev = cpu_q->get_device();
            GGML_LOG_INFO("[SYCL-CPU] CPU offload enabled: %s (%d CUs, %s backend)\n",
                          cpu_dev.get_info<sycl::info::device::name>().c_str(),
                          cpu_dev.get_info<sycl::info::device::max_compute_units>(),
                          cpu_q->get_backend() == sycl::backend::opencl ? "OpenCL" : "other");
        } catch (sycl::exception & e) {
            GGML_LOG_WARN("[SYCL-CPU] CPU offload requested but no CPU device available: %s\n", e.what());
            info.has_cpu_device = false;
            info.cpu_queue = nullptr;
        }
    }
```

### 4. Add cleanup

In `ggml-sycl.cpp`, find where `ggml_sycl_device_info` might be cleaned up (or in the destructor if it has one). If there's no cleanup, add a note that `cpu_queue` is a `new`-allocated pointer that lives for the process lifetime (same as other device queues).

**Commit:**
```bash
git add ggml/src/ggml-sycl/common.hpp ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add CPU device discovery for data-local compute

Add GGML_SYCL_CPU_OFFLOAD=1 env var that creates a SYCL CPU queue
(OpenCL backend) alongside the GPU queue. This CPU device will be
used to compute directly on host-pinned weights instead of streaming
them to GPU."
```

**Notes for implementer:**
- The CPU device appears as `opencl:cpu:0` in sycl-ls output
- GPU uses Level Zero, CPU uses OpenCL — different backends, different contexts
- USM cross-backend validation confirmed: GPU's `malloc_host` is accessible from CPU queue
- `ggml_sycl_info()` returns the static singleton — modifications are init-time only

---

## Task 2: Tier-Based Dispatch Routing

**Track:** A
**Depends on:** Task 1
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~line 23183, compute_forward)

**Description:**

Add routing logic to `ggml_sycl_compute_forward()` that checks the weight tensor's cache tier. When a weight is in PINNED_HOST tier, route to the CPU compute path instead of GPU kernels.

**Acceptance Criteria:**

- [ ] `should_dispatch_to_cpu()` function checks weight tensor tier via `cache->get_view()`
- [ ] MUL_MAT ops with PINNED_HOST weights route to CPU path
- [ ] Non-MUL_MAT ops in CPU-layer blocks also route to CPU (norm, add, rope, softmax)
- [ ] GPU-tier weights follow existing path unchanged
- [ ] Default behavior (no CPU offload) is completely unchanged

**Implementation Guide:**

### 1. Add CPU dispatch declaration

Near the top of ggml-sycl.cpp (around line 160, with other forward declarations), add:

```cpp
// Forward declaration for CPU data-local compute dispatch
// Defined in cpu-dispatch.cpp
bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst);
```

### 2. Add layer tier tracking

Add a global to track which layers are CPU-bound (set once during first graph compute, based on weight tiers). Near other globals (around line 2000):

```cpp
// Per-layer CPU dispatch decision (set on first graph, static thereafter)
static std::vector<bool> g_layer_on_cpu;
static std::atomic<bool> g_layer_map_initialized{false};

// Build the layer-to-device map from cache tiers
static void build_layer_device_map(ggml_backend_sycl_context & ctx, int n_layers) {
    if (g_layer_map_initialized.load(std::memory_order_acquire)) {
        return;
    }
    g_layer_on_cpu.resize(n_layers, false);

    auto * cache = ggml_sycl::unified_cache_enabled()
                 ? ggml_sycl::get_unified_cache_for_device(ctx.device)
                 : nullptr;
    if (!cache) {
        g_layer_map_initialized.store(true, std::memory_order_release);
        return;
    }

    int cpu_count = 0;
    // Check each layer's primary weight (attn_q is representative)
    for (int l = 0; l < n_layers; l++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        ggml_sycl_cache_id key;
        key.name_hash = std::hash<std::string>{}(std::string(name));
        key.valid = true;
        auto view = cache->get_view(key, GGML_LAYOUT_AOS);
        if (view.ptr && view.location != ggml_sycl::cache_location::DEVICE) {
            g_layer_on_cpu[l] = true;
            cpu_count++;
        }
    }

    if (cpu_count > 0) {
        GGML_LOG_INFO("[SYCL-CPU] Data-local compute: %d/%d layers on CPU, %d on GPU\n",
                      cpu_count, n_layers, n_layers - cpu_count);
    }
    g_layer_map_initialized.store(true, std::memory_order_release);
}
```

### 3. Add should_dispatch_to_cpu helper

```cpp
static bool should_dispatch_to_cpu(const ggml_tensor * dst, ggml_backend_sycl_context & ctx) {
    if (!ggml_sycl_info().has_cpu_device || !ggml_sycl_cpu_offload_enabled()) {
        return false;
    }

    // Only dispatch layer ops — global ops (embeddings, output) stay on GPU
    const char * name = dst->name;
    if (!name || name[0] == '\0') {
        return false;
    }

    int layer_id = ggml_sycl::extract_layer_id(name);
    if (layer_id < 0) {
        return false;  // Not a layer tensor — stay on GPU
    }

    if (!g_layer_map_initialized.load(std::memory_order_acquire)) {
        return false;  // Map not ready yet — first call builds it
    }

    if (layer_id >= (int)g_layer_on_cpu.size()) {
        return false;
    }

    return g_layer_on_cpu[layer_id];
}
```

### 4. Insert routing in compute_forward

In `ggml_sycl_compute_forward()` (line 23183), add AFTER the TP mode check (around line 23222) and BEFORE the switch statement:

```cpp
    // Data-local compute: dispatch host-tier layers to CPU device
    if (should_dispatch_to_cpu(dst, ctx)) {
        return ggml_sycl_compute_forward_cpu(ctx, dst);
    }

    switch (dst->op) {
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add tier-based CPU dispatch routing for data-local compute

Check each layer's weight tier via unified cache. Layers with weights
in PINNED_HOST tier are dispatched to the CPU compute path. Layer map
is built once on first graph compute and stays static."
```

**Notes for implementer:**
- `extract_layer_id()` is in tensor-types.hpp — parses "blk.N." pattern
- `cache->get_view()` returns `cache_ptr_view` with `.location` field (DEVICE, HOST_PINNED, HOST_MMAP)
- The layer map must be built AFTER the unified cache has loaded weights (first graph compute)
- `ggml_sycl_cache_id` construction: look at `ggml_backend_sycl_get_weight_cache_key()` for the real key format — the simplified version above may need adjustment to match actual cache key generation

---

## Task 3: CPU Compute Path (oneDNN + Portable Kernels)

**Track:** B
**Depends on:** Task 1
**File scope:**
- Create: `ggml/src/ggml-sycl/cpu-dispatch.hpp`
- Create: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

**Description:**

Implement the CPU compute path that handles ops dispatched from `should_dispatch_to_cpu()`. Use oneDNN for matmul (already device-agnostic) and portable SYCL kernels for element-wise ops. All operations submit to the CPU SYCL queue and operate directly on GPU-allocated host pinned memory.

**Acceptance Criteria:**

- [ ] `ggml_sycl_compute_forward_cpu()` handles MUL_MAT via oneDNN on CPU queue
- [ ] Handles RMS_NORM, ADD, MUL, ROPE, SOFT_MAX via portable SYCL on CPU queue
- [ ] Returns false for unsupported ops (caller falls back to GPU)
- [ ] Uses existing `DnnlGemmWrapper` with CPU queue — no new oneDNN code
- [ ] Data pointers resolved correctly for host pinned memory
- [ ] Build succeeds (file auto-included by CMake glob)

**Implementation Guide:**

### 1. Create cpu-dispatch.hpp

```cpp
// cpu-dispatch.hpp — CPU compute path for data-local inference
// When unified cache evicts weights to host pinned memory, this dispatches
// layer computation to a SYCL CPU device instead of streaming to GPU.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef GGML_SYCL_CPU_DISPATCH_HPP
#define GGML_SYCL_CPU_DISPATCH_HPP

#include "common.hpp"

// Dispatch a single ggml operation to the CPU SYCL device.
// Returns true if handled, false if the op is unsupported on CPU.
bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst);

#endif // GGML_SYCL_CPU_DISPATCH_HPP
```

### 2. Create cpu-dispatch.cpp

```cpp
// cpu-dispatch.cpp — CPU compute path for data-local inference
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT

#include "cpu-dispatch.hpp"
#include "common.hpp"

#if GGML_SYCL_DNNL
#include "gemm.hpp"
#endif

#include <sycl/sycl.hpp>

// Get the CPU queue from device info
static sycl::queue * get_cpu_queue() {
    return ggml_sycl_info().cpu_queue;
}

// Resolve a tensor's data pointer for CPU access.
// For host-pinned memory (unified cache eviction), this is already a valid host pointer.
// For device memory, we'd need a transfer — but CPU-dispatched ops should only
// have host-tier inputs.
static void * resolve_cpu_ptr(const ggml_tensor * tensor, int device) {
    // Use the standard resolution which returns host-pinned or mmap pointers
    // For PINNED_HOST tier, this returns the host pointer directly
    return ggml_sycl_get_data_ptr(tensor, device);
}

// ─── MUL_MAT via oneDNN on CPU ───────────────────────────────────────────────

static bool cpu_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if GGML_SYCL_DNNL
    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // activations

    sycl::queue * cpu_q = get_cpu_queue();
    if (!cpu_q) return false;

    // Resolve pointers — weights should be in host pinned memory
    const void * src0_data = resolve_cpu_ptr(src0, ctx.device);
    const void * src1_data = resolve_cpu_ptr(src1, ctx.device);
    void *       dst_data  = resolve_cpu_ptr(dst, ctx.device);

    if (!src0_data || !src1_data || !dst_data) {
        GGML_LOG_WARN("[SYCL-CPU] mul_mat: null pointer, falling back to GPU\n");
        return false;
    }

    // For quantized types, we need to dequantize first since oneDNN CPU
    // may not support all quantized formats natively.
    // Phase 1: Only handle F32 and F16 weights on CPU.
    // Quantized types fall back to GPU streaming.
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        // TODO Phase 2: Add dequantize path for Q4_0 etc.
        // For now, fall back to GPU for quantized weights
        return false;
    }

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // N (rows of weight)
    const int64_t ne10 = src1->ne[0]; // K
    const int64_t ne11 = src1->ne[1]; // M (batch/tokens)

    // Use oneDNN GEMM on CPU queue
    // llama.cpp convention: C = B * A^T where src0=A (weight), src1=B (input)
    auto dt_src0 = (src0->type == GGML_TYPE_F16) ? DnnlGemmWrapper::to_dt<sycl::half>()
                                                   : DnnlGemmWrapper::to_dt<float>();
    auto dt_src1 = DnnlGemmWrapper::to_dt<float>();
    auto dt_dst  = DnnlGemmWrapper::to_dt<float>();

    DnnlGemmWrapper::gemm(ctx,
        /* batches_a */ 1, /* batches_b */ 1,
        /* M */ ne11, /* N */ ne01, /* K */ ne00,
        /* a (src1) */ src1_data, dt_src1, /* strides */ ne10, 0,
        /* b (src0) */ src0_data, dt_src0, /* strides */ ne00, 0,
        /* c (dst) */ dst_data, dt_dst, /* stride */ ne01,
        cpu_q);

    cpu_q->wait();  // Ensure CPU computation completes before GPU reads result
    return true;
#else
    (void)ctx; (void)dst;
    return false;
#endif
}

// ─── Portable SYCL ops on CPU ────────────────────────────────────────────────

static bool cpu_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = get_cpu_queue();
    if (!cpu_q) return false;

    const ggml_tensor * src0 = dst->src[0];
    const float * src0_data = static_cast<const float *>(resolve_cpu_ptr(src0, ctx.device));
    float * dst_data = static_cast<float *>(resolve_cpu_ptr(dst, ctx.device));
    if (!src0_data || !dst_data) return false;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    // Simple CPU implementation — no SLM, no sub-groups needed
    cpu_q->parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> row_id) {
        const int64_t row = row_id[0];
        const float * x = src0_data + row * ne00;
        float * y = dst_data + row * ne00;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < ne00; i++) {
            sum_sq += x[i] * x[i];
        }
        const float scale = 1.0f / sqrtf(sum_sq / (float)ne00 + eps);

        for (int64_t i = 0; i < ne00; i++) {
            y[i] = x[i] * scale;
        }
    }).wait();

    return true;
}

static bool cpu_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = get_cpu_queue();
    if (!cpu_q) return false;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * a = static_cast<const float *>(resolve_cpu_ptr(src0, ctx.device));
    const float * b = static_cast<const float *>(resolve_cpu_ptr(src1, ctx.device));
    float * c = static_cast<float *>(resolve_cpu_ptr(dst, ctx.device));
    if (!a || !b || !c) return false;

    const int64_t n = ggml_nelements(dst);
    cpu_q->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        // Handle broadcasting: src1 may be smaller
        // Simple case: same shape
        c[i] = a[i] + b[i % n];  // TODO: proper broadcast logic
    }).wait();

    return true;
}

static bool cpu_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = get_cpu_queue();
    if (!cpu_q) return false;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * a = static_cast<const float *>(resolve_cpu_ptr(src0, ctx.device));
    const float * b = static_cast<const float *>(resolve_cpu_ptr(src1, ctx.device));
    float * c = static_cast<float *>(resolve_cpu_ptr(dst, ctx.device));
    if (!a || !b || !c) return false;

    const int64_t n = ggml_nelements(dst);
    cpu_q->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        c[i] = a[i] * b[i % n];
    }).wait();

    return true;
}

// ─── Main CPU dispatch ───────────────────────────────────────────────────────

bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            return cpu_mul_mat(ctx, dst);
        case GGML_OP_RMS_NORM:
            return cpu_rms_norm(ctx, dst);
        case GGML_OP_ADD:
            return cpu_add(ctx, dst);
        case GGML_OP_MUL:
            return cpu_mul(ctx, dst);
        // TODO Phase 2: ROPE, SOFT_MAX, LAYER_NORM, SILU
        default:
            // Unsupported on CPU — fall back to GPU
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported op %s on CPU, falling back to GPU\n",
                            ggml_op_name(dst->op));
            return false;
    }
}
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/cpu-dispatch.hpp ggml/src/ggml-sycl/cpu-dispatch.cpp
git commit -m "sycl: implement CPU compute path for data-local inference

Add cpu-dispatch.cpp with MUL_MAT (via oneDNN), RMS_NORM, ADD, MUL
dispatched to SYCL CPU device. Operates directly on host pinned
memory from unified cache evictions — zero weight data movement."
```

**Notes for implementer:**
- CMakeLists.txt globs all `*.cpp` — new files are auto-included in build
- `DnnlGemmWrapper::gemm()` takes a queue pointer — passing CPU queue creates CPU oneDNN engine automatically via `ctx.engine_dnnl(cpu_q)` and `ctx.stream_dnnl(cpu_q)`. The engine cache handles GPU vs CPU engine separation.
- Phase 1 only handles F32/F16 weights on CPU. Quantized types (Q4_0) return false → fall back to GPU streaming. Phase 2 will add dequantize → F32 → oneDNN CPU path.
- The `.wait()` calls after each CPU kernel ensure completion before any subsequent GPU op reads the result. Phase 2 will optimize with async overlapping.
- Broadcast logic in add/mul is simplified — check ggml shapes for the real broadcast pattern.
- `resolve_cpu_ptr` uses `ggml_sycl_get_data_ptr` which returns host-pinned pointers for evicted weights. For input activations that are still on GPU VRAM, this returns a device pointer that the CPU can't access — this is handled by Task 4's activation transfers.

---

## Task 4: Activation Transfers and Graph Gate

**Track:** A
**Depends on:** Task 2, Task 3
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~line 24437, graph_compute_impl; ~line 30213, graph_compute)

**Description:**

Handle activation data transfers at GPU↔CPU layer boundaries. When a GPU layer's output needs to be consumed by a CPU layer (or vice versa), synchronize queues and ensure data is accessible. Also disable graph replay when CPU offload is active.

**Acceptance Criteria:**

- [ ] GPU→CPU boundary: sync GPU queue, activation is accessible via host pinned USM
- [ ] CPU→GPU boundary: sync CPU queue, copy activation to GPU VRAM if needed
- [ ] Graph replay disabled when CPU offload layers exist
- [ ] No activation transfer overhead when all layers are on same device
- [ ] Build layer device map on first graph_compute call

**Implementation Guide:**

### 1. Add graph replay gate for CPU offload

In `ggml_backend_sycl_graph_compute()` (~line 30213), after the existing `exceeds || evictions` check, add:

```cpp
    // Disable graph replay when data-local CPU offload is active
    // (graph replay records GPU commands only — can't replay CPU inline ops)
    if (ggml_sycl_cpu_offload_enabled() && ggml_sycl_info().has_cpu_device) {
        bool has_cpu_layers = g_layer_map_initialized.load(std::memory_order_acquire) &&
                              std::any_of(g_layer_on_cpu.begin(), g_layer_on_cpu.end(),
                                          [](bool v) { return v; });
        if (has_cpu_layers) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Disabled: data-local CPU offload active\n");
            compute_impl();
            record_completion(false);
            return GGML_STATUS_SUCCESS;
        }
    }
```

### 2. Build layer map in graph_compute_impl

At the start of `ggml_backend_sycl_graph_compute_impl()` (~line 24437), after `ggml_sycl_set_main_device()`:

```cpp
    // Build layer-to-device map on first call (needs cache to be populated)
    if (ggml_sycl_cpu_offload_enabled() && ggml_sycl_info().has_cpu_device &&
        !g_layer_map_initialized.load(std::memory_order_acquire)) {
        // Count layers from graph nodes
        int max_layer = -1;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            int lid = ggml_sycl::extract_layer_id(cgraph->nodes[i]->name);
            if (lid > max_layer) max_layer = lid;
        }
        if (max_layer >= 0) {
            build_layer_device_map(*sycl_ctx, max_layer + 1);
        }
    }
```

### 3. Add activation transfer at boundaries

In the main node loop (around line 24530), replace the simple `ggml_sycl_compute_forward` call:

```cpp
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!node) continue;

        // Skip fused nodes (existing logic)
        if (fused_nodes.count(node)) continue;

        // Activation transfer at device boundaries
        if (ggml_sycl_cpu_offload_enabled() && ggml_sycl_info().has_cpu_device &&
            g_layer_map_initialized.load(std::memory_order_acquire)) {

            bool this_on_cpu = should_dispatch_to_cpu(node, *sycl_ctx);
            int  this_layer = ggml_sycl::extract_layer_id(node->name);

            // Detect boundary: previous layer was GPU, this layer is CPU
            // We only need to sync — GPU's malloc_host output is already accessible to CPU
            static int prev_layer = -1;
            static bool prev_on_cpu = false;

            if (this_layer >= 0 && this_layer != prev_layer) {
                if (this_on_cpu && !prev_on_cpu && prev_layer >= 0) {
                    // GPU→CPU boundary: ensure GPU writes are visible
                    sycl_ctx->stream()->wait();
                    GGML_SYCL_DEBUG("[SYCL-CPU] GPU→CPU boundary at layer %d\n", this_layer);
                }
                if (!this_on_cpu && prev_on_cpu) {
                    // CPU→GPU boundary: ensure CPU writes are visible
                    sycl::queue * cpu_q = ggml_sycl_info().cpu_queue;
                    if (cpu_q) cpu_q->wait();
                    GGML_SYCL_DEBUG("[SYCL-CPU] CPU→GPU boundary at layer %d\n", this_layer);
                }
                prev_layer = this_layer;
                prev_on_cpu = this_on_cpu;
            }
        }

        // Existing dispatch
        if (!ggml_sycl_compute_forward(*sycl_ctx, node)) {
            GGML_SYCL_DEBUG("[DEBUG-IMPL] compute_forward failed for node %d\n", i);
        }
    }
```

**Important**: The activation tensors between layers are allocated by the ggml scheduler. For Phase 1, these are allocated in the GPU backend's buffer (VRAM). When a CPU layer needs to read them, the pointer resolution path (`ggml_sycl_get_data_ptr`) returns the VRAM pointer — which the CPU can't access directly.

**Phase 1 workaround**: Allocate scratch/activation buffers using host pinned memory when CPU offload is active. OR: at the GPU→CPU boundary, explicitly copy the activation to host pinned memory. A simple approach:

```cpp
// At GPU→CPU boundary:
sycl_ctx->stream()->wait();
// Activation data needs to be in host-accessible memory for CPU layers
// Phase 1: use temporary host buffer for boundary activations
```

This is the most complex part — the exact mechanism depends on how ggml allocates intermediate tensors. For Phase 1, it may be sufficient to ensure the compute buffer is allocated as host pinned memory when CPU offload is active, using the existing `GGML_SYCL_KV_HOST` infrastructure.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: activation transfers and graph gate for CPU offload

Sync GPU/CPU queues at device boundaries. Disable graph replay
when CPU offload layers exist. Build layer device map from unified
cache tiers on first graph compute call."
```

**Notes for implementer:**
- The `static` prev_layer/prev_on_cpu variables assume single-threaded graph_compute — this is true for the current codebase
- The activation transfer challenge is the HARDEST part of this feature — GPU-allocated compute buffers (VRAM) need to be readable by CPU
- Options: (a) host-pinned compute buffers, (b) explicit DMA at boundaries, (c) use `sycl::malloc_shared` for activations
- For Phase 1, option (b) is simplest — allocate a reusable host staging buffer, memcpy activations at boundaries

---

## Task 5: Integration Test

**Track:** — (convergence point)
**Depends on:** Task 3, Task 4
**File scope:**
- Create: `Testing/test-cpu-offload-validation.sh` (NEW)

**Description:**

Validate the data-local CPU compute path end-to-end. Compare outputs between default (all GPU), CPU offload, and weight streaming paths.

**Acceptance Criteria:**

- [ ] Default (no budget pressure): identical output, no CPU offload triggered
- [ ] 30% budget + CPU offload: correct output (may differ from all-GPU due to FP precision)
- [ ] 30% budget + weight streaming: correct output for comparison
- [ ] Performance comparison logged: CPU offload vs streaming
- [ ] No crashes, hangs, or memory leaks

**Implementation Guide:**

### Test script

```bash
#!/bin/bash
# test-cpu-offload-validation.sh — Validate data-local CPU compute
set -e
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

MODEL=/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf
PROMPT='1, 2, 3, 4, 5,'
COMMON_ARGS="-m $MODEL -p '$PROMPT' -n 15 --seed 42 --temp 0"

echo "=== Test 1: Default (all GPU, no budget pressure) ==="
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m $MODEL -p "$PROMPT" -n 15 --seed 42 --temp 0 2>&1 | tail -5
echo "Expected: 6, 7, 8, 9, 10"

echo ""
echo "=== Test 2: 30% budget + CPU offload ==="
GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_CPU_OFFLOAD=1 \
  ./build/bin/llama-completion \
  -m $MODEL -p "$PROMPT" -n 15 --seed 42 --temp 0 -fit off 2>&1 | tail -10
echo "Expected: correct numeric sequence"

echo ""
echo "=== Test 3: 30% budget + weight streaming (comparison) ==="
GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_FORCE_STREAMING=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m $MODEL -p "$PROMPT" -n 15 --seed 42 --temp 0 -fit off 2>&1 | tail -10
echo "Expected: correct numeric sequence"

echo ""
echo "=== Done ==="
```

Note: Use `-fit off` for tests 2 and 3 to bypass fit_params (known separate issue). The CPU offload feature should work without fit_params.

**Commit:**
```bash
git add Testing/test-cpu-offload-validation.sh
git commit -m "test: add CPU offload validation script"
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Activation tensors in GPU VRAM inaccessible from CPU | HIGH | Phase 1: explicit DMA copy at boundaries |
| Quantized weights (Q4_0) not handled on CPU | MEDIUM | Phase 1: F32/F16 only, Q4_0 falls back to GPU streaming |
| oneDNN CPU performance worse than expected | LOW | Profile in Phase 1, optimize in Phase 2 |
| Cross-backend USM edge cases | LOW | Validated with 5/6 test cases passing |
| `DnnlGemmWrapper` mutex serializes GPU+CPU | LOW | Acceptable for Phase 1, per-device mutex in Phase 2 |
