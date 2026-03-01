# Token Generation (TG) Performance Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve Token Generation throughput from ~16 t/s to ~40 t/s on Intel Arc B580

**Architecture:** Two complementary optimization strategies:
1. **SYCL Graph Optimization** - Reduce kernel launch overhead via graph recording/replay
2. **Persistent Kernel** - Eliminate launch overhead entirely with single long-running kernel

Both use the unified cache for all memory management.

**Tech Stack:** SYCL 2020, Intel oneAPI, ESIMD intrinsics, Unified Kernel, Unified Cache

---

## Performance Context

| Metric | Value | Notes |
|--------|-------|-------|
| Current TG | ~16 t/s | DMMV kernel path |
| Target TG | ~40 t/s | Previously achieved |
| Theoretical max | ~111 t/s | 456 GB/s ÷ 4.1 GB weights |
| Kernel launches/forward | 280-392 | 10-14 kernels × 28 layers |
| Estimated launch overhead | 2.8-7.8 ms | 10-20 µs × 280-392 launches |
| Launch overhead % at target | 11-31% | At 40 t/s = 25ms/token |

---

## Existing Infrastructure

### Unified Kernel (`unified-kernel.hpp/cpp`)
- **Dispatch paths**: DMMV (M=1), MMVQ (M<8), ESIMD_DPAS (M≥8), ESIMD_LARGE_TILE (M≥128)
- **Entry point**: `launch_unified_matmul(queue, UnifiedKernelArgs)`
- **Tile configuration**: From TuningEngine via `TunedParams`
- **Layout support**: AoS, SoA, COALESCED, XMX_COALESCED

### Unified Cache (`unified-cache.hpp`)
- **Three-tier memory**: Device VRAM → Pinned Host → Mmap
- **Key APIs**:
  - `ensure_cached()` - Synchronous load with layout conversion
  - `try_get_cached_fast()` - Fast path with reader-writer lock
  - `ensure_cached_layout()` - Async fill with graph dependency tracking
  - `pin()/unpin()` - Prevent eviction during kernel execution
- **MoE support**: `prestage_routed_experts()`, `unpin_routed_experts()`
- **Scratch buffers**: `reserve_onednn_scratch()`, `get_onednn_scratch()`

### SYCL Graph Infrastructure
- **Graph storage**: `common.hpp:2339-2346` defines `exec_graph`, tracking fields
- **Graph recording**: `ggml-sycl.cpp:24727-24770` implements record/finalize flow
- **Graph replay**: `ggml-sycl.cpp:24715-24720` uses `ext_oneapi_graph()`
- **Warmup tracking**: `warmup_decode_n_nodes`, `warmup_prompt_n_nodes`

### Existing Persistent Patterns
- **MoE ESIMD kernel**: `fused-moe-esimd.hpp:375-583` - Work-stealing, cooperative processing
- **MoE XMX fused**: `moe-xmx-fused.hpp:25-700` - Layer-persistent for MoE blocks
- **Split barriers**: `GGML_SYCL_SPLIT_BARRIER_SUPPORT` for cooperative sync

---

## Part 1: SYCL Graph Optimization

### Current State Analysis
- Graph infrastructure EXISTS and IS functional
- Graph replay code at `:24715-24720` uses `ext_oneapi_graph()`
- Commented code at `:24524-24542` is old UPDATE logic (not replay)
- May not be triggering for TG due to warmup or node count issues

### Tasks

#### Task 1.1: Instrument Graph State Logging

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:24700-24800`

**Step 1: Add graph state logging**

Add logging after graph execution decision point:

```cpp
// Around line 24710, after checking graph conditions
if (ctx.graphs_disabled) {
    GGML_LOG_DEBUG("SYCL Graph: disabled globally\n");
} else if (!ctx.exec_graph) {
    GGML_LOG_DEBUG("SYCL Graph: no executable graph cached\n");
} else if (ctx.exec_graph_n_nodes != n_nodes) {
    GGML_LOG_DEBUG("SYCL Graph: node count mismatch (cached=%d, current=%d)\n",
                   ctx.exec_graph_n_nodes, n_nodes);
} else if (ctx.exec_graph_is_decode != is_decode) {
    GGML_LOG_DEBUG("SYCL Graph: phase mismatch (cached=%s, current=%s)\n",
                   ctx.exec_graph_is_decode ? "decode" : "prompt",
                   is_decode ? "decode" : "prompt");
} else {
    GGML_LOG_DEBUG("SYCL Graph: replaying cached graph (%d nodes)\n", n_nodes);
}
```

**Step 2: Rebuild and test**

Run: `ninja -C build ggml-sycl`

**Step 3: Test with debug output**

Run: `GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 32 2>&1 | grep -i graph`

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: Add graph state debug logging for TG optimization"
```

---

#### Task 1.2: Fix Graph Warmup for TG Path

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Verify warmup runs for decode phase**

Search for warmup logic and ensure it handles TG (decode) separately from PP (prompt).

**Step 2: Add TG-specific warmup if missing**

```cpp
// During backend initialization or first decode
if (ctx.warmup_decode_n_nodes == 0 && is_decode) {
    // Run warmup iterations for decode graph
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        compute_impl(ctx, cgraph, is_decode);
    }
    ctx.warmup_decode_n_nodes = n_nodes;
}
```

**Step 3: Test graph capture after warmup**

Run: `GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 64`

**Step 4: Benchmark improvement**

Run: `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128 -r 3`

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: Fix graph warmup for decode (TG) phase"
```

---

#### Task 1.3: Stabilize Graph Node Count

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Add node count tracking**

```cpp
// Track node counts across iterations
static thread_local int last_decode_nodes = 0;
if (is_decode && n_nodes != last_decode_nodes) {
    GGML_LOG_DEBUG("SYCL Graph: decode node count changed %d -> %d\n",
                   last_decode_nodes, n_nodes);
    last_decode_nodes = n_nodes;
}
```

**Step 2: Identify variance sources**

Common causes: dynamic tensor shapes, conditional dispatch, debug kernels.

**Step 3: Fix or tolerate variance**

If small (±1-2 nodes), consider hash-based caching instead of node count.

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: Stabilize graph node count for reliable replay"
```

---

## Part 2: Persistent Kernel Architecture

### Design Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PERSISTENT TG KERNEL                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Work-Group 0 │    │ Work-Group 1 │    │ Work-Group N │   │
│  │  (tile 0,0)  │    │  (tile 0,1)  │    │  (tile M,N)  │   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              PERSISTENT LOOP                         │    │
│  │  for layer = 0..n_layers:                           │    │
│  │    for op in [attn_norm, qkv, rope, attn, out,      │    │
│  │               ffn_norm, gate_up, silu, down]:       │    │
│  │      tile_idx = atomic_fetch_add(&work_counter)     │    │
│  │      compute_tile(layer, op, tile_idx)              │    │
│  │      split_barrier_sync()                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Memory Management via Unified Cache:                        │
│  - Weights: pre-pinned via cache->pin() before launch        │
│  - Activations: intermediate buffers from scratch pool       │
│  - Output: final token logits                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Memory Layout

```
Unified Cache Allocations:
├─ Weight Tensors (pinned for kernel lifetime)
│  ├─ Layer 0: attn_norm, q_proj, k_proj, v_proj, o_proj
│  │           ffn_norm, gate_proj, up_proj, down_proj
│  ├─ Layer 1: ...
│  └─ Layer N: ...
│
├─ Persistent Scratch Buffers (new allocation type)
│  ├─ Intermediate activations: [n_layers × hidden_dim × sizeof(half)]
│  ├─ KV cache pointers: [n_layers × 2 × sizeof(void*)]
│  └─ Work counter: sizeof(atomic<int>)
│
└─ Output Buffer
   └─ Logits: [vocab_size × sizeof(float)]
```

### Tasks

#### Task 2.1: Add Persistent Scratch API to Unified Cache

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp`

**Step 1: Read current scratch buffer implementation**

Read `unified-cache.hpp` to understand existing `reserve_onednn_scratch()` pattern.

**Step 2: Add persistent scratch allocation API**

```cpp
// In unified_cache class:

// Reserve scratch buffer for persistent kernels
bool reserve_persistent_scratch(
    const std::string& buffer_name,
    size_t size_bytes,
    bool pin = true
);

// Get persistent scratch buffer pointer
void* get_persistent_scratch(const std::string& buffer_name);

// Release persistent scratch buffer
void release_persistent_scratch(const std::string& buffer_name);

private:
    struct persistent_scratch_entry {
        void* device_ptr;
        size_t size;
        bool pinned;
    };
    std::unordered_map<std::string, persistent_scratch_entry> persistent_scratches_;
    std::mutex persistent_scratch_mutex_;
```

**Step 3: Implement allocation from device pool**

```cpp
bool unified_cache::reserve_persistent_scratch(
    const std::string& buffer_name,
    size_t size_bytes,
    bool pin
) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    if (persistent_scratches_.count(buffer_name)) {
        auto& entry = persistent_scratches_[buffer_name];
        if (entry.size >= size_bytes) {
            return true;
        }
        release_persistent_scratch_impl(buffer_name);
    }

    void* ptr = device_alloc(size_bytes);
    if (!ptr) {
        evict_for_size(size_bytes);
        ptr = device_alloc(size_bytes);
    }

    if (!ptr) {
        GGML_LOG_ERROR("Failed to allocate persistent scratch '%s' (%zu bytes)\n",
                       buffer_name.c_str(), size_bytes);
        return false;
    }

    persistent_scratches_[buffer_name] = {ptr, size_bytes, pin};
    return true;
}
```

**Step 4: Test scratch allocation**

Run: `ninja -C build ggml-sycl && ./build/bin/test-backend-ops -o SYCL0`

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp
git commit -m "sycl: Add persistent scratch buffer API to unified cache"
```

---

#### Task 2.2: Add Weight Pre-Pinning Infrastructure

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp`

**Step 1: Add bulk pinning API**

```cpp
struct layer_weight_set {
    ggml_sycl_cache_id attn_norm;
    ggml_sycl_cache_id q_proj, k_proj, v_proj, o_proj;
    ggml_sycl_cache_id ffn_norm;
    ggml_sycl_cache_id gate_proj, up_proj, down_proj;
};

int pin_layer_weights(int layer_id, const layer_weight_set& weights, ggml_layout_mode layout);
void unpin_layer_weights(int layer_id);
int pin_model_weights(int n_layers, const std::vector<layer_weight_set>& layers, ggml_layout_mode layout);
```

**Step 2: Implement using existing pin/unpin**

```cpp
int unified_cache::pin_layer_weights(
    int layer_id,
    const layer_weight_set& weights,
    ggml_layout_mode layout
) {
    int pinned = 0;
    auto try_pin = [&](const ggml_sycl_cache_id& key) {
        if (key.valid && try_get_cached_fast(key, layout)) {
            pin(key, layout);
            pinned++;
        }
    };

    try_pin(weights.attn_norm);
    try_pin(weights.q_proj);
    try_pin(weights.k_proj);
    try_pin(weights.v_proj);
    try_pin(weights.o_proj);
    try_pin(weights.ffn_norm);
    try_pin(weights.gate_proj);
    try_pin(weights.up_proj);
    try_pin(weights.down_proj);

    return pinned;
}
```

**Step 3: Test pinning**

Verify cache eviction still works for non-pinned entries.

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp
git commit -m "sycl: Add bulk weight pinning for persistent kernels"
```

---

#### Task 2.3: Create Persistent TG Kernel Header

**Files:**
- Create: `ggml/src/ggml-sycl/persistent-tg-kernel.hpp`

**Step 1: Define argument structures**

```cpp
#pragma once

#include "common.hpp"
#include "unified-kernel.hpp"

namespace ggml_sycl {

struct PersistentTGArgs {
    int n_layers;
    int hidden_dim;
    int n_heads;
    int head_dim;
    int intermediate_dim;
    int vocab_size;
    int quant_type;
    ggml_layout_mode layout;

    struct LayerWeights {
        const void* attn_norm;
        const void* q_proj, *k_proj, *v_proj, *o_proj;
        const void* ffn_norm;
        const void* gate_proj, *up_proj, *down_proj;
    };
    const LayerWeights* layer_weights;

    struct KVCache {
        void* k_cache;
        void* v_cache;
        int seq_len;
    };
    KVCache* kv_caches;

    const float* input_embedding;
    float* output_logits;
    void* intermediate_buffer;
    int* work_counter;
    int total_tiles;
};

struct PersistentTGConfig {
    int tile_m, tile_n, tile_k;
    int n_workgroups;
    int workgroup_size;
    bool use_split_barriers;
};

sycl::event launch_persistent_tg_kernel(sycl::queue& q, const PersistentTGArgs& args, const PersistentTGConfig& config);
bool can_use_persistent_tg(int n_layers, int hidden_dim, int quant_type, const XMXConfig& xmx_config);

}  // namespace ggml_sycl
```

**Step 2: Commit**

```bash
git add ggml/src/ggml-sycl/persistent-tg-kernel.hpp
git commit -m "sycl: Add persistent TG kernel header"
```

---

#### Task 2.4: Implement Persistent Kernel Core Loop

**Files:**
- Create: `ggml/src/ggml-sycl/persistent-tg-kernel.cpp`

**Step 1: Implement work-stealing persistent loop**

```cpp
#include "persistent-tg-kernel.hpp"

namespace ggml_sycl {

template<int TILE_N, int TILE_K>
class PersistentDMMVKernel {
public:
    PersistentDMMVKernel(const PersistentTGArgs& args, sycl::local_accessor<float, 1> slm, sycl::nd_item<1> item)
        : args_(args), slm_(slm), item_(item) {}

    void run() {
        const int local_id = item_.get_local_linear_id();

        while (true) {
            int work_idx;
            if (local_id == 0) {
                work_idx = sycl::atomic_ref<int,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(
                        *args_.work_counter).fetch_add(1);
            }
            work_idx = sycl::group_broadcast(item_.get_group(), work_idx, 0);

            if (work_idx >= args_.total_tiles) break;

            int layer, op, tile;
            decode_work_item(work_idx, layer, op, tile);
            dispatch_operation(layer, op, tile);

            item_.barrier(sycl::access::fence_space::global_and_local);
        }
    }

private:
    static constexpr int OP_ATTN_NORM = 0, OP_QKV_PROJ = 1, OP_ATTN = 2, OP_OUT_PROJ = 3;
    static constexpr int OP_FFN_NORM = 4, OP_GATE_UP = 5, OP_SILU = 6, OP_DOWN = 7;
    static constexpr int OPS_PER_LAYER = 8;

    void decode_work_item(int work_idx, int& layer, int& op, int& tile);
    void dispatch_operation(int layer, int op, int tile);

    const PersistentTGArgs& args_;
    sycl::local_accessor<float, 1> slm_;
    sycl::nd_item<1> item_;
};

sycl::event launch_persistent_tg_kernel(sycl::queue& q, const PersistentTGArgs& args, const PersistentTGConfig& config) {
    const int slm_size = config.tile_n * config.tile_k * sizeof(float);

    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm(slm_size / sizeof(float), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(config.n_workgroups * config.workgroup_size, config.workgroup_size),
            [=](sycl::nd_item<1> item) {
                PersistentDMMVKernel<64, 32> kernel(args, slm, item);
                kernel.run();
            }
        );
    });
}

}  // namespace ggml_sycl
```

**Step 2: Test compilation**

Run: `ninja -C build ggml-sycl`

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/persistent-tg-kernel.cpp
git commit -m "sycl: Add persistent TG kernel core loop"
```

---

#### Task 2.5: Integrate with Dispatch System

**Files:**
- Modify: `ggml/src/ggml-sycl/dispatch.hpp`
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Add persistent TG dispatch path**

```cpp
// In dispatch.hpp kernel selection
if (is_decode_phase && batch_size == 1 &&
    can_use_persistent_tg(n_layers, hidden_dim, quant_type, xmx_config) &&
    env_persistent_tg_enabled()) {
    return KernelType::PERSISTENT_TG;
}
```

**Step 2: Add integration in ggml-sycl.cpp**

```cpp
case KernelType::PERSISTENT_TG: {
    PersistentTGArgs args = build_persistent_tg_args(ctx, cgraph);
    PersistentTGConfig config = get_persistent_tg_config(ctx);

    pin_model_weights_for_persistent(ctx, args);
    sycl::event e = launch_persistent_tg_kernel(ctx.stream(), args, config);
    e.wait();
    unpin_model_weights(ctx);
    return;
}
```

**Step 3: Test dispatch**

Run: `GGML_SYCL_DEBUG=1 GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 16`

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/dispatch.hpp ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: Integrate persistent TG kernel with dispatch"
```

---

## Part 3: Validation and Optimization

#### Task 3.1: Validate Graph Optimization Correctness

**Files:** None (testing)

**Step 1: Baseline output**

```bash
GGML_SYCL_GRAPH=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > baseline_no_graph.txt
```

**Step 2: Graph-enabled output**

```bash
GGML_SYCL_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > with_graph.txt
```

**Step 3: Compare**

```bash
diff baseline_no_graph.txt with_graph.txt
```

---

#### Task 3.2: Validate Persistent Kernel Correctness

**Files:** None (testing)

**Step 1: Baseline output**

```bash
GGML_SYCL_PERSISTENT_TG=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > baseline_no_persistent.txt
```

**Step 2: Persistent kernel output**

```bash
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > with_persistent.txt
```

**Step 3: Compare**

```bash
diff baseline_no_persistent.txt with_persistent.txt
```

---

#### Task 3.3: Profile and Benchmark Both Approaches

**Files:** None (profiling)

**Step 1: Benchmark graph optimization**

```bash
GGML_SYCL_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128 -r 3
```

**Step 2: Benchmark persistent kernel**

```bash
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128 -r 3
```

**Step 3: Compare launch counts**

```bash
ZE_ENABLE_TRACING_LAYER=1 GGML_SYCL_GRAPH=0 GGML_SYCL_PERSISTENT_TG=0 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 8 2>&1 | \
  grep -c zeKernel
```

---

#### Task 3.4: Add Split Barrier Optimization

**Files:**
- Modify: `ggml/src/ggml-sycl/persistent-tg-kernel.cpp`

**Step 1: Replace full barriers with split barriers**

```cpp
#ifdef GGML_SYCL_SPLIT_BARRIER_SUPPORT
    split_barrier_arrive(item_);
    split_barrier_wait(item_);
#else
    item_.barrier(sycl::access::fence_space::global_and_local);
#endif
```

**Step 2: Test and benchmark**

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/persistent-tg-kernel.cpp
git commit -m "sycl: Use split barriers in persistent TG kernel"
```

---

## Task Dependencies

```
Part 1: SYCL Graph Optimization
  Task 1.1 (Graph Logging)
    └─► Task 1.2 (Graph Warmup Fix) ─► depends on 1.1
        └─► Task 1.3 (Node Count Stabilization) ─► depends on 1.2

Part 2: Persistent Kernel
  Task 2.1 (Persistent Scratch API)
    └─► Task 2.2 (Weight Pinning) ─► depends on 2.1
        └─► Task 2.3 (Kernel Header) ─► depends on 2.2
            └─► Task 2.4 (Kernel Implementation) ─► depends on 2.3
                └─► Task 2.5 (Dispatch Integration) ─► depends on 2.4

Part 3: Validation (depends on Parts 1 & 2)
  Task 3.1 (Graph Validation) ─► depends on 1.3
  Task 3.2 (Persistent Validation) ─► depends on 2.5
  Task 3.3 (Benchmarking) ─► depends on 3.1, 3.2
  Task 3.4 (Split Barrier Opt) ─► depends on 3.2
```

---

## Success Criteria

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| TG (M=1) | 16 t/s | 25 t/s | 40 t/s |
| Kernel launches/token | 280-392 | <50 (graph) / 1 (persistent) | 1 |
| Launch overhead | 2.8-7.8 ms | <1 ms | ~0 ms |
| Correctness | - | 100% | 100% |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Graph recording breaks | Medium | High | Comprehensive warmup, fallback |
| Persistent kernel occupancy | Medium | High | Tune workgroup count |
| Synchronization deadlocks | Medium | High | Careful barrier placement |
| Numerical accuracy | Low | Medium | FP32 accumulators, validation |

---

## References

- `ggml/src/ggml-sycl/unified-kernel.hpp` - Unified kernel dispatch
- `ggml/src/ggml-sycl/unified-cache.hpp` - Cache API
- `ggml/src/ggml-sycl/ggml-sycl.cpp:24700-24800` - Graph execution flow
- `ggml/src/ggml-sycl/fused-moe-esimd.hpp:375-583` - Persistent pattern
- `ggml/src/ggml-sycl/common.hpp:2339-2346` - Graph state
