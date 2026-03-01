# TG Performance Recovery: Closing the 5x Gap to Master

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Close the 5x TG performance gap (15.73 vs 76 tok/s) by fixing kernel selection, streamlining graph_compute, and eliminating per-token overhead — while preserving the unified cache and unified kernel architecture.

**Architecture:** Three parallel optimization tracks: (A) fix kernel selection mismatch so batch=1 uses MMVQ+SOA instead of unified DMMV, (B) cache expensive O(n_nodes) computations in graph_compute to make the replay path near-zero-overhead, (C) restructure data pointer resolution as a hot/cold split to eliminate get_pointer_type() driver round-trips.

**Tech Stack:** C++17, Intel SYCL (icpx), oneAPI Level Zero, oneDNN

---

## Team Topology

**Recommended implementers:** 3 (based on 3 parallel tracks after diagnostics)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 4 | Diagnostics + TG kernel fast-path in mul_mat dispatch |
| B | 2 | graph_compute streamlining (cached checks, pre-cached input tensors) |
| C | 3 | Data pointer O(1) fast path + get_pointer_type elimination |
| — | 5 | Integration benchmark (convergence point) |

### Dependency Graph

```
digraph dependencies {
    rankdir=LR;
    1 [label="Task 1: Diagnostics"];
    2 [label="Task 2: Graph Streamlining"];
    3 [label="Task 3: Pointer Fast Path"];
    4 [label="Task 4: TG Kernel Parity"];
    5 [label="Task 5: Benchmark"];
    1 -> 4;
    2 -> 5;
    3 -> 5;
    4 -> 5;
}
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 23158-23180 | 1 | None (add timing to synchronize) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29350-29940 | 1, 2 | Sequential (same track concept, but Task 1 adds timing, Task 2 adds caching — **different sub-regions**) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 19073-19430 | 4 | None (single task) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 19490-19494 | 3 | None (single task) |
| `ggml/src/ggml-sycl/common.hpp` lines 1707-1820 | 3 | None (single task) |
| `ggml/src/ggml-sycl/common.hpp` lines 2169+ | 2 | None (single task, adds context fields) |

---

## Background (for implementers with zero context)

### What is this codebase?

`llama.cpp` is a C/C++ inference engine for large language models. The SYCL backend (`ggml/src/ggml-sycl/`) runs on Intel GPUs. The branch `feature/sycl-coalescing` adds a "unified cache" (tiered VRAM/host/mmap memory) and a "unified kernel" (auto-tuned matmul dispatch) that are architecturally important but introduced a 5x TG regression.

### Key terms

- **TG (text generation)**: Generating one token at a time (batch=1 matmul). Target: 76 tok/s.
- **PP (prompt processing)**: Processing many tokens at once (batch=512). Currently 1303 tok/s — no regression.
- **MMVQ**: Matrix-multiply-vector-quantized — operates directly on quantized Q4_0 data. Fast for batch=1.
- **DMMV**: Dequantize-then-multiply — converts Q4_0 to FP32 first. Slower for batch=1 on Intel GPUs.
- **SOA reorder**: Structure-of-arrays layout for quantized weights. Improves memory coalescing.
- **Graph replay**: SYCL records GPU commands once, replays on subsequent tokens (avoids per-node dispatch).
- **Unified cache**: Tiered memory system (VRAM → pinned host → mmap) for models that exceed VRAM.

### Build commands

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
ninja -C build -j $(nproc)
```

**Build time**: ~10 minutes with ccache. Always source oneAPI before building or running.

### Test commands

```bash
source /opt/intel/oneapi/setvars.sh --force

# Correctness (must match expected output)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected output includes: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

# Performance benchmark
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Targets: PP512 >= 1300 tok/s, TG128 >= 70 tok/s

# WARNING: ONEAPI_DEVICE_SELECTOR=level_zero:0 is REQUIRED (multi-GPU hangs without it)
```

---

### Task 1: Diagnostic Timing — Measure GPU Execution Time vs CPU Overhead

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 23158-23180 (synchronize function)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29392-29440 (graph_compute entry)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29927-29937 (graph refresh + replay)

**Description:**

The existing timing instrumentation only covers 3 sections (finalize_layouts, graph compatibility, graph replay). There are UNTIMED gaps where overhead may hide. More critically, we don't measure the actual GPU execution time (how long `wait_and_throw()` blocks). This task adds complete timing to answer: "Is the 5x gap from slower GPU kernels or CPU-side overhead?"

**Acceptance Criteria:**

- [ ] `ggml_backend_sycl_synchronize` prints `wait_and_throw` duration every 50th call when `GGML_SYCL_DEBUG=1`
- [ ] `graph_compute` prints total function time (entry to return) every 50th call
- [ ] `graph_refresh_input_tensors` duration is measured and included in the timing output
- [ ] Output format: `[TG-DIAG] call #N: total=%.2fms sync=%.2fms refresh=%.2fms replay=%.2fms`
- [ ] No timing overhead when `GGML_SYCL_DEBUG` is not set (compile-time or flag check)
- [ ] Build succeeds, correctness test passes
- [ ] Diagnostic output collected for TG128 benchmark run

**Implementation Guide:**

1. **Add timing to `ggml_backend_sycl_synchronize`** (line 23158):

```cpp
static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    const queue_ptr             stream   = sycl_ctx->stream(sycl_ctx->device, 0);

    // Timing: measure actual GPU execution time (how long we wait for GPU to finish)
    static int sync_call_count = 0;
    sync_call_count++;
    auto t_sync_start = std::chrono::high_resolution_clock::now();

    // Existing conditional mutex logic for graph recording
    std::unique_lock<std::mutex> graph_lock(g_sycl_graph_compute_mutex, std::defer_lock);
    if (ggml_sycl_graph_recording_active()) {
        graph_lock.lock();
    }
    stream->wait_and_throw();

    if (g_ggml_sycl_debug && sync_call_count % 50 == 0) {
        double sync_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_sync_start).count();
        GGML_SYCL_DEBUG("[TG-DIAG] synchronize call #%d: wait_and_throw=%.2fms\n",
                        sync_call_count, sync_ms);
    }
    GGML_UNUSED(backend);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}
```

Run: `ninja -C build -j $(nproc)` — should compile. No test yet (need full build).

2. **Add total function timing to `graph_compute`** (line 29392):

Right after the opening brace of `ggml_backend_sycl_graph_compute`, add:
```cpp
    auto t_graph_compute_entry = std::chrono::high_resolution_clock::now();
```

Before each `return GGML_STATUS_SUCCESS;` (there are multiple return points), add:
```cpp
    if (g_ggml_sycl_debug) {
        static int gc_call_count = 0;
        gc_call_count++;
        if (gc_call_count % 50 == 0) {
            double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_graph_compute_entry).count();
            GGML_SYCL_DEBUG("[TG-DIAG] graph_compute call #%d: total=%.2fms\n",
                            gc_call_count, total_ms);
        }
    }
```

**Note:** There are ~8 return points in graph_compute. Use a helper lambda or RAII guard to avoid duplication:

```cpp
    // At the top of graph_compute, after t_graph_compute_entry:
    static int gc_diag_count = 0;
    gc_diag_count++;
    struct GCTimingGuard {
        int call_num;
        std::chrono::high_resolution_clock::time_point start;
        double refresh_ms = 0;
        ~GCTimingGuard() {
            if (g_ggml_sycl_debug && call_num % 50 == 0) {
                double total_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - start).count();
                GGML_SYCL_DEBUG("[TG-DIAG] graph_compute #%d: total=%.2fms refresh=%.2fms\n",
                                call_num, total_ms, refresh_ms);
            }
        }
    } gc_timing{gc_diag_count, t_graph_compute_entry};
```

3. **Add timing around `graph_refresh_input_tensors`** (line 29929):

```cpp
    // Before:
    graph_refresh_input_tensors(sycl_ctx, cgraph);

    // After:
    {
        auto t_refresh = std::chrono::high_resolution_clock::now();
        graph_refresh_input_tensors(sycl_ctx, cgraph);
        gc_timing.refresh_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_refresh).count();
    }
```

4. **Build and run diagnostic benchmark:**

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Correctness
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Diagnostic TG run
GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 0 -n 128 > /tmp/tg-diag.txt 2>&1

# Extract diagnostics
grep 'TG-DIAG' /tmp/tg-diag.txt
```

**Expected output format:**
```
[TG-DIAG] graph_compute #50: total=0.85ms refresh=0.15ms
[TG-DIAG] synchronize call #50: wait_and_throw=62.30ms
```

If `wait_and_throw` is ~60ms, the GPU kernels themselves are slow (confirms kernel mismatch).
If `wait_and_throw` is ~12ms but total graph_compute is much higher, the CPU overhead is the bottleneck.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add diagnostic timing to synchronize and graph_compute

Measure wait_and_throw duration to determine if TG gap is GPU execution
time or CPU overhead. Prints every 50th call with GGML_SYCL_DEBUG=1."
```

**Notes for implementer:**
- The `<chrono>` header is already included at line 22 of ggml-sycl.cpp
- `g_ggml_sycl_debug` is the global debug flag, already available
- IMPORTANT: Do NOT add `.wait()` calls anywhere — we're measuring, not changing behavior
- There are ~8 return points in `graph_compute`. The RAII GCTimingGuard handles them all automatically.
- Redirect GGML_SYCL_DEBUG output to file — it produces 16K+ lines for PP512 and 78K+ for TG128

---

### Task 2: Cache Expensive O(n) Computations in graph_compute

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/common.hpp` lines 2169+ (add fields to `ggml_backend_sycl_context`)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 25306-25357 (`graph_refresh_input_tensors`)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29350-29390 (`should_use_persistent_tg`)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29460-29470 (persistent TG check call site)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29648-29690 (graph hash + phase detection)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29927-29937 (graph replay path)

**Description:**

For steady-state TG with graph replay, `graph_compute` runs 5+ O(n_nodes) scans per token: `should_use_persistent_tg` (3 internal scans), `ggml_sycl_graph_signature` (~41K hash ops for 350 nodes), phase detection, and `graph_refresh_input_tensors` (iterates all nodes × GGML_MAX_SRC). Since graph topology is stable during TG, cache all of these and only recompute when n_nodes changes.

**Acceptance Criteria:**

- [ ] `should_use_persistent_tg` result cached; only recomputed when `cgraph->n_nodes` changes
- [ ] `ggml_sycl_graph_signature` hash cached; only recomputed when n_nodes changes or exec_graph is null
- [ ] Phase detection (decode vs prompt) cached alongside persistent_tg result
- [ ] `graph_refresh_input_tensors` uses pre-cached input tensor list instead of full graph scan
- [ ] Input tensor list populated during graph recording, reused on all subsequent replays
- [ ] Build succeeds, correctness test passes, PP512 >= 1300 tok/s (no regression)

**Implementation Guide:**

1. **Add caching fields to `ggml_backend_sycl_context`:**

In `ggml/src/ggml-sycl/common.hpp`, find `struct ggml_backend_sycl_context` (around line 2169). Add these fields alongside the existing `layouts_finalized` fields:

```cpp
    // === Cached per-graph computations (reset when n_nodes changes) ===
    int  cached_persistent_n_nodes = -1;   // n_nodes when persistent check was cached
    bool cached_persistent_result  = false; // cached should_use_persistent_tg result
    bool cached_is_decode_phase    = false; // cached phase detection result

    uint64_t cached_graph_sig         = 0;  // cached graph signature hash
    int      cached_graph_sig_n_nodes = -1; // n_nodes when hash was cached

    // Pre-cached input tensor set for graph_refresh (populated during recording)
    std::vector<ggml_tensor *> cached_input_tensors;
    bool input_tensors_cached = false;
```

Run: `ninja -C build -j $(nproc)` — verify compile.

2. **Cache `should_use_persistent_tg` result:**

At line ~29464 in `ggml_backend_sycl_graph_compute`, replace:
```cpp
    // BEFORE:
    bool persistent_eligible = should_use_persistent_tg(*sycl_ctx, cgraph);
```

With:
```cpp
    // AFTER: Cache persistent TG check (3x O(n) scans avoided on steady-state TG)
    bool persistent_eligible;
    bool is_decode = false;
    if (sycl_ctx->cached_persistent_n_nodes == cgraph->n_nodes) {
        persistent_eligible = sycl_ctx->cached_persistent_result;
        is_decode = sycl_ctx->cached_is_decode_phase;
    } else {
        persistent_eligible = should_use_persistent_tg(*sycl_ctx, cgraph);
        // Phase detection: find first MUL_MAT and check ne[1]
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && cgraph->nodes[i]->src[1]) {
                is_decode = (cgraph->nodes[i]->src[1]->ne[1] == 1);
                break;
            }
        }
        sycl_ctx->cached_persistent_n_nodes = cgraph->n_nodes;
        sycl_ctx->cached_persistent_result  = persistent_eligible;
        sycl_ctx->cached_is_decode_phase    = is_decode;
    }
```

3. **Cache graph signature hash:**

At line ~29649 where `ggml_sycl_graph_signature` is called, replace:
```cpp
    // BEFORE:
    const uint64_t graph_hash = ggml_sycl_graph_signature(cgraph);
```

With:
```cpp
    // AFTER: Cache graph hash (saves ~41K FNV-1a hash mix ops per token)
    uint64_t graph_hash;
    if (sycl_ctx->cached_graph_sig_n_nodes == cgraph->n_nodes && sycl_ctx->exec_graph) {
        graph_hash = sycl_ctx->cached_graph_sig;
    } else {
        graph_hash = ggml_sycl_graph_signature(cgraph);
        sycl_ctx->cached_graph_sig = graph_hash;
        sycl_ctx->cached_graph_sig_n_nodes = cgraph->n_nodes;
    }
```

4. **Use cached `is_decode` for phase detection** — replace the phase detection loop at lines 29668-29688:

```cpp
    // BEFORE: Loop to find first MUL_MAT for phase detection
    bool is_decode_phase = false;
    bool is_prompt_phase = false;
    for (int i = 0; i < cgraph->n_nodes; i++) { ... }

    // AFTER: Use cached result from step 2
    bool is_decode_phase = is_decode;
    bool is_prompt_phase = !is_decode && cgraph->n_nodes > 0;
```

5. **Pre-cache input tensor list during graph recording:**

In `graph_refresh_input_tensors` (line 25306), change to populate cache on first call and reuse:

```cpp
static void graph_refresh_input_tensors(ggml_backend_sycl_context * ctx, const ggml_cgraph * cgraph) {
    if (!ctx || !cgraph || cgraph->n_nodes == 0) {
        return;
    }

    // Use pre-cached input tensor list if available
    if (ctx->input_tensors_cached && !ctx->cached_input_tensors.empty()) {
        int refreshed_count = 0;
        for (ggml_tensor * tensor : ctx->cached_input_tensors) {
            if (!tensor || !tensor->data) continue;
            void * resolved_ptr = ggml_sycl_get_data_ptr(tensor, ctx->device);
            refreshed_count++;
            if (g_ggml_sycl_debug) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] refreshed cached input %s -> %p\n",
                                tensor->name ? tensor->name : "?", resolved_ptr);
            }
        }
        if (g_ggml_sycl_debug && refreshed_count > 0) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] refreshed %d cached input tensors\n", refreshed_count);
        }
        return;
    }

    // First call: discover and cache input tensors
    std::unordered_set<const void *> seen_ptrs;
    ctx->cached_input_tensors.clear();

    auto discover_input = [&](ggml_tensor * tensor) {
        if (!tensor || !tensor->data) return;
        if (ggml_sycl_tensor_is_weight(tensor)) return;
        if (!(tensor->flags & GGML_TENSOR_FLAG_INPUT)) return;
        if (!seen_ptrs.insert(tensor->data).second) return;

        ctx->cached_input_tensors.push_back(tensor);
        void * resolved_ptr = ggml_sycl_get_data_ptr(tensor, ctx->device);
        if (g_ggml_sycl_debug) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] discovered input %s -> %p\n",
                            tensor->name ? tensor->name : "?", resolved_ptr);
        }
    };

    for (int i = 0; i < cgraph->n_leafs; ++i) {
        discover_input(cgraph->leafs[i]);
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (!cgraph->nodes[i]) continue;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            discover_input(cgraph->nodes[i]->src[j]);
        }
    }

    ctx->input_tensors_cached = true;
    GGML_SYCL_DEBUG("[SYCL-GRAPH] cached %zu input tensors for future replays\n",
                    ctx->cached_input_tensors.size());
}
```

6. **Invalidate caches when graph is invalidated** — at line ~29809 where `exec_graph.reset()` is called:

```cpp
    if (invalidate) {
        sycl_ctx->exec_graph.reset();
        sycl_ctx->exec_graph_n_nodes = 0;
        sycl_ctx->exec_graph_hash    = 0;
        // Invalidate cached computations
        sycl_ctx->cached_persistent_n_nodes = -1;
        sycl_ctx->cached_graph_sig_n_nodes  = -1;
        sycl_ctx->input_tensors_cached      = false;
        sycl_ctx->cached_input_tensors.clear();
        // ... existing unpin calls ...
```

7. **Build and test:**

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Correctness
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Performance (no regression)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/common.hpp
git commit -m "sycl: cache O(n_nodes) computations in graph_compute for steady-state TG

Cache should_use_persistent_tg result, graph signature hash, phase
detection, and input tensor list. Eliminates 5+ O(n_nodes) scans per
token during graph replay. Caches invalidated when n_nodes changes."
```

**Notes for implementer:**
- `ggml_backend_sycl_context` is in common.hpp around line 2169
- The `cached_input_tensors` vector uses `ggml_tensor *` (not const) because `ggml_sycl_get_data_ptr` takes non-const
- The `input_tensors_cached` flag prevents re-scanning on every call
- Be careful: `graph_refresh_input_tensors` takes `const ggml_cgraph *` — the tensor pointers within are still mutable
- There are multiple `invalidate` code paths (lines 29809, 29896). Add cache invalidation to ALL of them.
- Test with GGML_SYCL_PERSISTENT_TG=1 too, to ensure persistent TG still works

---

### Task 3: Data Pointer O(1) Fast Path and get_pointer_type Elimination

**Track:** C
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/common.hpp` lines 1707-1820 (`ggml_sycl_get_data_ptr`)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 19490-19498 (remove 3 `get_pointer_type` calls in unified dispatch)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 3002-3117 (`buffer_init_tensor`, expand data_device population)

**Description:**

`ggml_sycl_get_data_ptr` is called for every tensor pointer resolution. Its current flow checks `g_tiered_enabled` atomic, then `extra->data_device`, then falls through to a 100-line chain including `get_pointer_type()` driver calls. For the common case (model fits in VRAM, `data_device` already cached), restructure as a 3-line hot path with a cold-path fallback. Also remove the 3 explicit `get_pointer_type()` validation calls in the unified dispatch path (lines 19492-19494) which are redundant with the data_device cache.

**Acceptance Criteria:**

- [ ] `ggml_sycl_get_data_ptr` fast path is 2 pointer dereferences + 1 null check (no atomic loads, no function calls)
- [ ] Cold path extracted to separate `ggml_sycl_get_data_ptr_slow` function
- [ ] 3 `get_pointer_type()` calls at lines 19492-19494 removed
- [ ] `buffer_init_tensor` populates `data_device[]` for ALL tensor types that have `extra`, not just quantized weights
- [ ] Build succeeds, correctness test passes
- [ ] Zero `get_pointer_type()` calls during steady-state TG (verify with debug counter if needed)

**Implementation Guide:**

1. **Restructure `ggml_sycl_get_data_ptr` as hot/cold split:**

In `ggml/src/ggml-sycl/common.hpp`, replace the current function at line 1707:

```cpp
// Cold path: full resolution chain (tiered cache, get_pointer_type, staging)
void * ggml_sycl_get_data_ptr_slow(const ggml_tensor * tensor, int device);

// Hot path: 2 dereferences + 1 null check for common case
inline void * ggml_sycl_get_data_ptr(const ggml_tensor * tensor, int device) {
    if (tensor == nullptr) {
        return nullptr;
    }
    // Fast path: data_device already cached (covers all tensors initialized via buffer_init_tensor)
    if (tensor->extra != nullptr) {
        void * ptr = static_cast<ggml_tensor_extra_gpu *>(tensor->extra)->data_device[device];
        if (ptr != nullptr) {
            // Input tensors need data refresh (positions, tokens change each token)
            if ((tensor->flags & GGML_TENSOR_FLAG_INPUT) && tensor->data) {
                ggml_sycl_refresh_cached_input_ptr(ptr, tensor->data, ggml_nbytes(tensor), device);
            }
            return ptr;
        }
    }
    // Cold path: full resolution
    return ggml_sycl_get_data_ptr_slow(tensor, device);
}
```

Move the entire body of the OLD `ggml_sycl_get_data_ptr` (lines 1707-1820) into a new function `ggml_sycl_get_data_ptr_slow` in ggml-sycl.cpp (NOT in the header — it's too large to inline). Keep the existing logic intact. Just rename it.

**IMPORTANT:** The old function body at lines 1711-1820 must be moved, not deleted. It handles tiered cache lookups, unified cache lookups, and staging — all needed for models that don't fit in VRAM.

2. **Remove 3 `get_pointer_type()` calls in unified dispatch:**

At lines 19490-19498, remove the validation block:
```cpp
    // REMOVE THIS BLOCK:
    //     const sycl::context & sycl_ctx = ctx.stream()->get_context();
    //     const sycl::usm::alloc src0_alloc = sycl::get_pointer_type(src0_data, sycl_ctx);
    //     const sycl::usm::alloc src1_alloc = sycl::get_pointer_type(src1_data, sycl_ctx);
    //     const sycl::usm::alloc dst_alloc  = sycl::get_pointer_type(dst_data,  sycl_ctx);
    //     const bool src0_ok = src0_alloc != sycl::usm::alloc::unknown;
    //     const bool src1_ok = src1_alloc != sycl::usm::alloc::unknown;
    //     const bool dst_ok  = dst_alloc  != sycl::usm::alloc::unknown;
```

Replace the subsequent `if (!src0_ok || !src1_ok || !dst_ok)` check with just `if (false)` or remove entirely. The `data_device` cache guarantees device pointers, and `ggml_sycl_get_layout_ptr_for` / `ggml_sycl_get_data_ptr` already validate.

3. **Expand `data_device` population in `buffer_init_tensor`:**

At line ~3090 (after the if/else-if chain for quantized types), the catch-all block already handles some cases. Strengthen it to cover ALL tensors:

Find the existing catch-all block (lines 3091-3100) and ensure it runs for all tensor types:
```cpp
    // After the if/else-if chain, this catch-all ensures ALL tensors with extra have data_device set
    if (tensor->extra != nullptr && tensor->data != nullptr) {
        auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
        if (extra->data_device[ctx->device] == nullptr) {
            if (!(ctx->is_tp_compute_buffer && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1)) {
                extra->data_device[ctx->device] = tensor->data;
            }
        }
    }
```

This block should already exist from our previous work. Verify it's present and covers the case.

4. **Build and test:**

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Correctness
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Performance
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/common.hpp
git commit -m "sycl: restructure get_data_ptr as hot/cold split, remove get_pointer_type calls

Hot path: 2 dereferences + 1 null check. Cold path extracted to
ggml_sycl_get_data_ptr_slow. Remove 3 redundant get_pointer_type
driver round-trips in unified dispatch path."
```

**Notes for implementer:**
- `ggml_sycl_refresh_cached_input_ptr` already exists in common.hpp — it handles memcpy for input tensors that change each token
- The `g_sycl_tp_config` check in buffer_init_tensor is needed because TP mode sets up per-device pointers differently
- When moving the old function body to `ggml_sycl_get_data_ptr_slow`, keep ALL the existing logic — it handles weight streaming, tiered cache, staging, unified cache fallbacks. These are essential for models that don't fit in VRAM.
- Search for all `get_pointer_type` call sites in ggml-sycl.cpp. There are ~41 total. Only remove the 3 at lines 19492-19494 in this task. Others may be needed for edge cases (leave them for now).

---

### Task 4: TG Kernel Selection Parity — MMVQ Fast-Path for Batch=1

**Track:** A
**Depends on:** Task 1 (diagnostics confirm GPU kernel time is the bottleneck)
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 19073-19100 (add fast-path before existing dispatch)

**Description:**

Master dispatches Q4_0 batch=1 through MMVQ with SOA reorder (fast). The branch routes it through the unified kernel DMMV path (slow). Add a TG fast-path at the top of `ggml_sycl_mul_mat` that bypasses the orchestrator, name parsing, prefetch tracking, TP checks, and unified kernel dispatch — routing directly to MMVQ (same as master). The unified kernel continues to handle PP (M >= 64) where oneDNN FP16 is critical.

**Acceptance Criteria:**

- [ ] Batch=1 quantized matmuls route to MMVQ with SOA reorder (same kernel as master)
- [ ] `opt_for_reorder` is called for SOA layout optimization
- [ ] Types not supported by MMVQ fall through to DMMV (not broken)
- [ ] F16 KQ/KQV matmuls still use their existing special paths (not affected)
- [ ] PP (batch >= 64) still uses unified kernel / oneDNN path (not affected)
- [ ] Build succeeds, correctness test passes
- [ ] TG128 performance improves (target: >= 40 tok/s)

**Implementation Guide:**

1. **Add TG fast-path at the TOP of `ggml_sycl_mul_mat`:**

After the debug scope at line 19082, BEFORE the KQV detection at line 19083, add:

```cpp
    // =====================================================================
    // TG fast-path: batch=1, single-device, quantized → MMVQ (same as master)
    // Bypasses orchestrator, name parsing, prefetch, TP checks for maximum speed.
    // The unified kernel handles PP (M >= 64) via oneDNN FP16 — unchanged.
    // =====================================================================
    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    if (src1->ne[1] == 1 && !split
        && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32
        && dst->type == GGML_TYPE_F32
        && !(g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1)) {

        // Match master's kernel selection logic (ggml-sycl.cpp:3310-3360 in master)
        const bool use_mmvq = (ggml_sycl_supports_mmvq(src0->type)
                               && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE);
        const bool use_dmmv = (ggml_sycl_supports_dmmv(src0->type)
                               && src0->ne[0] % GGML_SYCL_DMMV_X == 0);

        if (use_mmvq) {
            // MMVQ path — direct match to master lines 3353-3360
            ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
            if (extra && extra->optimized_feature.reorder
                && ggml_sycl_supports_reorder_mmvq(src0->type)) {
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(
                    ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
            } else {
                ggml_sycl_op_mul_mat<quantize_q8_1>(
                    ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
            }
            return;
        } else if (use_dmmv) {
            // DMMV fallback for types not supported by MMVQ
            ggml_sycl_op_mul_mat<no_quantize_q8_1>(
                ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
            return;
        }
        // Neither MMVQ nor DMMV available: fall through to full dispatch
    }
```

**NOTE:** The existing `const bool split = ...` declaration at line 19256 will now conflict since we declared it earlier. Either:
- Remove the duplicate at line 19256 and use the one from the fast-path check
- Or scope the fast-path `split` inside the if-block

**Recommended:** Move the `split` declaration to line 19082 (before the fast-path check) and remove the duplicate at line 19256.

2. **Verify `ggml_sycl_supports_mmvq` is available:**

Search for its declaration — it should be in `ggml-sycl.cpp` or a header. On master it's at line ~3281. On branch, find it with:
```bash
grep -n 'ggml_sycl_supports_mmvq\|can_use_mul_mat_vec_q' ggml/src/ggml-sycl/ggml-sycl.cpp | head -5
```

The branch has `can_use_mul_mat_vec_q` at line 18247 with a 4-parameter signature. We can use it directly:
```cpp
    const bool use_mmvq = can_use_mul_mat_vec_q(src0, src1, dst, ctx.device);
```

**BUT** the branch's `can_use_mul_mat_vec_q` (line 18247) has complex unified-cache-aware logic for Q6_K. For the fast-path, we want the SIMPLE check. Use the underlying support function instead:
```cpp
    const bool use_mmvq = ggml_sycl_supports_mmvq(src0->type)
                          && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

If `ggml_sycl_supports_mmvq` doesn't exist on branch, check for `ggml_sycl_supports_reorder_mmvq` (line 15230) — it exists. Use:
```cpp
    const bool use_mmvq = ggml_is_quantized(src0->type)
                          && src1->type == GGML_TYPE_F32
                          && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

3. **Build and test:**

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Correctness (CRITICAL — kernel change must produce identical output)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Performance
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Verify MMVQ path is taken (check debug output for first few matmuls)
GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 0 -n 32 > /tmp/kernel-path.txt 2>&1
grep -c 'mul_mat_vec_q' /tmp/kernel-path.txt  # Should show MMVQ kernel calls
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add TG fast-path for batch=1 matmul via MMVQ

Route batch=1 quantized matmuls directly to MMVQ with SOA reorder,
bypassing the orchestrator and unified kernel. Matches master's kernel
selection. Unified kernel still used for PP (M >= 64)."
```

**Notes for implementer:**
- This is the HIGHEST IMPACT change. If TG improves from 15 to 40+ tok/s, this confirms the kernel mismatch hypothesis.
- MMVQ_MAX_BATCH_SIZE is defined as a macro — search for it in the codebase
- `quantize_and_reorder_q8_1_soa` is a template parameter, not a function call — it tells `ggml_sycl_op_mul_mat` how to quantize activations
- The `opt_for_reorder` function from master doesn't exist on branch. Instead, the SOA reorder is already handled by the `optimized_feature.reorder` flag which is set during `buffer_init_tensor` / `finalize_layouts`.
- If correctness fails: the fast-path may need adjustment. Check if `ggml_sycl_op_mul_mat` expects specific state that the orchestrator normally sets up. If so, add that state setup before the dispatch call.
- F16 matmuls (KQ, KQV) are handled BEFORE the fast-path check in the existing code (they check `src0->type == GGML_TYPE_F16`). Since `ggml_is_quantized(GGML_TYPE_F16)` returns false, the fast-path won't intercept them.

---

### Task 5: Integration Benchmark and Verification

**Track:** — (convergence point)
**Depends on:** Tasks 1, 2, 3, 4
**File scope:**
- No file changes — purely verification

**Description:**

After all tasks are merged, run comprehensive benchmarks and correctness verification. Collect before/after metrics.

**Acceptance Criteria:**

- [ ] Correctness: seed 42 completion output matches expected
- [ ] TG128 >= 40 tok/s (minimum target, up from 15.73)
- [ ] PP512 >= 1300 tok/s (no regression)
- [ ] No crashes or hangs across 3 consecutive benchmark runs
- [ ] Diagnostic timing output collected and analyzed (from Task 1)

**Implementation Guide:**

1. **Full build from clean state:**

```bash
source /opt/intel/oneapi/setvars.sh --force
# Touch all modified files to force recompilation
touch ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/common.hpp
ninja -C build -j $(nproc)
```

2. **Correctness verification (3 runs):**

```bash
for i in 1 2 3; do
  echo "=== Run $i ==="
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
    -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
done
```

All 3 runs must produce identical output.

3. **Performance benchmark (3 runs, take median):**

```bash
for i in 1 2 3; do
  echo "=== Benchmark Run $i ==="
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
done
```

4. **Diagnostic analysis:**

```bash
GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 0 -n 128 > /tmp/final-diag.txt 2>&1
grep 'TG-DIAG' /tmp/final-diag.txt
grep 'TIMING' /tmp/final-diag.txt
```

5. **Non-graph baseline (optional — check dispatch overhead without graph replay):**

```bash
GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128
```

6. **Report results in this format:**

```
| Metric | Before | After | Target | Status |
|--------|--------|-------|--------|--------|
| TG128 tok/s | 15.73 | ??? | >= 40 | PASS/FAIL |
| PP512 tok/s | 1303 | ??? | >= 1300 | PASS/FAIL |
| Correctness | PASS | ??? | PASS | PASS/FAIL |
| GPU exec time (sync) | ~62ms | ??? | ~13ms | PASS/FAIL |
| graph_compute total | ??? | ??? | < 2ms | PASS/FAIL |
```

**Commit:** None (verification only). If issues found, create follow-up tasks.

---

## Summary

| Task | Track | Dependencies | Focus | Files |
|------|-------|-------------|-------|-------|
| 1 | A | None | Diagnostic timing | ggml-sycl.cpp (sync + graph_compute) |
| 2 | B | None | Cache O(n) computations | common.hpp + ggml-sycl.cpp (graph_compute) |
| 3 | C | None | Pointer hot/cold split | common.hpp + ggml-sycl.cpp (init + unified dispatch) |
| 4 | A | Task 1 | MMVQ fast-path for batch=1 | ggml-sycl.cpp (mul_mat dispatch) |
| 5 | — | All | Benchmark + verification | None (testing only) |
