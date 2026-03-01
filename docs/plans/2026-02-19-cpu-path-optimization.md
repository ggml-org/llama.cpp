# CPU Path Optimization for Tensor Split — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Optimize the CPU portion of tensor split from ~140 ms/token overhead to ~8-13 ms,
enabling 70+ tok/s TG for full-VRAM models and paving the way for 18-29 tok/s offloading.

**Architecture:** Three optimizations compound: (1) eliminate D2H weight copies by using host
mmap pointers from `g_host_ptr_map`, (2) batch all CPU vec_dot work into a single TBB
parallel_for with pre-quantized src1, (3) batch all H2D output copies with a single wait.

**Tech Stack:** C++17, Intel oneAPI SYCL, oneTBB, ggml quantization API

---

## End Goal Context

This plan implements **P0** from the tensor split optimization roadmap. The ultimate goal
is running large models (GPT-OSS 120B ~63 GB, dense 70B+ ~35 GB) that exceed VRAM (28 GB
across Arc B580 + Arc Pro B50) at maximum speed using all available bandwidth:

| Device | Device BW | PCIe BW | Role |
|--------|-----------|---------|------|
| Arc B580 | 280 GB/s | 13 GB/s (4.0 x8) | Primary GPU |
| Arc Pro B50 | 224 GB/s | 25 GB/s (5.0 x8) | Secondary GPU |
| CPU (Arrow Lake) | — | 40-75 GB/s (DDR5) | CPU compute |

**How this plan fits the roadmap:**

| Priority | What | Impact | Status |
|----------|------|--------|--------|
| **P0 (THIS PLAN)** | CPU path optimization | 6.86 → 70+ tok/s (graph replay) | Implementing |
| P1 | Single-GPU tensor split for host layers | 12.5 → 23 tok/s | Next |
| P2 | Multi-GPU tensor split (add B50) | 23 → 29 tok/s | Future |
| P3 | Graph replay + per-layer hybrid | 29 → 35 tok/s | Future |
| P4 | Persistent kernel integration | 35 → 45+ tok/s | Long-term |

**Design intent:**
- **Unified cache** owns all memory. Tensor split queries weight locations (device vs host)
  through existing cache APIs, never duplicates the cache's job.
- **Unified kernel** (unified-kernel.cpp) is the long-term dispatch path. The batched
  vec_dot function created here will be reused by the persistent kernel's CPU dispatch.
- Host pointer access via `g_host_ptr_map` (cpu-dispatch.cpp:76) is the standard way to
  access weight data on host. This plan uses that API consistently.

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1 | Host pointer shortcut — eliminate D2H weight copies |
| B | 2 | Batched CPU vec_dot with pre-quantized src1 dedup |
| — | 3 | Rewrite `split_execute_cpu_work` using Tasks 1+2 |
| — | 4 | Build, correctness test, benchmark |

### Dependency Graph

```
Task 1 (Track A) ──┐
                    ├──→ Task 3 ──→ Task 4
Task 2 (Track B) ──┘
```

### File Ownership Map

| File | Tasks | Conflict Risk |
|------|-------|---------------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 1, 3 | Sequential (1 before 3) |
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 2 | None (single task) |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | 2 | None (single task) |

---

## Current State (What We're Optimizing)

### The Problem: 224 Sequential D2H + vec_dot Steps

`split_execute_cpu_work()` at `ggml-sycl.cpp:19937` processes the CPU work queue
after GPU graph replay. Profiling shows **140 ms/token** broken down as:

| Phase | Time | % | Root Cause |
|-------|------|---|-----------|
| D2H src1 copies | 70 ms | 47% | 224 sequential `q->memcpy().wait()` calls |
| CPU vec_dot | 80 ms | 53% | 224 separate TBB `parallel_for` invocations |
| H2D output | 1 ms | <1% | Already batched (single wait at end) |

The current code (simplified):
```cpp
// ggml-sycl.cpp:19960-19977
for (size_t wi = 0; wi < n_work; wi++) {
    const auto & w = g_split_cpu_queue[wi];
    q->memcpy(g_split_staging.src1_host, w.src1_device, w.src1_bytes).wait();  // 0.31ms each × 224
    ggml_sycl_cpu_vec_dot_rows(w.type, w.ne00, w.host_weights,                // 0.36ms each × 224
                                g_split_staging.src1_host,
                                g_split_cpu_output.data, w.n_rows);
    q->memcpy(w.dst_device + w.N_gpu, g_split_cpu_output.data,
              w.n_rows * sizeof(float));
}
q->wait();
```

### Target: 3-Phase Batched Execution

After optimization:
```
Phase 1: Batch D2H — stage ~128 unique src1 values (2 MB total, ONE q->wait())   ~0.2 ms
Phase 2: Batch vec_dot — ONE TBB parallel_for over all 177K rows                 ~7-13 ms
Phase 3: Batch H2D — all output copies submitted, ONE q->wait()                  ~0.1 ms
                                                                          Total: ~8-13 ms
```

---

### Task 1: Host Pointer Shortcut — Eliminate D2H Weight Copies

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:20075-20145` (recording + non-recording paths in `ggml_sycl_mul_mat_tensor_split`)

**Description:**

Replace D2H weight copies from device AOS data with direct host mmap pointer access.
The `g_host_ptr_map` in `cpu-dispatch.cpp:76` already stores host pointers for all weight
tensors, registered during model loading via `ggml_sycl_cpu_dispatch_register_host_ptr()`.
Currently, tensor split D2H-copies weight rows from device AOS → host-pinned cache
(`g_split_weight_cache`). This task eliminates that copy by reading directly from the
host mmap pointer, offset to the CPU row range `[N_gpu, N)`.

**Why this works:** Weight data is immutable after model loading. The mmap pointer is
valid for the model's lifetime. For CPU offload mode, `g_host_ptr_map` stores an
`aligned_alloc(64)` copy — even better (AVX-aligned, persistent).

**Acceptance Criteria:**

- [ ] Non-recording path uses host mmap pointer when available
- [ ] Recording path (graph replay warmup) uses host mmap pointer for work descriptors
- [ ] Falls back to D2H copy if host pointer unavailable (backward compat)
- [ ] `g_split_weight_cache` is no longer populated when host pointer is available
- [ ] GPU-only baseline unaffected: PP512 >= 1200, TG128 >= 68 tok/s
- [ ] Tensor split produces correct output (deterministic completion test)
- [ ] Commit with descriptive message

**Implementation Guide:**

1. **Modify the non-recording path** (`ggml-sycl.cpp:20120-20145`)

Current code at line 20120:
```cpp
    // Non-recording mode (Phase 1 path): execute CPU portion immediately.
    // Cache weights for future graph replay use.
    void * host_weights = nullptr;
    if (src0->name) {
        host_weights = split_get_cached_weights(src0->name, src0_aos, N_gpu,
                                                 src0_row_bytes, cpu_weight_bytes, stream);
    }
    if (!host_weights) {
        // Fallback: use transient staging buffer
        split_weight_staging_ensure(cpu_weight_bytes, stream);
        if (!g_split_weight_staging.data) {
            return false;
        }
        stream->memcpy(g_split_weight_staging.data, src0_aos + N_gpu * src0_row_bytes, cpu_weight_bytes).wait();
        host_weights = g_split_weight_staging.data;
    }
```

Replace with:
```cpp
    // Non-recording mode: use host mmap pointer if available (zero-copy).
    // Falls back to D2H from device AOS if host pointer not registered.
    const void * host_weights = nullptr;
    if (src0->name) {
        const void * host_ptr_full = ggml_sycl_cpu_dispatch_get_host_ptr(src0->name);
        if (host_ptr_full) {
            // Direct host access — no D2H copy needed
            host_weights = (const char *) host_ptr_full + N_gpu * src0_row_bytes;
        }
    }
    if (!host_weights) {
        // Fallback: D2H from device AOS data
        split_weight_staging_ensure(cpu_weight_bytes, stream);
        if (!g_split_weight_staging.data) {
            return false;
        }
        stream->memcpy(g_split_weight_staging.data,
                        src0_aos + N_gpu * src0_row_bytes, cpu_weight_bytes).wait();
        host_weights = g_split_weight_staging.data;
    }
```

2. **Modify the recording path** (`ggml-sycl.cpp:20080-20105`)

Current code at line 20080:
```cpp
        void * cached_weights = nullptr;
        if (src0->name) {
            auto it = g_split_weight_cache.find(src0->name);
            if (it != g_split_weight_cache.end() && it->second.bytes >= cpu_weight_bytes) {
                cached_weights = it->second.data;
            }
        }
        if (!cached_weights) {
            GGML_LOG_WARN("[TENSOR-SPLIT] No cached weights for '%s' during recording, skipping CPU work\n",
                          src0->name ? src0->name : "?");
            return true;
        }
```

Replace with:
```cpp
        // Try host mmap pointer first (zero-copy, always available)
        const void * cached_weights = nullptr;
        if (src0->name) {
            const void * host_ptr_full = ggml_sycl_cpu_dispatch_get_host_ptr(src0->name);
            if (host_ptr_full) {
                cached_weights = (const char *) host_ptr_full + N_gpu * src0_row_bytes;
            }
        }
        if (!cached_weights) {
            // Fallback: check D2H weight cache (populated during warmup)
            if (src0->name) {
                auto it = g_split_weight_cache.find(src0->name);
                if (it != g_split_weight_cache.end() && it->second.bytes >= cpu_weight_bytes) {
                    cached_weights = it->second.data;
                }
            }
        }
        if (!cached_weights) {
            GGML_LOG_WARN("[TENSOR-SPLIT] No host pointer or cached weights for '%s' during recording\n",
                          src0->name ? src0->name : "?");
            return true;
        }
```

3. **Update the `split_cpu_work` struct** to use `const void *` for `host_weights`

At line 19900:
```cpp
struct split_cpu_work {
    const void * host_weights;  // Changed: was void*, now const void* for mmap safety
    void *    src1_device;
    float *   dst_device;
    ggml_type type;
    int       ne00;
    int       n_rows;
    int64_t   N_gpu;
    size_t    src1_bytes;
};
```

4. **Verify** the change:

Build:
```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

Test correctness (graph disabled, isolates non-recording path):
```bash
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

Test correctness (graph enabled, recording path):
```bash
GGML_SYCL_TENSOR_SPLIT=13 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

GPU-only baseline (tensor split OFF):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68 (zero regression)

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: use host mmap pointers for tensor split CPU weights

Eliminate D2H weight copies in tensor split by using host mmap pointers
from g_host_ptr_map (registered during model loading). Both non-recording
and graph-replay recording paths now check for host pointers before
falling back to D2H from device AOS data.

This removes ~70ms of per-token overhead (224 weight staging operations)
from the graph replay CPU work path.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- `ggml_sycl_cpu_dispatch_get_host_ptr()` is declared in `cpu-dispatch.hpp:69` and already
  included in ggml-sycl.cpp via the header chain
- The `const void *` change to `split_cpu_work::host_weights` is a type-safety improvement;
  the queue push at line 20097 may need a `const_cast` removed or type adjusted
- `g_split_weight_cache` and `split_get_cached_weights` are still needed as fallback but
  will no longer be the primary path. Do NOT delete them yet.
- `g_split_weight_staging` (transient staging buffer) is only needed for the D2H fallback

---

### Task 2: Batched CPU vec_dot with Pre-Quantized src1 Dedup

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (add new function after `ggml_sycl_cpu_vec_dot_rows` at line ~310)
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.hpp` (add declarations after line ~67)

**Description:**

Add a batched CPU vec_dot function that processes multiple tensor split work items in a
single TBB `parallel_for` invocation. The function deduplicates src1 quantization — within
a transformer layer, Q/K/V projections share the same src1 (hidden state), and gate/up
share another src1 (FFN input). Instead of quantizing src1 224 times (once per work item),
the batched function quantizes ~128 unique src1 values and reuses them.

This eliminates 224 separate TBB `parallel_for` calls (each with overhead from thread pool
wake/dispatch) and replaces them with ONE invocation distributing all ~177K rows.

**Acceptance Criteria:**

- [ ] New struct `cpu_vec_dot_batch_item` declared in `cpu-dispatch.hpp`
- [ ] New function `ggml_sycl_cpu_vec_dot_batched()` implemented in `cpu-dispatch.cpp`
- [ ] Function deduplicates src1 quantization (same src1 pointer → one quantize call)
- [ ] Single TBB `parallel_for` over all work items
- [ ] Handles mixed quant types (different items can have different `ggml_type`)
- [ ] Falls back to serial processing when `n_items < 2` or TBB unavailable
- [ ] Existing `ggml_sycl_cpu_vec_dot_rows()` is NOT modified (backward compat)
- [ ] Compiles cleanly with `-Wall -Werror`
- [ ] Commit with descriptive message

**Implementation Guide:**

1. **Add the batch item struct to `cpu-dispatch.hpp`** (after line 67):

```cpp
// Batch item for ggml_sycl_cpu_vec_dot_batched().
// Each item represents one tensor's CPU row range from tensor split.
struct cpu_vec_dot_batch_item {
    const void * weight_data;   // host pointer to first CPU row (pre-offset by caller)
    const float * src1_host;    // host float32 activation [ne00]
    float *       output;       // output buffer for this item [n_rows]
    ggml_type     type;         // weight quant type (e.g. GGML_TYPE_Q4_0)
    int           ne00;         // K columns per row
    int           n_rows;       // number of CPU rows for this item
};

// Process multiple tensor split work items in a single TBB parallel_for.
// Deduplicates src1 quantization: items sharing the same src1_host pointer
// share one quantized copy. Distributes work across all TBB threads.
void ggml_sycl_cpu_vec_dot_batched(const cpu_vec_dot_batch_item * items, int n_items);
```

2. **Implement the function in `cpu-dispatch.cpp`** (after `ggml_sycl_cpu_vec_dot_rows` at ~line 310):

```cpp
void ggml_sycl_cpu_vec_dot_batched(const cpu_vec_dot_batch_item * items, int n_items) {
    if (n_items <= 0 || !items) {
        return;
    }

    // --- Phase 1: Pre-quantize unique src1 values ---
    // Many items share the same src1 (Q/K/V share hidden state, gate/up share FFN input).
    // Map: src1_host pointer → quantized Q8 data.
    // We store quantized buffers in a vector and index by src1_host pointer.
    struct src1_q_entry {
        std::vector<uint8_t> data;
    };
    std::unordered_map<const float *, src1_q_entry> src1_q_map;

    for (int i = 0; i < n_items; i++) {
        const auto & item = items[i];
        if (!item.src1_host || !item.weight_data || item.n_rows <= 0) {
            continue;
        }
        if (src1_q_map.count(item.src1_host)) {
            continue;  // Already quantized
        }

        const auto * cpu_traits = ggml_get_type_traits_cpu(item.type);
        if (!cpu_traits || !cpu_traits->vec_dot) {
            continue;
        }
        const ggml_type        vdt        = cpu_traits->vec_dot_type;
        const auto *           vdt_traits = ggml_get_type_traits_cpu(vdt);
        ggml_from_float_t      from_float = vdt_traits ? vdt_traits->from_float : nullptr;
        if (!from_float) {
            continue;
        }

        const size_t q_size = ggml_row_size(vdt, item.ne00);
        auto & entry = src1_q_map[item.src1_host];
        entry.data.resize(q_size);
        from_float(item.src1_host, entry.data.data(), item.ne00);
    }

    if (src1_q_map.empty()) {
        return;
    }

    // Build per-item Q8 pointer array for O(1) access in the parallel loop.
    std::vector<const uint8_t *> item_src1_q(n_items, nullptr);
    for (int i = 0; i < n_items; i++) {
        auto it = src1_q_map.find(items[i].src1_host);
        if (it != src1_q_map.end()) {
            item_src1_q[i] = it->second.data.data();
        }
    }

    // --- Phase 2: Single batched parallel_for ---
    // Parallelize at the work-item level. Each item has ~500-1500 rows,
    // providing sufficient work per TBB task for good load balance.
    const cpu_vec_dot_batch_item * items_ptr = items;
    const uint8_t ** q8_ptrs = item_src1_q.data();

#if GGML_SYCL_HAS_TBB
    ggml_sycl_cpu_arena().execute([&] {
        ggml_sycl_tbb::parallel_for(
            ggml_sycl_tbb::blocked_range<int>(0, n_items, 1),
            [items_ptr, q8_ptrs](const ggml_sycl_tbb::blocked_range<int> & range) {
                for (int ii = range.begin(); ii < range.end(); ii++) {
                    const auto & item    = items_ptr[ii];
                    const uint8_t * q8   = q8_ptrs[ii];
                    if (!q8 || !item.weight_data || !item.output || item.n_rows <= 0) {
                        continue;
                    }

                    const auto * cpu_traits = ggml_get_type_traits_cpu(item.type);
                    const size_t row_stride = ggml_row_size(item.type, item.ne00);

                    for (int r = 0; r < item.n_rows; r++) {
                        const void * row =
                            (const char *) item.weight_data + (size_t) r * row_stride;
                        float dot = 0.0f;
                        cpu_traits->vec_dot(item.ne00, &dot, sizeof(float),
                                            row, 0, q8, 0, 1);
                        item.output[r] = dot;
                    }
                }
            });
    });
#else
    // No-TBB fallback: sequential
    for (int ii = 0; ii < n_items; ii++) {
        const auto & item  = items_ptr[ii];
        const uint8_t * q8 = q8_ptrs[ii];
        if (!q8 || !item.weight_data || !item.output || item.n_rows <= 0) {
            continue;
        }

        const auto * cpu_traits = ggml_get_type_traits_cpu(item.type);
        const size_t row_stride = ggml_row_size(item.type, item.ne00);

        for (int r = 0; r < item.n_rows; r++) {
            const void * row =
                (const char *) item.weight_data + (size_t) r * row_stride;
            float dot = 0.0f;
            cpu_traits->vec_dot(item.ne00, &dot, sizeof(float),
                                row, 0, q8, 0, 1);
            item.output[r] = dot;
        }
    }
#endif

    GGML_SYCL_DEBUG("[TENSOR-SPLIT] Batched vec_dot: %d items, %d unique src1\n",
                    n_items, (int) src1_q_map.size());
}
```

3. **Make `ggml_sycl_cpu_arena()` accessible** within the file.

The function `ggml_sycl_cpu_arena()` is already static in `cpu-dispatch.cpp` at line 1135.
The new `ggml_sycl_cpu_vec_dot_batched()` function is in the same file, so it can call
it directly. The forward declaration at line 246 already covers this:
```cpp
static ggml_sycl_tbb::task_arena & ggml_sycl_cpu_arena();
```

No change needed — just place the new function AFTER the existing forward declaration.

4. **Verify** the new function compiles:

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```
Expected: Clean build, no warnings.

The function is not yet called (Task 3 wires it in), but it must compile.

**Commit:**
```bash
git add ggml/src/ggml-sycl/cpu-dispatch.cpp ggml/src/ggml-sycl/cpu-dispatch.hpp
git commit -m "sycl: add batched CPU vec_dot for tensor split

Add ggml_sycl_cpu_vec_dot_batched() that processes multiple tensor split
work items in a single TBB parallel_for. Deduplicates src1 quantization
across items that share the same activation pointer (Q/K/V share hidden
state, gate/up share FFN input within each transformer layer).

Replaces 224 separate TBB parallel_for invocations with ONE, eliminating
thread pool wake/dispatch overhead per work item.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- `ggml_get_type_traits_cpu()` is declared in `ggml-cpu-traits.h` (already included
  transitively). If compilation fails, add `#include "ggml-cpu-traits.h"` to cpu-dispatch.cpp.
- The `GGML_SYCL_DEBUG` macro is defined in `common.hpp` (already included).
- `ggml_row_size()` is in `ggml.h` (already included).
- The `blocked_range` grain_size of 1 means each work item (averaging ~700 rows) is a TBB
  task. With ~20 threads and 224 items, this gives good load distribution.
- The `src1_q_map` uses `std::vector<uint8_t>` for storage. The total allocation is
  ~128 unique × ~2.3 KB per Q8 buffer = ~300 KB. Negligible.

---

### Task 3: Rewrite `split_execute_cpu_work` Using Batched Infrastructure

**Track:** — (convergence point)
**Depends on:** Task 1, Task 2
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:19937-19983` (rewrite `split_execute_cpu_work`)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:19912-19935` (grow `g_split_cpu_output` for batched use)

**Description:**

Rewrite `split_execute_cpu_work()` to use three-phase batched execution:
1. **Batch D2H src1**: Deduplicate device→host src1 copies, submit all async, one wait
2. **Batch vec_dot**: Build work item array, call `ggml_sycl_cpu_vec_dot_batched()`
3. **Batch H2D output**: Submit all output copies async, one wait

The host weights come from Task 1's mmap pointer path (stored in `split_cpu_work::host_weights`).
The batched vec_dot comes from Task 2's `ggml_sycl_cpu_vec_dot_batched()`.

**Acceptance Criteria:**

- [ ] `split_execute_cpu_work()` rewritten with 3-phase batched execution
- [ ] D2H src1 copies are deduplicated (same device pointer → one copy)
- [ ] ONE TBB `parallel_for` for all CPU vec_dot work (via Task 2's function)
- [ ] H2D output copies batched with single final `q->wait()`
- [ ] `g_split_cpu_output` grows to hold ALL items' output (not just max single item)
- [ ] Correct output with graph replay: deterministic completion test
- [ ] Correct output without graph replay: deterministic completion test
- [ ] GPU-only baseline unaffected
- [ ] Commit with descriptive message

**Implementation Guide:**

1. **Add src1 staging infrastructure** (before `split_execute_cpu_work`, ~line 19912):

```cpp
// Persistent src1 D2H staging buffer for batched tensor split.
// Sized to hold all unique src1 values across the work queue.
static struct {
    void *  data = nullptr;
    size_t  bytes = 0;
} g_split_src1_staging;

static void split_src1_staging_ensure(size_t needed, sycl::queue * q) {
    if (g_split_src1_staging.bytes >= needed) {
        return;
    }
    if (g_split_src1_staging.data) {
        sycl::free(g_split_src1_staging.data, *q);
    }
    g_split_src1_staging.data = sycl::malloc_host(needed, *q);
    g_split_src1_staging.bytes = g_split_src1_staging.data ? needed : 0;
}
```

2. **Modify `split_cpu_output_ensure`** to support total rows across ALL items:

The existing function at line 19920 only ensures enough space for the largest single item.
Change it to accept the TOTAL output size needed:

At line 19920, the function already takes a `needed` size parameter and does grows-only
allocation. No change needed to the function — just change how it's called (step 3).

3. **Rewrite `split_execute_cpu_work()`** (replace lines 19937-19983):

```cpp
static void split_execute_cpu_work(sycl::queue * q) {
    if (g_split_cpu_queue.empty()) {
        return;
    }
    const size_t n_work = g_split_cpu_queue.size();

    // --- Phase 0: Compute buffer sizes ---
    size_t total_output_rows = 0;
    size_t max_src1_bytes    = 0;
    for (const auto & w : g_split_cpu_queue) {
        total_output_rows += w.n_rows;
        if (w.src1_bytes > max_src1_bytes) {
            max_src1_bytes = w.src1_bytes;
        }
    }

    // Count unique src1 device pointers for D2H dedup
    std::unordered_map<void *, size_t> src1_unique;  // device_ptr → offset in staging buf
    size_t src1_staging_needed = 0;
    for (const auto & w : g_split_cpu_queue) {
        if (src1_unique.count(w.src1_device)) {
            continue;
        }
        src1_unique[w.src1_device] = src1_staging_needed;
        src1_staging_needed += w.src1_bytes;
    }

    // Ensure staging buffers
    split_src1_staging_ensure(src1_staging_needed, q);
    split_cpu_output_ensure(total_output_rows * sizeof(float), q);

    if (!g_split_src1_staging.data || !g_split_cpu_output.data) {
        GGML_LOG_WARN("[TENSOR-SPLIT] Failed to allocate staging buffers\n");
        return;
    }

    // --- Phase 1: Batch D2H src1 copies (deduplicated) ---
    for (const auto & [dev_ptr, offset] : src1_unique) {
        // Find the first work item with this src1 to get the byte count
        size_t bytes = 0;
        for (const auto & w : g_split_cpu_queue) {
            if (w.src1_device == dev_ptr) {
                bytes = w.src1_bytes;
                break;
            }
        }
        void * host_dst = (char *) g_split_src1_staging.data + offset;
        q->memcpy(host_dst, dev_ptr, bytes);  // async, no wait
    }
    q->wait();  // ONE wait for all D2H copies

    // --- Phase 2: Build batch work items and call batched vec_dot ---
    std::vector<cpu_vec_dot_batch_item> batch_items(n_work);
    size_t output_offset = 0;
    for (size_t wi = 0; wi < n_work; wi++) {
        const auto & w = g_split_cpu_queue[wi];
        size_t src1_off = src1_unique[w.src1_device];
        batch_items[wi] = {
            w.host_weights,                                         // weight_data
            (const float *) ((char *) g_split_src1_staging.data + src1_off),  // src1_host
            g_split_cpu_output.data + output_offset,                // output
            w.type,                                                 // type
            w.ne00,                                                 // ne00
            w.n_rows,                                               // n_rows
        };
        output_offset += w.n_rows;
    }

    ggml_sycl_cpu_vec_dot_batched(batch_items.data(), static_cast<int>(n_work));

    // --- Phase 3: Batch H2D output copies ---
    output_offset = 0;
    for (size_t wi = 0; wi < n_work; wi++) {
        const auto & w = g_split_cpu_queue[wi];
        q->memcpy(w.dst_device + w.N_gpu,
                  g_split_cpu_output.data + output_offset,
                  w.n_rows * sizeof(float));  // async, no wait
        output_offset += w.n_rows;
    }
    q->wait();  // ONE wait for all H2D copies

    GGML_SYCL_DEBUG("[TENSOR-SPLIT] Batched: %zu items, %zu unique src1, %zu total rows\n",
                    n_work, src1_unique.size(), total_output_rows);
}
```

4. **Include the new header type.** At the top of ggml-sycl.cpp, `cpu-dispatch.hpp` is
already included (it's in the existing include chain). The `cpu_vec_dot_batch_item` struct
from Task 2 is declared there. No additional includes needed.

5. **Verify correctness:**

GPU-only baseline (tensor split OFF):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200, TG128 >= 68

Tensor split with graph replay:
```bash
GGML_SYCL_TENSOR_SPLIT=13 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

Tensor split without graph replay:
```bash
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: batch tensor split CPU work into 3-phase pipeline

Rewrite split_execute_cpu_work() to use batched execution:
- Phase 1: Deduplicate and batch D2H src1 copies (224 waits → 1)
- Phase 2: Single TBB parallel_for via ggml_sycl_cpu_vec_dot_batched()
- Phase 3: Batch H2D output copies (already batched, now explicit)

Expected improvement: ~140 ms → ~8-13 ms per token for the CPU work
portion of tensor split graph replay.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- The `src1_unique` map uses `void *` keys (device pointers). These are stable within a
  graph replay session because `graph_refresh_input_tensors()` updates data in-place.
- `g_split_cpu_output.data` now holds ALL items' output (not just largest single item).
  Total size for Mistral 7B: ~177K rows × 4 bytes = ~700 KB. Well within sycl::malloc_host.
- The `batch_items` vector is allocated per-call (~224 entries × ~48 bytes = ~11 KB). The
  allocator will cache this. For zero-alloc, could use a static vector, but clarity wins.
- After this task, `g_split_staging` (the old per-item staging) is no longer used by
  `split_execute_cpu_work()`. It's still used by the non-recording path in
  `ggml_sycl_mul_mat_tensor_split()`. Leave it in place.

---

### Task 4: Build, Correctness Test, and Performance Benchmark

**Track:** — (final verification)
**Depends on:** Task 3
**File scope:**
- No file changes (verification only)

**Description:**

Full build from clean state, comprehensive correctness testing across all tensor split
configurations, and performance benchmarking to quantify the improvement. Compare against
the profiled baseline (6.86 tok/s with graph replay, ~64 tok/s GPU-only).

**Acceptance Criteria:**

- [ ] Clean build from scratch (no stale objects)
- [ ] GPU-only baseline: PP512 >= 1200, TG128 >= 68 tok/s (zero regression)
- [ ] Tensor split 13% with graph: TG128 improved (target: 50-70 tok/s, up from 6.86)
- [ ] Tensor split 13% without graph: correct output
- [ ] Tensor split at 10%, 15%, 20%, 25%: correct output at each
- [ ] Performance sweep logged with 30s cooldown between runs
- [ ] No memory leaks (valgrind or manual check of staging buffer lifecycle)

**Verification Steps:**

1. **Clean rebuild:**
```bash
source /opt/intel/oneapi/setvars.sh --force
rm -rf build && cmake -B build -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
ninja -C build -j $(nproc)
```

2. **GPU-only baseline (tensor split disabled):**
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
Expected: PP512 >= 1200 tok/s, TG128 >= 68 tok/s

**Wait 30 seconds** (thermal cooldown for Arc B580)

3. **Tensor split correctness — graph replay:**
```bash
GGML_SYCL_TENSOR_SPLIT=13 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

4. **Tensor split correctness — no graph:**
```bash
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

5. **Performance sweep (graph replay):**
```bash
for pct in 10 13 15 20 25; do
  echo "=== TENSOR_SPLIT=$pct ==="
  GGML_SYCL_TENSOR_SPLIT=$pct \
    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
  sleep 30  # thermal cooldown
done
```
Expected: Best TG128 > 50 tok/s (up from 6.86 baseline)

6. **Correctness across split percentages:**
```bash
for pct in 10 15 20 25; do
  echo "=== TENSOR_SPLIT=$pct ==="
  GGML_SYCL_TENSOR_SPLIT=$pct \
    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
    -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
  echo ""
done
```
Expected: All produce `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

7. **Report results.** Report to team lead:
   - GPU-only baseline: PP512 = ?, TG128 = ?
   - Tensor split perf per split %: TG128 at 10/13/15/20/25%
   - Optimal split % (best TG128)
   - Correctness: PASS/FAIL for each configuration
   - Any issues or anomalies

**Commit:** No commit (verification only). If issues found, report to team lead.

**Notes for implementer:**
- **Arc B580 thermal throttling**: MUST wait 30+ seconds between benchmark runs.
  Back-to-back runs cause 29x slowdown (70 → 2.4 tok/s).
- **PTI library issue**: If you see GPU device initialization failure, filter PTI from
  LD_LIBRARY_PATH: `NEW_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v pti | tr '\n' ':' | sed 's/:$//')`
  and `export LD_LIBRARY_PATH="$NEW_PATH"`
- **UMF missing**: If "No platforms found", add:
  `export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.0/lib:$LD_LIBRARY_PATH"`
- The 6.86 tok/s baseline was measured with 224 sequential D2H + vec_dot steps. After
  optimization, the GPU graph time (~14 ms at 13% split) becomes the bottleneck, so expect
  TG128 ≈ 1000/14.3 ≈ 70 tok/s (matching or slightly below GPU-only).
- DO NOT use VTune for profiling — it caused a kernel panic on this system.

---

## Performance Model

### Before (Measured)

| Phase | Time | Detail |
|-------|------|--------|
| GPU graph replay (partial MMVQ) | 12.2 ms | 87% of weights at 280 GB/s |
| D2H src1 × 224 | 70 ms | Sequential `.wait()` per copy |
| CPU vec_dot × 224 | 80 ms | Sequential TBB invocations |
| H2D output (batched) | 1 ms | Already good |
| **Total** | **~163 ms** | GPU + max(D2H+vec_dot, 0) |
| **TG128** | **6.86 tok/s** | 1000/145.7 |

### After (Estimated)

| Phase | Time | Detail |
|-------|------|--------|
| GPU graph replay (partial MMVQ) | 12.2 ms | Same as before |
| D2H src1 (batched, deduped) | 0.2 ms | ~128 unique × 16 KB = 2 MB |
| CPU vec_dot (single parallel_for) | 7-13 ms | ~500 MB at 40-75 GB/s |
| H2D output (batched) | 0.1 ms | ~700 KB total |
| **Total** | **~13-14 ms** | max(GPU, D2H+vec_dot+H2D) |
| **TG128** | **~70 tok/s** | Limited by GPU graph time |

### Why ~70 tok/s (Not Higher)

With graph replay, the GPU graph time (~12.2 ms for 87% of weights) is the FLOOR.
CPU work at 7-13 ms runs concurrently with GPU graph execution (submitted via
`sycl_ctx->stream()->ext_oneapi_graph()` which is async). The CPU work just needs
to finish before the next token's graph replay.

If CPU finishes before GPU: TG = 1000 / (12.2 + overhead) ≈ 70 tok/s
If GPU finishes before CPU: TG = 1000 / (CPU_time + overhead) ≈ 65-70 tok/s

At 13% CPU split, the GPU graph handles 87% of weights. The GPU time DECREASES
proportionally: 87% × 13.9 ms = 12.1 ms. So GPU is faster, CPU has less work.
The net effect: approximately GPU-only speed with a small CPU bonus.

---

## What This Enables (Future Work)

After P0 is complete, the tensor split infrastructure is ready for:

- **P1: HOST_COMPUTE integration** — For host-resident layers (VRAM budget < 100%), the GPU
  streams weight rows over PCIe while CPU reads from DRAM. Uses the same batched vec_dot.
  Requires: GPU DMMV/MMVQ on host-pinned memory, per-layer barrier coordination.

- **P2: Multi-GPU (B50)** — B50 adds 25 GB/s PCIe 5.0 bandwidth. Each GPU gets rows
  proportional to its PCIe bandwidth. Same batched vec_dot for CPU portion.
  Requires: multi-device queue management, per-GPU row assignment.

- **P3: Graph replay hybrid** — GPU-resident layers use graph replay (fast). Host-resident
  layers use tensor split with per-layer sync. Requires: mixed graph/non-graph dispatch.

- **P4: Persistent kernel** — Single GPU kernel launch with per-layer CPU sync via USM
  atomics. The batched vec_dot function from Task 2 becomes the CPU worker in the
  persistent kernel loop.
