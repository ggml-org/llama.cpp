# Batched host_task: Single Submission per CPU Block

> **For Claude:** Use subagent-driven-development to implement this plan. Use Serena MCP for all code exploration and editing.

**Goal:** Reduce ~480 individual host_task submissions per token to ~2-4 (one per CPU block), eliminating SYCL runtime overhead that dominates CPU offload TG latency.

**Architecture:** In HOST_COMPUTE mode, instead of each CPU kernel submitting its own host_task to gpu_q, the dispatch loop in `graph_compute_impl` collects consecutive CPU nodes into a batch, then submits ONE host_task that iterates through them all synchronously. Inside the batched host_task, `g_host_task_mode` is set to `false` so each kernel takes its sync/direct path.

**Tech Stack:** C++ / SYCL / Intel oneAPI

---

## Problem Analysis

### Current Per-Op Submission Pattern (HOST_COMPUTE mode)

When `GGML_SYCL_HOST_COMPUTE=1`, compute buffers are host-pinned USM. CPU ops access `t->data` directly. Each CPU kernel submits its own host_task to the in-order gpu_q:

```
graph_compute_impl loop:
  for each node:
    if node_on_cpu:
      compute_forward(node)                    // dispatches to CPU kernel
        → cpu_rms_norm() or cpu_add() etc:
          host_task=true  (host_task_mode_active())
          get_host_ptr(src) → returns t->data   // host-pinned, no staging
          get_host_output_ptr(dst) → returns t->data  // host-pinned, no staging
          gpu_q->submit({ cgh.host_task({ ... }) })  // INDIVIDUAL SUBMISSION
          // no flush_output — skipped in host_task path
        → cpu_mul_mat():
          async_mode=true  (ggml_sycl_cpu_offload_async_enabled())
          get_host_ptr(src0) → mmap weight ptr
          get_host_ptr(src1) → returns t->data
          get_host_output_ptr(dst) → staging (!)  // see issue below
          cpu_submit_async(gpu_q, deps, { cgh.host_task({...}) })  // INDIVIDUAL SUBMISSION
          flush_output(dst)  // staging→device copy (!)
```

**Per-token overhead for Mistral 7B (32 layers, all CPU):**
- ~15 CPU ops per layer × 32 layers = ~480 host_task submissions
- Each submission: SYCL handler creation + event bookkeeping + runtime dispatch ≈ 10-20µs
- Total: 480 × 15µs ≈ 7.2ms of pure submission overhead per token
- At 128 tok/s target → 7.8ms per token budget → submission overhead alone exceeds budget

### Two Distinct Kernel Dispatch Patterns

**Pattern A — Element-wise kernels** (RMS_NORM, ADD, MUL, SILU, GLU, SOFT_MAX, NORM, SCALE, CPY, ROPE):
```cpp
const bool host_task = host_task_mode_active();  // checks g_host_task_mode
// In host_task mode:
//   get_host_ptr() → returns t->data (host-pinned USM)
//   gpu_q->submit({ cgh.host_task({ compute... }) })
//   NO flush_output (output goes directly to host-pinned t->data)
```

**Pattern B — MUL_MAT and fused ops**:
```cpp
const bool async_mode = ggml_sycl_cpu_offload_async_enabled();  // DIFFERENT check
// In async mode:
//   get_host_ptr(src0) → weight from mmap/cache (host-accessible)
//   get_host_ptr(src1) → t->data (host-pinned USM)
//   get_host_output_ptr(dst) → staging buffer (!)
//   cpu_submit_async(gpu_q, ...) → host_task on gpu_q
//   flush_output(dst) → memcpy staging→device (but device ptr is host-pinned!)
```

**Key issue in Pattern B**: `get_host_output_ptr()` doesn't recognize HOST_COMPUTE buffers as host-accessible (because `ggml_backend_buffer_is_host()` returns false for SYCL buffer types), so it falls through to staging. Then `flush_output()` copies staging→"device" but the "device" pointer is actually host-pinned USM. This is a wasted memcpy but not incorrect.

### What Batching Saves

Instead of 480 individual host_task submissions:
```
[host_task: RMS_NORM] → [host_task: MUL] → [host_task: MUL_MAT] →
[host_task: ADD] → [host_task: RMS_NORM] → ... (480 submissions)
```

Batch into 2 per layer × 32 layers ≈ 64:
```
[host_task: { RMS_NORM; MUL; MUL_MAT; SILU; MUL_MAT; ADD; RMS_NORM; MUL; MUL_MAT; ... }]
```

Or better — one per CPU segment if no GPU islands interrupt:
```
[host_task: { all ~480 CPU ops for 32 layers }]
```

---

## Architecture

### Batch Collection in graph_compute_impl

The dispatch loop at `ggml-sycl.cpp:25573` already identifies CPU nodes via `node_cpu_flags`. We add batch collection:

```
                  graph_compute_impl
                         │
                    ┌────▼────┐
                    │ for each │
                    │  node    │
                    └────┬────┘
                         │
                 ┌───────▼───────┐
                 │ node_on_cpu?  │──no──► GPU dispatch (unchanged)
                 └───────┬───────┘
                         │ yes
                 ┌───────▼───────┐
                 │ Collect into  │
                 │ cpu_batch[]   │
                 └───────┬───────┘
                         │
              ┌──────────▼──────────┐
              │ End of CPU segment? │──no──► continue collecting
              │ (GPU node or end)   │
              └──────────┬──────────┘
                         │ yes
              ┌──────────▼──────────┐
              │ Submit ONE host_task│
              │ containing ALL ops  │
              │ in cpu_batch[]      │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ Inside host_task:   │
              │ g_host_task_mode=F  │
              │ for each node:      │
              │   compute_forward() │
              │ g_host_task_mode=T  │
              └─────────────────────┘
```

### Inside the Batched host_task

When `g_host_task_mode` is set to `false` inside the batched host_task, each kernel takes its **sync path**. The sync path for non-MUL_MAT kernels (`cpu_rms_norm`, `cpu_add`, etc.) does:

```cpp
// host_task = false → takes the else branch
sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();  // may be null
// if !host_task && !cpu_q → return false  ← PROBLEM: needs fix
```

This is a **critical issue**: the sync path requires a `cpu_q`, but in HOST_COMPUTE mode there may not be one. We need a **third path**: direct synchronous execution without any queue.

### The Three Execution Modes

After batching, CPU kernels need to support three modes:

| Mode | When | Queue | Execution |
|------|------|-------|-----------|
| `host_task` | HOST_COMPUTE, per-op (legacy) | gpu_q | `gpu_q->submit({ cgh.host_task({...}) })` |
| `batched` | HOST_COMPUTE, inside batch | none | Direct function call (no queue submission) |
| `async` | Non-HOST_COMPUTE, cpu_q available | cpu_q | `cpu_q->submit({ parallel_for/host_task })` |

The `batched` mode is the simplest: just call the compute lambda directly. No staging, no events, no queue submission.

### Data Flow in Batched Mode

In HOST_COMPUTE mode, compute buffers are host-pinned USM. Inside the batched host_task:

1. **Inputs**: `get_host_ptr()` returns `t->data` directly (host-pinned USM recognized by USM pointer type check at line 802-834)
2. **Weights**: `get_host_ptr()` returns mmap/cache pointers (already host-accessible)
3. **Outputs**: Write directly to `t->data` (host-pinned USM) — no staging needed
4. **Between ops**: Previous op's output at `t->data` is immediately readable by next op — zero copies

**Exception: MUL_MAT output path** — `get_host_output_ptr()` currently returns staging because `ggml_backend_buffer_is_host()` fails. Fix: add HOST_COMPUTE check.

### Staging/Event State Safety

Inside the batched host_task, we run on a SYCL runtime thread. The following must be safe:

| State | Issue | Mitigation |
|-------|-------|------------|
| `g_staging_bank` | Toggled by `staging_begin_op()` per kernel call | Not needed — HOST_COMPUTE skips staging |
| `g_staging_flush_pending` | Set by `flush_output()` | Not needed — no flush in batched mode |
| `g_cpu_chain_event_valid` | Set by `cpu_submit_async()` | Not used — no async submission in batch |
| `g_host_task_mode` | Must be false inside batch | Set false before loop, restore after |
| `g_retained_active` | Retained mode disabled | No-op, safe |
| Thread-local buffers | `src1_q_buf`, `src0_f32_buf` | Thread-local, initialized on first use per-thread — safe |

---

## Detailed Changes

### Change 1: Add `batched_mode` flag to cpu-dispatch.cpp (~15 lines)

Add a new global flag for batched execution mode, where kernels run as direct function calls:

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (after line 566)

```cpp
// BATCHED host_task mode: when active, CPU ops run as direct function calls
// inside a single batched host_task, not individual submissions.
// Activated by graph_compute_impl when collecting CPU segments.
static bool g_batched_mode = false;

static inline bool batched_mode_active() {
    return g_batched_mode;
}
```

**File**: `ggml/src/ggml-sycl/cpu-dispatch.hpp` (add declarations)

```cpp
void ggml_sycl_batched_mode_set(bool active);
bool ggml_sycl_batched_mode_active();
```

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (add implementation)

```cpp
void ggml_sycl_batched_mode_set(bool active) {
    g_batched_mode = active;
}

bool ggml_sycl_batched_mode_active() {
    return g_batched_mode;
}
```

### Change 2: Add batched execution path to all element-wise CPU kernels (~5 lines each, 10 kernels)

Each kernel that uses `host_task_mode_active()` needs a third path. The pattern is identical for all:

**Before** (example: `cpu_rms_norm`):
```cpp
static bool cpu_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool host_task = host_task_mode_active();
    sycl::queue * cpu_q = host_task ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !cpu_q) {
        return false;
    }
    // ... staging + data acquisition ...
    if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() { /* compute */ });
        });
    } else {
        // async path on cpu_q
    }
}
```

**After**:
```cpp
static bool cpu_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();
    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }
    // ... staging + data acquisition (unchanged — get_host_ptr returns t->data in HOST_COMPUTE) ...
    if (batched) {
        // Direct synchronous execution — no queue submission
        /* same compute code as host_task lambda body */
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() { /* compute */ });
        });
    } else {
        // async path on cpu_q (unchanged)
    }
}
```

**Affected kernels** (10 total):
1. `cpu_rms_norm` (line 1267)
2. `cpu_binary_op` (line 1370) — covers `cpu_add` and `cpu_mul`
3. `cpu_silu` (line 1522) — covers `cpu_unary`
4. `cpu_glu` (line 1588)
5. `cpu_soft_max` (line 1724)
6. `cpu_norm` (line 1865)
7. `cpu_scale` (line 1974)
8. `cpu_cpy` (line 2040)
9. `cpu_rope` (line 2100)

**Implementation**: Each kernel already has a compute lambda or inline code in its `host_task` branch. The `batched` path just calls that code directly. For kernels that define a `run_*` lambda (like `cpu_mul_mat`), call the lambda directly.

### Change 3: Add batched path to cpu_mul_mat (~10 lines)

`cpu_mul_mat` uses a different async mechanism. Add batched mode check:

**Before** (line 1239-1261):
```cpp
    if (async_mode) {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, &e1, gpu_q);
        sycl::event cpu_evt = cpu_submit_async(gpu_q, deps, ...);
        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt, true);
        }
        return true;
    }
    // Legacy sync path
    cpu_wait_chain_event();
    offload_wait_event(e0, offload_wait_reason::FALLBACK);
    offload_wait_event(e1, offload_wait_reason::FALLBACK);
    run_mul_mat();
    ...
```

**After**:
```cpp
    if (batched_mode_active()) {
        // Direct execution inside batched host_task — no submission, no events.
        // In HOST_COMPUTE mode: src0 from mmap, src1/dst from host-pinned t->data.
        run_mul_mat();
        // No flush_output: dst_data already points to host-pinned t->data
        // (after Change 4 fixes get_host_output_ptr).
        return true;
    }

    if (async_mode) {
        // ... unchanged ...
    }

    // Legacy sync path — unchanged
```

### Change 4: Fix get_host_output_ptr for HOST_COMPUTE buffers (~8 lines)

Currently `get_host_output_ptr` falls through to staging for HOST_COMPUTE buffers because `ggml_backend_buffer_is_host()` returns false. Add the same USM pointer type check that `get_host_ptr` uses (line 802-834):

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`, function `get_host_output_ptr` (line 946)

**Before**:
```cpp
static void * get_host_output_ptr(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }
    size_t nbytes = ggml_nbytes(t);
    return staging_ensure(g_staging_bank, 2, nbytes, gpu_q);
}
```

**After**:
```cpp
static void * get_host_output_ptr(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }
    // HOST_COMPUTE: SYCL-allocated host-pinned USM buffers are host-accessible
    // but ggml_backend_buffer_is_host() returns false for SYCL buffer types.
    // Check USM pointer type to detect host-accessible compute buffers.
    if (t->data) {
        try {
            sycl::context    ctx = ggml_sycl_get_device(device).default_queue().get_context();
            sycl::usm::alloc pt  = sycl::get_pointer_type(t->data, ctx);
            if (pt != sycl::usm::alloc::device) {
                return t->data;
            }
        } catch (...) {}
    }
    size_t nbytes = ggml_nbytes(t);
    return staging_ensure(g_staging_bank, 2, nbytes, gpu_q);
}
```

**Optimization**: Use the same `host_ptr_cache` that `get_host_ptr` uses (line 803) to avoid repeated `get_pointer_type` calls. Extract the cache to a shared helper:

```cpp
// Shared helper: check if a pointer is host-accessible via USM type.
// Caches results per base pointer to avoid repeated runtime queries.
static bool is_host_accessible_usm(void * ptr, int device) {
    static std::unordered_map<void *, bool> cache;
    auto it = cache.find(ptr);
    if (it != cache.end()) {
        return it->second;
    }
    bool is_host = true;  // assume host unless proven device
    try {
        sycl::context    ctx = ggml_sycl_get_device(device).default_queue().get_context();
        sycl::usm::alloc pt  = sycl::get_pointer_type(ptr, ctx);
        is_host              = (pt != sycl::usm::alloc::device);
    } catch (...) {}
    cache[ptr] = is_host;
    return is_host;
}
```

Then both `get_host_ptr` (line 802-834) and `get_host_output_ptr` use this shared helper.

### Change 5: Fix flush_output for HOST_COMPUTE buffers (~3 lines)

When output was written directly to host-pinned `t->data` (via fixed `get_host_output_ptr`), `flush_output` should be a no-op. Currently it checks `ggml_backend_buffer_is_host()` (which fails). Add the same check:

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`, function `flush_output` (line 901)

**After the existing buffer_is_host check** (line 906-908):
```cpp
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return;
    }
    // HOST_COMPUTE: host-pinned buffers don't need staging flush.
    if (t->data && is_host_accessible_usm(t->data, device)) {
        return;
    }
```

### Change 6: Fused ops batched path (~10 lines each, 2 fused ops)

The fused ops (`ggml_sycl_compute_fused_rms_norm_mul`, `ggml_sycl_compute_fused_add_rms_norm`) use `ggml_sycl_cpu_offload_async_enabled()` like MUL_MAT. Add batched path:

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

Each fused op already defines a `run_fused` lambda. Add before the async check:

```cpp
    if (batched_mode_active()) {
        run_fused();
        return true;
    }
```

### Change 7: Batch collection and submission in graph_compute_impl (~50 lines)

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`, inside `ggml_backend_sycl_graph_compute_impl`

Replace the per-node compute_forward dispatch for CPU nodes with batch collection:

**After the boundary sync block** (line ~25700, the `if (!node_is_island) { prev_on_cpu = ... }` block):

Replace the existing per-node dispatch (lines 25709-25755) with:

```cpp
            if (node_on_cpu) {
                // CPU op fusion check (unchanged, but only in non-batch mode)
                // ... existing fusion code ...

                // HOST_COMPUTE batching: collect consecutive CPU nodes
                if (host_compute_enabled) {
                    // Collect this node and all subsequent CPU nodes until
                    // a GPU node, GPU island, or end of graph.
                    std::vector<ggml_tensor *> cpu_batch;
                    cpu_batch.push_back(node);

                    int j = i + 1;
                    while (j < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[j];
                        if (!next || ggml_sycl_is_noop(next)) {
                            j++;
                            continue;
                        }
                        int8_t jflag = (j < (int)node_cpu_flags.size()) ? node_cpu_flags[j] : FLAG_UNKNOWN;
                        if (jflag != FLAG_CPU) {
                            break;  // hit GPU node or island → end of batch
                        }
                        cpu_batch.push_back(next);
                        j++;
                    }

                    // Submit ONE host_task for the entire CPU batch.
                    // Inside: set batched_mode=true so kernels run synchronously
                    // without individual queue submissions.
                    auto & sctx = *sycl_ctx;
                    gpu_q->submit([&](sycl::handler & cgh) {
                        cgh.host_task([&sctx, cpu_batch]() {
                            ggml_sycl_batched_mode_set(true);
                            ggml_sycl_host_task_mode_set(false);
                            for (ggml_tensor * n : cpu_batch) {
                                bool ok = ggml_sycl_compute_forward(sctx, n);
                                GGML_ASSERT(ok);
                            }
                            ggml_sycl_batched_mode_set(false);
                            ggml_sycl_host_task_mode_set(true);
                        });
                    });

                    // Advance loop index past the batch
                    i = j - 1;  // -1 because loop will i++
                    continue;
                }

                // Non-HOST_COMPUTE: existing per-op dispatch (unchanged)
            }
```

**Critical detail**: The `cpu_batch` vector is captured **by value** in the host_task lambda. The `sctx` reference must remain valid (it's the backend context, which outlives the host_task). The `ggml_tensor *` pointers in `cpu_batch` point to graph nodes that remain valid for the graph's lifetime.

**Op fusion integration**: Op fusion (RMS_NORM+MUL, ADD+RMS_NORM) should happen **inside** the batched host_task, not before it. Move the fusion logic into the batched dispatch:

```cpp
                        cgh.host_task([&sctx, cpu_batch]() {
                            ggml_sycl_batched_mode_set(true);
                            ggml_sycl_host_task_mode_set(false);
                            for (size_t bi = 0; bi < cpu_batch.size(); bi++) {
                                ggml_tensor * n = cpu_batch[bi];
                                // Try fusion with next node in batch
                                if (bi + 1 < cpu_batch.size()) {
                                    ggml_tensor * next = cpu_batch[bi + 1];
                                    if (n->op == GGML_OP_RMS_NORM && next->op == GGML_OP_MUL
                                        && next->src[0] == n) {
                                        if (ggml_sycl_compute_fused_rms_norm_mul(sctx, n, next)) {
                                            bi++;
                                            continue;
                                        }
                                    }
                                    if (n->op == GGML_OP_ADD && next->op == GGML_OP_RMS_NORM
                                        && next->src[0] == n) {
                                        if (ggml_sycl_compute_fused_add_rms_norm(sctx, n, next)) {
                                            bi++;
                                            continue;
                                        }
                                    }
                                }
                                bool ok = ggml_sycl_compute_forward(sctx, n);
                                GGML_ASSERT(ok);
                            }
                            ggml_sycl_batched_mode_set(false);
                            ggml_sycl_host_task_mode_set(true);
                        });
```

### Change 8: Skip staging_begin_op() in batched mode (~2 lines)

`staging_begin_op()` toggles staging banks and waits on pending events. In batched mode with HOST_COMPUTE, there's no staging to manage. Add early return:

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`, function `staging_begin_op` (line 664)

```cpp
static void staging_begin_op() {
    // Batched mode in HOST_COMPUTE: no staging buffers used.
    if (g_batched_mode) {
        return;
    }
    // ... existing code ...
```

### Change 9: Skip get_retained_or_staging_output staging fallback in batched mode (~3 lines)

In batched mode, outputs always go to `t->data` directly (host-pinned USM). The retained/staging output helper shouldn't allocate staging:

**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`, function `get_retained_or_staging_output` (line 958)

```cpp
static void * get_retained_or_staging_output(ggml_tensor * dst, int device, sycl::queue * gpu_q, bool * retained) {
    // Batched mode: output directly to host-pinned t->data
    if (g_batched_mode && dst->data && is_host_accessible_usm(dst->data, device)) {
        *retained = false;
        return dst->data;
    }
    // ... existing code ...
```

---

## Critical Files

| File | Change | Description |
|------|--------|-------------|
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | Changes 1-6, 8-9 | Batched mode flag, kernel paths, staging fixes, USM helper |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | Change 1 | API declarations for batched mode |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Change 7 | Batch collection and submission in dispatch loop |

## What This Does NOT Change

- GPU kernel code — unaffected
- Non-HOST_COMPUTE CPU offload — the async/parallel_for path remains unchanged
- Graph replay — batched host_task still goes through the graph recording path
- Unified cache — no changes
- Buffer allocation — HOST_COMPUTE buffer type unchanged
- Op fusion — moved inside batch but same logic
- Boundary sync — GPU↔CPU transitions unchanged

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Thread-local buffers in kernels | Thread-local init happens on host_task thread — same thread runs all batched ops, so buffers persist across ops in the batch |
| `staging_begin_op` modifies globals | Skipped entirely in batched mode (Change 8) |
| `flush_output` called from sync path | Becomes no-op via is_host_accessible_usm check (Change 5) |
| `cpu_wait_chain_event` in sync path | No-op because `g_cpu_chain_event_valid` is false in HOST_COMPUTE mode |
| `offload_wait_event` on staging events | No-op because events are default-constructed (never submitted) |
| SYCL runtime thread safety | All globals are single-threaded within one host_task — safe |
| Graph recording compatibility | Batch submission is a single host_task — records as one node in SYCL graph |

## Verification

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# 1. GPU-only correctness (HOST_COMPUTE not active → batching inactive)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# 2. HOST_COMPUTE + CPU offload 43% — mixed GPU/CPU layers
GGML_SYCL_HOST_COMPUTE=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=43 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# 3. HOST_COMPUTE + CPU offload 30% — all CPU layers
GGML_SYCL_HOST_COMPUTE=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expect: "6, 7, 8, 9, 10"

# 4. GPU-only performance (wait 30s between runs)
LD_LIBRARY_PATH="build/bin:$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v pti | tr '\n' ':' | sed 's/:$//')" \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expect: PP512 >= 1200, TG128 >= 68 (no regression)

# 5. HOST_COMPUTE TG performance
GGML_SYCL_HOST_COMPUTE=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
# Before: baseline per-op host_task rate
# After: expect improvement from 480→~64 host_task submissions
```
