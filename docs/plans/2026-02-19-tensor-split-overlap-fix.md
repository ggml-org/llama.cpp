# Tensor Split GPU/CPU Overlap Fix

> **For Claude:** Use superpowers:subagent-driven-development to implement this plan.

**Goal:** Fix tensor split serialization so GPU and CPU actually run in parallel, achieving ~84 tok/s TG (up from 72 tok/s GPU-only baseline).

**Architecture:** Two stacking optimizations: (1) Reorder operations in `ggml_sycl_mul_mat_tensor_split()` to stage src1 before GPU work for GPU/CPU overlap (+14%). (2) Drop `stream->wait()` and `memcpy().wait()` after CPU vec_dot — in-order queue implicit ordering guarantees correctness without explicit sync (+5-7%). Auto-disable graph replay for TG when tensor split enabled (graph only adds 2.8%).

---

## Background: Why Tensor Split Is Currently Broken

**Problem 1 — Dead code:** With graph replay enabled (default), tensor split only runs during the warmup pass (token 1). The recording pass captures full MMVQ, and all subsequent tokens replay the graph — `compute_impl()` is never called, so tensor split never executes.

**Problem 2 — Serialization:** Even when tensor split runs (graph disabled), the `.wait()` on the src1 D2H memcpy (`ggml-sycl.cpp:20111`) blocks until the preceding GPU MMVQ completes (in-order queue semantics). CPU vec_dot starts only AFTER GPU finishes — zero overlap.

**Current performance with `GGML_SYCL_TENSOR_SPLIT=13`:**

| Config | TG128 tok/s |
|--------|-------------|
| GPU-only + graph replay | 72.03 |
| GPU-only, no graph | 70.04 |
| Tensor split 13%, no graph (serialized) | 10.93 |
| Tensor split 13%, host_task graph | 5.37 |

**Target:** ~84 tok/s (two optimizations stack: overlap + async H2D)

**Math:** GPU BW = 280 GB/s, CPU DDR5-5600 BW ~40-50 GB/s. Optimal split = cpu_bw/(gpu_bw+cpu_bw) = 12.5-15%. At optimal split, both devices finish simultaneously and total time = max(GPU_time, CPU_time) ≈ 0.875 * GPU_only_time.

| Optimization | Mechanism | TG128 estimate |
|---|---|---|
| GPU/CPU overlap (operation reorder) | Pre-stage src1 before GPU work | ~80 tok/s |
| Async H2D (drop unnecessary waits) | In-order queue implicit ordering | ~84 tok/s |

---

## Task 1: Extract shared `ggml_sycl_tensor_split_pct()` helper

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

The `GGML_SYCL_TENSOR_SPLIT` env var is read as a static local in `ggml_sycl_mul_mat()` (line 20164). We also need it in the graph compute function (Task 3). Extract to a shared helper.

### Implementation

Add before `ggml_sycl_mul_mat_tensor_split()` (around line 19828, before `g_split_staging`):

```cpp
// Tensor split CPU percentage (0 = disabled).
// Shared between mul_mat dispatch and graph compute.
static int ggml_sycl_tensor_split_pct() {
    static const int pct = []() {
        const char * env = std::getenv("GGML_SYCL_TENSOR_SPLIT");
        return env ? std::atoi(env) : 0;
    }();
    return pct;
}
```

Replace the static local in `ggml_sycl_mul_mat()` (lines 20164-20167):
```cpp
// Before:
static const int tensor_split_cpu_pct = []() {
    const char * env = std::getenv("GGML_SYCL_TENSOR_SPLIT");
    return env ? std::atoi(env) : 0;
}();
if (tensor_split_cpu_pct > 0) {
    if (ggml_sycl_mul_mat_tensor_split(ctx, src0, src1, dst,
                                        tensor_split_cpu_pct, effective_layout)) {

// After:
if (ggml_sycl_tensor_split_pct() > 0) {
    if (ggml_sycl_mul_mat_tensor_split(ctx, src0, src1, dst,
                                        ggml_sycl_tensor_split_pct(), effective_layout)) {
```

### Verification
```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)
# Build must succeed
```

---

## Task 2: Fix GPU/CPU overlap — pre-stage src1 before GPU work

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_sycl_mul_mat_tensor_split()`

### Root Cause

The function has a common section (lines 19988-20003) that submits Q8 quantize + MMVQ to the GPU queue BEFORE branching into recording vs non-recording paths. In the non-recording path, `stream->memcpy(src1_host, src1->data, ...).wait()` at line 20111 is queued BEHIND the already-submitted MMVQ on the in-order queue. The `.wait()` blocks until the memcpy event completes, which requires all preceding ops (Q8+MMVQ) to finish first. CPU vec_dot only starts after GPU finishes.

### Fix Strategy

Move weight lookup + src1 D2H staging into the common section BEFORE Q8 quantize. This way the D2H memcpy is the FIRST operation queued, `.wait()` returns after just the 16KB copy (~0.4us), and GPU Q8+MMVQ run async while CPU vec_dot runs concurrently.

Additionally, eliminate both `stream->wait()` and `memcpy().wait()` after CPU vec_dot by relying on in-order queue implicit ordering. The H2D output memcpy is queued after MMVQ, so subsequent ops execute after it. The NEXT MUL_MAT's src1 D2H `.wait()` ensures output_buf is consumed before reuse. `ggml_backend_sycl_synchronize()` at line 24227 calls `stream->wait_and_throw()` after graph compute, ensuring the final H2D completes.

**Safety proof for dropping waits:**
1. `cpu_vec_dot()` completes before we queue H2D → output_buf is fully written
2. H2D is queued after MMVQ on in-order queue → executes after GPU finishes
3. Next ops (NORM, ADD, etc.) are queued after H2D → they see complete `dst`
4. Next MUL_MAT's `src1 D2H .wait()` drains preceding queue ops → output_buf safe to reuse
5. `ggml_backend_sycl_synchronize()` does `wait_and_throw()` after all tokens → final H2D completes

### Exact Changes

**Step A**: After line 19985 (`const size_t cpu_weight_bytes = N_cpu * src0_row_bytes;`) and BEFORE line 19988 (`ggml_sycl_pool_alloc<char> src1_q8_alloc`), insert pre-staging block:

```cpp
    // --- Pre-stage for non-recording overlap ---
    // Move weight lookup + src1 D2H staging BEFORE GPU work so CPU vec_dot
    // can overlap with GPU Q8 quantize + MMVQ. Only for non-recording path;
    // graph recording path handles staging via graph nodes.
    const size_t out_bytes = N_cpu * sizeof(float);
    void * overlap_weights = nullptr;
    bool   src1_prestaged  = false;

    if (!g_ggml_sycl_graph_recording) {
        // Weight lookup (cached after first token, no .wait())
        if (src0->name) {
            overlap_weights = split_get_cached_weights(
                src0->name, src0_aos, N_gpu, src0_row_bytes, cpu_weight_bytes, stream);
        }
        if (!overlap_weights) {
            split_weight_staging_ensure(cpu_weight_bytes, stream);
            if (g_split_weight_staging.data) {
                stream->memcpy(g_split_weight_staging.data,
                               src0_aos + N_gpu * src0_row_bytes, cpu_weight_bytes).wait();
                overlap_weights = g_split_weight_staging.data;
            }
        }

        if (overlap_weights) {
            split_staging_ensure(src1_bytes, out_bytes, stream);
            if (g_split_staging.src1_host && g_split_staging.output) {
                // D2H src1: queued FIRST on empty queue, .wait() returns in ~0.4us
                stream->memcpy(g_split_staging.src1_host, src1->data, src1_bytes).wait();
                src1_prestaged = true;
            }
        }
    }
```

**Step B**: Replace the non-recording path (lines 20081-20124) with the overlap version:

```cpp
    // ===================================================================
    // NON-RECORDING PATH: overlapped GPU + CPU execution
    // ===================================================================

    if (src1_prestaged) {
        // GPU Q8 + MMVQ already in flight (submitted above).
        // CPU vec_dot runs CONCURRENTLY with GPU.
        ggml_sycl_cpu_vec_dot_rows(src0->type, static_cast<int>(ne00),
                                    overlap_weights,
                                    g_split_staging.src1_host,
                                    g_split_staging.output,
                                    static_cast<int>(N_cpu));

        // Queue H2D copy — NO stream->wait(), NO .wait().
        // In-order queue guarantees: MMVQ completes → H2D executes → next op sees complete dst.
        // Next MUL_MAT's src1 D2H .wait() ensures output_buf is consumed before reuse.
        // ggml_backend_sycl_synchronize() at graph compute end ensures final H2D completes.
        stream->memcpy(dst_dd + N_gpu, g_split_staging.output, out_bytes);
    } else {
        // Fallback: serialized path (pre-staging failed or first-time alloc issue)
        void * host_weights = nullptr;
        if (src0->name) {
            host_weights = split_get_cached_weights(src0->name, src0_aos, N_gpu,
                                                     src0_row_bytes, cpu_weight_bytes, stream);
        }
        if (!host_weights) {
            split_weight_staging_ensure(cpu_weight_bytes, stream);
            if (!g_split_weight_staging.data) {
                return false;
            }
            stream->memcpy(g_split_weight_staging.data,
                           src0_aos + N_gpu * src0_row_bytes, cpu_weight_bytes).wait();
            host_weights = g_split_weight_staging.data;
        }

        split_staging_ensure(src1_bytes, out_bytes, stream);
        if (!g_split_staging.src1_host || !g_split_staging.output) {
            return false;
        }
        stream->memcpy(g_split_staging.src1_host, src1->data, src1_bytes).wait();

        ggml_sycl_cpu_vec_dot_rows(src0->type, static_cast<int>(ne00),
                                    host_weights,
                                    g_split_staging.src1_host,
                                    g_split_staging.output,
                                    static_cast<int>(N_cpu));

        // Same async H2D as overlap path — no waits needed.
        stream->memcpy(dst_dd + N_gpu, g_split_staging.output, out_bytes);
    }

    return true;
```

### Verification
```bash
# Correctness (graph disabled to test overlap path directly)
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15"

# Performance (no graph, overlap only)
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
# Expected: TG128 > 60 tok/s (was 10.93 with serialized, should be much faster)
```

---

## Task 3: Auto-disable graph for TG when tensor split enabled

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`, function `ggml_backend_sycl_graph_compute()`

### Why

Graph replay captures full MMVQ (all rows GPU). On replay, `compute_impl()` never runs, so tensor split never executes for tokens 3+. Graph replay adds only 2.8% for TG, while tensor split overlap adds ~14%. Auto-disabling graph for decode phase is a net +11.2% win.

PP (prompt processing) continues to use graph replay (tensor split only applies to batch=1 TG).

### Implementation

After line 31848 (the `moe_graphs_disabled_once` reset block) and before line 31850 (`if (use_sycl_graph) {`), add:

```cpp
    // Tensor split provides ~14% TG improvement via GPU/CPU overlap, while
    // graph replay only adds ~2.8% for TG (MMVQ fast-path already eliminates
    // most kernel dispatch overhead). Auto-disable graph for decode phase
    // when tensor split is enabled to allow overlap for every token.
    if (use_sycl_graph && cached_is_decode && ggml_sycl_tensor_split_pct() > 0) {
        static bool logged = false;
        if (!logged) {
            GGML_LOG_INFO("[SYCL] Tensor split %d%%: disabling graph replay for TG (GPU/CPU overlap)\n",
                          ggml_sycl_tensor_split_pct());
            logged = true;
        }
        use_sycl_graph = false;
    }
```

### Verification
```bash
# Tensor split + graph auto-disable produces correct output
GGML_SYCL_TENSOR_SPLIT=13 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15"

# GPU-only (no tensor split) still uses graph replay
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DEBUG=1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 16 2>&1 | \
  grep -c "SYCL-GRAPH.*execute"
# Expected: > 0 (graph replay still active for GPU-only)
```

---

## Task 4: Remove dead host_task graph recording path

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

With Task 3 auto-disabling graph for TG when tensor split is enabled, the host_task graph recording path (lines 20008-20078) is unreachable dead code.

### Changes

1. Remove `ggml_sycl_tensor_split_graph_enabled()` function (lines 19922-19931)
2. Simplify recording guard to unconditional return:
   ```cpp
   // During graph recording, tensor split is disabled. Graph is auto-disabled
   // for TG when tensor split is active (Task 3), so this only fires during
   // PP recording or if graph is forced on for diagnostic purposes.
   if (g_ggml_sycl_graph_recording) {
       return false;
   }
   ```
3. Remove the entire `if (g_ggml_sycl_graph_recording) { ... }` block (lines 20008-20078)

### Verification
```bash
ninja -C build -j $(nproc)
# Build must succeed

# Same correctness and performance tests as Tasks 2-3
```

---

## Task 5: Performance benchmark and sweep

### Benchmarks (30-45s cooldown between each)
```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# 1. GPU-only baseline
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expected: PP512 >= 1200, TG128 ~72 tok/s

sleep 45

# 2. Tensor split sweep
for pct in 10 13 15 20 25; do
  echo "=== TENSOR_SPLIT=$pct ==="
  GGML_SYCL_TENSOR_SPLIT=$pct \
    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
  sleep 45
done
# Target: best TG128 > 78 tok/s (conservative; theoretical ~84 with async H2D)

# 3. Verify PP unaffected (tensor split only for TG batch=1)
# PP numbers from sweep should be >= 1200 tok/s
```

### Success Criteria
- TG128 with best tensor split % > 78 tok/s (conservative; target 84)
- PP512 unchanged from baseline (>= 1200 tok/s)
- Deterministic output matches GPU-only baseline
- GPU-only path (TENSOR_SPLIT=0) has zero regression
