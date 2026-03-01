# HeteGen-Style Tensor Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Split each TG MUL_MAT across GPU (MMVQ on SOA device data) and CPU (vec_dot on AOS host data) to utilize idle CPU memory bandwidth for 7-25% TG throughput improvement.

**Architecture:** During TG (batch=1), the weight matrix is row-split: GPU processes rows [0, N_gpu) via existing MMVQ kernel (SOA layout), CPU processes rows [N_gpu, N) via vec_dot (AOS/mmap host data). Both compute simultaneously, writing non-overlapping output regions. Phase 1 validates correctness without graph replay; Phase 2 integrates with SYCL graph replay for full performance (CPU work runs concurrent with graph submission).

**Tech Stack:** SYCL (oneAPI), TBB (parallel_for), ggml quantized vec_dot, MMVQ kernels with existing row_low/row_high support

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1 | CPU vec_dot row helper + host_ptr lookup (cpu-dispatch files) |
| B | 2 | GPU partial MMVQ dispatch + staging + gate (ggml-sycl.cpp) |
| — | 3 | Integration: wire CPU into split + correctness test |
| — | 4 | Phase 2: graph replay integration |

### Dependency Graph

```
1 ──┐
    ├──► 3 ──► 4
2 ──┘
```

Task 1 and Task 2 are fully parallel (different files, no shared state).
Task 3 merges both tracks. Task 4 adds graph replay.

### File Ownership Map

| File | Tasks | Conflict Risk |
|------|-------|---------------|
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 1 | None (Track A only) |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | 1 | None (Track A only) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 2, 3, 4 | Sequential (same track after merge) |

---

## Background Context for Implementers

### System Overview
- **Arc B580 GPU**: 280 GB/s device memory bandwidth
- **Core Ultra 7 265K CPU**: DDR5-5600 dual-channel, ~50-70 GB/s usable bandwidth
- **Current TG128**: ~72 tok/s (GPU-only with graph replay), ~70 tok/s (without graph — TG_FAST path is already fast, graph adds only +3%)
- **Target**: 75-88 tok/s (Phase 2 with graph replay + tensor split)

### Key Existing Infrastructure (do NOT modify these)
- **MMVQ row splitting**: `mmvq.cpp:4679` `ggml_sycl_op_mul_mat_vec_q()` already accepts `row_low`/`row_high` params. Kernels use `total_nrows` (full ne01) for SOA offset math, `nrows` (row_diff) for bounds. Built for multi-GPU TP.
- **Host weight pointers**: `cpu-dispatch.cpp:75` `g_host_ptr_map` stores mmap host pointers for all weights, registered during `set_tensor`. Lookup: `cpu_dispatch_lookup_host_ptr(name)`.
- **TBB arena**: `ggml_sycl_cpu_arena()` returns persistent `tbb::task_arena` (cpu-dispatch.cpp). Used for parallel CPU work.
- **vec_dot per quant type**: `ggml_get_type_traits_cpu(type)->vec_dot` — e.g. Q4_0 uses Q8_0 activations.
- **Quantize from float**: `ggml_get_type_traits_cpu(vec_dot_type)->from_float` — quantizes float32 activations.

### ggml MUL_MAT Convention
```
C^T = A * B^T  =>  C = B * A^T
src0 = A (weights)     [ne00=K, ne01=N]
src1 = B (activations) [ne10=K, ne11=M]
dst  = C (output)      [ne0=N,  ne1=M]
```
For TG batch=1: M=1, so dst is a vector of N floats.

### Build and Test Commands
```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# Correctness test
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"

# Performance benchmark (wait 30s between runs for thermal cooldown)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
```

---

### Task 1: CPU vec_dot Row Helper

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (add ~50 lines after line 238)
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.hpp` (add 5 lines after line 59)

**Description:**

Add a standalone function that computes vec_dot for a contiguous range of weight rows
on the CPU. This is the CPU-side compute kernel for tensor split. It quantizes the
float32 activation to the appropriate Q8 type, then runs vec_dot per row using TBB
parallel_for. Also expose the existing host_ptr lookup as a public API.

**Acceptance Criteria:**

- [ ] `ggml_sycl_cpu_vec_dot_rows()` compiles and is callable from other TUs
- [ ] `ggml_sycl_cpu_dispatch_get_host_ptr()` is public and returns correct pointers
- [ ] Function uses existing `ggml_sycl_cpu_arena()` for TBB parallelism
- [ ] Handles Q4_0, Q8_0, Q4_K, Q6_K quant types (all supported by vec_dot)
- [ ] Build succeeds: `ninja -C build -j $(nproc)` with zero new warnings
- [ ] GPU-only baseline unaffected (no regression — code is only called when tensor split is active)

**Implementation Guide:**

1. **Add header declarations to `cpu-dispatch.hpp`**

Open `ggml/src/ggml-sycl/cpu-dispatch.hpp`. After the existing `ggml_sycl_is_host_accessible_usm` declaration (around line 59), add:

```cpp
// Tensor-split: compute vec_dot for a contiguous range of weight rows on CPU.
// src0_host must point to the FIRST row to process (pre-offset by caller).
// Output buffer receives n_rows float results.
void ggml_sycl_cpu_vec_dot_rows(ggml_type type, int ne00,
                                 const void * src0_host, const float * src1_host,
                                 float * output, int n_rows);

// Lookup the registered host (mmap) pointer for a weight tensor by name.
// Returns nullptr if not registered.
const void * ggml_sycl_cpu_dispatch_get_host_ptr(const char * name);
```

2. **Implement `ggml_sycl_cpu_dispatch_get_host_ptr()` in `cpu-dispatch.cpp`**

After the existing `cpu_dispatch_lookup_host_ptr()` (line 238), add a public wrapper:

```cpp
const void * ggml_sycl_cpu_dispatch_get_host_ptr(const char * name) {
    return cpu_dispatch_lookup_host_ptr(name);
}
```

3. **Implement `ggml_sycl_cpu_vec_dot_rows()` in `cpu-dispatch.cpp`**

Add after the public host_ptr wrapper. This function:
- Gets type traits for the weight type
- Quantizes src1 (float32 activation) to the vec_dot input type (e.g., Q8_0)
- Runs vec_dot per row using TBB parallel_for

```cpp
void ggml_sycl_cpu_vec_dot_rows(ggml_type type, int ne00,
                                 const void * src0_host, const float * src1_host,
                                 float * output, int n_rows) {
    if (n_rows <= 0 || !src0_host || !src1_host || !output) {
        return;
    }

    const auto * cpu_traits = ggml_get_type_traits_cpu(type);
    if (!cpu_traits || !cpu_traits->vec_dot) {
        GGML_LOG_WARN("[TENSOR-SPLIT] No vec_dot for type %d, skipping CPU rows\n", type);
        return;
    }

    const ggml_type        vec_dot_type  = cpu_traits->vec_dot_type;
    const auto *           vdt_traits    = ggml_get_type_traits_cpu(vec_dot_type);
    ggml_from_float_t      from_float_fn = vdt_traits ? vdt_traits->from_float : nullptr;
    if (!from_float_fn) {
        GGML_LOG_WARN("[TENSOR-SPLIT] No from_float for vec_dot_type %d\n", vec_dot_type);
        return;
    }

    // Quantize src1 (activation) from float32 to Q8 format
    const size_t q_row_size = ggml_row_size(vec_dot_type, ne00);
    // Thread-local to avoid repeated allocation across tokens
    static thread_local std::vector<uint8_t> src1_q_buf;
    src1_q_buf.resize(q_row_size);
    from_float_fn(src1_host, src1_q_buf.data(), ne00);

    const size_t   row_stride  = ggml_row_size(type, ne00);
    uint8_t *      src1_q_data = src1_q_buf.data();

#if GGML_SYCL_HAS_TBB
    if (n_rows > 1) {
        ggml_sycl_cpu_arena().execute([&] {
            ggml_sycl_tbb::parallel_for(
                ggml_sycl_tbb::blocked_range<int>(0, n_rows),
                [&, src1_q_data](const ggml_sycl_tbb::blocked_range<int> & r) {
                    for (int i = r.begin(); i < r.end(); i++) {
                        const void * row = (const char *) src0_host + (size_t) i * row_stride;
                        float        dot_result;
                        cpu_traits->vec_dot(ne00, &dot_result, sizeof(float),
                                            row, 0, src1_q_data, 0, 1);
                        output[i] = dot_result;
                    }
                });
        });
        return;
    }
#endif
    // Single-row or no-TBB fallback
    for (int i = 0; i < n_rows; i++) {
        const void * row = (const char *) src0_host + (size_t) i * row_stride;
        float        dot_result;
        cpu_traits->vec_dot(ne00, &dot_result, sizeof(float),
                            row, 0, src1_q_buf.data(), 0, 1);
        output[i] = dot_result;
    }
}
```

**Important notes for `ggml_sycl_cpu_arena()`**: This function is already declared
and defined in cpu-dispatch.cpp. Search for `ggml_sycl_cpu_arena` to find the
declaration — it returns a `tbb::task_arena &`. The function might be declared with
a different signature. Check line ~1050-1100 area. If it's declared as
`static tbb::task_arena &`, you may need to make it non-static or use an extern.
Actually, check the header — it may already be declared there.

4. **Build and verify no regression**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: `6, 7, 8, 9, 10` ... (normal output, no regression — new code not yet called)

**Commit:**
```bash
git add ggml/src/ggml-sycl/cpu-dispatch.cpp ggml/src/ggml-sycl/cpu-dispatch.hpp
git commit -m "sycl: add CPU vec_dot row helper for tensor split"
```

---

### Task 2: GPU Partial MMVQ + Staging + Tensor Split Gate

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (add ~100 lines: staging struct, tensor_split function, gate)

**Description:**

Add the tensor split dispatch infrastructure to ggml-sycl.cpp. This includes:
(a) persistent staging buffers for src1 and CPU output,
(b) the main `ggml_sycl_mul_mat_tensor_split()` function that computes the row split,
    submits GPU partial MMVQ, and copies CPU output to device,
(c) the gate in `ggml_sycl_mul_mat()` TG fast-path that activates tensor split.

In this task, the CPU portion is a PLACEHOLDER (fills output with zeros). Task 3 wires
in the actual CPU vec_dot call.

**Acceptance Criteria:**

- [ ] `GGML_SYCL_TENSOR_SPLIT=0` (default) → normal GPU-only path, zero overhead
- [ ] `GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1` → builds and runs
- [ ] GPU partial MMVQ produces correct output for rows [0, N_gpu)
- [ ] CPU output portion is zeros (placeholder) — Task 3 adds real computation
- [ ] Build succeeds with zero new warnings
- [ ] GPU-only baseline unaffected (tensor split disabled by default)

**Implementation Guide:**

1. **Add staging buffers** (before `ggml_sycl_mul_mat`, around line 19825)

```cpp
// ---------------------------------------------------------------------------
// Tensor split: GPU+CPU cooperative MUL_MAT staging
// ---------------------------------------------------------------------------
static struct {
    float * src1_host = nullptr;   // host-pinned src1 staging (float32)
    float * output    = nullptr;   // host-pinned CPU output (float32)
    size_t  src1_size = 0;         // allocated bytes for src1
    size_t  out_size  = 0;         // allocated bytes for output
    sycl::queue * alloc_q = nullptr; // queue used for allocation (for free)
} g_split_staging;

static void split_staging_ensure(size_t src1_bytes, size_t out_bytes, sycl::queue * q) {
    if (g_split_staging.src1_size < src1_bytes) {
        if (g_split_staging.src1_host) {
            sycl::free(g_split_staging.src1_host, *q);
        }
        g_split_staging.src1_host = (float *) sycl::malloc_host(src1_bytes, *q);
        g_split_staging.src1_size = src1_bytes;
        g_split_staging.alloc_q   = q;
    }
    if (g_split_staging.out_size < out_bytes) {
        if (g_split_staging.output) {
            sycl::free(g_split_staging.output, *q);
        }
        g_split_staging.output  = (float *) sycl::malloc_host(out_bytes, *q);
        g_split_staging.out_size = out_bytes;
        g_split_staging.alloc_q  = q;
    }
}
```

2. **Implement `ggml_sycl_mul_mat_tensor_split()`**

Add after the staging helpers:

```cpp
// Tensor split: cooperative GPU+CPU MUL_MAT for TG (batch=1).
// Splits weight rows: GPU gets [0, N_gpu) via MMVQ, CPU gets [N_gpu, N) via vec_dot.
// Returns true if handled, false to fall back to normal GPU-only dispatch.
static bool ggml_sycl_mul_mat_tensor_split(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor *         src0,
        const ggml_tensor *         src1,
        ggml_tensor *               dst,
        int                         cpu_pct,
        layout_mode                 src0_layout) {

    const int64_t ne00 = src0->ne[0];  // K (columns)
    const int64_t ne01 = src0->ne[1];  // N (rows, output size)

    // Compute split
    int64_t N_cpu = ne01 * cpu_pct / 100;
    int64_t N_gpu = ne01 - N_cpu;

    // Too few CPU rows — not worth the overhead
    if (N_cpu < 32) {
        return false;
    }

    // Get host weight pointer for CPU portion
    const void * host_ptr = ggml_sycl_cpu_dispatch_get_host_ptr(src0->name);
    if (!host_ptr) {
        return false;  // No host pointer registered — fallback to GPU-only
    }

    GGML_SYCL_DEBUG("[TENSOR-SPLIT] %s: N=%lld N_gpu=%lld N_cpu=%lld cpu_pct=%d\n",
                    src0->name ? src0->name : "?",
                    (long long) ne01, (long long) N_gpu, (long long) N_cpu, cpu_pct);

    sycl::queue * stream = ctx.stream();
    const int     device = ctx.device;

    // Ensure staging buffers are allocated
    const size_t src1_bytes = ggml_nbytes(src1);
    const size_t out_bytes  = N_cpu * sizeof(float);
    split_staging_ensure(src1_bytes, out_bytes, stream);

    // Stage src1 (activation) to host — tiny for TG (16KB)
    stream->memcpy(g_split_staging.src1_host, src1->data, src1_bytes).wait();

    // --- GPU portion: partial MMVQ for rows [0, N_gpu) ---

    // Quantize src1 for GPU (Q8_1 format)
    const int64_t K         = ne00;
    const int64_t K_padded  = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t  q8_1_size = K_padded / QK8_1 * sizeof(block_q8_1);

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), q8_1_size);
    char * src1_ddq = src1_q8_alloc.get();

    // Quantize src1 to Q8_1 with SOA reorder
    const float * src1_ddf = (const float *) src1->data;
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
        src1_ddf, src1_ddq, K, 1, K_padded, stream);

    // Get GPU weight pointer (SOA layout)
    const char * src0_dd = (const char *) ggml_sycl_get_layout_ptr_for(src0, device, src0_layout);
    if (!src0_dd) {
        return false;  // SOA pointer unavailable
    }

    // GPU output: first N_gpu floats of dst
    float * dst_dd = (float *) dst->data;

    // Zero the GPU output region (MMVQ may not write all elements cleanly)
    stream->memset(dst_dd, 0, N_gpu * sizeof(float));

    // Submit GPU MMVQ for partial rows
    ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                                src0_dd,                // src0_dd_i
                                nullptr,                // src1_ddf_i (not used by MMVQ)
                                src1_ddq,               // src1_ddq_i (Q8_1 quantized)
                                dst_dd,                 // dst_dd_i
                                0,                      // row_low
                                N_gpu,                  // row_high
                                1,                      // src1_ncols (batch=1)
                                K_padded,               // src1_padded_row_size
                                stream);

    // --- CPU portion: placeholder (zeros) ---
    // Task 3 will replace this with actual ggml_sycl_cpu_vec_dot_rows() call
    memset(g_split_staging.output, 0, out_bytes);

    // Wait for GPU to complete
    stream->wait();

    // Copy CPU output to device dst[N_gpu..N-1]
    stream->memcpy(dst_dd + N_gpu, g_split_staging.output, out_bytes).wait();

    return true;
}
```

3. **Add the tensor split gate in `ggml_sycl_mul_mat()`**

Inside the TG fast-path block (around line 19863), BEFORE the existing
`if (has_soa_reorder)` dispatch, add:

```cpp
            if (has_soa_reorder) {
                // Tensor split: cooperative GPU+CPU MUL_MAT
                static const int tensor_split_cpu_pct = []() {
                    const char * env = std::getenv("GGML_SYCL_TENSOR_SPLIT");
                    return env ? std::atoi(env) : 0;
                }();
                if (tensor_split_cpu_pct > 0 && !g_ggml_sycl_graph_recording) {
                    if (ggml_sycl_mul_mat_tensor_split(ctx, src0, src1, dst,
                                                        tensor_split_cpu_pct, effective_layout)) {
                        return;
                    }
                    // Fall through to normal GPU dispatch if split failed
                }

                // MMVQ with SOA reorder -- matches master's fast path
                GGML_SYCL_DEBUG("[TG-FAST] batch=1 MMVQ+SOA for %s (type=%d layout=%d)\n",
```

**IMPORTANT**: The gate goes INSIDE the existing `if (has_soa_reorder)` block,
before the MMVQ dispatch. You are NOT replacing the MMVQ dispatch — you are
adding a check before it that optionally redirects to tensor split.

The modified block should look like:
```cpp
            if (has_soa_reorder) {
                // NEW: Tensor split gate
                static const int tensor_split_cpu_pct = ...;
                if (tensor_split_cpu_pct > 0 && !g_ggml_sycl_graph_recording) {
                    if (ggml_sycl_mul_mat_tensor_split(...)) {
                        return;
                    }
                }
                // EXISTING: MMVQ with SOA reorder
                GGML_SYCL_DEBUG("[TG-FAST] batch=1 MMVQ+SOA ...");
                ggml_sycl_op_mul_mat<...>(...);
                return;
            }
```

4. **Include the cpu-dispatch header** (if not already included)

At the top of ggml-sycl.cpp, verify `#include "cpu-dispatch.hpp"` is present.
If not, add it near the other local includes.

5. **Forward-declare the CPU function** (since Task 1 may not be merged yet)

Near the top of ggml-sycl.cpp (after includes), add:
```cpp
// Forward declaration — implemented in cpu-dispatch.cpp (Task 1)
extern void ggml_sycl_cpu_vec_dot_rows(ggml_type type, int ne00,
                                        const void * src0_host, const float * src1_host,
                                        float * output, int n_rows);
extern const void * ggml_sycl_cpu_dispatch_get_host_ptr(const char * name);
```

6. **Build and test**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# GPU-only baseline (tensor split disabled)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10" (normal, unaffected)

# Tensor split with CPU placeholder (output will be WRONG — expected!)
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0
# Expected: GARBAGE output (CPU portion is zeros — this is correct for Task 2)
# This verifies the path executes without crashing.
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add tensor split GPU dispatch with CPU placeholder"
```

**Notes for implementer:**
- `quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>` is a template function
  already used in the TG fast-path. Search for it around line 19867 to see the
  existing usage pattern.
- `MATRIX_ROW_PADDING` and `QK8_1` are defined in the SYCL headers.
- `ggml_sycl_get_layout_ptr_for()` is in `common.hpp` — it returns the SOA device pointer.
- The `ggml_sycl_op_mul_mat_vec_q()` function is in `mmvq.hpp`/`mmvq.cpp`.
- `g_ggml_sycl_graph_recording` is a thread_local bool at line 142.

---

### Task 3: Wire CPU vec_dot + End-to-End Correctness

**Track:** — (convergence)
**Depends on:** Task 1, Task 2
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~5 lines: replace placeholder with real call)

**Description:**

Replace the CPU placeholder (memset zeros) in `ggml_sycl_mul_mat_tensor_split()` with
the actual `ggml_sycl_cpu_vec_dot_rows()` call. Remove the forward declarations (now
resolved via header). Verify end-to-end correctness.

**Acceptance Criteria:**

- [ ] `GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1` produces correct output
- [ ] Output matches GPU-only reference: `"6, 7, 8, 9, 10"` for counting prompt
- [ ] GPU-only baseline still unaffected
- [ ] Works for multiple CPU percentages (10%, 13%, 20%, 25%)

**Implementation Guide:**

1. **Remove forward declarations** from ggml-sycl.cpp (the header from Task 1 provides them)

Delete these lines (added in Task 2):
```cpp
extern void ggml_sycl_cpu_vec_dot_rows(...);
extern const void * ggml_sycl_cpu_dispatch_get_host_ptr(...);
```

2. **Replace CPU placeholder with real vec_dot call**

In `ggml_sycl_mul_mat_tensor_split()`, find the placeholder:
```cpp
    // --- CPU portion: placeholder (zeros) ---
    memset(g_split_staging.output, 0, out_bytes);
```

Replace with:
```cpp
    // --- CPU portion: vec_dot on host AOS data ---
    const size_t src0_row_bytes = ggml_row_size(src0->type, ne00);
    const void * cpu_src0 = (const char *) host_ptr + N_gpu * src0_row_bytes;
    ggml_sycl_cpu_vec_dot_rows(src0->type, static_cast<int>(ne00),
                                cpu_src0,
                                g_split_staging.src1_host,
                                g_split_staging.output,
                                static_cast<int>(N_cpu));
```

3. **Test correctness**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# GPU-only baseline
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10"

# Tensor split correctness (MUST match GPU-only output)
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10" (MUST MATCH)

# Test multiple split ratios
for pct in 10 15 20 25; do
  echo "=== TENSOR_SPLIT=$pct ==="
  GGML_SYCL_TENSOR_SPLIT=$pct GGML_SYCL_DISABLE_GRAPH=1 \
    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
    -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
done
# Expected: ALL produce "6, 7, 8, 9, 10"
```

4. **Run GPU-only regression check**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expected: PP512 >= 1200, TG128 >= 68 (no regression)
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: wire CPU vec_dot into tensor split — Phase 1 complete"
```

---

### Task 4: Phase 2 — Graph Replay Integration

**Track:** — (sequential, after Task 3)
**Depends on:** Task 3
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~50 lines: CPU work queue + replay dispatch)

**Description:**

Phase 2 integrates tensor split with SYCL graph replay for full performance. During
graph recording, partial MUL_MATs are captured in the graph while CPU work descriptors
are saved. During graph replay, the GPU graph executes as one async submission while
CPU runs all vec_dot work concurrently. This is the performance-critical path.

**Architecture:**

```
RECORDING:
  compute_impl() records partial MUL_MATs → command graph
  tensor_split() saves CPU work descriptors → g_split_cpu_queue vector

REPLAY:
  ext_oneapi_graph() → GPU runs partial MUL_MATs (async)
  for each cpu_work: vec_dot_rows() → CPU runs concurrently
  stream->wait() → sync
  memcpy CPU outputs to device
```

**Acceptance Criteria:**

- [ ] `GGML_SYCL_TENSOR_SPLIT=13` (graph replay ON) produces correct output
- [ ] TG128 performance exceeds GPU-only graph replay baseline (~70 tok/s)
- [ ] GPU-only baseline (TENSOR_SPLIT=0) still unaffected
- [ ] Graph replay without tensor split still works
- [ ] Works for multiple CPU percentages

**Implementation Guide:**

1. **Add CPU work queue struct** (near g_split_staging)

```cpp
struct split_cpu_work {
    const void * host_src0;    // AOS host ptr, pre-offset to first CPU row
    ggml_type    type;         // weight quant type
    int          ne00;         // K columns
    int          n_rows;       // N_cpu rows
    size_t       dst_byte_off; // byte offset in dst for GPU→CPU boundary
};
static thread_local std::vector<split_cpu_work> g_split_cpu_queue;
```

2. **Modify tensor_split to support recording mode**

In `ggml_sycl_mul_mat_tensor_split()`, change the behavior based on
`g_ggml_sycl_graph_recording`:

- **Recording**: Record GPU partial MMVQ (captured by graph), push CPU work to queue, skip CPU execution
- **Non-recording**: Execute both GPU and CPU as before (Phase 1 behavior)

Replace the existing CPU portion + wait + copy block with:

```cpp
    if (g_ggml_sycl_graph_recording) {
        // Recording mode: GPU partial MMVQ was just recorded into the graph.
        // Save CPU work descriptor for replay-time execution.
        const size_t src0_row_bytes = ggml_row_size(src0->type, ne00);
        g_split_cpu_queue.push_back({
            (const char *) host_ptr + N_gpu * src0_row_bytes,
            src0->type,
            static_cast<int>(ne00),
            static_cast<int>(N_cpu),
            static_cast<size_t>(N_gpu) * sizeof(float)
        });
        return true;
    }

    // Non-recording: execute CPU portion immediately (Phase 1 path)
    const size_t src0_row_bytes = ggml_row_size(src0->type, ne00);
    const void * cpu_src0 = (const char *) host_ptr + N_gpu * src0_row_bytes;
    ggml_sycl_cpu_vec_dot_rows(src0->type, static_cast<int>(ne00),
                                cpu_src0, g_split_staging.src1_host,
                                g_split_staging.output, static_cast<int>(N_cpu));

    stream->wait();
    stream->memcpy(dst_dd + N_gpu, g_split_staging.output, out_bytes).wait();
    return true;
```

3. **Remove the `!g_ggml_sycl_graph_recording` gate from Step 1 of the tensor split gate**

In `ggml_sycl_mul_mat()`, change:
```cpp
if (tensor_split_cpu_pct > 0 && !g_ggml_sycl_graph_recording) {
```
to:
```cpp
if (tensor_split_cpu_pct > 0) {
```

This allows tensor split to run during recording (GPU portion gets captured, CPU
portion queued for later).

4. **Clear CPU work queue at start of compute_impl**

In `ggml_backend_sycl_graph_compute_impl()` (around line 25024), at the very beginning,
add:
```cpp
    g_split_cpu_queue.clear();
```

5. **Execute CPU work after graph replay**

In the graph replay path (around line 31862-31870), after `ext_oneapi_graph()`, add:

```cpp
        } else if (sycl_ctx->exec_graph) {
            // Default: replay cached graph
            GGML_SYCL_DEBUG("[SYCL-GRAPH] execute existing graph...\n");
            graph_refresh_input_tensors(sycl_ctx, cgraph);
            graph_executed = true;
            sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));

            // Tensor split: run CPU work concurrently with GPU graph replay
            if (!g_split_cpu_queue.empty()) {
                // Stage current src1 to host (find first MUL_MAT node's src1)
                for (int i = 0; i < cgraph->n_nodes && !g_split_staging.src1_host; i++) {
                    if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && cgraph->nodes[i]->src[1]) {
                        const size_t src1_bytes = ggml_nbytes(cgraph->nodes[i]->src[1]);
                        split_staging_ensure(src1_bytes, g_split_staging.out_size, sycl_ctx->stream());
                        sycl_ctx->stream()->memcpy(g_split_staging.src1_host,
                                                    cgraph->nodes[i]->src[1]->data, src1_bytes);
                        break;
                    }
                }

                // Execute all CPU vec_dot work items
                for (const auto & work : g_split_cpu_queue) {
                    // Ensure output buffer is large enough
                    const size_t out_bytes = work.n_rows * sizeof(float);
                    split_staging_ensure(0, out_bytes, sycl_ctx->stream());

                    ggml_sycl_cpu_vec_dot_rows(work.type, work.ne00,
                                                work.host_src0,
                                                g_split_staging.src1_host,
                                                g_split_staging.output,
                                                work.n_rows);
                }

                // Wait for GPU graph to finish
                sycl_ctx->stream()->wait();

                // Copy all CPU output portions to device
                for (const auto & work : g_split_cpu_queue) {
                    // Find the corresponding dst tensor
                    for (int i = 0; i < cgraph->n_nodes; i++) {
                        ggml_tensor * node = cgraph->nodes[i];
                        if (node->op == GGML_OP_MUL_MAT && node->data) {
                            float * dst_dd = (float *) node->data;
                            sycl_ctx->stream()->memcpy(
                                (char *) dst_dd + work.dst_byte_off,
                                g_split_staging.output,
                                work.n_rows * sizeof(float));
                        }
                    }
                }
                sycl_ctx->stream()->wait();
            }

            GGML_SYCL_DEBUG("[SYCL-GRAPH] execute done\n");
```

**IMPORTANT NOTE**: The CPU work queue's `dst_byte_off` and output buffer management
needs careful handling — each MUL_MAT has its OWN dst tensor. The work descriptors
should also store the dst pointer or a reference to it. The code above is a sketch;
the implementer should store `dst->data` in the work descriptor and use it directly.

Update the struct:
```cpp
struct split_cpu_work {
    const void * host_src0;
    void *       dst_device;    // device pointer to dst->data
    ggml_type    type;
    int          ne00;
    int          n_rows;
    size_t       dst_byte_off;  // byte offset into dst_device for CPU portion
};
```

And in the replay loop, simplify to:
```cpp
for (const auto & work : g_split_cpu_queue) {
    sycl_ctx->stream()->memcpy(
        (char *) work.dst_device + work.dst_byte_off,
        /* per-work output buffer */, work.n_rows * sizeof(float));
}
```

**IMPORTANT NOTE 2**: Each CPU work item needs its OWN output buffer, since they run
sequentially but the outputs go to different tensors. Either:
(a) allocate one big output buffer and partition it, or
(b) use a per-work output pointer into a large pinned buffer.
Option (a) is simpler: pre-allocate `max_N_cpu * n_mul_mats * sizeof(float)`.

6. **Test correctness**

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# Correctness with graph replay + tensor split
GGML_SYCL_TENSOR_SPLIT=13 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10" (MUST MATCH GPU-only)

# Performance sweep
for pct in 10 13 15 20 25; do
  echo "=== TENSOR_SPLIT=$pct ==="
  GGML_SYCL_TENSOR_SPLIT=$pct ONEAPI_DEVICE_SELECTOR=level_zero:0 \
    ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
  sleep 30
done
# Expected: best TG128 > 70 tok/s (exceeds graph-only baseline)

# GPU-only regression check
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expected: PP512 >= 1200, TG128 >= 68
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: Phase 2 tensor split with graph replay integration"
```

**Notes for implementer:**
- The src1 staging during replay is tricky: each MUL_MAT has a different src1
  (activations change as data flows through layers). The work descriptor should
  also store a handle to the src1 data or stage it per-MUL_MAT. For TG batch=1,
  all MUL_MATs in a layer share the same hidden state as src1, but attention Q/K/V
  MUL_MATs have different src1 shapes. This may require per-work src1 staging.
- If graph replay + tensor split shows correct output but no speedup, the bottleneck
  is likely per-MUL_MAT src1 staging. In that case, consider batching all CPU work
  with a shared src1 and only splitting layers where src1 matches.
- The graph replay submits ALL recorded ops as ONE async call. CPU vec_dot runs on
  the calling thread during GPU execution. GPU finishes all partial MUL_MATs in
  graph order; CPU processes them sequentially. Total time = max(GPU graph, CPU total).
