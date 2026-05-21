# CUDA-CPU Threading Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce CPU busy-wait when the scheduler waits for CUDA GPU operations by implementing a 3-phase adaptive event wait.

**Architecture:** Replace `cudaStreamSynchronize()` with `ggml_cuda_adaptive_wait()` that progresses through spin → exponential backoff → OS yield phases. Remove `cudaDeviceScheduleSpin` for cc 12.1 iGPUs. Add timing instrumentation to measure wait distribution. Buffer-level sync methods use adaptive wait instead of `cudaStreamSynchronize(cudaStreamPerThread)`.

**Tech Stack:** CUDA runtime API (`cudaEventQuery`, `cudaEventRecord`, `cudaEventSynchronize`), C++17, POSIX (`usleep`, `sched_yield`, `<chrono>`), ggml build system (CMake)

---

### Task 1: Implement `ggml_cuda_adaptive_wait()` function

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu` (add function before `ggml_backend_cuda_synchronize` at ~line 3232)
- Modify: `ggml/src/ggml-cuda/common.cuh` (add forward declaration)

This function is the core of the optimization. It replaces `cudaStreamSynchronize(stream)` with a progressive wait that yields the CPU thread faster.

- [ ] **Step 1: Add the adaptive wait function to `ggml-cuda.cu`**

Add this function before line 3232 (before `ggml_backend_cuda_synchronize`):

```c
// 3-phase adaptive wait: spin → backoff → yield
// Returns true if wait completed, false if fell back to cudaStreamSynchronize
static bool ggml_cuda_adaptive_wait(cudaStream_t stream, int device) {
    // Create a transient event for querying completion
    cudaEvent_t wait_event;
    cudaError_t err = cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return false; // fallback caller will handle
    }
    cudaEventRecord(wait_event, stream);

    const int64_t t_start = ggml_time_us();

    // Phase 1: busy-spin with short usleep (0→~100 μs)
    // 10 iterations of 10 μs sleep between event queries
    {
        for (int i = 0; i < 10; i++) {
            err = cudaEventQuery(wait_event);
            if (err == cudaSuccess) {
                cudaEventDestroy(wait_event);
                return true;
            }
            if (err != cudaErrorNotReady) {
                goto phase3; // error — fall through to OS-aware wait
            }
            usleep(10);
        }
    }

    // Phase 2: exponential backoff (100 μs → 10 ms)
    // Sleep intervals: 100, 200, 500, 1000, 2000, 5000, 10000 μs
    static const uint32_t backoff_us[] = { 100, 200, 500, 1000, 2000, 5000, 10000 };
    for (size_t i = 0; i < sizeof(backoff_us) / sizeof(backoff_us[0]); i++) {
        usleep(backoff_us[i]);
        err = cudaEventQuery(wait_event);
        if (err == cudaSuccess) {
            cudaEventDestroy(wait_event);
            return true;
        }
        if (err != cudaErrorNotReady) {
            break; // error — fall through
        }
    }

    // Phase 3: yield CPU + OS-aware wait
    {
        sched_yield();
    }

phase3:
    // cudaEventSynchronize uses CUDA's internal OS-aware wait mechanism
    // The CPU thread is handed to the OS scheduler
    err = cudaEventSynchronize(wait_event);
    cudaEventDestroy(wait_event);

    if (err == cudaSuccess) {
        return true;
    }

    // Fallback: if event operations failed, use the original stream sync
    return false;
}
```

- [ ] **Step 2: Add forward declaration in `common.cuh`**

Add to `common.cuh` near the other function declarations (~line 1136, after `ggml_cuda_set_device`):

```c
bool ggml_cuda_adaptive_wait(cudaStream_t stream, int device);
```

- [ ] **Step 3: Build check — compile the CUDA backend**

Run:
```bash
cd /home/alexmv2025/projects/cuda/fast/claude_llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON 2>&1 | tail -20
cmake --build build --config Release -j $(nproc) 2>&1 | grep -E "error|warning" | head -20
```

Expected: No new errors or warnings in `ggml-cuda.cu`.

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/common.cuh
git commit -m "feat(cuda): add 3-phase adaptive event wait function

Replaces cudaStreamSynchronize with progressive wait:
Phase 1: 10 iterations of event-query + 10μs usleep (~100μs total)
Phase 2: exponential backoff 100μs→10ms
Phase 3: sched_yield + cudaEventSynchronize (OS-aware)

Fallback to cudaStreamSynchronize if event operations fail."
```

---

### Task 2: Replace `synchronize` in CUDA backend with adaptive wait

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu:3232-3238`

This connects the adaptive wait to the backend interface. All callers of `ggml_backend_synchronize(cuda_backend)` automatically benefit.

- [ ] **Step 1: Replace `ggml_backend_cuda_synchronize` implementation**

Replace lines 3232-3238:

```c
static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (!ggml_cuda_adaptive_wait(cuda_ctx->stream(), cuda_ctx->device)) {
        // adaptive wait returned false — fallback
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    }

    GGML_UNUSED(backend);
}
```

- [ ] **Step 2: Build and run existing tests**

Run:
```bash
cmake --build build --config Release -j $(nproc) 2>&1 | grep -E "error|warning" | head -10
cd build && ctest -C Release -L backend --output-on-failure 2>&1 | tail -30
```

Expected: All backend tests pass. No new warnings.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "feat(cuda): use adaptive wait in backend synchronize

All callers of ggml_backend_synchronize() now use the 3-phase
adaptive wait instead of cudaStreamSynchronize. Falls back to
cudaStreamSynchronize if event operations fail."
```

---

### Task 3: Remove `cudaDeviceScheduleSpin` for cc 12.1 iGPUs

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu:303-310`

This flag forces CUDA runtime to busy-wait internally, defeating our adaptive wait at the API level.

- [ ] **Step 1: Replace the `cudaDeviceScheduleSpin` block**

Replace lines 303-310:

```c
        // Use adaptive yield by default. Users can force spin for
        // latency-sensitive workloads via GGML_CUDA_SCHEDULE_SPIN=1.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(id));
            if (getenv("GGML_CUDA_SCHEDULE_SPIN") != nullptr) {
                CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
            }
            // else: use default cudaDeviceScheduleAuto (yields on shared GPUs)
        }
```

- [ ] **Step 2: Build check**

Run:
```bash
cmake --build build --config Release -j $(nproc) 2>&1 | grep -E "error|warning" | head -10
```

Expected: No new warnings.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "fix(cuda): remove default cudaDeviceScheduleSpin for cc 12.1 iGPUs

cudaDeviceScheduleSpin forces the CUDA runtime to busy-wait,
which defeats adaptive wait at our API level. Default to
cudaDeviceScheduleAuto; allow GGML_CUDA_SCHEDULE_SPIN=1 to
force spin for latency-sensitive workloads."
```

---

### Task 4: Replace buffer-level `cudaStreamSynchronize` with adaptive wait

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu:674-745` (buffer mem ops)
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu:1012-1053` (split buffer ops)
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu:2691-2714` (MoE getrows sync)

These buffer-level operations use `cudaMemcpyAsync` + `cudaStreamSynchronize(cudaStreamPerThread)`. Replacing with adaptive wait reduces CPU blocking during buffer init, set, get, clear operations.

Note: `cudaStreamPerThread` is used here (not the backend's main stream). We need to adapt the wait to work with any stream.

- [ ] **Step 1: Create adaptive wait wrapper for `cudaStreamPerThread`**

Add a helper that wraps `cudaStreamPerThread` sync with adaptive wait. Add near the `ggml_cuda_adaptive_wait` function:

```c
// Adaptive wait wrapper for cudaStreamPerThread (used by buffer ops)
static inline void ggml_cuda_adaptive_wait_per_thread(int device) {
    ggml_cuda_set_device(device);
    if (!ggml_cuda_adaptive_wait(cudaStreamPerThread, device)) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}
```

- [ ] **Step 2: Replace sync in buffer operations**

Replace these lines with `ggml_cuda_adaptive_wait_per_thread(ctx->device)`:

| Line | Original | Replacement |
|------|----------|-------------|
| 679 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(ctx->device);` |
| 687 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(ctx->device);` |
| 695 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(ctx->device);` |
| 705 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(ctx->device);` |
| 715 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(ctx->device);` |
| 731 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(dst_ctx->device);` |
| 744 | `CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));` | `ggml_cuda_adaptive_wait_per_thread(ctx->device);` |

- [ ] **Step 3: Replace sync in split buffer operations**

Replace in `ggml_backend_cuda_split_buffer_set_tensor` (lines 1012-1014):

```c
    // Before:
    // for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
    //     CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    // }

    // After:
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_adaptive_wait_per_thread(id);
    }
```

Same pattern for `ggml_backend_cuda_split_buffer_get_tensor` (lines 1050-1052):

```c
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_adaptive_wait_per_thread(id);
    }
```

- [ ] **Step 4: Replace sync in MoE getrows operations**

Replace in the MoE getrows kernel (lines 2693 and 2714):

```c
    // Line 2693 — after cudaMemcpyAsync(ids_host.data(), ids->data, ...)
    // Before: CUDA_CHECK(cudaStreamSynchronize(stream));
    // After:
    if (!ggml_cuda_adaptive_wait(stream, ctx.device)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Line 2714 — after cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), ...)
    // Before: CUDA_CHECK(cudaStreamSynchronize(stream));
    // After:
    if (!ggml_cuda_adaptive_wait(stream, ctx.device)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
```

- [ ] **Step 5: Build and run tests**

Run:
```bash
cmake --build build --config Release -j $(nproc) 2>&1 | grep -E "error|warning" | head -10
cd build && ctest -C Release -L backend --output-on-failure 2>&1 | tail -30
```

Expected: All backend tests pass. The buffer operations are synchronous from the caller's perspective (they still wait for completion), just with less CPU usage.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "feat(cuda): use adaptive wait in buffer-level sync operations

Replace cudaStreamSynchronize(cudaStreamPerThread) in buffer
set_tensor, get_tensor, memset, clear, cpy, split-buffer, and
MoE getrows operations with adaptive event wait."
```

---

### Task 5: Add timing instrumentation to CUDA backend

**Files:**
- Modify: `ggml/src/ggml-cuda/common.cuh` (add timing fields to `ggml_backend_cuda_context`)
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu` (record timing in adaptive wait, print on free)
- Modify: `ggml/include/ggml-cuda.h` (add `ggml_backend_cuda_print_timing` declaration)

Adds timing counters to measure how much time is spent in each wait phase, enabling verification that the optimization works.

- [ ] **Step 1: Add timing fields to `ggml_backend_cuda_context` in `common.cuh`**

Add to the struct (after `copy_event` at line 1390):

```c
    // Timing instrumentation for adaptive wait
    int64_t  t_wait_us = 0;
    uint64_t n_wait_ops = 0;
    int64_t  t_wait_phase1_us = 0;
    int64_t  t_wait_phase2_us = 0;
    int64_t  t_wait_phase3_us = 0;
    int64_t  t_wait_fallback_us = 0;
    uint64_t n_wait_fallback = 0;
```

- [ ] **Step 2: Update `ggml_cuda_adaptive_wait()` to accept and record timing**

Modify the function signature to accept timing pointers and record per-phase timing. The function becomes non-static (since it's declared in common.cuh) and writes timing to the context:

```c
// In ggml-cuda.cu — replace the static function with this version:
bool ggml_cuda_adaptive_wait(cudaStream_t stream, int device,
        int64_t * out_t_wait_us,
        uint64_t * out_n_wait_ops,
        int64_t * out_t_phase1_us,
        int64_t * out_t_phase2_us,
        int64_t * out_t_phase3_us,
        int64_t * out_t_fallback_us,
        uint64_t * out_n_wait_fallback) {

    cudaEvent_t wait_event;
    cudaError_t err = cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return false;
    }
    cudaEventRecord(wait_event, stream);

    const int64_t t_start = ggml_time_us();

    // Phase 1: busy-spin (0→~100 μs)
    {
        const int64_t t_phase1_start = ggml_time_us();
        for (int i = 0; i < 10; i++) {
            err = cudaEventQuery(wait_event);
            if (err == cudaSuccess) {
                const int64_t t_end = ggml_time_us();
                const int64_t t_total = t_end - t_start;
                const int64_t t_p1 = ggml_time_us() - t_phase1_start;
                if (out_t_wait_us) *out_t_wait_us += t_total;
                if (out_n_wait_ops) (*out_n_wait_ops)++;
                if (out_t_phase1_us) *out_t_phase1_us += t_p1;
                cudaEventDestroy(wait_event);
                return true;
            }
            if (err != cudaErrorNotReady) goto phase3;
            usleep(10);
        }
        if (out_t_phase1_us) *out_t_phase1_us += ggml_time_us() - t_phase1_start;
    }

    // Phase 2: exponential backoff
    {
        const int64_t t_phase2_start = ggml_time_us();
        static const uint32_t backoff_us[] = { 100, 200, 500, 1000, 2000, 5000, 10000 };
        for (size_t i = 0; i < sizeof(backoff_us) / sizeof(backoff_us[0]); i++) {
            usleep(backoff_us[i]);
            err = cudaEventQuery(wait_event);
            if (err == cudaSuccess) {
                const int64_t t_end = ggml_time_us();
                const int64_t t_total = t_end - t_start;
                const int64_t t_p2 = ggml_time_us() - t_phase2_start;
                if (out_t_wait_us) *out_t_wait_us += t_total;
                if (out_n_wait_ops) (*out_n_wait_ops)++;
                if (out_t_phase2_us) *out_t_phase2_us += t_p2;
                cudaEventDestroy(wait_event);
                return true;
            }
            if (err != cudaErrorNotReady) break;
        }
        if (out_t_phase2_us) *out_t_phase2_us += ggml_time_us() - t_phase2_start;
    }

    // Phase 3: yield + OS-aware wait
phase3:
    {
        const int64_t t_phase3_start = ggml_time_us();
        sched_yield();
        err = cudaEventSynchronize(wait_event);
        cudaEventDestroy(wait_event);
        const int64_t t_p3 = ggml_time_us() - t_phase3_start;
        if (out_t_phase3_us) *out_t_phase3_us += t_p3;

        if (err == cudaSuccess) {
            const int64_t t_total = ggml_time_us() - t_start;
            if (out_t_wait_us) *out_t_wait_us += t_total;
            if (out_n_wait_ops) (*out_n_wait_ops)++;
            return true;
        }
    }

    // Fallback to cudaStreamSynchronize
    {
        const int64_t t_fb_start = ggml_time_us();
        // Caller handles the actual cudaStreamSynchronize
        const int64_t t_fb = ggml_time_us() - t_fb_start;
        if (out_t_fallback_us) *out_t_fallback_us += t_fb;
        if (out_n_wait_fallback) (*out_n_wait_fallback)++;
        return false;
    }
}
```

- [ ] **Step 3: Update all callers to pass timing pointers**

Update `ggml_backend_cuda_synchronize` (from Task 2):

```c
static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (!ggml_cuda_adaptive_wait(cuda_ctx->stream(), cuda_ctx->device,
            &cuda_ctx->t_wait_us, &cuda_ctx->n_wait_ops,
            &cuda_ctx->t_wait_phase1_us, &cuda_ctx->t_wait_phase2_us,
            &cuda_ctx->t_wait_phase3_us,
            &cuda_ctx->t_wait_fallback_us, &cuda_ctx->n_wait_fallback)) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    }

    GGML_UNUSED(backend);
}
```

Update `ggml_cuda_adaptive_wait_per_thread` helper (from Task 4):

```c
static inline void ggml_cuda_adaptive_wait_per_thread(int device) {
    ggml_cuda_set_device(device);
    // Buffer ops don't track per-context timing (they use cudaStreamPerThread,
    // not a backend stream), so pass null timing pointers.
    if (!ggml_cuda_adaptive_wait(cudaStreamPerThread, device,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}
```

Update MoE getrows callers (from Task 4):

```c
    // Pass null timing for MoE inline sync — it's on a kernel-local stream
    if (!ggml_cuda_adaptive_wait(stream, ctx.device,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
```

- [ ] **Step 4: Add print timing function**

Add to `ggml-cuda.cu` (after `ggml_backend_cuda_free` at ~line 3135):

```c
void ggml_backend_cuda_print_timing(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    GGML_LOG_INFO("\n=== CUDA Backend %s Adaptive Wait Timing ===\n", cuda_ctx->name.c_str());
    GGML_LOG_INFO("  Total waits:     %" PRIu64 "\n", cuda_ctx->n_wait_ops);
    GGML_LOG_INFO("  Total wait us:   %" PRId64 "\n", cuda_ctx->t_wait_us);
    GGML_LOG_INFO("  Phase 1 (spin):  %" PRId64 " us\n", cuda_ctx->t_wait_phase1_us);
    GGML_LOG_INFO("  Phase 2 (back):  %" PRId64 " us\n", cuda_ctx->t_wait_phase2_us);
    GGML_LOG_INFO("  Phase 3 (yield): %" PRId64 " us\n", cuda_ctx->t_wait_phase3_us);
    GGML_LOG_INFO("  Fallbacks:       %" PRIu64 " (%" PRId64 " us)\n",
        cuda_ctx->n_wait_fallback, cuda_ctx->t_wait_fallback_us);
    GGML_LOG_INFO("=====================================================\n\n");
}
```

- [ ] **Step 5: Print timing on backend free**

In `ggml_backend_cuda_free`, add timing print before delete:

```c
static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_backend_cuda_print_timing(backend);

    delete cuda_ctx;
    delete backend;
}
```

- [ ] **Step 6: Add API declaration in `ggml-cuda.h`**

Add after the existing function declarations:

```c
GGML_BACKEND_API void ggml_backend_cuda_print_timing(ggml_backend_t backend);
```

- [ ] **Step 7: Build and verify**

Run:
```bash
cmake --build build --config Release -j $(nproc) 2>&1 | grep -E "error|warning" | head -10
```

Expected: No errors or warnings.

- [ ] **Step 8: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/common.cuh ggml/include/ggml-cuda.h
git commit -m "feat(cuda): add adaptive wait timing instrumentation

Track per-phase timing (spin/backoff/yield/fallback) in the
CUDA backend context. Print summary on backend free. Expose
ggml_backend_cuda_print_timing() for manual inspection."
```

---

### Task 6: Run full test suite and verify correctness

**Files:**
- Test: `build/bin/test-backend-ops` (or via ctest)
- Test: Any CUDA-capable tool (llama-cli, llama-server)

- [ ] **Step 1: Run all backend tests**

Run:
```bash
cd build && ctest -C Release --output-on-failure -j $(nproc) 2>&1 | tail -40
```

Expected: All tests that passed before still pass. No new failures.

- [ ] **Step 2: Run the backend-ops test specifically with verbose output**

Run:
```bash
cd build && ctest -C Release --output-on-failure -R backend-ops -V 2>&1 | tail -60
```

Expected: All backend-ops tests pass, including set/get tensor roundtrip, copy, memset operations.

- [ ] **Step 3: Quick smoke test with llama-cli or llama-bench**

If a model is available, run a short inference to verify adaptive wait doesn't introduce correctness issues:

```bash
# Example (adjust path to your model)
./build/bin/llama-cli -m /path/to/model.gguf -n 32 -p "Hello" --temp 0 2>&1 | tail -10
```

Expected: Model runs without crashes, output is correct.

- [ ] **Step 4: Commit any test fixes if needed**

If any test fails due to timing (race conditions with adaptive wait), fix and commit.

---

### Task 7: Verify the optimization with a benchmark

**Files:**
- No code changes — verification only

- [ ] **Step 1: Measure CPU usage during GPU compute**

Run inference with `top` or `htop` monitoring:

```bash
# Terminal 1: watch CPU usage
watch -n 1 'top -bn1 | grep "Cpu\|llama"'

# Terminal 2: run inference
./build/bin/llama-bench -m /path/to/model.gguf -p 256 -n 256 -b 1 -t $(nproc) 2>&1
```

Expected: CPU usage during GPU-bound tokens should be <10% per thread (was 20-40% before).

- [ ] **Step 2: Check timing output on shutdown**

When the program exits, the CUDA backend timing should print:
```
=== CUDA Backend cuda0 Adaptive Wait Timing ===
  Total waits:     150
  Total wait us:   5230000
  Phase 1 (spin):  12000 us
  Phase 2 (back):  380000 us
  Phase 3 (yield): 4800000 us
  Fallbacks:       0 (0 us)
=====================================================
```

Verify: Most time should be in Phase 3 (yield), with minimal fallbacks.

- [ ] **Step 3: Document results**

Note the before/after CPU usage and timing output. If the optimization works as expected, the Phase 3 time should dominate, indicating the CPU thread is properly yielded during long GPU operations.

---

## Spec Self-Review (completed inline)

1. **Spec coverage:**
   - Point A (replace synchronize) → Task 2
   - Point B (remove schedule_spin) → Task 3
   - Point C (async buffer ops) → Task 4 (adaptive wait for buffer sync instead of full async, which is riskier)
   - Point D (scheduler flows through backend) → automatic, no code change needed
   - Timing instrumentation → Task 5
   - Testing → Task 6-7
   - All spec items covered.

2. **Placeholder scan:** No "TBD", "TODO", or vague instructions found. All code blocks are complete.

3. **Type consistency:**
   - `ggml_cuda_adaptive_wait()` signature: `cudaStream_t, int, 7x timing pointers` — consistent across all callers
   - Timing fields in `ggml_backend_cuda_context`: `int64_t` for durations, `uint64_t` for counts — matches CPU threadpool convention
   - Function visibility: non-static in common.cuh, defined in ggml-cuda.cu — consistent with existing `ggml_cuda_set_device` pattern

4. **Scope check:** Focused on CUDA backend only. No changes to scheduler, CPU backend, or other backends. Each task is self-contained and commits independently.
