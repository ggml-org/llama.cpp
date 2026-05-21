# CUDA-CPU Threading Optimization Design

**Date:** 2026-05-21
**Branch:** refactor/CUDA-backend

## Problem

When the ggml scheduler coordinates work between CPU and CUDA backends, the main CPU thread blocks on `cudaStreamSynchronize()` during `ggml_backend_sched_compute_splits()`. Each split requires:

1. `ggml_backend_synchronize(backend)` — blocks CPU until GPU finishes previous work
2. Cross-backend tensor copy — blocks CPU during `cudaMemcpy` + implicit sync
3. `ggml_backend_graph_compute_async()` — launches kernels, returns immediately
4. Another sync for MoE weight offloading

For multi-device workloads (CPU + GPU offload), the CPU thread spins/yields while waiting for GPU, consuming 20-40% CPU per waiting thread with zero productive work.

Additionally:
- `cudaDeviceScheduleSpin` is set for cc 12.1 iGPUs, forcing CUDA runtime to busy-wait
- `set_tensor`/`get_tensor` use `cudaMemcpyAsync` + immediate `cudaStreamSynchronize`
- No backend-level timing exists to measure how much time is spent waiting

## Solution: 3-Phase Adaptive Event Wait

Replace `cudaStreamSynchronize()` with a new `ggml_cuda_adaptive_wait()` function that progresses through three phases of increasing CPU yield:

### Phase 1: Busy-spin (0→100 μs)
- Tight loop: `cudaEventQuery()` + 10 iterations of ~10μs `usleep()`
- Targets short GPU ops (norm, small matmul) that finish quickly
- CPU stays on-core, minimal latency

### Phase 2: Exponential backoff (100 μs → 10 ms)
- Sleep intervals: 100μs → 200μs → 500μs → 1ms → 2ms → 5ms → 10ms
- Between sleeps: `cudaEventQuery()` check
- Targets medium GPU ops (attention layers, FFN)

### Phase 3: Yield + event sync (10 ms → completion)
- Record a join event on the CUDA stream
- Call `sched_yield()` to hand CPU time slice to OS scheduler
- Block on `cudaEventSynchronize(join_event)` — CUDA's internal OS-aware wait
- CPU thread is fully yielded to the OS

### Fallback
If `cudaEventQuery` returns an error (not `cudaErrorNotReady`), fall back to `cudaStreamSynchronize()`. This ensures correctness even if CUDA context is damaged.

## Implementation Points

### Point A: Replace `synchronize` in CUDA backend

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu:3232-3238`

```c
// Before:
static int ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    return 0;
}

// After:
static int ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_cuda_adaptive_wait(cuda_ctx->stream(), cuda_ctx->device);
    return 0;
}
```

This affects ALL callers: scheduler, tools, examples. Every time CPU waits for GPU, it uses adaptive wait.

### Point B: Remove `cudaDeviceScheduleSpin`

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu:303-310`

Remove the `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` call for cc 12.1 GPUs. This flag forces CUDA runtime to busy-wait internally, which defeats the adaptive wait at our level.

Replace with:
- Default: `cudaDeviceScheduleAuto` (CUDA chooses based on dedicated/shared GPU)
- Env var override: `GGML_CUDA_SCHEDULE_SPIN=1` to force spin (for latency-sensitive workloads)

### Point C: Async `set_tensor`/`get_tensor`

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu:674-715`

Current: `cudaMemcpyAsync(dst, src, sz, kind) → cudaStreamSynchronize(cudaStreamPerThread)`

New:
1. Track per-buffer `cudaEvent_t copy_done_events`
2. `set_tensor`: `cudaMemcpyAsync` → record event → return immediately (no sync)
3. In `graph_compute`: before each CUDA kernel that reads from a CPU→GPU buffer, `cudaStreamWaitEvent(compute_stream, copy_done_event)` for any pending copies
4. `get_tensor`: must still sync because CPU reads the data, but use `ggml_cuda_adaptive_wait()` instead of `cudaStreamSynchronize()`

Safety: GPU stream waits on copy event before consuming data. CPU only reads after explicit `get_tensor` sync.

### Point D: Scheduler cross-backend sync

**File:** `ggml/src/ggml-backend.cpp` in `ggml_backend_sched_compute_splits()`

The scheduler calls `ggml_backend_synchronize(backend)` before cross-backend copies. With Point A, these calls automatically use adaptive wait. No scheduler code changes needed — the improvement flows through the backend interface.

## Timing Instrumentation

Add backend-level timing to `ggml_backend_cuda_context`:

```c
int64_t  t_wait_us;       // total time spent in adaptive_wait
uint64_t n_wait_ops;      // number of wait operations
int64_t  t_wait_phase1_us; // time spent in spin phase
int64_t  t_wait_phase2_us; // time spent in backoff phase
int64_t  t_wait_phase3_us; // time spent in yield phase
```

Expose via new API:
- `ggml_backend_cuda_print_timing(ggml_backend_t)` — prints per-backend timing
- Logged on backend free, alongside CPU threadpool timings

## Testing

1. **Unit test:** `ggml_cuda_adaptive_wait()` with immediately-ready event → should return in <50μs (Phase 1)
2. **Unit test:** delayed event (via `cudaLaunchHostFunc`) → should progress through all 3 phases
3. **Integration:** Existing `test-backend-ops.cpp` tests should pass unchanged
4. **Correctness:** Set/get tensor roundtrip — verify GPU doesn't read stale data when copies are async
5. **Benchmark:** Measure CPU% during GPU compute: expect <5% per thread vs 20-40% before

## Files to Change

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Add `ggml_cuda_adaptive_wait()`, replace sync calls, fix schedule_spin, async buffer ops |
| `ggml/src/ggml-cuda/common.cuh` | Add copy tracking events to `ggml_backend_cuda_context` |
| `ggml/include/ggml-cuda.h` | Add timing API declaration |
| `ggml/src/ggml-backend.cpp` | No changes — flows through backend interface |

## Out of Scope

- Vulkan/HIP/SYCL backends (same pattern can be applied later)
- Metal backend (uses different sync model — `MTLCommandBuffer.waitUntilCompleted`)
- Server-level threading (llama-server thread pool, HTTP workers)
- CPU threadpool barrier optimization (already has adaptive barrier since commit 79da4de98)
