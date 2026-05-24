# Async Pipelined Prefill for AMD EPYC 9V74 + RTX 5090 Hybrid Inference

> **Date:** 2026-05-24
> **Platform:** AMD EPYC 9V74 (80-core Zen 4, 12-channel DDR5, 768GB) + NVIDIA RTX 5090 (Blackwell SM 120, 32GB VRAM)
> **Depends on:** Phase 1 CPU optimizations (NUMA interleave, CCD affinity, AVX-512 VNNI, pinned RAM, TMA descriptor types) — already on `refactor/AMD-EPYC`

**Goal:** Eliminate the single-threaded prefill bottleneck by introducing a 3-stage async pipeline (CPU compute → TMA transfer → GPU compute) that saturates all 80 EPYC cores and the RTX 5090 simultaneously during large batch (4096+) prompt processing.

**Architecture:** A new `ggml_backend_sched_pipelined` scheduler wraps the existing split-graph creation and replaces the sequential `compute_splits` loop with a depth-3 pipeline. KV cache is double-buffered between pinned RAM and VRAM. CPU execution uses two CCD-paired threadpools (CCD0+1 and CCD6+7) for NUMA-local L3 locality.

**Tech Stack:** CUDA 13.x TMA (`cp.async.bulk`), pinned RAM (`mmap(MAP_LOCKED)`), AVX-512 VNNI, sysfs topology probing, CUDA event-based synchronization.

---

## Problem Statement

During prompt prefill (Phase 1: NUMA/CCD/VNNI already implemented), the system achieves ~41 t/s but hardware is severely underutilized:

- **CPU**: Only a fraction of 80 cores active during graph compute. Many ops (GET_ROWS, SCALE, ROPE, etc.) run with `n_tasks = 1`, leaving 79 threads idle at the barrier.
- **GPU**: ~50% utilization. The sequential split execution (`compute_splits` in `ggml-backend.cpp:1541-1681`) processes CPU and GPU splits one at a time — CPU computes, copies, then GPU computes. No overlap.
- **PCIe**: `cudaMemcpyAsync` on the compute stream. Single-thread copy overhead, no TMA despite infrastructure (`tma.cuh`) existing.

Root causes identified in codebase exploration:

| Root Cause | Location | Impact |
|------------|----------|--------|
| Sequential split execution | `ggml-backend.cpp:1541` | GPU waits for CPU, CPU waits for GPU |
| Barrier-after-every-node model | `ggml-cpu.c:3354` | Single-thread ops stall all 80 threads |
| KV cache on compute stream | `ggml-cuda.cu:762-790` | Copy blocks compute stream |
| TMA not in data path | `tma.cuh` unused | Zero CPU transfer not realized |
| Single threadpool | `llama-context.cpp:2221` | No CCD-parallel execution within pipeline |

---

## Design Decisions

### Pipeline Depth: 3 (CPU compute → TMA transfer → GPU compute)

Three concurrent stages. In steady state:

```
t=0:  CPU pool A: Split 0  | TMA: —              | GPU: —
t=1:  CPU pool A: Split 1  | TMA: Split 0 KV     | GPU: Split 0
t=2:  CPU pool A: Split 2  | TMA: Split 1 KV     | GPU: Split 1
t=3:  CPU pool B: Split 3  | TMA: Split 2 KV     | GPU: Split 2
t=4:  CPU pool B: Split 4  | TMA: Split 3 KV     | GPU: Split 3
t=5:  —                    | TMA: Split 4 KV     | GPU: Split 4
```

Rationale: Depth=2 would underutilize the TMA engine. Depth=4 adds memory overhead (~32-48GB extra activations) for diminishing returns at 768GB RAM. Depth=3 achieves ~85-90% utilization with ~16-32GB overhead.

### Split Granularity: Groups of 8 layers

For 80-layer model (40 CPU + 40 GPU): 5 CPU splits × 8 layers + 5 GPU splits × 8 layers = 10 total splits. Fine-grained (per-layer) would create 80 CUDA events and scheduling overhead. Coarse (all 40 at once) creates load imbalance — if CPU and GPU speeds diverge, pipeline bubbles. Groups of 8 allow the scheduler to rebalance.

### KV Cache: Double-buffered with TMA prefetch

KV cache stored in pinned RAM (primary, via `ggml_backend_cpu_pinned_buffer_type()`) and VRAM (prefetched copy). TMA transfers KV split N+1 to VRAM while GPU computes split N. The GPU reads from VRAM (fast, no PCIe latency per attention). Double buffering: splits 0,2,4 use Buffer A; splits 1,3 use Buffer B. This prevents write-after-read hazards between CPU writes and TMA reads.

### CPU Threadpool: Dual pool, one per CCD pair

Two threadpools of 20 cores each:
- **Pool A**: CCD 0 + CCD 1 (NUMA node 0) — 40 threads with SMT
- **Pool B**: CCD 6 + CCD 7 (NUMA node 1) — 40 threads with SMT

CCD0+1 and CCD6+7 are chosen for symmetry: one pair from each NUMA node, equal distance to both memory controllers. The pipeline rotates pools per split (split 0 → pool A, split 1 → pool B, split 2 → pool A...), eliminating false-sharing — each pool writes to its own KV buffer on its own NUMA node.

Intermediate CCDs (CCD 2-5) remain available for server I/O, sampling, chat templating.

---

## Component Specification

### Component 1: Pipeline Scheduler (`ggml_backend_sched_pipelined`)

**Files:** `ggml/src/ggml-backend-pipeline.h` (new), `ggml/src/ggml-backend-pipeline.cpp` (new)

**API:**

```c
// Creation — wraps an existing scheduler
ggml_backend_sched_pipelined_t ggml_backend_sched_pipelined_init(
    ggml_backend_sched_t base,       // existing scheduler (split creation, alloc)
    int pipeline_depth,              // 3
    int split_size_layers,           // 8 (layers per split)
    ggml_threadpool_t cpu_tp[2],     // dual threadpools
    ggml_backend_t gpu_backend);     // CUDA backend

// Execute — replaces ggml_backend_sched_graph_compute
void ggml_backend_sched_pipelined_compute(
    ggml_backend_sched_pipelined_t sched,
    ggml_cgraph * gf);

// Cleanup
void ggml_backend_sched_pipelined_free(ggml_backend_sched_pipelined_t sched);
```

**State structure:**

```c
struct ggml_backend_sched_pipelined {
    ggml_backend_sched_t base;
    int depth;                       // 3
    int split_count;                 // populated from base scheduler

    // TMA transfer infrastructure
    cudaStream_t tma_stream;
    cudaEvent_t stage_events[2];     // ping-pong ring

    // CPU threadpools
    ggml_threadpool_t cpu_tp[2];
    int active_pool;                 // rotates per split

    // KV double buffers (managed by kv-cache component)
    void * kv_buffer[2];
};
```

**Execution algorithm:**

1. Call `ggml_backend_sched_alloc_graph(base, gf)` to create and allocate splits as usual.
2. Iterate splits sequentially by submission order, but execute in a depth-N window:
   - Submit CPU split i for compute (synchronous on current pool).
   - If i >= 1 and split i-1 is CPU: launch TMA transfer of split i-1's KV to VRAM on `tma_stream`, record `stage_events[i % 2]`.
   - If i >= depth: make compute stream wait on `stage_events[(i - depth) % 2]`, then submit GPU split i-depth.
3. Rotate `active_pool` each iteration.
4. Final `ggml_backend_synchronize(gpu_backend)`.

The key insight: submission is sequential (one loop), but execution is overlapping via async streams. CPU is blocking (simplifies barrier semantics), TMA and GPU are async. The pipeline depth naturally emerges from the event dependencies.

### Component 2: TMA KV Transfer (`ggml_tma_transfer`)

**Files:** `ggml/src/ggml-cuda/tma-transfer.cu` (new), `ggml/src/ggml-cuda/tma-transfer.h` (new)

**Host API:**

```c
struct ggml_tma_transfer {
    void * desc;                     // device-side TMA descriptor (16 bytes)
    cudaStream_t stream;
    size_t num_elements;
    size_t stride;
};

void ggml_tma_init_transfer(ggml_tma_transfer_t * out,
    void * src_pinned,    // pinned RAM source
    void * dst_vram,      // VRAM destination
    size_t num_elements,  // element count (float16/bf16)
    size_t stride,        // row stride in elements
    cudaStream_t stream);

void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer);
void ggml_tma_free_transfer(ggml_tma_transfer_t transfer);
```

**Device kernel (SM 100+ only):**

```cuda
#if __CUDA_ARCH__ >= 1000
__global__ void ggml_tma_kv_transfer_kernel(
    ggml_cuda_tma_desc desc,
    int num_elements,
    int elem_per_thread);
#endif
```

One thread per chunk. Loads TMA descriptor into descriptor cache slot via `cp.async.bulk.shared::global` PTX, calls `ggml_cuda_tma_commit_group()` then `ggml_cuda_tma_wait_group(0)`.

**Fallback for SM < 100:** `cudaMemcpyAsync` on the dedicated TMA stream. The scheduler's event synchronization remains identical — only the transfer mechanism changes.

**Fallback for no pinned RAM:** If `mmap(MAP_LOCKED)` fails (ulimit), attempt `cudaHostAlloc()`. If that fails, use regular `cudaMemcpyAsync` with a warning log.

### Component 3: Dual CCD Threadpool

**Files:** `ggml/src/ggml-cpu/ggml-cpu.c` (extension of existing CCD code)

**API additions:**

```c
// Probe CCD pairs from sysfs topology
int ggml_cpu_probe_ccd_pairs(
    struct ggml_cpu_ccd_pair pairs[],
    int max_pairs);

// Initialize dual threadpools with CCD pair affinity
void ggml_cpu_init_dual_threadpool(
    ggml_threadpool_t tp_out[2],
    int threads_per_pair,
    int fill_smt);

// Cleanup
void ggml_cpu_free_dual_threadpool(ggml_threadpool_t tp[2]);
```

**Sysfs probing:** Reads `/sys/devices/system/cpu/cpu*/cache/index*/id` to group CPUs by L3 cache domain. Forms pairs from first and last CCDs (CCD0+1, CCD6+7) for NUMA symmetry. Falls back to single global pool if sysfs is unavailable (non-Linux).

**Fill physical cores first:** The existing CCD affinity code already has `fill_smt` logic. Each pair's pool first fills 20 physical cores, then 20 SMT siblings.

### Component 4: Double-Buffered KV Cache

**Files:** `src/llama-kv-cache.cpp` (extension), `src/llama-context.cpp` (wiring)

**Allocation changes:** When pipeline is enabled (`cparams.pipeline_depth > 0`):
1. KV cache allocated via `ggml_backend_cpu_pinned_buffer_type()` — pinned RAM.
2. Two VRAM copies allocated via `ggml_backend_cuda_buffer_type()`.
3. The KV cache cell write path remains unchanged — CPU writes to pinned RAM as before.
4. At split boundaries, the pipeline scheduler triggers TMA to the appropriate VRAM buffer.

**State save/restore compatibility:** `llama_state_get_size/copy/load` operates on the pinned RAM buffer directly — no change to the save/restore API. The VRAM buffers are transient (reconstructed on demand via TMA).

### Component 5: Server Integration

**Files:** `src/llama-context.cpp` (pipeline compute path in `graph_compute()`)

**Configuration via `llama_context_params`:**

```c
struct llama_context_params {
    // ... existing fields ...
    int32_t pipeline_depth;         // 0 = disabled (default)
    int32_t pipeline_split_size;    // layers per split (default 8)
};
```

**Integration in `llama_context::graph_compute()`:**

When `cparams.pipeline_depth > 0` and the batch is a prefill (`batched == true` and `n_tokens > pipeline_split_size`):
1. Construct `ggml_backend_sched_pipelined` from the base scheduler.
2. Call `ggml_backend_sched_pipelined_compute()` instead of `ggml_backend_sched_graph_compute_async()`.
3. Free the pipelined scheduler.

For decoding (batch=1), the pipeline is never used — the sequential path has lower overhead for single-token evaluation.

---

## Backward Compatibility

- The pipeline is strictly opt-in. `pipeline_depth = 0` (the default) uses the existing sequential `compute_splits`.
- All existing APIs (`ggml_backend_sched_graph_compute`, `llama_decode`, KV cache state save/restore) remain unchanged.
- The dual threadpool is only created when `pipeline_depth > 0`. The single threadpool path is untouched.
- TMA transfer falls back to `cudaMemcpyAsync` on SM < 100 or when pinned RAM is unavailable.
- Non-Linux platforms: CCD probing falls back to a single global pool. Pipeline still functions with reduced locality.

---

## Testing Strategy

1. **Unit test (`test-pipeline-scheduler`):** Mock GPU backend + simple CPU graph. Verify split execution order and event synchronization. Test all three fallback paths (no TMA, no pinned, no CCD).
2. **Micro benchmark (extend `pcie-bench`):** Add `--tma-kv-prefill` mode. Measures pipeline throughput with real KV-sized transfers on pinned RAM.
3. **Integration test (extend `llama-bench`):** Add `--pipeline` flag. Compare t/s with and without pipeline. Expected: ~2x prefill speedup on batch=4096, ~1.3x on batch=1024, neutral on batch=1.
4. **Regression test:** Run existing `test-backend-ops.cpp` suite — the base scheduler path must produce identical results.

---

## Expected Performance

| Metric | Before (Phase 1) | After (Pipeline) |
|--------|-----------------|------------------|
| Prefill t/s (batch 4096) | ~41 | ~80-100 |
| CPU utilization (prefill) | ~45% | ~90% |
| GPU utilization (prefill) | ~50% | ~85% |
| PCIe utilization | Burst-only | Sustained ~45 GB/s |
| Decode t/s (batch 1) | ~28 | ~28 (unchanged) |
| Memory overhead | — | +16-32 GB (double-buffered KV) |

---

## Out of Scope

- Decoding optimization (batch=1) — pipeline overhead exceeds benefit for single tokens.
- Work-stealing at the node level within a split — Phase 1's barrier model suffices for mul_mat-dominated work; node-level stealing is a separate architectural change.
- Dynamic split sizing — the 8-layer granularity is fixed for this implementation. Adaptive sizing is a follow-up.
- Multi-GPU — single RTX 5090 target. Multi-instance training (MIG) or NVLink is future work.
