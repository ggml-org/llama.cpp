# CPU Dispatch Performance: Analysis & Options

**Date**: 2026-02-11
**Branch**: feature/sycl-coalescing
**Problem**: CPU dispatch path (`GGML_SYCL_CPU_OFFLOAD=1`) gives 0.96-0.99 tok/s vs fit_params' 6.79-8.80 tok/s

---

## Root Cause Analysis

### Current Performance Comparison (Mistral 7B Q4_0, Arc B580)

| Path | 40% VRAM TG (tok/s) | 30% VRAM TG (tok/s) | Mechanism |
|------|---------------------|---------------------|-----------|
| **fit_params** | **8.80** | **6.79** | Reduces ngl → ggml-cpu handles entire CPU layers |
| **Layer streaming** | 2.99 | 3.01 | GPU fetches weights from host per-layer |
| **CPU dispatch** | 0.99 | 0.96 | CPU computes host-resident MUL_MATs only |

### Why fit_params Is Fast

fit_params + ggml_backend_sched:
1. Decides `n_gpu_layers` at load time based on memory budget
2. Allocates CPU layers' tensors (weights AND activations) in HOST memory
3. Backend scheduler creates "splits" — contiguous groups of ops on one backend
4. Within a CPU split: ALL ops (MUL_MAT, RMS_NORM, ADD, ROPE, attention, etc.) execute on CPU with HOST memory buffers
5. **ONE copy at the boundary** between last CPU layer and first GPU layer
6. Activations never touch device memory for CPU layers

### Why Our CPU Dispatch Is 9x Slower

Debug output with `GGML_SYCL_DEBUG=1` reveals the per-layer dispatch pattern:

```
GPU: attn_norm (RMS_NORM)       ← stays on GPU
CPU: Qcur MUL_MAT               ← GPU→CPU transition #1
GPU: RESHAPE(Qcur)              ← CPU→GPU transition #2 (RESHAPE is a NO-OP!)
CPU: Vcur MUL_MAT, Kcur MUL_MAT ← GPU→CPU transition #3
GPU: RESHAPE(K), RESHAPE(V), ROPE(Q), ROPE(K), PERMUTE(Q), PERMUTE(K),
     MUL_MAT(Q*K^T), SOFT_MAX, MUL_MAT(attn*V), PERMUTE, CONT
                                 ← CPU→GPU transition #4
CPU: attn_out MUL_MAT            ← GPU→CPU transition #5
GPU: ADD(residual), RMS_NORM     ← CPU→GPU transition #6
CPU: ffn_gate MUL_MAT, ffn_up MUL_MAT  ← GPU→CPU transition #7
GPU: SWIGLU                      ← CPU→GPU transition #8
CPU: ffn_out MUL_MAT             ← GPU→CPU transition #9
GPU: ADD(residual)               ← CPU→GPU transition #10
```

**Result: 10 transitions per layer × 32 CPU layers = 320 GPU↔CPU transitions per token**

### Three Compounding Problems

1. **Only MUL_MAT dispatches to CPU** — RMS_NORM, ADD, SWIGLU, ROPE, SOFT_MAX all stay on GPU even for CPU-classified layers, creating constant ping-pong. Debug confirms: 451 CPU ops total, ALL are MUL_MAT, ZERO are anything else.

2. **View ops (RESHAPE/PERMUTE) are no-ops that needlessly trigger transitions** — They don't move data (just `break` in compute_forward) but `should_dispatch_to_cpu()` returns false for them, so they break CPU block continuity and trigger CPU→GPU→CPU flip-flops.

3. **Activation buffers are in DEVICE memory** — The SYCL backend owns all compute buffers on the GPU. Even when all ops in a layer could theoretically run on CPU, each op must: D2H copy activations → compute on CPU → H2D copy result. The next CPU op reads the result back from device again.

### Contrast with fit_params Architecture

| Aspect | fit_params + backend_sched | Our CPU dispatch |
|--------|---------------------------|------------------|
| Who decides placement | fit_params (static, load-time) | Unified cache (dynamic, runtime) |
| Activation buffer location | HOST memory for CPU layers | DEVICE memory (always) |
| Ops dispatched to CPU | ALL ops in CPU layers | Only MUL_MAT |
| Transitions per layer | 0 (within layer) | 10 |
| Transitions per token | 1-2 (at layer boundary) | 320 |
| Staging overhead | Zero (host buffers) | ~640 D2H/H2D copies per token |

---

## Options

### Option A: "Whole Layer" CPU Dispatch (Recommended First Step)

**Goal**: Reduce transitions from 320/token to ~28/token by dispatching ALL ops in CPU-classified layers to CPU.

**Changes**:

1. **Broaden `should_dispatch_to_cpu()` for non-MUL_MAT ops** (ggml-sycl.cpp:23448-23465):
   Currently, non-MUL_MAT ops only dispatch to CPU if the layer was "already classified" AND the op has a `blk.N.` name. Change to: if the layer is classified as CPU, ALL ops with that layer ID go to CPU — including attention MUL_MATs where src[0] is not a weight tensor.

2. **Make view ops transparent to transition detection** (ggml-sycl.cpp:25174):
   RESHAPE/PERMUTE/VIEW/TRANSPOSE are no-ops. Don't count them as "GPU ops" in the `node_on_cpu != prev_on_cpu` check. Simply inherit the previous op's classification.

3. **Handle attention MUL_MATs on CPU** (cpu-dispatch.cpp):
   Attention MUL_MATs (Q*K^T and attn*V) don't use weight tensors — src[0] is an intermediate activation. For batch=1 TG, these are tiny operations (1×N vectors). `cpu_mul_mat` already handles any MUL_MAT, just needs the dispatch routing to send them there.

4. **Re-enable retention within whole-layer CPU blocks** (cpu-dispatch.cpp):
   With entire layers on CPU, CPU blocks become 20+ consecutive ops instead of 1-2. Retention eliminates all intermediate staging. Only boundary D2H/H2D at layer entry/exit.

**Expected result**:
- Transitions: 320/token → ~28/token (2 per CPU layer: one at entry, one at exit)
- With retention: staging drops from ~640 copies to ~28 boundary copies
- Estimated TG: 5-8 tok/s (approaching fit_params' 6.79-8.80)

**Risk**: Low-medium. All CPU kernels exist. Main new work is routing attention MUL_MATs to CPU and fixing the view op transition detection.

**Implementation effort**: ~100-150 lines changed across ggml-sycl.cpp and cpu-dispatch.cpp.

---

### Option B: Host Compute Buffers for CPU Layers

**Goal**: Eliminate ALL staging by putting activation buffers in HOST memory for CPU-classified layers.

**Changes**:

1. **Classify layers before graph allocation**: During model load or first inference, determine which layers are CPU-classified (unified cache query).

2. **Allocate host-side compute buffers**: In `ggml_backend_sycl_buffer_init_tensor`, if the tensor belongs to a CPU-classified layer's compute graph, allocate from host-pinned memory pool instead of device USM.

3. **Modify graph refresh**: The graph replay path (`graph_refresh_input_tensors`) needs to handle mixed device/host tensor addresses.

4. **Boundary copy management**: At the last CPU layer → first GPU layer boundary, insert an explicit host→device copy. This mirrors what ggml_backend_sched does with splits.

**Expected result**:
- Zero staging overhead for CPU layers (activations never touch device)
- Matches fit_params performance exactly
- Estimated TG: 7-9 tok/s

**Risk**: Medium-high.
- Requires changes to buffer allocation path (currently assumes all SYCL buffers are device-side)
- Graph replay assumes all tensors have device pointers — host pointers would break pointer caching
- Interaction with unified cache eviction (if a weight gets evicted mid-inference, which buffer pool does the activation use?)
- May need to disable graph replay for mixed-buffer graphs

**Implementation effort**: ~300-500 lines. Touches buffer allocation, graph replay, tensor initialization.

---

### Option C: Surrender Gracefully (Remove CPU Dispatch)

**Goal**: Accept that fit_params + ggml_backend_sched already solves the problem better.

**Changes**:

1. Remove `GGML_SYCL_CPU_OFFLOAD` code path (~1700 lines in cpu-dispatch.cpp/hpp)
2. Remove orchestration code in ggml-sycl.cpp (~100 lines)
3. Ensure fit_params works correctly with unified cache budget (already working: 8.80 tok/s at 40%)

**Expected result**:
- Simpler codebase (-1800 lines)
- Performance unchanged (fit_params already at 8.80 tok/s)
- No maintenance burden for CPU dispatch path

**Risk**: Low.

**Downside**: Loses the ability for the SYCL backend to handle memory pressure dynamically. With fit_params, layer placement is decided at load time. If the unified cache needs to evict a GPU layer mid-inference (e.g., MoE expert rotation), there's no CPU fallback within the SYCL backend.

**Implementation effort**: Deletion only.

---

### Option D: Hybrid — fit_params for Static, CPU Dispatch for Dynamic

**Goal**: Best of both worlds. Let fit_params handle the initial layer split (fast path), keep CPU dispatch as a fallback for dynamically-evicted layers.

**Changes**:

1. **Default path**: fit_params sets `n_gpu_layers` based on unified cache budget → ggml_backend_sched splits graph → CPU layers use ggml-cpu backend with host buffers (fast)

2. **Fallback path**: If a GPU-assigned layer's weights get evicted to host during inference (MoE expert cycling, memory pressure from context growth), the SYCL backend's CPU dispatch handles it transparently.

3. **CPU dispatch improvements from Option A** applied only to the fallback path (fewer layers, less critical for latency since it's a rare event).

**Expected result**:
- Normal operation: fit_params performance (8.80 tok/s)
- Under dynamic pressure: graceful degradation with CPU dispatch fallback
- No inference failure from OOM — weights evicted → CPU compute fallback

**Risk**: Low (just layering two existing systems).

**Implementation effort**: ~50 lines to wire fit_params default + keep existing CPU dispatch as fallback.

---

## Recommendation

**Short-term**: Option D (Hybrid). fit_params already works at 8.80 tok/s. Keep CPU dispatch as-is for dynamic fallback. Minimal work, maximum benefit.

**Medium-term**: Option A (Whole Layer dispatch). If we want the SYCL backend to be self-sufficient (not dependent on fit_params), this gets us to 5-8 tok/s with moderate effort. Most valuable for MoE models where expert weights are dynamically loaded/evicted and fit_params can't predict placement.

**Long-term**: Option B (Host compute buffers). Full parity with fit_params, but significant architectural change. Only worth it if Option A proves insufficient and we need the SYCL backend to fully own the CPU/GPU split.

**Avoid**: Option C (removal) unless we're certain we'll never need dynamic weight migration.

---

## Key Metrics to Track

| Metric | Current | Option A Target | Option B Target | fit_params Ref |
|--------|---------|-----------------|-----------------|----------------|
| Transitions/token | 320 | 28 | 0 | 1-2 |
| CPU ops/layer | 7 (MUL_MAT only) | 30+ (all ops) | 30+ (all ops) | 30+ (all ops) |
| Staging copies/token | ~640 | ~28 | 0 | 0 |
| TG 40% VRAM (tok/s) | 0.99 | 5-8 | 7-9 | 8.80 |
| TG 30% VRAM (tok/s) | 0.96 | 4-7 | 6-8 | 6.79 |
| GPU-only regression | None | None | None | None |
