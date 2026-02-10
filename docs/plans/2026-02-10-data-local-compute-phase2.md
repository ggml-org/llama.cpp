# Data-Local CPU Compute — Phase 2: Performance & Coverage

## Context

Phase 1 (epic llama.cpp-d7h4, 12 commits `7b28a92cd..f344f0bd9`) established the foundation:
- CPU device discovery via `sycl::cpu_selector_v` with `GGML_SYCL_CPU_OFFLOAD=1`
- Tier-based dispatch routing with per-layer classification
- CPU compute kernels: MUL_MAT (oneDNN), RMS_NORM, ADD, MUL (SYCL parallel_for)
- Activation staging (3 host-pinned slots for device↔host transfer)
- GPU fallthrough for unsupported ops, graph replay gate
- Boundary sync at GPU↔CPU layer transitions

**Phase 1 performance at 30% VRAM** (Mistral 7B Q4_0, Arc B580):
- PP512 = 247 tok/s, TG128 = 1.7 tok/s (CPU offload mode)
- PP512 = 251 tok/s, TG128 = 2.7 tok/s (auto-streaming mode)
- Default (100% VRAM): PP512 = 1249, TG128 = 70.9

Phase 2 eliminates the main performance bottlenecks and extends op coverage so that
CPU-offloaded layers run entirely on CPU without GPU fallthrough round-trips.

## Improvement Areas (from code analysis)

### 1. Host-Pinned Compute Buffers — Eliminate Staging Overhead

**Current state** (`cpu-dispatch.cpp:36-112`):
Every CPU op executes this pattern:
```
get_host_ptr()    → gpu_q->memcpy(host, dev_ptr, n).wait()   // stage in
cpu_q->submit()   → cpu_q->wait()                            // compute
flush_output()    → gpu_q->memcpy(dev_ptr, host, n).wait()   // stage out
```
Two synchronous `.wait()` calls per op (lines 96, 111) stall the pipeline.
With 5+ ops per transformer layer × 20+ CPU layers, this adds ~200+ sync points per token.

**Fix**: Allocate the ggml compute buffer for CPU-offloaded layers using
`sycl::malloc_host()` instead of `sycl::malloc_device()`. When activations are already
in host-pinned memory, `get_host_ptr()` returns the original pointer (line 86-88)
and `flush_output()` is a no-op (line 103-104). Zero staging, zero waits.

**Implementation**: Add a host-pinned buffer type to the SYCL backend that the scheduler
uses for compute buffers when all nodes in the allocation are CPU-dispatched. The existing
`ggml_backend_sycl_host_buffer_type()` (line 9427) provides the foundation — extend it to
handle compute buffer allocation (not just weight buffers).

**Files**: `ggml-sycl.cpp` (buffer type registration), `cpu-dispatch.cpp` (remove staging)

### 2. Additional CPU Kernels

**Current state** (`cpu-dispatch.cpp:442-458`):
Only 4 ops handled: MUL_MAT, RMS_NORM, ADD, MUL. A typical transformer layer has ~12 ops.
Unsupported ops fall through to GPU, requiring activation transfer back to device memory,
GPU execution, then transfer back — defeating the purpose of CPU dispatch.

**Ops needed** (marked as TODO at line 452 + analysis of transformer layer ops):

| Op | Used For | Complexity |
|----|----------|-----------|
| SILU (UNARY) | Activation function in SwiGLU FFN | Simple: `x * sigmoid(x)` element-wise |
| GLU | Fused gate×up in FFN (REGLU, SWIGLU, GEGLU) | Medium: element-wise with split |
| SOFT_MAX | Attention score normalization | Medium: row-wise max+sum+div |
| ROPE | Rotary positional embeddings | Medium: sin/cos rotation per dim pair |
| NORM | Layer norm (GPT-2 style models) | Simple: mean+variance normalization |
| CPY/CONT | Tensor reshape/copy operations | Simple: memcpy or no-op for contiguous |
| SCALE | Scale tensor by constant | Simple: element-wise multiply |
| SQR/SQRT | Squared/square root | Simple: element-wise |

Priority order: SILU/GLU → ROPE → SOFT_MAX → NORM → CPY/CONT/SCALE → SQR/SQRT

**Files**: `cpu-dispatch.cpp` (new kernel functions + dispatch entries)

### 3. Quantized Type Support for CPU MUL_MAT

**Current state** (`cpu-dispatch.cpp:153-157`):
```cpp
if (!src0_f32 && !src0_f16) {
    return false;  // Q4_0, Q4_K, Q8_0 etc. all fall back to GPU
}
```
This is the most impactful gap. Real-world models use quantized types (Q4_0, Q4_K_M, Q8_0).
With only F32/F16 support, CPU MUL_MAT never fires for typical models — the primary compute
op falls through to GPU streaming, making CPU dispatch mostly useless for quantized models.

**Options**:
1. **oneDNN WOQ (weight-only quantization)**: `DnnlGemmWrapper::woq_gemm_q4_0()` already
   exists in `gemm.hpp:335`. Test if it works with CPU queue (oneDNN is device-agnostic
   through SYCL). If so, this is the fastest path for Q4_0.
2. **Dequant→F32→GEMM**: For types without native WOQ support, dequantize weights to F32
   on CPU, then use standard oneDNN GEMM. The dequantization functions exist in ggml
   (`ggml_internal_get_type_traits_map`).
3. **ggml-cpu backend delegation**: For complex quantized types (Q4_K, Q6_K), consider
   delegating to the ggml-cpu backend's optimized AVX/AMX kernels. This avoids reimplementing
   quantized GEMM on SYCL CPU.

**Files**: `cpu-dispatch.cpp` (cpu_mul_mat expansion), possibly `gemm.hpp`

### 4. Async Activation Pipeline (Double-Buffer Staging)

**Current state**: Even with host-pinned compute buffers (Task 1), weight tensors that remain
in device VRAM may still need staging. Also, CPU→GPU boundary transitions require data
movement that could be overlapped.

**Fix**: Use SYCL events instead of `.wait()` for data transfers. Double-buffer the staging
slots so one op's output staging overlaps with the next op's input staging. The layer
streaming manager (`layer-streaming.hpp:34`) already implements this pattern — adapt it.

**Dependencies**: Task 1 (host-pinned buffers) should land first since it may eliminate
most staging. This task handles the remaining cases.

**Files**: `cpu-dispatch.cpp` (event-based staging), `ggml-sycl.cpp` (boundary sync)

### 5. Broadcast Support Expansion

**Current state** (`cpu-dispatch.cpp:369, 430`):
ADD and MUL return false for non-trivial broadcast patterns:
```cpp
return false;  // Unsupported broadcast for Phase 1
```
Some transformer architectures use multi-dimensional broadcasting for bias addition
and attention masking.

**Fix**: Implement general ND broadcast following ggml's stride-based iteration pattern
(see `ggml-cpu/ops.cpp` for reference implementations).

**Files**: `cpu-dispatch.cpp` (cpu_add, cpu_mul broadcast expansion)

### 6. DnnlGemmWrapper Mutex Separation

**Current state** (`common.hpp:2198-2267`):
```cpp
std::mutex dnnl_mutex;
```
All oneDNN calls (CPU and GPU) serialize through a single per-context mutex. When CPU
layers do MUL_MAT via oneDNN while GPU layers also use oneDNN (PP path), they contend.

**Fix**: Create separate `dnnl_cpu_mutex` for CPU-queue oneDNN calls. CPU dispatch uses
its own CPU SYCL queue (`ggml_sycl_get_cpu_queue()`), which creates a separate oneDNN
engine — the calls don't need to share a mutex. The unified-cache also has
`onednn_scratch_mutex_` (unified-cache.hpp:902) which may need similar treatment.

**Files**: `common.hpp` (add cpu mutex), `cpu-dispatch.cpp` (use cpu mutex)

### 7. Layer Classification Cost Model

**Current state** (`ggml-sycl.cpp:23263-23310`):
`should_dispatch_to_cpu()` uses binary classification: if any weight tensor in the layer
is host-resident (cache PINNED_HOST or MMAP tier), the entire layer runs on CPU.

**Fix**: Add a cost model that compares estimated CPU throughput vs GPU streaming throughput
for each layer. Factors: layer size, batch size (PP vs TG), available GPU bandwidth,
CPU core count. Small layers might be faster on GPU even with streaming; large layers
definitely benefit from CPU compute. This enables optimal split point selection.

**Files**: `ggml-sycl.cpp` (should_dispatch_to_cpu enhancement)

### 8. Boundary Sync Optimization

**Current state** (`ggml-sycl.cpp:25036-25050`):
```cpp
if (node_on_cpu != prev_on_cpu) {
    if (node_on_cpu) {
        // GPU→CPU: drain entire GPU queue
```
Full queue drain at every device transition. With interleaved CPU/GPU layers, this
creates many synchronization barriers.

**Fix**: Use per-tensor SYCL event dependencies instead of full queue drain. Track the
event from the last GPU op that produces each tensor, and wait only on that event
when the tensor is consumed by a CPU op. This allows independent GPU work to continue.

**Dependencies**: Task 4 (async pipeline) — both address synchronization overhead.

**Files**: `ggml-sycl.cpp` (boundary sync in graph_compute_impl)

### 9. fit_params Budget Guard

**Current state** (`llama-model.cpp:8540-8583, 8707-8744`):
SYCL tensor inventory runs BEFORE the `no_alloc` early return at line 8746, causing:
- 236 MB streaming buffer allocation per probe
- Accumulated `unified_cache_add_runtime_bytes()` across probe iterations
- Eventually OOM or deadlock during fit_params binary search

**Fix**: Guard both inventory blocks with `if (!ml.no_alloc)`. During measurement-only
probes, inventory serves no purpose since the model is freed immediately.

A plan already exists at `~/.claude/plans/fluffy-prancing-treehouse.md` with the exact
two-line fix. This is the simplest and highest-priority fix.

**Files**: `src/llama-model.cpp` (two `if (!ml.no_alloc)` guards)

## Task Breakdown

### Track A: Staging Elimination (Tasks 1, 4)
These remove the synchronous staging overhead — the biggest perf bottleneck.

### Track B: Op Coverage (Tasks 2, 5)
These extend the set of CPU-runnable ops so layers don't fall through to GPU.

### Track C: MUL_MAT Coverage (Task 3)
This enables CPU MUL_MAT for quantized models — the most impactful single change.

### Track D: Infrastructure (Tasks 6, 7, 8, 9)
Mutex separation, cost model, boundary sync, and fit_params.

### Task Dependencies

```
Task 9 (fit_params)       → no dependencies, standalone fix
Task 3 (quantized types)  → no dependencies, standalone
Task 2 (new kernels)      → no dependencies, standalone
Task 6 (mutex separation) → no dependencies, standalone
Task 1 (host-pinned buf)  → no dependencies, standalone
Task 5 (broadcast)        → no dependencies, standalone
Task 7 (cost model)       → after Task 3 (needs quantized MUL_MAT to compare)
Task 4 (async pipeline)   → after Task 1 (host-pinned may eliminate most staging)
Task 8 (boundary sync)    → after Task 4 (same synchronization concern)
```

Parallel tracks: Tasks 1, 2, 3, 5, 6, 9 can all run independently.
Task 7 depends on Task 3. Tasks 4, 8 are sequential after Task 1.

### Priority Order (impact × effort)

1. **Task 9** — fit_params guard (2 lines, unblocks budget-constrained users)
2. **Task 3** — Quantized MUL_MAT (high impact, enables CPU compute for real models)
3. **Task 1** — Host-pinned compute buffers (eliminates staging overhead entirely)
4. **Task 2** — Additional CPU kernels (SILU/GLU/ROPE/SOFT_MAX)
5. **Task 6** — DnnlGemmWrapper mutex separation (simple, reduces contention)
6. **Task 5** — Broadcast expansion (medium, enables more models)
7. **Task 4** — Async activation pipeline (cleanup remaining staging)
8. **Task 7** — Layer classification cost model (smart placement decisions)
9. **Task 8** — Boundary sync optimization (fine-grained events)

## Verification

Each task should verify:
1. **Default performance** (no regression): PP512 >= 1200, TG128 >= 68
2. **CPU offload correctness**: `GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30`
   with `ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0'` produces correct output
3. **Auto-streaming correctness**: `GGML_SYCL_VRAM_BUDGET_PCT=30` produces correct output
4. Task-specific benchmarks comparing before/after

Target: 30% VRAM budget should achieve PP512 > 400 tok/s, TG128 > 5 tok/s after Phase 2
(vs current 247/1.7).

## Critical Files

| File | Role |
|------|------|
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | CPU kernels, staging, dispatch |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | Public interface |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Classification, boundary sync, buffer types |
| `ggml/src/ggml-sycl/common.hpp` | DnnlGemmWrapper mutex, context |
| `ggml/src/ggml-sycl/gemm.hpp` | oneDNN GEMM wrappers (woq_gemm_q4_0) |
| `ggml/src/ggml-sycl/unified-cache.cpp` | oneDNN scratch mutex |
| `ggml/src/ggml-sycl/layer-streaming.hpp` | Double-buffer prefetch pattern |
| `src/llama-model.cpp` | fit_params inventory guard |
