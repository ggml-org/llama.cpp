# Mixed CPU/GPU Token Generation Speed Improvements

**Date**: 2026-02-10
**Branch**: feature/sycl-coalescing
**Current performance**: TG128 = 1.67 tok/s at 30% VRAM budget (Mistral 7B Q4_0, Arc B580)

## Current Bottleneck Analysis

With `GGML_SYCL_VRAM_BUDGET_PCT=30`, ~18 of 32 layers are evicted to host memory and computed on CPU. Each token takes ~600ms, broken down as:

| Component | Time (ms) | % of Total | Notes |
|-----------|-----------|------------|-------|
| CPU MUL_MAT (dequant + dnnl_sgemm) | ~500 | 83% | THE bottleneck |
| Staging transfers (device<->host) | ~50 | 8% | Double-buffered, overlapped |
| Element-wise ops (RMS_NORM, ADD, etc.) | ~30 | 5% | SYCL parallel_for on CPU queue |
| Dispatch/sync overhead | ~20 | 3% | Per-tensor events, negligible |

### Why CPU MUL_MAT is so slow

`cpu_mul_mat` in `cpu-dispatch.cpp` does this for every MUL_MAT on a CPU layer:

1. **Dequantize** entire N x K weight matrix from Q4_0 to F32 (~100-150ms)
   - For attention: 4096 x 4096 = 16M floats = 64 MB temporary buffer
   - For MLP: 14336 x 4096 = 58M floats = 224 MB temporary buffer
2. **Call dnnl_sgemm** for F32 GEMV with M=1 (~300-400ms)
   - oneDNN's BLAS is tuned for large M (batch); M=1 has massive overhead
   - Reads entire 64-224 MB dequant buffer for a single output row

Per layer: 7 MUL_MAT ops x ~70ms each = ~500ms. That's 83% of the token time.

---

## Option 1: CPU Quantized Dot Product (P0 - Highest Impact)

**Expected improvement**: 3-8x (target: 5-12 tok/s)
**Effort**: Medium (1 file, ~50 lines changed)
**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

### What

Replace the dequant->F32->dnnl_sgemm path with ggml's hand-optimized quantized dot products for M<=4 (TG).

### How

ggml-cpu provides `ggml_get_type_traits_cpu(type)` which returns:
- `vec_dot`: Quantized dot product function (e.g., `ggml_vec_dot_q4_0_q8_0`)
- `vec_dot_type`: Type to quantize activations to (Q8_0 for Q4_0 weights)
- `from_float`: F32 -> vec_dot_type quantization function

For M=1 TG:
1. Quantize the 1 x K activation vector from F32 to Q8_0 (~4 KB, cheap)
2. For each of N output rows: call `vec_dot(K, &result, weight_row_q4_0, activation_q8_0)`
3. No weight dequantization at all

### Why it's fast

| Metric | Current (dequant+GEMM) | Proposed (vec_dot) |
|--------|------------------------|--------------------|
| Weight reads per layer | 64 MB (F32 dequant buffer) | 9 MB (raw Q4_0) |
| Activation reads | 16 KB (F32) | 4 KB (Q8_0) |
| Temp buffer | 64 MB (N x K floats) | 4 KB (1 x K Q8_0) |
| L1 cache fit | No (64 MB thrashes everything) | Yes (2.2 KB weight row + 4 KB activation) |
| BLAS overhead | High (M=1 GEMV setup) | Zero (direct function call) |

The vec_dot functions are hand-tuned with AVX2/AVX512 intrinsics in ggml-cpu. They're exactly what the CPU backend uses for its own MUL_MAT and are heavily optimized for this workload.

### Compatibility

- `ggml-sycl.cpp` already `#include "ggml-cpu.h"` (line 52)
- Works for all quantized types that have a vec_dot implementation (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, K-quants)
- Falls back to existing dequant+sgemm for M>4 (prompt processing) or unsupported types

---

## Option 2: Async Double-Buffered Weight Prefetch (P2)

**Expected improvement**: 15-30%
**Effort**: High (touches layer-streaming, cpu-dispatch, ggml-sycl graph loop)
**Files**: `layer-streaming.cpp`, `cpu-dispatch.cpp`, `ggml-sycl.cpp`

### What

Prefetch the NEXT layer's host-pinned weights into L2/L3 cache while the current layer computes on CPU. Overlaps memory access latency with computation.

### How

The layer-streaming infrastructure already has `layer_streaming_prefetch_next()`. Wire it to the CPU dispatch path:

1. At start of each CPU layer's first MUL_MAT, issue `__builtin_prefetch()` or `_mm_prefetch()` for next layer's weight addresses
2. While current layer's 7 MUL_MATs execute (~70ms), next layer's weights warm into L2/L3
3. When next layer starts, weights are already in cache -> fewer memory stalls

### Limitation

Only helps if CPU compute is slower than memory prefetch (true for large layers). For small layers, prefetch may not complete before next layer starts.

---

## Option 3: CPU Op Fusion (P3)

**Expected improvement**: 5-10%
**Effort**: Medium (cpu-dispatch.cpp restructuring)
**File**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

### What

Fuse sequential element-wise operations to reduce staging round-trips. Currently each op does: stage-in -> compute -> stage-out. Fusing N ops saves (N-1) staging round-trips.

### Candidates

- `RMS_NORM` -> `MUL` (always sequential in transformer blocks)
- `ADD` -> `SILU` (residual + activation)
- `ADD` -> `MUL` (residual + scale)

### How

1. Detect fusible op sequences in the graph before dispatch
2. Execute fused kernel that keeps data in host staging buffer between ops
3. Only stage-out after the final op in the fused sequence

### Limitation

Saves ~30ms/token (staging overhead), which is only 5% of total. Not transformative.

---

## Option 4: Hybrid Per-Op Dispatch (P4)

**Expected improvement**: Variable (depends on op mix)
**Effort**: High (dispatch logic in ggml-sycl.cpp)
**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

### What

Currently, ALL ops in a CPU layer go to CPU. But some ops are memory-light and would be faster on GPU even with the PCIe transfer cost:

- `SOFT_MAX`: Reads/writes small attention matrix, compute-heavy (exp, normalize)
- `ROPE`: Small tensor, transcendental math
- `SCALE`: Trivial element-wise

### How

Per-op cost model: if GPU_compute_time + 2*PCIe_transfer_time < CPU_compute_time, route to GPU.

### Limitation

Increases GPU<->CPU transitions (currently only 2 per token). Each transition has ~20-35us overhead, but many transitions could add up. Need careful profiling.

---

## Option 5: GPU Subgraph Replay (P5)

**Expected improvement**: Minor for TG, moderate for PP
**Effort**: High (graph infrastructure in ggml-sycl.cpp)
**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

### What

Currently, graph replay is ENTIRELY disabled when any CPU layers exist (`has_cpu_layers` gate at line 30582-30594). This means all 1158 graph nodes execute individually, even the ~500 GPU-only nodes.

### How

Split the graph into GPU-only subgraphs and CPU sections:
1. Record GPU subgraphs (layers 0-14 as one graph)
2. Execute CPU layers individually (layers 15-31)
3. Replay GPU subgraphs on subsequent tokens

### Limitation

For TG, the GPU portion already runs fast (~14ms for 15 layers). Replay savings would be ~1-2ms/token at most. The bottleneck is CPU layers (500ms), so this doesn't help much.

More impactful for PP where batch>4 routes everything to GPU.

---

## Option 6: fit_params Budget Fix (P1 - Correctness)

**Expected improvement**: Correctness (not performance)
**Effort**: Low (already applied!)
**File**: `src/llama-model.cpp`

### Status: ALREADY DONE

Both SYCL tensor inventory blocks in `load_tensors()` already have `if (!ml.no_alloc)` guards:
- Line 8546: Early inventory block
- Line 8715: Post-allocation refresh block

This prevents `fit_params` from hanging when `GGML_SYCL_VRAM_BUDGET_PCT < 100` by skipping inventory collection during measurement probes.

---

## Recommendation

**Start with Option 1**. It's the highest-impact change (3-8x improvement) with moderate effort (single file, ~50 lines). It directly addresses the root cause: dequantization + BLAS GEMV overhead for M=1.

After Option 1, the bottleneck shifts from CPU MUL_MAT to staging/element-wise ops, making Options 2 and 3 more impactful as follow-ups.

### Priority Order

1. **Option 1** - CPU quantized dot product (3-8x TG improvement)
2. **Option 2** - Async weight prefetch (15-30% on top of Option 1)
3. **Option 3** - Op fusion (5-10% from staging reduction)
4. **Option 5** - GPU subgraph replay (minor TG, helps PP)
5. **Option 4** - Hybrid dispatch (needs profiling data)
