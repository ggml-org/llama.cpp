# Data-Local Compute: Detailed Implementation Plan

## Overview

Replace weight streaming (and eventually fit_params) with **data-local compute**: when the unified cache tiers a weight to host pinned memory, dispatch that layer's computation to a SYCL CPU device instead of streaming the weight to GPU. Zero weight movement — compute goes to where data lives.

## Architecture

```
Unified Cache tiers weights at load time:
  VRAM tier    → dispatch to GPU queue (XMX/ESIMD/MMVQ kernels)
  PINNED_HOST  → dispatch to CPU queue (oneDNN + portable SYCL kernels)
  MMAP         → dispatch to CPU queue (oneDNN, direct file-backed access)

Activations transfer at device boundaries:
  [GPU layers 0..N] → one GPU→CPU transfer → [CPU layers N+1..31]
```

## Why It Works

1. **oneDNN is device-agnostic**: `dnnl::sycl_interop::make_engine(dev, ctx)` works on both GPU and CPU devices. The engine/stream/primitive infrastructure already handles this.
2. **Host pinned memory is CPU-accessible**: `sycl::malloc_host()` returns regular host pointers. CPU computes directly, zero copy.
3. **Primitive caching handles multi-engine**: The existing `DnnlPrimitiveCache` in gemm.hpp compares engines and invalidates when device changes. GPU primitives and CPU primitives coexist.
4. **CPU device available**: `opencl:cpu:0` — Intel Core Ultra 7 265K (24 cores, AVX-512).

## Design Decisions

- **Graph replay**: GPU-only graph, CPU layers execute inline between graph segments
- **Tier-to-device mapping**: Static at load time (unified cache decides tiers, assignment is fixed)
- **KV cache (Phase 1)**: All on host pinned memory (both GPU and CPU access it)
- **KV cache (Phase 2)**: Per-device — GPU layers' KV in VRAM, CPU layers' KV on host
- **Weight streaming**: Kept as alternative path, env var toggle to compare

## Phase 1: CPU Queue + oneDNN Dispatch (Prove Concept)

### Goal
Dispatch host-tier layers to CPU via oneDNN. No graph replay. Validate correctness and measure baseline perf.

### Task 1.1: CPU Device Discovery and Queue Creation

**File**: `ggml/src/ggml-sycl/common.hpp` (ggml_sycl_device_info)

Add CPU device support to device initialization:

```cpp
struct ggml_sycl_device_info {
    // ... existing fields ...
    bool has_cpu_device = false;
    int  cpu_device_id = -1;         // Index into devices[] for CPU
    sycl::queue * cpu_queue = nullptr;
};
```

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~line 2708, ggml_sycl_init)

During device enumeration:
1. After enumerating Level Zero GPU devices, probe for OpenCL CPU device
2. Create CPU queue: `sycl::queue cpu_q{sycl::cpu_selector_v}`
3. Create CPU oneDNN engine: `dnnl::sycl_interop::make_engine(cpu_dev, cpu_ctx)`
4. Store in device_info for later access
5. Gate with env var: `GGML_SYCL_CPU_OFFLOAD=1` (opt-in during development)

**Key concern**: CPU device uses OpenCL backend, GPU uses Level Zero. USM pointers from `sycl::malloc_host()` on one backend may not be directly accessible from the other. Need to verify cross-backend USM compatibility, or allocate host memory that both can access.

**Verification**: Allocate `sycl::malloc_host()` on GPU context, read it from CPU queue. If this fails, we need `sycl::malloc_shared()` or a common platform context.

### Task 1.2: Tier-Based Device Routing in compute_forward

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~line 23183, ggml_sycl_compute_forward)

Add routing logic before the main switch statement:

```cpp
bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    // NEW: Check if this op's primary weight is in host tier
    if (g_device_info.has_cpu_device && should_dispatch_to_cpu(dst, ctx)) {
        return ggml_sycl_compute_forward_cpu(ctx, dst);
    }

    // EXISTING: GPU dispatch (unchanged)
    switch (dst->op) { ... }
}
```

**Routing logic** (`should_dispatch_to_cpu`):
- For MUL_MAT: check src0 (weight tensor) tier
- If src0 tier is PINNED_HOST or MMAP → dispatch to CPU
- For non-MUL_MAT ops (norm, add, rope, softmax): check if we're in a "CPU layer block"
  - Track current layer index from tensor name (e.g., "blk.15.attn_q.weight" → layer 15)
  - If layer's weight tier is PINNED_HOST → all ops in that layer go to CPU
- For global ops (embeddings, output): stay on GPU (they're typically in VRAM)

### Task 1.3: CPU Compute Path

**New file**: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

Implement `ggml_sycl_compute_forward_cpu()`:

```cpp
bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = g_device_info.cpu_queue;

    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            // Use oneDNN on CPU queue
            return ggml_sycl_mul_mat_cpu(ctx, dst, cpu_q);

        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_ROPE:
        case GGML_OP_SOFT_MAX:
            // Use portable SYCL kernels on CPU queue
            return ggml_sycl_op_generic_cpu(ctx, dst, cpu_q);

        default:
            // Unsupported op on CPU — fall back to GPU with DMA
            return false;
    }
}
```

For `ggml_sycl_mul_mat_cpu`: Reuse the existing oneDNN GEMM path but submit to CPU queue:
- Get/create CPU oneDNN engine from cpu_queue
- Call `DnnlGemmWrapper::gemm()` with CPU queue instead of GPU queue
- The primitive cache handles engine mismatch automatically

For portable ops: The existing norm/add/rope/softmax kernels are pure SYCL (no XMX/ESIMD). Submit them to cpu_queue instead of gpu_queue.

### Task 1.4: Activation Transfers at Device Boundaries

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph_compute_impl, ~line 24437)

At the boundary between GPU and CPU layers:

```cpp
// In the main node dispatch loop:
for (int i = 0; i < cgraph->n_nodes; i++) {
    ggml_tensor * node = cgraph->nodes[i];

    bool this_on_cpu = should_dispatch_to_cpu(node, ctx);
    bool prev_on_cpu = (i > 0) ? should_dispatch_to_cpu(cgraph->nodes[i-1], ctx) : false;

    // Device boundary: GPU→CPU
    if (this_on_cpu && !prev_on_cpu && i > 0) {
        // Sync GPU queue, copy activation to host pinned
        ctx.stream()->wait();
        // Activation is already accessible from CPU via USM
        // (if USM host/shared, no explicit copy needed)
    }

    // Device boundary: CPU→GPU
    if (!this_on_cpu && prev_on_cpu) {
        // Sync CPU queue, ensure activation is visible to GPU
        g_device_info.cpu_queue->wait();
        // Copy activation from host to GPU VRAM if needed
    }

    ggml_sycl_compute_forward(ctx, node);
}
```

**Key insight**: With USM, host-allocated activations may be accessible from both devices without explicit copies. Need to verify: does the GPU queue see data written by the CPU queue on host pinned memory? If yes, we only need queue synchronization (wait), not explicit memcpy.

### Task 1.5: Disable Graph Replay for Mixed Execution

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp` (~line 30213, graph_compute)

When CPU offload is active, disable graph replay for the first implementation:

```cpp
bool has_cpu_layers = g_device_info.has_cpu_device &&
                      unified_cache_has_host_tier_weights();

if (has_cpu_layers) {
    // Phase 1: no graph replay with CPU layers
    // Phase 3 will add segmented graphs
    return ggml_backend_sycl_graph_compute_impl(backend, cgraph);
}
```

### Task 1.6: Integration Test

Test scenarios:
1. **Default (no budget pressure)**: All weights in VRAM → all GPU → should match baseline perf
2. **30% budget, CPU offload**: ~15 layers GPU, ~17 layers CPU → correct output, measure perf
3. **30% budget, weight streaming**: Same budget but `GGML_SYCL_FORCE_STREAMING=1` → compare perf
4. **Correctness**: Both paths should produce identical output with seed 42, temp 0

Environment variables:
- `GGML_SYCL_CPU_OFFLOAD=1`: Enable data-local CPU compute (new)
- `GGML_SYCL_FORCE_STREAMING=1`: Force weight streaming (existing)
- Neither: Default behavior (unified cache tiers, no CPU offload, no streaming)

## Phase 2: Activation Transfer Optimization

### Task 2.1: Minimize Transfers

With contiguous layer blocks (GPU first, CPU last — which is what unified cache's LRU eviction naturally produces since later layers are loaded last and evicted first):
- Only ONE GPU→CPU boundary
- Activation size: ~8 MB (4096 * 4096 * fp16 / 2 for half the attention)
- Use async DMA: `gpu_queue.memcpy(host_buf, gpu_buf, size)` overlapped with compute

### Task 2.2: Per-Device KV Cache

Instead of all KV on host:
- GPU layers (0..N): KV allocated in VRAM via normal path
- CPU layers (N+1..31): KV allocated in host pinned memory
- The unified cache tracks this per-layer
- Eliminates KV DMA for both GPU and CPU layers

### Task 2.3: CPU Kernel Optimization

Profile CPU-executed layers and optimize hot paths:
- oneDNN matmul on CPU may need different tiling/blocking than GPU
- Norm/softmax might benefit from OpenMP or TBB parallelism via SYCL CPU runtime
- Consider: are there quantized matmul primitives in oneDNN for CPU? (WOQ support)

## Phase 3: Segmented Graph Replay

### Task 3.1: GPU Graph Segments

Split the computation graph into GPU-only segments:
```
[GPU Graph Segment 1: layers 0..N]
  → record once, replay on subsequent tokens
[CPU Inline: layers N+1..31]
  → execute normally each token
```

This restores the 12.5x graph replay benefit for GPU layers while CPU layers execute inline.

### Task 3.2: Implementation

In `graph_compute`:
1. Scan graph nodes, identify contiguous GPU-only segments
2. For each GPU segment: record as SYCL graph, replay on subsequent calls
3. Between segments: sync GPU, execute CPU ops, sync CPU, resume GPU graph

## Phase 4: Remove fit_params Dependency

### Task 4.1: Auto-Enable CPU Offload Under Budget Pressure

When unified cache detects model exceeds VRAM budget:
1. Auto-enable CPU offload (no env var needed)
2. Auto-enable KV host for CPU layers
3. Skip fit_params entirely for SYCL backends

### Task 4.2: Context Auto-Sizing

After weight loading, query remaining VRAM:
- If KV fits in VRAM: use full context
- If not: either enable KV host (preserve context) or reduce context to fit

### Task 4.3: Remove fit_params for SYCL

In `common_init_result` (common.cpp):
```cpp
if (backend_is_sycl_with_unified_cache()) {
    // Skip fit_params — unified cache + CPU offload handles everything
    params.fit = false;
}
```

## Performance Expectations

| Scenario | Weight Streaming | Data-Local CPU | Why |
|----------|-----------------|----------------|-----|
| 30% budget, TG | ~2.8 tok/s | ~10-15 tok/s | CPU compute >> PCIe streaming |
| 30% budget, PP | ~? | ~20-30 tok/s | oneDNN CPU is well-optimized |
| Default (all VRAM) | N/A | ~70.5 tok/s | No change — all GPU |

Data-local should be 3-5x faster than streaming because:
- Zero weight data movement (weights stay on host, CPU computes directly)
- Only 8 MB activation transfer at one boundary vs 117 MB/layer weight streaming
- oneDNN on Intel CPU (AVX-512, AMX) is much faster than PCIe bandwidth-limited GPU streaming

## Risk Assessment

### USM Cross-Backend Compatibility (HIGH RISK)
`sycl::malloc_host()` on Level Zero context may not be accessible from OpenCL CPU context. If this fails, we need:
- Option A: Use `sycl::malloc_shared()` (accessible from all devices but may be slower)
- Option B: Create a common SYCL platform context spanning both backends
- Option C: Explicit memcpy between backends (defeats zero-copy goal)
- **Mitigation**: Test USM compatibility in Task 1.1 before committing to architecture

### oneDNN CPU Performance with Quantized Types (MEDIUM RISK)
oneDNN on CPU may not have optimized kernels for Q4_0 dequantization. GPU path uses specialized MMVQ kernels. CPU path would need:
- Dequantize Q4_0 → FP32/FP16 → oneDNN matmul
- Or: Use oneDNN WOQ (weight-only quantization) if available for CPU
- **Mitigation**: Profile in Phase 1, optimize in Phase 2

### Graph Replay Regression (LOW RISK)
Phase 1 disables graph replay entirely for mixed execution. This affects TG performance for GPU layers. Phase 3 restores segmented replay.
- **Mitigation**: Segmented graph replay is Phase 3 priority

## Files to Create/Modify

| File | Change | Phase |
|------|--------|-------|
| `ggml-sycl/common.hpp` | Add CPU device fields to device_info | 1 |
| `ggml-sycl/ggml-sycl.cpp` | CPU device init, dispatch routing, boundary transfers | 1 |
| `ggml-sycl/cpu-dispatch.cpp` (NEW) | CPU compute forward implementation | 1 |
| `ggml-sycl/cpu-dispatch.hpp` (NEW) | CPU dispatch API | 1 |
| `ggml-sycl/CMakeLists.txt` | Add new source files | 1 |
| `ggml-sycl/unified-cache.cpp` | Query tier per tensor for routing | 1 |
| `ggml-sycl/ggml-sycl.cpp` | Segmented graph replay | 3 |
| `common/common.cpp` | Skip fit_params for SYCL | 4 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_SYCL_CPU_OFFLOAD=1` | OFF | Enable data-local CPU compute for host-tier weights |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Force weight streaming (existing, for comparison) |
| `GGML_SYCL_CPU_OFFLOAD_AUTO=1` | OFF (Phase 4: ON) | Auto-enable CPU offload when model exceeds budget |
