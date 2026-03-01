# SYCL Prompt Processing Optimization Design Document

## Executive Summary

This document outlines a parallelization strategy for improving prompt processing (PP) performance in the llama.cpp SYCL backend targeting Intel Arc GPUs. The goal is to maximize throughput by exploiting all available parallelism in the hardware.

**Current State**: PP512 achieves 38.20 tok/s on Arc B580
**Target**: Maximize memory bandwidth utilization (currently ~66%) and XMX utilization (~16%)

---

## CRITICAL ARCHITECTURAL CONSTRAINTS

### No Duplicate Kernels

All optimizations MUST be implemented within the existing unified kernel architecture:

1. **Single Dispatch Point**: All matmul operations route through `unified_matmul_dispatch()` in `unified-kernel.cpp`
2. **Extend KernelPath Enum**: New variants are added to the existing `KernelPath` enum, not as separate kernels
3. **Use Unified Cache**: All weight caching uses `UnifiedCache` from `unified-cache.hpp`
4. **Respect Layout System**: Work with existing `ggml_layout_mode` (AOS/SOA) infrastructure

### Unified Kernel Extension Points

```cpp
// unified-kernel.hpp - Add new path variant
enum class KernelPath {
    DMMV,        // batch=1, memory-bound
    MMVQ,        // small batch
    ESIMD_DPAS,  // large batch (current)
    ESIMD_LARGE_TILE  // NEW: larger tiles for PP optimization
};

// unified-kernel.cpp - Add dispatch logic
if (batch_size >= large_tile_threshold && can_use_large_tile(...)) {
    // Dispatch to new large-tile variant within same kernel file
    esimd_large_tile_kernel_impl<...>(...);
}
```

### Unified Cache Integration

The unified cache already supports:
- **Layout-aware caching**: `ggml_layout_mode` parameter in all cache operations
- **SoA/AoS layouts**: Used by DMMV path for memory coalescing
- **Fast lookup path**: `try_get_cached_fast()` with reader-writer lock
- **Async fill**: `ensure_cached_layout()` for graph-safe operations

New kernel variants MUST:
- Use `cache.get()` / `cache.try_get_cached_fast()` for weight lookup
- Respect `data_layout` parameter from dispatch
- Not introduce new caching mechanisms

---

## 1. Hardware Analysis: Intel Arc B580

### 1.1 Compute Resources

| Resource | Count | Notes |
|----------|-------|-------|
| Xe-cores | 20 | Each contains 8 XVE + 8 XMX engines |
| XVE (Vector Engines) | 160 | SIMD16 FP16/FP32 ALUs |
| XMX (Matrix Engines) | 160 | Systolic arrays for matrix ops |
| Max Threads per XVE | 8 | Hardware thread contexts |
| Total HW Threads | 1,280 | 160 XVE × 8 threads |

### 1.2 Memory Hierarchy

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| Registers | 4KB/thread | ~TB/s | 1 cycle |
| SLM (Shared Local Memory) | 64KB/work-group | ~2 TB/s | ~20 cycles |
| L1 Cache | 192KB/Xe-core | ~1 TB/s | ~40 cycles |
| L2 Cache | 18MB total | ~500 GB/s | ~100 cycles |
| GDDR6 VRAM | 12GB | 456 GB/s | ~300 cycles |

### 1.3 XMX Instruction Characteristics

The `dpas` (Dot Product Accumulate Systolic) instruction:
- **Tile size**: 8×16×16 (M×N×K) for FP16
- **Throughput**: 1 dpas/cycle per XMX engine
- **Peak**: 160 engines × 8×16×16 × 2 ops × 2.85 GHz = ~47 TFLOPS FP16

---

## 2. Current Implementation Analysis

### 2.1 Prompt Processing Kernel Structure

```
For each transformer layer (32 layers in Mistral-7B):
├── QKV Projection: [B×512, 4096] × [4096, 4096×3] → [B×512, 12288]
├── Attention:
│   ├── Q×K^T: [B×512, 128] × [128, 512] → [B×512, 512] per head
│   ├── Softmax
│   └── Score×V: [B×512, 512] × [512, 128] → [B×512, 128] per head
├── Output Projection: [B×512, 4096] × [4096, 4096] → [B×512, 4096]
├── FFN Gate/Up: [B×512, 4096] × [4096, 14336×2] → [B×512, 28672]
├── FFN Activation (SiLU + multiply)
└── FFN Down: [B×512, 14336] × [14336, 4096] → [B×512, 4096]
```

### 2.2 Current Tile Configuration

```cpp
// XMX joint_matrix path (used for PP > 1)
constexpr int XMX_TM = 8;   // Output rows per tile
constexpr int XMX_TN = 16;  // Output columns per tile
constexpr int XMX_TK = 32;  // K-dimension per iteration

// Cooperative ESIMD path
constexpr int COOP_WG_TILES_M = 2;  // 2 M-tiles (16 rows)
constexpr int COOP_WG_TILES_N = 1;  // 1 N-tile (16 cols)
constexpr int COOP_WG_SIZE = 32;    // 32 work-items
```

**Current output tile**: 16×16 = 256 elements per work-group

### 2.3 Identified Bottlenecks

1. **Small Work-Group Output Size**: 16×16 output means high dispatch overhead
2. **Low SLM Utilization**: Using 1KB of 64KB available SLM
3. **Single K-Buffer**: No double-buffering to hide memory latency
4. **Insufficient N-Parallelism**: Only 1 N-tile per work-group

---

## 3. Parallelization Strategy

### 3.1 Levels of Parallelism

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LEVEL                             │
│  • Multiple transformer layers can overlap (pipeline)            │
│  • KV cache operations parallel with compute                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      KERNEL LEVEL                                │
│  • M-dimension: Parallelize over output rows (batch × seq)       │
│  • N-dimension: Parallelize over output columns (model dim)      │
│  • K-dimension: Sequential with accumulation                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    WORK-GROUP LEVEL                              │
│  • Multiple sub-groups cooperate on shared tile                  │
│  • SLM used for weight/activation sharing                        │
│  • Barrier synchronization for cooperative loading               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SUB-GROUP LEVEL (SIMD)                        │
│  • 16-wide SIMD execution (SIMD16)                               │
│  • XMX dpas instruction for 8×16×16 matrix tiles                 │
│  • Register blocking for data reuse                              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Proposed Work Distribution

**Target Configuration**:
```cpp
// Large cooperative kernel
constexpr int LARGE_WG_TILES_M = 4;   // 4 M-tiles (32 rows)
constexpr int LARGE_WG_TILES_N = 4;   // 4 N-tiles (64 cols)
constexpr int LARGE_WG_SIZE = 256;    // 16 sub-groups
constexpr int LARGE_K_TILES = 4;      // 4 K-tiles in SLM (double-buffer × 2)

// Output per work-group: 32×64 = 2048 elements (8× current)
```

**Work Distribution**:
```
PP512 with M=512, N=4096, K=4096:

Grid: ceil(512/32) × ceil(4096/64) = 16 × 64 = 1024 work-groups
Each WG: 256 threads = 16 sub-groups
Total threads: 262,144 (fits in 160 XVE × 8 = 1280 concurrent)

Sub-group assignment within work-group:
  SG 0-3:   Load weights (4 N-tiles × K_TILE)
  SG 4-7:   Load activations (4 M-tiles × K_TILE)
  SG 8-15:  Compute (16 sub-groups × 8×16 = 16 output tiles)
```

---

## 4. Implementation Design (Unified Kernel Integration)

### 4.1 File Structure - NO NEW FILES

All changes go in existing unified kernel files:

```
ggml/src/ggml-sycl/
├── unified-kernel.hpp    # Add ESIMD_LARGE_TILE to KernelPath enum
├── unified-kernel.cpp    # Add large_tile_esimd_kernel_impl() function
├── unified-cache.hpp     # No changes needed
└── dispatch.hpp          # May need threshold config
```

### 4.2 Unified Kernel Dispatch Integration

```cpp
// In unified-kernel.cpp, within unified_matmul_dispatch():

// After existing ESIMD_DPAS check, before scalar fallback:
#if GGML_SYCL_ESIMD_AVAILABLE
    // Large-tile ESIMD path for prompt processing (PP > threshold)
    if (esimd_enabled_by_gating &&
        batch_size >= get_large_tile_min_batch() &&  // e.g., 128
        can_use_large_tile_esimd(args.M, args.N, args.K)) {

        // Dispatch to large-tile variant (same file, different template params)
        large_tile_esimd_kernel_impl<LARGE_TILE_M, LARGE_TILE_N, LARGE_TILE_K>(
            q, args, slm_weights, slm_activations);
        return;
    }

    // Existing cooperative ESIMD path (smaller tiles)
    if (esimd_enabled_by_gating && can_use_cooperative_esimd(...)) {
        // ... existing code ...
    }
#endif
```

### 4.3 Large Tile Kernel Implementation

The new kernel is implemented as a template function in `unified-kernel.cpp`:

```cpp
// New constants alongside existing COOP_* constants
constexpr int LARGE_TILE_M = 32;    // 4× current (16→32)
constexpr int LARGE_TILE_N = 64;    // 4× current (16→64)
constexpr int LARGE_TILE_K = 32;    // Same K-tile
constexpr int LARGE_WG_SIZE = 256;  // 8× current (32→256)

/**
 * Large-tile ESIMD kernel for prompt processing optimization.
 *
 * This is NOT a separate kernel - it's a variant within unified_matmul_dispatch()
 * that uses larger output tiles for better memory bandwidth utilization.
 *
 * Integrates with:
 * - UnifiedCache: weights obtained via args.weights_data (already cached)
 * - LayoutMode: respects args.layout_mode (AOS/SOA)
 * - KernelPath: selected via ESIMD_LARGE_TILE path
 */
template <int TILE_M, int TILE_N, int TILE_K>
SYCL_ESIMD_FUNCTION void large_tile_esimd_kernel_impl(
    const KernelArgs& args,
    int64_t m_start,
    int64_t n_start,
    sycl::local_accessor<sycl::half, 1>& slm_weights,
    sycl::local_accessor<sycl::half, 1>& slm_activations
) {
    // Uses same KernelArgs structure as existing kernels
    // Weights already fetched from UnifiedCache by caller
    // Layout already applied based on args.layout_mode

    // Phase 1: Cooperative loading with double-buffering
    // Phase 2: XMX compute with register blocking
    // Phase 3: Write-back with boundary handling
}
```

### 4.2 SLM Layout

```
SLM Budget: 64KB per work-group

Weights buffer (double-buffered):
  Size: 64 cols × 32 K × 2 bytes × 2 buffers = 8KB

Activations buffer (double-buffered):
  Size: 32 rows × 32 K × 2 bytes × 2 buffers = 4KB

Total: 12KB (leaves 52KB for additional optimizations)

Alternative: Larger K-tile for better bandwidth utilization
  Weights: 64 × 64 × 2 × 2 = 16KB
  Activations: 32 × 64 × 2 × 2 = 8KB
  Total: 24KB
```

### 4.3 Double-Buffering Pipeline

```
Iteration k=0:
  Buffer A: Load W[0:64, 0:32], A[0:32, 0:32]
  Buffer B: (empty)
  Compute: (idle, first iteration)

Iteration k=1:
  Buffer A: Compute W[A] × A[A] → C
  Buffer B: Load W[0:64, 32:64], A[0:32, 32:64]

Iteration k=2:
  Buffer A: Load W[0:64, 64:96], A[0:32, 64:96]
  Buffer B: Compute W[B] × A[B] → C

... (continue alternating)
```

### 4.4 Memory Access Pattern

**Current (inefficient)**:
```
Each work-group: loads full K-strip independently
Multiple work-groups with same M-index: redundant A loads
Multiple work-groups with same N-index: redundant W loads
```

**Proposed (optimized)**:
```
Work-groups along M-dimension share weights via L2 cache
Work-groups along N-dimension share activations via L2 cache
Tiling chosen to maximize L2 reuse before eviction
```

### 4.5 Unified Cache Integration (REQUIRED)

The large-tile kernel MUST use the existing unified cache, not create new caching:

```cpp
// In ggml-sycl.cpp, init_tensor() or compute_mul_mat():

// 1. Weight lookup uses existing cache path
void * weight_ptr = nullptr;
if (unified_cache_enabled()) {
    // Fast path: try shared_lock lookup first
    weight_ptr = cache.try_get_cached_fast(cache_id, data_layout);

    if (!weight_ptr) {
        // Slow path: ensure cached with layout conversion
        auto result = cache.ensure_cached_layout({
            .key = cache_id,
            .src_ptr = tensor->data,
            .src_size = ggml_nbytes(tensor),
            .dst_size = dst_size,
            .type = cache_entry_type::DENSE_WEIGHT,
            .layer_id = layer_id,
            .layout = data_layout,  // AOS or SOA based on batch size
            // ... other fields
        }, deps);
        weight_ptr = result.device_ptr;
    }
}

// 2. Pass to kernel via existing KernelArgs
KernelArgs args = {
    .weights_data = weight_ptr,    // From unified cache
    .activations_data = src1_ptr,  // Activations (not cached)
    .output_data = dst_ptr,
    .M = M, .N = N, .K = K,
    .layout_mode = static_cast<int>(data_layout),
    // ... other fields
};

// 3. Dispatch to large-tile kernel (inside unified_matmul_dispatch)
unified_matmul_dispatch(q, args, device_id);
```

**Key Points**:
- Weights are ALREADY cached before kernel dispatch
- Kernel receives `device_ptr` from cache, not raw tensor data
- Layout (AOS/SOA) is determined at cache time, kernel respects it
- No duplicate caching or layout conversion inside kernel

### 4.6 Environment Variables for Control

```bash
# Enable large-tile path (default: auto based on batch size)
GGML_SYCL_UNIFIED_LARGE_TILE=1

# Minimum batch size for large-tile dispatch (default: 128)
GGML_SYCL_LARGE_TILE_MIN_BATCH=128

# Force specific tile configuration (for testing)
GGML_SYCL_LARGE_TILE_M=32
GGML_SYCL_LARGE_TILE_N=64
```

---

## 5. Flash Attention Optimization

### 5.1 Current Implementation

The SYCL backend already has Flash Attention v2:
- `fattn-xmx-f16.hpp`: XMX-accelerated attention
- `fattn-v2-partition.hpp`: Partitioned KV for long contexts
- `fattn-esimd.hpp`: ESIMD variant

### 5.2 Opportunities for Improvement

1. **Head Parallelism**: Distribute attention heads across Xe-cores
2. **Sequence Parallelism**: For PP512, process multiple query positions in parallel
3. **Fused QKV**: Avoid intermediate buffer for QKV projection

```cpp
// Current: Separate kernels
Q = linear(X, Wq);  // Kernel 1
K = linear(X, Wk);  // Kernel 2
V = linear(X, Wv);  // Kernel 3
Attn = flash_attn(Q, K, V);  // Kernel 4

// Proposed: Fused QKV + Attention
QKV = fused_qkv_linear(X, Wqkv);  // Single kernel
Attn = flash_attn_from_qkv(QKV);  // Integrated attention
```

---

## 6. Quantization-Aware Optimizations

### 6.1 Q4_0 Dequantization Pipeline

Current: Sequential dequantization per block
```cpp
for each block in K-dimension:
    load 18 bytes (16 weights + 2 byte scale)
    dequantize to 32 FP16 values
    accumulate
```

Proposed: Vectorized dequantization
```cpp
// Load 4 blocks at once (72 bytes = 128 weights)
simd<uint8_t, 64> packed_weights = block_load<64>(weight_ptr);
simd<half, 4> scales = block_load<4>(scale_ptr);

// Vectorized extraction using shuffle operations
simd<half, 128> weights = vectorized_dequant(packed_weights, scales);
```

### 6.2 INT8 Dynamic Quantization

For compute-bound operations, dynamic INT8 can improve XMX throughput:
- INT8 dpas: 8×16×32 per instruction (2× K-dimension)
- Requires activation quantization per tile
- Trade-off: Slight accuracy loss for 2× throughput

---

## 7. Implementation Phases

### Phase 1: Larger Cooperative Tiles (2 weeks)

**Files Modified**: `unified-kernel.hpp`, `unified-kernel.cpp` ONLY

- Add `ESIMD_LARGE_TILE` to `KernelPath` enum
- Add `large_tile_esimd_kernel_impl<32, 64, 32>()` template function
- Add dispatch logic in `unified_matmul_dispatch()` with batch threshold
- Add environment variable `GGML_SYCL_UNIFIED_LARGE_TILE`
- Use existing `KernelArgs` structure (no new structs)
- Get weights from unified cache (already passed via `args.weights_data`)

**Validation**:
- Run existing `test-kernel-dispatch` tests
- Run numerical accuracy comparison vs baseline
- Run `llama-bench` PP sweep

**Expected improvement**: 20-30%

### Phase 2: Double-Buffered K-Dimension (1 week)

**Files Modified**: `unified-kernel.cpp` ONLY

- Add double-buffer logic within `large_tile_esimd_kernel_impl()`
- Use ESIMD named barriers for synchronization
- Overlap K-tile loading with compute

**No changes to cache or dispatch logic**

**Expected improvement**: 10-20%

### Phase 3: SLM Weight Caching (2 weeks)

**Files Modified**: `unified-kernel.cpp` ONLY

- Increase SLM allocation for weight tiles
- Cache 2-4 K-tiles worth of weights in SLM
- Reuse across multiple dpas iterations

**No changes to unified cache** (SLM is per-kernel, not persistent)

**Expected improvement**: 15-25%

### Phase 4: Fused Operations (3 weeks)

**Files Modified**: May require new function in `unified-kernel.cpp`

- Fuse QKV projection (requires graph-level change)
- This phase needs more investigation - may require ggml graph fusion
- Consider as separate RFC

**Expected improvement**: 10-15%

---

## 8. Testing Strategy

### 8.1 Correctness Tests

```bash
# Numerical accuracy test
./build/bin/llama-completion -m model.gguf -p "test" -n 100 --seed 42

# Compare outputs between paths
GGML_SYCL_UNIFIED_LARGE_TILE=0 ./build/bin/llama-completion ...  # Baseline
GGML_SYCL_UNIFIED_LARGE_TILE=1 ./build/bin/llama-completion ...  # Optimized
```

### 8.2 Performance Benchmarks

```bash
# Prompt processing sweep
for pp in 128 256 512 1024 2048; do
    ./build/bin/llama-bench -m model.gguf -p $pp -n 0 -r 5
done

# Memory bandwidth test
./build/bin/test-backend-ops -b SYCL
```

### 8.3 Regression Tests

- Run full test suite before/after changes
- Monitor for numerical drift (softmax precision)
- Track peak memory usage

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Register pressure | High | Medium | Reduce tile size, use SLM spilling |
| SLM bank conflicts | Medium | Medium | Pad SLM arrays, align accesses |
| Numerical precision | Low | High | Use Kahan summation, FP32 accumulators |
| ESIMD compiler bugs | Medium | High | Add fallback to joint_matrix path |
| Performance regression | Low | Medium | Feature flags for each optimization |

---

## 10. Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| PP512 tok/s | 38.20 | 60+ | llama-bench |
| Memory BW utilization | 66% | 85%+ | Calculated from tok/s |
| XMX utilization | ~16% | ~25%+ | VTune (if available) |

---

## 11. Open Questions

1. **Batch size threshold**: At what batch size should we switch from DMMV to large-tile GEMM?
2. **Tile size tuning**: Should we auto-tune tile sizes per model architecture?
3. **Mixed precision**: Is INT8 dynamic quantization acceptable for prompt processing?
4. **Multi-GPU**: How should we distribute work across multiple Intel GPUs?

---

## Appendix A: Related Work

- **cuBLAS/CUTLASS**: NVIDIA's approach uses 128×128×32 tiles with warp-level parallelism
- **oneDNN**: Intel's GEMM uses auto-tuned kernels with JIT compilation
- **Flash Attention**: Memory-efficient attention with O(N) memory vs O(N²)
- **CLBlast**: OpenCL GEMM with 14-parameter tuning space

## Appendix B: Code References

- `ggml/src/ggml-sycl/unified-kernel.cpp` - Current unified kernel implementation
- `ggml/src/ggml-sycl/fattn-xmx-f16.hpp` - Flash attention XMX kernel
- `ggml/src/ggml-sycl/tuning-engine.hpp` - Auto-tuning infrastructure
- `ggml/src/ggml-sycl/dispatch.hpp` - Kernel dispatch logic
