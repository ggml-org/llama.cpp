# Coalesced Memory Layout Optimization for SYCL Backend

**Date:** 2025-12-30
**Status:** Design Complete
**Scope:** All SoA-capable quantization formats (Q4_0, Q4_K, Q6_K, Q8_0, MXFP4)

## Overview

This design adds GPU-coalesced memory access patterns to the SYCL backend, ensuring adjacent threads access adjacent memory addresses for maximum memory bandwidth utilization on Intel Arc GPUs.

**Target:** >80% of theoretical memory bandwidth (~448 GB/s on Arc B50 Pro)

## 1. Profiling Phase & Workflow

### Baseline Profiling
Before implementing coalesced layouts, profile current SoA implementation:

```bash
# VTune GPU Hotspots with memory bandwidth analysis
vtune -collect gpu-hotspots -knob gpu-sampling-interval=1 \
  -result-dir vtune_coalescing_baseline -- \
  ./build/bin/llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -fa 1
```

### Metrics to Capture
- Memory bandwidth per kernel (GB/s achieved vs theoretical)
- L3 cache hit/miss rates
- Memory access pattern efficiency (% coalesced accesses)

### Decision Criteria
- Kernels with <80% bandwidth efficiency are candidates for coalescing
- Kernels already at >90% may not benefit (SoA layout sufficient)

## 2. Coalesced Memory Layout Strategy

### Current SoA Layout (Q4_0 example)
```
Row 0: [qs0..qs15][d0] [qs16..qs31][d1] ...  // 16 nibbles + scale per block
Row 1: [qs0..qs15][d0] [qs16..qs31][d1] ...
```

### Proposed Coalesced Layout
```
// Interleave across WARP_SIZE (32) threads for coalesced access
// Thread 0 accesses byte 0, 32, 64...
// Thread 1 accesses byte 1, 33, 65...
Block 0-31:  [qs_t0_b0][qs_t1_b0]...[qs_t31_b0][qs_t0_b1]...
Scales:      [d0][d1]...[d31] (grouped after all qs for block group)
```

### Key Principle
Adjacent threads in a warp access adjacent memory addresses, maximizing memory transaction efficiency.

### Reorder Kernel Pattern
```cpp
// Reorder from SoA to Coalesced
__kernel void reorder_soa_to_coalesced_q4_0(
    const int8_t* src_soa,
    int8_t* dst_coalesced,
    int num_blocks,
    sycl::nd_item<1> it)
{
    int tid = it.get_local_id(0);  // 0-31 within warp
    int block_group = it.get_group(0);

    // Each warp processes 32 blocks together
    int src_block = block_group * WARP_SIZE + tid;
    if (src_block >= num_blocks) return;

    // Read from SoA layout
    block_q4_0_soa src = load_block_soa(src_soa, src_block);

    // Write to coalesced positions (tid determines offset within group)
    store_coalesced(dst_coalesced, block_group, tid, src);
}
```

## 3. Kernel Modifications

### Dispatch Pattern
```cpp
// Helper function in optimized_feature or extra data structure
bool is_coalesced() const { return mode == reorder_mode::COALESCED; }

// Three-way dispatch in kernel entry points
void ggml_sycl_op_dmmv_q4_0(
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst,
    dpct::queue_ptr stream)
{
    auto* extra = (ggml_tensor_extra_gpu*)src0->extra;

    if (extra->optimized_feature.is_coalesced()) {
        dmmv_q4_0_coalesced(src0, src1, dst, stream);
    } else if (extra->optimized_feature.is_soa()) {
        dmmv_q4_0_soa(src0, src1, dst, stream);
    } else {
        dmmv_q4_0(src0, src1, dst, stream);  // AoS fallback
    }
}
```

### Kernel Implementation Pattern
```cpp
template<int BLOCK_SIZE>
void dmmv_q4_0_coalesced_kernel(
    const int8_t* __restrict__ x_coalesced,
    const float* __restrict__ y,
    float* __restrict__ dst,
    int ncols,
    sycl::nd_item<1> it)
{
    int tid = it.get_local_id(0);
    int row = it.get_group(0);

    float sum = 0.0f;

    // Coalesced access: thread tid reads bytes at offset tid, tid+32, tid+64...
    for (int block_group = 0; block_group < ncols / (QK4_0 * WARP_SIZE); block_group++) {
        // All 32 threads read 32 consecutive bytes (one memory transaction)
        int8_t qs = x_coalesced[coalesced_offset(row, block_group, tid)];
        float d = load_scale_coalesced(x_coalesced, row, block_group, tid);

        // Dequantize and accumulate
        sum += dequantize_q4_0(qs, d) * y[block_group * WARP_SIZE + tid];
    }

    // Warp reduction
    sum = warp_reduce_sum(sum, it);
    if (tid == 0) dst[row] = sum;
}
```

## 4. Error Handling & Fallback Strategy

### Graceful Degradation
```cpp
reorder_mode select_reorder_mode(const ggml_tensor* tensor, dpct::queue_ptr stream) {
    // Check hardware support
    if (!device_supports_coalesced_access(stream)) {
        return tensor->supports_soa() ? reorder_mode::SOA : reorder_mode::NONE;
    }

    // Check tensor alignment requirements
    if (tensor->ne[0] % WARP_SIZE != 0) {
        // Fall back to SoA for non-WARP-aligned dimensions
        return reorder_mode::SOA;
    }

    // Coalesced is viable
    return reorder_mode::COALESCED;
}
```

### Runtime Validation
- First-access sanity check: compare coalesced output vs SoA for first block
- If mismatch detected, log warning and fall back to SoA for that tensor
- Controlled via `GGML_SYCL_VALIDATE_COALESCED=1` (off by default)

### Cache Invalidation
- Reorder mode stored per-tensor in extra data
- Mode change invalidates cached reordered buffer
- Existing unified-cache eviction logic handles this

## 5. Testing Strategy

### Unit Tests
```cpp
// tests/test-coalesced-layout.cpp

// Layout verification
TEST(CoalescedLayout, Q4_0_LayoutCorrectness) {
    // Create reference AoS data, reorder to coalesced
    // Verify specific byte positions match expected coalesced layout
}

// Round-trip validation
TEST(CoalescedRoundtrip, Q4_0_Dequantize) {
    // AoS -> Coalesced -> dequantize -> compare with AoS -> dequantize
}
```

### Integration Tests
```bash
# A/B comparison script (scripts/test-coalesced.sh)
GGML_SYCL_REORDER_MODE=soa ./build/bin/llama-completion ... > soa_output.txt
GGML_SYCL_REORDER_MODE=coalesced ./build/bin/llama-completion ... > coalesced_output.txt
diff soa_output.txt coalesced_output.txt  # Must match exactly
```

### Performance Regression Tests
- Baseline benchmark captured before changes
- CI runs `llama-bench` with both modes
- Fails if coalesced is >5% slower than SoA
- VTune bandwidth metrics tracked per-kernel

### Format Coverage
Each format (Q4_0, Q4_K, Q6_K, Q8_0, MXFP4) gets dedicated tests on real model inference.

## 6. Implementation Rollout Plan

### Phase 1: Profiling Baseline (no code changes)
- Run VTune GPU Hotspots on current SoA implementation
- Identify kernels with <80% bandwidth efficiency
- Document memory access patterns per kernel

### Phase 2: Infrastructure
- Add `reorder_mode::COALESCED` to enum
- Add `is_coalesced()` helper function
- Add `GGML_SYCL_REORDER_MODE` environment variable
- Default remains `soa` until coalesced is validated

### Phase 3: Kernel Implementation (one format at a time)
1. Q4_0 DMMV -> test -> benchmark
2. Q4_0 MMQ -> test -> benchmark
3. Q4_0 MMVQ -> test -> benchmark
4. Repeat for Q8_0, Q6_K, Q4_K, MXFP4

### Phase 4: Validation & Default Switch
- All formats passing tests
- Performance regression tests green
- Switch default to `coalesced` if bandwidth improvement confirmed
- Keep `soa` as fallback option

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `GGML_SYCL_REORDER_MODE` | `none`, `soa`, `coalesced` | `soa` | Weight reorder strategy |
| `GGML_SYCL_VALIDATE_COALESCED` | `0`, `1` | `0` | Runtime output validation |

## Success Criteria

- Memory bandwidth utilization >80% on profiled kernels
- No correctness regressions (identical token output)
- Performance improvement on token generation (tg) benchmarks
- All quantization formats supported
