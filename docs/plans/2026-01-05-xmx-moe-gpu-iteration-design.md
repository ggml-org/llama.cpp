# XMX MoE GPU-Side Expert Iteration Design

**Date**: 2026-01-05
**Status**: Design Complete
**Goal**: Make XMX MoE kernels compatible with SYCL command graphs by eliminating host synchronization

## Problem Statement

The current XMX MoE implementation achieves only ~24 t/s pp512 compared to ESIMD+graphs at ~671 t/s. The root cause is host-side expert iteration requiring:

1. `memcpy` expert counts/offsets to host (2 waits)
2. Host loop over active experts launching separate kernels
3. Each kernel launch requires host-device sync

This pattern is fundamentally incompatible with SYCL command graphs, which require the entire computation graph to be recordable without host intervention.

## Solution: Fused Single-Kernel with GPU Work Assignment

Replace host expert iteration with a single fused kernel where work-groups self-assign to experts by reading `expert_offsets` directly from device memory.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Fused XMX MoE Kernel                        │
│                                                             │
│  Input (device memory):                                     │
│  - sorted_tokens[total_tokens × hidden_dim]                 │
│  - expert_offsets[n_experts + 1]  ← GPU prefix sum          │
│  - expert_tile_offsets[n_experts + 1] ← GPU computed        │
│  - expert_weights[n_experts][out_dim × hidden_dim]          │
│                                                             │
│  Work-group Assignment:                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ wg_id → binary_search(expert_tile_offsets) → expert  │  │
│  │ local_tile = wg_id - expert_tile_offsets[expert]     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Per Work-group:                                            │
│  1. Determine expert_idx from wg_id via binary search       │
│  2. Load expert weights for that expert                     │
│  3. Process assigned token tiles via XMX                    │
│  4. Write results to correct output positions               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Work-group to Expert Mapping**: Binary search on `expert_tile_offsets` - O(log n_experts) ≈ 6 comparisons for 64 experts

2. **Tile Granularity**: Launch `total_tiles` work-groups where each expert contributes `ceil(tokens/tile_M)` tiles

3. **Weights Access**: `expert_weights + expert_idx * weight_stride` computed on GPU

4. **Event Chaining**: All operations connected via SYCL events, no `.wait()` calls

## Implementation Tasks

### Task 1: GPU Tile Mapping Function

Create function to compute expert tile offsets and total tile count on GPU:

```cpp
// File: moe-sort.hpp
sycl::event moe_compute_tile_mapping(
    const int32_t* expert_counts,     // [n_experts] - input
    int32_t* expert_tile_offsets,     // [n_experts + 1] - output
    int32_t* total_tiles,             // [1] - output scalar
    int64_t n_experts,
    int64_t tile_M,                   // typically 32
    sycl::queue& q,
    sycl::event dep);
```

Implementation: Sequential GPU kernel (single work-item) computing prefix sum of `ceil(count/tile_M)`.

### Task 2: Work-group Assignment Device Function

```cpp
// File: moe-xmx.hpp
// Device function: binary search to find expert from work-group ID
inline int find_expert_for_workgroup(
    int wg_id,
    const int32_t* expert_tile_offsets,
    int n_experts) {
    int lo = 0, hi = n_experts;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (expert_tile_offsets[mid + 1] <= wg_id) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
```

### Task 3: Fused XMX Kernel

Refactor `launch_xmx_moe_gemm_q8_0` to accept tile offsets and perform GPU-side expert assignment:

```cpp
template <typename AccT>
sycl::event launch_fused_xmx_moe(
    sycl::queue& q,
    sycl::event dep,
    const sycl::half* sorted_tokens,    // [total_tokens × hidden_dim]
    const int32_t* expert_offsets,      // [n_experts + 1] token offsets
    const int32_t* expert_tile_offsets, // [n_experts + 1] tile offsets
    int32_t total_tiles,                // total work-groups to launch
    const void* expert_weights,         // all expert weights contiguous
    sycl::half* output,                 // [total_tokens × out_dim]
    int64_t hidden_dim,
    int64_t out_dim,
    int64_t n_experts,
    int64_t weight_stride);             // bytes between expert weights
```

Each work-group:
1. Computes `expert_idx = find_expert_for_workgroup(wg_id, expert_tile_offsets, n_experts)`
2. Computes `local_tile = wg_id - expert_tile_offsets[expert_idx]`
3. Gets token range from `expert_offsets[expert_idx]`
4. Processes tile using existing XMX logic

### Task 4: Update Dispatch in ggml-sycl.cpp

Replace host iteration pattern:

```cpp
// BEFORE (host iteration - breaks graphs):
stream->memcpy(h_counts.data(), expert_counts, ...).wait();
stream->memcpy(h_offsets.data(), expert_offsets, ...).wait();
for (int e = 0; e < n_experts; e++) {
    if (h_counts[e] > 0) {
        launch_xmx_for_expert(e, h_offsets[e], h_counts[e], ...);
    }
}

// AFTER (single GPU launch - graph compatible):
auto tile_event = moe_compute_tile_mapping(
    expert_counts, expert_tile_offsets, &total_tiles,
    n_experts, TILE_M, *stream, sort_event);

auto gemm_event = launch_fused_xmx_moe(
    *stream, tile_event,
    sorted_tokens, expert_offsets, expert_tile_offsets, total_tiles,
    expert_weights, output, hidden_dim, out_dim, n_experts, weight_stride);
```

### Task 5: Remove Host Sync Points

Delete from `try_xmx_sorted_moe`:
- `std::vector<int32_t> h_counts, h_offsets` host allocations
- All `stream->memcpy(...).wait()` patterns
- Replace synchronous `moe_compute_expert_offsets` with `moe_compute_expert_offsets_gpu`

### Task 6: Allocate Tile Mapping Buffers

Add to buffer management:
```cpp
int32_t* expert_tile_offsets;  // [n_experts + 1]
int32_t* total_tiles;          // [1] scalar
```

Pre-allocate for max experts (64) to enable graph recording.

### Task 7: Testing and Benchmarking

Test command (graphs enabled by default):
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
```

Benchmark:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -p 512 -n 128 -ngl 99 -fa 1
```

Target: Match ESIMD+graphs performance (~671 t/s pp512)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Binary search overhead | Low | Low | 6 comparisons negligible vs XMX compute |
| Load imbalance | Medium | Low | XMX already handles variable tiles |
| Graph recording with variable tiles | Medium | Medium | Pre-allocate max, use actual at runtime |
| Debugging complexity | Medium | Medium | Add debug flag for single-expert fallback |

## Success Criteria

1. XMX MoE works with `GGML_SYCL_DISABLE_GRAPH=0` (default)
2. Zero `.wait()` calls in hot path
3. Performance within 10% of ESIMD+graphs baseline
4. Correct output verified against reference model

## Files to Modify

- `ggml/src/ggml-sycl/moe-sort.hpp` - Add tile mapping function
- `ggml/src/ggml-sycl/moe-xmx.hpp` - Refactor to fused kernel
- `ggml/src/ggml-sycl/ggml-sycl.cpp` - Update dispatch pattern (lines ~11450-11900)
