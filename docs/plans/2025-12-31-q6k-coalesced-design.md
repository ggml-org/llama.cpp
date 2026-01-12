# Q6_K Coalesced Memory Layout Design

**Date**: 2025-12-31
**Status**: Approved
**Author**: Claude + User

## Summary

Implement coalesced memory layout support for Q6_K tensors in the SYCL backend, matching the existing Q4_0 coalesced implementation pattern.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Full Q6_K coalescing (all 3 arrays) | Consistency with Q4_0 |
| Tile size | 16 blocks/tile | Match Q4_0, simpler code |
| Layout | Interleaved arrays per tile | Each array independently coalesced |
| Thread mapping | 32 threads/warp, more iterations | Match Q4_0 structure |
| Reorder approach | Extend existing reorder_q6_K() | Single code path, mode-aware |

## Memory Layout

### Q6_K Block Structure (210 bytes/block, 256 elements)
- `ql[128]` - lower 4-bit quants (128 bytes)
- `qh[64]` - upper 2-bit quants (64 bytes)
- `scales[16]` - 8-bit scales (16 bytes)
- `d` - fp16 super-block scale (2 bytes, stored separately)

### Coalesced Tile Layout (16 blocks/tile)

```
Per-tile quants (3328 bytes):
  [ql word-major: 2048 bytes]    // 16 blocks × 128 bytes, reordered
  [qh word-major: 1024 bytes]    // 16 blocks × 64 bytes, reordered
  [scales word-major: 256 bytes] // 16 blocks × 16 bytes, reordered

Tensor layout:
  [row 0 tiles][row 1 tiles]...[row N tiles]  // all quant data
  [d values: nrows × blocks_per_row × 2 bytes] // scales at tensor end
```

### Word-Major Reordering Pattern
- Source (block-major): `[B0.W0-Wn][B1.W0-Wn]...[B15.W0-Wn]`
- Dest (word-major): `[W0:B0-B15][W1:B0-B15]...[Wn:B0-B15]`

## Kernel Implementation

### Thread Mapping
```
32 threads per row (matching Q4_0):
- Threads 0-15:  blocks 0-15, process first half of each block's data
- Threads 16-31: blocks 0-15, process second half of each block's data
```

### Kernel Pseudocode
```cpp
static void mul_mat_vec_q6_k_coalesced(vx, vy, dst, ncols, nrows, nd_item) {
    const int lane_id = sg.get_local_linear_id();
    const int block_in_tile = lane_id % 16;  // 0-15
    const int is_upper_half = lane_id / 16;  // 0 or 1

    // Base pointers (coalesced layout)
    const uint8_t* x_ql = vx;
    const uint8_t* x_qh = x_ql + nrows * ql_row_stride;
    const int8_t* x_sc = x_qh + nrows * qh_row_stride;
    const ggml_half* x_d = ...;  // d values at tensor end

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Coalesced loads from each array (64-byte stride)

        // Process 8 sub-blocks (256 elements / 32 per Q8_1)
        for (int sub = 0; sub < 4; sub++) {  // 4 per half
            // Load ql, qh, combine to 6-bit values
            // dp4a with Y data
            // Accumulate partial sums
        }
    }

    // Warp reduction and write result
}
```

## CPU-side Reordering

Extend existing `reorder_q6_K()` in ggml-sycl.cpp:

```cpp
if (g_ggml_sycl_reorder_mode == reorder_mode::COALESCED) {
    constexpr int TILE_BLOCKS = 16;

    // Reorder ql (128 bytes/block, 32 words/block)
    // Reorder qh (64 bytes/block, 16 words/block)
    // Reorder scales (16 bytes/block, 4 words/block)
    // Each uses word-major pattern within tiles
} else {
    // Existing SoA reorder logic
}
```

## Multi-Path Support

| Path | When Used | Implementation |
|------|-----------|----------------|
| MMVQ | batch 1-8 | Add coalesced kernel (primary focus) |
| DMMV | batch 1 fallback | Fall back to AoS initially |
| MMQ | batch >8 | Fall back to AoS initially |

**Phase 1**: MMVQ coalesced + DMMV/MMQ AoS fallback
**Phase 2**: Add coalesced support to DMMV/MMQ if needed

## Files to Modify

1. **ggml-sycl.cpp**: Extend `reorder_q6_K()` for COALESCED mode
2. **mmvq.cpp**: Add `mul_mat_vec_q6_k_coalesced()` kernel and dispatch
3. **common.hpp**: Add `GGML_TYPE_Q6_K` to `is_coalesced_supported()`
4. **dmmv.cpp**: Add AoS fallback for coalesced tensors
5. **mmq.cpp**: Add AoS fallback for coalesced tensors

## Testing

Primary test (pure Q6_K model):
```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Secondary test (Q4_0 with Q6_K output layer):
```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`
