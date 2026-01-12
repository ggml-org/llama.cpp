# Q6_K Variable Tile Coalescing Design

**Date**: 2026-01-02
**Status**: Approved
**Author**: Claude + User

## Summary

Enable coalesced memory access for Q6_K tensors of any dimension using adaptive power-of-2 tile decomposition with multi-warp parallelism.

## Problem

Current fixed 32-block tile size fails for Mistral Q6_K:
- 4096-dim → 16 blocks (smaller than tile)
- 14336-dim → 56 blocks (not divisible by 32)
- 32000-dim → 125 blocks (not divisible by 32)

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Tile selection | Per-tensor at reorder time | Decided once, no runtime overhead |
| Remainder handling | Recursive power-of-2 decomposition | Any count supported, no padding waste |
| Tile structure | Encoded in memory layout | No metadata, kernel computes from block count |
| Thread mapping | Multiple warps, one tile each | Better bandwidth, more parallelism |
| Reduction | Shared memory | Single kernel, low overhead for 3-6 warps |

## Memory Layout

### Tile Decomposition (Binary)

Any block count decomposes into power-of-2 tiles (largest first):

| Blocks | Decomposition | Tiles |
|--------|--------------|-------|
| 16 | 16 | 1×16 |
| 56 | 32 + 16 + 8 | 1×32, 1×16, 1×8 |
| 125 | 64 + 32 + 16 + 8 + 4 + 1 | 2×32, 1×16, 1×8, 1×4, 1×1 |

### Per-Tile Layout (Q6_K)

Each tile stores data contiguously in word-major order:
```
[ql bytes: tile_size × 128]   // lower 4-bit quants
[qh bytes: tile_size × 64]    // upper 2-bit quants
[scales:   tile_size × 16]    // 8-bit scales
```

The `d` (fp16 scale) values remain at tensor end as in current SoA layout.

### Row Layout Example (56 blocks = 32+16+8)

```
Row memory (contiguous):
├── Tile 0 (32 blocks): [ql: 4096B][qh: 2048B][sc: 512B]
├── Tile 1 (16 blocks): [ql: 2048B][qh: 1024B][sc: 256B]
└── Tile 2 (8 blocks):  [ql: 1024B][qh: 512B] [sc: 128B]

Total row: 6656 + 3328 + 1664 = 11648 bytes
```

### Word-Major Ordering Within Tiles

Same as current coalesced layout - threads read consecutive bytes across blocks:
- Source (block-major): `[B0.W0-Wn][B1.W0-Wn]...[Bn.W0-Wn]`
- Dest (word-major): `[W0:B0-Bn][W1:B0-Bn]...[Wn:B0-Bn]`

## Kernel Implementation

### Launch Configuration

```cpp
// Compute tile count for this row
int num_tiles = popcount(blocks_per_row);  // Binary decomposition = set bits
int warps_per_row = num_tiles;
int threads_per_row = warps_per_row * WARP_SIZE;

// Launch: one work-group per row
sycl::range<3> grid(1, nrows, 1);
sycl::range<3> block(1, 1, threads_per_row);
```

### Thread Mapping

```cpp
const int warp_id = thread_id / WARP_SIZE;
const int lane_id = thread_id % WARP_SIZE;

// Find which tile this warp handles (walk decomposition)
int tile_idx = 0, block_offset = 0, tile_size = 0;
int remaining = blocks_per_row;
for (int ts = 32; ts >= 1; ts >>= 1) {
    while (remaining >= ts) {
        if (tile_idx == warp_id) {
            tile_size = ts;
            goto found;
        }
        block_offset += ts;
        remaining -= ts;
        tile_idx++;
    }
}
found:

// Only lanes < tile_size are active
if (lane_id >= tile_size) return;  // Early exit for inactive threads
```

### Reduction Phase

```cpp
// Warp-level reduction first
float warp_sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<>());

// Write to shared memory (one slot per warp)
if (lane_id == 0) shared_partials[warp_id] = warp_sum;
barrier();

// First warp reduces all partials
if (warp_id == 0 && lane_id < num_tiles) {
    float final = sycl::reduce_over_group(sg, shared_partials[lane_id], sycl::plus<>());
    if (lane_id == 0) dst[row] = final;
}
```

## CPU-Side Reordering

```cpp
void reorder_q6_K_coalesced_variable(
    const block_q6_K* src,
    uint8_t* dst,
    int64_t nrows,
    int64_t blocks_per_row
) {
    // Compute tile decomposition once
    std::vector<int> tile_sizes;
    int remaining = blocks_per_row;
    for (int ts = 32; ts >= 1; ts >>= 1) {
        while (remaining >= ts) {
            tile_sizes.push_back(ts);
            remaining -= ts;
        }
    }

    for (int64_t row = 0; row < nrows; row++) {
        int block_idx = 0;
        uint8_t* row_dst = dst + row * row_stride;

        for (int tile = 0; tile < tile_sizes.size(); tile++) {
            int tile_size = tile_sizes[tile];

            // Word-major reorder ql (128 bytes/block)
            for (int word = 0; word < 32; word++) {
                for (int b = 0; b < tile_size; b++) {
                    // Copy 4 bytes per word position
                    memcpy(row_dst, &src[...].ql[word * 4], 4);
                    row_dst += 4;
                }
            }
            // Similar for qh (16 words) and scales (4 words)

            block_idx += tile_size;
        }
    }
}
```

**Fallback**: Single-block rows fall back to SoA (no tiling benefit).

## Files to Modify

| File | Changes |
|------|---------|
| `common.hpp` | Add tile decomposition helpers: `count_tiles()`, `get_tile_size()`, `get_tile_offset()` |
| `convert.cpp` | Add `reorder_q6_K_coalesced_variable()` |
| `dmmv.cpp` | Add variable-tile Q6_K DMMV kernel |
| `mmvq.cpp` | Add variable-tile Q6_K MMVQ kernel |
| `ggml-sycl.cpp` | Wire up new reorder, update dispatch logic |

## Testing

### Correctness

```bash
# Pure Q6_K model (all layers use variable tiling)
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15,"
```

### Performance

```bash
# Compare vs SoA baseline
GGML_SYCL_LAYOUT_OVERRIDE=soa ./build/bin/llama-bench \
  -m mistral-7b-v0.1.Q6_K.gguf -p 512 -n 128 -ngl 99 -fa 1

GGML_SYCL_LAYOUT_OVERRIDE=coalesced ./build/bin/llama-bench \
  -m mistral-7b-v0.1.Q6_K.gguf -p 512 -n 128 -ngl 99 -fa 1
```

### Success Criteria

- Correct output on Mistral Q6_K (currently fails with fixed tiles)
- No performance regression vs SoA for well-aligned models
- Bandwidth improvement for decode (single-token) path

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Thread divergence for small tiles | Early exit for inactive lanes |
| Tile decomposition overhead | Minimal - just bit operations |
| Shared memory pressure | ~128 bytes/row, well within limits |
| Complex kernel logic | Thorough unit testing per tile size |
