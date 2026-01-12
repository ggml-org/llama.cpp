# Q6_K Variable Tile Coalescing - All Kernel Paths

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable coalesced memory access for Q6_K tensors of any dimension across ALL kernel paths (MMVQ, MMQ, DMMV, GET_ROWS).

**Architecture:** Binary decomposition splits any block count into power-of-2 tiles (32, 16, 8, 4, 2, 1). Each tile is processed by one warp. All kernels use `tile_count()`, `tile_size_at()`, `tile_offset_at()` helpers.

**Tech Stack:** SYCL, C++17, Intel oneAPI

**Prerequisites:** Tasks 1-4 from the original plan are complete (helpers, reorder, MMVQ kernel, MMVQ dispatch).

---

## Task 5: Variable Tile MMQ Loader

**Files:**
- Modify: `ggml/src/ggml-sycl/mmq.cpp:1695-1809`

**Step 1: Read the existing loader**

Read `mmq.cpp` lines 1695-1809 to understand `load_tiles_q6_K_coalesced`.

**Step 2: Update tile computation**

Replace the fixed tile calculation at line ~1717:

```cpp
// OLD (broken):
constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 32
const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

// NEW (variable):
const int num_tiles = tile_count(blocks_per_row);
```

**Step 3: Update row stride calculation**

Replace fixed stride with variable computation:

```cpp
// OLD:
const size_t tile_ql_size = TILE_BLOCKS * 128;
const size_t tile_qh_size = TILE_BLOCKS * 64;
const size_t tile_sc_size = TILE_BLOCKS * 16;
const size_t tile_total = tile_ql_size + tile_qh_size + tile_sc_size;
const size_t row_stride = tiles_per_row * tile_total;

// NEW:
size_t row_stride = 0;
for (int t = 0; t < num_tiles; t++) {
    int ts = tile_size_at(blocks_per_row, t);
    row_stride += ts * (128 + 64 + 16);
}
```

**Step 4: Update tile/block indexing**

Replace fixed tile indexing with variable computation:

```cpp
// OLD:
const int tile = block_in_row / TILE_BLOCKS;
const int block_in_tile = block_in_row % TILE_BLOCKS;
const int tile_size = TILE_BLOCKS;

// NEW: Find which tile this block belongs to
int tile = 0, tile_offset = 0, tile_size = 0;
int remaining = blocks_per_row;
int acc = 0;
for (int t = 0; remaining > 0; t++) {
    int ts = tile_size_at(blocks_per_row, t);
    if (block_in_row < acc + ts) {
        tile = t;
        tile_size = ts;
        tile_offset = acc;
        break;
    }
    acc += ts;
    remaining -= ts;
}
const int block_in_tile = block_in_row - tile_offset;
```

**Step 5: Update data offset calculation**

Update the offset to reach the correct tile:

```cpp
// Compute byte offset to this tile
size_t tile_byte_offset = 0;
for (int t = 0; t < tile; t++) {
    int ts = tile_size_at(blocks_per_row, t);
    tile_byte_offset += ts * (128 + 64 + 16);
}

const uint8_t * tile_base = x_base + row * row_stride + tile_byte_offset;
const uint8_t * tile_ql = tile_base;
const uint8_t * tile_qh = tile_ql + tile_size * 128;
const int8_t * tile_sc = (const int8_t *)(tile_qh + tile_size * 64);
```

**Step 6: Update word-major access pattern**

Replace fixed TILE_BLOCKS with tile_size in access calculations:

```cpp
// Word-major layout: stride is tile_size * 4 bytes
const int ql_offset = word_idx * tile_size * 4 + block_in_tile * 4;
const int qh_offset = word_idx * tile_size * 4 + block_in_tile * 4;
const int sc_offset = word_idx * tile_size * 4 + block_in_tile * 4;
```

**Step 7: Rebuild and verify compilation**

```bash
./scripts/quick-rebuild.sh mmq.cpp
```
Expected: Build succeeds

---

## Task 6: MMQ Dispatch row_quants_bytes

**Files:**
- Modify: `ggml/src/ggml-sycl/mmq.cpp:5607-5640`

**Step 1: Read current dispatch**

Read the Q6_K dispatch section around line 5607.

**Step 2: Update row_quants_bytes calculation**

Replace fixed calculation with variable:

```cpp
// OLD:
const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
const size_t row_quants_bytes = tiles_per_row * MMVQ_COALESCED_TILE_BLOCKS * (128 + 64 + 16);

// NEW:
const int num_tiles = tile_count(blocks_per_row);
size_t row_quants_bytes = 0;
for (int t = 0; t < num_tiles; t++) {
    int ts = tile_size_at(blocks_per_row, t);
    row_quants_bytes += ts * (128 + 64 + 16);
}
```

**Step 3: Rebuild**

```bash
./scripts/quick-rebuild.sh mmq.cpp
```

**Step 4: Quick test**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 5 --seed 42 --temp 0
```
Expected: Correct output (may still fail if DMMV/GET_ROWS are hit first)

---

## Task 7: Variable Tile DMMV Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/dmmv.cpp:1455-1646`

**Step 1: Read existing kernel**

Read `dmmv.cpp` lines 1455-1646 to understand `dequantize_mul_mat_vec_q6_k_coalesced`.

**Step 2: Create variable tile version**

Add new kernel after the existing one (around line 1647):

```cpp
// Q6_K variable tile coalesced DMMV kernel
static void dequantize_mul_mat_vec_q6_k_coalesced_variable(
    const void * __restrict__ vx,
    const float * __restrict__ y,
    float * __restrict__ dst,
    const int ncols,
    const int nrows,
    const int nrows_full,
    const int row_low,
    const sycl::nd_item<3> & item)
{
    const int blocks_per_row = ncols / QK_K;
    const int num_tiles = tile_count(blocks_per_row);

    const int row = item.get_group(2);
    if (row >= nrows) return;

    const int warp_id = item.get_local_id(2) / WARP_SIZE;
    const int lane_id = item.get_local_id(2) % WARP_SIZE;

    // Early exit for warps beyond tile count
    if (warp_id >= num_tiles) {
        // Still participate in reduction with zero
        // ... (barrier and reduction code)
        return;
    }

    // Get tile info
    const int tile_size = tile_size_at(blocks_per_row, warp_id);
    const int tile_block_offset = tile_offset_at(blocks_per_row, warp_id);

    // Compute row stride
    size_t row_stride = 0;
    for (int t = 0; t < num_tiles; t++) {
        row_stride += tile_size_at(blocks_per_row, t) * (128 + 64 + 16);
    }

    // Compute tile offset within row
    size_t tile_byte_offset = 0;
    for (int t = 0; t < warp_id; t++) {
        tile_byte_offset += tile_size_at(blocks_per_row, t) * (128 + 64 + 16);
    }

    const uint8_t * x_base = (const uint8_t *)vx;
    const int global_row = row_low + row;
    const uint8_t * tile_base = x_base + global_row * row_stride + tile_byte_offset;

    // D values at end of tensor
    const ggml_half * x_d = (const ggml_half *)(x_base + nrows_full * row_stride);

    float partial_sum = 0.0f;

    if (lane_id < tile_size) {
        const int block_idx = global_row * blocks_per_row + tile_block_offset + lane_id;
        const float d = x_d[block_idx];

        const uint8_t * tile_ql = tile_base;
        const uint8_t * tile_qh = tile_ql + tile_size * 128;
        const int8_t * tile_sc = (const int8_t *)(tile_qh + tile_size * 64);

        // Process Q6_K block with word-major access
        // ... (same math as MMVQ variable tile kernel)

        for (int j = 0; j < QK_K; j += 128) {
            // Dequantize and multiply with y
            // Access pattern: word * tile_size * 4 + lane_id * 4
            // ... (detailed implementation)
        }
    }

    // Warp reduction + cross-warp reduction via shared memory
    // ... (same pattern as MMVQ)
}
```

**Step 3: Add dispatch function**

```cpp
static void dequantize_mul_mat_vec_q6_k_coalesced_variable_sycl(
    const void * vx, const float * y, float * dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    dpct::queue_ptr stream)
{
    const int blocks_per_row = ncols / QK_K;
    const int num_tiles = tile_count(blocks_per_row);
    const int threads_per_row = num_tiles * WARP_SIZE;

    sycl::range<3> grid(1, 1, nrows);
    sycl::range<3> block(1, 1, threads_per_row);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(num_tiles), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q6_k_coalesced_variable(
                    vx, y, dst, ncols, nrows, nrows_full, row_low, item);
            });
    });
}
```

**Step 4: Rebuild**

```bash
./scripts/quick-rebuild.sh dmmv.cpp
```

---

## Task 8: DMMV Dispatch Wiring

**Files:**
- Modify: `ggml/src/ggml-sycl/dmmv.cpp:3034-3080`

**Step 1: Update Q6_K coalesced dispatch**

In the Q6_K case around line 3034, call the variable tile version:

```cpp
case GGML_TYPE_Q6_K:
{
    ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)src0->extra;
    reorder_mode mode = extra ? extra->optimized_feature.get_reorder() : reorder_mode::NONE;

    if (mode == reorder_mode::COALESCED) {
        // Use variable tile kernel
        dequantize_mul_mat_vec_q6_k_coalesced_variable_sycl(
            src0_dd_i, src1_ddf_i, dst_dd_i,
            ne00, row_diff, storage_ne01, global_row_low, stream);
    } else if (mode == reorder_mode::SOA) {
        // ... existing SoA path
    } else {
        // ... existing AoS path
    }
    break;
}
```

**Step 2: Rebuild and test**

```bash
./scripts/quick-rebuild.sh dmmv.cpp
```

---

## Task 9: Variable Tile GET_ROWS Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/getrows.cpp:328-486`

**Step 1: Read existing kernel**

Read `getrows.cpp` lines 328-486 to understand `k_get_rows_q6_k_coalesced`.

**Step 2: Update tile computation in kernel**

Modify the kernel to use variable tiles:

```cpp
template <typename dst_t>
static void k_get_rows_q6_k_coalesced_variable(
    const void * __restrict__ x,
    const int32_t * __restrict__ indices,
    dst_t * __restrict__ dst,
    const int64_t blocks_per_row,
    const int64_t row_quants_bytes,
    const int64_t total_nrows,
    const sycl::nd_item<3> & item)
{
    const int num_tiles = tile_count(blocks_per_row);

    const int row_idx = item.get_group(2);      // Which row (from indices)
    const int block_id = item.get_group(1);     // Which block in the row
    const int tid = item.get_local_id(2);       // Thread within work-group

    const int src_row = indices[row_idx];

    // Find which tile this block belongs to
    int tile_idx = 0, tile_offset = 0, tile_size = 0;
    int acc = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        if (block_id < acc + ts) {
            tile_idx = t;
            tile_size = ts;
            tile_offset = acc;
            break;
        }
        acc += ts;
    }
    const int block_in_tile = block_id - tile_offset;

    // Compute byte offset to tile
    size_t tile_byte_offset = 0;
    for (int t = 0; t < tile_idx; t++) {
        tile_byte_offset += tile_size_at(blocks_per_row, t) * (128 + 64 + 16);
    }

    const uint8_t * x_base = (const uint8_t *)x;
    const uint8_t * tile_base = x_base + src_row * row_quants_bytes + tile_byte_offset;

    const uint8_t * tile_ql = tile_base;
    const uint8_t * tile_qh = tile_ql + tile_size * 128;
    const int8_t * tile_sc = (const int8_t *)(tile_qh + tile_size * 64);

    // D values at end
    const ggml_half * x_d = (const ggml_half *)(x_base + total_nrows * row_quants_bytes);
    const float d = x_d[src_row * blocks_per_row + block_id];

    // Dequantize with word-major access pattern
    // Each thread handles 4 values, 64 threads per block = 256 values = QK_K
    // Access: word * tile_size * 4 + block_in_tile * 4
    // ... (dequantize logic)

    // Write to output
    dst_t * out = dst + row_idx * blocks_per_row * QK_K + block_id * QK_K;
    // ... (write dequantized values)
}
```

**Step 3: Add dispatch function**

```cpp
template <typename dst_t>
static void get_rows_q6_k_coalesced_variable_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const int32_t * src1_dd, dst_t * dst_dd,
    dpct::queue_ptr stream)
{
    const int64_t blocks_per_row = src0->ne[0] / QK_K;
    const int64_t nrows = src1->ne[0];
    const int64_t total_nrows = src0->ne[1];

    // Compute variable row_quants_bytes
    const int num_tiles = tile_count(blocks_per_row);
    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        row_quants_bytes += tile_size_at(blocks_per_row, t) * (128 + 64 + 16);
    }

    sycl::range<3> grid(1, blocks_per_row, nrows);
    sycl::range<3> block(1, 1, 64);  // 64 threads per Q6_K block

    stream->parallel_for(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item) {
            k_get_rows_q6_k_coalesced_variable<dst_t>(
                src0_dd, src1_dd, dst_dd,
                blocks_per_row, row_quants_bytes, total_nrows, item);
        });
}
```

**Step 4: Rebuild**

```bash
./scripts/quick-rebuild.sh getrows.cpp
```

---

## Task 10: GET_ROWS Dispatch Wiring

**Files:**
- Modify: `ggml/src/ggml-sycl/getrows.cpp:1053-1080`

**Step 1: Update dispatch**

In the Q6_K coalesced case, call the variable tile version:

```cpp
case GGML_TYPE_Q6_K:
{
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->src[0]->extra;
    reorder_mode mode = extra ? extra->optimized_feature.get_reorder() : reorder_mode::NONE;

    if (mode == reorder_mode::COALESCED) {
        // Variable tile row_quants_bytes
        const int64_t blocks_per_row = dst->src[0]->ne[0] / QK_K;
        const int num_tiles = tile_count(blocks_per_row);
        size_t row_quants_bytes = 0;
        for (int t = 0; t < num_tiles; t++) {
            row_quants_bytes += tile_size_at(blocks_per_row, t) * (128 + 64 + 16);
        }

        get_rows_q6_k_coalesced_variable_sycl(ctx, dst->src[0], dst->src[1], dst,
            src0_d, src1_d, dst_d, row_quants_bytes, stream);
    } else if (mode == reorder_mode::SOA) {
        get_rows_q6_k_soa_sycl(ctx, dst->src[0], dst->src[1], dst,
            src0_d, src1_d, dst_d, stream);
    } else {
        get_rows_q6_k_aos_sycl(ctx, dst->src[0], dst->src[1], dst,
            src0_d, src1_d, dst_d, stream);
    }
    break;
}
```

**Step 2: Rebuild**

```bash
./scripts/quick-rebuild.sh getrows.cpp
```

---

## Task 11: Integration Testing

**Step 1: Full correctness test**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected output: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15,`

**Step 2: Compare with SoA baseline**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=soa ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected: Same output as coalesced

**Step 3: Verify Q4_0 still works (no regression)**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Step 4: Benchmark**

```bash
# SoA baseline
GGML_SYCL_LAYOUT_OVERRIDE=soa ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -p 512 -n 128 -ngl 99 -fa 1

# Variable tile coalesced
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/mmq.cpp ggml/src/ggml-sycl/dmmv.cpp ggml/src/ggml-sycl/getrows.cpp
git commit -m "feat(sycl): add Q6_K variable tile support to MMQ, DMMV, GET_ROWS

Complete variable tile coalescing for Q6_K quantization across all
kernel paths. This enables coalesced memory access for any tensor
dimension using power-of-2 tile decomposition.

Kernels updated:
- MMQ: load_tiles_q6_K_coalesced now uses variable tiles
- DMMV: dequantize_mul_mat_vec_q6_k_coalesced_variable kernel
- GET_ROWS: k_get_rows_q6_k_coalesced_variable kernel

All kernels use tile_count(), tile_size_at(), tile_offset_at()
helpers for consistent tile decomposition.

Tested with Mistral 7B Q6_K (56 blocks = 32+16+8 tiles).
"
```

---

## Summary

| Task | Description | File | Status |
|------|-------------|------|--------|
| 1-4 | Infrastructure + MMVQ | common.hpp, convert.cpp, mmvq.cpp | ✅ Done |
| 5 | MMQ variable tile loader | mmq.cpp | TODO |
| 6 | MMQ dispatch | mmq.cpp | TODO |
| 7 | DMMV variable tile kernel | dmmv.cpp | TODO |
| 8 | DMMV dispatch | dmmv.cpp | TODO |
| 9 | GET_ROWS variable tile kernel | getrows.cpp | TODO |
| 10 | GET_ROWS dispatch | getrows.cpp | TODO |
| 11 | Integration testing | Runtime | TODO |

**Total remaining: 7 tasks**
