# Q6_K Variable Tile Coalescing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable coalesced memory access for Q6_K tensors of any dimension using power-of-2 tile decomposition.

**Architecture:** Binary decomposition splits any block count into power-of-2 tiles (32, 16, 8, 4, 2, 1). Each tile is processed by one warp. Shared memory reduction combines partial sums.

**Tech Stack:** SYCL, C++17, Intel oneAPI

---

## Task 1: Add Tile Decomposition Helpers

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp:251-260`
- Test: `tests/test-tile-decomposition.cpp` (create)

**Step 1: Write the test file**

Create `tests/test-tile-decomposition.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include <vector>

// Inline the helpers for testing (will be in common.hpp)
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        count++;
        blocks -= tile_size;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    int idx = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        if (idx == tile_idx) return tile_size;
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

inline int tile_offset_at(int blocks, int tile_idx) {
    int idx = 0, offset = 0;
    while (blocks > 0 && idx < tile_idx) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        offset += tile_size;
        blocks -= tile_size;
        idx++;
    }
    return offset;
}

int main() {
    // Test 16 blocks: single tile of 16
    assert(tile_count(16) == 1);
    assert(tile_size_at(16, 0) == 16);
    assert(tile_offset_at(16, 0) == 0);
    printf("PASS: 16 blocks = 1x16\n");

    // Test 56 blocks: 32 + 16 + 8
    assert(tile_count(56) == 3);
    assert(tile_size_at(56, 0) == 32);
    assert(tile_size_at(56, 1) == 16);
    assert(tile_size_at(56, 2) == 8);
    assert(tile_offset_at(56, 0) == 0);
    assert(tile_offset_at(56, 1) == 32);
    assert(tile_offset_at(56, 2) == 48);
    printf("PASS: 56 blocks = 32+16+8\n");

    // Test 125 blocks: 64+32+16+8+4+1 = 2x32+16+8+4+1
    assert(tile_count(125) == 6);
    assert(tile_size_at(125, 0) == 32);
    assert(tile_size_at(125, 1) == 32);
    assert(tile_size_at(125, 2) == 32);
    assert(tile_size_at(125, 3) == 16);
    assert(tile_size_at(125, 4) == 8);
    assert(tile_size_at(125, 5) == 5);  // Remainder
    printf("PASS: 125 blocks decomposition\n");

    // Test 32 blocks: single full tile
    assert(tile_count(32) == 1);
    assert(tile_size_at(32, 0) == 32);
    printf("PASS: 32 blocks = 1x32\n");

    printf("\nAll tile decomposition tests passed!\n");
    return 0;
}
```

**Step 2: Compile and run the test**

```bash
g++ -std=c++17 -o build/bin/test-tile-decomposition tests/test-tile-decomposition.cpp
./build/bin/test-tile-decomposition
```
Expected: All tests pass

**Step 3: Add helpers to common.hpp**

Add after line 256 in `ggml/src/ggml-sycl/common.hpp`:

```cpp
// Variable tile decomposition helpers (power-of-2, largest first, max 32)
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        count++;
        blocks -= tile_size;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    int idx = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        if (idx == tile_idx) return tile_size;
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

inline int tile_offset_at(int blocks, int tile_idx) {
    int idx = 0, offset = 0;
    while (blocks > 0 && idx < tile_idx) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        offset += tile_size;
        blocks -= tile_size;
        idx++;
    }
    return offset;
}
```

**Step 4: Rebuild to verify compilation**

```bash
./scripts/quick-rebuild.sh
```
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp tests/test-tile-decomposition.cpp
git commit -m "feat(sycl): add tile decomposition helpers for variable Q6_K coalescing"
```

---

## Task 2: Add Variable Tile Reorder Function

**Files:**
- Modify: `ggml/src/ggml-sycl/convert.cpp:823-900`
- Test: `tests/test-q6k-variable-reorder.cpp` (create)

**Step 1: Write reorder test**

Create `tests/test-q6k-variable-reorder.cpp`:

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>

// Simplified Q6_K block for testing
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

// Helper from common.hpp (inline for test)
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        count++;
        blocks -= tile_size;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    int idx = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        if (idx == tile_idx) return tile_size;
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

void reorder_q6_K_variable_tile(
    const block_q6_K* src,
    uint8_t* dst,
    int nrows,
    int blocks_per_row
) {
    const int num_tiles = tile_count(blocks_per_row);

    // Compute row stride (sum of all tile sizes)
    int row_stride = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_stride += ts * (128 + 64 + 16);  // ql + qh + scales per block
    }

    for (int row = 0; row < nrows; row++) {
        uint8_t* row_dst = dst + row * row_stride;
        int block_idx = 0;

        for (int tile = 0; tile < num_tiles; tile++) {
            int tile_size = tile_size_at(blocks_per_row, tile);

            // Reorder ql: word-major (32 words of 4 bytes each per block)
            for (int word = 0; word < 32; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K& blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.ql[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder qh: word-major (16 words of 4 bytes each per block)
            for (int word = 0; word < 16; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K& blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.qh[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder scales: word-major (4 words of 4 bytes each per block)
            for (int word = 0; word < 4; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K& blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.scales[word * 4], 4);
                    row_dst += 4;
                }
            }

            block_idx += tile_size;
        }
    }
}

int main() {
    // Test with 56 blocks (Mistral FFN dimension / QK_K)
    const int blocks_per_row = 56;
    const int nrows = 2;

    std::vector<block_q6_K> src(nrows * blocks_per_row);

    // Fill with identifiable pattern
    for (int r = 0; r < nrows; r++) {
        for (int b = 0; b < blocks_per_row; b++) {
            block_q6_K& blk = src[r * blocks_per_row + b];
            for (int i = 0; i < 128; i++) blk.ql[i] = (r * 100 + b) & 0xFF;
            for (int i = 0; i < 64; i++) blk.qh[i] = (r * 100 + b + 1) & 0xFF;
            for (int i = 0; i < 16; i++) blk.scales[i] = (r * 100 + b + 2) & 0x7F;
            blk.d = r * 100 + b;
        }
    }

    // Compute expected size
    int num_tiles = tile_count(blocks_per_row);
    int row_stride = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_stride += ts * (128 + 64 + 16);
    }

    std::vector<uint8_t> dst(nrows * row_stride);
    reorder_q6_K_variable_tile(src.data(), dst.data(), nrows, blocks_per_row);

    // Verify tile structure: 56 = 32 + 16 + 8
    assert(num_tiles == 3);
    printf("PASS: 56 blocks -> %d tiles\n", num_tiles);

    // Verify first word of first tile contains block 0 data
    assert(dst[0] == (0 & 0xFF));  // First byte of row 0, block 0
    printf("PASS: First byte correct\n");

    printf("\nVariable tile reorder test passed!\n");
    return 0;
}
```

**Step 2: Compile and run test**

```bash
g++ -std=c++17 -o build/bin/test-q6k-variable-reorder tests/test-q6k-variable-reorder.cpp
./build/bin/test-q6k-variable-reorder
```
Expected: All tests pass

**Step 3: Add reorder function to convert.cpp**

Add after existing Q6_K reorder (around line 900) in `ggml/src/ggml-sycl/convert.cpp`:

```cpp
// Q6_K variable tile reorder for arbitrary block counts
void reorder_q6_K_variable_tile(
    const block_q6_K * src,
    uint8_t * dst,
    int64_t nrows,
    int64_t blocks_per_row,
    int64_t row_stride
) {
    const int num_tiles = tile_count(blocks_per_row);

    for (int64_t row = 0; row < nrows; row++) {
        uint8_t * row_dst = dst + row * row_stride;
        int block_idx = 0;

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_size = tile_size_at(blocks_per_row, tile);

            // Reorder ql: word-major (32 words of 4 bytes each per block)
            for (int word = 0; word < 32; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.ql[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder qh: word-major (16 words of 4 bytes each per block)
            for (int word = 0; word < 16; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.qh[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder scales: word-major (4 words of 4 bytes each per block)
            for (int word = 0; word < 4; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.scales[word * 4], 4);
                    row_dst += 4;
                }
            }

            block_idx += tile_size;
        }
    }
}
```

**Step 4: Rebuild**

```bash
./scripts/quick-rebuild.sh
```
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/convert.cpp tests/test-q6k-variable-reorder.cpp
git commit -m "feat(sycl): add Q6_K variable tile reorder function"
```

---

## Task 3: Add Variable Tile MMVQ Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:264-400`
- Test: Integration test via llama-completion

**Step 1: Add variable tile kernel**

Add after existing `mul_mat_vec_q6_k_coalesced` (around line 400) in `ggml/src/ggml-sycl/mmvq.cpp`:

```cpp
// Q6_K variable tile kernel - handles any block count via power-of-2 decomposition
// Each warp processes one tile, shared memory reduction for final sum
static void mul_mat_vec_q6_k_variable_tile(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int blocks_per_row,
    float * __restrict__ shared_partials,
    const sycl::nd_item<3> & nd_item)
{
    const auto sg = nd_item.get_sub_group();
    const int warp_id = nd_item.get_local_id(2) / WARP_SIZE;
    const int lane_id = sg.get_local_linear_id();
    const int row = nd_item.get_group(2);

    if (row >= nrows) return;

    const int num_tiles = tile_count(blocks_per_row);

    // Early exit for warps beyond tile count
    if (warp_id >= num_tiles) return;

    // Get this warp's tile info
    const int tile_size = tile_size_at(blocks_per_row, warp_id);
    const int tile_block_offset = tile_offset_at(blocks_per_row, warp_id);

    // Only lanes < tile_size are active
    if (lane_id >= tile_size) {
        // Still need to participate in reduction
        float zero = 0.0f;
        float warp_sum = sycl::reduce_over_group(sg, zero, sycl::plus<float>());
        if (lane_id == 0) {
            shared_partials[warp_id] = warp_sum;
        }
        sycl::group_barrier(nd_item.get_group());

        // First warp does final reduction
        if (warp_id == 0 && lane_id < num_tiles) {
            float val = shared_partials[lane_id];
            float final_sum = sycl::reduce_over_group(sg, val, sycl::plus<float>());
            if (lane_id == 0) {
                dst[row] = final_sum;
            }
        }
        return;
    }

    // Compute row stride (variable due to different tile sizes)
    int row_stride = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_stride += ts * (128 + 64 + 16);  // ql + qh + scales bytes per block
    }

    // Compute offset to this tile within the row
    int tile_offset = 0;
    for (int t = 0; t < warp_id; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        tile_offset += ts * (128 + 64 + 16);
    }

    const uint8_t * x_base = (const uint8_t *)vx;
    const uint8_t * tile_base = x_base + row * row_stride + tile_offset;

    // Tile layout: [ql: tile_size * 128][qh: tile_size * 64][scales: tile_size * 16]
    const uint8_t * tile_ql = tile_base;
    const uint8_t * tile_qh = tile_ql + tile_size * 128;
    const int8_t * tile_sc = (const int8_t *)(tile_qh + tile_size * 64);

    // D values at end of tensor
    const ggml_half * x_d = (const ggml_half *)(x_base + nrows * row_stride);
    const int block_idx = row * blocks_per_row + tile_block_offset + lane_id;
    const float d = x_d[block_idx];

    // Y pointers
    const int8_t * y_qs = (const int8_t *)vy;
    const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)vy + ncols);

    float partial_sum = 0.0f;

    // Process this block (lane_id corresponds to block within tile)
    const int y_block_base = (tile_block_offset + lane_id) * 8;

    constexpr int WORD_PLANE_STRIDE_QL = 4;  // 4 bytes per word position per block
    constexpr int WORD_PLANE_STRIDE_QH = 4;
    constexpr int WORD_PLANE_STRIDE_SC = 4;

    // Simplified Q6_K dot product (same math as fixed tile version)
    for (int iqs = 0; iqs < QI6_K; ++iqs) {
        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int u_index = iqs % QI8_1;

        // Word-major layout: words are interleaved across blocks
        // For tile_size blocks: [word0 for all blocks][word1 for all blocks]...
        const int ql_offset = iqs * tile_size * 4 + lane_id * 4;
        const int qh_word_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
        const int qh_offset = qh_word_idx * tile_size * 4 + lane_id * 4;

        const int vl = *((const int *)(tile_ql + ql_offset));
        const int vh_raw = *((const int *)(tile_qh + qh_offset));
        const int vh = vh_raw >> vh_shift;

        const int sc_idx0 = scale_offset;
        const int sc_idx1 = scale_offset + 4;
        const int sc_word0 = sc_idx0 / 4;
        const int sc_byte0 = sc_idx0 % 4;
        const int sc_word1 = sc_idx1 / 4;
        const int sc_byte1 = sc_idx1 % 4;

        const int sc_offset0 = sc_word0 * tile_size * 4 + lane_id * 4 + sc_byte0;
        const int sc_offset1 = sc_word1 * tile_size * 4 + lane_id * 4 + sc_byte1;
        const int8_t sc0 = tile_sc[sc_offset0];
        const int8_t sc1 = tile_sc[sc_offset1];

        // Unpack and combine 6-bit values
        const int v0 = (((vl >> 0) & 0x0F0F0F0F) | (((vh >> 0) & 0x03030303) << 4)) - 0x20202020;
        const int v1 = (((vl >> 4) & 0x0F0F0F0F) | (((vh >> 2) & 0x03030303) << 4)) - 0x20202020;

        // Get Y values
        const sycl::half2 yds = y_ds[y_block_base + bq8_offset + u_index / 4];
        const int8_t * u = y_qs + (y_block_base + bq8_offset) * QK8_1 + u_index * 4;

        const int u0 = *((const int *)u);
        const int u1 = *((const int *)(u + QK8_1));

        partial_sum += d * (float)yds[0] * (sc0 * (dpct::dp4a(v0, u0, 0)) +
                                            sc1 * (dpct::dp4a(v1, u1, 0)));
    }

    // Warp-level reduction
    float warp_sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());

    // Write to shared memory
    if (lane_id == 0) {
        shared_partials[warp_id] = warp_sum;
    }

    sycl::group_barrier(nd_item.get_group());

    // First warp reduces all partial sums
    if (warp_id == 0 && lane_id < num_tiles) {
        float val = shared_partials[lane_id];
        float final_sum = sycl::reduce_over_group(sg, val, sycl::plus<float>());
        if (lane_id == 0) {
            dst[row] = final_sum;
        }
    }
}
```

**Step 2: Add dispatch function**

Add after the kernel:

```cpp
static void variable_tile_mul_mat_vec_q6_k_q8_1_sycl(
    const void * vx, const void * vy, float * dst,
    const int ncols, const int nrows,
    sycl::queue & stream)
{
    const int blocks_per_row = ncols / QK_K;
    const int num_tiles = tile_count(blocks_per_row);
    const int threads_per_row = num_tiles * WARP_SIZE;

    // Shared memory for partial sums (one float per warp)
    const int shared_size = num_tiles * sizeof(float);

    sycl::range<3> grid(1, 1, nrows);
    sycl::range<3> block(1, 1, threads_per_row);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_partials(sycl::range<1>(num_tiles), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> nd_item) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q6_k_variable_tile(
                    vx, vy, dst, ncols, nrows, blocks_per_row,
                    shared_partials.get_pointer(), nd_item);
            });
    });
}
```

**Step 3: Rebuild**

```bash
./scripts/quick-rebuild.sh mmvq.cpp
```
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): add Q6_K variable tile MMVQ kernel with multi-warp parallelism"
```

---

## Task 4: Wire Up Variable Tile Dispatch

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:9380-9400`
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:3650-3670`

**Step 1: Update ggml-sycl.cpp reorder logic**

Find the Q6_K coalesced reorder section (around line 9380) and modify to always succeed:

```cpp
// Replace the divisibility check with variable tile support
case GGML_TYPE_Q6_K: {
    const size_t blocks_per_row = ne00 / QK_K;

    // Compute variable tile row stride
    const int num_tiles = tile_count(blocks_per_row);
    size_t row_stride = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_stride += ts * (128 + 64 + 16);
    }

    // D values go at the end
    const size_t data_size = ne01 * row_stride + ne01 * blocks_per_row * sizeof(ggml_half);

    reorder_q6_K_variable_tile(
        (const block_q6_K *)host_buf,
        (uint8_t *)reordered_buf,
        ne01, blocks_per_row, row_stride);

    // Copy D values to end
    const ggml_half * src_d = ...;  // Extract from AoS
    ggml_half * dst_d = (ggml_half *)((uint8_t *)reordered_buf + ne01 * row_stride);
    // ... copy D values

    GGML_SYCL_DEBUG("[CPU-REORDER] Q6_K variable tile: blocks_per_row=%zu, tiles=%d\n",
                   blocks_per_row, num_tiles);
    break;
}
```

**Step 2: Update mmvq.cpp dispatch**

Find the Q6_K dispatch (around line 3650) and add variable tile path:

```cpp
case GGML_TYPE_Q6_K:
    if (is_coalesced) {
        const int blocks_per_row = ne00 / QK_K;
        // Use variable tile kernel (handles any block count)
        GGML_SYCL_DEBUG("Calling variable_tile_mul_mat_vec_q6_k_q8_1_sycl\n");
        variable_tile_mul_mat_vec_q6_k_q8_1_sycl(
            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs,
            ne00, row_diff, stream);
    } else {
        // Fall back to AoS kernel
        mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs,
                                   ne00, row_diff, stream);
    }
    break;
```

**Step 3: Rebuild**

```bash
./scripts/quick-rebuild.sh
```
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): wire up Q6_K variable tile dispatch for any dimension"
```

---

## Task 5: Integration Test with Mistral Q6_K

**Files:**
- None (runtime test)

**Step 1: Test correctness**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15,`

**Step 2: Verify no regression on Q4_0**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15,`

**Step 3: Benchmark comparison**

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

**Step 4: Document results and commit**

```bash
git commit --allow-empty -m "test(sycl): verify Q6_K variable tile coalescing on Mistral 7B

Results:
- Correctness: PASS (counting sequence correct)
- Q4_0 regression: NONE
- Performance: [INSERT BENCHMARK RESULTS]
"
```

---

## Summary

| Task | Description | Estimated Steps |
|------|-------------|-----------------|
| 1 | Tile decomposition helpers | 5 |
| 2 | Variable tile reorder function | 5 |
| 3 | Variable tile MMVQ kernel | 4 |
| 4 | Wire up dispatch | 4 |
| 5 | Integration testing | 4 |

**Total: 22 steps**

Each step is atomic (2-5 minutes). Follow TDD: write test, verify fail, implement, verify pass, commit.
