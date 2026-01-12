# Q6_K Coalesced Memory Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement coalesced memory layout support for Q6_K quantized tensors in the SYCL backend.

**Architecture:** Extend existing coalesced infrastructure (Q4_0 pattern) to Q6_K. Add kernel, conversion function, and dispatch logic. Handle three Q6_K arrays (ql, qh, scales) with word-major reordering within 16-block tiles.

**Tech Stack:** SYCL C++, Intel oneAPI, llama.cpp ggml backend

---

### Task 1: Enable Q6_K in is_coalesced_supported()

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp:306-318`

**Step 1: Add Q6_K case to switch statement**

```cpp
// At line 308, after case GGML_TYPE_Q4_0:
inline bool is_coalesced_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        case GGML_TYPE_Q6_K:
            return true;
        // Future: add Q8_0, MXFP4, etc. as kernels are verified
        default:
            return false;
    }
}
```

**Step 2: Build to verify compilation**

Run: `./scripts/quick-rebuild.sh` or `cmake --build build -j 16`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): enable Q6_K in is_coalesced_supported()"
```

---

### Task 2: Add Q6_K Coalesced Conversion Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp` (add after convert_mxfp4_to_coalesced_sycl around line 1165)

**Step 1: Add the conversion kernel**

Insert before `convert_tensor_to_coalesced()` (around line 1165):

```cpp
// Q6_K Coalesced Conversion
// SoA layout: [all ql][all qh][all scales][all d]
// Coalesced layout per tile (16 blocks):
//   [ql word-major: 2048 bytes] [qh word-major: 1024 bytes] [scales word-major: 256 bytes]
// D values remain at tensor end (not coalesced within tiles)
static void convert_q6_k_to_coalesced_kernel(
    const uint8_t * __restrict__ src,   // SoA format
    uint8_t * __restrict__ dst,          // Coalesced format
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item)
{
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16
    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // SoA region sizes per row
    const int ql_per_row = blocks_per_row * (QK_K / 2);       // 128 bytes/block
    const int qh_per_row = blocks_per_row * (QK_K / 4);       // 64 bytes/block
    const int sc_per_row = blocks_per_row * (QK_K / 16);      // 16 bytes/block

    // SoA base pointers
    const uint8_t * src_ql = src;
    const uint8_t * src_qh = src_ql + nrows * ql_per_row;
    const uint8_t * src_sc = src_qh + nrows * qh_per_row;
    // D values stay in place (at tensor end)

    // Coalesced tile sizes
    constexpr int ql_tile_bytes = TILE_BLOCKS * (QK_K / 2);   // 16 * 128 = 2048
    constexpr int qh_tile_bytes = TILE_BLOCKS * (QK_K / 4);   // 16 * 64 = 1024
    constexpr int sc_tile_bytes = TILE_BLOCKS * (QK_K / 16);  // 16 * 16 = 256
    constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;  // 3328 bytes

    // Destination base for quant data (coalesced tiles)
    uint8_t * dst_tiles = dst;
    // D values remain at same location as SoA (after all ql+qh+sc)

    // Each thread handles one word (4 bytes) reordering
    const int global_id = item.get_global_linear_id();
    const int total_threads = item.get_global_range().size();

    // Process ql words (32 words per block, 512 words per tile)
    const int ql_words_per_tile = TILE_BLOCKS * 32;
    const int total_ql_words = nrows * tiles_per_row * ql_words_per_tile;

    for (int i = global_id; i < total_ql_words; i += total_threads) {
        const int tile_word = i % ql_words_per_tile;
        const int tile_idx = (i / ql_words_per_tile) % tiles_per_row;
        const int row = i / (tiles_per_row * ql_words_per_tile);

        const int word_in_block = tile_word / TILE_BLOCKS;
        const int block_in_tile = tile_word % TILE_BLOCKS;

        // Source: SoA sequential blocks
        const int src_offset = row * ql_per_row + (tile_idx * TILE_BLOCKS + block_in_tile) * (QK_K / 2) + word_in_block * 4;

        // Dest: word-major within tile
        const int dst_tile_base = row * tiles_per_row * tile_total + tile_idx * tile_total;
        const int dst_offset = dst_tile_base + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

        *((int *)(dst_tiles + dst_offset)) = *((const int *)(src_ql + src_offset));
    }

    // Process qh words (16 words per block, 256 words per tile)
    const int qh_words_per_tile = TILE_BLOCKS * 16;
    const int total_qh_words = nrows * tiles_per_row * qh_words_per_tile;

    for (int i = global_id; i < total_qh_words; i += total_threads) {
        const int tile_word = i % qh_words_per_tile;
        const int tile_idx = (i / qh_words_per_tile) % tiles_per_row;
        const int row = i / (tiles_per_row * qh_words_per_tile);

        const int word_in_block = tile_word / TILE_BLOCKS;
        const int block_in_tile = tile_word % TILE_BLOCKS;

        const int src_offset = row * qh_per_row + (tile_idx * TILE_BLOCKS + block_in_tile) * (QK_K / 4) + word_in_block * 4;
        const int dst_tile_base = row * tiles_per_row * tile_total + tile_idx * tile_total + ql_tile_bytes;
        const int dst_offset = dst_tile_base + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

        *((int *)(dst_tiles + dst_offset)) = *((const int *)(src_qh + src_offset));
    }

    // Process scales words (4 words per block, 64 words per tile)
    const int sc_words_per_tile = TILE_BLOCKS * 4;
    const int total_sc_words = nrows * tiles_per_row * sc_words_per_tile;

    for (int i = global_id; i < total_sc_words; i += total_threads) {
        const int tile_word = i % sc_words_per_tile;
        const int tile_idx = (i / sc_words_per_tile) % tiles_per_row;
        const int row = i / (tiles_per_row * sc_words_per_tile);

        const int word_in_block = tile_word / TILE_BLOCKS;
        const int block_in_tile = tile_word % TILE_BLOCKS;

        const int src_offset = row * sc_per_row + (tile_idx * TILE_BLOCKS + block_in_tile) * (QK_K / 16) + word_in_block * 4;
        const int dst_tile_base = row * tiles_per_row * tile_total + tile_idx * tile_total + ql_tile_bytes + qh_tile_bytes;
        const int dst_offset = dst_tile_base + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

        *((int *)(dst_tiles + dst_offset)) = *((const int *)(src_sc + src_offset));
    }
}

static void convert_q6_k_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    GGML_ASSERT((ncols / QK_K) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;

    // Total quant bytes (ql + qh + scales, excluding d)
    const size_t quant_bytes_per_row = blocks_per_row * ((QK_K/2) + (QK_K/4) + (QK_K/16));
    const size_t total_quant_bytes = nrows * quant_bytes_per_row;

    // Allocate temp buffer for in-place conversion
    uint8_t * temp = (uint8_t *)sycl::malloc_device(total_quant_bytes, *stream);
    GGML_ASSERT(temp != nullptr);

    // Copy current data to temp
    stream->memcpy(temp, data, total_quant_bytes).wait();

    // Launch conversion kernel
    const int total_work = nrows * tiles_per_row * MMVQ_COALESCED_TILE_BLOCKS * 32;  // ql dominant
    const sycl::range<3> global_size(1, 1, (total_work + 255) / 256 * 256);
    const sycl::range<3> local_size(1, 1, 256);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) {
                convert_q6_k_to_coalesced_kernel(temp, (uint8_t *)data, ncols, nrows, item);
            });
    });

    // Free temp after kernel completes
    stream->submit([&](sycl::handler & cgh) {
        cgh.host_task([=]() {
            sycl::free(temp, *stream);
        });
    });
}
```

**Step 2: Build to verify compilation**

Run: `./scripts/quick-rebuild.sh mmvq.cpp`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): add Q6_K coalesced conversion kernel"
```

---

### Task 3: Add Q6_K Coalesced MMVQ Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp` (add after mul_mat_vec_q4_0_coalesced around line 270)

**Step 1: Add the coalesced kernel**

```cpp
// Q6_K coalesced kernel - processes 16 blocks per tile with word-major layout
// Q6_K block: 256 elements, ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
// Coalesced tile: [ql: 2048 bytes][qh: 1024 bytes][scales: 256 bytes] = 3328 bytes
static void mul_mat_vec_q6_k_coalesced(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & nd_item)
{
    const auto sg = nd_item.get_sub_group();
    const int sg_range = sg.get_group_linear_range();
    const int workgroup_id = nd_item.get_group_linear_id();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int row = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16
    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Thread mapping: 32 threads per warp
    // Threads 0-15: blocks 0-15, first set of sub-blocks
    // Threads 16-31: blocks 0-15, second set of sub-blocks
    const int block_in_tile = lane_id % TILE_BLOCKS;
    const int thread_half = lane_id / TILE_BLOCKS;  // 0 or 1

    // Tile layout sizes
    constexpr int ql_tile_bytes = TILE_BLOCKS * (QK_K / 2);   // 2048
    constexpr int qh_tile_bytes = TILE_BLOCKS * (QK_K / 4);   // 1024
    constexpr int sc_tile_bytes = TILE_BLOCKS * (QK_K / 16);  // 256
    constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;

    // X base pointers (coalesced layout)
    const uint8_t * x_base = (const uint8_t *)vx;
    const int x_row_stride = tiles_per_row * tile_total;

    // D values are after all coalesced quant data
    const ggml_half * x_d = (const ggml_half *)(x_base + nrows * x_row_stride);

    // Y pointers (standard Q8_1 format)
    const block_q8_1 * y = (const block_q8_1 *)vy;

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        const int tile_base = row * x_row_stride + tile * tile_total;

        // Pointers to this tile's arrays
        const uint8_t * tile_ql = x_base + tile_base;
        const uint8_t * tile_qh = tile_ql + ql_tile_bytes;
        const int8_t * tile_sc = (const int8_t *)(tile_qh + qh_tile_bytes);

        // Get super-block scale for this block
        const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
        const float d = x_d[block_idx];

        // Process 8 Q8_1 sub-blocks per Q6_K block (256/32 = 8)
        // Thread half determines which 4 sub-blocks to process
        const int sub_start = thread_half * 4;

        int sumi = 0;

        for (int sub = 0; sub < 4; sub++) {
            const int sub_idx = sub_start + sub;
            const int y_block_idx = (tile * TILE_BLOCKS + block_in_tile) * 8 + sub_idx;

            // Load Y data
            const block_q8_1 * y_blk = &y[y_block_idx];

            // Load ql (16 bytes per sub-block, coalesced access)
            // sub_idx determines which 16-byte chunk within the block's 128-byte ql
            const int ql_word = sub_idx * 4;  // 4 words = 16 bytes per sub-block
            for (int w = 0; w < 4; w++) {
                const int ql_offset = (ql_word + w) * (TILE_BLOCKS * 4) + block_in_tile * 4;
                const int ql_val = *((const int *)(tile_ql + ql_offset));

                // Load qh (8 bytes per sub-block)
                const int qh_word = sub_idx * 2 + w / 2;
                const int qh_offset = qh_word * (TILE_BLOCKS * 4) + block_in_tile * 4;
                const int qh_val = *((const int *)(tile_qh + qh_offset));

                // Load scale for this sub-block
                const int sc_word = sub_idx / 4;
                const int sc_offset = sc_word * (TILE_BLOCKS * 4) + block_in_tile * 4;
                const int8_t sc = tile_sc[sc_offset + sub_idx % 4];

                // Combine low 4 bits (ql) with high 2 bits (qh) to get 6-bit values
                // Then dot product with Y
                int8_t x_vals[4];
                for (int j = 0; j < 4; j++) {
                    const int ql_nibble = (ql_val >> (j * 8)) & 0xFF;
                    const int qh_bits = (qh_val >> (j * 2 + (w % 2) * 8)) & 0x03;
                    x_vals[j] = (int8_t)((ql_nibble & 0x0F) | (qh_bits << 4)) - 32;
                }

                // dp4a equivalent
                for (int j = 0; j < 4; j++) {
                    sumi += x_vals[j] * y_blk->qs[w * 4 + j];
                }
            }
        }

        // Apply scale
        partial_sum += d * sumi;
    }

    // Warp reduction
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}
```

**Step 2: Build to verify compilation**

Run: `./scripts/quick-rebuild.sh mmvq.cpp`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): add Q6_K coalesced MMVQ kernel"
```

---

### Task 4: Add Q6_K Coalesced Dispatch Function

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp` (add after coalesced_mul_mat_vec_mxfp4_q8_1_sycl around line 1750)

**Step 1: Add dispatch function**

```cpp
// Dispatch function for coalesced Q6_K MMVQ kernel
static void coalesced_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                  const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = 16;
    const int block_num_y = ceil_div(nrows, (int)num_subgroups);

    const sycl::range<3> global_size(1, 1, block_num_y * num_subgroups * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, 1, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q6_k_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}
```

**Step 2: Build to verify compilation**

Run: `./scripts/quick-rebuild.sh mmvq.cpp`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): add Q6_K coalesced dispatch function"
```

---

### Task 5: Add Q6_K to convert_tensor_to_coalesced()

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:1198-1227` (convert_tensor_to_coalesced switch)

**Step 1: Add Q6_K case**

After the MXFP4 case (around line 1221), add:

```cpp
        case GGML_TYPE_Q6_K:
            block_size = QK_K;
            if ((ncols / block_size) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
                return false;  // Not tile-aligned
            }
            convert_q6_k_to_coalesced_sycl(tensor->data, ncols, nrows, stream);
            break;
```

**Step 2: Build to verify compilation**

Run: `./scripts/quick-rebuild.sh mmvq.cpp`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): add Q6_K case to convert_tensor_to_coalesced()"
```

---

### Task 6: Update Q6_K MMVQ Dispatch

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:3316-3329` (Q6_K dispatch case)

**Step 1: Replace the current dispatch logic**

Replace the existing Q6_K case with:

```cpp
            case GGML_TYPE_Q6_K:
                {
                    auto * extra = (ggml_tensor_extra_gpu *) dst->src[0]->extra;

                    if (extra && extra->optimized_feature.is_coalesced()) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_q6_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q6_k_coalesced", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                        coalesced_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else if (extra && extra->optimized_feature.is_soa()) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q6_k_soa", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld", (long long)ne00, (long long)row_diff, (long long)ne01, (long long)row_low);
                        const void * soa_base = ggml_sycl_get_data_ptr(src0, ctx.device);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(soa_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, ne01, row_low, stream);
                    } else {
                        GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q6_k_aos", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                        mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
```

**Step 2: Build to verify compilation**

Run: `./scripts/quick-rebuild.sh mmvq.cpp`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): update Q6_K MMVQ dispatch for coalesced layout"
```

---

### Task 7: Full Build and Test

**Files:**
- None (testing only)

**Step 1: Full rebuild**

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake --build build -j 16
```

Expected: Builds successfully with no errors

**Step 2: Test with Q6_K model**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q6_K.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

**Step 3: Test with Q4_0 model (has Q6_K output layer)**

```bash
GGML_SYCL_LAYOUT_OVERRIDE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

**Step 4: Commit final changes (if any fixes needed)**

```bash
git add -A
git commit -m "fix(sycl): Q6_K coalesced kernel corrections"
```

---

## Notes

- The Q6_K coalesced kernel is more complex than Q4_0 due to three separate arrays (ql, qh, scales)
- D values (super-block scales) remain at tensor end, not reordered within tiles
- Each thread processes one block from the tile, with 32 threads covering 16 blocks × 2 halves
- DMMV and MMQ will automatically fall back to AoS for coalesced tensors (Phase 2 optimization)
