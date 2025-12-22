#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "quantize.hpp"
#include "vecdotq.hpp"
#include "sycl-profiling.hpp"

// Kernel name classes for VTune/profiler visibility
// Note: Using int instead of ggml_type because SYCL kernel names require fixed underlying types
template <int qtype> class mmvq_kernel_name;
template <int qtype> class mmvq_reorder_kernel_name;
template <int qtype> class mmvq_reorder_slm_kernel_name;
template <int qtype> class mmvq_coalesced_kernel_name;
template <int qtype> class mmvq_id_kernel_name;

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// Multi-row reordered kernel with SLM Y-vector sharing
// All subgroups in the work-group share Y-vector cached in SLM
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder_slm(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                      const int ncols, const int nrows, const sycl::nd_item<3> & nd_item,
                                      int8_t * __restrict__ slm_y_qs, sycl::half2 * __restrict__ slm_y_ds) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  lane_id      = sg.get_local_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    const int blocks_per_row = ncols / block_traits::qk;

    // Step 1: First subgroup loads Y-vector to SLM cooperatively
    // Y is in reordered format: quants at vy[0..ncols-1], ds at vy[ncols..]
    if (sg_id == 0) {
        // Load Y quants (int8) - ncols bytes total
        const int8_t * y_qs = (const int8_t *)vy;
        for (int i = lane_id; i < ncols; i += WARP_SIZE) {
            slm_y_qs[i] = y_qs[i];
        }

        // Load Y ds (half2) - blocks_per_row entries
        const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)vy + ncols);
        for (int i = lane_id; i < blocks_per_row; i += WARP_SIZE) {
            slm_y_ds[i] = y_ds[i];
        }
    }

    // Barrier: wait for Y to be loaded to SLM
    nd_item.barrier(sycl::access::fence_space::local_space);

    if (row >= nrows) {
        return;
    }

    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * blocks_per_row;

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = lane_id / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        // Use SLM-cached Y data instead of device memory
        const int8_t* q8_1_quant_ptr = slm_y_qs + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = slm_y_ds + iby;

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (lane_id % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// Warp-coalesced MMVQ kernel for Q4_0
// Tensor layout: within each 16-block tile, data is word-major instead of block-major
// Word w of block b is at: tile_offset + w * 64 + b * 4
// This achieves 100% cache line utilization (vs 50% with strided access in standard reorder)
//
// Thread mapping (32 threads process 16 blocks per iteration):
// - Threads 0-15:  process blocks 0-15, lower half (X bytes 0-7 = elements 0-7 + 16-23)
// - Threads 16-31: process blocks 0-15, upper half (X bytes 8-15 = elements 8-15 + 24-31)
//
// Memory access pattern for first load:
// - T0: offset 0, T1: offset 4, T2: offset 8, ... T15: offset 60 (perfect 4-byte coalescing)
// - T16: offset 128, T17: offset 132, ... T31: offset 188 (another perfect cache line)
static void mul_mat_vec_q4_0_coalesced(
    const void * __restrict__ vx,        // Coalesced X weights
    const void * __restrict__ vy,        // Reordered Y activations
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

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16 blocks per tile
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Thread role: which block in tile and which half (lower=words 0,1 vs upper=words 2,3)
    const int block_in_tile = lane_id % TILE_BLOCKS;  // 0-15
    const int is_upper_half = lane_id / TILE_BLOCKS;   // 0 or 1

    // X base pointers (coalesced layout: quants first, then scales)
    // Quants: tiles_per_row * TILE_BLOCKS * 16 bytes per row = ncols/2 bytes
    const uint8_t * x_qs = (const uint8_t *)vx;
    const int x_row_stride = ncols / 2;  // bytes per row of quants

    // Scales are after all quants in the tensor
    const ggml_half * x_d = (const ggml_half *)((const char *)vx + nrows * x_row_stride);

    // Y base pointers (standard reordered format: quants, then ds)
    const int8_t * y_qs = (const int8_t *)vy;
    const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)vy + ncols);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Base offset for this tile's quants (256 bytes per tile)
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES;

        // Coalesced load: word w of block b at offset w*64 + b*4
        // Thread loads 2 words (8 bytes total) from its assigned block
        // Lower half (is_upper_half=0): words 0,1 at offsets 0+b*4, 64+b*4
        // Upper half (is_upper_half=1): words 2,3 at offsets 128+b*4, 192+b*4
        const int word_base = is_upper_half * 128;  // 0 or 128
        const int word0_offset = word_base + block_in_tile * 4;       // word 0 or 2
        const int word1_offset = word_base + 64 + block_in_tile * 4;  // word 1 or 3

        // Perfectly coalesced 4-byte loads
        const int v0 = *((const int *)(x_qs + tile_base + word0_offset));
        const int v1 = *((const int *)(x_qs + tile_base + word1_offset));

        // Get scale for this block (scales are NOT coalesced, remain block-sequential)
        const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
        const float d = x_d[block_idx];

        // Y block index and base offset
        const int y_block = tile * TILE_BLOCKS + block_in_tile;
        const int y_base = y_block * QK8_1;

        // Load Y data matching the X half we're processing
        // For lower half (is_upper_half=0): Y elements 0-3, 16-19, 4-7, 20-23
        // For upper half (is_upper_half=1): Y elements 8-11, 24-27, 12-15, 28-31
        const int y_offset = is_upper_half * 8;  // 0 or 8 (in terms of bytes)
        const int u0 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4);      // Y[0:3] or Y[8:11]
        const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 4);  // Y[16:19] or Y[24:27]
        const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);  // Y[4:7] or Y[12:15]
        const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 5);  // Y[20:23] or Y[28:31]

        // Extract nibbles and compute dp4a
        const int vi0_0 = (v0 >> 0) & 0x0F0F0F0F;  // low nibbles of word 0
        const int vi1_0 = (v0 >> 4) & 0x0F0F0F0F;  // high nibbles of word 0
        const int vi0_1 = (v1 >> 0) & 0x0F0F0F0F;  // low nibbles of word 1
        const int vi1_1 = (v1 >> 4) & 0x0F0F0F0F;  // high nibbles of word 1

        int sumi = 0;
        sumi = dpct::dp4a(vi0_0, u0, sumi);
        sumi = dpct::dp4a(vi1_0, u1, sumi);
        sumi = dpct::dp4a(vi0_1, u2, sumi);
        sumi = dpct::dp4a(vi1_1, u3, sumi);

        // Apply scales: result = d4 * (sumi * ds8.x - 4 * ds8.y)
        // The 4 comes from: 8 elements * (subtract 8 offset) / 16 = 4
        const sycl::half2 ds8 = y_ds[y_block];
        const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();
        partial_sum += d * (sumi * ds8f.x() - 4.0f * ds8f.y());
    }

    // Warp reduction using subgroup intrinsic
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Hoist invariant iqs calculation outside the loop
    constexpr int qi_div_vdr = qi / vdr;
    const int lane_id = item_ct1.get_local_id(2);
    const int base_iqs = vdr * (lane_id % qi_div_vdr);
    const int row_offset = row * blocks_per_row;

    // 4-way accumulator for better ILP (matches Xe2 4-cycle FMA latency)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int i = lane_id / qi_div_vdr;
    const int stride = blocks_per_warp;
    const int stride4 = 4 * stride;

    // Main loop: 4x unrolled for ILP
    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y[iby0], iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y[iby1], iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y[iby2], iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y[iby3], iqs);
        }
    }

    // Handle remainder
    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // Combine accumulators (tree reduction for fewer dependencies)
    float tmp = (acc0 + acc1) + (acc2 + acc3);

    // Use subgroup reduce for final reduction (more efficient than manual XOR)
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

// MoE-aware kernel: routes to different experts based on ids tensor
// Handles 2D iteration: (iid1, id) over tokens and expert selections
// For MUL_MAT_ID: reads ids[iid1][id] to determine which expert weights to use
// This allows GPU-side expert routing without host sync
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_id(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                              const int32_t * __restrict__ ids, const int ncols, const int nrows_per_expert,
                              const int n_ids, const int ne11,
                              const int64_t stride_expert_x,
                              const int64_t ids_nb0, const int64_t ids_nb1,
                              const int64_t nb11, const int64_t nb12,
                              const int64_t nb1, const int64_t nb2,
                              const sycl::nd_item<3> & item_ct1) {
    // batch_idx from block.y dimension (linearized over iid1 * n_ids + id)
    const int batch_idx = item_ct1.get_group(1);
    // row within expert from block.z dimension
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - 2D iteration structure
    const int iid1 = batch_idx / n_ids;  // Token position
    const int id = batch_idx % n_ids;    // Expert selection index

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *)((const char*)ids + iid1 * ids_nb1 + id * ids_nb0);

    // Compute src1 and dst offsets matching host-side logic
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1 = id;
    const int64_t i2 = iid1;

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    assert(blocks_per_warp > 0);

    // Expert weights: offset by expert_id * stride_expert_x
    const block_q_t *  x = (const block_q_t *) ((const char*)vx + expert_id * stride_expert_x);
    // Input: offset using proper 2D indexing
    const block_q8_1 * y = (const block_q8_1 *) ((const char*)vy + i11 * nb11 + i12 * nb12);

    // Hoist invariant calculations outside the loop
    constexpr int qi_div_vdr = qi / vdr;
    const int lane_id = item_ct1.get_local_id(2);
    const int base_iqs = vdr * (lane_id % qi_div_vdr);
    const int row_offset = row * blocks_per_row;

    // 4-way accumulator for better ILP (matches Xe2 4-cycle FMA latency)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int i = lane_id / qi_div_vdr;
    const int stride = blocks_per_warp;
    const int stride4 = 4 * stride;

    // Main loop: 4x unrolled for ILP
    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y[iby0], iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y[iby1], iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y[iby2], iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y[iby3], iqs);
        }
    }

    // Handle remainder
    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // Combine accumulators (tree reduction for fewer dependencies)
    float tmp = (acc0 + acc1) + (acc2 + acc3);

    // Use subgroup reduce for final reduction (more efficient than manual XOR)
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        // Output: offset using proper 2D indexing
        float * dst_out = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
        dst_out[row] = tmp;
    }
}

// Multi-row MMVQ kernel: processes multiple output rows per work-group
// Shares Y-vector in SLM across all rows to reduce memory bandwidth
// Expected +15-25% improvement for token generation
template <int qk, int qi, typename block_q_t, int vdr,
          float (*vec_dot_q_slm)(const void *, const int *, const sycl::half2 *, int, int, const int &),
          int nrows_per_wg>
static void mul_mat_vec_q_multirow(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                   const int ncols, const int nrows,
                                   const sycl::nd_item<3> & item_ct1,
                                   int * __restrict__ slm_y_qs,
                                   sycl::half2 * __restrict__ slm_y_ds) {
    // Work-group layout: (1, nrows_per_wg, WARP_SIZE)
    // Each warp handles one row, all warps share Y-vector in SLM
    const int local_row = item_ct1.get_local_id(1);  // Which row within work-group (0 to nrows_per_wg-1)
    const int lane_id = item_ct1.get_local_id(2);    // Thread within warp (0 to 31)
    const int wg_idx = item_ct1.get_group(2);        // Work-group index
    const int row = wg_idx * nrows_per_wg + local_row;  // Global row index

    const int blocks_per_row = ncols / qk;

    // Step 1: First warp (local_row == 0) loads Y-vector to SLM
    // All threads in the first warp cooperatively load Y data
    if (local_row == 0) {
        const block_q8_1 * y = (const block_q8_1 *) vy;

        // Each thread loads its share of blocks
        for (int blk = lane_id; blk < blocks_per_row; blk += WARP_SIZE) {
            // Load 8 ints (32 bytes) of quantized data per block
            const int slm_offset = blk * MMVQ_SLM_Y_QS_STRIDE;
            #pragma unroll
            for (int j = 0; j < QI8_1; ++j) {
                slm_y_qs[slm_offset + j] = get_int_from_int8_aligned(y[blk].qs, j);
            }
            // Load ds (scale and sum as half2)
            slm_y_ds[blk] = *((const sycl::half2 *) &y[blk].ds);
        }
    }

    // Barrier: wait for Y-vector to be loaded to SLM
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Step 2: Each warp computes its row using Y from SLM
    if (row >= nrows) {
        return;
    }

    const block_q_t * x = (const block_q_t *) vx;

    // Hoist invariant iqs calculation outside the loop
    constexpr int qi_div_vdr = qi / vdr;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    const int base_iqs = vdr * (lane_id % qi_div_vdr);
    const int row_offset = row * blocks_per_row;

    // 4-way accumulator for better ILP
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int i = lane_id / qi_div_vdr;
    const int stride = blocks_per_warp;
    const int stride4 = 4 * stride;

    // Main loop: 4x unrolled for ILP
    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        // Y block indices (for SLM lookup)
        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx0], slm_y_qs, slm_y_ds, iby0, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc1 += vec_dot_q_slm(&x[ibx1], slm_y_qs, slm_y_ds, iby1, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc2 += vec_dot_q_slm(&x[ibx2], slm_y_qs, slm_y_ds, iby2, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc3 += vec_dot_q_slm(&x[ibx3], slm_y_qs, slm_y_ds, iby3, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    // Handle remainder
    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx], slm_y_qs, slm_y_ds, iby, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    // Combine accumulators
    float tmp = (acc0 + acc1) + (acc2 + acc3);

    // Subgroup reduce
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

// Kernel name class for multi-row MMVQ
template <int qtype> class mmvq_multirow_kernel_name;

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q4_0>>(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

// Q8_0 reorder MMVQ dispatch function
static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q8_0>>(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

// MXFP4 reorder MMVQ dispatch function
static void reorder_mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                 const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_MXFP4>>(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_item);
                         });
    });
}

// GPU kernel to convert Q4_0 reordered format to warp-coalesced format
// Input layout (per 16-block tile):  [B0.qs[0:15]][B1.qs[0:15]]...[B15.qs[0:15]] = block-major
// Output layout (per 16-block tile): [W0:B0..B15][W1:B0..B15][W2:B0..B15][W3:B0..B15] = word-major
// Where W0 = bytes 0-3, W1 = bytes 4-7, W2 = bytes 8-11, W3 = bytes 12-15 of each block
static void convert_q4_0_to_coalesced_kernel(
    const uint8_t * __restrict__ src,   // Reordered format
    uint8_t * __restrict__ dst,          // Coalesced format
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item)
{
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int bytes_per_row = ncols / 2;  // 16 bytes per block, 32 elements per block

    // Grid: one work-item per 4-byte word in the tensor
    // Total words per row = blocks_per_row * 4 = tiles_per_row * 16 * 4
    const int global_id = item.get_global_linear_id();
    const int total_words_per_row = blocks_per_row * 4;
    const int total_words = nrows * total_words_per_row;

    if (global_id >= total_words) {
        return;
    }

    // Decompose global_id into (row, word_in_row)
    const int row = global_id / total_words_per_row;
    const int word_in_row = global_id % total_words_per_row;

    // Decompose word_in_row into (tile, block_in_tile, word_in_block)
    const int words_per_tile = TILE_BLOCKS * 4;  // 64 words per tile
    const int tile = word_in_row / words_per_tile;
    const int word_in_tile = word_in_row % words_per_tile;
    const int block_in_tile = word_in_tile / 4;
    const int word_in_block = word_in_tile % 4;

    // Source offset (block-major): row * bytes_per_row + tile * 256 + block * 16 + word * 4
    const int src_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 16) + block_in_tile * 16 + word_in_block * 4;

    // Destination offset (word-major): row * bytes_per_row + tile * 256 + word * 64 + block * 4
    const int dst_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 16) + word_in_block * 64 + block_in_tile * 4;

    // Copy 4 bytes
    *((int *)(dst + dst_offset)) = *((const int *)(src + src_offset));
}

// Convert Q4_0 reordered tensor to coalesced layout in-place
// Note: This modifies the tensor data directly. Only call once per tensor.
// WARNING: This must be called OUTSIDE of graph recording mode (cannot use wait() during recording)
static void convert_q4_0_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    GGML_ASSERT((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int blocks_per_row = ncols / QK4_0;
    const int total_words = nrows * blocks_per_row * 4;  // 4 words per block

    // Allocate temporary buffer for conversion
    const int bytes_per_row = ncols / 2;
    const int total_bytes = nrows * bytes_per_row;

    uint8_t * temp = sycl::malloc_device<uint8_t>(total_bytes, *stream);

    // Copy original quants to temp
    sycl::event copy_event = stream->memcpy(temp, data, total_bytes);

    // Convert from temp to data (now coalesced)
    const int block_size = 256;
    const int num_blocks = (total_words + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        // Depend on copy completing before conversion
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_q4_0_to_coalesced_kernel(temp, (uint8_t *)data, ncols, nrows, item);
            });
    });

    // Free temp buffer after conversion completes using host_task
    // host_task waits for convert_event then frees, without blocking the host thread
    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task([temp, stream]() {
            sycl::free(temp, *stream);
        });
    });
}

// Public API for coalesced conversion - call at model load time, after reorder
bool ggml_sycl_convert_to_coalesced_q4_0(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    // Check if coalesced mode is enabled
    static bool coalesced_enabled = (std::getenv("GGML_SYCL_MMVQ_COALESCED") != nullptr);
    if (!coalesced_enabled) {
        return false;
    }

    // Only for Q4_0 type
    if (tensor->type != GGML_TYPE_Q4_0) {
        return false;
    }

    // Check if already converted
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;
    if (!extra || extra->optimized_feature.coalesced) {
        return false;
    }

    // Must be reordered first
    if (!extra->optimized_feature.reorder) {
        return false;
    }

    // Skip TP-sharded tensors
    if (extra->tp_sharded) {
        return false;
    }

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];

    // Check if tensor dimensions are compatible with coalesced kernel
    if ((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        return false;
    }

    GGML_SYCL_DEBUG("Converting Q4_0 tensor to coalesced layout at model load: %s\n", tensor->name);

    // Use tensor->data directly (same as reorder_qw does - this is the device pointer at load time)
    convert_q4_0_to_coalesced_sycl(tensor->data, ncols, nrows, stream);
    extra->optimized_feature.coalesced = true;

    return true;
}

// Dispatch function for coalesced Q4_0 MMVQ kernel
static void coalesced_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                  const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    GGML_ASSERT((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_coalesced_kernel_name<GGML_TYPE_Q4_0>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q4_0_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// ============================================================================
// Q8_0 Warp-Coalesced MMVQ Kernel
// ============================================================================
// Q8_0 block: 32 int8 quants (32 bytes) + fp16 scale (2 bytes) = 34 bytes
// Coalesced layout: group consecutive words (4 bytes) across 16 blocks per tile
//
// Memory layout per tile (16 blocks, 512 bytes quants):
// Source (block-major): [B0.W0-W7][B1.W0-W7]...[B15.W0-W7]
// Dest (word-major):    [W0:B0-B15][W1:B0-B15]...[W7:B0-B15]
//
// Thread mapping (32 threads process 16 blocks per iteration):
// - Threads 0-15:  process blocks 0-15, lower half (words 0-3 = elements 0-15)
// - Threads 16-31: process blocks 0-15, upper half (words 4-7 = elements 16-31)
static void mul_mat_vec_q8_0_coalesced(
    const void * __restrict__ vx,        // Coalesced X weights
    const void * __restrict__ vy,        // Reordered Y activations
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

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16 blocks per tile
    const int blocks_per_row = ncols / QK8_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Thread role: which block in tile and which half (lower=words 0-3 vs upper=words 4-7)
    const int block_in_tile = lane_id % TILE_BLOCKS;  // 0-15
    const int is_upper_half = lane_id / TILE_BLOCKS;  // 0 or 1

    // X base pointers (coalesced layout: quants first, then scales)
    // Quants: tiles_per_row * TILE_BLOCKS * 32 bytes per row = ncols bytes
    const uint8_t * x_qs = (const uint8_t *)vx;
    const int x_row_stride = ncols;  // bytes per row of quants (32 bytes/block * blocks)

    // Scales are after all quants in the tensor
    const ggml_half * x_d = (const ggml_half *)((const char *)vx + nrows * x_row_stride);

    // Y base pointers (standard reordered format: quants, then ds)
    const int8_t * y_qs = (const int8_t *)vy;
    const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)vy + ncols);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Base offset for this tile's quants (512 bytes per tile for Q8_0)
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES_Q8_0;

        // Coalesced load: word w of block b at offset w*64 + b*4
        // Thread loads 4 words (16 bytes total) from its assigned block
        // Lower half (is_upper_half=0): words 0-3 at offsets 0+b*4, 64+b*4, 128+b*4, 192+b*4
        // Upper half (is_upper_half=1): words 4-7 at offsets 256+b*4, 320+b*4, 384+b*4, 448+b*4
        const int word_base = is_upper_half * 256;  // 0 or 256 (4 words * 64 bytes stride)

        // Perfectly coalesced 4-byte loads
        const int v0 = *((const int *)(x_qs + tile_base + word_base + 0 * 64 + block_in_tile * 4));
        const int v1 = *((const int *)(x_qs + tile_base + word_base + 1 * 64 + block_in_tile * 4));
        const int v2 = *((const int *)(x_qs + tile_base + word_base + 2 * 64 + block_in_tile * 4));
        const int v3 = *((const int *)(x_qs + tile_base + word_base + 3 * 64 + block_in_tile * 4));

        // Get scale for this block (scales are NOT coalesced, remain block-sequential)
        const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
        const float d = x_d[block_idx];

        // Y block index and base offset
        const int y_block = tile * TILE_BLOCKS + block_in_tile;
        const int y_base = y_block * QK8_1;

        // Load Y data matching the X half we're processing
        // For lower half (is_upper_half=0): Y elements 0-15
        // For upper half (is_upper_half=1): Y elements 16-31
        const int y_offset = is_upper_half * 16;  // 0 or 16 (in bytes)
        const int u0 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 0);  // Y[0:3] or Y[16:19]
        const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);  // Y[4:7] or Y[20:23]
        const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 2);  // Y[8:11] or Y[24:27]
        const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 3);  // Y[12:15] or Y[28:31]

        // dp4a: compute dot product of 4 int8 pairs
        int sumi = 0;
        sumi = dpct::dp4a(v0, u0, sumi);
        sumi = dpct::dp4a(v1, u1, sumi);
        sumi = dpct::dp4a(v2, u2, sumi);
        sumi = dpct::dp4a(v3, u3, sumi);

        // Apply scales: Q8_0 × Q8_1 = d8_0 * d8_1 * sumi
        const sycl::half2 ds8 = y_ds[y_block];
        const float d8_1 = ds8[0];
        partial_sum += d * d8_1 * sumi;
    }

    // Warp reduction using subgroup intrinsic
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// GPU kernel to convert Q8_0 reordered format to warp-coalesced format
// Input layout (per 16-block tile):  [B0.qs[0:31]][B1.qs[0:31]]...[B15.qs[0:31]] = block-major
// Output layout (per 16-block tile): [W0:B0..B15][W1:B0..B15]...[W7:B0..B15] = word-major
// Where W0 = bytes 0-3, W1 = bytes 4-7, ..., W7 = bytes 28-31 of each block
static void convert_q8_0_to_coalesced_kernel(
    const uint8_t * __restrict__ src,   // Reordered format
    uint8_t * __restrict__ dst,          // Coalesced format
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item)
{
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16
    const int blocks_per_row = ncols / QK8_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int bytes_per_row = ncols;  // 32 bytes per block, 32 elements per block

    // Grid: one work-item per 4-byte word in the tensor
    // Total words per row = blocks_per_row * 8 = tiles_per_row * 16 * 8
    const int global_id = item.get_global_linear_id();
    const int total_words_per_row = blocks_per_row * 8;  // 8 words per Q8_0 block
    const int total_words = nrows * total_words_per_row;

    if (global_id >= total_words) {
        return;
    }

    // Decompose global_id into (row, word_in_row)
    const int row = global_id / total_words_per_row;
    const int word_in_row = global_id % total_words_per_row;

    // Decompose word_in_row into (tile, block_in_tile, word_in_block)
    const int words_per_tile = TILE_BLOCKS * 8;  // 128 words per tile
    const int tile = word_in_row / words_per_tile;
    const int word_in_tile = word_in_row % words_per_tile;
    const int block_in_tile = word_in_tile / 8;
    const int word_in_block = word_in_tile % 8;

    // Source offset (block-major): row * bytes_per_row + tile * 512 + block * 32 + word * 4
    const int src_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 32) + block_in_tile * 32 + word_in_block * 4;

    // Destination offset (word-major): row * bytes_per_row + tile * 512 + word * 64 + block * 4
    const int dst_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 32) + word_in_block * 64 + block_in_tile * 4;

    // Copy 4 bytes
    *((int *)(dst + dst_offset)) = *((const int *)(src + src_offset));
}

// Convert Q8_0 reordered tensor to coalesced layout in-place
static void convert_q8_0_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    GGML_ASSERT((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int blocks_per_row = ncols / QK8_0;
    const int total_words = nrows * blocks_per_row * 8;  // 8 words per block

    // Allocate temporary buffer for conversion
    const int bytes_per_row = ncols;
    const int total_bytes = nrows * bytes_per_row;

    uint8_t * temp = sycl::malloc_device<uint8_t>(total_bytes, *stream);

    // Copy original quants to temp
    sycl::event copy_event = stream->memcpy(temp, data, total_bytes);

    // Convert from temp to data (now coalesced)
    const int block_size = 256;
    const int num_blocks = (total_words + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_q8_0_to_coalesced_kernel(temp, (uint8_t *)data, ncols, nrows, item);
            });
    });

    // Free temp buffer after conversion completes using host_task
    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task([temp, stream]() {
            sycl::free(temp, *stream);
        });
    });
}

// Public API for Q8_0 coalesced conversion - call at model load time, after reorder
bool ggml_sycl_convert_to_coalesced_q8_0(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    // Check if coalesced mode is enabled
    static bool coalesced_enabled = (std::getenv("GGML_SYCL_MMVQ_COALESCED") != nullptr);
    if (!coalesced_enabled) {
        return false;
    }

    // Only for Q8_0 type
    if (tensor->type != GGML_TYPE_Q8_0) {
        return false;
    }

    // Check if already converted
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;
    if (!extra || extra->optimized_feature.coalesced) {
        return false;
    }

    // Must be reordered first
    if (!extra->optimized_feature.reorder) {
        return false;
    }

    // Skip TP-sharded tensors
    if (extra->tp_sharded) {
        return false;
    }

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];

    // Check if tensor dimensions are compatible with coalesced kernel
    if ((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        GGML_SYCL_DEBUG("Q8_0 coalesced SKIP %s: ncols=%ld, blocks=%ld, mod=%ld\n",
                        tensor->name, (long)ncols, (long)(ncols / QK8_0),
                        (long)((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS));
        return false;
    }

    GGML_SYCL_DEBUG("Converting Q8_0 tensor to coalesced layout at model load: %s (ncols=%ld)\n", tensor->name, (long)ncols);

    convert_q8_0_to_coalesced_sycl(tensor->data, ncols, nrows, stream);
    extra->optimized_feature.coalesced = true;

    return true;
}

// Dispatch function for coalesced Q8_0 MMVQ kernel
static void coalesced_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                  const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    GGML_ASSERT((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_coalesced_kernel_name<GGML_TYPE_Q8_0>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q8_0_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// ============================================================================
// MXFP4 Warp-Coalesced MMVQ Kernel
// ============================================================================
// MXFP4 block: 16 packed bytes (32 4-bit elements) + 1 byte E8M0 exponent = 17 bytes
// Same coalesced layout as Q4_0 (16 bytes quants per block)
static void mul_mat_vec_mxfp4_coalesced(
    const void * __restrict__ vx,        // Coalesced X weights
    const void * __restrict__ vy,        // Reordered Y activations
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

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16 blocks per tile
    const int blocks_per_row = ncols / QK_MXFP4;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Thread role: which block in tile and which half (lower=words 0,1 vs upper=words 2,3)
    const int block_in_tile = lane_id % TILE_BLOCKS;  // 0-15
    const int is_upper_half = lane_id / TILE_BLOCKS;  // 0 or 1

    // X base pointers (coalesced layout: quants first, then scales)
    const uint8_t * x_qs = (const uint8_t *)vx;
    const int x_row_stride = ncols / 2;  // bytes per row of quants (16 bytes/block)

    // Scales are after all quants in the tensor (1 byte E8M0 per block)
    const uint8_t * x_e = (const uint8_t *)vx + nrows * x_row_stride;

    // Y base pointers (standard reordered format: quants, then ds)
    const int8_t * y_qs = (const int8_t *)vy;
    const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)vy + ncols);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Base offset for this tile's quants (256 bytes per tile for MXFP4)
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES_MXFP4;

        // Coalesced load: word w of block b at offset w*64 + b*4
        const int word_base = is_upper_half * 128;  // 0 or 128
        const int word0_offset = word_base + block_in_tile * 4;       // word 0 or 2
        const int word1_offset = word_base + 64 + block_in_tile * 4;  // word 1 or 3

        // Perfectly coalesced 4-byte loads
        const int v0 = *((const int *)(x_qs + tile_base + word0_offset));
        const int v1 = *((const int *)(x_qs + tile_base + word1_offset));

        // Get E8M0 exponent for this block
        const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
        const uint8_t e8m0 = x_e[block_idx];
        const float scale = ggml_sycl_e8m0_to_fp32(e8m0) * 0.5f;

        // Y block index and base offset
        const int y_block = tile * TILE_BLOCKS + block_in_tile;
        const int y_base = y_block * QK8_1;

        // Load Y data matching the X half we're processing
        const int y_offset = is_upper_half * 8;  // 0 or 8 (in terms of bytes)

        // Use MXFP4 lookup table for dequantization
        // Process 8 elements per load (2 words of 4 nibbles each = 8 FP4 values)
        const sycl::int2 dq0 = get_int_from_table_16(v0, kvalues_mxfp4);
        const sycl::int2 dq1 = get_int_from_table_16(v1, kvalues_mxfp4);

        // Load corresponding Y values
        const int u0 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4);      // Y[0:3] or Y[8:11]
        const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 4);  // Y[16:19] or Y[24:27]
        const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);  // Y[4:7] or Y[12:15]
        const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 5);  // Y[20:23] or Y[28:31]

        // dp4a: compute dot product
        int sumi = 0;
        sumi = ggml_sycl_dp4a(dq0.x(), u0, sumi);
        sumi = ggml_sycl_dp4a(dq0.y(), u1, sumi);
        sumi = ggml_sycl_dp4a(dq1.x(), u2, sumi);
        sumi = ggml_sycl_dp4a(dq1.y(), u3, sumi);

        // Apply scales
        const sycl::half2 ds8 = y_ds[y_block];
        const float d8_1 = ds8[0];
        partial_sum += scale * d8_1 * sumi;
    }

    // Warp reduction using subgroup intrinsic
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// GPU kernel to convert MXFP4 reordered format to warp-coalesced format
// Same layout as Q4_0 (16 bytes quants per block)
static void convert_mxfp4_to_coalesced_kernel(
    const uint8_t * __restrict__ src,
    uint8_t * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item)
{
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;  // 16
    const int blocks_per_row = ncols / QK_MXFP4;
    const int bytes_per_row = ncols / 2;  // 16 bytes per block, 32 elements per block

    const int global_id = item.get_global_linear_id();
    const int total_words_per_row = blocks_per_row * 4;  // 4 words per MXFP4 block
    const int total_words = nrows * total_words_per_row;

    if (global_id >= total_words) {
        return;
    }

    const int row = global_id / total_words_per_row;
    const int word_in_row = global_id % total_words_per_row;

    const int words_per_tile = TILE_BLOCKS * 4;
    const int tile = word_in_row / words_per_tile;
    const int word_in_tile = word_in_row % words_per_tile;
    const int block_in_tile = word_in_tile / 4;
    const int word_in_block = word_in_tile % 4;

    const int src_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 16) + block_in_tile * 16 + word_in_block * 4;
    const int dst_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 16) + word_in_block * 64 + block_in_tile * 4;

    *((int *)(dst + dst_offset)) = *((const int *)(src + src_offset));
}

static void convert_mxfp4_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    GGML_ASSERT((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int blocks_per_row = ncols / QK_MXFP4;
    const int total_words = nrows * blocks_per_row * 4;

    const int bytes_per_row = ncols / 2;
    const int total_bytes = nrows * bytes_per_row;

    uint8_t * temp = sycl::malloc_device<uint8_t>(total_bytes, *stream);

    sycl::event copy_event = stream->memcpy(temp, data, total_bytes);

    const int block_size = 256;
    const int num_blocks = (total_words + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_mxfp4_to_coalesced_kernel(temp, (uint8_t *)data, ncols, nrows, item);
            });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task([temp, stream]() {
            sycl::free(temp, *stream);
        });
    });
}

// Public API for MXFP4 coalesced conversion
bool ggml_sycl_convert_to_coalesced_mxfp4(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    static bool coalesced_enabled = (std::getenv("GGML_SYCL_MMVQ_COALESCED") != nullptr);
    if (!coalesced_enabled) {
        return false;
    }

    if (tensor->type != GGML_TYPE_MXFP4) {
        return false;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;
    if (!extra || extra->optimized_feature.coalesced) {
        return false;
    }

    if (!extra->optimized_feature.reorder) {
        return false;
    }

    if (extra->tp_sharded) {
        return false;
    }

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];

    if ((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        GGML_SYCL_DEBUG("MXFP4 coalesced SKIP %s: ncols=%ld, blocks=%ld, mod=%ld\n",
                        tensor->name, (long)ncols, (long)(ncols / QK_MXFP4),
                        (long)((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS));
        return false;
    }

    GGML_SYCL_DEBUG("Converting MXFP4 tensor to coalesced layout at model load: %s (ncols=%ld)\n", tensor->name, (long)ncols);

    convert_mxfp4_to_coalesced_sycl(tensor->data, ncols, nrows, stream);
    extra->optimized_feature.coalesced = true;

    return true;
}

// Dispatch function for coalesced MXFP4 MMVQ kernel
static void coalesced_mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                   const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    GGML_ASSERT((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_coalesced_kernel_name<GGML_TYPE_MXFP4>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_mxfp4_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);

    // Use multi-row kernel with SLM Y-vector sharing for better bandwidth utilization
    constexpr int NROWS_PER_WG = MMVQ_NROWS_PER_WG;
    const int block_num_z = (nrows + NROWS_PER_WG - 1) / NROWS_PER_WG;
    const sycl::range<3> block_nums(1, 1, block_num_z);
    const sycl::range<3> block_dims(1, NROWS_PER_WG, WARP_SIZE);

    const int blocks_per_row = ncols / QK4_0;
    const int slm_y_qs_size = blocks_per_row * MMVQ_SLM_Y_QS_STRIDE;
    const int slm_y_ds_size = blocks_per_row + 1;  // +1 for padding

    stream->submit([&](sycl::handler & cgh) {
        // Allocate SLM for Y-vector (shared across all rows in work-group)
        sycl::local_accessor<int, 1> slm_y_qs(slm_y_qs_size, cgh);
        sycl::local_accessor<sycl::half2, 1> slm_y_ds(slm_y_ds_size, cgh);

        cgh.parallel_for<mmvq_multirow_kernel_name<GGML_TYPE_Q4_0>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_multirow<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                                       vec_dot_q4_0_q8_1_slm, NROWS_PER_WG>(
                    vx, vy, dst, ncols, nrows, item_ct1,
                    slm_y_qs.get_pointer(), slm_y_ds.get_pointer());
            });
    });
}

// MoE dispatch: Q4_0 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_q4_0_q8_1_id_sycl(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                          const int ncols, const int nrows_per_expert,
                                          const int total_batches, const int n_ids, const int ne11,
                                          const int64_t stride_expert_x,
                                          const int64_t ids_nb0, const int64_t ids_nb1,
                                          const int64_t nb11, const int64_t nb12,
                                          const int64_t nb1, const int64_t nb2,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // 3D dispatch: (1, total_batches, block_num_z) - batch in y dimension, rows in z
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_id_kernel_name<GGML_TYPE_Q4_0>>(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                 vx, vy, dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, stride_expert_x,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q4_1>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_MXFP4>>(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}


static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q5_0>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q5_1>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    // Use multi-row kernel with SLM Y-vector sharing for better bandwidth utilization
    constexpr int NROWS_PER_WG = MMVQ_NROWS_PER_WG;
    const int block_num_z = (nrows + NROWS_PER_WG - 1) / NROWS_PER_WG;
    const sycl::range<3> block_nums(1, 1, block_num_z);
    const sycl::range<3> block_dims(1, NROWS_PER_WG, WARP_SIZE);

    const int blocks_per_row = ncols / QK8_0;
    const int slm_y_qs_size = blocks_per_row * MMVQ_SLM_Y_QS_STRIDE;
    const int slm_y_ds_size = blocks_per_row + 1;  // +1 for padding

    stream->submit([&](sycl::handler & cgh) {
        // Allocate SLM for Y-vector (shared across all rows in work-group)
        sycl::local_accessor<int, 1> slm_y_qs(slm_y_qs_size, cgh);
        sycl::local_accessor<sycl::half2, 1> slm_y_ds(slm_y_ds_size, cgh);

        cgh.parallel_for<mmvq_multirow_kernel_name<GGML_TYPE_Q8_0>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_multirow<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                       vec_dot_q8_0_q8_1_slm, NROWS_PER_WG>(
                    vx, vy, dst, ncols, nrows, item_ct1,
                    slm_y_qs.get_pointer(), slm_y_ds.get_pointer());
            });
    });
}

// MoE dispatch: Q8_0 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_q8_0_q8_1_id_sycl(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                          const int ncols, const int nrows_per_expert,
                                          const int total_batches, const int n_ids, const int ne11,
                                          const int64_t stride_expert_x,
                                          const int64_t ids_nb0, const int64_t ids_nb1,
                                          const int64_t nb11, const int64_t nb12,
                                          const int64_t nb1, const int64_t nb2,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_id_kernel_name<GGML_TYPE_Q8_0>>(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                                 vx, vy, dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, stride_expert_x,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q2_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q3_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q4_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q4_K>>(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}


static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q5_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q6_K>>(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}
static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q6_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_IQ4_NL>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

// MoE-aware MXFP4 kernel: routes to different experts based on ids tensor
// Handles 2D iteration: (iid1, id) over tokens and expert selections
template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_mxfp4_q8_1_id(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                         const int32_t * __restrict__ ids, const int ncols, const int nrows_per_expert,
                                         const int n_ids, const int ne11,
                                         const int64_t stride_expert_x,
                                         const int64_t ids_nb0, const int64_t ids_nb1,
                                         const int64_t nb11, const int64_t nb12,
                                         const int64_t nb1, const int64_t nb2,
                                         const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - 2D iteration structure
    const int iid1 = batch_idx / n_ids;  // Token position
    const int id = batch_idx % n_ids;    // Expert selection index

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *)((const char*)ids + iid1 * ids_nb1 + id * ids_nb0);

    // Compute src1 and dst offsets matching host-side logic
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1 = id;
    const int64_t i2 = iid1;

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);

    float tmp = 0.0f;

    // Expert weights: offset by expert_id * stride_expert_x
    const block_q_t *  x = (const block_q_t *) ((const char*)vx + expert_id * stride_expert_x);
    // Input: offset using proper 2D indexing
    const block_q8_1 * y = (const block_q8_1 *) ((const char*)vy + i11 * nb11 + i12 * nb12);

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);
        const int iqs = vdr * (item_ct1.get_local_id(2) % (qi / vdr));

        tmp += vec_dot_mxfp4_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        // Output: offset using proper 2D indexing
        float * dst_out = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
        dst_out[row] = tmp;
    }
}

// MoE dispatch: MXFP4 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_mxfp4_q8_1_id_sycl(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                           const int ncols, const int nrows_per_expert,
                                           const int total_batches, const int n_ids, const int ne11,
                                           const int64_t stride_expert_x,
                                           const int64_t ids_nb0, const int64_t ids_nb1,
                                           const int64_t nb11, const int64_t nb12,
                                           const int64_t nb1, const int64_t nb2,
                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    // Use generic template with vec_dot_mxfp4_q8_1 function pointer
    // This matches how Q4_0 and Q8_0 work
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                 vx, vy, dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, stride_expert_x,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

// MoE dispatch: MXFP4 with SoA layout (reordered weights) + expert routing
// SoA layout: [all qs for all experts][all scales for all experts]
// After reorder_qw_mxfp4: qs at offset 0, scales at offset (ncols/2)*total_rows
static void mul_mat_vec_mxfp4_q8_1_soa_id_kernel(
    const uint8_t * __restrict__ vx,      // Base pointer to reordered tensor
    const block_q8_1 * __restrict__ vy,
    float * __restrict__ dst,
    const int32_t * __restrict__ ids,
    const int ncols,
    const int nrows_per_expert,
    const int n_ids,
    const int ne11,
    const int64_t total_qs_size,          // Offset to scale region
    const int64_t ids_nb0,
    const int64_t ids_nb1,
    const int64_t nb11,
    const int64_t nb12,
    const int64_t nb1,
    const int64_t nb2,
    const sycl::nd_item<3> & item_ct1)
{
    const int batch_idx = item_ct1.get_group(1);
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id)
    const int iid1 = batch_idx / n_ids;
    const int id = batch_idx % n_ids;

    // Read expert ID from ids tensor
    const int32_t expert_id = *(const int32_t *)((const char*)ids + iid1 * ids_nb1 + id * ids_nb0);

    // Compute src1 and dst offsets
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1 = id;
    const int64_t i2 = iid1;

    const int blocks_per_row = ncols / QK_MXFP4;

    // SoA layout: compute absolute row offset for this expert
    // total_rows = nrows_per_expert * num_experts, flattened during reorder
    const int64_t abs_row = expert_id * nrows_per_expert + row;
    const int64_t row_qs_offset = abs_row * blocks_per_row * (QK_MXFP4 / 2);  // 16 bytes per block
    const int64_t row_scale_offset = total_qs_size + abs_row * blocks_per_row;

    const uint8_t * qs_row = vx + row_qs_offset;
    const uint8_t * scale_row = vx + row_scale_offset;

    // Input: Q8_1 quantized
    const block_q8_1 * y = (const block_q8_1 *)((const char*)vy + i11 * nb11 + i12 * nb12);

    const int lane_id = item_ct1.get_local_id(2);
    constexpr int blocks_per_warp = WARP_SIZE;  // Each thread handles one block

    float acc = 0.0f;

    for (int b = lane_id; b < blocks_per_row; b += blocks_per_warp) {
        // Load E8M0 scale for this block from SoA scale region
        const uint8_t e8m0 = scale_row[b];
        const float d = ggml_sycl_e8m0_to_fp32(e8m0) * 0.5f;

        // Load Q8_1 block
        const block_q8_1 * q8_blk = &y[b];
        const float d8 = (*q8_blk).ds[0];

        // Load 16 packed bytes (32 4-bit values) from SoA qs region
        const uint8_t * qs = qs_row + b * (QK_MXFP4 / 2);

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < QK_MXFP4 / 2; i += 4) {
            // Load 4 packed bytes at once
            const int aux_q4 = *((const int*)(qs + i));
            const sycl::int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

            // DP4A: 4-way int8 dot product
            sumi = ggml_sycl_dp4a(v.x(), ((const int*)q8_blk->qs)[i/4], sumi);
            sumi = ggml_sycl_dp4a(v.y(), ((const int*)q8_blk->qs)[i/4 + 4], sumi);
        }

        acc += d * d8 * sumi;
    }

    // Warp reduction
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        acc += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), acc, mask);
    }

    if (lane_id == 0) {
        float * dst_out = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
        dst_out[row] = acc;
    }
}

// MoE dispatch: MXFP4 SoA layout with expert routing
static void reorder_mul_mat_vec_mxfp4_q8_1_id_sycl(
    const void * vx,
    const void * vy,
    float * dst,
    const int32_t * ids,
    const int ncols,
    const int nrows_per_expert,
    const int num_experts,
    const int total_batches,
    const int n_ids,
    const int ne11,
    const int64_t ids_nb0,
    const int64_t ids_nb1,
    const int64_t nb11,
    const int64_t nb12,
    const int64_t nb1,
    const int64_t nb2,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    // SoA layout: qs_size = (ncols / 2) * total_rows
    const int64_t total_rows = (int64_t)nrows_per_expert * num_experts;
    const int64_t total_qs_size = (ncols / 2) * total_rows;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_mxfp4_q8_1_soa_id_kernel(
                                 (const uint8_t *)vx,
                                 (const block_q8_1 *)vy,
                                 dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, total_qs_size,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2,
                                 item_ct1);
                         });
    });
}

#ifdef GGML_SYCL_GRAPH
// Pre-allocate Q8_1 buffers for all MUL_MAT_ID operations before graph recording.
// This must be called during decode phase, before graph recording starts.
// MUL_MAT_ID normally allocates Q8_1 buffers dynamically via ggml_sycl_pool_alloc,
// which is incompatible with SYCL graph recording.
void ggml_sycl_moe_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Skip if already initialized with sufficient buffers
    if (ctx.moe_buffers.initialized) {
        ctx.moe_buffers.reset_usage();
        return;
    }

    queue_ptr stream = ctx.stream();

    // Count MUL_MAT_ID nodes and find max dimensions
    int moe_count = 0;
    int64_t max_ne10 = 0;
    int64_t max_src1_rows = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];  // Expert weights
        const ggml_tensor * src1 = node->src[1];  // Input activations

        // Only count graph-compatible types
        if (!ggml_is_quantized(src0->type)) continue;
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4:
                break;
            default:
                continue;  // Skip unsupported types
        }

        moe_count++;

        // Track max dimensions
        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t ne12 = src1->ne[2];
        const int64_t total_rows = ne11 * ne12;

        if (ne10 > max_ne10) max_ne10 = ne10;
        if (total_rows > max_src1_rows) max_src1_rows = total_rows;
    }

    if (moe_count == 0) {
        return;  // No MoE operations
    }

    // Calculate buffer size (use max dimensions for all buffers)
    const int64_t ne10_padded = GGML_PAD(max_ne10, QK8_1);
    const int64_t q8_1_row_size = ne10_padded * sizeof(block_q8_1) / QK8_1;
    const size_t buffer_size = max_src1_rows * q8_1_row_size;

    GGML_SYCL_DEBUG("[MOE-GRAPH] Pre-allocating %d Q8_1 buffers, %zu bytes each (ne10=%lld, rows=%lld)\n",
                   moe_count, buffer_size, (long long)max_ne10, (long long)max_src1_rows);

    // Allocate buffers
    ctx.moe_buffers.q8_1_buffers.resize(moe_count);
    ctx.moe_buffers.q8_1_sizes.resize(moe_count);

    for (int i = 0; i < moe_count; i++) {
        ctx.moe_buffers.q8_1_buffers[i] = sycl::malloc_device(buffer_size, *stream);
        ctx.moe_buffers.q8_1_sizes[i] = buffer_size;

        if (!ctx.moe_buffers.q8_1_buffers[i]) {
            GGML_LOG_ERROR("[MOE-GRAPH] Failed to allocate Q8_1 buffer %d\n", i);
            // Cleanup and abort
            for (int j = 0; j < i; j++) {
                sycl::free(ctx.moe_buffers.q8_1_buffers[j], *stream);
            }
            ctx.moe_buffers.q8_1_buffers.clear();
            ctx.moe_buffers.q8_1_sizes.clear();
            return;
        }
    }

    ctx.moe_buffers.max_ne10 = max_ne10;
    ctx.moe_buffers.max_src1_rows = max_src1_rows;
    ctx.moe_buffers.initialized = true;
    ctx.moe_buffers.reset_usage();

    // Wait for allocations to complete
    stream->wait();

    GGML_SYCL_DEBUG("[MOE-GRAPH] Pre-allocated %d buffers successfully\n", moe_count);
}
#endif

// MoE-aware MUL_MAT_ID dispatch: GPU-side expert routing without host sync
// This allows SYCL graph recording to work with MoE models
// Returns true if handled, false to fall back to host-side routing
bool ggml_sycl_mul_mat_id_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, const ggml_tensor *ids, ggml_tensor *dst) {

    GGML_TENSOR_BINARY_OP_LOCALS;

    // Supports both ne12 == 1 (decode) and ne12 > 1 (prompt)
    // The kernel dispatches over (iid1, id) pairs in parallel

    // Only handle quantized types that have _id kernels
    if (!ggml_is_quantized(src0->type)) {
        return false;
    }

    // Check for supported types
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return false;  // Fall back to host routing for other types
    }

    const queue_ptr stream = ctx.stream();

    // Quantize src1 to Q8_1 format - need to handle all rows (ne11 * ne12)
    const int64_t ne10_padded = GGML_PAD(ne10, QK8_1);
    const int64_t q8_1_row_size = ne10_padded * sizeof(block_q8_1) / QK8_1;

    // Total rows = ne11 * ne12 (e.g., 4 expert outputs * 1 token = 4 rows)
    const int64_t total_src1_rows = ne11 * ne12;
    const size_t required_size = total_src1_rows * q8_1_row_size;

    // Get pointers early for cache check
    const void * src0_d = src0->data;
    const float * src1_d = (const float *)src1->data;
    const int32_t * ids_d = (const int32_t *)ids->data;
    float * dst_d = (float *)dst->data;

    // Check Q8_1 quantization cache - MoE uses same input for gate/up/down projections
    void* q8_1_buffer = nullptr;
    bool using_cached = false;
    bool using_preallocated = false;

#ifdef GGML_SYCL_GRAPH
    // Check cache first - avoids re-quantizing same input across MoE projections
    if (ctx.moe_q8_cache.matches(src1_d, ne10, total_src1_rows)) {
        q8_1_buffer = ctx.moe_q8_cache.cached_q8_1;
        using_cached = true;
        GGML_SYCL_DEBUG("[MOE-CACHE] Cache HIT: src1=%p, ne10=%lld, rows=%lld\n",
                       src1_d, (long long)ne10, (long long)total_src1_rows);
    }
#endif

    // Fall back to allocation + quantization if cache miss
    ggml_sycl_pool_alloc<int8_t> src1_q8_1_pool(ctx.pool());
    if (!using_cached) {
#ifdef GGML_SYCL_GRAPH
        // Try pre-allocated buffer for graph recording
        if (g_ggml_sycl_graph_recording && ctx.moe_buffers.initialized) {
            q8_1_buffer = ctx.moe_buffers.get_next_buffer(required_size);
            if (q8_1_buffer) {
                using_preallocated = true;
                GGML_SYCL_DEBUG("[MOE-GRAPH] Using pre-allocated buffer %d\n",
                               ctx.moe_buffers.current_buffer_idx - 1);
            }
        }
#endif

        // Fall back to pool allocation if no pre-allocated buffer available
        if (!using_preallocated) {
            src1_q8_1_pool.alloc(required_size);
            q8_1_buffer = src1_q8_1_pool.get();
        }

        // Quantize all rows to Q8_1
        quantize_row_q8_1_sycl<quantize_q8_1>(src1_d, (char*)q8_1_buffer, ne10, total_src1_rows, ne10_padded, stream);

#ifdef GGML_SYCL_GRAPH
        // Cache the quantized result for subsequent gate/up/down calls
        // Only cache if using pre-allocated buffer (pool buffers get freed)
        if (using_preallocated) {
            ctx.moe_q8_cache.cached_q8_1 = q8_1_buffer;
            ctx.moe_q8_cache.cached_src = src1_d;
            ctx.moe_q8_cache.cached_ne10 = ne10;
            ctx.moe_q8_cache.cached_rows = total_src1_rows;
            ctx.moe_q8_cache.cached_size = required_size;
            ctx.moe_q8_cache.valid = true;
            GGML_SYCL_DEBUG("[MOE-CACHE] Cache STORE: src1=%p, ne10=%lld, rows=%lld, buffer=%p\n",
                           src1_d, (long long)ne10, (long long)total_src1_rows, q8_1_buffer);
        }
#endif
    }

    // Calculate strides from tensors (matching host-side logic)
    const int64_t n_ids = ids->ne[0];  // Number of expert selections per token
    const int64_t num_tokens = ids->ne[1];  // Number of tokens (should equal ne12)

    // Total batches = tokens * expert_selections_per_token
    const int64_t total_batches = num_tokens * n_ids;

    // stride_expert_x: offset between expert weight matrices
    const int64_t stride_expert_x = nb02;

    // Quantized src1 strides (map to Q8_1 layout)
    // For Q8_1: each row is q8_1_row_size bytes
    // q8_nb11: stride between consecutive rows (dimension 1)
    // q8_nb12: stride between tokens (dimension 2) = ne11 rows per token
    const int64_t q8_nb11 = q8_1_row_size;
    const int64_t q8_nb12 = ne11 * q8_1_row_size;

    // Dispatch based on type
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_id_sycl(
                src0_d, q8_1_buffer, dst_d, ids_d,
                ne00,  // ncols
                ne01,  // nrows_per_expert
                total_batches,
                n_ids,
                ne11,
                stride_expert_x,
                ids->nb[0], ids->nb[1],  // ids strides
                q8_nb11, q8_nb12,        // Q8_1 strides
                nb1, nb2,                // dst strides
                stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_id_sycl(
                src0_d, q8_1_buffer, dst_d, ids_d,
                ne00,  // ncols
                ne01,  // nrows_per_expert
                total_batches,
                n_ids,
                ne11,
                stride_expert_x,
                ids->nb[0], ids->nb[1],  // ids strides
                q8_nb11, q8_nb12,        // Q8_1 strides
                nb1, nb2,                // dst strides
                stream);
            break;
        case GGML_TYPE_MXFP4:
            {
                // Check if weights are reordered to SoA layout
                auto * extra = (ggml_tensor_extra_gpu *)src0->extra;
                bool use_soa = extra && extra->optimized_feature.reorder;

                GGML_SYCL_DEBUG("[MMVQ-MXFP4] use_soa=%d extra=%p\n", use_soa, extra);

                if (use_soa) {
                    // SoA layout - use reordered kernel
                    reorder_mul_mat_vec_mxfp4_q8_1_id_sycl(
                        src0_d, q8_1_buffer, dst_d, ids_d,
                        ne00,  // ncols
                        ne01,  // nrows_per_expert
                        ne02,  // num_experts
                        total_batches,
                        n_ids,
                        ne11,
                        ids->nb[0], ids->nb[1],  // ids strides
                        q8_nb11, q8_nb12,        // Q8_1 strides
                        nb1, nb2,                // dst strides
                        stream);
                } else {
                    // Original AoS layout
                    mul_mat_vec_mxfp4_q8_1_id_sycl(
                        src0_d, q8_1_buffer, dst_d, ids_d,
                        ne00,  // ncols
                        ne01,  // nrows_per_expert
                        total_batches,
                        n_ids,
                        ne11,
                        stride_expert_x,
                        ids->nb[0], ids->nb[1],  // ids strides
                        q8_nb11, q8_nb12,        // Q8_1 strides
                        nb1, nb2,                // dst strides
                        stream);
                }
            }
            break;
        default:
            GGML_ABORT("Unsupported type for MoE GPU dispatch");
    }

    return true;
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    GGML_SYCL_PROFILE_SCOPE_MMVQ("mmvq");
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    // DEBUG: Check dimensions and src1 values for layer 0 projections
    static int mmvq_dbg = 0;
    static int mmvq_b1_dbg = 0;
    static int mmvq_wq_dbg = 0;
    bool is_attn_out_l0 = src0->name && strstr(src0->name, "blk.0.attn_output");
    bool is_wq_l0 = src0->name && strstr(src0->name, "blk.0.attn_q");
    bool do_mmvq_dbg = (mmvq_dbg++ < 5 && is_attn_out_l0) || (src1_ncols == 1 && is_attn_out_l0 && mmvq_b1_dbg++ < 5);

    // DEBUG wq projection for batch=1 NaN issue
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    if (g_ggml_sycl_tp_debug && is_wq_l0 && src1_ncols == 1 && mmvq_wq_dbg++ < 5) {
        fprintf(stderr, "TP DEBUG MMVQ wq layer0 batch=1: ne00=%lld, row_diff=%lld, src1_ddf_i=%p, src1_ddq_i=%p\n",
                (long long)ne00, (long long)row_diff, (void*)src1_ddf_i, (void*)src1_ddq_i);

        // Check input values (RMS norm output, quantized as Q8_1)
        struct block_q8_1_debug {
            sycl::half d;
            sycl::half s;
            int8_t qs[32];
        };
        block_q8_1_debug blk;
        stream->memcpy(&blk, src1_ddq_i, sizeof(blk)).wait();
        float d_f = static_cast<float>(blk.d);
        fprintf(stderr, "TP DEBUG MMVQ wq input (norm output): d=%f, qs[0..3]=[%d,%d,%d,%d]\n",
                d_f, blk.qs[0], blk.qs[1], blk.qs[2], blk.qs[3]);

        // Check for NaN/inf in d
        if (d_f != d_f || d_f == std::numeric_limits<float>::infinity() || d_f == -std::numeric_limits<float>::infinity()) {
            fprintf(stderr, "TP DEBUG MMVQ wq input: INVALID d value detected!\n");
        }
    }

    // DEBUG: Layer 30 and 31 FFN gate Q8_1 quantized input check
    static int mmvq_l30_gate_dbg = 0;
    static int mmvq_l31_gate_dbg = 0;
    bool is_ffn_gate_l30 = src0->name && strstr(src0->name, "blk.30.ffn_gate");
    bool is_ffn_gate_l31 = src0->name && strstr(src0->name, "blk.31.ffn_gate");

    // Check layer 30 first to compare
    if (g_ggml_sycl_tp_debug && is_ffn_gate_l30 && src1_ncols == 1 && mmvq_l30_gate_dbg++ < 3) {
        struct block_q4_0_l30 {
            sycl::half d;
            uint8_t qs[16];
        };
        block_q4_0_l30 wblk;
        stream->memcpy(&wblk, src0_dd_i, sizeof(wblk)).wait();
        float wd = static_cast<float>(wblk.d);
        int w0 = (wblk.qs[0] & 0xF) - 8;
        int w1 = (wblk.qs[0] >> 4) - 8;
        fprintf(stderr, "TP DEBUG MMVQ L30 Q4_0 weight: ptr=%p d=%f raw[0..1]=[%d,%d] -> deq=[%f,%f]\n",
                (void*)src0_dd_i, wd, w0, w1, w0 * wd, w1 * wd);
    }
    if (g_ggml_sycl_tp_debug && is_ffn_gate_l31 && src1_ncols == 1 && mmvq_l31_gate_dbg++ < 5) {
        fprintf(stderr, "TP DEBUG MMVQ L31 FFN_GATE batch=1: ne00=%lld (K), row_diff=%lld (N), src1_ddq_i=%p\n",
                (long long)ne00, (long long)row_diff, (void*)src1_ddq_i);

        // Read first Q8_1 block (ffn_norm output quantized)
        struct block_q8_1_l31 {
            sycl::half d;
            sycl::half s;
            int8_t qs[32];
        };
        block_q8_1_l31 blk;
        stream->memcpy(&blk, src1_ddq_i, sizeof(blk)).wait();
        float d_f = static_cast<float>(blk.d);
        float s_f = static_cast<float>(blk.s);
        float v0 = blk.qs[0] * d_f;
        float v1 = blk.qs[1] * d_f;
        float v2 = blk.qs[2] * d_f;
        float v3 = blk.qs[3] * d_f;

        bool d_invalid = std::isnan(d_f) || std::isinf(d_f);
        fprintf(stderr, "TP DEBUG MMVQ L31 Q8_1 input: d=%f (%s), s=%f, qs[0..3]=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d_f, d_invalid ? "INVALID" : "ok", s_f,
                blk.qs[0], blk.qs[1], blk.qs[2], blk.qs[3], v0, v1, v2, v3);

        // Also check weight Q4_0 block at layer 31
        struct block_q4_0_l31 {
            sycl::half d;
            uint8_t qs[16];  // 32 x 4-bit
        };
        block_q4_0_l31 wblk;
        stream->memcpy(&wblk, src0_dd_i, sizeof(wblk)).wait();
        float wd = static_cast<float>(wblk.d);
        int w0 = (wblk.qs[0] & 0xF) - 8;
        int w1 = (wblk.qs[0] >> 4) - 8;
        bool wd_invalid = std::isnan(wd) || std::isinf(wd);
        fprintf(stderr, "TP DEBUG MMVQ L31 Q4_0 weight: d=%f (%s), raw[0..1]=[%d,%d] -> deq=[%f,%f]\n",
                wd, wd_invalid ? "INVALID" : "ok", w0, w1, w0 * wd, w1 * wd);

        // DEBUG: Check if dst pointer overlaps with weight
        uintptr_t weight_start = (uintptr_t)src0_dd_i;
        uintptr_t weight_end = weight_start + (ne00 / 32) * 18 * row_diff;  // Q4_0 block size
        uintptr_t dst_start = (uintptr_t)dst_dd_i;
        uintptr_t dst_end = dst_start + row_diff * src1_ncols * sizeof(float);
        bool overlap = (dst_start < weight_end && dst_end > weight_start);
        fprintf(stderr, "TP DEBUG MMVQ L31 ptrs: weight=[%p,%p), dst=[%p,%p), overlap=%d\n",
                (void*)weight_start, (void*)weight_end, (void*)dst_start, (void*)dst_end, overlap);
    }
    if (g_ggml_sycl_tp_debug && do_mmvq_dbg) {
        fprintf(stderr, "TP DEBUG MMVQ %s: ne00=%lld (K), ne10=%lld (src1_K), row_diff=%lld (N), src1_ncols=%lld, padded=%lld, src1_ddq_i=%p\n",
                src0->name, (long long)ne00, (long long)ne10, (long long)row_diff,
                (long long)src1_ncols, (long long)src1_padded_col_size, (void*)src1_ddq_i);

        // Read and dequantize first Q8_1 block to check input values
        // Q8_1 block: ggml_half d (2 bytes), ggml_half s (2 bytes), int8_t qs[32]
        struct block_q8_1_debug {
            sycl::half d;
            sycl::half s;
            int8_t qs[32];
        };
        block_q8_1_debug blk;
        stream->memcpy(&blk, src1_ddq_i, sizeof(blk)).wait();
        // Dequantize first 4 values: val[i] = qs[i] * d
        float d_f = static_cast<float>(blk.d);
        float s_f = static_cast<float>(blk.s);
        float v0 = blk.qs[0] * d_f;
        float v1 = blk.qs[1] * d_f;
        float v2 = blk.qs[2] * d_f;
        float v3 = blk.qs[3] * d_f;
        fprintf(stderr, "TP DEBUG MMVQ src1 (attn_out quantized): d=%f, s=%f, qs[0..3]=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d_f, s_f, blk.qs[0], blk.qs[1], blk.qs[2], blk.qs[3], v0, v1, v2, v3);

        // Also read some middle values from quantized input (block 32 = position 1024-1055)
        block_q8_1_debug blk_mid;
        stream->memcpy(&blk_mid, src1_ddq_i + 32*sizeof(blk_mid), sizeof(blk_mid)).wait();
        float d_mid = static_cast<float>(blk_mid.d);
        fprintf(stderr, "TP DEBUG MMVQ src1 middle (pos 1024): d=%f, qs[0..3]=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d_mid, blk_mid.qs[0], blk_mid.qs[1], blk_mid.qs[2], blk_mid.qs[3],
                blk_mid.qs[0]*d_mid, blk_mid.qs[1]*d_mid, blk_mid.qs[2]*d_mid, blk_mid.qs[3]*d_mid);
    }

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));

    // DEBUG: Also print device ID and weight data for layer 0 attn_output
    static int mmvq_dev_dbg = 0;
    if (g_ggml_sycl_tp_debug && mmvq_dev_dbg++ < 10 && src0->name && strstr(src0->name, "blk.0.attn_output")) {
        fprintf(stderr, "TP DEBUG MMVQ device=%d for %s, src0_dd_i=%p, ne00=%lld, row_diff=%lld\n",
                id, src0->name, (void*)src0_dd_i, (long long)ne00, (long long)row_diff);

        // Read first Q4_0 block from weight (output row 0, first 32 input elements)
        struct block_q4_0_debug {
            sycl::half d;
            uint8_t qs[16];  // 32 x 4-bit values packed
        };
        block_q4_0_debug w_blk0, w_blk64;
        stream->memcpy(&w_blk0, src0_dd_i, sizeof(w_blk0)).wait();
        // Also read block 64 (second row of output, position [1,0]) for TP mode
        // Block index 64 = ne00/32 blocks per row = 64 for TP, 128 for single
        size_t blocks_per_row = ne00 / 32;
        stream->memcpy(&w_blk64, src0_dd_i + blocks_per_row * sizeof(w_blk0), sizeof(w_blk64)).wait();

        float d0 = static_cast<float>(w_blk0.d);
        float d64 = static_cast<float>(w_blk64.d);
        // Dequantize first 4 values: val[i] = (qs[i/2] >> (4*(i%2)) & 0xF) - 8) * d
        int v0 = (w_blk0.qs[0] & 0xF) - 8;
        int v1 = (w_blk0.qs[0] >> 4) - 8;
        int v2 = (w_blk0.qs[1] & 0xF) - 8;
        int v3 = (w_blk0.qs[1] >> 4) - 8;
        fprintf(stderr, "TP DEBUG MMVQ weight row0 blk0: d=%f, qs[0-1]=0x%02x%02x, vals=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d0, w_blk0.qs[0], w_blk0.qs[1], v0, v1, v2, v3, v0*d0, v1*d0, v2*d0, v3*d0);

        int v0_64 = (w_blk64.qs[0] & 0xF) - 8;
        int v1_64 = (w_blk64.qs[0] >> 4) - 8;
        fprintf(stderr, "TP DEBUG MMVQ weight row1 blk0: d=%f, qs[0]=0x%02x, vals=[%d,%d] -> deq=[%f,%f]\n",
                d64, w_blk64.qs[0], v0_64, v1_64, v0_64*d64, v1_64*d64);
    }

    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                {
                    // Use src0->extra for reorder check since src0 is the actual weight tensor being used
                    // This is important for TP where we may pass different tensors than dst->src[0]
                    auto * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
                    bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

                    // For TP: DISABLE reorder completely to rule out issues
                    if (src0_extra && src0_extra->tp_sharded) {
                        use_reorder = false;
                    }

                    // Check if tensor was converted to coalesced layout at model load time
                    // (conversion happens in ggml_sycl_reorder_weights when GGML_SYCL_MMVQ_COALESCED is set)
                    bool use_coalesced = src0_extra && src0_extra->optimized_feature.coalesced;

                    if (use_coalesced) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_q4_0_q8_1_sycl\n");
                        coalesced_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else if (use_reorder) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else {
                        GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                        mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q8_0:
                {
                    auto * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
                    bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;
                    bool use_coalesced = src0_extra && src0_extra->optimized_feature.coalesced;

                    // Disable optimized paths for TP-sharded tensors
                    if (src0_extra && src0_extra->tp_sharded) {
                        use_reorder = false;
                        use_coalesced = false;
                    }

                    if (use_coalesced) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_q8_0_q8_1_sycl\n");
                        coalesced_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else if (use_reorder) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else {
                        mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_MXFP4:
                {
                    auto * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
                    bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;
                    bool use_coalesced = src0_extra && src0_extra->optimized_feature.coalesced;

                    // Disable optimized paths for TP-sharded tensors
                    if (src0_extra && src0_extra->tp_sharded) {
                        use_reorder = false;
                        use_coalesced = false;
                    }

                    if (use_coalesced) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        coalesced_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else if (use_reorder) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else {
                        mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_Q2_K:
                mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q3_K:
                mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }

        // DEBUG: Check output values for attn_output after kernel (layer 0 only)
        // Controlled by GGML_SYCL_TP_DEBUG environment variable
        static int mmvq_out_dbg = 0;
        if (g_ggml_sycl_tp_debug && mmvq_out_dbg++ < 10 && src0->name && strstr(src0->name, "blk.0.attn_output")) {
            stream->wait();  // Wait for kernel to complete
            float out_vals[8];
            stream->memcpy(out_vals, dst_dd_i_bs, 8*sizeof(float)).wait();
            fprintf(stderr, "TP DEBUG MMVQ output device=%d %s col=%d dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                    id, src0->name, i, out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                    out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
        }

        // DEBUG: Check FFN down output for layer 0 (works in both TP and single GPU)
        // Controlled by GGML_SYCL_TP_DEBUG environment variable
        static int ffn_down_dbg = 0;
        if (g_ggml_sycl_tp_debug && ffn_down_dbg++ < 5 && src0->name && strstr(src0->name, "blk.0.ffn_down")) {
            stream->wait();
            float out_vals[8];
            stream->memcpy(out_vals, dst_dd_i_bs, 8*sizeof(float)).wait();
            fprintf(stderr, "DEBUG FFN_DOWN layer0 device=%d col=%d ne00=%lld row_diff=%lld dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                    id, (int)i, (long long)ne00, (long long)row_diff, out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                    out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
