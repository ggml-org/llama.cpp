//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "ggml-impl.h"
#include "common.hpp"
#include "dequantize.hpp"
#include "getrows.hpp"


template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

// SoA (Structure of Arrays) version of k_get_rows for reordered quantized tensors
// In SoA layout: all qs bytes come first, then all d (scale) bytes
// This requires computing separate pointers for qs and d data
template<int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder, typename dst_t>
static void k_get_rows_reorder(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            int64_t d_offset,
            const sycl::nd_item<3> &item_ct1) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];  // Row index in source tensor

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    // For SoA layout, compute block index within the full tensor
    const int64_t blocks_per_row = ne00 / qk;
    const int ib_local = i00/qk;  // block index within row
    const int64_t ib_global = i01 * blocks_per_row + ib_local;  // global block index

    const int iqs = (i00%qk)/qr;  // quant index within block
    const int iybs = i00 - i00%qk;  // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // In SoA layout:
    // - qs data is contiguous from start: offset = row * bytes_per_row + block_offset + elem_offset
    // - d data starts at d_offset: offset = d_offset + global_block_index * sizeof(half)
    // Byte sizes depend on quantization type:
    //   Q4_0 (qr=2): 4 bits/elem → ne00/2 bytes/row, qk/2 bytes/block
    //   Q8_0 (qr=1): 8 bits/elem → ne00 bytes/row, qk bytes/block
    const size_t bytes_per_row = ne00 / qr;
    const size_t bytes_per_block = qk / qr;
    const void * d_ptr = (const char *)src0 + d_offset;
    const void * qs_ptr = (const char *)src0 + i01 * bytes_per_row + ib_local * bytes_per_block + iqs;

    // dequantize using SoA kernel
    dfloat2 v;
    dequantize_kernel_reorder(d_ptr, ib_global, qs_ptr, 0, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

// Specialized Q6_K AoS kernel for GET_ROWS (standard block layout)
template<typename dst_t>
static void k_get_rows_q6_k_aos(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {

    // Each thread processes 4 values (Q6_K block structure)
    // Thread layout: tid = ip * 32 + il, where ip in {0,1}, il in {0..31}
    const int tid = item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    // Block index within row (each block has QK_K=256 elements)
    const int block_in_row = item_ct1.get_group(2);
    const int64_t blocks_per_row = ne00 / QK_K;

    if (block_in_row >= blocks_per_row) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];  // Row index
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    // Get pointer to the Q6_K block (AoS layout)
    const block_q6_K * x = (const block_q6_K *)((const char *)src0 + i01 * nb01 + i11 * nb02 + i12 * nb03);
    const block_q6_K * bx = x + block_in_row;

    // Thread decomposition: ip (0 or 1), il (0..31)
    const int ip = tid / 32;
    const int il = tid % 32;
    const int is = 8 * ip + il / 16;

    // Destination position for this thread's 4 values
    dst_t * y = dst_row + block_in_row * QK_K + 128 * ip + il;

    // Read data for this thread from AoS block
    const uint8_t * ql = bx->ql + 64 * ip + il;
    const uint8_t qh = bx->qh[32 * ip + il];
    const int8_t * sc = bx->scales + is;
    const float d = static_cast<float>(bx->d);

    // Dequantize 4 values
    y[0]  = d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K AoS GET_ROWS
static void get_rows_q6_k_aos_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K uses 64 threads (2 phases * 32 threads) per block
    const sycl::range<3> block_dims(1, 1, 64);
    // One work-group per Q6_K block in each row
    const int64_t blocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, blocks_per_row);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_q6_k_aos<float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Specialized Q6_K SoA kernel for GET_ROWS
// Q6_K has 4 sections: [all ql (n*128)][all qh (n*64)][all scales (n*16)][all d (n*2)]
template<typename dst_t>
static void k_get_rows_q6_k_soa(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {

    // Each thread processes 4 values (Q6_K block structure)
    // Thread layout: tid = ip * 32 + il, where ip in {0,1}, il in {0..31}
    const int tid = item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    // Block index within row (each block has QK_K=256 elements)
    const int block_in_row = item_ct1.get_group(2);
    const int64_t blocks_per_row = ne00 / QK_K;

    if (block_in_row >= blocks_per_row) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];  // Row index
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    // Global block index for SoA offset calculation
    const int64_t n_blocks = ne01 * blocks_per_row;  // Total blocks in tensor
    const int64_t ib_global = i01 * blocks_per_row + block_in_row;

    // Thread decomposition: ip (0 or 1), il (0..31)
    const int ip = tid / 32;
    const int il = tid % 32;
    const int is = 8 * ip + il / 16;

    // SoA layout offsets
    const uint8_t * base_ptr = static_cast<const uint8_t *>(src0);
    const int64_t ql_offset = ib_global * (QK_K / 2);  // 128 bytes per block
    const int64_t qh_offset = (QK_K / 2) * n_blocks + (QK_K / 4) * ib_global;
    const int64_t scales_offset = (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks + (QK_K / 16) * ib_global;
    const int64_t d_offset = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks;

    const uint8_t * ql_ptr = base_ptr + ql_offset;
    const uint8_t * qh_ptr = base_ptr + qh_offset;
    const uint8_t * scales_ptr = base_ptr + scales_offset;
    const sycl::half * d_ptr = (const sycl::half *)(base_ptr + d_offset) + ib_global;

    // Destination position for this thread's 4 values
    dst_t * y = dst_row + block_in_row * QK_K + 128 * ip + il;

    // Read data for this thread
    const uint8_t * ql = ql_ptr + 64 * ip + il;
    const uint8_t qh = *(qh_ptr + 32 * ip + il);
    const int8_t * sc = reinterpret_cast<const int8_t *>(scales_ptr + is);
    const float d = *d_ptr;

    // Dequantize 4 values
    y[0]  = d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K SoA GET_ROWS
static void get_rows_q6_k_soa_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K uses 64 threads (2 phases * 32 threads) per block
    const sycl::range<3> block_dims(1, 1, 64);
    // One work-group per Q6_K block in each row
    const int64_t blocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, blocks_per_row);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_q6_k_soa<float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Q6_K Coalesced layout kernel for GET_ROWS
// Coalesced layout: word-major within tiles for each component
//   - Tile structure: [ql_word_major...][qh_word_major...][scales_word_major...]
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - D values follow all quant tiles contiguously
template<typename dst_t>
static void k_get_rows_q6_k_coalesced(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            int64_t d_offset,
            const sycl::nd_item<3> &item_ct1) {

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    // Thread layout: same as Q6_K SoA (64 threads: 2 phases × 32 threads)
    const int tid = item_ct1.get_local_id(2);
    const int ip = tid / 32;
    const int il = tid % 32;
    const int is = 8 * ip + il / 16;

    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    // Block index within row (each block has QK_K=256 elements)
    const int block_in_row = item_ct1.get_group(2);
    const int64_t blocks_per_row = ne00 / QK_K;

    if (block_in_row >= blocks_per_row) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];  // Row index
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    // Global block index
    const int64_t ib_global = i01 * blocks_per_row + block_in_row;

    // Coalesced tile calculations
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int tile = block_in_row / TILE_BLOCKS;
    const int block_in_tile = block_in_row % TILE_BLOCKS;

    // Tile sizes
    constexpr int ql_tile_bytes = TILE_BLOCKS * (QK_K / 2);   // 2048
    constexpr int qh_tile_bytes = TILE_BLOCKS * (QK_K / 4);   // 1024
    constexpr int sc_tile_bytes = TILE_BLOCKS * (QK_K / 16);  // 256
    constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;

    // Base pointer
    const uint8_t * base_ptr = static_cast<const uint8_t *>(src0);
    const sycl::half * d_ptr = (const sycl::half *)(base_ptr + d_offset) + ib_global;

    // Destination position
    dst_t * y = dst_row + block_in_row * QK_K + 128 * ip + il;

    // Tile base for this row/tile
    const int64_t tile_row_base = i01 * tiles_per_row * tile_total;
    const int64_t ql_tile_base = tile_row_base + tile * tile_total;
    const int64_t qh_tile_base = ql_tile_base + ql_tile_bytes;
    const int64_t sc_tile_base = qh_tile_base + qh_tile_bytes;

    // === Read two ql bytes ===
    const int ql_pos0 = 64 * ip + il;
    const int ql_pos1 = ql_pos0 + 32;
    const int ql_word0 = ql_pos0 / 4;
    const int ql_byte0 = ql_pos0 % 4;
    const int ql_word1 = ql_pos1 / 4;
    const int ql_byte1 = ql_pos1 % 4;
    const int64_t ql_offset0 = ql_tile_base + ql_word0 * WORD_PLANE_STRIDE + block_in_tile * 4 + ql_byte0;
    const int64_t ql_offset1 = ql_tile_base + ql_word1 * WORD_PLANE_STRIDE + block_in_tile * 4 + ql_byte1;
    const uint8_t ql0 = base_ptr[ql_offset0];
    const uint8_t ql1 = base_ptr[ql_offset1];

    // === Read qh byte ===
    const int qh_pos = 32 * ip + il;
    const int qh_word = qh_pos / 4;
    const int qh_byte = qh_pos % 4;
    const int64_t qh_offset = qh_tile_base + qh_word * WORD_PLANE_STRIDE + block_in_tile * 4 + qh_byte;
    const uint8_t qh = base_ptr[qh_offset];

    // === Read scales ===
    const int sc_idx0 = is + 0;
    const int sc_idx2 = is + 2;
    const int sc_idx4 = is + 4;
    const int sc_idx6 = is + 6;
    const int sc_word0 = sc_idx0 / 4;
    const int sc_byte0 = sc_idx0 % 4;
    const int sc_word2 = sc_idx2 / 4;
    const int sc_byte2 = sc_idx2 % 4;
    const int sc_word4 = sc_idx4 / 4;
    const int sc_byte4 = sc_idx4 % 4;
    const int sc_word6 = sc_idx6 / 4;
    const int sc_byte6 = sc_idx6 % 4;
    const int64_t sc_offset0 = sc_tile_base + sc_word0 * WORD_PLANE_STRIDE + block_in_tile * 4 + sc_byte0;
    const int64_t sc_offset2 = sc_tile_base + sc_word2 * WORD_PLANE_STRIDE + block_in_tile * 4 + sc_byte2;
    const int64_t sc_offset4 = sc_tile_base + sc_word4 * WORD_PLANE_STRIDE + block_in_tile * 4 + sc_byte4;
    const int64_t sc_offset6 = sc_tile_base + sc_word6 * WORD_PLANE_STRIDE + block_in_tile * 4 + sc_byte6;
    const int8_t sc0 = static_cast<int8_t>(base_ptr[sc_offset0]);
    const int8_t sc2 = static_cast<int8_t>(base_ptr[sc_offset2]);
    const int8_t sc4 = static_cast<int8_t>(base_ptr[sc_offset4]);
    const int8_t sc6 = static_cast<int8_t>(base_ptr[sc_offset6]);
    const float d = *d_ptr;

    // Dequantize (same logic as SoA)
    y[0]  = d * sc0 * ((int8_t)((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc2 * ((int8_t)((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc4 * ((int8_t)((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc6 * ((int8_t)((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K Coalesced GET_ROWS
static void get_rows_q6_k_coalesced_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          int64_t d_offset,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K uses 64 threads (2 phases × 32 threads) per block
    const sycl::range<3> block_dims(1, 1, 64);
    const int64_t blocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, blocks_per_row);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_q6_k_coalesced<float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, d_offset, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Q4_0 Coalesced layout kernel for GET_ROWS
// Coalesced layout: word-major within tiles
//   - For word w of block b in tile: offset = tile_base + w*stride + b*4
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - Scales (d values) follow all quant tiles contiguously
template<typename dst_t>
static void k_get_rows_q4_0_coalesced(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            int64_t d_offset,
            const sycl::nd_item<3> &item_ct1) {

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;  // 16 bytes of quants per block
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;  // bytes between word planes

    // Each thread dequantizes 2 values (like k_get_rows)
    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];  // Row index in source tensor

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    // Calculate block/element positions
    const int64_t blocks_per_row = ne00 / QK4_0;
    const int ib_local = i00 / QK4_0;  // Block index within row
    const int64_t ib_global = i01 * blocks_per_row + ib_local;  // Global block index

    const int iqs = (i00 % QK4_0) / 2;  // Quant index within block (0-15)
    const int iybs = i00 - i00 % QK4_0;  // Dst block start index
    const int y_offset = QK4_0 / 2;  // 16

    // Calculate coalesced layout positions
    const int64_t tile = ib_local / TILE_BLOCKS;
    const int block_in_tile = ib_local % TILE_BLOCKS;

    // Which word contains our quant byte?
    const int word_idx = iqs / 4;  // Which of 4 words (0-3)
    const int byte_in_word = iqs % 4;  // Byte within word (0-3)

    // Row's quant section base
    const int64_t row_quants_bytes = ne00 / 2;  // bytes per row of quants
    const int64_t row_base = i01 * row_quants_bytes;

    // Tile base within row
    const int64_t tile_base = row_base + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

    // Coalesced offset: word-major within tile
    const int64_t qs_offset = tile_base + word_idx * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word;

    // Read the quant byte from coalesced layout
    const uint8_t * qs_ptr = (const uint8_t *)src0 + qs_offset;
    const uint8_t qs_byte = *qs_ptr;

    // Read scale from contiguous d array after all quants
    const sycl::half * d_ptr = (const sycl::half *)((const char *)src0 + d_offset) + ib_global;
    const float d = (float)(*d_ptr);

    // Dequantize: Q4_0 stores 2 values per byte (low 4 bits, high 4 bits)
    const int vl = (qs_byte & 0xF) - 8;
    const int vh = (qs_byte >> 4) - 8;

    dst_row[iybs + iqs + 0] = d * vl;
    dst_row[iybs + iqs + y_offset] = d * vh;
}

// Dispatch function for Q4_0 Coalesced GET_ROWS
static void get_rows_q4_0_coalesced_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          int64_t d_offset,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_q4_0_coalesced<float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, d_offset, item_ct1);
                         });

    // DEBUG: avoid queue waits during graph recording

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Q8_0 Coalesced layout kernel for GET_ROWS
// Coalesced layout: word-major within tiles
//   - For word w of block b in tile: offset = tile_base + w*stride + b*4
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - Scales (d values) follow all quant tiles contiguously
template<typename dst_t>
static void k_get_rows_q8_0_coalesced(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            int64_t d_offset,
            const sycl::nd_item<3> &item_ct1) {

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const int64_t blocks_per_row = ne00 / QK8_0;
    const int ib_local = i00 / QK8_0;
    const int64_t ib_global = i01 * blocks_per_row + ib_local;

    const int iqs = i00 % QK8_0;
    const int iybs = i00 - i00 % QK8_0;

    const int64_t tile = ib_local / TILE_BLOCKS;
    const int block_in_tile = ib_local % TILE_BLOCKS;

    const int word_idx0 = iqs / 4;
    const int byte_in_word0 = iqs % 4;
    const int word_idx1 = (iqs + 1) / 4;
    const int byte_in_word1 = (iqs + 1) % 4;

    const int64_t row_quants_bytes = ne00;
    const int64_t row_base = i01 * row_quants_bytes;
    const int64_t tile_base = row_base + tile * (TILE_BLOCKS * QK8_0);

    const int64_t qs_offset0 = tile_base + word_idx0 * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word0;
    const int64_t qs_offset1 = tile_base + word_idx1 * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word1;

    const int8_t q0 = (int8_t)*((const uint8_t *)src0 + qs_offset0);
    const int8_t q1 = (int8_t)*((const uint8_t *)src0 + qs_offset1);

    const sycl::half * d_ptr = (const sycl::half *)((const char *)src0 + d_offset) + ib_global;
    const float d = (float)(*d_ptr);

    dst_row[iybs + iqs + 0] = d * (float)q0;
    dst_row[iybs + iqs + 1] = d * (float)q1;
}

// Dispatch function for Q8_0 Coalesced GET_ROWS
static void get_rows_q8_0_coalesced_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          int64_t d_offset,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_q8_0_coalesced<float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, d_offset, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// SoA dispatch function for reordered Q4_0/Q8_0 tensors
template <int qk, int qr, dequantize_kernel_t_reorder dq_reorder>
static void get_rows_sycl_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          int64_t d_offset,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_reorder<qk, qr, dq_reorder, float>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, d_offset, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template<typename src0_t, typename dst_t>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
            });
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int device = ctx.device;
    const void * src0_d = ggml_sycl_get_data_ptr(dst->src[0], device);
    const int32_t * src1_i32 = (const int32_t *) ggml_sycl_get_data_ptr(dst->src[1], device);
    float * dst_d = (float *) ggml_sycl_get_data_ptr(dst, device);

    // TP DEBUG (controlled by GGML_SYCL_TP_DEBUG environment variable)
    bool is_tok_embd = dst->src[0]->name && strstr(dst->src[0]->name, "token_embd");
    int64_t n_rows = dst->src[1]->ne[0];  // Number of rows to fetch
    static int getrows_b1_dbg = 0;
    if (g_ggml_sycl_tp_debug && is_tok_embd && n_rows == 1 && getrows_b1_dbg++ < 5) {
        // Read token ID
        int32_t tok_id;
        ctx.stream()->memcpy(&tok_id, src1_i32, sizeof(int32_t)).wait();
        fprintf(stderr, "TP DEBUG GET_ROWS tok_embd batch=1: device=%d, tok_id=%d, src0_d=%p, src1_i32=%p, dst_d=%p\n",
                device, tok_id, src0_d, (void*)src1_i32, (void*)dst_d);

        // Check if embedding table has per-device pointers (extra->data_device)
        ggml_tensor * emb_tensor = dst->src[0];
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) emb_tensor->extra;
        if (extra) {
            fprintf(stderr, "TP DEBUG GET_ROWS: emb extra=%p, data_device[0]=%p, data_device[1]=%p\n",
                    extra, extra->data_device[0], extra->data_device[1]);
        } else {
            fprintf(stderr, "TP DEBUG GET_ROWS: emb extra=NULL (no per-device pointers)\n");
        }

        // Check embedding table values at tok_id position
        // Embedding table has shape [vocab_size, n_embd] = [32000, 4096]
        // For F32: row_offset = tok_id * n_embd * sizeof(float)
        // For F16: row_offset = tok_id * n_embd * sizeof(sycl::half)
        int64_t n_embd = dst->src[0]->ne[0];
        size_t elem_size = ggml_type_size(dst->src[0]->type);
        size_t row_offset = tok_id * n_embd * elem_size;

        fprintf(stderr, "TP DEBUG GET_ROWS embd table: ne[0]=%lld (n_embd), ne[1]=%lld (vocab), type=%s, elem_size=%zu\n",
                (long long)dst->src[0]->ne[0], (long long)dst->src[0]->ne[1],
                ggml_type_name(dst->src[0]->type), elem_size);

        // Read Q4_0 block at tok_id row (first block)
        // Q4_0 block: 2 bytes scale (fp16) + 16 bytes quants (32 4-bit values)
        // Block offset = tok_id * blocks_per_row * block_size = tok_id * (4096/32) * 18
        struct block_q4_0_t {
            sycl::half d;
            uint8_t qs[16];
        };
        int64_t blocks_per_row = n_embd / 32;  // 4096/32 = 128 blocks per row
        size_t q4_row_offset = tok_id * blocks_per_row * sizeof(block_q4_0_t);

        block_q4_0_t blk;
        const char * q4_ptr = (const char*)src0_d + q4_row_offset;
        ctx.stream()->memcpy(&blk, q4_ptr, sizeof(blk)).wait();

        float d_val = (float)blk.d;
        // Dequantize first 4 values
        int v0 = (blk.qs[0] & 0xF) - 8;
        int v1 = (blk.qs[0] >> 4) - 8;
        int v2 = (blk.qs[1] & 0xF) - 8;
        int v3 = (blk.qs[1] >> 4) - 8;
        fprintf(stderr, "TP DEBUG GET_ROWS Q4_0[tok=%d]: ptr=%p, d=%.6f, qs[0-1]=0x%02x%02x, deq=[%.6f,%.6f,%.6f,%.6f]\n",
                tok_id, q4_ptr, d_val, blk.qs[0], blk.qs[1], v0*d_val, v1*d_val, v2*d_val, v3*d_val);

        // Also check token 0 and token 1 to verify embedding table has data
        block_q4_0_t blk0, blk1;
        ctx.stream()->memcpy(&blk0, src0_d, sizeof(blk0)).wait();  // Token 0, block 0
        ctx.stream()->memcpy(&blk1, (const char*)src0_d + blocks_per_row * sizeof(block_q4_0_t), sizeof(blk1)).wait();  // Token 1, block 0
        fprintf(stderr, "TP DEBUG GET_ROWS tok0: d=%.6f, qs[0]=0x%02x | tok1: d=%.6f, qs[0]=0x%02x\n",
                (float)blk0.d, blk0.qs[0], (float)blk1.d, blk1.qs[0]);

        // Check which device the queue is actually on
        sycl::device queue_dev = ctx.stream()->get_device();
        fprintf(stderr, "TP DEBUG GET_ROWS: ctx.device=%d, queue_device='%s'\n",
                device, queue_dev.get_info<sycl::info::device::name>().c_str());

        // Also check tensor->data vs resolved pointer
        fprintf(stderr, "TP DEBUG GET_ROWS: tensor->data=%p, resolved src0_d=%p (match=%d)\n",
                dst->src[0]->data, src0_d, (dst->src[0]->data == src0_d));
    }

    // DEBUG: Check F32 get_rows for inp_out_ids reduction (attention output to FFN)
    // This is triggered when src0 is F32 and batch reduces from >1 to 1
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int f32_getrows_dbg = 0;
    bool is_f32 = dst->src[0]->type == GGML_TYPE_F32;
    bool is_reduction = dst->src[0]->ne[1] > 1 && n_rows == 1;  // Batch >1 reduced to 1
    if (g_ggml_sycl_tp_debug && is_f32 && is_reduction && f32_getrows_dbg++ < 10) {
        const char * name = dst->src[0]->name ? dst->src[0]->name : "?";
        int32_t row_idx;
        ctx.stream()->memcpy(&row_idx, src1_i32, sizeof(int32_t)).wait();

        // Read values at the row being extracted
        int64_t ne0 = dst->src[0]->ne[0];  // Row width
        size_t row_offset = row_idx * ne0 * sizeof(float);
        float src_vals[4];
        const float * row_ptr = (const float*)((const char*)src0_d + row_offset);
        ctx.stream()->memcpy(src_vals, row_ptr, 4*sizeof(float)).wait();

        fprintf(stderr, "TP DEBUG GET_ROWS F32 reduction: src0=%s ne=[%lldx%lld], extracting row %d, values=[%f,%f,%f,%f]\n",
                name, (long long)ne0, (long long)dst->src[0]->ne[1], row_idx,
                src_vals[0], src_vals[1], src_vals[2], src_vals[3]);

        // Check if any values are NaN
        bool has_nan = false;
        for (int i = 0; i < 4; i++) if (std::isnan(src_vals[i])) has_nan = true;
        if (has_nan) {
            fprintf(stderr, "TP DEBUG GET_ROWS F32: WARNING - NaN found in source row %d!\n", row_idx);
        }
    }

    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)src0_d,
                                src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
        {
            // Check reorder mode for proper kernel dispatch
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->src[0]->extra;
            bool is_soa = extra && extra->optimized_feature.is_soa();

            if (is_soa) {
                // SoA layout: all qs first, then all d
                // d_offset = total qs bytes = ne01 * ne00 / 2 (for Q4_0: 16 bytes qs per 32 values)
                const int64_t ne00 = dst->src[0]->ne[0];
                const int64_t ne01 = dst->src[0]->ne[1];
                const int64_t d_offset = ne01 * ne00 / 2;
                get_rows_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, d_offset, ctx.stream());
            } else if (extra && extra->optimized_feature.is_coalesced()) {
                // Coalesced layout: word-major within tiles
                // d_offset = total qs bytes = ne01 * ne00 / 2 (for Q4_0: 16 bytes qs per 32 values)
                const int64_t ne00 = dst->src[0]->ne[0];
                const int64_t ne01 = dst->src[0]->ne[1];
                const int64_t d_offset = ne01 * ne00 / 2;
                get_rows_q4_0_coalesced_sycl(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, d_offset, ctx.stream());
            } else {
                // AoS (original) layout
                get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst,
                    (const float *)src0_d, src1_i32, dst_d, ctx.stream());
            }
        }
        break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)src0_d,
            src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
        {
            // Check reorder mode for proper kernel dispatch
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->src[0]->extra;
            if (extra && extra->optimized_feature.is_soa()) {
                // SoA layout: all qs first, then all d
                // d_offset = total qs bytes = ne01 * ne00 (for Q8_0: 32 bytes qs per 32 values)
                const int64_t ne00 = dst->src[0]->ne[0];
                const int64_t ne01 = dst->src[0]->ne[1];
                const int64_t d_offset = ne01 * ne00;
                get_rows_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, d_offset, ctx.stream());
            } else if (extra && extra->optimized_feature.is_coalesced()) {
                // Coalesced layout: word-major within tiles
                // d_offset = total qs bytes = ne01 * ne00 (for Q8_0: 32 bytes qs per 32 values)
                const int64_t ne00 = dst->src[0]->ne[0];
                const int64_t ne01 = dst->src[0]->ne[1];
                const int64_t d_offset = ne01 * ne00;
                get_rows_q8_0_coalesced_sycl(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, d_offset, ctx.stream());
            } else {
                // AoS (original) layout
                get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst,
                    (const float *)src0_d, src1_i32, dst_d, ctx.stream());
            }
        }
        break;
        case GGML_TYPE_Q6_K:
        {
            // Check reorder mode for proper kernel dispatch
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->src[0]->extra;
            if (extra && extra->optimized_feature.is_coalesced()) {
                // Coalesced layout: word-major reordering within tiles for each component
                // Calculate d_offset: total quant bytes (ql+qh+sc) across all rows
                const int64_t ne00 = dst->src[0]->ne[0];
                const int64_t ne01 = dst->src[0]->ne[1];
                const int64_t blocks_per_row = ne00 / QK_K;
                const int64_t ql_per_row = blocks_per_row * (QK_K / 2);
                const int64_t qh_per_row = blocks_per_row * (QK_K / 4);
                const int64_t sc_per_row = blocks_per_row * (QK_K / 16);
                const int64_t row_quants_bytes = ql_per_row + qh_per_row + sc_per_row;
                const int64_t d_offset = ne01 * row_quants_bytes;
                get_rows_q6_k_coalesced_sycl(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, d_offset, ctx.stream());
            } else if (extra && extra->optimized_feature.is_soa()) {
                // Standard SoA layout: [all ql (n*128)][all qh (n*64)][all scales (n*16)][all d (n*2)]
                get_rows_q6_k_soa_sycl(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, ctx.stream());
            } else {
                // AoS (original) layout - uses specialized kernel due to Q6_K block complexity
                get_rows_q6_k_aos_sycl(ctx, dst->src[0], dst->src[1], dst,
                    src0_d, src1_i32, dst_d, ctx.stream());
            }
        }
        break;
        default:
            // TODO: other k-quants (Q2_K, Q3_K, Q4_K, Q5_K)
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }

    // DEBUG: Check output after kernel for token embedding batch=1
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int getrows_out_dbg = 0;
    bool is_tok_embd_out = dst->src[0]->name && strstr(dst->src[0]->name, "token_embd");
    int64_t n_rows_out = dst->src[1]->ne[0];
    if (g_ggml_sycl_tp_debug && is_tok_embd_out && n_rows_out == 1 && getrows_out_dbg++ < 5) {
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst_d, std::min((size_t)8*sizeof(float), dst->ne[0]*sizeof(float))).wait();

        // Check for zeros
        int zero_count = 0;
        for (int i = 0; i < 8; i++) {
            if (out_vals[i] == 0.0f) zero_count++;
        }

        fprintf(stderr, "TP DEBUG GET_ROWS tok_embd OUTPUT batch=1: device=%d, dst[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f zeros=%d/8\n",
                device, out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                out_vals[4], out_vals[5], out_vals[6], out_vals[7], zero_count);
    }
}
