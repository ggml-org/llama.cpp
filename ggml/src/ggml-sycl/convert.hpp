//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_CONVERT_HPP
#define GGML_SYCL_CONVERT_HPP

#include "common.hpp"

template <typename T>
using to_t_sycl_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
typedef to_t_sycl_t<float>      to_fp32_sycl_t;
typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;

// full_tensor: Set to false when processing row slices to disable SoA-aware kernels
// (SoA kernels compute d_offset from k, which is wrong for row slices)
to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor = true);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor = true);

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = void (*)(const void * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

typedef to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t get_to_fp16_nc_sycl(ggml_type type);

// =============================================================================
// Q4_0 Coalesced Layout Reorder Functions
// =============================================================================
// These functions convert Q4_0 data between different memory layouts:
//
// AoS (Array of Structures): Original block_q4_0 format
//   [block0: d,qs[16]][block1: d,qs[16]]...
//
// Coalesced: Warp-optimized layout for GPU memory access
//   Tiles of MMVQ_COALESCED_TILE_BLOCKS blocks with interleaved qs for coalesced access:
//   [tile0: qs interleaved, d values][tile1...]
//
// SoA (Structure of Arrays): Intermediate format
//   [all qs bytes contiguous][all d values contiguous]
// =============================================================================

// Convert Q4_0 from AoS (block_q4_0) to SoA layout
void reorder_q4_0_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q8_0 from AoS (block_q8_0) to SoA layout
void reorder_q8_0_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q4_K from AoS (block_q4_K) to SoA layout
void reorder_q4_k_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t nblocks,
    dpct::queue_ptr stream);

// Convert Q6_K from AoS (block_q6_K) to SoA layout
void reorder_q6_k_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t nblocks,
    dpct::queue_ptr stream);

// Convert MXFP4 from AoS (block_mxfp4) to SoA layout
void reorder_mxfp4_aos_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q4_0 from AoS (block_q4_0) to Coalesced layout
void reorder_q4_0_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,  // number of elements per row
    int64_t ne01,  // number of rows
    dpct::queue_ptr stream);

// Convert Q8_0 from AoS (block_q8_0) to Coalesced layout
void reorder_q8_0_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert MXFP4 from AoS (block_mxfp4) to Coalesced layout
void reorder_mxfp4_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q6_K from AoS (block_q6_K) to Coalesced layout
void reorder_q6_k_aos_to_coalesced_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

// Convert Q4_0 from Coalesced layout to SoA layout (for testing/debugging)
void reorder_q4_0_coalesced_to_soa_sycl(
    const void * src,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream);

#endif  // GGML_SYCL_CONVERT_HPP
