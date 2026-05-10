//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TURBO_WHT_HPP
#define GGML_SYCL_TURBO_WHT_HPP

#include <sycl/sycl.hpp>
#include "common.hpp"

/**
 * Fast Walsh-Hadamard Transform (FWHT) optimized for Intel Arc GPUs.
 * 
 * This implementation uses subgroup shuffles for intra-subgroup stages
 * and Shared Local Memory (SLM) for inter-subgroup stages.
 * 
 * @tparam D The dimension of the transform. Must be a power of 2.
 * @tparam T The data type (e.g., float, sycl::half).
 * @tparam SG_SIZE Subgroup size, defaults to 32 for Intel Arc.
 * @param val The value held by the current thread. Each thread in the 
 *            work-group (of size D) should hold one element.
 * @param item_ct1 The nd_item for the current kernel execution.
 * @param shared_mem Pointer to Shared Local Memory of size D * sizeof(T).
 *                   Required if D > SG_SIZE.
 */
template <int D, typename T, int SG_SIZE = WARP_SIZE, int DIM = 1>
static __dpct_inline__ void turbo_wht(T &val, const sycl::nd_item<DIM> &item_ct1, T *shared_mem = nullptr) {
    auto sg = item_ct1.get_sub_group();
    const int hw_sg_size = sg.get_local_linear_range();
    const int lane_id = sg.get_local_linear_id();
    const int tid = item_ct1.get_local_linear_id(); // Use linear ID for multi-dimensional work-groups

    // Intra-subgroup stages: step < hw_sg_size
    // Inter-subgroup stages: step >= hw_sg_size
    #pragma unroll
    for (int step = 1; step < D; step <<= 1) {
        if (step < hw_sg_size) {
            T other = dpct::permute_sub_group_by_xor(sg, val, step, hw_sg_size);
            if (lane_id & step) {
                val = other - val;
            } else {
                val = val + other;
            }
        } else {
            if (tid < D) shared_mem[tid] = val;
            item_ct1.barrier(sycl::access::fence_space::local_space);
            
            if (tid < D) {
                T other = shared_mem[tid ^ step];
                if (tid & step) {
                    val = other - val;
                } else {
                    val = val + other;
                }
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }
}

/**
 * Verification of WHT:
 * WHT is its own inverse up to a scale factor of D.
 * To verify correctness:
 * 1. Apply turbo_wht(x)
 * 2. Apply turbo_wht(x) again
 * 3. Result should be D * original_x
 * 
 * Alternatively, apply turbo_wht(x) and divide by sqrt(D) for a unitary transform.
 */

/**
 * Example verification kernel:
 * 
 * template <int D, typename T>
 * void verify_wht_kernel(T *data, const sycl::nd_item<3> &item_ct1, T *shared_mem) {
 *     const int tid = item_ct1.get_local_id(2);
 *     T val = data[tid];
 *     
 *     // Round-trip
 *     turbo_wht<D>(val, item_ct1, shared_mem);
 *     turbo_wht<D>(val, item_ct1, shared_mem);
 *     
 *     // After two transforms, data should be original * D
 *     data[tid] = val / (T)D; 
 * }
 */

#endif // GGML_SYCL_TURBO_WHT_HPP
