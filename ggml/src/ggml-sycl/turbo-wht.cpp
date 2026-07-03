//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "turbo-wht.hpp"
#include "turbo-quants.hpp"
#include "common.hpp"

template <int direction, int group_size>
static void k_turbo_wht_f32_sycl(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const float * __restrict__ scale_inv,
        const int64_t n_groups,
        const int64_t head_dim,
        const int64_t groups_per_head,
        const sycl::nd_item<1> &item_ct1,
        float *shared_mem) {

    static_assert(group_size == 128 || group_size == 64 || group_size == 32,
                  "group_size must be 128, 64, or 32");

    const int64_t g = item_ct1.get_group(0);
    if (g >= n_groups) return;

    const int t = item_ct1.get_local_id(0);

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * group_size;

    float val = src[base + t];

    if (direction == 0 && scale_inv != nullptr) {
        val *= scale_inv[t % group_size];
    }

    if (group_size == 128) {
        val *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    } else if (group_size == 64) {
        val *= (direction == 0) ? TURBO_WHT_SIGNS1_64[t] : TURBO_WHT_SIGNS2_64[t];
    } else {
        val *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    }

    turbo_wht<group_size>(val, item_ct1, shared_mem);

    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f :
                               (group_size == 64)  ? 0.125f :
                                                     0.17677669529663688f;

    float result;
    if (group_size == 128) {
        result = val * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    } else if (group_size == 64) {
        result = val * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2_64[t] : TURBO_WHT_SIGNS1_64[t]);
    } else {
        result = val * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    }

    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}


void ggml_sycl_op_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale_tensor = dst->src[1];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    memcpy(&direction, dst->op_params + 0, sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int), sizeof(int));

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
    // same invariant as ggml_turbo_wht(); a tail would silently bypass the WHT
    GGML_ASSERT(head_dim % group_size == 0);
    const int64_t groups_per_head = head_dim / group_size;
    const int64_t n_groups        = groups_per_head * n_heads;

    queue_ptr stream = ctx.stream();

    const float * src_ptr        = (const float *) src->data;
    float       * dst_ptr        = (float       *) dst->data;
    const float * scale_inv_ptr  = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    if (n_groups > 0) {
        if (group_size == 128) {
            constexpr int D = 128;
            if (direction == 0) {
                stream->submit([&](sycl::handler &h) {
                    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(D), h);
                    h.parallel_for(
                        sycl::nd_range<1>(n_groups * D, D),
                        [=](sycl::nd_item<1> item_ct1) {
                            k_turbo_wht_f32_sycl<0, D>(
                                src_ptr, dst_ptr, scale_inv_ptr,
                                n_groups, head_dim, groups_per_head,
                                item_ct1,
                                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
                            );
                        }
                    );
                });
            } else {
                stream->submit([&](sycl::handler &h) {
                    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(D), h);
                    h.parallel_for(
                        sycl::nd_range<1>(n_groups * D, D),
                        [=](sycl::nd_item<1> item_ct1) {
                            k_turbo_wht_f32_sycl<1, D>(
                                src_ptr, dst_ptr, scale_inv_ptr,
                                n_groups, head_dim, groups_per_head,
                                item_ct1,
                                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
                            );
                        }
                    );
                });
            }
        } else if (group_size == 64) {
            constexpr int D = 64;
            if (direction == 0) {
                stream->submit([&](sycl::handler &h) {
                    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(D), h);
                    h.parallel_for(
                        sycl::nd_range<1>(n_groups * D, D),
                        [=](sycl::nd_item<1> item_ct1) {
                            k_turbo_wht_f32_sycl<0, D>(
                                src_ptr, dst_ptr, scale_inv_ptr,
                                n_groups, head_dim, groups_per_head,
                                item_ct1,
                                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
                            );
                        }
                    );
                });
            } else {
                stream->submit([&](sycl::handler &h) {
                    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(D), h);
                    h.parallel_for(
                        sycl::nd_range<1>(n_groups * D, D),
                        [=](sycl::nd_item<1> item_ct1) {
                            k_turbo_wht_f32_sycl<1, D>(
                                src_ptr, dst_ptr, scale_inv_ptr,
                                n_groups, head_dim, groups_per_head,
                                item_ct1,
                                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
                            );
                        }
                    );
                });
            }
        } else {
            constexpr int D = 32;
            if (direction == 0) {
                stream->submit([&](sycl::handler &h) {
                    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(D), h);
                    h.parallel_for(
                        sycl::nd_range<1>(n_groups * D, D),
                        [=](sycl::nd_item<1> item_ct1) {
                            k_turbo_wht_f32_sycl<0, D>(
                                src_ptr, dst_ptr, scale_inv_ptr,
                                n_groups, head_dim, groups_per_head,
                                item_ct1,
                                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
                            );
                        }
                    );
                });
            } else {
                stream->submit([&](sycl::handler &h) {
                    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(D), h);
                    h.parallel_for(
                        sycl::nd_range<1>(n_groups * D, D),
                        [=](sycl::nd_item<1> item_ct1) {
                            k_turbo_wht_f32_sycl<1, D>(
                                src_ptr, dst_ptr, scale_inv_ptr,
                                n_groups, head_dim, groups_per_head,
                                item_ct1,
                                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
                            );
                        }
                    );
                });
            }
        }
    }
}
