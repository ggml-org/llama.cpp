//
// cpu-dispatch.cpp — CPU compute path for data-local inference
//
// Executes operations on a CPU SYCL queue when weight data resides in host
// pinned memory (unified cache PINNED_HOST or MMAP tier).  Avoids unnecessary
// host-to-device transfers for layers evicted from VRAM.
//
// MUL_MAT uses oneDNN via DnnlGemmWrapper (device-agnostic — passing a CPU
// queue creates a CPU oneDNN engine automatically).  Element-wise ops
// (RMS_NORM, ADD, MUL) use portable SYCL parallel_for on the CPU queue.
//
// Phase 1: F32 and F16 weight types only.  Quantized types (Q4_0 etc.)
// return false so the caller falls back to GPU streaming.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-dispatch.hpp"

#include "ggml.h"

#if GGML_SYCL_DNNL
#include "gemm.hpp"
#endif

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Pointer resolution
// ---------------------------------------------------------------------------

// Resolve a host-accessible pointer for a tensor.
// For CPU-dispatched ops, data is in host pinned memory or mmap — use
// tensor->data directly.  The unified cache ensures evicted weights are
// accessible from the host.
static void * resolve_cpu_ptr(const ggml_tensor * t) {
    return t->data;
}

// ---------------------------------------------------------------------------
// MUL_MAT  (oneDNN on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_mul_mat(ggml_backend_sycl_context & ctx,
                        ggml_tensor * dst,
                        sycl::queue * cpu_q) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(cpu_q);
    return false;
#else
    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // activations

    if (!src0 || !src1) {
        return false;
    }

    // Phase 1: F32 and F16 weights, F32 activations
    const bool src0_f32 = (src0->type == GGML_TYPE_F32);
    const bool src0_f16 = (src0->type == GGML_TYPE_F16);
    if (!src0_f32 && !src0_f16) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // ggml MUL_MAT convention:  C^T = A * B^T  =>  C = B * A^T
    //   src0 = A  (weights)     [ne00 x ne01]  (ne00 = K, ne01 = N)
    //   src1 = B  (activations) [ne10 x ne11]  (ne10 = K, ne11 = M)
    //   dst  = C  (output)      [ne0  x ne1 ]  (ne0  = N, ne1  = M)
    //
    // We need: C[M,N] = src1[M,K] * src0^T[K,N]
    // DnnlGemmWrapper::gemm() with custom strides expresses the transpose.

    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // N
    const int64_t ne10 = src1->ne[0];  // K
    const int64_t ne11 = src1->ne[1];  // M

    GGML_ASSERT(ne00 == ne10);

    const int M = static_cast<int>(ne11);
    const int N = static_cast<int>(ne01);
    const int K = static_cast<int>(ne00);

    const void * src0_data = resolve_cpu_ptr(src0);
    const void * src1_data = resolve_cpu_ptr(src1);
    void *       dst_data  = resolve_cpu_ptr(dst);

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    using dt = DnnlGemmWrapper::dt;
    const dt a_type = dt::f32;                        // activations always F32
    const dt b_type = src0_f16 ? dt::f16 : dt::f32;  // weights
    const dt c_type = dt::f32;                        // output always F32

    // Batch dimensions (broadcast src0 if ne02/ne03 < ne12/ne13)
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb1  = dst->nb[1];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];

    const int64_t src0_elem_size = static_cast<int64_t>(ggml_type_size(src0->type));
    const int64_t src1_elem_size = static_cast<int64_t>(ggml_type_size(src1->type));
    const int64_t dst_elem_size  = static_cast<int64_t>(ggml_type_size(dst->type));

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i02 = i12 % ne02;
            const int64_t i03 = i13 % ne03;

            const char * src0_batch = static_cast<const char *>(src0_data) + i02 * nb02 + i03 * nb03;
            const char * src1_batch = static_cast<const char *>(src1_data) + i12 * nb12 + i13 * nb13;
            char *       dst_batch  = static_cast<char *>(dst_data) + i12 * nb2 + i13 * nb3;

            // A = src1 [M, K], row-major
            const int64_t src1_stride_col = 1;
            const int64_t src1_stride_row = nb11 / src1_elem_size;

            // B = src0^T [K, N] — transpose via swapped strides
            const int64_t src0_stride_col = nb01 / src0_elem_size;  // row stride in memory = col stride of transpose
            const int64_t src0_stride_row = 1;                      // col stride in memory = row stride of transpose

            DnnlGemmWrapper::gemm(ctx, M, N, K,
                src1_batch, a_type, src1_stride_col, src1_stride_row, static_cast<int64_t>(M) * K,
                src0_batch, b_type, src0_stride_col, src0_stride_row, static_cast<int64_t>(N) * K,
                dst_batch,  c_type, cpu_q, 1, 1,
                static_cast<int>(nb1 / dst_elem_size));
        }
    }

    cpu_q->wait();
    return true;
#endif
}

// ---------------------------------------------------------------------------
// RMS_NORM  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_rms_norm(ggml_backend_sycl_context & ctx,
                         ggml_tensor * dst,
                         sycl::queue * cpu_q) {
    GGML_UNUSED(ctx);

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    float eps;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const float * src_data = static_cast<const float *>(resolve_cpu_ptr(src0));
    float *       dst_data = static_cast<float *>(resolve_cpu_ptr(dst));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    // One work-item per row — each computes RMS and normalizes.
    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
            const int64_t row     = static_cast<int64_t>(row_id[0]);
            const float * src_row = src_data + row * ne00;
            float *       dst_row = dst_data + row * ne00;

            float sum_sq = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum_sq += src_row[j] * src_row[j];
            }

            const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(ne00) + eps);

            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] = src_row[j] * scale;
            }
        });
    });

    cpu_q->wait();
    return true;
}

// ---------------------------------------------------------------------------
// ADD  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_add(ggml_backend_sycl_context & ctx,
                    ggml_tensor * dst,
                    sycl::queue * cpu_q) {
    GGML_UNUSED(ctx);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const float * src0_data = static_cast<const float *>(resolve_cpu_ptr(src0));
    const float * src1_data = static_cast<const float *>(resolve_cpu_ptr(src1));
    float *       dst_data  = static_cast<float *>(resolve_cpu_ptr(dst));

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    const int64_t ne_total  = ggml_nelements(dst);
    const int64_t ne1_total = ggml_nelements(src1);

    if (ne1_total == ne_total) {
        // Same shape — element-wise
        cpu_q->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(ne_total)), [=](sycl::id<1> i) {
                dst_data[i] = src0_data[i] + src1_data[i];
            });
        });
    } else if (ne1_total == src0->ne[0]) {
        // Row-vector broadcast (bias add)
        const int64_t ncols = src0->ne[0];
        cpu_q->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(ne_total)), [=](sycl::id<1> idx) {
                const int64_t col = static_cast<int64_t>(idx[0]) % ncols;
                dst_data[idx] = src0_data[idx] + src1_data[col];
            });
        });
    } else {
        return false;  // Unsupported broadcast for Phase 1
    }

    cpu_q->wait();
    return true;
}

// ---------------------------------------------------------------------------
// MUL  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_mul(ggml_backend_sycl_context & ctx,
                    ggml_tensor * dst,
                    sycl::queue * cpu_q) {
    GGML_UNUSED(ctx);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const float * src0_data = static_cast<const float *>(resolve_cpu_ptr(src0));
    const float * src1_data = static_cast<const float *>(resolve_cpu_ptr(src1));
    float *       dst_data  = static_cast<float *>(resolve_cpu_ptr(dst));

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    const int64_t ne_total  = ggml_nelements(dst);
    const int64_t ne1_total = ggml_nelements(src1);

    if (ne1_total == ne_total) {
        // Same shape — element-wise
        cpu_q->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(ne_total)), [=](sycl::id<1> i) {
                dst_data[i] = src0_data[i] * src1_data[i];
            });
        });
    } else if (ne1_total == src0->ne[0]) {
        // Row-vector broadcast (RMS norm weights)
        const int64_t ncols = src0->ne[0];
        cpu_q->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(ne_total)), [=](sycl::id<1> idx) {
                const int64_t col = static_cast<int64_t>(idx[0]) % ncols;
                dst_data[idx] = src0_data[idx] * src1_data[col];
            });
        });
    } else {
        return false;  // Unsupported broadcast for Phase 1
    }

    cpu_q->wait();
    return true;
}

// ---------------------------------------------------------------------------
// Main dispatch entry point
// ---------------------------------------------------------------------------

bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            return cpu_mul_mat(ctx, dst, cpu_q);
        case GGML_OP_RMS_NORM:
            return cpu_rms_norm(ctx, dst, cpu_q);
        case GGML_OP_ADD:
            return cpu_add(ctx, dst, cpu_q);
        case GGML_OP_MUL:
            return cpu_mul(ctx, dst, cpu_q);
        // TODO Phase 2: ROPE, SOFT_MAX, LAYER_NORM, SILU
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported op %s on CPU, falling back to GPU\n",
                            ggml_op_name(dst->op));
            return false;
    }
}
