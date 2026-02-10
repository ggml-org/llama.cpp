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
// Activation staging: Intermediate tensors live in GPU VRAM (compute buffer).
// CPU kernels can't access device memory directly.  Staging buffers copy
// activations host↔device at each CPU op.  Weights are already host-pinned
// so they need no staging.  Phase 2 will use host-pinned compute buffers
// to eliminate staging overhead entirely.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-dispatch.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#if GGML_SYCL_DNNL
#include "gemm.hpp"
#endif

// ---------------------------------------------------------------------------
// Activation staging: reusable host-pinned buffers for device↔host transfer
// ---------------------------------------------------------------------------
//
// GPU compute buffers use sycl::malloc_device() which CPU can't access.
// Weight tensors are already host-pinned (that's why we're here).
// Activation/output tensors need staging: copy device→host before CPU compute,
// then host→device after CPU writes output.
//
// Three staging slots (grown on demand, never shrunk):
//   slot 0: first source tensor (weights or activations)
//   slot 1: second source tensor (activations)
//   slot 2: output tensor

static struct {
    void * ptr = nullptr;
    size_t cap = 0;
} g_cpu_staging[3];

static sycl::queue * g_cpu_staging_gpu_q = nullptr;

static void * staging_ensure(int slot, size_t nbytes, sycl::queue * gpu_q) {
    if (slot < 0 || slot > 2) {
        return nullptr;
    }
    if (nbytes <= g_cpu_staging[slot].cap && g_cpu_staging[slot].ptr) {
        return g_cpu_staging[slot].ptr;
    }
    if (g_cpu_staging[slot].ptr && g_cpu_staging_gpu_q) {
        sycl::free(g_cpu_staging[slot].ptr, *g_cpu_staging_gpu_q);
    }
    g_cpu_staging_gpu_q = gpu_q;
    g_cpu_staging[slot].ptr = sycl::malloc_host(nbytes, *gpu_q);
    g_cpu_staging[slot].cap = nbytes;
    return g_cpu_staging[slot].ptr;
}

// Get host-accessible pointer for a tensor.
// If tensor is in host-accessible memory, returns original pointer.
// If tensor is in device memory, copies to staging slot and returns host ptr.
static void * get_host_ptr(const ggml_tensor * t, int device, int slot, sycl::queue * gpu_q) {
    void * ptr = ggml_sycl_get_data_ptr(t, device);
    if (!ptr) {
        return nullptr;
    }

    // Host-accessible buffer (host pinned, mmap, or no buffer) → use directly
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return ptr;
    }

    // Device-resident → copy to host staging
    size_t nbytes = ggml_nbytes(t);
    void * host = staging_ensure(slot, nbytes, gpu_q);
    if (!host) {
        return nullptr;
    }
    gpu_q->memcpy(host, ptr, nbytes).wait();
    return host;
}

// Copy output from host staging back to device memory.
// No-op if tensor is already in host-accessible memory.
static void flush_output(ggml_tensor * t, int device, int slot, sycl::queue * gpu_q) {
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return;
    }
    void * dev_ptr = ggml_sycl_get_data_ptr(t, device);
    if (!dev_ptr || slot < 0 || slot > 2 || !g_cpu_staging[slot].ptr) {
        return;
    }
    size_t nbytes = ggml_nbytes(t);
    gpu_q->memcpy(dev_ptr, g_cpu_staging[slot].ptr, nbytes).wait();
}

// Get host pointer for output tensor (staging slot 2).
// If device-resident, ensures staging buffer is allocated but doesn't copy
// (the kernel will write fresh data).
static void * get_host_output_ptr(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    void * ptr = ggml_sycl_get_data_ptr(t, device);
    if (!ptr) {
        return nullptr;
    }
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return ptr;
    }
    // Device-resident: allocate staging but don't copy (will be written by kernel)
    size_t nbytes = ggml_nbytes(t);
    return staging_ensure(2, nbytes, gpu_q);
}

// ---------------------------------------------------------------------------
// MUL_MAT  (oneDNN on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

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

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    // Stage tensors: weights already host-pinned, activations may be in VRAM
    const void * src0_data = get_host_ptr(src0, device, 0, gpu_q);
    const void * src1_data = get_host_ptr(src1, device, 1, gpu_q);
    void *       dst_data  = get_host_output_ptr(dst, device, gpu_q);

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

    // Compute byte offsets relative to staged base pointers.
    // For staged tensors, nb values still reflect the original layout since
    // we copied the full contiguous ggml_nbytes() block.
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

    // Flush output from host staging back to device if needed
    flush_output(dst, device, 2, gpu_q);
    return true;
#endif
}

// ---------------------------------------------------------------------------
// RMS_NORM  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

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
    memcpy(&eps, dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    float *       dst_data = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

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

            const float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(ne00) + eps);

            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] = src_row[j] * scale;
            }
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// ADD  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

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

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();
    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    const float * src1_data = static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q));
    float *       dst_data  = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

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
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// MUL  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

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

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();
    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    const float * src1_data = static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q));
    float *       dst_data  = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

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
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// Main dispatch entry point
// ---------------------------------------------------------------------------

bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            return cpu_mul_mat(ctx, dst);
        case GGML_OP_RMS_NORM:
            return cpu_rms_norm(ctx, dst);
        case GGML_OP_ADD:
            return cpu_add(ctx, dst);
        case GGML_OP_MUL:
            return cpu_mul(ctx, dst);
        // TODO Phase 2: ROPE, SOFT_MAX, LAYER_NORM, SILU
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported op %s on CPU, falling back to GPU\n",
                            ggml_op_name(dst->op));
            return false;
    }
}
