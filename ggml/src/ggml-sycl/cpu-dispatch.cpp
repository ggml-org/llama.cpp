//
// cpu-dispatch.cpp — CPU compute path for data-local inference
//
// Executes operations on a CPU SYCL queue when weight data resides in host
// pinned memory (unified cache PINNED_HOST or MMAP tier).  Avoids unnecessary
// host-to-device transfers for layers evicted from VRAM.
//
// MUL_MAT uses dnnl_sgemm (pure CPU BLAS, no SYCL queue required) after
// dequantizing weights to F32 via ggml type traits.  Element-wise ops
// (RMS_NORM, ADD, MUL) use portable SYCL parallel_for on the CPU queue.
//
// Supports F32, F16, and quantized types (via dequantize-to-F32 + sgemm).
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

#include <cmath>
#include <cstring>

#if GGML_SYCL_DNNL
#include "gemm.hpp"          // Provides dnnl.hpp → dnnl_sgemm()
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
    // Free old buffer using the same queue context it was allocated with.
    // Assert queue consistency — in practice ctx.stream() returns the same
    // queue for a given device, but guard against stale-pointer bugs.
    if (g_cpu_staging[slot].ptr && g_cpu_staging_gpu_q) {
        GGML_ASSERT(gpu_q == g_cpu_staging_gpu_q && "staging queue changed between calls");
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
    // For host-accessible buffers (weight mmap, host-pinned), return tensor->data directly.
    // DO NOT use ggml_sycl_get_data_ptr() for these — it returns the cached DEVICE copy
    // from extra->data_device[device] (unified cache), which is not host-accessible.
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }

    // Device-resident buffer (compute buffer for activations) → stage to host
    void * ptr = ggml_sycl_get_data_ptr(t, device);
    if (!ptr) {
        return nullptr;
    }
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

// Get host pointer for output tensor.
// Always uses staging slot 2 (by convention: slot 0 = src0, slot 1 = src1, slot 2 = dst).
// If device-resident, ensures staging buffer is allocated but doesn't copy
// (the kernel will write fresh data).
static void * get_host_output_ptr(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    // Host-accessible buffer → use tensor->data directly (same logic as get_host_ptr)
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
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
    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // activations

    if (!src0 || !src1) {
        return false;
    }

    // Activations must be F32, output must be F32
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Supported weight types: F32, F16, or any quantized type with to_float
    const bool src0_f32       = (src0->type == GGML_TYPE_F32);
    const bool src0_quantized = !src0_f32 && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));

    if (!src0_f32 && !src0_quantized) {
        return false;
    }

    const auto * type_traits = src0_quantized ? ggml_get_type_traits(src0->type) : nullptr;
    if (src0_quantized && (!type_traits || !type_traits->to_float)) {
        return false;
    }

    // ggml MUL_MAT convention:  C^T = A * B^T  =>  C = B * A^T
    //   src0 = A  (weights)     [ne00 x ne01]  (ne00 = K, ne01 = N)
    //   src1 = B  (activations) [ne10 x ne11]  (ne10 = K, ne11 = M)
    //   dst  = C  (output)      [ne0  x ne1 ]  (ne0  = N, ne1  = M)
    //
    // We need: C[M,N] = src1[M,K] * src0^T[K,N]

    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const dnnl_dim_t M = static_cast<dnnl_dim_t>(src1->ne[1]);
    const dnnl_dim_t N = static_cast<dnnl_dim_t>(src0->ne[1]);
    const dnnl_dim_t K = static_cast<dnnl_dim_t>(src0->ne[0]);

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    // Stage all tensors to host-accessible memory.
    // Weights (src0) are in SYCL device buffer (tensor->data = USM device ptr).
    // get_host_ptr resolves via unified cache / staging and copies to host.
    // Activations (src1) are in GPU compute buffer — also staged to host.
    static int dbg = 0;
    if (dbg < 5) GGML_LOG_INFO("[CPU-MM-DBG] staging src0 %s data=%p buf=%p...\n", src0->name, src0->data, (void*)src0->buffer);
    const void * src0_data = get_host_ptr(src0, device, 0, gpu_q);
    if (dbg < 5) GGML_LOG_INFO("[CPU-MM-DBG] src0 staged to %p\n", src0_data);
    if (dbg < 5) GGML_LOG_INFO("[CPU-MM-DBG] staging src1 %s data=%p buf=%p...\n", src1->name, src1->data, (void*)src1->buffer);
    const void * src1_data = get_host_ptr(src1, device, 1, gpu_q);
    if (dbg < 5) GGML_LOG_INFO("[CPU-MM-DBG] src1 staged to %p\n", src1_data);
    void *       dst_data  = get_host_output_ptr(dst, device, gpu_q);
    if (dbg < 5) GGML_LOG_INFO("[CPU-MM-DBG] dst staged to %p, calling dequant+sgemm...\n", dst_data);
    dbg++;
    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    // Batch dimensions (broadcast src0 if ne02/ne03 < ne12/ne13)
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];

    const dnnl_dim_t ldc = static_cast<dnnl_dim_t>(dst->nb[1] / sizeof(float));

    // Dequant/conversion buffer for non-F32 weights (reused across batch iters)
    std::vector<float> src0_f32_buf;
    if (src0_quantized) {
        src0_f32_buf.resize(static_cast<size_t>(N) * K);
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i02 = i12 % ne02;
            const int64_t i03 = i13 % ne03;

            const char * src0_batch = static_cast<const char *>(src0_data)
                                      + i02 * nb02 + i03 * nb03;
            const float * src1_batch = reinterpret_cast<const float *>(
                static_cast<const char *>(src1_data) + i12 * nb12 + i13 * nb13);
            float * dst_batch = reinterpret_cast<float *>(
                static_cast<char *>(dst_data) + i12 * nb2 + i13 * nb3);

            const float * weight_f32;
            dnnl_dim_t    weight_ld = K;

            if (src0_f32) {
                // F32 weights: use directly from mmap
                weight_f32 = reinterpret_cast<const float *>(src0_batch);
                weight_ld  = static_cast<dnnl_dim_t>(nb01 / sizeof(float));
            } else {
                // F16 / quantized → dequantize each row to F32
                for (dnnl_dim_t row = 0; row < N; row++) {
                    const void * row_data = src0_batch + row * nb01;
                    type_traits->to_float(row_data, src0_f32_buf.data() + row * K, K);
                }
                weight_f32 = src0_f32_buf.data();
            }

            // Pure CPU GEMM via oneDNN BLAS (no SYCL queue needed).
            // Row-major C[M,N] = src1[M,K] * weight^T[K,N] is computed
            // via Fortran-convention trick: C_f = weight_f * src1_f^T
            //   transa='T': weight [N,K] row-major → Fortran [K,N], transposed = [N,K]
            //   transb='N': src1   [M,K] row-major → Fortran [K,M], used as-is
            //   result C:   dst    [M,N] row-major → Fortran [N,M]
            dnnl_sgemm('T', 'N',
                       N, M, K,
                       1.0f,
                       weight_f32, weight_ld,
                       src1_batch, K,
                       0.0f,
                       dst_batch, ldc);
        }
    }

    // Copy output from host staging back to device compute buffer
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
// ADD / MUL  (SYCL parallel_for on CPU queue, general ND broadcast)
// ---------------------------------------------------------------------------
//
// General broadcast: src1 dimensions can be 1 where src0 dimensions are > 1.
// The broadcast pattern uses modulo indexing across all 4 dimensions,
// following the same stride-based approach as ggml-cpu/binary-ops.cpp.
// src0 and dst always have the same shape.

enum class binary_op_type { OP_ADD, OP_MUL };

static bool cpu_binary_op(ggml_backend_sycl_context & ctx, ggml_tensor * dst, binary_op_type op) {
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
    // src0 and dst must be contiguous; src1 must be contiguous along dim 0
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float)) {
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

    // dst/src0 dimensions
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // src1 dimensions (may be smaller for broadcasting)
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    // src1 strides in floats
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);

    // Number of column repetitions within a row
    const int64_t nr0 = ne00 / ne10;

    // Total rows = ne01 * ne02 * ne03
    const int64_t total_rows = ne01 * ne02 * ne03;

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_rows)), [=](sycl::id<1> row_id) {
            const int64_t ir  = static_cast<int64_t>(row_id[0]);
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            // Broadcast: map src0 indices to src1 via modulo
            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            // src0 and dst are contiguous, so row offset is straightforward
            const int64_t src0_row_off = ir * ne00;
            const int64_t dst_row_off  = ir * ne00;

            // src1 uses stride-based indexing (may not be contiguous in higher dims)
            const float * src1_row = src1_data + i13 * s13 + i12 * s12 + i11 * s11;

            const float * sp0 = src0_data + src0_row_off;
            float *       dp  = dst_data  + dst_row_off;

            if (op == binary_op_type::OP_ADD) {
                for (int64_t r = 0; r < nr0; r++) {
                    for (int64_t j = 0; j < ne10; j++) {
                        dp[r * ne10 + j] = sp0[r * ne10 + j] + src1_row[j];
                    }
                }
            } else {
                for (int64_t r = 0; r < nr0; r++) {
                    for (int64_t j = 0; j < ne10; j++) {
                        dp[r * ne10 + j] = sp0[r * ne10 + j] * src1_row[j];
                    }
                }
            }
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

static bool cpu_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return cpu_binary_op(ctx, dst, binary_op_type::OP_ADD);
}

static bool cpu_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return cpu_binary_op(ctx, dst, binary_op_type::OP_MUL);
}

// ---------------------------------------------------------------------------
// SILU  (x * sigmoid(x), element-wise)
// ---------------------------------------------------------------------------

static bool cpu_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    float *       dst_data = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t n = ggml_nelements(dst);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
            const float x = src_data[i];
            dst_data[i] = x / (1.0f + sycl::exp(-x));
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// GLU  (SWIGLU, REGLU, GEGLU variants — fused gate*up)
// ---------------------------------------------------------------------------

static bool cpu_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1 && !ggml_is_contiguous(src1)) {
        return false;
    }

    const enum ggml_glu_op glu_op = ggml_get_glu_op(dst);
    const int32_t swapped = ((const int32_t *)(dst->op_params))[1];

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    const float * src1_data = src1 ? static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q))
                                   : src0_data;

    if (!src0_data || !src1_data) {
        return false;
    }

    float * dst_data = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));
    if (!dst_data) {
        return false;
    }

    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nrows = ggml_nrows(src0);

    // For single-source GLU, gate and up are split halves of src0
    // swapped controls which half is gate vs up
    const int64_t src0_row_stride = src0->nb[1] / sizeof(float);
    const int64_t src1_row_stride = src1 ? (src1->nb[1] / sizeof(float)) : src0_row_stride;
    const int64_t dst_row_stride  = dst->nb[1] / sizeof(float);

    const int64_t gate_offset = (!src1 && swapped) ? nc : 0;
    const int64_t up_offset   = (!src1 && swapped) ? 0 : nc;
    const bool has_src1 = (src1 != nullptr);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows * nc)), [=](sycl::id<1> idx) {
            const int64_t row = static_cast<int64_t>(idx[0]) / nc;
            const int64_t col = static_cast<int64_t>(idx[0]) % nc;

            float gate_val, up_val;
            if (has_src1) {
                gate_val = src0_data[row * src0_row_stride + col];
                up_val   = src1_data[row * src1_row_stride + col];
            } else {
                gate_val = src0_data[row * src0_row_stride + gate_offset + col];
                up_val   = src0_data[row * src0_row_stride + up_offset + col];
            }

            float activated;
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                    activated = gate_val / (1.0f + sycl::exp(-gate_val));  // silu
                    break;
                case GGML_GLU_OP_REGLU:
                    activated = gate_val > 0.0f ? gate_val : 0.0f;  // relu
                    break;
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                    activated = 0.5f * gate_val * (1.0f + sycl::erf(gate_val * 0.7071067811865475f));  // gelu
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    activated = gate_val / (1.0f + sycl::exp(-1.702f * gate_val));  // quick_gelu
                    break;
                default:
                    activated = gate_val;
                    break;
            }

            dst_data[row * dst_row_stride + col] = activated * up_val;
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// SOFT_MAX  (row-wise softmax with scale and optional mask)
// ---------------------------------------------------------------------------

static bool cpu_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // Optional mask (src1) — F32 only for CPU path
    const ggml_tensor * src1 = dst->src[1];
    if (src1 && src1->type != GGML_TYPE_F32) {
        return false;
    }

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // ALiBi not supported on CPU path for simplicity
    if (max_bias != 0.0f) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    const float * src_data  = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    const float * mask_data = src1 ? static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q))
                                   : nullptr;
    float *       dst_data  = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t ne00  = src0->ne[0];  // row width
    const int64_t nrows = ggml_nrows(src0);

    const int64_t mask_ne1 = src1 ? src1->nb[1] / sizeof(float) : 0;

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
            const int64_t row = static_cast<int64_t>(row_id[0]);
            const float * sp  = src_data + row * ne00;
            float *       dp  = dst_data + row * ne00;

            // Mask row — broadcast across rows within each head
            const float * mp = nullptr;
            if (mask_data) {
                const int64_t mask_row = row % (mask_ne1 > 0 ? (nrows) : 1);
                mp = mask_data + mask_row * ne00;
            }

            // 1. Scale + mask + find max
            float max_val = -INFINITY;
            for (int64_t j = 0; j < ne00; j++) {
                float v = sp[j] * scale;
                if (mp) {
                    v += mp[j];
                }
                dp[j] = v;
                if (v > max_val) {
                    max_val = v;
                }
            }

            // 2. exp(x - max) and sum
            float sum = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                dp[j] = sycl::exp(dp[j] - max_val);
                sum += dp[j];
            }

            // 3. Normalize
            const float inv_sum = 1.0f / sum;
            for (int64_t j = 0; j < ne00; j++) {
                dp[j] *= inv_sum;
            }
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// NORM  (layer normalization: mean-subtract, variance-normalize)
// ---------------------------------------------------------------------------

static bool cpu_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
            const int64_t row     = static_cast<int64_t>(row_id[0]);
            const float * src_row = src_data + row * ne00;
            float *       dst_row = dst_data + row * ne00;

            // Compute mean
            float sum = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum += src_row[j];
            }
            const float mean = sum / static_cast<float>(ne00);

            // Compute variance and mean-subtract
            float var = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                float d = src_row[j] - mean;
                dst_row[j] = d;
                var += d * d;
            }
            var /= static_cast<float>(ne00);

            // Normalize
            const float scale = 1.0f / sycl::sqrt(var + eps);
            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] *= scale;
            }
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// SCALE  (multiply all elements by a scalar)
// ---------------------------------------------------------------------------

static bool cpu_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    float *       dst_data = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t n = ggml_nelements(dst);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
            dst_data[i] = src_data[i] * scale;
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// CPY / CONT  (copy or contiguify tensor data)
// ---------------------------------------------------------------------------

static bool cpu_cpy(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }

    // Simple case: both contiguous, same type → memcpy
    if (src0->type != dst->type) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();
    const void * src_data = get_host_ptr(src0, device, 0, gpu_q);
    void *       dst_data = get_host_output_ptr(dst, device, gpu_q);

    if (!src_data || !dst_data) {
        return false;
    }

    const size_t nbytes = ggml_nbytes(dst);
    memcpy(dst_data, src_data, nbytes);

    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// ROPE  (rotary positional embeddings — NEOX and NORMAL modes, F32 only)
// ---------------------------------------------------------------------------

static bool cpu_rope(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];  // input tensor
    const ggml_tensor * src1 = dst->src[1];  // positions (int32)
    const ggml_tensor * src2 = dst->src[2];  // freq_factors (optional)

    if (!src0 || !src1) {
        return false;
    }
    // F32 only for CPU path
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Extract op_params
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    // Only support NORMAL and NEOX modes on CPU
    const bool is_neox   = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool is_normal = (mode == GGML_ROPE_TYPE_NORMAL);
    if (!is_neox && !is_normal) {
        return false;  // MROPE, VISION, IMROPE not supported
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    const float *   src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q));
    const int32_t * pos_data = static_cast<const int32_t *>(get_host_ptr(src1, device, 1, gpu_q));
    float *         dst_data = static_cast<float *>(get_host_output_ptr(dst, device, gpu_q));

    if (!src_data || !pos_data || !dst_data) {
        return false;
    }

    // Freq factors (optional src2) — must be host-accessible (no staging slot available)
    const float * freq_factors_data = nullptr;
    if (src2) {
        void * src2_ptr = ggml_sycl_get_data_ptr(src2, device);
        if (!src2_ptr) {
            return false;
        }
        // Only handle host-accessible freq_factors (no staging available for 4th tensor)
        if (src2->buffer && !ggml_backend_buffer_is_host(src2->buffer)) {
            return false;
        }
        freq_factors_data = static_cast<const float *>(src2_ptr);
    }

    const int64_t ne0 = src0->ne[0];  // head dim
    const int64_t ne1 = src0->ne[1];  // num heads
    const int64_t ne2 = src0->ne[2];  // seq len
    const int64_t ne3 = src0->ne[3];  // batch

    // Strides in float units
    const int64_t s01 = src0->nb[1] / sizeof(float);
    const int64_t s02 = src0->nb[2] / sizeof(float);
    const int64_t s03 = src0->nb[3] / sizeof(float);
    const int64_t d01 = dst->nb[1] / sizeof(float);
    const int64_t d02 = dst->nb[2] / sizeof(float);
    const int64_t d03 = dst->nb[3] / sizeof(float);

    // YaRN correction dims
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    // Total work items = ne3 * ne2 * ne1 (one per row)
    const int64_t total_rows = ne3 * ne2 * ne1;

    // Capture locals for kernel
    const float cd0 = corr_dims[0];
    const float cd1 = corr_dims[1];

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_rows)), [=](sycl::id<1> work_id) {
            const int64_t idx = static_cast<int64_t>(work_id[0]);
            const int64_t i3  = idx / (ne2 * ne1);
            const int64_t i2  = (idx / ne1) % ne2;
            const int64_t i1  = idx % ne1;

            const float * src_row = src_data + i3 * s03 + i2 * s02 + i1 * s01;
            float *       dst_row = dst_data + i3 * d03 + i2 * d02 + i1 * d01;

            const int32_t p = pos_data[i2];
            const float theta_base = static_cast<float>(p);

            float theta = theta_base;
            for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                const float ff = freq_factors_data ? freq_factors_data[i0 / 2] : 1.0f;

                // YaRN rope_yarn inline
                const float theta_extrap = theta / ff;
                float theta_interp = freq_scale * theta_extrap;
                float theta_val = theta_interp;
                float mscale = attn_factor;

                if (ext_factor != 0.0f) {
                    const float y = (i0 / 2.0f - cd0) / sycl::fmax(0.001f, cd1 - cd0);
                    const float ramp_mix = (1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y))) * ext_factor;
                    theta_val = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                }

                const float cos_theta = sycl::cos(theta_val) * mscale;
                const float sin_theta = sycl::sin(theta_val) * mscale;

                if (is_normal) {
                    // NORMAL: pairs are adjacent (i0, i0+1)
                    const float x0 = src_row[i0];
                    const float x1 = src_row[i0 + 1];
                    dst_row[i0]     = x0 * cos_theta - x1 * sin_theta;
                    dst_row[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
                } else {
                    // NEOX: pairs are (i0/2, i0/2 + n_dims/2)
                    const int64_t ic = i0 / 2;
                    const float x0 = src_row[ic];
                    const float x1 = src_row[ic + n_dims / 2];
                    dst_row[ic]              = x0 * cos_theta - x1 * sin_theta;
                    dst_row[ic + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
                }

                theta *= theta_scale;
            }

            // Copy remaining dimensions beyond n_dims
            if (!is_normal) {
                for (int64_t i0 = n_dims; i0 < ne0; i0++) {
                    dst_row[i0] = src_row[i0];
                }
            } else {
                for (int64_t i0 = n_dims; i0 < ne0; i0++) {
                    dst_row[i0] = src_row[i0];
                }
            }
        });
    });

    cpu_q->wait();
    flush_output(dst, device, 2, gpu_q);
    return true;
}

// ---------------------------------------------------------------------------
// UNARY dispatch (SILU and others)
// ---------------------------------------------------------------------------

static bool cpu_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const enum ggml_unary_op op = ggml_get_unary_op(dst);
    switch (op) {
        case GGML_UNARY_OP_SILU:
            return cpu_silu(ctx, dst);
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported unary op %d on CPU\n", static_cast<int>(op));
            return false;
    }
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
        case GGML_OP_UNARY:
            return cpu_unary(ctx, dst);
        case GGML_OP_GLU:
            return cpu_glu(ctx, dst);
        case GGML_OP_SOFT_MAX:
            return cpu_soft_max(ctx, dst);
        case GGML_OP_NORM:
            return cpu_norm(ctx, dst);
        case GGML_OP_SCALE:
            return cpu_scale(ctx, dst);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return cpu_cpy(ctx, dst);
        case GGML_OP_ROPE:
            return cpu_rope(ctx, dst);
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported op %s on CPU, falling back to GPU\n",
                            ggml_op_name(dst->op));
            return false;
    }
}
