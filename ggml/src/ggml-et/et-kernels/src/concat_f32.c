//******************************************************************************
// Concat F32 Kernel
// Concatenates two F32 tensors along a specified dimension.
// All copies are aligned to cacheline boundaries (64 bytes = 16 floats).
//
// For dim >= 1, entire rows are copied from src0 or src1 into dst.
// For dim == 0, each row is two halves: [src0_part | src1_part], both
// cacheline-aligned since ne00 % 16 == 0 is enforced by supports_op.
//******************************************************************************

#include <stdint.h>
#include <string.h>
#include "ggml_tensor.h"
#include "platform.h"

struct ggml_et_concat_params {
    struct ggml_tensor src0;  // F32 input tensor 0
    struct ggml_tensor src1;  // F32 input tensor 1
    struct ggml_tensor dst;   // F32 output tensor
    int32_t dim;              // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
static inline void copy_row_aligned(float* dst, const float* src, int32_t n) {
    for (int32_t i = 0; i < n; i += 8) {
        __asm__ volatile(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst[i])
            : [src_vec] "m"(*(const float(*)[8])&src[i])
            : "f11"
        );
    }
}

int entry_point(struct ggml_et_concat_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst  = &params->dst;
    int32_t dim = params->dim;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    float* src0_data = (float*)src0->data;
    float* src1_data = (float*)src1->data;
    float* dst_data  = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne0  = dst->ne[0],  ne1  = dst->ne[1],  ne2  = dst->ne[2],  ne3  = dst->ne[3];

    // src strides in bytes
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    const size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions - parallelize over these
    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = thread_id; row < total_rows; row += num_threads) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        float* dst_row = (float*)((char*)dst_data + i1*dnb1 + i2*dnb2 + i3*dnb3);

        if (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            const float* s0_row = (const float*)((const char*)src0_data + i1*nb01 + i2*nb02 + i3*nb03);
            const float* s1_row = (const float*)((const char*)src1_data + i1*nb11 + i2*nb12 + i3*nb13);

            copy_row_aligned(dst_row, s0_row, (int32_t)ne00);
            copy_row_aligned(dst_row + ne00, s1_row, (int32_t)ne10);

        } else if (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            if (i1 < ne01) {
                const float* s0_row = (const float*)((const char*)src0_data + i1*nb01 + i2*nb02 + i3*nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t)ne0);
            } else {
                const float* s1_row = (const float*)((const char*)src1_data + (i1 - ne01)*nb11 + i2*nb12 + i3*nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t)ne0);
            }

        } else if (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            if (i2 < ne02) {
                const float* s0_row = (const float*)((const char*)src0_data + i1*nb01 + i2*nb02 + i3*nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t)ne0);
            } else {
                const float* s1_row = (const float*)((const char*)src1_data + i1*nb11 + (i2 - ne02)*nb12 + i3*nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t)ne0);
            }

        } else {
            // dim == 3: first ne03 batches from src0, rest from src1
            if (i3 < ne03) {
                const float* s0_row = (const float*)((const char*)src0_data + i1*nb01 + i2*nb02 + i3*nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t)ne0);
            } else {
                const float* s1_row = (const float*)((const char*)src1_data + i1*nb11 + i2*nb12 + (i3 - ne03)*nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t)ne0);
            }
        }
    }

    return 0;
}
