//******************************************************************************
// Bare Metal SET_ROWS F32 Kernel
// Writes source data rows to specific indices in destination tensor
//
// Algorithm:
// 1. Read row indices from src1 (int64 tensor)
// 2. For each source row, write it to destination at the specified index
// 3. Handle type conversion: F32 source -> F32/F16 destination
// 4. Support multi-dimensional tensor operations
//
// Operation: dst[indices[i]] = src[i] for i = 0..num_source_rows
// This is the inverse of GET_ROWS operation
//
// Features supported:
// - F32 source data (always F32 input)
// - F32 and F16 destination data (with transcoding)
// - Int64 row indices (vs Int32 in GET_ROWS)
// - Multi-dimensional tensor support
// - Sequential source reads, scattered destination writes
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

struct ggml_et_set_rows_params {
    struct ggml_tensor src0;     // F32 source data tensor
    struct ggml_tensor src1;     // I64 row indices tensor
    struct ggml_tensor dst;      // F32/F16 destination tensor
};

KERNEL_TRAMPOLINE();

// Copy a row of F32 data to F32 destination (no conversion)
static void copy_f32_to_f32_row(float* dst, const float* src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

// Copy a row of F32 data to F16 destination (with conversion)
static void copy_f32_to_f16_row(uint16_t* dst, const float* src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

int entry_point(struct ggml_et_set_rows_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    // Single-threaded implementation: only thread 0 does the work
    // All other threads return early
    if (thread_id != 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0;  // Source data tensor (F32)
    struct ggml_tensor* src1 = &params->src1;  // Row indices tensor (I64)
    struct ggml_tensor* dst = &params->dst;    // Destination tensor (F32/F16)

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        return -1; // Invalid source types
    }

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return -1; // Unsupported destination type
    }

    float* src0_data = (float*)src0->data;
    int64_t* src1_data = (int64_t*)src1->data;
    void* dst_data = dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    const int64_t ne00 = src0->ne[0];  // Source columns (row width)
    const int64_t ne01 = src0->ne[1];  // Source rows (number of rows to write)
    const int64_t ne02 = src0->ne[2];  // Source batch dimension
    const int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    const int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    const int64_t ne12 = src1->ne[2];  // Batch dimension for indices

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    const int64_t ne_dst1 = dst->ne[1]; // Number of rows in destination (for bounds checking)

    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    // Validate that number of indices matches number of source rows
    if (ne10 != ne01) {
        return -1; // Number of indices must match number of source rows
    }

    // Naive single-threaded implementation - process all rows sequentially
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                // Calculate byte offset in indices tensor using strides
                const int64_t i12 = i03 % ne12;
                const int64_t i11 = i02 % ne11;
                const int64_t i10 = i01;

                // Get destination row index using stride-based addressing (in bytes)
                const int64_t index_byte_offset = i10*nb10 + i11*nb11 + i12*nb12;
                const int64_t dst_row_index = *(int64_t*)((char*)src1_data + index_byte_offset);

                if (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                    return -1; // Index out of bounds
                }

                const char* src_row_ptr = (char*)src0_data + i01*nb01 + i02*nb02 + i03*nb03;
                const float* src_row = (const float*)src_row_ptr;

                char* dst_row_ptr = (char*)dst_data + dst_row_index*nb1 + i02*nb2 + i03*nb3;

                if (dst->type == GGML_TYPE_F32) {
                    // F32 destination: direct copy
                    float* dst_row = (float*)dst_row_ptr;
                    copy_f32_to_f32_row(dst_row, src_row, ne00);

                } else if (dst->type == GGML_TYPE_F16) {
                    // F16 destination: convert while copying
                    uint16_t* dst_row = (uint16_t*)dst_row_ptr;
                    copy_f32_to_f16_row(dst_row, src_row, ne00);
                }
            }
        }
    }

    return 0;
}
