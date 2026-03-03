//******************************************************************************
// Bare Metal GET_ROWS F32 Kernel
// Extracts specific rows from a source tensor based on row indices
//
// Algorithm:
// 1. Read row indices from src1 (int32 tensor)
// 2. For each index, extract the corresponding row from src0
// 3. Copy the row data to the output tensor dst
// 4. Handle different input types: F32 and Q8_0 (quantized)
//
// Operation: dst[i] = src0[indices[i]] for i = 0..num_indices
//
// Features supported:
// - F32 input data (direct copy)
// - Q8_0 quantized input data (dequantized to F32)
// - Int32 row indices
// - Multi-dimensional tensor support
// - Memory-efficient row extraction
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "quants.h"

struct ggml_et_get_rows_params {
    struct ggml_tensor src0;     // Data tensor (F32 or Q8_0)
    struct ggml_tensor src1;     // Row indices tensor (I32)
    struct ggml_tensor dst;      // Output tensor (F32)
};

#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_ELEMENTS(elem_size) (CACHE_LINE_SIZE_BYTES / (elem_size))

// Copy a row of F32 data from source to destination
static void copy_f32_row(float* dst, const float* src, int64_t num_elements) {
    // Simple memcpy for F32 data - no conversion needed
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
static void copy_row_cache_align(float* dst, const float* src, int64_t n_bytes) {
    int num_f32_elem = n_bytes / sizeof(float);

    // Unrolled to do an entire cache line at a time
    __asm__ volatile (
        "1: \n\t"
        // --- Process 64 Bytes (1 Cache Line) ---
        // Load 256 bits (32 bytes) into f0 and the other into f1
        "flq2 f0, 0(%[src]) \n\t"
        "flq2 f1, 32(%[src]) \n\t"

        // Store 256 bits (32 bytes) from f0 and f1
        "fsq2 f0, 0(%[dst]) \n\t"
        "fsq2 f1, 32(%[dst]) \n\t"

        // Increment Pointers by 64 bytes
        "addi %[src], %[src], 64 \n\t"
        "addi %[dst], %[dst], 64 \n\t"

        // Decrement count by 16 elements
        "addi %[n], %[n], -16 \n\t"

        // Loop if at least 16 elements remain
        "bge %[n], %[stride_count], 1b \n\t"

        : [dst] "+r" (dst), [src] "+r" (src), [n] "+r" (num_f32_elem)
        : [stride_count] "r" (16L)
        : "f0", "f1", "memory"
    );
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
static void copy_q8_0_row(float* dst, const block_q8_0* src_blocks, int64_t num_elements) {
    // Number of Q8_0 blocks needed for this row
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        float temp_buffer[QK8_0];
        dequantize_q8_0_block(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

static void dequantize_q8_0_block_cache_aligned(const block_q8_0* block, float* dst) {
    const int8_t* qs_ptr = block->qs;

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    const int32_t __attribute__((aligned(32))) vec_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float scale = fp16_to_fp32(block->d);
    __asm__ volatile (
        "fbcx.ps     f0, %0       \n\t" // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t" // Load gether indicies
        :: "r"(scale), "r"(vec_indices)
        : "f0", "f1"
    );

    for (int i = 0; i < 4; i++) {
        __asm__ volatile (
            "fgb.ps      f2, f1(%0)   \n\t" // Loads 8 bytes from (qs_ptr + indices) and sign-extends to 32-bit int.
            "fcvt.ps.pw  f2, f2, rne  \n\t" // Convert Int32 to Float32
            "fmul.ps     f2, f2, f0   \n\t" // f2 = f2 * f0 (scale)
            "fsq2        f2, 0(%1)    \n\t" // Store 256 bits (8 floats) to dst.

            :: "r"(qs_ptr), "r"(dst)
            : "f2", "memory"
        );

        // Advance pointers in C
        qs_ptr += 8;
        dst += 8;
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
static void copy_q8_0_row_cache_aligned(float* dst, const block_q8_0* src_blocks, int64_t num_elements) {
    // Number of Q8_0 blocks needed for this row
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        float temp_buffer[QK8_0];
        dequantize_q8_0_block_cache_aligned(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}


static int get_row_f32_mc_row_cache_aligned(struct ggml_et_get_rows_params* params, void* env)
{
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    struct ggml_tensor* src0 = &params->src0;  // Data tensor (F32 or Q8_0)
    struct ggml_tensor* src1 = &params->src1;  // Row indices tensor (I32)
    struct ggml_tensor* dst = &params->dst;    // Output tensor (F32)

    const int64_t ne00 = src0->ne[0];  // Source columns (row width)
    const int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    const int64_t ne02 = src0->ne[2];  // Source batch dimension
    const int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    const int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    const int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    const int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    const int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    for (int64_t i = thread_id; i < total_rows_to_extract; i+=num_threads) {
        // Calculate multi-dimensional index for the current output position
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        void* src0_data = src0->data;
        int32_t* src1_data = (int32_t*)src1->data;
        float* dst_data = (float*)dst->data;
        // Get the row index from src1
        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 +
                                    i12_idx * ne11 * ne10 +
                                    i11_idx * ne10 +
                                    i10_idx;
        const int32_t row_index = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) {
            return -1; // Index out of bounds
        }

        const int64_t batch_offset = i11_idx * ne01 * ne00 +
                                     i12_idx * ne02 * ne01 * ne00 +
                                     i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t dst_offset = i;

        if (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            const float* src_row = (const float*)src0_data + row_index * ne00 + batch_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_row_cache_align(dst_row, src_row, ne00 * sizeof(float));
        }
        else if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            const int64_t blocks_per_row = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q8_0* src_blocks = (const block_q8_0*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q8_0_row_cache_aligned(dst_row, src_blocks, ne00);
        }
    }

    return 0;
}

int entry_point(struct ggml_et_get_rows_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) {
        return -1;
    }

    struct ggml_tensor* src0 = &params->src0;  // Data tensor (F32 or Q8_0)
    struct ggml_tensor* src1 = &params->src1;  // Row indices tensor (I32)
    struct ggml_tensor* dst = &params->dst;    // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    if((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q8_0) && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32
        && dst->ne[0] % CACHE_ELEMENTS(sizeof(float)) == 0) {
        return get_row_f32_mc_row_cache_aligned(params, env);
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    if (thread_id != 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        return -1; // Invalid output or index type
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q8_0) {
        return -1; // Unsupported input type
    }

    void* src0_data = src0->data;
    int32_t* src1_data = (int32_t*)src1->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    const int64_t ne00 = src0->ne[0];  // Source columns (row width)
    const int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    const int64_t ne02 = src0->ne[2];  // Source batch dimension
    const int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    const int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    const int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    const int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    const int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    // Naive single-threaded implementation - process all rows sequentially
    for (int64_t i = 0; i < total_rows_to_extract; i++) {
        // Calculate multi-dimensional index for the current output position
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 +
                                    i12_idx * ne11 * ne10 +
                                    i11_idx * ne10 +
                                    i10_idx;
        const int32_t row_index = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) {
            return -1; // Index out of bounds
        }

        const int64_t batch_offset = i11_idx * ne01 * ne00 +
                                     i12_idx * ne02 * ne01 * ne00 +
                                     i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t dst_offset = i;

        if (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            const float* src_row = (const float*)src0_data + row_index * ne00 + batch_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_f32_row(dst_row, src_row, ne00);

        } else if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            const int64_t blocks_per_row = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q8_0* src_blocks = (const block_q8_0*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q8_0_row(dst_row, src_blocks, ne00);
        }
    }

    return 0;
}
