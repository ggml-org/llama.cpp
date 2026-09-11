//******************************************************************************
// Bare Metal GET_ROWS F32 Kernel
// Extracts specific rows from a source tensor based on row indices
//
// Algorithm:
// 1. Read row indices from src1 (int32 tensor)
// 2. For each index, extract the corresponding row from src0
// 3. Copy the row data to the output tensor dst
// 4. Handle different input types: F32, Q8_0, Q4_0, and Q4_K (quantized)
//
// Operation: dst[i] = src0[indices[i]] for i = 0..num_indices
//
// Features supported:
// - F32 input data (direct copy)
// - Q4_0 quantized input data (dequantized to F32)
// - Q8_0 quantized input data (dequantized to F32)
// - Q4_K quantized input data (dequantized to F32)
// - Int32 row indices
// - Multi-dimensional tensor support
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"
#include "quants.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#define CACHE_LINE_SIZE_BYTES 64

struct ggml_et_get_rows_params {
    struct ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    struct ggml_tensor src1;  // Row indices tensor (I32)
    struct ggml_tensor dst;   // Output tensor (F32)
};

#define CACHE_LINE_SIZE_BYTES     64
#define CACHE_ELEMENTS(elem_size) (CACHE_LINE_SIZE_BYTES / (elem_size))

// Copy a row of F32 data from source to destination
static void copy_f32_row(float * dst, const float * src, int64_t num_elements) {
    // Simple memcpy for F32 data - no conversion needed
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

static void copy_f16_row(float * dst, const uint16_t * src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
static void copy_row_cache_align(float * dst, const float * src, int64_t n_bytes) {
    int num_f32_elem = n_bytes / sizeof(float);

    // Unrolled to do an entire cache line at a time
    __asm__ volatile(
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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(num_f32_elem)
        : [stride_count] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
static void copy_q4_0_row(float * dst, const block_q4_0 * src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK4_0 - 1) / QK4_0;

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ? (num_elements - block_idx * QK4_0) : QK4_0;

        float temp_buffer[QK4_0];
        dequantize_q4_0_block(&src_blocks[block_idx], temp_buffer);

        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK4_0 + i] = temp_buffer[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
static void copy_q8_0_row(float * dst, const block_q8_0 * src_blocks, int64_t num_elements) {
    // Number of Q8_0 blocks needed for this row
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block =
            (block_idx == num_blocks - 1) ? (num_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        float temp_buffer[QK8_0];
        dequantize_q8_0_block(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
static void copy_q4_K_row(float * dst, const block_q4_K * src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK_K - 1) / QK_K;

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ? (num_elements - block_idx * QK_K) : QK_K;

        float temp_buffer[QK_K];
        dequantize_q4_K_block(&src_blocks[block_idx], temp_buffer);

        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK_K + i] = temp_buffer[i];
        }
    }
}

static void dequantize_q8_0_block_cache_aligned(const block_q8_0 * block, float * dst) {
    const int8_t * qs_ptr = block->qs;

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    const int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    float                                      scale          = fp16_to_fp32(block->d);
    __asm__ volatile(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    for (int i = 0; i < 4; i++) {
        __asm__ volatile(
            "fgb.ps      f2, f1(%0)   \n\t"  // Loads 8 bytes from (qs_ptr + indices) and sign-extends to 32-bit int.
            "fcvt.ps.pw  f2, f2, rne  \n\t"  // Convert Int32 to Float32
            "fmul.ps     f2, f2, f0   \n\t"  // f2 = f2 * f0 (scale)
            "fsq2        f2, 0(%1)    \n\t"  // Store 256 bits (8 floats) to dst.

            ::"r"(qs_ptr),
            "r"(dst)
            : "f2", "memory");

        // Advance pointers in C
        qs_ptr += 8;
        dst += 8;
    }
    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
static void copy_q4_0_row_cache_aligned(float * dst, const block_q4_0 * src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    const int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    const int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ volatile(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_0 * block     = &src_blocks[block_idx];
        const uint8_t *    qs        = block->qs;
        float *            block_dst = dst + block_idx * QK4_0;

        float scale = fp16_to_fp32(block->d);
        float bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ volatile(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        for (int i = 0; i < 4; i++) {
            __asm__ volatile(
                "fgb.ps      f2, f1(%0)    \n\t"  // Gather: [b0,b0,b1,b1,b2,b2,b3,b3]
                "mov.m.x     m0, x0, 0xAA  \n\t"  // Odd lanes only (fills gather latency)
                "fsrli.pi    f2, f2, 4     \n\t"  // Odd lanes: byte >> 4 (high nibble)
                "mov.m.x     m0, x0, 0xFF  \n\t"  // Restore full mask
                "fslli.pi    f2, f2, 28    \n\t"  // Isolate low 4 bits: shift left 28
                "fsrli.pi    f2, f2, 28    \n\t"  //   then right 28 -> nibble in [3:0]
                "fcvt.ps.pw  f2, f2, rne   \n\t"  // Int32 -> Float32
                "fmul.ps     f2, f2, f0    \n\t"  // * scale
                "fadd.ps     f2, f2, f3    \n\t"  // + bias -> (nibble - 8) * scale
                "fscw.ps     f2, f4(%1)    \n\t"  // Scatter to GGML positions

                ::"r"(qs),
                "r"(block_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            block_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
static void copy_q8_0_row_cache_aligned(float * dst, const block_q8_0 * src_blocks, int64_t num_elements) {
    // Number of Q8_0 blocks needed for this row
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block =
            (block_idx == num_blocks - 1) ? (num_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        float temp_buffer[QK8_0];
        dequantize_q8_0_block_cache_aligned(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
static void copy_q4_K_row_cache_aligned(float * dst, const block_q4_K * src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    const int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ volatile("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_K * block     = &src_blocks[block_idx];
        const uint8_t *    qs        = block->qs;
        float *            block_dst = dst + block_idx * QK_K;

        const float d   = fp16_to_fp32(block->d);
        const float min = fp16_to_fp32(block->dmin);

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, block->scales, &sc, &m);
            const float d1     = d * sc;
            const float neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, block->scales, &sc, &m);
            const float d2     = d * sc;
            const float neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ volatile(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            const uint8_t * qs_lo  = qs;
            float *         dst_lo = block_dst + j;
            for (int k = 0; k < 4; k++) {
                __asm__ volatile(
                    "fgb.ps      f2, f1(%0)   \n\t"   // Gather 8 bytes, sign-extend to int32
                    "fandi.pi    f2, f2, 0xF   \n\t"  // Mask low nibble (imm10=15)
                    "fcvt.ps.pw  f2, f2, rne   \n\t"  // Int32 -> Float32
                    "fmadd.ps    f2, f2, f0, f3\n\t"  // d1 * nibble + (-m1)
                    "fsq2        f2, 0(%1)     \n\t"  // Store 8 floats
                    ::"r"(qs_lo),
                    "r"(dst_lo)
                    : "f2", "memory");
                qs_lo += 8;
                dst_lo += 8;
            }

            // High nibbles: 32 elements using d2, neg_m2
            __asm__ volatile(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            const uint8_t * qs_hi  = qs;
            float *         dst_hi = block_dst + j + 32;
            for (int k = 0; k < 4; k++) {
                __asm__ volatile(
                    "fgb.ps      f2, f1(%0)   \n\t"   // Gather 8 bytes, sign-extend to int32
                    "fsrli.pi    f2, f2, 4     \n\t"  // Shift right 4: high nibble
                    "fandi.pi    f2, f2, 0xF   \n\t"  // Mask to 4 bits (clean any sign-ext artifacts)
                    "fcvt.ps.pw  f2, f2, rne   \n\t"  // Int32 -> Float32
                    "fmadd.ps    f2, f2, f0, f3\n\t"  // d2 * nibble + (-m2)
                    "fsq2        f2, 0(%1)     \n\t"  // Store 8 floats
                    ::"r"(qs_hi),
                    "r"(dst_hi)
                    : "f2", "memory");
                qs_hi += 8;
                dst_hi += 8;
            }

            qs += 32;  // Advance to next 32 packed bytes
            is += 2;
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
static int64_t get_elements_per_work_unit(int type) {
    const int64_t elements_per_cacheline = CACHE_LINE_SIZE_BYTES / sizeof(float);  // 16
    switch (type) {
        case GGML_TYPE_Q8_0:
            return QK8_0;                   // 32 elements = 2 cachelines
        case GGML_TYPE_Q4_0:
            return QK4_0;                   // 32 elements = 2 cachelines
        case GGML_TYPE_Q4_K:
            return QK_K;                    // 256 elements = 16 cachelines
        default:
            return elements_per_cacheline;  // 16 elements = 1 cacheline
    }
}

static int get_row_f32_mc_cacheline_aligned(struct ggml_et_get_rows_params * params, void * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    int                    thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int                    num_threads = get_num_threads(kernel_env->shire_mask);

    struct ggml_tensor * src0 = &params->src0;  // Data tensor
    struct ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    struct ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    const int64_t ne00 = src0->ne[0];           // Source columns (row width)
    const int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    const int64_t ne02 = src0->ne[2];           // Source batch dimension
    const int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    const int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    const int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    const int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    const int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    const int64_t elements_per_wu = get_elements_per_work_unit(src0->type);
    const int64_t wus_per_row     = ne00 / elements_per_wu;
    const int64_t total_wus       = total_rows_to_extract * wus_per_row;

    // Distribute work units across threads (contiguous ranges)
    const int64_t wus_per_thread = (total_wus + num_threads - 1) / num_threads;
    const int64_t wu_start       = thread_id * wus_per_thread;
    int64_t       wu_end         = wu_start + wus_per_thread;
    if (wu_end > total_wus) {
        wu_end = total_wus;
    }

    void *    src0_data = src0->data;
    int32_t * src1_data = (int32_t *) src1->data;
    float *   dst_data  = (float *) dst->data;

    int64_t wu = wu_start;
    while (wu < wu_end) {
        // Determine which row this work unit belongs to and offset within row
        const int64_t row_idx   = wu / wus_per_row;
        const int64_t wu_in_row = wu % wus_per_row;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_remaining_in_row = wus_per_row - wu_in_row;
        int64_t wus_to_process       = wu_end - wu;
        if (wus_remaining_in_row < wus_to_process) {
            wus_to_process = wus_remaining_in_row;
        }

        // Calculate multi-dimensional index for this row
        const int64_t i       = row_idx;
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        const int32_t row_index    = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) {
            return -1;  // Index out of bounds
        }

        const int64_t batch_offset =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t elem_offset_in_row = wu_in_row * elements_per_wu;
        const int64_t num_elements       = wus_to_process * elements_per_wu;

        float * dst_row = dst_data + row_idx * ne00 + elem_offset_in_row;

        if (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            const float * src_row = (const float *) src0_data + row_index * ne00 + batch_offset + elem_offset_in_row;
            copy_row_cache_align(dst_row, src_row, num_elements * sizeof(float));
        } else if (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            const uint16_t * src_row =
                (const uint16_t *) src0_data + row_index * ne00 + batch_offset + elem_offset_in_row;
            copy_f16_row(dst_row, src_row, num_elements);
        } else if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            const int64_t      blocks_per_row   = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            const int64_t      block_start      = elem_offset_in_row / QK8_0;
            const block_q8_0 * src_blocks       = (const block_q8_0 *) src0_data + src_block_offset + block_start;
            copy_q8_0_row_cache_aligned(dst_row, src_blocks, num_elements);
        } else if (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            const int64_t      blocks_per_row   = (ne00 + QK4_0 - 1) / QK4_0;
            const int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            const int64_t      block_start      = elem_offset_in_row / QK4_0;
            const block_q4_0 * src_blocks       = (const block_q4_0 *) src0_data + src_block_offset + block_start;
            copy_q4_0_row_cache_aligned(dst_row, src_blocks, num_elements);
        } else if (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            const int64_t      blocks_per_row   = (ne00 + QK_K - 1) / QK_K;
            const int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            const int64_t      block_start      = elem_offset_in_row / QK_K;
            const block_q4_K * src_blocks       = (const block_q4_K *) src0_data + src_block_offset + block_start;
            copy_q4_K_row_cache_aligned(dst_row, src_blocks, num_elements);
        }

        wu += wus_to_process;
    }

    return 0;
}

static inline size_t tensor_bytes(const struct ggml_tensor * t) {
    return (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

int entry_point(struct ggml_et_get_rows_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    struct ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    struct ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(sizeof(float)) == 0) {
        return get_row_f32_mc_cacheline_aligned(params, env);
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    if (thread_id != 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;  // Invalid pointer
    }

    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        return -1;  // Invalid output or index type
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        return -1;  // Unsupported input type
    }

    void *    src0_data = src0->data;
    int32_t * src1_data = (int32_t *) src1->data;
    float *   dst_data  = (float *) dst->data;
#ifdef ET_UBERKERNEL
    evict_region_past_l2(src0_data, tensor_bytes(src0));
    evict_region_past_l2(src1_data, tensor_bytes(src1));
    evict_region_past_l2(dst_data, tensor_bytes(dst));
#endif

    if (!src0_data || !src1_data || !dst_data) {
        return -1;  // Null data pointer
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
#ifdef ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    for (int64_t i = 0; i < total_rows_to_extract; i++) {
        // Calculate multi-dimensional index for the current output position
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        const int32_t row_index    = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) {
            return -1;  // Index out of bounds
        }

        const int64_t batch_offset =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t dst_offset = i;

        if (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            const float * src_row = (const float *) src0_data + row_index * ne00 + batch_offset;
            float *       dst_row = dst_data + dst_offset * ne00;
            copy_f32_row(dst_row, src_row, ne00);
        } else if (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            const uint16_t * src_row = (const uint16_t *) src0_data + row_index * ne00 + batch_offset;
            float *          dst_row = dst_data + dst_offset * ne00;
            copy_f16_row(dst_row, src_row, ne00);
        } else if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            const int64_t      blocks_per_row   = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            const block_q8_0 * src_blocks       = (const block_q8_0 *) src0_data + src_block_offset;
            float *            dst_row          = dst_data + dst_offset * ne00;
            copy_q8_0_row(dst_row, src_blocks, ne00);
        } else if (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            const int64_t      blocks_per_row   = (ne00 + QK4_0 - 1) / QK4_0;
            const int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            const block_q4_0 * src_blocks       = (const block_q4_0 *) src0_data + src_block_offset;
            float *            dst_row          = dst_data + dst_offset * ne00;
            copy_q4_0_row(dst_row, src_blocks, ne00);
        } else if (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            const int64_t      blocks_per_row   = (ne00 + QK_K - 1) / QK_K;
            const int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            const block_q4_K * src_blocks       = (const block_q4_K *) src0_data + src_block_offset;
            float *            dst_row          = dst_data + dst_offset * ne00;
            copy_q4_K_row(dst_row, src_blocks, ne00);
        }
    }

    return 0;
}
