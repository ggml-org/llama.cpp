//******************************************************************************
// Element-wise Map F32 Kernel
// Element-wise operations: dst[i] = src0[i] op src1[i]
// Supports: MUL, ADD (more operations to be added later)
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

// TODO: only even threads

// Block operation implementations using ET vector instructions
static inline void block_mul_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector multiplication
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fmul.ps f12, f10, f11\n"          // dst = src0 * src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static inline void block_add_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector addition
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fadd.ps f12, f10, f11\n"          // dst = src0 + src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static inline void block_sub_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector addition
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fsub.ps f12, f10, f11\n"          // dst = src0 + src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}


// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes via fbc.ps
static inline void block_mul_broadcast(float* dst_block, const float* src0_block, float scalar, int elements) {
    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "fbc.ps f11, %[s]\n"
            "fmul.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [s] "m"(scalar)
            : "f10", "f11", "f12"
        );
    }
}

static inline void block_add_broadcast(float* dst_block, const float* src0_block, float scalar, int elements) {
    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "fbc.ps f11, %[s]\n"
            "fadd.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [s] "m"(scalar)
            : "f10", "f11", "f12"
        );
    }
}

static inline void block_sub_broadcast(float* dst_block, const float* src0_block, float scalar, int elements) {
    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "fbc.ps f11, %[s]\n"
            "fsub.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [s] "m"(scalar)
            : "f10", "f11", "f12"
        );
    }
}

int entry_point(struct ggml_et_binary_params* params, void* env) {
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
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    float* src0_data = (float*)src0->data;
    float* src1_data = (float*)src1->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    enum ggml_op operation = dst->op;

    if (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        return -1; // Unsupported operation
    }

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Calculate total number of rows (flatten dimensions 1,2,3)
    const int64_t total_rows = ne1 * ne2 * ne3;

    // Distribute rows across threads using ceiling division to handle remainder
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t start_row = thread_id * rows_per_thread;
    const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    if (start_row >= total_rows) {
        return 0;
    }

    bool cache_aligned = (dst->ne[0] % 16 == 0);
    if(!cache_aligned) {
        return 1;
    }

    for (int64_t ir = start_row; ir < end_row; ir++) {
        // Convert flat row index to 3D coordinates
        const int64_t i03 = ir / (ne2 * ne1);
        const int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        const int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        float* dst_ptr = (float*)((char*)dst_data + i03*nb3 + i02*nb2 + i01*nb1);
        const float* src0_ptr = (const float*)((const char*)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
        const float* src1_ptr = (const float*)((const char*)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

        if (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            float scalar = src1_ptr[0];
            switch (operation) {
                case GGML_OP_MUL:
                    block_mul_broadcast(dst_ptr, src0_ptr, scalar, (int)ne0);
                    break;
                case GGML_OP_ADD:
                    block_add_broadcast(dst_ptr, src0_ptr, scalar, (int)ne0);
                    break;
                case GGML_OP_SUB:
                    block_sub_broadcast(dst_ptr, src0_ptr, scalar, (int)ne0);
                    break;
                default:
                    return 1;
            }
        } else {
            // Broadcasting in dimension 0: src1 repeats across src0
            const int64_t nr0 = ne0 / ne10;

            for (int64_t r = 0; r < nr0; r++) {
                const float* src0_block = src0_ptr + r * ne10;
                float* dst_block = dst_ptr + r * ne10;

                switch (operation) {
                    case GGML_OP_MUL:
                        block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10);
                        break;
                    case GGML_OP_ADD:
                        block_add_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10);
                        break;
                    case GGML_OP_SUB:
                        block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10);
                        break;
                    default:
                        return 1;
                }
            }
        }
    }

    return 0;
}
