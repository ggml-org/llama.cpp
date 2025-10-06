//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
static inline float compute_block_dot_product_q8_0(const block_q8_0* a_block, const float* b_col_start) {
    const float scale = fp16_to_fp32(a_block->d);

    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk * 8;

        // Vectorized int8→float conversion + multiply-accumulate
        // Using gather pattern for byte loading and vector conversion
        static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
            "flw.ps f31, %[gather]\n"                // Load gather pattern into f31
            "fgb.ps f11, f31(%[a_ptr])\n"            // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"                  // Convert int8 vector to float vector
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (floats)
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            "fsw.ps f10, %[result]\n"                // Store back to accumulator

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_ptr] "r"(&a_block->qs[offset]),
              [b_vec] "m"(*(const float(*)[8])&b_col_start[offset]),
              [gather] "m"(*(const int32_t(*)[8])gather_pattern)
            : "f10", "f11", "f12", "f31"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    return final_sum * scale;
}

// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
// Scalar implementation for debugging - no vectorization
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16_naive(const uint16_t* a_block, const float* b_col_start) {
    float sum = 0.0f;

    // Simple scalar loop - convert each f16 to f32 and multiply
    for (int i = 0; i < QK_F16; i++) {
        float a_val = fp16_to_fp32(a_block[i]);
        float b_val = b_col_start[i];
        sum += a_val * b_val;
    }

    return sum;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16(const uint16_t* a_block, const float* b_col_start) {
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Process 32 f16 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk * 8;

        // Vectorized f16→f32 conversion + multiply-accumulate
        // Using gather pattern for f16 loading and vector conversion
        static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
            "flw.ps f31, %[gather]\n"                // Load gather pattern into f31
            "fgh.ps f11, f31(%[a_ptr])\n"            // Gather 8 f16 values from A using pattern
            "fcvt.ps.f16 f11, f11\n"                 // Convert f16 vector to f32 vector (8 values)
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (already f32)
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            "fsw.ps f10, %[result]\n"                // Store back to accumulator

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_ptr] "r"((const char*)a_block + offset * sizeof(uint16_t)),
              [b_vec] "m"(*(const float(*)[8])(b_col_start + offset)),
              [gather] "m"(*(const int32_t(*)[8])gather_pattern)
            : "f10", "f11", "f12", "f31"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    return final_sum;
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f32(const float* a_block, const float* b_col_start) {
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Process 16 f32 elements in 2 chunks of 8 elements each
    for (int chunk = 0; chunk < 2; chunk++) {
        int offset = chunk * 8;

        // Vectorized f32 multiply-accumulate
        __asm__ volatile(
            "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
            "flw.ps f11, %[a_vec]\n"                 // Load 8 A values (f32)
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (f32)
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            "fsw.ps f10, %[result]\n"                // Store back to accumulator

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_vec] "m"(*(const float(*)[8])(a_block + offset)),
              [b_vec] "m"(*(const float(*)[8])(b_col_start + offset))
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    return final_sum;
}

KERNEL_TRAMPOLINE();

// Main entry point for MUL_MAT Q8_0 x F32 kernel
int entry_point(struct ggml_et_binary_params* params, void* env) {
    // Cast env to proper type
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    // Validate environment pointer
    if (!kernel_env) {
        return -1;
    }

    // Get thread coordination info for B row-based parallelization
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    // Handle invalid thread ID
    if (thread_id < 0) {
        return -1;
    }

    // HACK: Only use even thread IDs (thread 0, 2, 4, ...) to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    if (thread_id & 1) {
        return 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    int effective_thread_id = thread_id / 2;
    int effective_num_threads = (num_threads + 1) / 2;  // Ceiling division

    // Basic safety check on params
    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Weight matrix A (q8_0)
    struct ggml_tensor* src1 = &params->src1; // Activation matrix B (f32)
    struct ggml_tensor* dst = &params->dst;   // Output matrix C (f32)

    // Validate tensor types - src1 and dst must be f32, src0 can be quantized
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination - src1 and dst must be f32
    }

    // FIXME: temporary, run F16 in single thread.
    if (src0->type == GGML_TYPE_F16) {
        effective_num_threads = 1;
        if (effective_thread_id != 0) {
            return 0;
        }
    }

    // Get data pointers
    const void* src0_data = src0->data;                          // Quantized weight blocks
    const float* src1_data = (const float*)src1->data;           // F32 activations
    float* dst_data = (float*)dst->data;                         // F32 output

    // Validate data pointers
    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    // Determine block size based on src0 type
    int block_size;
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            block_size = QK8_0;
            break;
        case GGML_TYPE_F16:
            block_size = QK_F16;
            break;
        case GGML_TYPE_F32:
            block_size = QK_F32;
            break;
        default:
            return -1; // Unsupported src0 type
    }

    // Get matrix dimensions (following GGML convention)
    // src0 (A): [ne00=K, ne01=M, ne02, ne03]
    // src1 (B): [ne10=K, ne11=N, ne12, ne13]
    // dst (C):  [ne0=M, ne1=N, ne2, ne3]
    const int64_t K  = src0->ne[0];  // Hidden dimension
    const int64_t M  = src0->ne[1];  // Output features (rows in result)
    const int64_t N  = src1->ne[1];  // Sequence length (cols in result)

    // Higher dimensions for batch processing
    const int64_t ne02 = src0->ne[2];  // src0 batch dim 2
    const int64_t ne03 = src0->ne[3];  // src0 batch dim 3
    const int64_t ne12 = src1->ne[2];  // src1 batch dim 2
    const int64_t ne13 = src1->ne[3];  // src1 batch dim 3
    const int64_t ne2  = dst->ne[2];   // dst batch dim 2
    const int64_t ne3  = dst->ne[3];   // dst batch dim 3

    // Strides (in bytes)
    const size_t nb01 = src0->nb[1];   // src0 row stride
    const size_t nb02 = src0->nb[2];   // src0 batch stride 2
    const size_t nb03 = src0->nb[3];   // src0 batch stride 3
    const size_t nb11 = src1->nb[1];   // src1 row stride
    const size_t nb12 = src1->nb[2];   // src1 batch stride 2
    const size_t nb13 = src1->nb[3];   // src1 batch stride 3
    const size_t nb1  = dst->nb[1];    // dst row stride
    const size_t nb2  = dst->nb[2];    // dst batch stride 2
    const size_t nb3  = dst->nb[3];    // dst batch stride 3

    // Verify K dimension alignment for quantization (must be multiple of block_size)
    if (K % block_size != 0) {
        return -1; // K dimension not aligned to quantization block size
    }

    // Verify first dimension is contiguous (required assumption)
    size_t expected_element_size_src0;
    if (src0->type == GGML_TYPE_Q8_0) {
        expected_element_size_src0 = sizeof(block_q8_0);  // Q8_0 blocks
    } else if (src0->type == GGML_TYPE_F16) {
        expected_element_size_src0 = sizeof(uint16_t);    // F16 elements
    } else if (src0->type == GGML_TYPE_F32) {
        expected_element_size_src0 = sizeof(float);       // F32 elements
    } else {
        return -1; // Unsupported type
    }

    const size_t expected_element_size_src1 = sizeof(float);
    const size_t expected_element_size_dst = sizeof(float);

    if (src0->nb[0] != expected_element_size_src0 ||
        src1->nb[0] != expected_element_size_src1 ||
        dst->nb[0] != expected_element_size_dst) {
        return -1; // First dimension must be contiguous
    }

    const int64_t K_blocks = K / block_size;  // Number of quantized blocks per row

    // Threading: distribute output elements across threads
    const uint64_t total_elements = M * N * ne2 * ne3;

    // Cache line is 64 bytes = 16 floats
    const uint64_t per_thread = 16;  // Process 16 elements per thread per iteration
    const uint64_t threads_stride = per_thread * effective_num_threads;

    // Early exit if this thread is beyond the data
    if (effective_thread_id * per_thread >= total_elements) {
        return 0;
    }

    // Broadcasting support (exactly like GGML CPU backend)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // Process elements assigned to this thread
    for (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        // Process up to per_thread elements
        for (uint64_t j = 0; j < per_thread; j++) {
            const uint64_t idx = base_idx + j;

            if (idx >= total_elements) {
                break;
            }

            // Decode linear index to (m, n, i2, i3)
            // Layout: m + M * (n + N * (i2 + ne2 * i3))
            const int64_t i3 = idx / (M * N * ne2);
            const int64_t rem3 = idx % (M * N * ne2);
            const int64_t i2 = rem3 / (M * N);
            const int64_t rem2 = rem3 % (M * N);
            const int64_t n = rem2 / M;
            const int64_t m = rem2 % M;

            // Broadcasting indices for src0
            const int64_t i03 = i3 / r3;
            const int64_t i02 = i2 / r2;

            // Broadcasting indices for src1
            const int64_t i13 = (ne13 > 1) ? i3 : 0;
            const int64_t i12 = (ne12 > 1) ? i2 : 0;

            // Compute dot product: A[m, :] . B[:, n]
            float sum = 0.0f;

            for (int64_t kb = 0; kb < K_blocks; kb++) {
                // Get pointer to B column at row kb*block_size
                const float* b_col_start = (const float*)((const char*)src1_data +
                                                         (kb * block_size) * src1->nb[0] +
                                                         n * nb11 + i12 * nb12 + i13 * nb13);

                // Compute block dot product based on type
                switch (src0->type) {
                    case GGML_TYPE_Q8_0: {
                        const block_q8_0* q8_row = (const block_q8_0*)((const char*)src0_data +
                                                                       m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_q8_0(&q8_row[kb], b_col_start);
                        break;
                    }
                    case GGML_TYPE_F16: {
                        const uint16_t* f16_row = (const uint16_t*)((const char*)src0_data +
                                                                    m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_start);
                        break;
                    }
                    case GGML_TYPE_F32: {
                        const float* f32_row = (const float*)((const char*)src0_data +
                                                              m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_start);
                        break;
                    }
                    default:
                        return -1;
                }
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line
            volatile float* c_element = (volatile float*)((char*)dst_data +
                                       m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    return 0;
}
