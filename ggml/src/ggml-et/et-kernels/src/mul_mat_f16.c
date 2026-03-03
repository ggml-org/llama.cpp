//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

// Main entry point for MUL_MAT Q8_0 x F32 kernel
int entry_point(struct ggml_et_binary_params* params, void* env) {
    // Cast env to proper type
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    uint64_t hart_id = get_hart_id();
    uint64_t minion_id = (hart_id >> 1) & 0x1F; // 2 harts per minion
    uint64_t shire_id_orig = hart_id >> 6;

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Weight matrix A (q8_0)
    struct ggml_tensor* src1 = &params->src1; // Activation matrix B (f32)
    struct ggml_tensor* dst = &params->dst;   // Output matrix C (f32)

    const uint64_t per_thread = 1;  // Process 16 elements per thread per iteration
    int effective_num_threads = 1 << 10; //1024;
    int effective_thread_id = hart_id >> 1; // x minion_id x




    // Get data pointers
    const void* src0_data = src0->data;                          // Quantized weight blocks
    const float* src1_data = (const float*)src1->data;           // F32 activations
    float* dst_data = (float*)dst->data;                         // F32 output


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
            block_size = 16;//QK_F32;
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

    const int64_t K_blocks = K / block_size;  // Number of quantized blocks per row

    // Threading: distribute output elements across threads
    const uint64_t total_elements = M * N * ne2 * ne3;

    // Cache line is 64 bytes = 16 floats
    // const uint64_t per_thread = 16;  // Process 16 elements per thread per iteration
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

            // Process full blocks
            for (int64_t kb = 0; kb < K_blocks; kb++) {
                // Get pointer to B column at row kb*block_size
                const float* b_col_start = (const float*)((const char*)src1_data +
                                                         (kb * block_size) * src1->nb[0] +
                                                         n * nb11 + i12 * nb12 + i13 * nb13);


                const uint16_t* f16_row = (const uint16_t*)((const char*)src0_data +
                                                                    m * nb01 + i02 * nb02 + i03 * nb03);
                sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_start);
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
