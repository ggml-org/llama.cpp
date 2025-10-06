//******************************************************************************
// Bare Metal MUL_MAT_ID Kernel (Mixture of Experts)
//
// ALGORITHM:
// MUL_MAT_ID performs batched matrix multiplication with expert routing.
// Each output element selects which expert matrix to use based on an index tensor.
//
// INPUTS:
//   src0 (as):  Expert weight matrices [K, M, n_expert]
//               - Stack of n_expert matrices, each of size [K, M]
//   src1 (b):   Activation vectors [K, n_cols, batch]
//               - n_cols can be 1 (broadcast) or n_expert_used (per-expert inputs)
//   src2 (ids): Expert selection indices [n_expert_used, batch] (int32)
//               - For each (slot, batch), specifies which expert from src0 to use
//
// OUTPUT:
//   dst:        Result [M, n_expert_used, batch, 1]
//
// COMPUTATION:
//   For each output position (m, slot, batch):
//     expert_id = ids[slot, batch]           // Which expert to use (0..n_expert-1)
//     col_idx = slot % src1.ne[1]            // Which column in src1 (handles broadcasting)
//     dst[m, slot, batch] = dot_product(
//       src0[0:K, m, expert_id],             // Row m from selected expert matrix
//       src1[0:K, col_idx, batch]            // Column from activations (may broadcast)
//     )
//
// BROADCASTING:
//   - When src1.ne[1] == 1: All expert slots use the same activation column
//   - When src1.ne[1] == n_expert_used: Each slot has its own activation column
//   - General case: col_idx = slot % src1.ne[1] (modulo handles both cases)
//
// MATH NOTATION:
//   C[m, s, b] = Sum(k=0 to K-1) A[k, m, ids[s,b]] x B[k, s % ne11, b]
//   where:
//     m: [0, M)               - output feature index
//     s: [0, n_expert_used)   - expert slot index
//     b: [0, batch)           - batch index
//     k: [0, K)               - hidden dimension
//     ne11 = src1->ne[1]       - number of columns in src1
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"

// Compute dot product between dequantized q8_0 block and f32 column vector
static inline float compute_block_dot_product_q8_0(const block_q8_0* a_block, const float* b_col_start) {
    const float scale = fp16_to_fp32(a_block->d);
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk * 8;
        static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"
            "flw.ps f31, %[gather]\n"
            "fgb.ps f11, f31(%[a_ptr])\n"
            "fcvt.ps.pw f11, f11\n"
            "flw.ps f12, %[b_vec]\n"
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_ptr] "r"(&a_block->qs[offset]),
              [b_vec] "m"(*(const float(*)[8])&b_col_start[offset]),
              [gather] "m"(*(const int32_t(*)[8])gather_pattern)
            : "f10", "f11", "f12", "f31"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    return final_sum * scale;
}

// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
static inline float compute_block_dot_product_f16_naive(const uint16_t* a_block, const float* b_col_start) {
    float sum = 0.0f;
    for (int i = 0; i < QK_F16; i++) {
        float a_val = fp16_to_fp32(a_block[i]);
        float b_val = b_col_start[i];
        sum += a_val * b_val;
    }
    return sum;
}

// Compute dot product between f32 block and f32 column vector
static inline float compute_block_dot_product_f32(const float* a_block, const float* b_col_start) {
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int chunk = 0; chunk < 2; chunk++) {
        int offset = chunk * 8;

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"
            "flw.ps f11, %[a_vec]\n"
            "flw.ps f12, %[b_vec]\n"
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_vec] "m"(*(const float(*)[8])(a_block + offset)),
              [b_vec] "m"(*(const float(*)[8])(b_col_start + offset))
            : "f10", "f11", "f12"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    return final_sum;
}

KERNEL_TRAMPOLINE();

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
int entry_point(struct ggml_et_mul_mat_id_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    // Get thread coordination info
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    if (thread_id & 1) {
        return 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    int effective_thread_id = thread_id / 2;
    int effective_num_threads = (num_threads + 1) / 2;  // Ceiling division

    // Validate params
    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Expert weight matrices [K, M, n_expert]
    struct ggml_tensor* src1 = &params->src1; // Activations [K, n_expert_used, batch]
    struct ggml_tensor* src2 = &params->src2; // Expert indices [n_expert_used, batch] (I32)
    struct ggml_tensor* dst = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        return -1;
    }

    // FIXME: temporary, run F16 in single thread
    if (src0->type == GGML_TYPE_F16) {
        effective_num_threads = 1;
        if (effective_thread_id != 0) {
            return 0;
        }
    }

    // Get data pointers
    const void* src0_data = src0->data;           // Expert matrices (Q8_0/F16/F32)
    const float* src1_data = (const float*)src1->data;  // Activations (F32)
    const int32_t* src2_data = (const int32_t*)src2->data;  // Expert IDs (I32)
    float* dst_data = (float*)dst->data;          // Output (F32)

    if (!src0_data || !src1_data || !src2_data || !dst_data) {
        return -1;
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
            return -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    const int64_t K = src0->ne[0];              // Hidden dimension
    const int64_t M = src0->ne[1];              // Output features
    const int64_t n_expert = src0->ne[2];       // Number of experts
    const int64_t n_expert_used = src2->ne[0];  // Experts used per token
    const int64_t batch = src2->ne[1];          // Batch size

    // Strides (in bytes)
    const size_t nb01 = src0->nb[1];   // src0 row stride
    const size_t nb02 = src0->nb[2];   // src0 expert stride
    const size_t nb11 = src1->nb[1];   // src1 column stride
    const size_t nb12 = src1->nb[2];   // src1 batch stride
    const size_t nb20 = src2->nb[0];   // src2 element stride
    const size_t nb21 = src2->nb[1];   // src2 batch stride
    const size_t nb1  = dst->nb[1];    // dst column stride
    const size_t nb2  = dst->nb[2];    // dst batch stride

    // Verify K dimension alignment
    if (K % block_size != 0) {
        return -1;
    }

    // Verify first dimension is contiguous
    size_t expected_element_size_src0;
    if (src0->type == GGML_TYPE_Q8_0) {
        expected_element_size_src0 = sizeof(block_q8_0);
    } else if (src0->type == GGML_TYPE_F16) {
        expected_element_size_src0 = sizeof(uint16_t);
    } else if (src0->type == GGML_TYPE_F32) {
        expected_element_size_src0 = sizeof(float);
    } else {
        return -1;
    }

    if (src0->nb[0] != expected_element_size_src0 ||
        src1->nb[0] != sizeof(float) ||
        src2->nb[0] != sizeof(int32_t) ||
        dst->nb[0] != sizeof(float)) {
        return -1;
    }

    const int64_t K_blocks = K / block_size;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    const uint64_t total_elements = M * n_expert_used * batch;

    const uint64_t per_thread = 16;
    const uint64_t threads_stride = per_thread * effective_num_threads;

    if (effective_thread_id * per_thread >= total_elements) {
        return 0;
    }

    // Process elements assigned to this thread
    for (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        for (uint64_t j = 0; j < per_thread; j++) {
            const uint64_t idx = base_idx + j;

            if (idx >= total_elements) {
                break;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            const int64_t batch_idx = idx / (M * n_expert_used);
            const int64_t rem = idx % (M * n_expert_used);
            const int64_t n_idx = rem / M;
            const int64_t m = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            const int32_t expert_id = *(const int32_t*)((const char*)src2_data +
                                                        n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            if (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                volatile float* dst_element = (volatile float*)((char*)dst_data +
                                              m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                continue;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            const int64_t col_idx = n_idx % src1->ne[1];
            float sum = 0.0f;

            for (int64_t kb = 0; kb < K_blocks; kb++) {
                // Get pointer to activation column at row kb*block_size
                const float* b_col_start = (const float*)((const char*)src1_data +
                                                         (kb * block_size) * src1->nb[0] +
                                                         col_idx * nb11 + batch_idx * nb12);

                // Compute block dot product based on expert matrix type
                switch (src0->type) {
                    case GGML_TYPE_Q8_0: {
                        const block_q8_0* expert_row = (const block_q8_0*)((const char*)src0_data +
                                                                           m * nb01 + expert_id * nb02);
                        sum += compute_block_dot_product_q8_0(&expert_row[kb], b_col_start);
                        break;
                    }
                    case GGML_TYPE_F16: {
                        const uint16_t* expert_row = (const uint16_t*)((const char*)src0_data +
                                                                       m * nb01 + expert_id * nb02);
                        sum += compute_block_dot_product_f16_naive(&expert_row[kb * block_size], b_col_start);
                        break;
                    }
                    case GGML_TYPE_F32: {
                        const float* expert_row = (const float*)((const char*)src0_data +
                                                                 m * nb01 + expert_id * nb02);
                        sum += compute_block_dot_product_f32(&expert_row[kb * block_size], b_col_start);
                        break;
                    }
                    default:
                        return -1;
                }
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            volatile float* dst_element = (volatile float*)((char*)dst_data +
                                          m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, sum);
        }
    }

    return 0;
}
