//******************************************************************************
// MUL_MAT_ID kernel specialized for Q8_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q8_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy mirrors mul_mat_id_Q4_0.c.
//******************************************************************************

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"

#include <stdint.h>

int entry_point(struct ggml_et_mul_mat_id_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * src2 = &params->src2;
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const void *    src0_data = src0->data;
    const float *   src1_data = (const float *) src1->data;
    const int32_t * src2_data = (const int32_t *) src2->data;
    float *         dst_data  = (float *) dst->data;
    if (!src0_data || !src1_data || !src2_data || !dst_data) {
        return -1;
    }

    const int64_t K             = src0->ne[0];
    const int64_t M             = src0->ne[1];
    const int64_t n_expert      = src0->ne[2];
    const int64_t n_expert_used = src2->ne[0];
    const int64_t batch         = src2->ne[1];
    const int64_t ne11          = src1->ne[1];

    if (K % QK8_0 != 0) {
        return -1;
    }

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb20 = src2->nb[0];
    const size_t nb21 = src2->nb[1];
    const size_t nbd0 = dst->nb[0];
    const size_t nbd1 = dst->nb[1];
    const size_t nbd2 = dst->nb[2];

    if (src0->nb[0] != sizeof(block_q8_0) || src1->nb[0] != sizeof(float) || src2->nb[0] != sizeof(int32_t) ||
        nbd0 != sizeof(float)) {
        return -1;
    }

    const int64_t K_blocks = K / QK8_0;
    const int     use_x2   = ((nb01 & 31) == 0);

    const uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    if (total_outputs == 0) {
        return 0;
    }

    const uint64_t chunk    = (total_outputs + (uint64_t) num_threads - 1) / (uint64_t) num_threads;
    const uint64_t my_start = (uint64_t) thread_id * chunk;
    if (my_start >= total_outputs) {
        return 0;
    }
    uint64_t my_end = my_start + chunk;
    if (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    const uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    while (idx < my_end) {
        const int64_t  batch_idx = (int64_t) (idx / per_batch);
        const uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        const int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        const int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        const uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        const uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        const int32_t expert_id =
            *(const int32_t *) ((const char *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        char * dst_slot = (char *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        if (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            for (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((volatile float *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            continue;
        }

        const int64_t col_idx = slot_idx % ne11;
        const float * b_col_base =
            (const float *) ((const char *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        const char * expert_base = (const char *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t left = run_len;

        if (use_x2) {
            while (left >= 2) {
                const block_q8_0 * row0 = (const block_q8_0 *) (expert_base + m * (int64_t) nb01);
                const block_q8_0 * row1 = (const block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                float              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((volatile float *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((volatile float *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                left -= 2;
            }
        }

        while (left > 0) {
            const block_q8_0 * row = (const block_q8_0 *) (expert_base + m * (int64_t) nb01);
            float              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((volatile float *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            left--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    return 0;
}
