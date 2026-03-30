//******************************************************************************
// SSM_SCAN F32 Kernel
// Scalar reference-style implementation matching ggml_compute_forward_ssm_scan_f32.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

struct ggml_et_ssm_scan_params {
    struct ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    struct ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    struct ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    struct ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    struct ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    struct ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    struct ggml_tensor src6;  // ids: [n_seqs] i32
    struct ggml_tensor dst;   // packed [y, final_state]
};

static inline float softplus_f32(float x) {
    return x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

int entry_point(struct ggml_et_ssm_scan_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * src2 = &params->src2;
    struct ggml_tensor * src3 = &params->src3;
    struct ggml_tensor * src4 = &params->src4;
    struct ggml_tensor * src5 = &params->src5;
    struct ggml_tensor * src6 = &params->src6;
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float * s_data  = (const float *) src0->data;
    const float * x_data  = (const float *) src1->data;
    const float * dt_data = (const float *) src2->data;
    const float * A_data  = (const float *) src3->data;
    const float * B_data  = (const float *) src4->data;
    const float * C_data  = (const float *) src5->data;
    const int32_t * ids   = (const int32_t *) src6->data;
    float * dst_data      = (float *) dst->data;

    if (!s_data || !x_data || !dt_data || !A_data || !B_data || !C_data || !ids || !dst_data) {
        return -1;
    }

    const int64_t d_state      = src0->ne[0];
    const int64_t head_dim     = src0->ne[1];
    const int64_t n_head       = src1->ne[1];
    const int64_t n_group      = src4->ne[1];
    const int64_t n_seq_tokens = src1->ne[2];
    const int64_t n_seqs       = src1->ne[3];
    const int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];

    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || src2->nb[0] != sizeof(float) ||
        src3->nb[0] != sizeof(float) || src4->nb[0] != sizeof(float) || src5->nb[0] != sizeof(float) ||
        src6->nb[0] != sizeof(int32_t) || dst->nb[0] != sizeof(float)) {
        return -1;
    }

    if (n_group <= 0 || n_head % n_group != 0) {
        return -1;
    }

    const int64_t heads_per_cacheline = head_dim >= 16 ? 1 : (16 / head_dim);
    const int64_t heads_per_block = heads_per_cacheline > 0 ? heads_per_cacheline : 1;
    const int64_t blocks_per_seq = (n_head + heads_per_block - 1) / heads_per_block;
    const int64_t total_blocks = n_seqs * blocks_per_seq;
    const int64_t blocks_per_thread = (total_blocks + num_threads - 1) / num_threads;
    const int64_t block_begin = (int64_t) thread_id * blocks_per_thread;
    int64_t block_end = block_begin + blocks_per_thread;

    if (block_begin >= total_blocks) {
        return 0;
    }

    if (block_end > total_blocks) {
        block_end = total_blocks;
    }

    for (int64_t block = block_begin; block < block_end; ++block) {
        const int64_t seq_idx = block / blocks_per_seq;
        const int64_t block_in_seq = block % blocks_per_seq;
        const int64_t head_begin = block_in_seq * heads_per_block;
        int64_t head_end = head_begin + heads_per_block;

        if (head_end > n_head) {
            head_end = n_head;
        }

        const int32_t state_seq = ids[seq_idx];

        for (int64_t head_idx = head_begin; head_idx < head_end; ++head_idx) {
            const int64_t group_idx = head_idx / (n_head / n_group);

            for (int64_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                const float * state_src = (const float *) ((const char *) s_data +
                    (size_t) dim_idx * src0->nb[1] +
                    (size_t) head_idx * src0->nb[2] +
                    (size_t) state_seq * src0->nb[3]);

                float * state_dst = (float *) ((char *) dst_data +
                    (size_t) y_elems * sizeof(float) +
                    (size_t) dim_idx * src0->nb[1] +
                    (size_t) head_idx * src0->nb[2] +
                    (size_t) seq_idx * src0->nb[3]);

                for (int64_t token_idx = 0; token_idx < n_seq_tokens; ++token_idx) {
                    const float * x_ptr = (const float *) ((const char *) x_data +
                        (size_t) dim_idx * src1->nb[0] +
                        (size_t) head_idx * src1->nb[1] +
                        (size_t) token_idx * src1->nb[2] +
                        (size_t) seq_idx * src1->nb[3]);

                    const float * dt_ptr = (const float *) ((const char *) dt_data +
                        (size_t) head_idx * src2->nb[0] +
                        (size_t) token_idx * src2->nb[1] +
                        (size_t) seq_idx * src2->nb[2]);

                    const float dt_softplus = softplus_f32(*dt_ptr);
                    const float x_dt = (*x_ptr) * dt_softplus;
                    float sumf = 0.0f;

                    for (int64_t state_idx = 0; state_idx < d_state; ++state_idx) {
                        const float prev_state = token_idx == 0 ? state_src[state_idx] : state_dst[state_idx];

                        const float * A_ptr = (const float *) ((const char *) A_data +
                            (size_t) (src3->ne[0] == 1 ? 0 : state_idx) * src3->nb[0] +
                            (size_t) head_idx * src3->nb[1]);

                        const float * B_ptr = (const float *) ((const char *) B_data +
                            (size_t) state_idx * src4->nb[0] +
                            (size_t) group_idx * src4->nb[1] +
                            (size_t) token_idx * src4->nb[2] +
                            (size_t) seq_idx * src4->nb[3]);

                        const float * C_ptr = (const float *) ((const char *) C_data +
                            (size_t) state_idx * src5->nb[0] +
                            (size_t) group_idx * src5->nb[1] +
                            (size_t) token_idx * src5->nb[2] +
                            (size_t) seq_idx * src5->nb[3]);

                        const float dA = et_expf(dt_softplus * (*A_ptr));
                        const float state = (prev_state * dA) + ((*B_ptr) * x_dt);

                        state_dst[state_idx] = state;
                        sumf += state * (*C_ptr);
                    }

                    dst_data[seq_idx * (n_seq_tokens * n_head * head_dim) +
                             token_idx * (n_head * head_dim) +
                             head_idx * head_dim +
                             dim_idx] = sumf;
                }
            }
        }
    }

    return 0;
}
