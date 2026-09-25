//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_cumsum_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

int entry_point(struct ggml_et_cumsum_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    if (!src0_data || !dst_data) {
        return -1;
    }

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const size_t snb0 = src0->nb[0];
    const size_t snb1 = src0->nb[1];
    const size_t snb2 = src0->nb[2];
    const size_t snb3 = src0->nb[3];

    const size_t dnb0 = dst->nb[0];
    const size_t dnb1 = dst->nb[1];
    const size_t dnb2 = dst->nb[2];
    const size_t dnb3 = dst->nb[3];

    if (snb0 != sizeof(float) || dnb0 != sizeof(float)) {
        return -1;
    }

    const int64_t total_rows     = ne1 * ne2 * ne3;
    const int64_t rows_per_group = et_rows_per_cacheline_group(ne0, sizeof(float));
    const int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    for (int64_t grp = thread_id; grp < total_groups; grp += num_threads) {
        const int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        if (row_end > total_rows) {
            row_end = total_rows;
        }

        for (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            const float * src_row = (const float *) ((const char *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            float *       dst_row = (float *) ((char *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            float acc = 0.0f;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    return 0;
}
