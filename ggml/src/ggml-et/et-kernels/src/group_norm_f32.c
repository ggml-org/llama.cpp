//******************************************************************************
// GROUP_NORM F32 Kernel
// Baseline scalar implementation:
//   normalize over (ne0 * ne1 * channels_in_group) for each (group, batch).
//
// Parallelization:
// - Work is partitioned across (group, batch) pairs.
// - For non-cache-aligned ne0, writes are emitted in row-groups so each thread's
//   destination write footprint still spans an integer number of cache lines.
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_group_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t            n_groups;
    float              eps;
};

int entry_point(struct ggml_et_group_norm_params * params, void * env) {
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

    const float * src0_data = (const float *) src0->data;
    float *       dst_data  = (float *) dst->data;

    if (!src0_data || !dst_data) {
        return -1;
    }

    const int32_t n_groups = params->n_groups;
    const float   eps      = params->eps;

    if (n_groups <= 0 || eps < 0.0f) {
        return -1;
    }

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1;
    }

    const int64_t nb1  = dst->nb[1];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    if (channels_per_group <= 0) {
        return -1;
    }

    const int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    const int64_t total_work           = active_groups * ne3;
    const int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, sizeof(float));

    for (int64_t work = thread_id; work < total_work; work += num_threads) {
        const int64_t i3        = work / active_groups;
        const int64_t group_idx = work % active_groups;

        const int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        if (channel_end > ne2) {
            channel_end = ne2;
        }

        const int64_t channel_count = channel_end - channel_start;
        if (channel_count <= 0) {
            continue;
        }

        float sum   = 0.0f;
        float denom = 0.0f;
        for (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const float * src_row = (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        const float mean = et_fdiv(sum, denom);

        float var_sum = 0.0f;
        for (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const float * src_row = (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        const float variance = et_fdiv(var_sum, denom);
        const float scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        if (ne0 % 16 == 0) {
            for (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const float * src_row =
                        (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    float * dst_row = (float *) ((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } else {
            const int64_t total_rows_in_group = channel_count * ne1;
            const int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            for (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                const int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                if (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                for (int64_t row = row_start; row < row_end; ++row) {
                    const int64_t local_i2 = row / ne1;
                    const int64_t i1       = row % ne1;
                    const int64_t i2       = channel_start + local_i2;

                    const float * src_row =
                        (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    float * dst_row = (float *) ((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    return 0;
}
