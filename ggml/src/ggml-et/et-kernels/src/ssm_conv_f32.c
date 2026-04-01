//******************************************************************************
// SSM_CONV F32 Kernel
// First-pass scalar implementation matching ggml_compute_forward_ssm_conv_f32.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

struct ggml_et_ssm_conv_params {
    struct ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    struct ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    struct ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

int entry_point(struct ggml_et_ssm_conv_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float * dst_data = (float *) dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }

    const int64_t nc  = src1->ne[0];
    const int64_t ncs = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    if (dst->ne[0] != nr ||
        src1->ne[1] != nr ||
        ncs != nc - 1 + n_t ||
        src0->nb[0] != sizeof(float) ||
        src1->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float) ||
        src0->nb[1] != (size_t) ncs * sizeof(float) ||
        src1->nb[1] != (size_t) nc * sizeof(float)) {
        return -1;
    }

    const int64_t chunk = 16;
    const int64_t n_chunks = (nr + chunk - 1) / chunk;

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *) ((const char *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            float * x = (float *) ((char *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            for (int64_t ci = thread_id; ci < n_chunks; ci += num_threads) {
                const int64_t i1_start = ci * chunk;
                const int64_t i1_end = i1_start + chunk < nr ? i1_start + chunk : nr;

                for (int64_t i1 = i1_start; i1 < i1_end; ++i1) {
                    const float * c = (const float *) ((const char *) src1_data + i1 * src1->nb[1]);
                    const float * s_row = s + i1 * ncs;
                    float sumf = 0.0f;
                    for (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    return 0;
}
