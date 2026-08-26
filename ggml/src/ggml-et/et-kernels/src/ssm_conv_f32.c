#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

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

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
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
    float *       dst_data  = (float *) dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }

    const int64_t nc  = src1->ne[0];
    const int64_t ncs = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    if (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != sizeof(float) ||
        src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src0->nb[1] != (size_t) ncs * sizeof(float) ||
        src1->nb[1] != (size_t) nc * sizeof(float)) {
        return -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    const int64_t chunk    = 16;
    const int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *) ((const char *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            float *       x = (float *) ((char *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            for (int64_t ci = thread_id; ci < n_chunks; ci += num_threads) {
                const int64_t i1_start = ci * chunk;
                const int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                for (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    float tmp_s[8], tmp_c[8];
                    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    for (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        for (int j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ volatile(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(float (*)[8]) acc)
                            : [acc] "m"(*(const float (*)[8]) acc), [sv] "m"(*(const float (*)[8]) tmp_s),
                              [cv] "m"(*(const float (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ volatile(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(float (*)[8])(x + i1))
                        : [acc] "m"(*(const float (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                for (; i1 < i1_end; ++i1) {
                    const float * c     = src1_data + i1 * nc;
                    const float * s_row = s + i1 * ncs;
                    float         sumf  = 0.0f;
                    for (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));

    return 0;
}
