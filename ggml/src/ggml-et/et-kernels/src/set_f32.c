//******************************************************************************
// SET F32 Kernel
// Minimal ET implementation for inplace F32 SET into a contiguous destination
// using a contiguous F32 source view and explicit destination view strides.
//
// Supported shape family:
// - dst/base is contiguous F32
// - src1 is contiguous F32
// - src1.ne[0] is cacheline-aligned (multiple of 16 floats)
// - destination view strides/offset are cacheline-aligned
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_set_params {
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

static inline void copy_row_aligned(float * dst, const float * src, int32_t n) {
    for (int32_t i = 0; i < n; i += 8) {
        __asm__ volatile(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float (*)[8]) & dst[i])
            : [src_vec] "m"(*(const float (*)[8]) & src[i])
            : "f11");
    }
}

int entry_point(struct ggml_et_set_params * params, void * env) {
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

    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * dst  = &params->dst;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;
    if (!src1_data || !dst_data) {
        return -1;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || ne10 % 16 != 0) {
        return -1;
    }

    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    const int64_t dnb1   = params->nb1;
    const int64_t dnb2   = params->nb2;
    const int64_t dnb3   = params->nb3;
    const int64_t offset = params->offset;

    const int64_t total_rows = ne11 * ne12 * ne13;

    for (int64_t row = thread_id; row < total_rows; row += num_threads) {
        const int64_t i1 = row % ne11;
        const int64_t i2 = (row / ne11) % ne12;
        const int64_t i3 = row / (ne11 * ne12);

        const float * src_row = (const float *) ((const char *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        float *       dst_row = (float *) ((char *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    return 0;
}
