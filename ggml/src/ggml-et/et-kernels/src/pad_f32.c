//******************************************************************************
// Bare Metal PAD F32 Kernel
// Zero-pads an F32 tensor along dimensions 1-3.
//
// Constraints:
//   - No dim0 padding (lp[0]==0, rp[0]==0)
//   - dst contiguous
//   - src nb[0] == 4 (dim0 contiguous for vectorized reads)
//   - Zero-pad only (no circular mode)
//
// Two paths:
//   Aligned (ne0 % 16 == 0): rows distributed across harts, vectorized.
//   Small   (16 % ne0 == 0): cache-line distributed, scalar per-element.
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_pad_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
static inline void vec_copy_f32(float * dst, const float * src, int32_t n) {
    int32_t       i       = 0;
    const int32_t vec_end = (n / 8) * 8;
    for (; i < vec_end; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(float (*)[8]) & dst[i])
            : [s] "m"(*(const float (*)[8]) & src[i])
            : "f10");
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

int entry_point(struct ggml_et_pad_params * params, void * env) {
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

    // Dst dimensions
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    const int64_t nb1_src = src0->nb[1];
    const int64_t nb2_src = src0->nb[2];
    const int64_t nb3_src = src0->nb[3];

    // Padding values
    const int32_t lp1 = params->lp[1];
    const int32_t rp1 = params->rp[1];
    const int32_t lp2 = params->lp[2];
    const int32_t rp2 = params->rp[2];
    const int32_t lp3 = params->lp[3];
    const int32_t rp3 = params->rp[3];

    const int64_t total_rows     = ne1 * ne2 * ne3;
    const int64_t total_elements = ne0 * total_rows;

    if (total_elements == 0) {
        return 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    float zero = 0.0f;
    __asm__ volatile("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    if (ne0 % 16 == 0) {
        for (int64_t row = thread_id; row < total_rows; row += num_threads) {
            const int64_t i3 = row / (ne1 * ne2);
            const int64_t i2 = (row / ne1) % ne2;
            const int64_t i1 = row % ne1;

            float * dst_row = dst_data + row * ne0;

            if (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                const float * src_row = (const float *) ((const char *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } else {
                int64_t       i       = 0;
                const int64_t vec_end = (ne0 / 8) * 8;
                for (; i < vec_end; i += 8) {
                    __asm__ volatile("fsw.ps f12, %[d]\n" : [d] "=m"(*(float (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        return 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    const int64_t elems_per_cl = 16;
    const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    const int64_t ne1_data_end = ne1 - rp1;
    const int64_t ne2_data_end = ne2 - rp2;
    const int64_t ne3_data_end = ne3 - rp3;

    for (int64_t cl = thread_id; cl < total_cl; cl += num_threads) {
        const int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        if (elem_end > total_elements) {
            elem_end = total_elements;
        }

        for (int64_t idx = elem_start; idx < elem_end; idx++) {
            const int64_t i0   = idx % ne0;
            const int64_t rem  = idx / ne0;
            const int64_t i1   = rem % ne1;
            const int64_t rem2 = rem / ne1;
            const int64_t i2   = rem2 % ne2;
            const int64_t i3   = rem2 / ne2;

            if (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                const float * sp = (const float *) ((const char *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } else {
                dst_data[idx] = 0.0f;
            }
        }
    }

    return 0;
}
