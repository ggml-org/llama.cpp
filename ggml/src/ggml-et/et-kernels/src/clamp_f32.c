//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_clamp_params {
    struct ggml_tensor src0;  // F32 input  (contiguous)
    struct ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    float              min_val;
    float              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
static inline void clamp_block_f32(float * dst, const float * src, float min_val, float max_val, int32_t n) {
    int32_t       i       = 0;
    const int32_t vec_end = (n / 8) * 8;

    if (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
        __asm__ volatile("mov.m.x m0, x0, 0xFF");

        for (; i < vec_end; i += 8) {
            __asm__ volatile(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(float (*)[8]) & dst[i])
                : [s] "m"(*(const float (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));
    }

    for (; i < n; i++) {
        float v = src[i];
        if (v < min_val) {
            v = min_val;
        }
        if (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

int entry_point(struct ggml_et_clamp_params * params, void * env) {
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

    const int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (total_elements <= 0) {
        return 0;
    }

    const float min_val = params->min_val;
    const float max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    const int64_t elems_per_cl = 16;
    const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    const int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    if (cl_end > total_cl) {
        cl_end = total_cl;
    }
    if (cl_start >= total_cl) {
        return 0;
    }

    const int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    if (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    return 0;
}
