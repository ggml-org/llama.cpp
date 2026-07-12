//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdbool.h>
#include <stdint.h>

struct ggml_et_cont_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

int entry_point(struct ggml_et_cont_params * params, void * env) {
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
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        return -1;
    }

    const char * src_data = (const char *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    if (!src_data || !dst_data) {
        return -1;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    if (total_elements == 0) {
        return 0;
    }

    // Check if src is contiguous F32
    const bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    const int64_t elems_per_cl = 32;
    const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    const int64_t cl_start      = thread_id * cl_per_thread;
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

    if (src_contiguous) {
        // Fast path: src is contiguous F32
        const float * src_f32 = (const float *) src_data;
        for (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } else {
        // General path: stride-aware read
        for (int64_t idx = es; idx < ee; ++idx) {
            const int64_t i00  = idx % ne00;
            const int64_t rem1 = idx / ne00;
            const int64_t i01  = rem1 % ne01;
            const int64_t rem2 = rem1 / ne01;
            const int64_t i02  = rem2 % ne02;
            const int64_t i03  = rem2 / ne02;

            const float val = *(const float *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    return 0;
}
