//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

// SQR kernel parameters structure (unary op: src0 -> dst)
struct ggml_et_sqr_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor dst;   // F32 output tensor
};

int entry_point(struct ggml_et_sqr_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* dst = &params->dst;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    float* src0_data = (float*)src0->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !dst_data) {
        return -1; // Null data pointer
    }

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Verify that src0 and dst have same shape
    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1; // Shape mismatch
    }

    // Parallelize across rows using simple striding
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {

            const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
            float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

            // Process 8 elements at a time: dst[i] = src[i] * src[i]
            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f10, %[x_vec]\n"            // Load 8 input values
                    "fmul.ps f11, f10, f10\n"           // x * x (8-wide)
                    "fsw.ps f11, %[result]\n"           // Store 8 results

                    : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                    : "f10", "f11"
                );
            }
            }
        }
    }

    return 0;
}
