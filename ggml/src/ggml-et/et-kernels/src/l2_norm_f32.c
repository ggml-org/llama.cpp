//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// L2 Norm kernel parameters structure
struct ggml_et_l2_norm_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor dst;   // F32 output tensor
    float eps;                // Epsilon parameter for numerical stability
};

int entry_point(struct ggml_et_l2_norm_params* params, void* env) {
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
    float eps = params->eps;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    float* src0_data = (float*)src0->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !dst_data) {
        return -1; // Null data pointer
    }

    if (eps < 0.0f) {
        return -1; // Invalid epsilon
    }

    const int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    const int64_t ne1 = dst->ne[1];  // Dimension 1
    const int64_t ne2 = dst->ne[2];  // Dimension 2
    const int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Suppress unused variable warnings
    (void)nb0; (void)nb00;

    // Verify that src0 and dst have same shape
    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1; // Shape mismatch
    }

    // L2 norm processes rows independently
    // Parallelize across rows using simple striding
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {

            // Calculate base pointers for this row using stride-based addressing
            const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
            float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

            // Step 1: Compute sum of squares using 8-wide vectors
            // ne0 is guaranteed to be a multiple of 16 (cache-aligned)

            // Zero the accumulator register
            float zero = 0.0f;
            __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f11, %[x_vec]\n"            // Load 8 input values
                    "fmadd.ps f10, f11, f11, f10\n"     // acc += x * x (fused multiply-add)
                    :
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                    : "f10", "f11"
                );
            }

            // Horizontal sum of 8 accumulated values in f10 to get total sum of squares
            float sum_sq;
            __asm__ __volatile__(
                "fswizz.ps f1, f10, 0xB1 \n\t"         // Swaps: e0<->e1 and e2<->e3
                "fadd.ps   f2, f10, f1, rne \n\t"
                "fswizz.ps f3, f2, 0x4E \n\t"           // Swaps: e0,e1 <-> e2,e3
                "fadd.ps   f4, f2, f3, rne \n\t"
                "fmvz.x.ps t0, f4, 4 \n\t"              // Move upper 128b half to scalar
                "fbcx.ps   f5, t0 \n\t"                  // Broadcast to vector
                "fadd.ps   %[vout], f4, f5, rne \n\t"
                : [vout] "=f" (sum_sq)
                :: "t0", "f1", "f2", "f3", "f4", "f5"
            );

            // Step 2: Compute scale = 1 / max(sqrt(sum_sq), eps)
            float l2_norm = et_powf(sum_sq, 0.5f);
            if (l2_norm < eps) {
                l2_norm = eps;
            }
            const float scale = et_fdiv(1.0f, l2_norm);

            // Step 3: Copy src to dst and apply scaling: y[i] = x[i] * scale
            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f11, %[x_vec]\n"            // Load 8 input values from src
                    "fbc.ps f12, %[scale_ptr]\n"        // Broadcast scale to all 8 elements
                    "fmul.ps f13, f11, f12\n"           // x * scale (8-wide)
                    "fsw.ps f13, %[result]\n"           // Store 8 scaled results

                    : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0]),
                      [scale_ptr] "m"(scale)
                    : "f11", "f12", "f13"
                );
            }
            }
        }
    }

    return 0;
}
