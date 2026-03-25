//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// Norm kernel parameters structure
struct ggml_et_norm_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor dst;   // F32 output tensor
    float eps;                // Epsilon parameter for numerical stability
};

int entry_point(struct ggml_et_norm_params* params, void* env) {
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

    // Verify that src0 and dst have same shape (required for norm)
    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1; // Shape mismatch
    }

    // Norm processes rows independently
    // Parallelize across rows using simple striding
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {

            // Calculate base pointers for this row using stride-based addressing
            const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
            float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

            // Step 1: Compute sum of elements for mean using 8-wide vectors
            // ne0 is guaranteed to be a multiple of 16 (cache-aligned)

            // Zero the accumulator register
            float zero = 0.0f;
            __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f11, %[x_vec]\n"            // Load 8 input values
                    "fadd.ps f10, f10, f11\n"            // acc += x (8-wide sum)
                    :
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                    : "f10", "f11"
                );
            }

            // Horizontal sum of 8 accumulated values in f10 to get total sum
            float sum;
            __asm__ __volatile__(
                "fswizz.ps f1, f10, 0xB1 \n\t"         // Swaps: e0<->e1 and e2<->e3
                "fadd.ps   f2, f10, f1, rne \n\t"
                "fswizz.ps f3, f2, 0x4E \n\t"           // Swaps: e0,e1 <-> e2,e3
                "fadd.ps   f4, f2, f3, rne \n\t"
                "fmvz.x.ps t0, f4, 4 \n\t"              // Move upper 128b half to scalar
                "fbcx.ps   f5, t0 \n\t"                  // Broadcast to vector
                "fadd.ps   %[vout], f4, f5, rne \n\t"
                : [vout] "=f" (sum)
                :: "t0", "f1", "f2", "f3", "f4", "f5"
            );

            // Step 2: Compute mean
            const float mean = et_fdiv(sum, (float)(int32_t)ne0);

            // Step 3: Compute y[i] = x[i] - mean and accumulate variance (sum of (x-mean)^2)
            // Zero the variance accumulator
            __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f11, %[x_vec]\n"            // Load 8 input values
                    "fbc.ps f12, %[mean_ptr]\n"         // Broadcast mean to all 8 elements
                    "fsub.ps f13, f11, f12\n"           // diff = x - mean (8-wide)
                    "fsw.ps f13, %[result]\n"           // Store centered values to dst
                    "fmadd.ps f10, f13, f13, f10\n"     // acc += diff * diff (fused multiply-add)
                    : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0]),
                      [mean_ptr] "m"(mean)
                    : "f10", "f11", "f12", "f13"
                );
            }

            // Horizontal sum of 8 accumulated variance values in f10
            float var_sum;
            __asm__ __volatile__(
                "fswizz.ps f1, f10, 0xB1 \n\t"
                "fadd.ps   f2, f10, f1, rne \n\t"
                "fswizz.ps f3, f2, 0x4E \n\t"
                "fadd.ps   f4, f2, f3, rne \n\t"
                "fmvz.x.ps t0, f4, 4 \n\t"
                "fbcx.ps   f5, t0 \n\t"
                "fadd.ps   %[vout], f4, f5, rne \n\t"
                : [vout] "=f" (var_sum)
                :: "t0", "f1", "f2", "f3", "f4", "f5"
            );

            // Step 4: Compute variance and scale factor
            const float variance = et_fdiv(var_sum, (float)(int32_t)ne0);
            const float scale = et_powf(variance + eps, -0.5f);

            // Numerical stability check
            if (!(scale > 0.0f)) {
                return -1; // Invalid scale factor
            }

            // Step 5: Apply scaling to centered values (already stored in dst)
            for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[y_vec]\n"            // Load 8 centered values from dst
                    "fbc.ps f13, %[scale_ptr]\n"        // Broadcast scale to all 8 elements
                    "fmul.ps f14, f12, f13\n"           // y * scale (8-wide)
                    "fsw.ps f14, %[result]\n"           // Store 8 scaled results

                    : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                    : [y_vec] "m"(*(const float(*)[8])&dst_ptr[i0]),
                      [scale_ptr] "m"(scale)
                    : "f12", "f13", "f14"
                );
            }
            }
        }
    }

    return 0;
}
