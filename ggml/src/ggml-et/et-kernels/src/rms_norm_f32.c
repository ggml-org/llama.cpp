//******************************************************************************
// RMS Norm F32 Kernel
// Root Mean Square normalization: y[i] = x[i] / sqrt(mean(x^2) + eps)
//******************************************************************************

#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// RMS norm kernel parameters structure
struct ggml_et_rms_norm_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor dst;   // F32 output tensor
    float eps;                // Epsilon parameter for numerical stability
};

int entry_point(struct ggml_et_rms_norm_params* params, void* env) {
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

    // Verify that src0 and dst have same shape (required for RMS norm)
    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1; // Shape mismatch
    }

    // RMS norm processes rows independently
    // Parallelize across rows using simple striding
    // TODO: ensure lines don't cross cache lines
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {

            // Calculate base pointers for this row using stride-based addressing
            const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
            float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

            // Step 1: Compute sum of squares for this row using 8-wide vectors
            float sum = 0.0f;

            // Process 8 elements at a time using vector instructions
            int32_t vec_end = (int32_t)((ne0 / 8) * 8);
            if (vec_end > 0) {
                float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                for (int32_t i0 = 0; i0 < vec_end; i0 += 8) {
                    // Use vector operations to compute x*x and accumulate
                    __asm__ volatile(
                        "flw.ps f10, %[acc]\n"              // Load current accumulator (8 floats)
                        "flw.ps f11, %[x_vec]\n"            // Load 8 input values
                        "fmadd.ps f10, f11, f11, f10\n"     // acc += x * x (fused multiply-add)
                        "fsw.ps f10, %[result]\n"           // Store back to accumulator

                        : [result] "=m"(*(float(*)[8])acc_vec)
                        : [acc] "m"(*(const float(*)[8])acc_vec),
                          [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                        : "f10", "f11"
                    );
                }

                // Sum the 8 accumulated values
                for (int i = 0; i < 8; i++) {
                    sum += acc_vec[i];
                }
            }

            // Handle remaining elements (< 8) with scalar operations
            for (int32_t i0 = vec_end; i0 < (int32_t)ne0; i0++) {
                const float x = src_ptr[i0];
                sum += x * x;
            }

            // Step 2: Compute mean of squares and scale factor
            const float mean = et_fdiv(sum, (float)(int32_t)ne0);
            const float scale = et_powf(mean + eps, -0.5f);

            // Numerical stability check
            if (!(scale > 0.0f)) {
                return -1; // Invalid scale factor
            }

            // Step 3: Apply scaling using 8-wide vectors
            // This approach works for both in-place and regular operations

            // Process 8 elements at a time using vector multiplication
            for (int32_t i0 = 0; i0 < vec_end; i0 += 8) {
                // Use vector operations to scale 8 elements at once
                __asm__ volatile(
                    "flw.ps f12, %[x_vec]\n"            // Load 8 input values
                    "fbc.ps f13, %[scale_ptr]\n"        // Broadcast scale to all 8 elements
                    "fmul.ps f14, f12, f13\n"           // x * scale (8-wide)
                    "fsw.ps f14, %[result]\n"           // Store 8 scaled results

                    : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                    : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0]),
                      [scale_ptr] "m"(scale)
                    : "f12", "f13", "f14"
                );
            }

            // Handle remaining elements (< 8) with scalar operations
            for (int32_t i0 = vec_end; i0 < (int32_t)ne0; i0++) {
                dst_ptr[i0] = src_ptr[i0] * scale;
            }
            }
        }
    }

    return 0;
}
