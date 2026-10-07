
// Fused RMS Norm + MUL F32 Kernel

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// Fused RMS norm + MUL kernel parameters structure
struct ggml_et_rms_norm_mul_params {
    struct ggml_tensor src0;  // F32 input tensor (to be normalized)
    struct ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    struct ggml_tensor dst;   // F32 output tensor
    float              eps;   // Epsilon for numerical stability
};

static inline size_t tensor_bytes(const struct ggml_tensor * t) {
    return (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

int entry_point(struct ggml_et_rms_norm_mul_params * params, void * env) {
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
        return -1;  // Invalid pointer
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * dst  = &params->dst;

    float eps = params->eps;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;  // Unsupported type combination
    }

    float * src0_data = (float *) src0->data;
    float * src1_data = (float *) src1->data;
    float * dst_data  = (float *) dst->data;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    if (!src0_data || !src1_data || !dst_data) {
        return -1;  // Null data pointer
    }

    if (eps < 0.0f) {
        return -1;  // Invalid epsilon
    }

    const int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    const int64_t ne1 = dst->ne[1];  // Dimension 1
    const int64_t ne2 = dst->ne[2];  // Dimension 2
    const int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        return -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    const float   inv_ne0       = et_fdiv(1.0f, (float) (int32_t) ne0);
    const int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    const int     shire_threads = SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;

    if (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {
                    const float * src_ptr =
                        (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    float * dst_ptr = (float *) ((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    const float * wgt_ptr = (const float *) ((const char *) src1_data + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    unsigned long saved_mask;
                    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ volatile("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ volatile("fbci.pi f10, 0" ::: "f10");
                    for (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    float sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    const float scale = et_powf(sum * inv_ne0 + eps, -0.5f);
                    if (!(scale > 0.0f)) {
                        __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
                        return -1;
                    }

                    uint32_t scale_bits;
                    __asm__ volatile("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                    __asm__ volatile("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                    for (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(const float (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
                }
            }
        }
    } else {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        int shire_tid       = thread_id % shire_threads;
        int threads_per_row = shire_threads / total_rows;
        int my_row          = shire_tid / threads_per_row;
        int local_tid       = shire_tid % threads_per_row;
        int group_base      = my_row * threads_per_row;

        // Excess threads within this shire
        if (my_row >= total_rows) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            return 0;
        }

        // Unflatten row index
        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        const float * src_ptr = (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        float *       dst_ptr = (float *) ((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        const float * wgt_ptr = (const float *) ((const char *) src1_data + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        const int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        if (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        if (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
        __asm__ volatile("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ volatile("fbci.pi f10, 0" ::: "f10");
        for (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        float partial_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        volatile float * my_slot = (volatile float *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_to_l2((const void *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        int workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        for (int t = 0; t < workers; t++) {
            volatile float * slot = (volatile float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((const void *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        float total_sum = 0.0f;
        for (int t = 0; t < workers; t++) {
            volatile float * slot = (volatile float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        const float scale = et_powf(total_sum * inv_ne0 + eps, -0.5f);
        if (!(scale > 0.0f)) {
            __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
            return -1;
        }

        // Apply scale * weights to own chunk
        if (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ volatile("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ volatile("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            for (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(const float (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    }

    return 0;
}
