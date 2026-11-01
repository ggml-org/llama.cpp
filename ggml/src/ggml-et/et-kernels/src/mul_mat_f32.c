//******************************************************************************
// MUL_MAT Kernel (F32 weights)
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//
// Decode-optimized distribution: one output element (row m of A dotted with
// column n of B) is the unit of work, striped by thread_id across every hart
// of every active shire. This keeps all 32 shires busy for a GEMV
// (M=4096, N=1) instead of leaving most idle. The reused B vector is staged
// into per-shire L2 SCP and the accumulator stays register-resident across K
// via f32_dot_*.
//******************************************************************************

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"
#include "tensor.h"

#include <stdint.h>

int entry_point(struct ggml_et_binary_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    // Thread coordination, use every hart of every active shire.
    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    struct ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    struct ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    struct ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    if (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float * src0_data = (const float *) src0->data;
    float *       dst_data  = (float *) dst->data;

    // Dimensions and strides
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    const int     block_size  = 32;   // 32 f32 per row-dot tile (128B)
    const int64_t K_blocks    = K / block_size;
    const int64_t K_remainder = K % block_size;

    // Broadcasting support
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const int      is_f32_b       = (src1->type == GGML_TYPE_F32);
    const uint64_t total_elements = (uint64_t) M * N * ne2 * ne3;

    // Stage the reused B activation vector into per-shire L2 SCP. Streaming the
    // weight matrix thrashes L2 and evicts B, so B is re-read from DRAM
    // repeatedly during the decode GEMV. Stage it once per shire into L2 SCP (a
    // separate SRAM partition cache streaming cannot evict) via
    // et_tensor_load_l2scp; one hart per shire issues the DMA loop, then a
    // shire barrier lets all harts read B on-chip. Only applies to the common
    // decode case: F32 B, a single contiguous non-broadcast B that fits the SCP
    // budget.
    const int      b_contig = (nb11 == (size_t) K * sizeof(float));
    const uint64_t b_lines  = ((uint64_t) N * K * sizeof(float) + 63) / 64;
    const int      stage_b  = is_f32_b && b_contig &&
                              ne12 == 1 && ne13 == 1 && ne02 == 1 && ne03 == 1 &&
                              b_lines <= 8192;   // <= 512 KB, within the SCP budget
    const float *  b_scp    = (const float *) et_shire_l2scp_local(0);

    if (stage_b) {
        if ((get_hart_id() & 63) == 0) {
            et_tensor_load_l2scp_conf_t conf;
            conf.use_tmask = false;
            conf.stride    = 64;   // advance one 64B cache line per line loaded
            uint64_t remaining = b_lines;
            uint64_t dst_ln    = 0;
            uint64_t addr      = (uint64_t) src1->data;
            while (remaining > 0) {
                uint64_t cl = (remaining >= 16) ? 16 : remaining;
                conf.dst_start = dst_ln;
                conf.addr      = addr;
                conf.num_lines = cl - 1;   // 4-bit field encodes (lines - 1)
                conf.id        = 0;
                et_tensor_load_l2scp(&conf);
                WAIT_TENSOR_LOAD_L2_0;
                dst_ln    += cl;
                addr      += cl * 64;
                remaining -= cl;
            }
        }
        et_barrier(ET_BARRIER_SHIRE);   // B is now resident in L2 SCP for all harts
    }

    // Set the vector mask (all 8 lanes) once for the whole row loop.
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (uint64_t idx = (uint64_t) thread_id; idx < total_elements; idx += (uint64_t) num_threads) {
        // Index decoding
        const int64_t i3   = idx / (M * N * ne2);
        const int64_t rem3 = idx % (M * N * ne2);
        const int64_t i2   = rem3 / (M * N);
        const int64_t rem2 = rem3 % (M * N);
        const int64_t n    = rem2 / M;
        const int64_t m    = rem2 % M;

        const int64_t i03 = i3 / r3, i02 = i2 / r2;
        const int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

        const float * f32_row =
            (const float *) ((const char *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

        float sum;
        if (is_f32_b) {
            const float * b_col = stage_b
                ? (b_scp + n * K)
                : (const float *) ((const char *) src1->data + n * nb11 + i12 * nb12 + i13 * nb13);

            f32_dot_reset();
            f32_dot_tile(f32_row, b_col, K_blocks);
            sum = f32_dot_reduce();

            if (K_remainder > 0) {
                const int64_t offset = K_blocks * block_size;
                sum += compute_block_dot_product_f32_partial(&f32_row[offset], &b_col[offset], K_remainder);
            }
        } else {
            const uint16_t * b_col =
                (const uint16_t *) ((const char *) src1->data + n * nb11 + i12 * nb12 + i13 * nb13);
            sum = compute_block_dot_product_f32_f16_partial(f32_row, b_col, K);
        }

        volatile float * c_element =
            (volatile float *) ((char *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
        atomic_store_f32(c_element, sum);
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return 0;
}
