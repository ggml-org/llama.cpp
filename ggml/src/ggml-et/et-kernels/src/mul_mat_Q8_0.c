//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

#define STRIDE_M        2048  /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT 1024  /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS 256   /* K >= 8192 elements */
#define KSPLIT_MAX_ROWS     8     /* max rows per minion for K-split */
#define TILE_KB           256     /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    const int64_t K    = params->src0.ne[0];
    const int64_t M    = params->src0.ne[1];
    const int64_t N    = params->src1.ne[1];
    const int64_t ne02 = params->src0.ne[2];
    const int64_t ne03 = params->src0.ne[3];
    const int64_t ne12 = params->src1.ne[2];
    const int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    const size_t nb01 = params->src0.nb[1];
    const size_t nb02 = params->src0.nb[2];
    const size_t nb03 = params->src0.nb[3];

    const size_t nb11 = params->src1.nb[1];
    const size_t nb12 = params->src1.nb[2];
    const size_t nb13 = params->src1.nb[3];

    const size_t nbd1 = params->dst.nb[1];
    const size_t nbd2 = params->dst.nb[2];
    const size_t nbd3 = params->dst.nb[3];

    // Q8_0 block size is 32
    const int64_t K_blocks = K / 32;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // K-split decision
    const int64_t minion_id = hart_id >> 1;               /* 0..1023 global */
    const int64_t local_minion = (hart_id >> 1) & 0x1F;   /* 0..31 within shire */
    const int is_hart1 = hart_id & 1;
    const int64_t rows_per_minion = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    const int64_t k_half = K_blocks / 2;
    /*
     * K-split when K is large enough to benefit, and either:
     *   - few rows (≤4): always safe, proven working
     *   - more rows (5-8): only if each hart's half fits in one tile,
     *     otherwise L1 thrashing from 2 harts × 8 rows kills performance
     */
    const int use_ksplit = (K_blocks >= KSPLIT_MIN_K_BLOCKS)
                        && (rows_per_minion <= KSPLIT_MAX_ROWS)
                        && (rows_per_minion <= 4 || k_half <= TILE_KB);

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    if (use_ksplit) {
        /* Each hart processes half the K dimension */
        const int64_t k_start = is_hart1 ? k_half : 0;
        const int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        volatile float* l2scp_slot =
            (volatile float*)et_shire_l2scp_local(local_minion * 64);

        for (int64_t i3 = 0; i3 < ne13; i3++) {
            const int64_t i03 = i3 / r3;
            const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
            const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
            char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

            for (int64_t i2 = 0; i2 < ne12; i2++) {
                const int64_t i02 = i2 / r2;
                const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
                const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
                char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

                for (int64_t n = 0; n < N; n++) {
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                    for (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);

                        q8_dot_reset();
                        q8_dot_tile(q_row + k_start, b_col_base + k_start * 32, k_len);
                        float partial = q8_dot_reduce();

                        if (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((const void*)l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } else {
                            et_sem_wait(ET_BARRIER_MINION);
                            float other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                            atomic_store_f32((volatile float*)dst_entry, partial + other);
                        }
                    }
                }
            }
        }
    } else if (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        for (int64_t i3 = 0; i3 < ne13; i3++) {
            const int64_t i03 = i3 / r3;
            const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
            const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
            char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

            for (int64_t i2 = 0; i2 < ne12; i2++) {
                const int64_t i02 = i2 / r2;
                const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
                const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
                char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

                for (int64_t n = 0; n < N; n++) {
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                    for (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        const int64_t m1 = m0 + STRIDE_M;
                        const int64_t m2 = m0 + STRIDE_M * 2;
                        const int64_t m3 = m0 + STRIDE_M * 3;

                        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        for (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            if (tile_len > TILE_KB) tile_len = TILE_KB;
                            const float* b_tile = b_col_base + kb * 32;

                            q8_dot_reset();
                            q8_dot_tile((const block_q8_0*)(src0_ptr2 + m0 * nb01) + kb,
                                        b_tile, tile_len);
                            s0 += q8_dot_reduce();
                            if (m1 < M) {
                                q8_dot_reset();
                                q8_dot_tile((const block_q8_0*)(src0_ptr2 + m1 * nb01) + kb,
                                            b_tile, tile_len);
                                s1 += q8_dot_reduce();
                            }
                            if (m2 < M) {
                                q8_dot_reset();
                                q8_dot_tile((const block_q8_0*)(src0_ptr2 + m2 * nb01) + kb,
                                            b_tile, tile_len);
                                s2 += q8_dot_reduce();
                            }
                            if (m3 < M) {
                                q8_dot_reset();
                                q8_dot_tile((const block_q8_0*)(src0_ptr2 + m3 * nb01) + kb,
                                            b_tile, tile_len);
                                s3 += q8_dot_reduce();
                            }
                        }

                        float* dst_base = (float*)(dst_ptr2 + n * nbd1);
                        atomic_store_f32((volatile float*)(dst_base + m0), s0);
                        if (m1 < M) atomic_store_f32((volatile float*)(dst_base + m1), s1);
                        if (m2 < M) atomic_store_f32((volatile float*)(dst_base + m2), s2);
                        if (m3 < M) atomic_store_f32((volatile float*)(dst_base + m3), s3);
                    }
                }
            }
        }
    } else {
        /* Simple path for small K (single tile, no B reuse benefit) */
        for (int64_t i3 = 0; i3 < ne13; i3++) {
            const int64_t i03 = i3 / r3;
            const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
            const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
            char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

            for (int64_t i2 = 0; i2 < ne12; i2++) {
                const int64_t i02 = i2 / r2;
                const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
                const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
                char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

                for (int64_t n = 0; n < N; n++) {
                    const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                    for (int64_t m = hart_id; m < M; m += STRIDE_M) {
                        const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);

                        q8_dot_reset();
                        q8_dot_tile(q_row, b_col_base, K_blocks);
                        float sum = q8_dot_reduce();

                        float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                        atomic_store_f32((volatile float*)dst_entry, sum);
                    }
                }
            }
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
    return 0;
}
