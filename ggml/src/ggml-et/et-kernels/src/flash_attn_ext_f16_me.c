//******************************************************************************
// Flash Attention with TensorFMA16A32 for QK^T
//
// Uses the matrix engine for the QK^T dot products (F16×F16→F32),
// scalar code for online softmax and V accumulation.
//
// Requirements:
//   - Q: F32 (converted to F16 internally)
//   - K, V: F16
//   - dk must be a multiple of 32 (TensorFMA16A32 K-tile)
//   - dv ≤ 128
//   - Only hart 0 per minion (matrix engine restriction)
//
// Parallelization: each minion independently processes one (qpos, head, batch)
// row, round-robin across all minion hart-0s.
//******************************************************************************

#include <etsoc/common/utils.h>
#include <stdint.h>
#include <string.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "math_fp.h"

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#define TILE_KV 16
#define TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#define A_L1_START 0
#define B_L1_START 16

// Max head dimensions for stack buffers
#define FA_DV_MAX 128   // max value head dim (dv)
#define FA_DK_MAX 256   // max key head dim (dk) — some models use hsk > hsv

typedef uint16_t et_fp16_t;

#define ET_NEG_INF_F (-3.402823466e+38f)

struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;     // Q (F32)
    struct ggml_tensor src1;     // K (F16)
    struct ggml_tensor src2;     // V (F16)
    struct ggml_tensor mask;     // mask (F16 or F32), zeroed when absent
    struct ggml_tensor dst;      // Output (F32)
    float scale;
    int32_t has_mask;
};

static inline float get_mask_val(const struct ggml_tensor * mask,
                                 int64_t iq1, int64_t ik1,
                                 int64_t iq2, int64_t iq3) {
    const char * base = (const char *) mask->data
        + iq1 * mask->nb[1]
        + (iq2 % mask->ne[2]) * mask->nb[2]
        + (iq3 % mask->ne[3]) * mask->nb[3];

    if (mask->type == GGML_TYPE_F32) {
        return *(const float *)(base + ik1 * mask->nb[0]);
    }
    return fp16_to_fp32(*(const uint16_t *)(base + ik1 * mask->nb[0]));
}

static inline const char * get_mask_row_base(const struct ggml_tensor * mask,
                                             int64_t iq1, int64_t iq2, int64_t iq3) {
    return (const char *) mask->data
        + iq1 * mask->nb[1]
        + (iq2 % mask->ne[2]) * mask->nb[2]
        + (iq3 % mask->ne[3]) * mask->nb[3];
}

static inline float get_mask_val_from_base(const struct ggml_tensor * mask,
                                           const char * base, int64_t ik1) {
    if (mask->type == GGML_TYPE_F32) {
        return *(const float *)(base + ik1 * mask->nb[0]);
    }
    return fp16_to_fp32(*(const uint16_t *)(base + ik1 * mask->nb[0]));
}

// Build B panel input for TensorLoadTranspose16.
//
// TensorLoadTranspose16 does: L1Scp[c].h[i] = input[i].h[c]
// We want the interleaved B: L1Scp[l].h[n*2+r] = K[n][dk_start + 2*l + r]
// So we need: input[n*2+r].h[l] = K[n][dk_start + 2*l + r]
//
// For each KV position n, produce two rows (de-interleave even/odd dk elements):
//   Row 2n:   K[n][dk+0], K[n][dk+2], ..., K[n][dk+30]  (16 evens)
//   Row 2n+1: K[n][dk+1], K[n][dk+3], ..., K[n][dk+31]  (16 odds)
//
// Output buffer: 32 rows × 32 halfwords (64-byte stride, 16 hw data + 16 hw pad)
//

// Prefetch K rows for one dk_chunk into L2.
// Each KV position needs 1 cache line (64B = 32 fp16 values).
// Uses prefetch_va (CSR 0x81F) with stride in x31 to batch all kv_count lines.
static inline void __attribute__((always_inline))
prefetch_k_to_l2(const char * k_head, int64_t kv_start, int64_t dk_start,
                 int64_t kv_count, int64_t nb1_k)
{
    uintptr_t base = (uintptr_t)k_head + kv_start * nb1_k + dk_start * 2;
    uint64_t num_lines_m1 = (uint64_t)(kv_count - 1);  // bits 3:0

    __asm__ __volatile__ (
        "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
        "mv    x31, %[stride]\n"          // Stride = nb1_k bytes
        "or    x3, x1, %[ptr]\n"          // Combine Dest + VA
        "or    x3, x3, %[sz]\n"           // Combine with NumLines
        "csrw  0x81f, x3\n"              // prefetch_va
        :
        : [ptr] "r" (base),
          [sz] "r" (num_lines_m1),
          [stride] "r" ((uint64_t)nb1_k)
        : "x1", "x3", "x31", "memory"
    );
}

static inline void __attribute__((always_inline))
pack_k_for_transpose16(et_fp16_t * out,
                       const char * k_base,
                       int64_t kv_start,
                       int64_t dk_start,
                       int64_t kv_count,
                       int64_t nb1_k)
{
    // save registers we use becayse TensorFMA uses all FP registers
    // and this function is used in the middle TensorFMAs
    uint32_t save_f28[8] __attribute__((aligned(32)));
    uint32_t save_f29[8] __attribute__((aligned(32)));
    uint32_t save_f30[8] __attribute__((aligned(32)));
    uint32_t save_f31[8] __attribute__((aligned(32)));
    unsigned long old_mask;

    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "fsw.ps    f28, 0(%[save28])\n\t"
        "fsw.ps    f29, 0(%[save29])\n\t"
        "fsw.ps    f30, 0(%[save30])\n\t"
        "fsw.ps    f31, 0(%[save31])\n\t"
        : [ms] "=&r"(old_mask)
        : [save28] "r"(save_f28),
          [save29] "r"(save_f29),
          [save30] "r"(save_f30),
          [save31] "r"(save_f31)
        : "f28", "f29", "f30", "f31", "memory"
    );

    for (int j = 0; j < (int)kv_count; ++j) {
        const et_fp16_t * k_row =
            (const et_fp16_t *)(k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t * even_row = out + (j * 2)     * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        {
            __asm__ volatile(
                "flw.ps    f30, 0(%[src0])  \n\t"
                "flw.ps    f31, 0(%[src1])  \n\t"
                "fpackreph.pi f28, f30      \n\t"
                "fsrli.pi  f29, f30, 16     \n\t"
                "fpackreph.pi f29, f29      \n\t"
                "fpackreph.pi f30, f31      \n\t"
                "fsrli.pi  f31, f31, 16     \n\t"
                "fpackreph.pi f31, f31      \n\t"
                "mov.m.x   m0, x0, 0x0F     \n\t"
                "fcmovm.ps f28, f28, f30    \n\t"
                "fcmovm.ps f29, f29, f31    \n\t"
                "mov.m.x   m0, x0, 0xFF     \n\t"
                "fsw.ps    f28, 0(%[even])  \n\t"
                "fsw.ps    f29, 0(%[odd])   \n\t"
                :
                : [src0] "r"(k_row),
                  [src1] "r"(k_row + 16),
                  [even] "r"(even_row),
                  [odd] "r"(odd_row)
                : "f28", "f29", "f30", "f31", "memory"
            );
        }
    }

    __asm__ volatile(
        "flw.ps    f28, 0(%[save28])\n\t"
        "flw.ps    f29, 0(%[save29])\n\t"
        "flw.ps    f30, 0(%[save30])\n\t"
        "flw.ps    f31, 0(%[save31])\n\t"
        "mova.m.x  %[ms]            \n\t"
        :
        : [ms] "r"(old_mask),
          [save28] "r"(save_f28),
          [save29] "r"(save_f29),
          [save30] "r"(save_f30),
          [save31] "r"(save_f31)
        : "f28", "f29", "f30", "f31", "memory"
    );

    for (int j = (int)kv_count; j < TILE_KV; ++j) {
        et_fp16_t * even_row = out + (j * 2)     * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        for (int l = 0; l < TILE_K / 2; ++l) {
            even_row[l] = 0;
            odd_row[l]  = 0;
        }
    }
}


static inline void __attribute__((always_inline))
convert_q_row_f32_to_f16(et_fp16_t * dst, const float * src, int64_t n) {
    static const int32_t __attribute__((aligned(32))) offsets[8] = {
        0, 2, 4, 6, 8, 10, 12, 14
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_mask)
        : [offs] "r"(offsets)
        : "f1"
    );

    for (int64_t d = 0; d < n; d += 8) {
        __asm__ volatile(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[ms]             \n\t"
        :
        : [ms] "r"(old_mask)
    );
}

static inline void __attribute__((always_inline))
accumulate_v_row_f16_contig(float * acc, const char * pv, int64_t dv, float vs) {
    static const int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 2, 4, 6, 8, 10, 12, 14
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[gidx])    \n\t"
        : [ms] "=&r"(old_mask)
        : [gidx] "r"(gather_idx)
        : "f1"
    );

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps      f2, 0(%[p_vs])   \n\t"
            "fgh.ps      f3, f1(%[pv])    \n\t"
            "fcvt.ps.f16 f3, f3           \n\t"
            "flw.ps      f4, 0(%[pa])     \n\t"
            "fmadd.ps    f4, f3, f2, f4   \n\t"
            "fsw.ps      f4, 0(%[pa])     \n\t"
            :
            : [p_vs] "r"(&vs), [pv] "r"(pv + d * 2), [pa] "r"(acc + d)
            : "f2", "f3", "f4", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[ms]             \n\t"
        :
        : [ms] "r"(old_mask)
    );
}

static inline void __attribute__((always_inline))
rescale_accumulate_v_row_f16_contig(float * acc, const char * pv, int64_t dv, float ms) {
    static const int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 2, 4, 6, 8, 10, 12, 14
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[msk]            \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[gidx])    \n\t"
        : [msk] "=&r"(old_mask)
        : [gidx] "r"(gather_idx)
        : "f1"
    );

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps      f2, 0(%[p_ms])   \n\t"
            "fgh.ps      f3, f1(%[pv])    \n\t"
            "fcvt.ps.f16 f3, f3           \n\t"
            "flw.ps      f4, 0(%[pa])     \n\t"
            "fmadd.ps    f4, f4, f2, f3   \n\t"
            "fsw.ps      f4, 0(%[pa])     \n\t"
            :
            : [p_ms] "r"(&ms), [pv] "r"(pv + d * 2), [pa] "r"(acc + d)
            : "f2", "f3", "f4", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[msk]            \n\t"
        :
        : [msk] "r"(old_mask)
    );
}

static inline void __attribute__((always_inline))
zero_acc_vec(float * acc, int64_t dv) {
    const float zero = 0.0f;
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps  f2, 0(%[z])     \n\t"
            "fsw.ps  f2, 0(%[a])     \n\t"
            :
            : [z] "r"(&zero), [a] "r"(acc + d)
            : "f2", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

static inline void __attribute__((always_inline))
scale_acc_vec(float * acc, int64_t dv, float scale) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

static inline void __attribute__((always_inline))
normalize_store_vec(float * out, float * acc, int64_t dv, float inv, int use_fast_store) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory"
        );
        if (use_fast_store) {
            __asm__ volatile(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(out + d)
                : "f4", "memory"
            );
        } else {
            atomic_store_f32((volatile float *) &out[d + 0], acc[d + 0]);
            atomic_store_f32((volatile float *) &out[d + 1], acc[d + 1]);
            atomic_store_f32((volatile float *) &out[d + 2], acc[d + 2]);
            atomic_store_f32((volatile float *) &out[d + 3], acc[d + 3]);
            atomic_store_f32((volatile float *) &out[d + 4], acc[d + 4]);
            atomic_store_f32((volatile float *) &out[d + 5], acc[d + 5]);
            atomic_store_f32((volatile float *) &out[d + 6], acc[d + 6]);
            atomic_store_f32((volatile float *) &out[d + 7], acc[d + 7]);
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

static inline float __attribute__((always_inline))
exp2f_et(float x) {
    unsigned long old_mask;
    float out;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x)
    );
    return out;
}

int entry_point(struct ggml_et_flash_attn_ext_params * params, void * env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;
    if (hart_id & 1) return 0;  // only hart 0 may issue tensor ops

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_id = shire_id * MINIONS_PER_SHIRE + local_minion;
    uint64_t total_minions = NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE;

    struct ggml_tensor * q   = &params->src0;
    struct ggml_tensor * k   = &params->src1;
    struct ggml_tensor * v   = &params->src2;
    struct ggml_tensor * dst = &params->dst;
    const int32_t has_mask   = params->has_mask;
    struct ggml_tensor * mask = has_mask ? &params->mask : (struct ggml_tensor *) 0;

    const char * q_data   = (const char *) q->data;
    const char * k_data   = (const char *) k->data;
    const char * v_data   = (const char *) v->data;
    char * dst_data       = (char *) dst->data;

    const int64_t dk  = q->ne[0];
    const int64_t nq  = q->ne[1];
    const int64_t nhq = q->ne[2];
    const int64_t no  = q->ne[3];
    const int64_t nk  = k->ne[1];
    const int64_t nhk = k->ne[2];
    const int64_t dv  = v->ne[0];

    if (dv > FA_DV_MAX || dk > FA_DK_MAX) return -1;
    if (k->nb[0] != 2 || v->nb[0] != 2) return -1;
    if ((dk % 8) != 0 || (dv % 16) != 0) return -1;

    const int64_t gqa_ratio = nhq / nhk;
    const int64_t total_rows = nq * nhq * no;
    const float scale = params->scale;
    const int use_fast_store = (dv % 16 == 0);

    // Enable L1 scratchpad for tensor engine
    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    // Interleaved K panel buffers
    et_fp16_t kpanel[32 * 32] __attribute__((aligned(64)));

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    float scores[TILE_KV] __attribute__((aligned(64)));

    for (int64_t row = (int64_t)my_id; row < total_rows; row += (int64_t)total_minions) {
        const int64_t iq3 = row / (nhq * nq);
        const int64_t rem = row % (nhq * nq);
        const int64_t iq2 = rem / nq;
        const int64_t iq1 = rem % nq;
        const int64_t ik2 = iq2 / gqa_ratio;

        // Read Q row (F32) and convert to F16
        const float * pq = (const float *)(q_data + iq1*q->nb[1] + iq2*q->nb[2] + iq3*q->nb[3]);
        convert_q_row_f32_to_f16(q_f16, pq, dk);

        // K/V base for this head + batch
        const char * k_head = k_data + ik2*k->nb[2] + iq3*k->nb[3];
        const char * v_head = v_data + ik2*v->nb[2] + iq3*v->nb[3];

        // Output pointer
        float * out = (float *)(dst_data + iq2*dst->nb[1] + iq1*dst->nb[2] + iq3*dst->nb[3]);

        float acc[FA_DV_MAX] __attribute__((aligned(32)));
        zero_acc_vec(acc, dv);
        float M = ET_NEG_INF_F;
        float S = 0.0f;
        const char * mask_base = has_mask ? get_mask_row_base(mask, iq1, iq2, iq3) : (const char *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_to_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        for (int64_t kv_base = 0; kv_base < nk; kv_base += TILE_KV) {
            const int64_t kv_count = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

            // Set tensor_mask for partial tiles
            if (kv_count < TILE_KV) {
                uint64_t tmask = (1ULL << kv_count) - 1;
                __asm__ __volatile__("csrw 0x805, %0" : : "r"(tmask));
            }

            // ============================================================
            // QK^T via TensorFMA16A32
            // A = Q_f16[1, dk] in L1SCP (1 row, processed in TILE_K chunks)
            // B = K^T_interleaved[TILE_K, kv_count] in L1SCP
            // Result: scores[1, kv_count] in vector registers
            // ============================================================

            // Prefetch K data for the first dk_chunk into L2
            prefetch_k_to_l2(k_head, kv_base, 0, kv_count, k->nb[1]);

            for (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {

                pack_k_for_transpose16(kpanel, k_head, kv_base, dk_chunk,
                                       kv_count, k->nb[1]);

                // Prefetch K data for the NEXT dk_chunk while tensor engine
                // processes the current one (overlaps DRAM latency with compute)
                if (dk_chunk + TILE_K < dk) {
                    prefetch_k_to_l2(k_head, kv_base, dk_chunk + TILE_K,
                                     kv_count, k->nb[1]);
                }

                FENCE;
                // split as `flush_to_l2` can only do 16 lines at a time
                flush_to_l2(kpanel, 16, 64);
                flush_to_l2(kpanel + 16 * 32, 16, 64);

                // Load Q_f16 chunk into L1SCP as A (1 row × 64B)
                tensor_load(
                    false, false,
                    A_L1_START,
                    TENSOR_LOAD_PLAIN,
                    0,
                    (uint64_t)(q_f16 + dk_chunk),
                    0,
                    0,  // 1 row (num_lines = 0 means 1 line)
                    64, // stride (doesn't matter for 1 row)
                    0
                );
                tensor_wait(TENSOR_LOAD_WAIT_0);

                // Make sure the current panel flush completed before tensor_load.
                WAIT_CACHEOPS;

                // New pack + TRANSPOSE16
                tensor_load(
                    false, false,
                    B_L1_START,
                    TENSOR_LOAD_TRANSPOSE16,
                    0,
                    (uint64_t)kpanel,
                    0,
                    15,  // 16 output lines
                    64,  // stride between source rows
                    0
                );
                tensor_wait(TENSOR_LOAD_WAIT_0);


                tensor_fma(
                    (kv_count < TILE_KV),  // use_tmask for partial tiles
                    3,                      // b_num_col (16 output cols)
                    0,                      // a_num_rows (1 query row)
                    15,                     // a_num_cols (32 F16 elements)
                    0,                      // offset
                    false,                  // tenc_loc
                    false,                  // tenb_unsigned
                    false,                  // tena_unsigned
                    false,                  // tenb_loc (B in L1SCP)
                    B_L1_START,
                    A_L1_START,
                    TENSOR_FMA_OP_FP16,
                    (dk_chunk == 0)         // first_pass
                );
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Extract QK^T scores from vector register file.
            // After TensorFMA16A32 with AROWS=0, BCOLS=3:
            //   C[0][0..7]  in f0, C[0][8..15] in f1
            // Apply scale via SIMD, then store to scores buffer.
            // Self-contained: save/restore mask, don't rely on f-reg state across blocks.
            __asm__ volatile("" ::: "f0", "f1");
            {
                unsigned long _ms;
                __asm__ volatile(
                    "mova.x.m  %[ms]                \n\t"
                    "mov.m.x   m0, x0, 0xFF         \n\t"
                    "fbc.ps    f2, 0(%[p_scale])    \n\t"
                    "fmul.ps   f0, f0, f2           \n\t"
                    "fmul.ps   f1, f1, f2           \n\t"
                    "fsw.ps    f0, 0(%[dst])        \n\t"
                    "fsw.ps    f1, 32(%[dst])       \n\t"
                    "mova.m.x  %[ms]                \n\t"
                    : [ms] "=&r"(_ms)
                    : [dst] "r"(scores), [p_scale] "r"(&scale)
                    : "f0", "f1", "f2", "memory"
                );
            }

            // ============================================================
            // Online softmax + vectorized V accumulation
            //
            // Each asm block is self-contained: loads its inputs from memory,
            // saves/restores the SIMD mask register, and declares all used
            // f-regs as clobbers. No f-reg state leaks between blocks.
            // ============================================================
            for (int64_t j = 0; j < kv_count; ++j) {
                float s = scores[j];

                if (has_mask) {
                    float mv = get_mask_val_from_base(mask, mask_base, kv_base + j);
                    if (mv == ET_NEG_INF_F || mv != mv) continue;
                    s += mv;
                }

                const float Mold = M;
                float ms, vs;

                if (s > M) {
                    M = s;
                    ms = exp2f_et((Mold - M) * 1.4426950408889634f);
                    vs = 1.0f;
                } else {
                    ms = 1.0f;
                    vs = exp2f_et((s - M) * 1.4426950408889634f);
                }

                const char * pv = v_head + (kv_base + j) * v->nb[1];
                if (ms != 1.0f) {
                    rescale_accumulate_v_row_f16_contig(acc, pv, dv, ms);
                } else {
                    accumulate_v_row_f16_contig(acc, pv, dv, vs);
                }

                S = S * ms + vs;
            }
        }

        // Write output: acc[d] / S  (vectorized, self-contained)
        const float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
    }

    FENCE;
    return 0;
}
