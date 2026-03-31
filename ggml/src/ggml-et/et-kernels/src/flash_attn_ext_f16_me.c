//******************************************************************************
// Flash Attention with TensorFMA16A32 for QK^T
//
// Uses the matrix engine for the QK^T dot products (F16×F16→F32),
// scalar code for online softmax and V accumulation.
//
// Hart 0: tensor engine (Q load, K load from SCP, FMA, softmax, V accum)
// Hart 1: pack K into double-buffered L2 SCP panels, flush for tensor_load
//
// Requirements:
//   - Q: F32 (converted to F16 internally)
//   - K, V: F16
//   - dk must be a multiple of 32 (TensorFMA16A32 K-tile)
//   - dv ≤ 256 (accumulator in shire-local L2 SCP)
//
// Parallelization: each minion independently processes one (qpos, head, batch)
// row, round-robin across all minion hart-0s. Hart 1 assists with K packing.
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

// Max head dimensions
#define FA_DV_MAX 256   // max value head dim (dv)
#define FA_DK_MAX 256   // max key head dim (dk) - some models use hsk > hsv

typedef uint16_t et_fp16_t;

#define ET_NEG_INF_F (-3.402823466e+38f)

// L2 SCP layout per minion:
//   [0..1023]       accumulator (FA_DV_MAX * sizeof(float))
//   [1024..3071]    kpanel buffer 0 (32 × 32 × 2 = 2048 bytes)
//   [3072..5119]    kpanel buffer 1 (2048 bytes)
// Sync: ET_BARRIER_MINION per dk_chunk.
// Double-buffering ensures hart 0 finishes buf[N%2] before hart 1
// overwrites it at chunk N+2.
#define SCP_ACC_OFF       0
#define SCP_ACC_STRIDE    (FA_DV_MAX * sizeof(float))           // 1024
#define SCP_KPANEL_SIZE   (32 * 32 * sizeof(et_fp16_t))         // 2048
#define SCP_KP0_OFF       SCP_ACC_STRIDE                        // 1024
#define SCP_KP1_OFF       (SCP_KP0_OFF + SCP_KPANEL_SIZE)       // 3072
#define SCP_PER_MINION    (SCP_KP1_OFF + SCP_KPANEL_SIZE)       // 5120

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

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
static inline void __attribute__((always_inline))
pack_k_for_transpose16(et_fp16_t * out,
                       const char * k_base,
                       int64_t kv_start,
                       int64_t dk_start,
                       int64_t kv_count,
                       int64_t nb1_k)
{
    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_mask)
        :
        :
    );

    for (int j = 0; j < (int)kv_count; ++j) {
        const et_fp16_t * k_row =
            (const et_fp16_t *)(k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t * even_row = out + (j * 2)     * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        __asm__ volatile(
            "flw.ps    f2, 0(%[src0])  \n\t"   // load row[0..15]
            "flw.ps    f3, 0(%[src1])  \n\t"   // load row[16..31]
            "fpackreph.pi f4, f2       \n\t"   // even_lo from src0
            "fpackreph.pi f6, f3       \n\t"   // even_lo from src1 (interleaved)
            "fsrli.pi  f5, f2, 16      \n\t"   // shift src0 for odd
            "fsrli.pi  f7, f3, 16      \n\t"   // shift src1 for odd (interleaved)
            "fpackreph.pi f5, f5       \n\t"   // odd from src0
            "fpackreph.pi f7, f7       \n\t"   // odd from src1
            "mov.m.x   m0, x0, 0x0F   \n\t"
            "fcmovm.ps f4, f4, f6      \n\t"   // merge even halves
            "fcmovm.ps f5, f5, f7      \n\t"   // merge odd halves
            "mov.m.x   m0, x0, 0xFF   \n\t"
            "fsw.ps    f4, 0(%[even])  \n\t"
            "fsw.ps    f5, 0(%[odd])   \n\t"
            :
            : [src0] "r"(k_row),
              [src1] "r"(k_row + 16),
              [even] "r"(even_row),
              [odd] "r"(odd_row)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[ms]            \n\t"
        :
        : [ms] "r"(old_mask)
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

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
static inline void __attribute__((always_inline))
pack_v_interleaved(et_fp16_t *out,
                   const char *v_head,
                   int64_t kv_base,
                   int64_t dv_start,
                   int64_t kv_count,
                   int64_t nb1_v)
{
    for (int k = 0; k < TILE_KV; ++k) {
        const int l = k >> 1;
        const int r = k & 1;
        et_fp16_t * const dst = out + l * 32 + r;
        if (k < (int)kv_count) {
            const et_fp16_t *v_row =
                (const et_fp16_t *)(v_head + (kv_base + k) * nb1_v) + dv_start;
            for (int n = 0; n < 16; ++n)
                dst[n * 2] = v_row[n];
        } else {
            for (int n = 0; n < 16; ++n)
                dst[n * 2] = 0;
        }
    }
}

// Prefetch KV rows for one chunk into L2.
static inline void __attribute__((always_inline))
prefetch_kv_to_l2(const char * head, int64_t kv_start, int64_t d_start,
                  int64_t kv_count, int64_t nb1)
{
    const void *base = (const void *)(head + kv_start * nb1 + d_start * 2);
    l2_prefetch(base, (uint64_t)kv_count, (uint64_t)nb1);
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
zero_acc_vec(float * acc, int64_t dv) {
    const float zero = 0.0f;
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    __asm__ volatile("fbc.ps  f2, 0(%[z])" :: [z] "r"(&zero) : "f2");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fsw.ps  f2, 0(%[a])     \n\t"
            :: [a] "r"(acc + d)
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

int entry_point(struct ggml_et_flash_attn_ext_params * params, void * env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int is_hart1 = hart_id & 1;
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

    // L2 SCP pointers for this minion
    uint64_t scp_base = local_minion * SCP_PER_MINION;
    et_fp16_t *scp_kp[2] = {
        (et_fp16_t *)et_shire_l2scp_local(scp_base + SCP_KP0_OFF),
        (et_fp16_t *)et_shire_l2scp_local(scp_base + SCP_KP1_OFF),
    };

    // ================================================================
    // Hart 1: K-panel packer (barrier-synced)
    // ================================================================
    if (is_hart1) {
        uint32_t chunk_id = 0;

        for (int64_t row = (int64_t)my_id; row < total_rows; row += (int64_t)total_minions) {
            const int64_t iq3 = row / (nhq * nq);
            const int64_t rem = row % (nhq * nq);
            const int64_t iq2 = rem / nq;
            const int64_t ik2 = iq2 / gqa_ratio;

            const char * k_head = k_data + ik2*k->nb[2] + iq3*k->nb[3];

            for (int64_t kv_base = 0; kv_base < nk; kv_base += TILE_KV) {
                const int64_t kv_count = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

                for (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                    int buf = chunk_id & 1;

                    // Prefetch K data for this chunk
                    prefetch_kv_to_l2(k_head, kv_base, dk_chunk, kv_count, k->nb[1]);

                    pack_k_for_transpose16(scp_kp[buf], k_head, kv_base, dk_chunk,
                                           kv_count, k->nb[1]);

                    FENCE;
                    flush_to_l2(scp_kp[buf], 16, 64);
                    flush_to_l2((et_fp16_t *)((char *)scp_kp[buf] + 1024), 16, 64);
                    WAIT_CACHEOPS;

                    // Barrier: signal hart 0 that data is ready
                    et_barrier(ET_BARRIER_MINION);

                    chunk_id++;
                }

            }
        }

        FENCE;
        return 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute (barrier-synced)
    // ================================================================
    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    float scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));    // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64))); // 512 bytes

    float * acc = (float *)et_shire_l2scp_local(scp_base + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    for (int64_t row = (int64_t)my_id; row < total_rows; row += (int64_t)total_minions) {
        const int64_t iq3 = row / (nhq * nq);
        const int64_t rem = row % (nhq * nq);
        const int64_t iq2 = rem / nq;
        const int64_t iq1 = rem % nq;
        const int64_t ik2 = iq2 / gqa_ratio;

        // Read Q row (F32) and convert to F16
        const float * pq = (const float *)(q_data + iq1*q->nb[1] + iq2*q->nb[2] + iq3*q->nb[3]);
        convert_q_row_f32_to_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        const char * v_head = v_data + ik2*v->nb[2] + iq3*v->nb[3];

        // Output pointer
        float * out = (float *)(dst_data + iq2*dst->nb[1] + iq1*dst->nb[2] + iq3*dst->nb[3]);

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
            // ============================================================

            for (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                int buf = chunk_id & 1;

                // Load Q
                tensor_load(
                    false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                    (uint64_t)(q_f16 + dk_chunk), 0, 0, 64, 0);

                // Barrier: wait for hart 1 to finish packing this K chunk
                et_barrier(ET_BARRIER_MINION);

                tensor_wait(TENSOR_LOAD_WAIT_0);

                // Load K from L2 SCP (hart 1 already flushed it)
                tensor_load(
                    false, false, B_L1_START, TENSOR_LOAD_TRANSPOSE16, 0,
                    (uint64_t)scp_kp[buf], 0, 15, 64, 0);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                tensor_fma(
                    (kv_count < TILE_KV), 3, 0, 15, 0,
                    false, false, false, false,
                    B_L1_START, A_L1_START,
                    TENSOR_FMA_OP_FP16, (dk_chunk == 0));
                tensor_wait(TENSOR_FMA_WAIT);

                chunk_id++;
            }

            // Prefetch V rows for this tile
            for (int64_t d = 0; d < dv; d += 32)
                prefetch_kv_to_l2(v_head, kv_base, d, kv_count, v->nb[1]);

            // Extract QK^T scores from vector register file
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
            // Two-phase softmax + V accumulation
            // ============================================================

            float weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                for (int64_t j = 0; j < kv_count; ++j) {
                    if (has_mask) {
                        float mv = get_mask_val_from_base(mask, mask_base, kv_base + j);
                        if (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } else {
                            scores[j] += mv;
                        }
                    }
                }
                for (int64_t j = kv_count; j < TILE_KV; ++j)
                    scores[j] = ET_NEG_INF_F;

                // A1b: SIMD horizontal max across all 16 scores
                float tile_max;
                {
                    unsigned long _ms;
                    __asm__ volatile(
                        "mova.x.m  %[ms]              \n\t"
                        "mov.m.x   m0, x0, 0xFF       \n\t"
                        "flw.ps    f2, 0(%[sc])       \n\t"
                        "flw.ps    f3, 32(%[sc])      \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fswizz.ps f3, f2, 0xB1       \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fswizz.ps f3, f2, 0x4E       \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fmvz.x.ps t0, f2, 4          \n\t"
                        "fbcx.ps   f3, t0             \n\t"
                        "fmax.ps   %[tm], f2, f3      \n\t"
                        "mova.m.x  %[ms]              \n\t"
                        : [ms] "=&r"(_ms), [tm] "=f"(tile_max)
                        : [sc] "r"(scores)
                        : "f2", "f3", "t0", "memory"
                    );
                }

                if (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    if (tile_max > M) {
                        float rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        const float log2e = 1.4426950408889634f;
                        float S_tile;
                        unsigned long _ms;
                        __asm__ volatile(
                            "mova.x.m  %[ms]              \n\t"
                            "mov.m.x   m0, x0, 0xFF       \n\t"
                            "flw.ps    f2, 0(%[sc])       \n\t"
                            "fbc.ps    f4, 0(%[pM])       \n\t"
                            "flw.ps    f3, 32(%[sc])      \n\t"
                            "fbc.ps    f5, 0(%[pL])       \n\t"
                            "fsub.ps   f2, f2, f4         \n\t"
                            "fsub.ps   f3, f3, f4         \n\t"
                            "fmul.ps   f2, f2, f5         \n\t"
                            "fmul.ps   f3, f3, f5         \n\t"
                            "fexp.ps   f2, f2             \n\t"
                            "fexp.ps   f3, f3             \n\t"
                            "fsw.ps    f2, 0(%[wt])       \n\t"
                            "fsw.ps    f3, 32(%[wt])      \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fswizz.ps f3, f2, 0xB1       \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fswizz.ps f3, f2, 0x4E       \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fmvz.x.ps t0, f2, 4          \n\t"
                            "fbcx.ps   f3, t0             \n\t"
                            "fadd.ps   %[st], f2, f3, rne \n\t"
                            "mova.m.x  %[ms]              \n\t"
                            : [ms] "=&r"(_ms), [st] "=f"(S_tile)
                            : [pM] "r"(&M), [pL] "r"(&log2e),
                              [sc] "r"(scores), [wt] "r"(weights)
                            : "f2", "f3", "f4", "f5", "t0", "memory"
                        );
                        S += S_tile;
                    }

                    // Phase B: weights @ V via TensorFMA16A32
                    {
                        // B1: convert weights F32 → F16
                        convert_q_row_f32_to_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_to_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Load A (weights)
                        tensor_load(false, false, A_L1_START,
                                    TENSOR_LOAD_PLAIN, 0,
                                    (uint64_t)w_f16_buf, 0, 0, 64, 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        // B2: process dv in chunks of 16
                        if (kv_count == TILE_KV) {
                            // Full tile: pipelined with double-buffered B
                            const uintptr_t v_base = (uintptr_t)v_head
                                + kv_base * v->nb[1];
                            const uint64_t nb1_v = (uint64_t)v->nb[1];
                            uint64_t b_cur = 8;

                            // Preload first chunk
                            tensor_load(false, false, b_cur,
                                        TENSOR_LOAD_INTERLEAVE16, 0,
                                        (uint64_t)v_base,
                                        0, 7, nb1_v, 0);
                            tensor_wait(TENSOR_LOAD_WAIT_0);

                            for (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                const uint64_t b_nxt = b_cur ^ 24;

                                if (dv_off + 16 < dv) {
                                    tensor_load(false, false, b_nxt,
                                                TENSOR_LOAD_INTERLEAVE16, 0,
                                                (uint64_t)(v_base + (dv_off + 16) * 2),
                                                0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0,
                                           false, false, false, false,
                                           b_cur, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ volatile("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ volatile(
                                        "mova.x.m  %[ms]            \n\t"
                                        "mov.m.x   m0, x0, 0xFF     \n\t"
                                        "flw.ps    f2, 0(%[pa])     \n\t"
                                        "flw.ps    f3, 32(%[pa])    \n\t"
                                        "fadd.ps   f0, f0, f2       \n\t"
                                        "fadd.ps   f1, f1, f3       \n\t"
                                        "fsw.ps    f0, 0(%[pa])     \n\t"
                                        "fsw.ps    f1, 32(%[pa])    \n\t"
                                        "mova.m.x  %[ms]            \n\t"
                                        : [ms] "=&r"(_ms)
                                        : [pa] "r"(acc + dv_off)
                                        : "f0", "f1", "f2", "f3", "memory"
                                    );
                                }

                                if (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_LOAD_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } else {
                            // Partial tile: software pack, no pipeline
                            for (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_head, kv_base,
                                                   dv_off, kv_count, v->nb[1]);
                                FENCE;
                                flush_to_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_load(false, false, B_L1_START,
                                            TENSOR_LOAD_PLAIN, 0,
                                            (uint64_t)vpanel_buf, 0, 7, 64, 0);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0,
                                           false, false, false, false,
                                           B_L1_START, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ volatile("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ volatile(
                                        "mova.x.m  %[ms]            \n\t"
                                        "mov.m.x   m0, x0, 0xFF     \n\t"
                                        "flw.ps    f2, 0(%[pa])     \n\t"
                                        "flw.ps    f3, 32(%[pa])    \n\t"
                                        "fadd.ps   f0, f0, f2       \n\t"
                                        "fadd.ps   f1, f1, f3       \n\t"
                                        "fsw.ps    f0, 0(%[pa])     \n\t"
                                        "fsw.ps    f1, 32(%[pa])    \n\t"
                                        "mova.m.x  %[ms]            \n\t"
                                        : [ms] "=&r"(_ms)
                                        : [pa] "r"(acc + dv_off)
                                        : "f0", "f1", "f2", "f3", "memory"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Write output: acc[d] / S
        const float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
    }

    FENCE;
    return 0;
}
