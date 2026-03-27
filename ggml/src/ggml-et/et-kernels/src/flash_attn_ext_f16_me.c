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

// Build interleaved B panel for TensorFMA16A32 from K^T.
// K layout: ne[0]=dk elements contiguous (nb[0]=2), nb[1]=stride between positions.
// We need B[k_idx, kv_idx] interleaved for the engine.
// Output: 16 L1SCP lines × 64 bytes = 16 × 32 F16 values.
//
static inline void __attribute__((always_inline))
pack_k_interleaved(et_fp16_t * out,
                   const char * k_base,  // K base for this head+batch
                   int64_t kv_start,     // first KV position
                   int64_t dk_start,     // first dk element
                   int64_t kv_count,     // number of KV positions (≤16)
                   int64_t nb1_k)        // K position stride
{
    for (int j = 0; j < (int)kv_count; ++j) {
        const et_fp16_t * k_row =
            (const et_fp16_t *)(k_base + (kv_start + j) * nb1_k) + dk_start;
        for (int l = 0; l < TILE_K / 2; ++l) {
            out[l * 32 + j * 2 + 0] = k_row[2 * l + 0];
            out[l * 32 + j * 2 + 1] = k_row[2 * l + 1];
        }
    }
    // Zero-pad unused columns when kv_count < 16
    for (int j = (int)kv_count; j < TILE_KV; ++j) {
        for (int l = 0; l < TILE_K / 2; ++l) {
            out[l * 32 + j * 2 + 0] = 0;
            out[l * 32 + j * 2 + 1] = 0;
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

    // Interleaved K panel buffers: ping-pong to overlap packing/flush of the
    // next chunk with tensor FMA on the current chunk.
    et_fp16_t kpanel[2][16 * 32] __attribute__((aligned(64)));

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
            int cur_buf = 0;

            // Prepare the first K panel before entering the pipelined loop.
            pack_k_interleaved(kpanel[cur_buf], k_head, kv_base, 0,
                               kv_count, k->nb[1]);
            FENCE;
            flush_to_l2(kpanel[cur_buf], 16, 64);

            for (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                const int has_next = (dk_chunk + TILE_K) < dk;
                const int next_buf = cur_buf ^ 1;

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

                // Make sure the current panel flush completed before tensor_load.
                WAIT_CACHEOPS;

                // Load K interleaved panel into L1SCP as B (16 lines × 64B)
                tensor_load(
                    false, false,
                    B_L1_START,
                    TENSOR_LOAD_PLAIN,
                    0,
                    (uint64_t)kpanel[cur_buf],
                    0,
                    15,  // 16 lines
                    64,  // contiguous stride
                    1
                );

                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_wait(TENSOR_LOAD_WAIT_1);

                // TensorFMA16A32:
                //   BCOLS  = 3   → 16 output columns (kv_count scores)
                //   AROWS  = 0   → 1 row (single query)
                //   ACOLS  = 15  → 32 F16 K elements
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

                // While the tensor engine computes this chunk, prepare and flush
                // the next panel in the alternate buffer.
                if (has_next) {
                    pack_k_interleaved(kpanel[next_buf], k_head, kv_base, dk_chunk + TILE_K,
                                       kv_count, k->nb[1]);
                    FENCE;
                    flush_to_l2(kpanel[next_buf], 16, 64);
                }

                tensor_wait(TENSOR_FMA_WAIT);
                cur_buf = next_buf;
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
                    float diff_l2 = (Mold - M) * 1.4426950408889634f;
                    {
                        unsigned long _ms;
                        float _exp0;
                        __asm__ volatile(
                            "mova.x.m  %[ms]               \n\t"
                            "mov.m.x   m0, x0, 1           \n\t"
                            "fexp.ps   %[exp0], %[diff]    \n\t"
                            "mova.m.x  %[ms]               \n\t"
                            : [ms] "=&r"(_ms), [exp0] "=&f"(_exp0)
                            : [diff] "f"(diff_l2)
                        );
                        ms = _exp0;
                    }
                    vs = 1.0f;
                } else {
                    ms = 1.0f;
                    float diff_l2 = (s - M) * 1.4426950408889634f;
                    {
                        unsigned long _ms;
                        float _exp0;
                        __asm__ volatile(
                            "mova.x.m  %[ms]               \n\t"
                            "mov.m.x   m0, x0, 1           \n\t"
                            "fexp.ps   %[exp0], %[diff]    \n\t"
                            "mova.m.x  %[ms]               \n\t"
                            : [ms] "=&r"(_ms), [exp0] "=&f"(_exp0)
                            : [diff] "f"(diff_l2)
                        );
                        vs = _exp0;
                    }
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
