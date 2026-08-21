//******************************************************************************
// F32 Flash Attention for ET backend
//
// Supports:
// - arbitrary dk/dv (up to 128)
// - GQA (n_head_q can differ from n_head_kv)
// - mask (F16 or F32, causal pattern)
// - F16 or F32 K and V (with non-contiguous strides from KV cache permute)
//
// Limitations:
// - Q and dst must be F32
// - no sinks, ALiBi, logit softcap
//
// Parallelization strategy:
// - flatten [query position, head, outer batch] into independent rows
// - assign rows round-robin across ET threads
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdbool.h>
#include <stdint.h>

struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;      // Q tensor (F32)
    struct ggml_tensor src1;      // K tensor (F16 or F32)
    struct ggml_tensor src2;      // V tensor (F16 or F32)
    struct ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    struct ggml_tensor dst;       // Output tensor (F32)
    float              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
static inline float read_kv_f32(const char * row_base, int64_t d, int64_t nb0, int type) {
    if (type == GGML_TYPE_F32) {
        return *(const float *) (row_base + d * nb0);
    }
    // F16
    return fp16_to_fp32(*(const uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
static inline float dot_qk(const float * q, const char * k_row, int64_t dk, int64_t k_nb0, int k_type) {
    float acc = 0.0f;
    if (k_type == GGML_TYPE_F32) {
        const float * kf = (const float *) k_row;
        for (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } else {
        // F16 stride-aware read
        for (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(const uint16_t *) (k_row + i * k_nb0));
        }
    }
    return acc;
}

static inline float get_mask_val(const struct ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    const char * base = (const char *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    if (mask->type == GGML_TYPE_F32) {
        return *(const float *) (base + ik1 * mask->nb[0]);
    }
    // F16
    return fp16_to_fp32(*(const uint16_t *) (base + ik1 * mask->nb[0]));
}

int entry_point(struct ggml_et_flash_attn_ext_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0 || num_threads <= 0) {
        return 0;
    }

    struct ggml_tensor * q        = &params->src0;
    struct ggml_tensor * k        = &params->src1;
    struct ggml_tensor * v        = &params->src2;
    struct ggml_tensor * dst      = &params->dst;
    const int32_t        has_mask = params->has_mask;
    struct ggml_tensor * mask     = has_mask ? &params->mask : (struct ggml_tensor *) 0;

    const char * q_data   = (const char *) q->data;
    const char * k_data   = (const char *) k->data;
    const char * v_data   = (const char *) v->data;
    char *       dst_data = (char *) dst->data;

    const int     k_type = k->type;
    const int     v_type = v->type;
    const int64_t k_nb0  = k->nb[0];
    const int64_t v_nb0  = v->nb[0];

    const int64_t dk  = q->ne[0];  // head dim for keys/queries
    const int64_t nq  = q->ne[1];  // number of query positions
    const int64_t nhq = q->ne[2];  // number of query heads
    const int64_t no  = q->ne[3];  // outer batch

    const int64_t nk  = k->ne[1];  // number of key/value positions
    const int64_t nhk = k->ne[2];  // number of kv heads
    const int64_t dv  = v->ne[0];  // head dim for values

    if (dv > FA_DV_MAX) {
        return -1;
    }

    // GQA: query heads per kv head
    const int64_t gqa_ratio = nhq / nhk;

    const int64_t total_rows = nq * nhq * no;
    const float   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    const int use_fast_store = (dv % 16 == 0);

    for (int64_t row = thread_id; row < total_rows; row += num_threads) {
        const int64_t iq3 = row / (nhq * nq);
        const int64_t rem = row % (nhq * nq);
        const int64_t iq2 = rem / nq;  // query head index
        const int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        const int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        const float * pq = (const float *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        float * out = (float *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        const int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        const int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        float acc[FA_DV_MAX];
        for (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        float M = -3.402823466e+38f;
        float S = 0.0f;

        for (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            float mask_val = 0.0f;
            if (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                if (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    continue;
                }
            }

            const char * pk = k_data + ik1 * k->nb[1] + kv_base;
            const char * pv = v_data + ik1 * v->nb[1] + vv_base;

            float       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            const float Mold = M;

            float ms = 1.0f;
            float vs = 1.0f;
            if (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                for (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } else {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            if (v_type == GGML_TYPE_F32) {
                const float * pvf = (const float *) pv;
                for (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } else {
                for (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(const uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        const float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        if (use_fast_store) {
            for (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } else {
            for (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((volatile float *) &out[d], acc[d] * S_inv);
            }
        }
    }

    return 0;
}
