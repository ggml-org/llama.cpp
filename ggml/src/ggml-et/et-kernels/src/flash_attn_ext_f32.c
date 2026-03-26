//******************************************************************************
// Very basic F32 Flash Attentoin
//
// Limitations:
// - Q, K, V, dst are all contiguous F32 tensors
// - no mask, sinks, ALiBi, logit softcap
//
// Parallelization strategy:
// - flatten [query position, head, outer batch] into independent rows
// - assign rows round-robin across ET threads
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;     // Q tensor (F32)
    struct ggml_tensor src1;     // K tensor (F32)
    struct ggml_tensor src2;     // V tensor (F32)
    struct ggml_tensor dst;      // Output tensor (F32)
    float scale;                 // Scale factor applied to QK
};

static inline float dot_f32(const float * a, const float * b, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

int entry_point(struct ggml_et_flash_attn_ext_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    const int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0 || num_threads <= 0) {
        return 0;
    }

    struct ggml_tensor * q   = &params->src0;
    struct ggml_tensor * k   = &params->src1;
    struct ggml_tensor * v   = &params->src2;
    struct ggml_tensor * dst = &params->dst;

    if (q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return -1;
    }

    float * q_data = (float *) q->data;
    float * k_data = (float *) k->data;
    float * v_data = (float *) v->data;
    float * dst_data = (float *) dst->data;
    if (!q_data || !k_data || !v_data || !dst_data) {
        return -1;
    }

    const int64_t dk = q->ne[0];
    const int64_t nq = q->ne[1];
    const int64_t nh = q->ne[2];
    const int64_t no = q->ne[3];
    const int64_t nk = k->ne[1];
    const int64_t dv = v->ne[0];

    if (dk != 16 ||
        k->ne[0] != 16 ||
        v->ne[0] != 16 ||
        dst->ne[0] != 16 ||
        q->ne[1] != dst->ne[2] ||
        q->ne[2] != dst->ne[1] ||
        q->ne[3] != dst->ne[3] ||
        k->ne[1] != v->ne[1] ||
        k->ne[2] != v->ne[2] ||
        k->ne[3] != v->ne[3] ||
        q->ne[2] != k->ne[2] ||
        q->ne[3] != k->ne[3]) {
        return -1;
    }

    const int64_t total_rows = nq * nh * no;
    const float scale = params->scale;

    for (int64_t row = thread_id; row < total_rows; row += num_threads) {
        const int64_t iq3 = row / (nh * nq);
        const int64_t rem = row % (nh * nq);
        const int64_t iq2 = rem / nq;
        const int64_t iq1 = rem % nq;

        const float * pq = (const float *) ((char *) q_data + iq1*q->nb[1] + iq2*q->nb[2] + iq3*q->nb[3]);
        float * out = (float *) ((char *) dst_data + iq2*dst->nb[1] + iq1*dst->nb[2] + iq3*dst->nb[3]);

        float acc[16];
        for (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        float M = -3.402823466e+38f;
        float S = 0.0f;

        for (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            const float * pk = (const float *) ((char *) k_data + ik1*k->nb[1] + iq2*k->nb[2] + iq3*k->nb[3]);
            const float * pv = (const float *) ((char *) v_data + ik1*v->nb[1] + iq2*v->nb[2] + iq3*v->nb[3]);

            const float s = dot_f32(pq, pk, dk) * scale;
            const float Mold = M;

            float ms = 1.0f;
            float vs = 1.0f;
            if (s > M) {
                M = s;
                ms = et_expf(Mold - M);
                for (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } else {
                vs = et_expf(s - M);
            }

            for (int64_t d = 0; d < dv; ++d) {
                acc[d] += pv[d] * vs;
            }

            S = S * ms + vs;
        }

        const float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        for (int64_t d = 0; d < dv; ++d) {
            out[d] = acc[d] * S_inv;
        }
    }

    return 0;
}
