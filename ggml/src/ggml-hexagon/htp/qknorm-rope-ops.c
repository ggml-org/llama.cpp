#include <HAP_farf.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "htp-ctx.h"
#include "hvx-norm.h"

struct qknorm_rope_context {
    struct htp_context * ctx;
    const float * src;
    const float * weight;
    const float * theta;
    float * dst;
    uint32_t head_dim;
    uint32_t n_heads;
    uint32_t n_tokens;
    uint32_t n_batches;
    uint32_t src_head_stride;
    uint32_t src_token_stride;
    uint32_t src_batch_stride;
    float epsilon;
    uint32_t n_threads;
};

static inline void qknorm_rope_pack_theta(float * restrict packed, const float * restrict src, uint32_t head_dim) {
    for (uint32_t i = 0; i < head_dim / 2; ++i) {
        packed[2 * i + 0] = src[4 * i + 0];
        packed[2 * i + 1] = src[4 * i + 2];
    }
}

static inline void qknorm_rope_apply_interleaved(float * restrict dst, const float * restrict src,
                                                  const float * restrict theta, uint32_t head_dim) {
    const HVX_Vector * vx = (const HVX_Vector *) src;
    const HVX_Vector * vt = (const HVX_Vector *) theta;
    HVX_Vector * vo = (HVX_Vector *) dst;

    for (uint32_t i = 0; i < head_dim / 64; ++i) {
        const HVX_Vector x0 = *vx++;
        const HVX_Vector x1 = *vx++;
        const HVX_Vector t0 = *vt++;
        const HVX_Vector t1 = *vt++;
        const HVX_VectorPair x_even_odd = Q6_W_vdeal_VVR(x1, x0, -4);
        const HVX_VectorPair cos_sin = Q6_W_vdeal_VVR(t1, t0, -4);

        const HVX_Vector x0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(x_even_odd), Q6_V_lo_W(cos_sin));
        const HVX_Vector x0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(x_even_odd), Q6_V_hi_W(cos_sin));
        const HVX_Vector x1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(x_even_odd), Q6_V_lo_W(cos_sin));
        const HVX_Vector x1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(x_even_odd), Q6_V_hi_W(cos_sin));
        const HVX_Vector even = Q6_Vqf32_vsub_Vqf32Vqf32(x0_c, x1_s);
        const HVX_Vector odd = Q6_Vqf32_vadd_Vqf32Vqf32(x0_s, x1_c);
        const HVX_VectorPair out = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(odd), Q6_Vsf_equals_Vqf32(even), -4);

        *vo++ = Q6_V_lo_W(out);
        *vo++ = Q6_V_hi_W(out);
    }
}

static void qknorm_rope_worker(unsigned int n, unsigned int i, void * data) {
    struct qknorm_rope_context * c = (struct qknorm_rope_context *) data;
    struct htp_thread_trace * tr = &c->ctx->trace[i];
    const uint32_t token_first = (uint32_t) (((uint64_t) c->n_tokens * i) / n);
    const uint32_t token_last = (uint32_t) (((uint64_t) c->n_tokens * (i + 1)) / n);
    float theta_packed[256] __attribute__((aligned(128)));
    float normalized[256] __attribute__((aligned(128)));

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, token_first);
    for (uint32_t token = token_first; token < token_last; ++token) {
        qknorm_rope_pack_theta(theta_packed, c->theta + (size_t) token * 2 * c->head_dim, c->head_dim);
        for (uint32_t batch = 0; batch < c->n_batches; ++batch) {
            for (uint32_t head = 0; head < c->n_heads; ++head) {
                const float * src = c->src + (size_t) batch * c->src_batch_stride +
                                    (size_t) token * c->src_token_stride +
                                    (size_t) head * c->src_head_stride;
                float * dst = c->dst + (((size_t) batch * c->n_heads + head) * c->n_tokens + token) * c->head_dim;
                hvx_fast_rms_norm_mul_f32((const uint8_t *) src, (const uint8_t *) c->weight,
                                           (uint8_t *) normalized, c->head_dim, c->epsilon);
                qknorm_rope_apply_interleaved(dst, normalized, theta_packed, c->head_dim);
            }
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, token_first);
}

int op_qknorm_rope(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * weight = octx->src[1];
    const struct htp_tensor * theta = octx->src[2];
    const struct htp_tensor * dst = octx->dst;

    if (!src || !weight || !theta || !dst || src->type != HTP_TYPE_F32 || weight->type != HTP_TYPE_F32 ||
        theta->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t head_dim = src->ne[0];
    const uint32_t n_heads = src->ne[1];
    const uint32_t n_tokens = src->ne[2];
    const uint32_t n_batches = src->ne[3];
    const uint64_t output_elements = (uint64_t) head_dim * n_heads * n_tokens * n_batches;
    const uint64_t weight_elements = (uint64_t) weight->ne[0] * weight->ne[1] * weight->ne[2] * weight->ne[3];
    const uint64_t theta_elements = (uint64_t) theta->ne[0] * theta->ne[1] * theta->ne[2] * theta->ne[3];
    const uint64_t dst_elements = (uint64_t) dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    if (head_dim == 0 || head_dim > 256 || head_dim % 64 != 0 || n_heads == 0 || n_tokens == 0 || n_batches == 0 ||
        weight_elements != head_dim || theta_elements < (uint64_t) n_tokens * 2 * head_dim ||
        dst_elements != output_elements || src->nb[0] != sizeof(float) ||
        src->nb[1] % sizeof(float) || src->nb[2] % sizeof(float) || src->nb[3] % sizeof(float)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    float epsilon = 0.0f;
    memcpy(&epsilon, &octx->op_params[0], sizeof(epsilon));
    const uint32_t n_threads = n_tokens < octx->n_threads ? n_tokens : octx->n_threads;
    struct qknorm_rope_context c = {
        .ctx = octx->ctx,
        .src = (const float *) src->data,
        .weight = (const float *) weight->data,
        .theta = (const float *) theta->data,
        .dst = (float *) dst->data,
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_tokens = n_tokens,
        .n_batches = n_batches,
        .src_head_stride = src->nb[1] / sizeof(float),
        .src_token_stride = src->nb[2] / sizeof(float),
        .src_batch_stride = src->nb[3] / sizeof(float),
        .epsilon = epsilon,
        .n_threads = n_threads,
    };

    work_queue_run(octx->ctx->work_queue, qknorm_rope_worker, &c, n_threads);
    return HTP_STATUS_OK;
}
