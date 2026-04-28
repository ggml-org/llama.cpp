#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define HTP_GDN_MAX_SV 128

struct htp_gdn_context {
    struct htp_ops_context * octx;
    uint32_t rows_per_thread;
    size_t state_bytes;
};

static inline HVX_Vector gdn_vec_add_f32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
#else
    return Q6_Vsf_vadd_VsfVsf(a, b);
#endif
}

static inline HVX_Vector gdn_vec_mul_f32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
#else
    return Q6_Vsf_vmpy_VsfVsf(a, b);
#endif
}

static inline float gdn_mul_dot_f32(float * restrict dst, const float * restrict mul,
        const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = gdn_vec_mul_f32(vd, vm);
        hvx_vec_store_u(dst + i * epv, 128, out);
        acc = gdn_vec_add_f32(acc, gdn_vec_mul_f32(out, vdot));
    }

    float sum = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
    const uint32_t off = nvec * epv;
    for (uint32_t i = 0; i < tail; ++i) {
        dst[off + i] *= mul[off + i];
        sum += dst[off + i] * dot[off + i];
    }
    return sum;
}

static inline float gdn_mul_scalar_dot_f32(float * restrict dst, float mul,
        const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = gdn_vec_mul_f32(vd, vmul);
        hvx_vec_store_u(dst + i * epv, 128, out);
        acc = gdn_vec_add_f32(acc, gdn_vec_mul_f32(out, vdot));
    }

    float sum = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
    const uint32_t off = nvec * epv;
    for (uint32_t i = 0; i < tail; ++i) {
        dst[off + i] *= mul;
        sum += dst[off + i] * dot[off + i];
    }
    return sum;
}

static inline float gdn_add_scaled_dot_f32(float * restrict dst, const float * restrict src,
        float scale, const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = gdn_vec_add_f32(vd, gdn_vec_mul_f32(vs, vscale));
        hvx_vec_store_u(dst + i * epv, 128, out);
        acc = gdn_vec_add_f32(acc, gdn_vec_mul_f32(out, vdot));
    }

    float sum = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
    const uint32_t off = nvec * epv;
    for (uint32_t i = 0; i < tail; ++i) {
        dst[off + i] += src[off + i] * scale;
        sum += dst[off + i] * dot[off + i];
    }
    return sum;
}

static inline void gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, const float * restrict mul,
        const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = gdn_vec_mul_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = gdn_vec_mul_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = gdn_vec_mul_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = gdn_vec_mul_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vec_store_u(dst0 + i * epv, 128, out0);
        hvx_vec_store_u(dst1 + i * epv, 128, out1);
        hvx_vec_store_u(dst2 + i * epv, 128, out2);
        hvx_vec_store_u(dst3 + i * epv, 128, out3);

        acc0 = gdn_vec_add_f32(acc0, gdn_vec_mul_f32(out0, vdot));
        acc1 = gdn_vec_add_f32(acc1, gdn_vec_mul_f32(out1, vdot));
        acc2 = gdn_vec_add_f32(acc2, gdn_vec_mul_f32(out2, vdot));
        acc3 = gdn_vec_add_f32(acc3, gdn_vec_mul_f32(out3, vdot));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));

    const uint32_t off = nvec * epv;
    for (uint32_t i = 0; i < tail; ++i) {
        dst0[off + i] *= mul[off + i];
        dst1[off + i] *= mul[off + i];
        dst2[off + i] *= mul[off + i];
        dst3[off + i] *= mul[off + i];
        sums[0] += dst0[off + i] * dot[off + i];
        sums[1] += dst1[off + i] * dot[off + i];
        sums[2] += dst2[off + i] * dot[off + i];
        sums[3] += dst3[off + i] * dot[off + i];
    }
}

static inline void gdn_mul_scalar_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float mul,
        const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = gdn_vec_mul_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = gdn_vec_mul_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = gdn_vec_mul_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = gdn_vec_mul_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vec_store_u(dst0 + i * epv, 128, out0);
        hvx_vec_store_u(dst1 + i * epv, 128, out1);
        hvx_vec_store_u(dst2 + i * epv, 128, out2);
        hvx_vec_store_u(dst3 + i * epv, 128, out3);

        acc0 = gdn_vec_add_f32(acc0, gdn_vec_mul_f32(out0, vdot));
        acc1 = gdn_vec_add_f32(acc1, gdn_vec_mul_f32(out1, vdot));
        acc2 = gdn_vec_add_f32(acc2, gdn_vec_mul_f32(out2, vdot));
        acc3 = gdn_vec_add_f32(acc3, gdn_vec_mul_f32(out3, vdot));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));

    const uint32_t off = nvec * epv;
    for (uint32_t i = 0; i < tail; ++i) {
        dst0[off + i] *= mul;
        dst1[off + i] *= mul;
        dst2[off + i] *= mul;
        dst3[off + i] *= mul;
        sums[0] += dst0[off + i] * dot[off + i];
        sums[1] += dst1[off + i] * dot[off + i];
        sums[2] += dst2[off + i] * dot[off + i];
        sums[3] += dst3[off + i] * dot[off + i];
    }
}

static inline void gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, const float * restrict src,
        const float * restrict scale, const float * restrict dot, uint32_t n,
        float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = gdn_vec_add_f32(hvx_vmemu(dst0 + i * epv), gdn_vec_mul_f32(vs, scale0));
        HVX_Vector out1 = gdn_vec_add_f32(hvx_vmemu(dst1 + i * epv), gdn_vec_mul_f32(vs, scale1));
        HVX_Vector out2 = gdn_vec_add_f32(hvx_vmemu(dst2 + i * epv), gdn_vec_mul_f32(vs, scale2));
        HVX_Vector out3 = gdn_vec_add_f32(hvx_vmemu(dst3 + i * epv), gdn_vec_mul_f32(vs, scale3));

        hvx_vec_store_u(dst0 + i * epv, 128, out0);
        hvx_vec_store_u(dst1 + i * epv, 128, out1);
        hvx_vec_store_u(dst2 + i * epv, 128, out2);
        hvx_vec_store_u(dst3 + i * epv, 128, out3);

        acc0 = gdn_vec_add_f32(acc0, gdn_vec_mul_f32(out0, vdot));
        acc1 = gdn_vec_add_f32(acc1, gdn_vec_mul_f32(out1, vdot));
        acc2 = gdn_vec_add_f32(acc2, gdn_vec_mul_f32(out2, vdot));
        acc3 = gdn_vec_add_f32(acc3, gdn_vec_mul_f32(out3, vdot));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));

    const uint32_t off = nvec * epv;
    for (uint32_t i = 0; i < tail; ++i) {
        dst0[off + i] += src[off + i] * scale[0];
        dst1[off + i] += src[off + i] * scale[1];
        dst2[off + i] += src[off + i] * scale[2];
        dst3[off + i] += src[off + i] * scale[3];
        sums[0] += dst0[off + i] * dot[off + i];
        sums[1] += dst1[off + i] * dot[off + i];
        sums[2] += dst2[off + i] * dot[off + i];
        sums[3] += dst3[off + i] * dot[off + i];
    }
}

static void gated_delta_net_f32_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];

    const uint32_t total_rows = H * n_seqs;
    if (ith >= total_rows) {
        return;
    }

    const uint32_t rq3 = n_seqs / q->ne[3];
    const uint32_t rk3 = n_seqs / k->ne[3];
    const float scale = 1.0f / sqrtf((float) S_v);

    float * dst_base       = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_base + (uint64_t) S_v * H * n_tokens * n_seqs;
    const float * state_in_base = (const float *) (uintptr_t) state->data;

    const bool kda = (g->ne[0] == S_v);
    float local_delta[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_gate[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_q[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_k[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_sums[4] __attribute__((aligned(128)));

    for (uint32_t ir = ith; ir < total_rows; ir += nth) {
        const uint32_t iv1 = ir % H;
        const uint32_t iv3 = ir / H;

        const uint32_t iq1 = iv1 % q->ne[1];
        const uint32_t ik1 = iv1 % k->ne[1];
        const uint32_t iq3 = iv3 / rq3;
        const uint32_t ik3 = iv3 / rk3;

        float * s_out = state_out_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
        const float * s_in = state_in_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
        float * s_work = s_out;
        float * delta = local_delta;
        memcpy(s_work, s_in, gctx->state_bytes);

        float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            memcpy(local_q, q_t, (size_t) S_v * sizeof(float));
            memcpy(local_k, k_t, (size_t) S_v * sizeof(float));

            // Per-4-row fusion: each 4-row batch performs phase A
            // (gate-multiply + dot with k -> delta) immediately followed
            // by phase B (add k * delta + dot with q -> attn). This keeps
            // the just-written decayed rows hot in L1/VTCM for phase B
            // and removes one full re-stream of s_work per token while
            // preserving FP order within each row.
            if (kda) {
                for (uint32_t i = 0; i < S_v; ++i) {
                    local_gate[i] = expf(g_t[i]);
                }
                uint32_t j = 0;
                for (; j + 4 <= S_v; j += 4) {
                    float * row0 = s_work + (uint64_t) (j + 0) * S_v;
                    float * row1 = s_work + (uint64_t) (j + 1) * S_v;
                    float * row2 = s_work + (uint64_t) (j + 2) * S_v;
                    float * row3 = s_work + (uint64_t) (j + 3) * S_v;
                    gdn_mul_dot4_f32(row0, row1, row2, row3, local_gate, local_k, S_v, local_sums);
                    const float d0 = (v_t[j + 0] - local_sums[0]) * beta_val;
                    const float d1 = (v_t[j + 1] - local_sums[1]) * beta_val;
                    const float d2 = (v_t[j + 2] - local_sums[2]) * beta_val;
                    const float d3 = (v_t[j + 3] - local_sums[3]) * beta_val;
                    delta[j + 0] = d0;
                    delta[j + 1] = d1;
                    delta[j + 2] = d2;
                    delta[j + 3] = d3;
                    float local_delta_b[4] __attribute__((aligned(128))) = { d0, d1, d2, d3 };
                    gdn_add_scaled_dot4_f32(row0, row1, row2, row3, local_k, local_delta_b, local_q, S_v, local_sums);
                    attn_data[j + 0] = local_sums[0] * scale;
                    attn_data[j + 1] = local_sums[1] * scale;
                    attn_data[j + 2] = local_sums[2] * scale;
                    attn_data[j + 3] = local_sums[3] * scale;
                }
                for (; j < S_v; ++j) {
                    float * row = s_work + (uint64_t) j * S_v;
                    const float sum = gdn_mul_dot_f32(row, local_gate, local_k, S_v);
                    const float dj = (v_t[j] - sum) * beta_val;
                    delta[j] = dj;
                    attn_data[j] = gdn_add_scaled_dot_f32(row, local_k, dj, local_q, S_v) * scale;
                }
            } else {
                const float gate = expf(g_t[0]);
                uint32_t j = 0;
                for (; j + 4 <= S_v; j += 4) {
                    float * row0 = s_work + (uint64_t) (j + 0) * S_v;
                    float * row1 = s_work + (uint64_t) (j + 1) * S_v;
                    float * row2 = s_work + (uint64_t) (j + 2) * S_v;
                    float * row3 = s_work + (uint64_t) (j + 3) * S_v;
                    gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, gate, local_k, S_v, local_sums);
                    const float d0 = (v_t[j + 0] - local_sums[0]) * beta_val;
                    const float d1 = (v_t[j + 1] - local_sums[1]) * beta_val;
                    const float d2 = (v_t[j + 2] - local_sums[2]) * beta_val;
                    const float d3 = (v_t[j + 3] - local_sums[3]) * beta_val;
                    delta[j + 0] = d0;
                    delta[j + 1] = d1;
                    delta[j + 2] = d2;
                    delta[j + 3] = d3;
                    float local_delta_b[4] __attribute__((aligned(128))) = { d0, d1, d2, d3 };
                    gdn_add_scaled_dot4_f32(row0, row1, row2, row3, local_k, local_delta_b, local_q, S_v, local_sums);
                    attn_data[j + 0] = local_sums[0] * scale;
                    attn_data[j + 1] = local_sums[1] * scale;
                    attn_data[j + 2] = local_sums[2] * scale;
                    attn_data[j + 3] = local_sums[3] * scale;
                }
                for (; j < S_v; ++j) {
                    float * row = s_work + (uint64_t) j * S_v;
                    const float sum = gdn_mul_scalar_dot_f32(row, gate, local_k, S_v);
                    const float dj = (v_t[j] - sum) * beta_val;
                    delta[j] = dj;
                    attn_data[j] = gdn_add_scaled_dot_f32(row, local_k, dj, local_q, S_v) * scale;
                }
            }

            attn_data += (uint64_t) S_v * H;
        }

    }
}

int op_gated_delta_net(struct htp_ops_context * octx) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    if (!q || !k || !v || !g || !beta || !state || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];

    if (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_tokens == 0 || n_seqs == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (state->ne[0] * state->ne[1] * state->ne[2] * state->ne[3] != S_v * S_v * H * n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct htp_gdn_context gctx;
    gctx.octx = octx;
    gctx.rows_per_thread = (H * n_seqs + octx->n_threads - 1) / octx->n_threads;
    gctx.state_bytes = (size_t) S_v * S_v * sizeof(float);

    worker_pool_run_func(octx->ctx->worker_pool, gated_delta_net_f32_thread, &gctx, octx->n_threads);

    return HTP_STATUS_OK;
}
