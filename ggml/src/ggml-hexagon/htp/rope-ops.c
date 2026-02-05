#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

// Redefined the types GGML_ROPE_TYPE_NORMAL & GGML_ROPE_TYPE_NEOX as we cant include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2

#define htp_rope_preamble              \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

struct rope_th_ctx {
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t sections[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    float theta_scale;
    float corr_dims[2];

    struct htp_ops_context * octx;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);

    return (1 - MIN(1, MAX(0, y)));
}

static void rope_cache_init(const float    theta_base,
                            const float    freq_scale,
                            const float *  freq_factors,
                            float *        corr_dims,
                            const uint32_t ne0,
                            const float    ext_factor,
                            const float    mscale,
                            float *        cache,
                            const float    theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;

    for (uint32_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;

        float theta_extrap = theta / ff;

        // Get n-d rotational scaling corrected for extrapolation
        float theta_interp = freq_scale * theta_extrap;
        float theta_final  = theta_interp;
        float mscale_final = mscale;

        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
            theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

            // Get n-d magnitude scaling corrected for interpolation
            mscale_final *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        cache[i0 + 0] = cosf(theta_final) * mscale_final;
        cache[i0 + 1] = sinf(theta_final) * mscale_final;

        theta *= theta_scale;
    }
}

#define M_PI 3.1415926535897932384626433

static void rope_corr_dims(int     n_dims,
                           int     n_ctx_orig,
                           float   freq_base,
                           float   beta_fast,
                           float   beta_slow,
                           float * dims) {
    float start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (float) M_PI)) / (2 * logf(freq_base)));
    float end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (float) M_PI)) / (2 * logf(freq_base)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, end);
}

static void init_rope_ctx(struct rope_th_ctx * rope_ctx, struct htp_ops_context * octx) {
    memset(rope_ctx, 0, sizeof(struct rope_th_ctx));

    const int32_t * op_params = &octx->op_params[0];

    rope_ctx->n_dims     = ((const int32_t *) op_params)[1];
    rope_ctx->mode       = ((const int32_t *) op_params)[2];
    rope_ctx->n_ctx_orig = ((const int32_t *) op_params)[4];

    memcpy(&rope_ctx->freq_base, (int32_t *) op_params + 5, sizeof(float));
    memcpy(&rope_ctx->freq_scale, (int32_t *) op_params + 6, sizeof(float));
    memcpy(&rope_ctx->ext_factor, (int32_t *) op_params + 7, sizeof(float));
    memcpy(&rope_ctx->attn_factor, (int32_t *) op_params + 8, sizeof(float));
    memcpy(&rope_ctx->beta_fast, (int32_t *) op_params + 9, sizeof(float));
    memcpy(&rope_ctx->beta_slow, (int32_t *) op_params + 10, sizeof(float));
    memcpy(&rope_ctx->sections, (int32_t *) op_params + 11, sizeof(int) * 4);

    rope_ctx->theta_scale = powf(rope_ctx->freq_base, -2.0f / rope_ctx->n_dims);

    rope_corr_dims(rope_ctx->n_dims, rope_ctx->n_ctx_orig, rope_ctx->freq_base, rope_ctx->beta_fast,
                   rope_ctx->beta_slow, rope_ctx->corr_dims);

    rope_ctx->octx = octx;
    FARF(HIGH, "rope-f32 n_dims:%d, ext_factor:%.6f, theta_scale:%.6f, attn_factor:%.6f\n", rope_ctx->n_dims,
         rope_ctx->ext_factor, rope_ctx->theta_scale, rope_ctx->attn_factor);
}

static void hvx_calc_rope_neox_f32(const float * restrict src0,
                                   float * restrict dst,
                                   const int num_elems,
                                   const float * restrict theta_cache) {
    // for (int i = 0; i < num_elems; i += 2) {
    //const float cos_theta = theta_cache[i + 0];
    //const float sin_theta = theta_cache[i + 1];

    //const float x0 = src[0];
    //const float x1 = src[num_elems/2];

    //dst[0] = x0*cos_theta - x1*sin_theta;
    //dst[num_elems/2] = x0*sin_theta + x1*cos_theta;

    //src += 1;
    //dst += 1;
    // }

    const uint8_t * restrict src0_curr  = (const uint8_t *) src0;
    const uint8_t * restrict theta_curr = (const uint8_t *) theta_cache;
    uint8_t * restrict dst_curr         = (uint8_t *) dst;

    int step_of_1 = num_elems >> 6;  // 6 because we process two vectors at once
    int half_size = (sizeof(float) * (num_elems / 2));

    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v0 = *(HVX_Vector *) src0_curr;
        HVX_Vector v1 = *(HVX_Vector *) (src0_curr + half_size);

        HVX_Vector v2 = *(HVX_Vector *) theta_curr;
        HVX_Vector v3 = *(HVX_Vector *) (theta_curr + VLEN);

        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);  // vcos_sin[0] = cos_theta, vcos_sin[1] = sin_theta

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        *(HVX_Vector *) dst_curr               = Q6_Vsf_equals_Vqf32(v4);
        *(HVX_Vector *) (dst_curr + half_size) = Q6_Vsf_equals_Vqf32(v5);

        src0_curr += VLEN;
        theta_curr += 2 * VLEN;
        dst_curr += VLEN;
    }
}

static void hvx_calc_rope_f32(const float * restrict src0,
                              float * restrict dst,
                              const int num_elems,
                              const float * restrict theta_cache) {
    // for (int i = 0; i < num_elems; i += 2) {
    //const float cos_theta = theta_cache[i + 0];
    //const float sin_theta = theta_cache[i + 1];

    //const float x0 = src[0];
    //const float x1 = src[1];

    //dst[0] = x0*cos_theta - x1*sin_theta;
    //dst[1] = x0*sin_theta + x1*cos_theta;

    //src += 2;
    //dst += 2;
    // }

    const uint8_t * restrict src0_curr  = (const uint8_t *) src0;
    const uint8_t * restrict theta_curr = (const uint8_t *) theta_cache;
    uint8_t * restrict dst_curr         = (uint8_t *) dst;

    int step_of_1 = num_elems >> 6;  // 6 because we process two vectors at once

    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v0 = *(HVX_Vector *) src0_curr;
        HVX_Vector v1 = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v2 = *(HVX_Vector *) theta_curr;
        HVX_Vector v3 = *(HVX_Vector *) (theta_curr + VLEN);

        HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);  // vx0_x1[0] = x0, vx0_x1[1] = x1
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);  // vcos_sin[0] = cos_theta, vcos_sin[1] = sin_theta

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        *(HVX_Vector *) dst_curr          = Q6_V_lo_W(vstore);
        *(HVX_Vector *) (dst_curr + VLEN) = Q6_V_hi_W(vstore);

        src0_curr += 2 * VLEN;
        theta_curr += 2 * VLEN;
        dst_curr += 2 * VLEN;
    }
}

static void rope_hex_f32(struct rope_th_ctx * rope_ctx,
                         const uint32_t       ir0,
                         const uint32_t       ir1,
                         int                  nth,
                         int                  ith,
                         const int            opt_path) {
    struct htp_ops_context * octx = rope_ctx->octx;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * src2 = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    const int32_t mode    = rope_ctx->mode;
    const bool    is_neox = mode & HTP_ROPE_TYPE_NEOX;

    htp_rope_preamble;

    const int32_t * pos = (const int32_t *) src1->data;

    float * wp0 = (float *) (octx->src0_spad.data + (ith * nb01));

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        freq_factors = (const float *) src2->data;
    }

    const uint32_t i1_end       = MIN(ir1, ne1);
    const int32_t  half_dims    = rope_ctx->n_dims / 2;
    const size_t   remain_bytes = (ne0 - rope_ctx->n_dims) * sizeof(float);
    for (uint32_t i3 = 0; i3 < ne3; i3++) {      // batch
        for (uint32_t i2 = 0; i2 < ne2; i2++) {  // seq-len
            const int32_t p = pos[i2];

            rope_cache_init(p, rope_ctx->freq_scale, freq_factors, rope_ctx->corr_dims, ne0, rope_ctx->ext_factor,
                            rope_ctx->attn_factor, wp0, rope_ctx->theta_scale);

            for (uint32_t i1 = ir0; i1 < i1_end; i1++) {  // attn-heads
                const float * src      = (float *) ((char *) src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                float *       dst_data = (float *) ((char *) dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                const float * src_loc      = src;
                float *       dst_data_loc = dst_data;

                if (1 == opt_path) {
                    if (is_neox) {
                        hvx_calc_rope_neox_f32(src_loc, dst_data_loc, rope_ctx->n_dims, wp0);
                    } else {
                        hvx_calc_rope_f32(src_loc, dst_data_loc, rope_ctx->n_dims, wp0);
                    }

                    src_loc += rope_ctx->n_dims;
                    dst_data_loc += rope_ctx->n_dims;
                } else {
                    for (uint32_t i0 = 0; i0 < rope_ctx->n_dims; i0 += 2) {
                        const float cos_theta = wp0[i0 + 0];
                        const float sin_theta = wp0[i0 + 1];

                        if (is_neox) {
                            const float x0 = src_loc[0];
                            const float x1 = src_loc[half_dims];

                            dst_data_loc[0]         = x0 * cos_theta - x1 * sin_theta;
                            dst_data_loc[half_dims] = x0 * sin_theta + x1 * cos_theta;

                            src_loc += 1;
                            dst_data_loc += 1;
                        } else {
                            const float x0 = src_loc[0];
                            const float x1 = src_loc[1];

                            dst_data_loc[0] = x0 * cos_theta - x1 * sin_theta;
                            dst_data_loc[1] = x0 * sin_theta + x1 * cos_theta;

                            src_loc += 2;
                            dst_data_loc += 2;
                        }
                    }

                    src_loc += (is_neox ? half_dims : 0);
                    dst_data_loc += (is_neox ? half_dims : 0);
                }

                // TODO: use simd to speed up the remaining elements copy
                memcpy(dst_data_loc, src_loc, remain_bytes);
            }
        }
    }
}

static void rope_job_f32_per_thread(struct rope_th_ctx * rope_ctx, int nth, int ith) {
    struct htp_ops_context * octx = rope_ctx->octx;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    htp_rope_preamble;

    const uint32_t src0_nrows            = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src0_nrows_per_thread = octx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    int is_aligned = 1;
    int opt_path   = 0;
    if ((0 == hex_is_aligned((void *) src0->data, VLEN)) || (0 == hex_is_aligned((void *) src1->data, VLEN)) ||
        (0 == hex_is_aligned((void *) dst->data, VLEN))) {
        FARF(HIGH, "rope-f32: unaligned addresses in rope op, possibly slower execution\n");
        is_aligned = 0;
    }
    if ((1 == is_aligned) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    rope_hex_f32(rope_ctx, src0_start_row, src0_end_row, nth, ith, opt_path);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "rope-f32: %d/%d/%d: (%u:%u) usec %u\n", ith, nth, opt_path, src0_start_row, src0_end_row,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void rope_job_dispatcher_f32(unsigned int n, unsigned int i, void * data) {
    struct rope_th_ctx * rope_ctx = (struct rope_th_ctx *) data;

    rope_job_f32_per_thread(rope_ctx, n, i);
}

static int execute_op_rope_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * src2 = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    worker_callback_t op_func;
    const char *      op_type = NULL;

    struct rope_th_ctx rope_ctx;

    switch (octx->op) {
        case HTP_OP_ROPE:
            op_func = rope_job_dispatcher_f32;
            op_type = "rope-f32";

            init_rope_ctx(&rope_ctx, octx);
            break;

        default:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_threads = octx->n_threads;

    const size_t src0_row_size = src0->nb[1];
    const size_t src1_row_size = src0_row_size;
    const size_t dst_row_size  = dst->nb[1];

    // VTCM scratchpads for all tensors
    // N rows per thread, padded to HVX vector size
    octx->dst_spad.size  = hex_round_up(dst_row_size, 128) * n_threads;
    octx->src0_spad.size = hex_round_up(src0_row_size, 128) * n_threads;
    octx->src1_spad.size = hex_round_up(src1_row_size, 128) * n_threads;

    size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

    if (src2->ne[0]) {
        FARF(HIGH,
             "%s: %ux%ux%ux%u (x %ux%ux%ux%u x %ux%ux%ux%u) -> %ux%ux%ux%u : src0-spad-size %u src1-spad-size %u "
             "dst-spad-size %u\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], src2->ne[0], src2->ne[1], src2->ne[2], src2->ne[3], dst->ne[0], dst->ne[1], dst->ne[2],
             dst->ne[3], octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size);
    } else {
        FARF(HIGH,
             "%s: %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], octx->src0_spad.size, octx->src1_spad.size,
             octx->dst_spad.size);
    }

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs             = MIN(n_threads, src0_nrows);
        octx->src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, op_func, &rope_ctx, n_jobs);
    }

    return err;
}

int op_rope(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_rope_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
