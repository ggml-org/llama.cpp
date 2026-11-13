#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <math.h>
#include <stdint.h>

#include "hex-common.h"
#include "htp-trace.h"

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "timestep-ops.h"

// ggml op_params layout for TIMESTEP_EMBEDDING:
//   op_params[0] - dim
//   op_params[1] - max_period

#define TIMESTEP_BLOCK 32

static const float __attribute__((aligned(128))) timestep_iota[TIMESTEP_BLOCK] = {
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
     8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
};

struct htp_timestep_context {
    struct htp_ops_context * octx;
    const struct htp_timestep_kernel_params * kparams;
    float    freq_scale;  // -ln(max_period) / half
    uint32_t blocks_per_thread;
};

static inline void timestep_store(float * dst, uint32_t n, HVX_Vector v) {
    if (n == TIMESTEP_BLOCK && ((uintptr_t) dst & 127) == 0) {
        hvx_vmem(dst) = v;
    } else {
        hvx_vec_store_u(dst, n * sizeof(float), v);
    }
}

static void timestep_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_timestep_context * tctx = (const struct htp_timestep_context *) data;
    const struct htp_timestep_kernel_params * kparams = tctx->kparams;
    struct htp_ops_context * octx = tctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    const uint32_t half   = kparams->half;
    const uint32_t n_rows = kparams->n_rows;

    const uint32_t ik0 = tctx->blocks_per_thread * ith;
    const uint32_t ik1 = MIN(ik0 + tctx->blocks_per_thread, kparams->total_blocks);

    const HVX_Vector v_iota  = hvx_vmem(timestep_iota);
    const HVX_Vector v_scale = hvx_vec_splat_f32(tctx->freq_scale);

    // Blocks are ordered j-block major so the frequency vector is reused across rows
    uint32_t  cur_jb = UINT32_MAX;
    HVX_Vector v_freq = Q6_V_vzero();

    for (uint32_t ik = ik0; ik < ik1; ++ik) {
        const uint32_t jb = fastdiv(ik, &kparams->div_rows);
        const uint32_t i  = ik - jb * n_rows;
        const uint32_t j0 = jb * TIMESTEP_BLOCK;

        if (jb != cur_jb) {
            const HVX_Vector v_j = hvx_vec_add_f32_f32(hvx_vec_splat_f32((float) j0), v_iota);
            v_freq = hvx_vec_exp_f32(hvx_vec_mul_f32_f32(v_j, v_scale));
            cur_jb = jb;
        }

        const float timestep = *(const float *) ((const uint8_t *) src0->data + (size_t) i * src0->nb[0]);
        const HVX_Vector v_arg = hvx_vec_mul_f32_f32(hvx_vec_splat_f32(timestep), v_freq);

        HVX_Vector v_cos;
        HVX_Vector v_sin;
        hvx_vec_sincos_f32(v_arg, &v_cos, &v_sin);

        float * embed = (float *) ((uint8_t *) dst->data + (size_t) i * dst->nb[1]);
        const uint32_t n = MIN(TIMESTEP_BLOCK, half - j0);
        timestep_store(embed + j0,        n, v_cos);
        timestep_store(embed + j0 + half, n, v_sin);
    }

    if (ith == 0 && (kparams->dim & 1) != 0) {
        for (uint32_t i = 0; i < n_rows; ++i) {
            float * embed = (float *) ((uint8_t *) dst->data + (size_t) i * dst->nb[1]);
            embed[2 * half] = 0.0f;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ik1);
    FARF(HIGH, "timestep-embedding %u/%u: blocks %u:%u\n", ith, nth, ik0, ik1);
}

int op_timestep_embedding(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_timestep_kernel_params * kparams =
        (const struct htp_timestep_kernel_params *) octx->kernel_params;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t max_period = (int32_t) octx->op_params[1];

    // Only device 0 writes the output when running multi-device
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx != 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, MAX(1u, kparams->total_blocks)))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    FARF(HIGH, "timestep-embedding: %u timesteps dim %u max-period %d\n", kparams->n_rows, kparams->dim, max_period);

    struct htp_timestep_context tctx = {
        .octx              = octx,
        .kparams           = kparams,
        .freq_scale        = kparams->half ? -logf((float) max_period) / (float) kparams->half : 0.0f,
        .blocks_per_thread = fastdiv(kparams->total_blocks + n_hvx - 1, &octx->n_hvx_div),
    };

    work_queue_run(octx->ctx->hvx_queue, timestep_thread, &tctx, n_hvx);

    return HTP_STATUS_OK;
}
