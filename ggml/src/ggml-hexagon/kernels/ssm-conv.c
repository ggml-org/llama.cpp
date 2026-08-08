#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-dma.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "hvx-utils.h"

#define htp_ssm_conv_tensors_preamble                           \
    const struct htp_tensor * restrict src0 = octx->src[0];     \
    const struct htp_tensor * restrict src1 = octx->src[1];     \
    const struct htp_tensor * restrict dst  = octx->dst;        \
    struct htp_spad * restrict src0_spad    = &octx->src0_spad; \
    struct htp_spad * restrict src1_spad    = &octx->src1_spad; \
    struct htp_spad * restrict dst_spad     = &octx->dst_spad;  \
                                                                \
    const uint32_t ne00 = src0->ne[0];                          \
    const uint32_t ne01 = src0->ne[1];                          \
    const uint32_t ne02 = src0->ne[2];                          \
    const uint32_t ne03 = src0->ne[3];                          \
                                                                \
    const uint32_t ne10 = src1->ne[0];                          \
    const uint32_t ne11 = src1->ne[1];                          \
    const uint32_t ne12 = src1->ne[2];                          \
    const uint32_t ne13 = src1->ne[3];                          \
                                                                \
    const uint32_t ne0 = dst->ne[0];                            \
    const uint32_t ne1 = dst->ne[1];                            \
    const uint32_t ne2 = dst->ne[2];                            \
    const uint32_t ne3 = dst->ne[3];                            \
                                                                \
    const uint32_t nb00 = src0->nb[0];                          \
    const uint32_t nb01 = src0->nb[1];                          \
    const uint32_t nb02 = src0->nb[2];                          \
    const uint32_t nb03 = src0->nb[3];                          \
                                                                \
    const uint32_t nb10 = src1->nb[0];                          \
    const uint32_t nb11 = src1->nb[1];                          \
    const uint32_t nb12 = src1->nb[2];                          \
    const uint32_t nb13 = src1->nb[3];                          \
                                                                \
    const uint32_t nb0 = dst->nb[0];                            \
    const uint32_t nb1 = dst->nb[1];                            \
    const uint32_t nb2 = dst->nb[2];                            \
    const uint32_t nb3 = dst->nb[3];

struct htp_ssm_conv_context {
    struct htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t d_inner_tile;
    uint64_t t_start;
};

#define htp_ssm_conv_preamble                                                   \
    struct htp_ssm_conv_context * scctx = (struct htp_ssm_conv_context *) data; \
    struct htp_ops_context *      octx  = scctx->octx;                          \
    htp_ssm_conv_tensors_preamble;                                              \
    dma_queue * dma_queue = octx->ctx->dma[ith];

// Scalar FP32 SSM_CONV implementation
static void ssm_conv_thread_f32_f32(unsigned int nth, unsigned int ith, void *data) {
    htp_ssm_conv_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t src0_stride_seq   = src0->nb[2] / sizeof(float); // stride for sequence dimension
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t dst_stride_token  = dst->nb[1]  / sizeof(float); // stride for token dimension
    const uint32_t dst_stride_seq    = dst->nb[2]  / sizeof(float); // stride for sequence dimension

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    // Calculate row range for this thread
    const uint32_t d_inner_per_thread = scctx->nrows_per_thread;
    const uint32_t d_inner_start = d_inner_per_thread * ith;
    const uint32_t d_inner_end   = MIN(d_inner_start + d_inner_per_thread, d_inner);

    // No work for this thread
    if (d_inner_start >= d_inner_end) {
        return;
    }

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            for (uint32_t i1 = d_inner_start; i1 < d_inner_end; ++i1) {
                float sumf = 0.0f;

                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    const uint32_t src0_idx = (i2 + i0) + i1 * src0_stride_inner + i3 * src0_stride_seq;
                    const uint32_t src1_idx = i0 + i1 * src1_stride_inner;

                    sumf += src0_data[src0_idx] * src1_data[src1_idx];
                }

                const uint32_t dst_idx = i1 + i2 * dst_stride_token + i3 * dst_stride_seq;
                dst_data[dst_idx] = sumf;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "ssm-conv-f32 %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], d_inner_start, d_inner_end,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}


// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
static inline void hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX FP32 SSM_CONV implementation - channel-vectorized HVX kernel with src0/src1
// transposed into VTCM.
//
// VTCM layouts (per thread):
//   src1_T : {d_inner_stride, d_conv}       - staged once per launch (small).
//   src0_T : {d_inner_tile,     ncs}        - staged per d_inner-tile.
//
// d_inner_tile is chosen so that per-thread VTCM stays under the budget.
// Each thread iterates ceil(d_inner_per_thread d_inner_tile) tiles serially.
#define HTP_SSM_CONV_VTCM_BUDGET (1u << 20) // 1 MiB per thread

// Scalar transpose: src1 {d_conv, d_inner} (DDR) -> {d_inner_stride, d_conv} (VTCM)
static inline void transpose_src1(const float * src1_data,
                                  uint32_t      src1_stride_inner,
                                  uint32_t      i1_off,
                                  uint32_t      d_inner_per_thread,
                                  uint32_t      d_inner_stride,
                                  uint32_t      d_conv,
                                  float *       src1_T) {
    for (uint32_t i = 0; i < d_inner_per_thread; ++i) {
        const float * src_row = src1_data + (i1_off + i) * src1_stride_inner;
        for (uint32_t j = 0; j < d_conv; ++j) {
            src1_T[j * d_inner_stride + i] = src_row[j];
        }
    }
}

// HVX 32x32 src0 transpose: src0 {ncs, d_inner} (DDR) -> src0_T {d_inner_tile, ncs} (VTCM)
static inline void transpose_src0_block(const float * src0_block,
                                        uint32_t      ncs,
                                        uint32_t      cb_n,
                                        uint32_t      d_inner_tile,
                                        float *       src0_T_block_dst,
                                        uint32_t      cb /* dst column offset */) {
    const uint32_t T_TILE = VLEN_FP32;

    HVX_Vector __attribute__((aligned(VLEN))) sub[32];

    for (uint32_t t0 = 0; t0 < ncs; t0 += T_TILE) {
        const uint32_t t_n = MIN(T_TILE, ncs - t0);

        // Load 32 rows (channels) of T_TILE samples; pad missing channels with zeros.
        for (uint32_t r = 0; r < cb_n; ++r) {
            const float * src_row = src0_block + r * ncs + t0;
            if (t_n == T_TILE) {
                sub[r] = *(const HVX_UVector *) src_row;
            } else {
                HVX_Vector v = hvx_vec_splat_f32(0.0f);
                hvx_vec_store_u(&v, t_n * sizeof(float), hvx_vec_splat_f32(0.0f));

                float __attribute__((aligned(VLEN))) tmp[VLEN_FP32] = { 0 };
                for (uint32_t k = 0; k < t_n; ++k) tmp[k] = src_row[k];
                v = *(const HVX_Vector *) tmp;
                sub[r] = v;
            }
        }
        for (uint32_t r = cb_n; r < T_TILE; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        // Store transposed sub-tile to src0_T at offsets (t0 + j) * d_inner_tile + cb.
        // Only write the valid t_n rows of the transposed result.
        for (uint32_t r = 0; r < t_n; ++r) {
            float * dst = src0_T_block_dst + (t0 + r) * d_inner_tile + cb;
            if (cb_n == T_TILE) {
                *(HVX_UVector *) dst = sub[r];
            } else {
                hvx_vec_store_u(dst, cb_n * sizeof(float), sub[r]);
            }
        }
    }
}

static void ssm_conv_thread_f32_f32_hvx(unsigned int nth, unsigned int ith, void *data) {
    htp_ssm_conv_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];
    const uint32_t ncs     = src0->ne[0];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float);
    const uint32_t src0_stride_seq   = src0->nb[2] / sizeof(float);
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float);
    const uint32_t dst_stride_token  = dst->nb[1]  / sizeof(float);
    const uint32_t dst_stride_seq    = dst->nb[2]  / sizeof(float);

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, d_inner);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = scctx->nrows_per_thread;
    const uint32_t d_inner_tile       = scctx->d_inner_tile;

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float       * dst_data  = (float       *) dst->data;

    // Per-thread VTCM regions.
    float * src0_T = (float *)(octx->src0_spad.data + ith * octx->src0_spad.size_per_thread);
    float * src1_T = (float *)(octx->src1_spad.data + ith * octx->src1_spad.size_per_thread);

    // Stage src1 weights once into VTCM in {d_inner_stride, d_conv} layout.
    transpose_src1(src1_data, src1_stride_inner, ir0, d_inner_per_thread, d_inner_stride, d_conv, src1_T);

    const uint32_t C_TILE = VLEN_FP32;

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        for (uint32_t tile_off = 0; tile_off < d_inner_per_thread; tile_off += d_inner_tile) {
            const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

            // Place src0 chunk into VTCM in {d_inner_tile, ncs} layout.
            const float * src0_block = src0_data + i3 * src0_stride_seq + (ir0 + tile_off) * src0_stride_inner;

            for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                transpose_src0_block(src0_block + cb * src0_stride_inner, ncs, cb_n, d_inner_tile, src0_T, cb);
            }

            for (uint32_t t = 0; t < n_t; ++t) {
                for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                    const uint32_t cb_n = MIN(C_TILE, tile_n - cb);

                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    for (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_Vector x = *(const HVX_Vector *) (src0_T + (t + j) * d_inner_tile + cb);
                        HVX_Vector w = *(const HVX_Vector *) (src1_T + j * d_inner_stride + tile_off + cb);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                    HVX_Vector res = Q6_Vsf_equals_Vqf32(acc);

                    float * dst_ptr = dst_data + i3 * dst_stride_seq + t * dst_stride_token + (ir0 + tile_off + cb);
                    if (cb_n == C_TILE) {
                        *(HVX_UVector *) dst_ptr = res;
                    } else {
                        hvx_vec_store_u(dst_ptr, cb_n * sizeof(float), res);
                    }
                }
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "ssm-conv-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) tile=%u * %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1, d_inner_tile,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    htp_ssm_conv_tensors_preamble;

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "ssm_conv: only (F32 x F32 -> F32) OPs supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_ssm_conv_context scctx = { 0 };
    scctx.octx = octx;

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];  // tokens per sequence
    const uint32_t n_s     = dst->ne[2];  // number of sequences in the batch

    const uint32_t n_threads = MIN(octx->n_threads, d_inner);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t use_hvx = 0;
        if (d_inner >= VLEN_FP32 && n_t >= VLEN_FP32) {
            use_hvx = 1;
        }

        scctx.nrows_per_thread = hex_round_up((d_inner + n_threads - 1) / n_threads, VLEN_FP32);

        const uint32_t d_inner_per_thread = scctx.nrows_per_thread;
        const uint32_t ncs                = src0->ne[0];

        const uint32_t src1_T_size = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 256);
        const uint32_t src0_T_max = HTP_SSM_CONV_VTCM_BUDGET > src1_T_size ? HTP_SSM_CONV_VTCM_BUDGET - src1_T_size : 0;

        uint32_t d_inner_tile = (src0_T_max / sizeof(float)) / ncs;
        d_inner_tile -= (d_inner_tile % VLEN_FP32);
        if (d_inner_tile == 0) {
            FARF(HIGH, "ssm_conv-f32: inner tile rounds to 0 (ncs=%u), falling back to scalar\n", ncs);
            use_hvx = 0;
        } else {
            scctx.d_inner_tile = d_inner_tile;

            octx->src0_spad.size_per_thread = hex_round_up(d_inner_tile * ncs * sizeof(float), 256);
            octx->src1_spad.size_per_thread = src1_T_size;
            octx->dst_spad.size_per_thread  = 0;

            octx->src0_spad.size = octx->src0_spad.size_per_thread * n_threads;
            octx->src1_spad.size = octx->src1_spad.size_per_thread * n_threads;
            octx->dst_spad.size  = 0;

            octx->src0_spad.data = octx->ctx->vtcm_base;
            octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
            octx->src0_spad.src  = NULL;
            octx->src1_spad.src  = NULL;

            const size_t total_spad = octx->src0_spad.size + octx->src1_spad.size;
            if (total_spad > octx->ctx->vtcm_size) {
                FARF(HIGH, "ssm_conv-f32: scratchpad %zu exceeds VTCM %zu, falling back to scalar\n",
                     total_spad, octx->ctx->vtcm_size);
                use_hvx = 0;
            }
        }

        FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : use_hvx %d\n", src0->ne[0],
             src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
             dst->ne[1], dst->ne[2], dst->ne[3], use_hvx);

        if (use_hvx) {
            worker_pool_run_func(octx->ctx->worker_pool, ssm_conv_thread_f32_f32_hvx, &scctx, n_threads);
        } else {
            worker_pool_run_func(octx->ctx->worker_pool, ssm_conv_thread_f32_f32, &scctx, n_threads);
        }
    }

    return HTP_STATUS_OK;
}

int op_ssm_conv(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    int err = HTP_STATUS_OK;

    switch (dst->type) {
        case HTP_TYPE_F32:
            err = op_ssm_conv_f32(octx);
            break;
        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
