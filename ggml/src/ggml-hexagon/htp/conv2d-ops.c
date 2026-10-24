#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "conv2d-ops.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-base.h"
#include "hvx-copy.h"
#include "hmx-queue.h"
#include "hmx-utils.h"

// fp16 elements between the HMX scales of two row tiles
#define CONV2D_HMX_SCALES_N (HTP_CONV2D_HMX_SCALES_SIZE / sizeof(__fp16))

struct htp_conv2d_hmx_job {
    __fp16 *       output;
    const __fp16 * kernel_tiles;
    const __fp16 * patch_tiles;
    const __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
};

// hvx_vec_f32_to_f16_shuff() on v81 turns an exact 0 into 2^-24, which shows up as a nonzero conv output where the result is 0
static inline HVX_Vector conv2d_f32_to_f16_shuff(HVX_Vector v0, HVX_Vector v1) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector q0   = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    const HVX_Vector q1   = Q6_Vqf32_vadd_VsfVsf(v1, zero);
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
}

static inline HVX_Vector conv2d_f32_to_f16(HVX_Vector v0, HVX_Vector v1) {
    return Q6_Vh_vdeal_Vh(conv2d_f32_to_f16_shuff(v0, v1));
}

// DMA pipelines: every step records how many DMAs it pushed and pops exactly that many before its buffer is used.
// nrows = 0 pushes a dummy that completes at once. A full queue retires its oldest DMA, which only makes a later pop
// return early, since pops complete DMAs in push order.

// DMA hangs if a stride is less than the row size, even for a single row
static inline void conv2d_dma_push(dma_queue * q, dma_data d, size_t dst_stride, size_t src_stride, size_t row, size_t nrows) {
    if (nrows == 1) {
        dst_stride = src_stride = row;
    }
    while (!dma_queue_push(q, d, dst_stride, src_stride, row, nrows)) {
        dma_queue_pop(q);
    }
}

static inline void conv2d_dma_pop(dma_queue * q, uint32_t n) {
    for (; n > 0; n--) {
        dma_queue_pop(q);
    }
}

// HMX output scale of one row tile: 2^-e with 2^e >= K * max|w|, so the fp16 result |y| * 2^-e <= max|x| does not overflow
static void conv2d_hmx_rt_scale(__fp16 * scales, const __fp16 * w, uint32_t n_tiles, uint32_t K) {
    const HVX_Vector * v = (const HVX_Vector *) w;
    const uint32_t     n = n_tiles * (HMX_FP16_TILE_N_ELMS / 64);

    // |w| fp16 bits compare like unsigned ints
    HVX_Vector m = Q6_V_vzero();
    for (uint32_t i = 0; i < n; i++) {
        m = Q6_Vuh_vmax_VuhVuh(m, hvx_vec_abs_f16(v[i]));
    }
    for (uint32_t s = 2; s < VLEN; s <<= 1) {
        m = Q6_Vuh_vmax_VuhVuh(m, Q6_V_vror_VR(m, s));
    }

    union { uint16_t u; __fp16 f; } wmax = { .u = (uint16_t) Q6_R_vextract_VR(m, 0) };
    union { float f; uint32_t u; } bound = { .f = (float) K * (float) wmax.f };

    int32_t e = (int32_t) (bound.u >> 23) - 127 + ((bound.u & 0x7fffff) != 0);
    e = MAX(0, MIN(e, 24));

    // 2^-e in fp16, subnormal below 2^-14
    const uint32_t s16 = e <= 14 ? (uint32_t) (15 - e) << 10 : 1u << (24 - e);
    hmx_init_column_scales(scales, Q6_V_vsplat_R(s16));
}

static inline float conv2d_hmx_rt_restore(const __fp16 * scales, uint32_t rt) {
    return 1.0f / (float) scales[(size_t) rt * CONV2D_HMX_SCALES_N];
}

static void conv2d_hmx_worker(void * data) {
    struct htp_conv2d_hmx_job * job = (struct htp_conv2d_hmx_job *) data;

    const size_t dot_stride = job->n_dot_tiles * HMX_FP16_TILE_N_ELMS;

    for (uint32_t r = 0; r < job->n_row_tiles; ++r) {
        asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) (job->scales + r * CONV2D_HMX_SCALES_N)));

        const __fp16 * row_base = job->kernel_tiles + r * dot_stride;
        const __fp16 * col_base = job->patch_tiles;
        __fp16 * out_tile = job->output + r * job->n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (uint32_t c = 0; c < job->n_col_tiles; ++c) {
            const __fp16 * row_tiles = row_base;
            const __fp16 * col_tiles = col_base;
            asm volatile(HMX_CLRACC_F16());

            uint32_t rem = job->n_dot_tiles;
            while (rem > 0) {
                const uint32_t n_tiles = MIN(rem, 32);
                const uint32_t range = 2048u * n_tiles - 1;
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
                row_tiles += n_tiles * HMX_FP16_TILE_N_ELMS;
                col_tiles += n_tiles * HMX_FP16_TILE_N_ELMS;
                rem -= n_tiles;
            }

            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");
            col_base += dot_stride;
            out_tile += HMX_FP16_TILE_N_ELMS;
        }
    }
}

struct htp_conv2d_hmx_context {
    struct htp_ops_context *                octx;
    const struct htp_conv2d_kernel_params * kp;
    __fp16 *                                scales;  // [rt] HMX scales
    __fp16 *                                w_tiles;
    __fp16 *                                act[2];
    __fp16 *                                out[2];
    uint8_t *                               thr_base;
    uint32_t                                t_base;
    uint32_t                                t_end;
    uint32_t                                rt0;
    uint32_t                                n_rt;
    int32_t                                 pack_chunk;
    int32_t                                 unpack_chunk;
    uint32_t                                pack_tpt;
    uint32_t                                unpack_tpt;
};

// col tiles [t0, t1) of chunk c for thread ith, returns first col tile of the chunk
static inline uint32_t conv2d_hmx_chunk_range(const struct htp_conv2d_hmx_context * hc, uint32_t c, uint32_t tpt,
                                              uint32_t ith, uint32_t * t0, uint32_t * t1) {
    const uint32_t cs = hc->t_base + c * hc->kp->ct_chunk;
    const uint32_t ce = MIN(cs + hc->kp->ct_chunk, hc->t_end);
    *t0 = MIN(cs + ith * tpt, ce);
    *t1 = MIN(*t0 + tpt, ce);
    return cs;
}

static void conv2d_hmx_wpack_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               kernel = octx->src[0];

    const uint32_t Cout   = kernel->ne[3];
    const uint32_t K      = kp->tiled.k_size;
    const uint32_t k_pad  = kp->tiled.n_k_tiles * HMX_FP16_TILE_N_COLS;
    const uint32_t WK     = kp->tiled.wk_chunk;
    const bool     is_f16 = kernel->type == HTP_TYPE_F16;
    const uint32_t es     = is_f16 ? sizeof(__fp16) : sizeof(float);

    dma_queue * dmaq = octx->ctx->dma[ith];
    uint8_t *   thr  = hc->thr_base + (size_t) ith * kp->thr_size;
    __fp16 *    wbuf = (__fp16 *) thr;
    float *     wraw = (float *) (thr + (size_t) WK * 32 * sizeof(__fp16));
    uint8_t *   wdma = is_f16 ? (uint8_t *) wbuf : (uint8_t *) wraw;

    const HVX_Vector zero = Q6_V_vzero();

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t rt = ith; rt < hc->n_rt; rt += nth) {
        const uint32_t co0 = (hc->rt0 + rt) * 32;
        const uint32_t nco = MIN(32, Cout - co0);

        for (uint32_t kc = 0; kc < k_pad; kc += WK) {
            const uint32_t kv = MIN(WK, K - kc);
            const uint32_t hd = MIN(WK, k_pad - kc);

            conv2d_dma_push(dmaq, dma_make_data(wdma, kernel->data + (size_t) co0 * kernel->nb[3] + (size_t) kc * es),
                           WK * es, kernel->nb[3], kv * es, nco);
            conv2d_dma_pop(dmaq, 1);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) kc);

            for (uint32_t r = 0; r < 32; r++) {
                HVX_Vector *       dv = (HVX_Vector *) (wbuf + (size_t) r * WK);
                const HVX_Vector * sv = (const HVX_Vector *) (wraw + (size_t) r * WK);
                for (uint32_t j = 0; j < hd; j += 64) {
                    HVX_Vector v = zero;
                    if (r < nco && j < kv) {
                        v = is_f16 ? dv[j / 64] : conv2d_f32_to_f16(sv[j / 32], sv[j / 32 + 1]);
                        if (kv - j < 64) {
                            v = Q6_V_vmux_QVV(Q6_Q_vsetq_R((kv - j) * sizeof(__fp16)), v, zero);
                        }
                    }
                    dv[j / 64] = v;
                }
            }

            __fp16 * tiles = hc->w_tiles + ((size_t) rt * kp->tiled.n_k_tiles + kc / 32) * HMX_FP16_TILE_N_ELMS;
            hmx_interleave_cols_to_tiles(tiles, wbuf, 32, hd, WK, 1, 0, 32);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) kc);
        }
    }
}

// Stage Cin*KH input rows for G col tiles: stage[(cin*KH + kh)*SR + j] = x[n][cin][ih(kh)][ow0 - p0 + j], zero outside.
// Returns the number of DMAs pushed.
static uint32_t conv2d_hmx_stage(const struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp,
                             dma_queue * dmaq, float * stage, uint32_t n, uint32_t oh, uint32_t ow0, uint32_t G) {
    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];

    const uint32_t KW  = kernel->ne[0];
    const uint32_t KH  = kernel->ne[1];
    const uint32_t Cin = kernel->ne[2];
    const int32_t  IW  = input->ne[0];
    const int32_t  IH  = input->ne[1];
    const int32_t  s1  = octx->op_params[1];
    const int32_t  p0  = octx->op_params[2];
    const int32_t  p1  = octx->op_params[3];
    const int32_t  d0  = octx->op_params[4];
    const int32_t  d1  = octx->op_params[5];
    const uint32_t SR  = kp->tiled.stage_row;

    const int32_t Wb  = 32 * G + (KW - 1) * d0;
    const int32_t iw0 = (int32_t) ow0 - p0;
    const int32_t lo  = MAX(0, -iw0);
    const int32_t hi  = MIN(Wb, IW - iw0);

    uint32_t nd = 0;
    for (uint32_t kh = 0; kh < KH; kh++) {
        const int32_t ih   = (int32_t) oh * s1 + (int32_t) kh * d1 - p1;
        float *       rows = stage + (size_t) kh * SR;

        if (ih < 0 || ih >= IH || hi <= lo) {
            for (uint32_t c = 0; c < Cin; c++) {
                hvx_splat_f32_u(rows + (size_t) c * KH * SR, 0.0f, Wb);
            }
            continue;
        }

        const dma_addr_t src = input->data + (size_t) n * input->nb[3] + (size_t) ih * input->nb[1] + (size_t) (iw0 + lo) * sizeof(float);
        conv2d_dma_push(dmaq, dma_make_data(rows + lo, src), (size_t) KH * SR * sizeof(float), input->nb[2],
                        (size_t) (hi - lo) * sizeof(float), Cin);
        nd++;

        if (lo > 0 || hi < Wb) {
            for (uint32_t c = 0; c < Cin; c++) {
                float * row = rows + (size_t) c * KH * SR;
                if (lo > 0) {
                    hvx_splat_f32_u(row, 0.0f, lo);
                }
                if (hi < Wb) {
                    hvx_splat_f32_u(row + hi, 0.0f, Wb - hi);
                }
            }
        }
    }
    return nd;
}

// One weight-side tile column: vector i holds k rows 2i and 2i+1 interleaved over 32 pixels
static inline void conv2d_hmx_pack_tile(__fp16 * restrict dst, const float * restrict src, uint32_t K, uint32_t KW,
                                        uint32_t d0, uint32_t SR, uint32_t n_vec) {
    HVX_Vector * restrict out  = (HVX_Vector *) dst;
    const HVX_Vector      zero = Q6_V_vzero();
    const uint32_t        wrap = SR - KW * d0;

    const float * p  = src;
    uint32_t      kw = 0;
    uint32_t      i  = 0;

    for (; i < K / 2; i++) {
        const HVX_Vector a = hvx_vmemu(p);
        p += d0;
        if (++kw == KW) {
            kw = 0;
            p += wrap;
        }
        const HVX_Vector b = hvx_vmemu(p);
        p += d0;
        if (++kw == KW) {
            kw = 0;
            p += wrap;
        }
        out[i] = conv2d_f32_to_f16_shuff(a, b);
    }
    if (K & 1) {
        out[i++] = conv2d_f32_to_f16_shuff(hvx_vmemu(p), zero);
    }
    for (; i < n_vec; i++) {
        out[i] = zero;
    }
}

// CWHN: stage[kh*SR + j*Cin + cin] = x[n][ih(kh)][ow0 - p0 + j][cin], zero outside. Returns the number of DMAs pushed.
static uint32_t conv2d_hmx_stage_cwhn(const struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp,
                                  dma_queue * dmaq, float * stage, uint32_t n, uint32_t oh, uint32_t ow0, uint32_t G) {
    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];

    const uint32_t KW  = kernel->ne[0];
    const uint32_t KH  = kernel->ne[1];
    const uint32_t Cin = kernel->ne[2];
    const int32_t  IW  = input->ne[0];
    const int32_t  IH  = input->ne[1];
    const int32_t  s1  = octx->op_params[1];
    const int32_t  p0  = octx->op_params[2];
    const int32_t  p1  = octx->op_params[3];
    const int32_t  d0  = octx->op_params[4];
    const int32_t  d1  = octx->op_params[5];
    const uint32_t SR  = kp->tiled.stage_row;

    const int32_t Wb  = 32 * G + (KW - 1) * d0;
    const int32_t iw0 = (int32_t) ow0 - p0;
    const int32_t lo  = MAX(0, -iw0);
    const int32_t hi  = MIN(Wb, IW - iw0);

    uint32_t nd = 0;
    for (uint32_t kh = 0; kh < KH; kh++) {
        const int32_t ih  = (int32_t) oh * s1 + (int32_t) kh * d1 - p1;
        float *       row = stage + (size_t) kh * SR;

        if (ih < 0 || ih >= IH || hi <= lo) {
            hvx_splat_f32_u(row, 0.0f, Wb * Cin);
            continue;
        }

        const dma_addr_t src  = input->data + (size_t) n * input->nb[3] + (size_t) ih * input->nb[1] + (size_t) (iw0 + lo) * input->nb[0];
        const size_t     size = (size_t) (hi - lo) * Cin * sizeof(float);
        conv2d_dma_push(dmaq, dma_make_data(row + lo * Cin, src), size, size, size, 1);
        nd++;

        if (lo > 0) {
            hvx_splat_f32_u(row, 0.0f, lo * Cin);
        }
        if (hi < Wb) {
            hvx_splat_f32_u(row + hi * Cin, 0.0f, (Wb - hi) * Cin);
        }
    }
    return nd;
}

static void conv2d_hmx_cwhn_f16(__fp16 * restrict dst, const float * restrict src, uint32_t KH, uint32_t SR, uint32_t n) {
    for (uint32_t kh = 0; kh < KH; kh++) {
        hvx_copy_f16_f32_aa((uint8_t *) (dst + (size_t) kh * SR), (const uint8_t *) (src + (size_t) kh * SR), n);
    }
}

// Stage the segment of col tiles starting at t, returns its number of col tiles and DMAs (*nd)
static uint32_t conv2d_hmx_issue_seg(const struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp,
                                     dma_queue * dmaq, float * stage, uint32_t t, uint32_t t1, uint32_t * nd) {
    const uint32_t OH    = octx->dst->ne[1];
    const uint32_t n_owt = kp->tiled.n_ow_tiles;
    const uint32_t row   = fastdiv(t, &kp->tiled.div_ow_tiles);
    const uint32_t owt   = t - row * n_owt;
    const uint32_t G     = MIN(MIN(kp->tiled.seg_tiles, t1 - t), n_owt - owt);
    const uint32_t n     = fastdiv(row, &kp->tiled.div_oh);
    const uint32_t oh    = row - n * OH;

    *nd = kp->cwhn ? conv2d_hmx_stage_cwhn(octx, kp, dmaq, stage, n, oh, owt * 32, G) :
                     conv2d_hmx_stage(octx, kp, dmaq, stage, n, oh, owt * 32, G);
    return G;
}

static inline void conv2d_vshuff_w(HVX_Vector * a, HVX_Vector * b) {
    const HVX_VectorPair p = Q6_W_vshuff_VVR(*b, *a, -4);
    *a = Q6_V_lo_W(p);
    *b = Q6_V_hi_W(p);
}

static inline void conv2d_transpose_finish(HVX_Vector * restrict scratch, HVX_Vector * restrict lo, HVX_Vector * restrict hi);

// 32x32 word transpose: out[i] word j = in[j * in_stride] word i; out 0..15 -> lo, 16..31 -> hi
static inline void conv2d_transpose_w32(const HVX_Vector * restrict in, uint32_t in_stride, HVX_Vector * restrict scratch,
                                        HVX_Vector * restrict lo, HVX_Vector * restrict hi) {
    for (uint32_t i = 0; i < 8; i++) {
        HVX_Vector v0 = in[(i +  0) * in_stride];
        HVX_Vector v1 = in[(i +  8) * in_stride];
        HVX_Vector v2 = in[(i + 16) * in_stride];
        HVX_Vector v3 = in[(i + 24) * in_stride];
        conv2d_vshuff_w(&v0, &v2);
        conv2d_vshuff_w(&v1, &v3);
        conv2d_vshuff_w(&v0, &v1);
        conv2d_vshuff_w(&v2, &v3);
        scratch[i +  0] = v0;
        scratch[i +  8] = v1;
        scratch[i + 16] = v2;
        scratch[i + 24] = v3;
    }
    conv2d_transpose_finish(scratch, lo, hi);
}

static inline void conv2d_transpose_finish(HVX_Vector * restrict scratch, HVX_Vector * restrict lo, HVX_Vector * restrict hi) {
    for (uint32_t g = 0; g < 4; g++) {
        HVX_Vector w0 = scratch[8 * g + 0];
        HVX_Vector w1 = scratch[8 * g + 1];
        HVX_Vector w2 = scratch[8 * g + 2];
        HVX_Vector w3 = scratch[8 * g + 3];
        HVX_Vector w4 = scratch[8 * g + 4];
        HVX_Vector w5 = scratch[8 * g + 5];
        HVX_Vector w6 = scratch[8 * g + 6];
        HVX_Vector w7 = scratch[8 * g + 7];
        conv2d_vshuff_w(&w0, &w4);
        conv2d_vshuff_w(&w1, &w5);
        conv2d_vshuff_w(&w2, &w6);
        conv2d_vshuff_w(&w3, &w7);
        conv2d_vshuff_w(&w0, &w2);
        conv2d_vshuff_w(&w1, &w3);
        conv2d_vshuff_w(&w4, &w6);
        conv2d_vshuff_w(&w5, &w7);
        conv2d_vshuff_w(&w0, &w1);
        conv2d_vshuff_w(&w2, &w3);
        conv2d_vshuff_w(&w4, &w5);
        conv2d_vshuff_w(&w6, &w7);
        HVX_Vector * o = (g < 2 ? lo : hi) + 8 * (g & 1);
        o[0] = w0;
        o[1] = w1;
        o[2] = w2;
        o[3] = w3;
        o[4] = w4;
        o[5] = w5;
        o[6] = w6;
        o[7] = w7;
    }
}

// Same transpose, but pixel j is base + j * stride. Skips the vin store/reload.
static inline void conv2d_transpose_strided(const __fp16 * restrict base, uint32_t stride, HVX_Vector * restrict scratch,
                                            HVX_Vector * restrict lo, HVX_Vector * restrict hi) {
    for (uint32_t i = 0; i < 8; i++) {
        HVX_Vector v0 = hvx_vmemu(base + (i +  0) * stride);
        HVX_Vector v1 = hvx_vmemu(base + (i +  8) * stride);
        HVX_Vector v2 = hvx_vmemu(base + (i + 16) * stride);
        HVX_Vector v3 = hvx_vmemu(base + (i + 24) * stride);
        conv2d_vshuff_w(&v0, &v2);
        conv2d_vshuff_w(&v1, &v3);
        conv2d_vshuff_w(&v0, &v1);
        conv2d_vshuff_w(&v2, &v3);
        scratch[i +  0] = v0;
        scratch[i +  8] = v1;
        scratch[i + 16] = v2;
        scratch[i + 24] = v3;
    }
    conv2d_transpose_finish(scratch, lo, hi);
}

// CWHN: k is n_groups groups of k_group contiguous values (per kh, or per (kh, kw) if d0 > 1), each padded to k_group_pad.
// A full 64-wide block is one vector per pixel and is transposed in place. A partial block is two masked loads.
static void conv2d_hmx_pack_tile_cwhn(__fp16 * restrict dst, const __fp16 * restrict src, HVX_Vector * restrict vin,
                                      HVX_Vector * restrict scratch, const struct htp_conv2d_kernel_params * kp,
                                      uint32_t gpr, uint32_t Cin, uint32_t d0) {
    const uint32_t k_pad = kp->tiled.n_k_tiles * HMX_FP16_TILE_N_COLS;
    const uint32_t KG    = kp->tiled.k_group;
    const uint32_t KGp   = kp->tiled.k_group_pad;
    const uint32_t NG    = kp->tiled.n_groups;
    const uint32_t SR    = kp->tiled.stage_row;

    HVX_Vector * restrict out  = (HVX_Vector *) dst;
    const HVX_VectorPred  q64  = Q6_Q_vsetq_R(64);

    uint32_t g    = 0;
    uint32_t gw   = 0;
    uint32_t r    = 0;
    size_t   roff = 0;
    size_t   goff = 0;

    for (uint32_t b = 0; b < k_pad / 64; b++) {
        const __fp16 * pa[2];
        uint32_t       va[2];
        for (uint32_t h = 0; h < 2; h++) {
            va[h] = g < NG && r < KG ? MIN(32, KG - r) : 0;
            pa[h] = g < NG ? src + goff + r - 32 * h : src;
            r += 32;
            if (r >= KGp) {
                r = 0;
                g++;
                if (++gw == gpr) {
                    gw   = 0;
                    roff += SR;
                    goff = roff;
                } else {
                    goff += d0 * Cin;
                }
            }
        }

        if (va[0] == 32 && va[1] == 32 && pa[0] == pa[1]) {
            conv2d_transpose_strided(pa[0], Cin, scratch, out + 32 * b, out + 32 * b + 16);
            continue;
        } else {
            const HVX_VectorPred q0 = Q6_Q_vsetq_R(va[0] * sizeof(__fp16));
            const HVX_VectorPred q1 = Q6_Q_and_QQn(Q6_Q_vsetq2_R(64 + va[1] * sizeof(__fp16)), q64);
            for (uint32_t j = 0; j < 32; j++) {
                const HVX_Vector v0 = hvx_vmemu(pa[0] + j * Cin);
                const HVX_Vector v1 = hvx_vmemu(pa[1] + j * Cin);
                vin[j] = Q6_V_vmux_QVV(q0, v0, Q6_V_vand_QV(q1, v1));
            }
        }

        conv2d_transpose_w32(vin, 1, scratch, out + 32 * b, out + 32 * b + 16);
    }
}

// Kernel stored as [k][cout] (cout contiguous): DMA k rows of 64 cout, transpose into two row tiles
static void conv2d_hmx_wpack_kmajor_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               kernel = octx->src[0];

    const uint32_t Cout   = kernel->ne[3];
    const uint32_t k_pad  = kp->tiled.n_k_tiles * HMX_FP16_TILE_N_COLS;
    const uint32_t WK     = kp->tiled.wk_chunk;
    const uint32_t KG     = kp->tiled.k_group;
    const uint32_t KGp    = kp->tiled.k_group_pad;
    const uint32_t NG     = kp->tiled.n_groups;
    const size_t   ks     = kp->tiled.w_kstride;
    const bool     is_f16 = kernel->type == HTP_TYPE_F16;
    const uint32_t es     = is_f16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t n_cb   = (hc->n_rt + 1) / 2;

    dma_queue *  dmaq    = octx->ctx->dma[ith];
    uint8_t *    wraw    = hc->thr_base + (size_t) ith * kp->thr_size;
    HVX_Vector * vin     = (HVX_Vector *) (wraw + (size_t) WK * 64 * sizeof(float));
    HVX_Vector * scratch = vin + 32;
    HVX_Vector * sink    = scratch + 32;

    const HVX_Vector zero = Q6_V_vzero();

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    uint32_t owner = 0;
    for (uint32_t cb = 0; cb < n_cb; cb++) {
        for (uint32_t g = 0; g < NG; g++) {
            const uint32_t span = g + 1 == NG ? k_pad - g * KGp : KGp;
            for (uint32_t kc = 0; kc < span; kc += WK) {
                const bool mine = owner == ith;
                owner = owner + 1 == nth ? 0 : owner + 1;
                if (!mine) {
                    continue;
                }

                const uint32_t rt  = 2 * cb;
                const uint32_t co0 = (hc->rt0 + rt) * 32;
                const uint32_t nco = MIN(64, Cout - co0);
                const uint32_t nk  = kc < KG ? MIN(WK, KG - kc) : 0;
                const uint32_t hk  = MIN(WK, span - kc);

                if (nk > 0) {
                    conv2d_dma_push(dmaq, dma_make_data(wraw, kernel->data + ((size_t) g * KG + kc) * ks + (size_t) co0 * es),
                                    64 * es, ks, nco * es, nk);
                    conv2d_dma_pop(dmaq, 1);
                }

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) kc);

                for (uint32_t kt = 0; kt < hk / 32; kt++) {
                    for (uint32_t j = 0; j < 32; j++) {
                        const uint32_t k = kt * 32 + j;
                        HVX_Vector     v = zero;
                        if (k < nk) {
                            const HVX_Vector * sv = (const HVX_Vector *) (wraw + (size_t) k * 64 * es);
                            v = is_f16 ? sv[0] : conv2d_f32_to_f16(sv[0], sv[1]);
                            if (nco < 64) {
                                v = Q6_V_vmux_QVV(Q6_Q_vsetq_R(nco * sizeof(__fp16)), v, zero);
                            }
                        }
                        vin[j] = v;
                    }

                    const uint32_t ktg = (g * KGp + kc) / 32 + kt;
                    HVX_Vector *   lo  = (HVX_Vector *) (hc->w_tiles + ((size_t) rt * kp->tiled.n_k_tiles + ktg) * HMX_FP16_TILE_N_ELMS);
                    HVX_Vector *   hi  = rt + 1 < hc->n_rt ?
                        (HVX_Vector *) (hc->w_tiles + ((size_t) (rt + 1) * kp->tiled.n_k_tiles + ktg) * HMX_FP16_TILE_N_ELMS) : sink;
                    conv2d_transpose_w32(vin, 1, scratch, lo, hi);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) kc);
            }
        }
    }
}

static void conv2d_hmx_pack(const struct htp_conv2d_hmx_context * hc, uint32_t ith) {
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               kernel = octx->src[0];

    const uint32_t KW    = kernel->ne[0];
    const uint32_t KH    = kernel->ne[1];
    const uint32_t Cin   = kernel->ne[2];
    const uint32_t d0    = octx->op_params[4];
    const uint32_t n_vec = kp->tiled.n_k_tiles * (HMX_FP16_TILE_N_ELMS / 64);

    dma_queue * dmaq  = octx->ctx->dma[ith];
    uint8_t *   thr   = hc->thr_base + (size_t) ith * kp->thr_size;
    float *     stages[2] = {
        (float *) thr,
        (float *) (thr + kp->stage_size),
    };
    __fp16 *    act   = hc->act[hc->pack_chunk & 1];

    // CWHN: fp16 stage (+256 bytes of over-read slack), then 32 + 32 vectors for the transpose
    const uint32_t n_stage = kp->tiled.n_stage > 1 ? 2 : 1;
    __fp16 *     stage16 = (__fp16 *) (thr + (size_t) n_stage * kp->stage_size + (size_t) 2 * 32 * kp->tiled.seg_tiles * 32 * sizeof(float));
    HVX_Vector * vin     = (HVX_Vector *) ((uint8_t *) stage16 + kp->stage_size / 2 + 256);
    HVX_Vector * scratch = vin + 32;
    const uint32_t gpr   = kp->tiled.n_groups == KH ? 1 : KW;

    uint32_t       t0, t1;
    const uint32_t cs = conv2d_hmx_chunk_range(hc, hc->pack_chunk, hc->pack_tpt, ith, &t0, &t1);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // push(N), push(N + 1), pop(N), pack(N), push(N + 2), ... with n_stage 2, push(N), pop(N), pack(N), ... with 1.
    // The first pop also retires the output DMAs left in flight by the unpack of this phase.
    uint32_t nd[2] = { dma_queue_depth(dmaq), 0 };
    uint32_t ng[2] = { 0, 0 };
    uint32_t tq    = t0;
    for (uint32_t i = 0; i < n_stage && tq < t1; i++) {
        uint32_t n;
        ng[i]  = conv2d_hmx_issue_seg(octx, kp, dmaq, stages[i], tq, t1, &n);
        nd[i] += n;
        tq    += ng[i];
    }

    uint32_t bi = 0;
    for (uint32_t t = t0; t < t1;) {
        const uint32_t G     = ng[bi];
        float *        stage = stages[bi];

        conv2d_dma_pop(dmaq, nd[bi]);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) t);
        if (kp->cwhn) {
            const uint32_t Wb = 32 * G + (KW - 1) * d0;
            conv2d_hmx_cwhn_f16(stage16, stage, KH, kp->tiled.stage_row, Wb * Cin);
            for (uint32_t g = 0; g < G; g++) {
                __fp16 * tile = act + (size_t) (t - cs + g) * n_vec * 64;
                conv2d_hmx_pack_tile_cwhn(tile, stage16 + (size_t) g * 32 * Cin, vin, scratch, kp, gpr, Cin, d0);
            }
        } else {
            for (uint32_t g = 0; g < G; g++) {
                __fp16 * tile = act + (size_t) (t - cs + g) * n_vec * 64;
                conv2d_hmx_pack_tile(tile, stage + g * 32, kp->tiled.k_size, KW, d0, kp->tiled.stage_row, n_vec);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) t);

        nd[bi] = 0;
        if (tq < t1) {
            ng[bi]  = conv2d_hmx_issue_seg(octx, kp, dmaq, stage, tq, t1, &nd[bi]);
            tq     += ng[bi];
        }
        t += G;
        bi = bi + 1 == n_stage ? 0 : bi + 1;
    }
}

static void conv2d_hmx_unpack(const struct htp_conv2d_hmx_context * hc, uint32_t ith) {
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               dst  = octx->dst;

    const uint32_t OW    = dst->ne[0];
    const uint32_t OH    = dst->ne[1];
    const uint32_t Cout  = dst->ne[2];
    const uint32_t n_owt = kp->tiled.n_ow_tiles;
    const uint32_t OS    = kp->tiled.seg_tiles * 32;

    dma_queue * dmaq      = octx->ctx->dma[ith];
    uint8_t *   thr       = hc->thr_base + (size_t) ith * kp->thr_size + (size_t) (kp->tiled.n_stage > 1 ? 2 : 1) * kp->stage_size;
    float *     ostage[2] = {
        (float *) thr,
        (float *) (thr + (size_t) 32 * OS * sizeof(float)),
    };
    const __fp16 * out = hc->out[hc->unpack_chunk & 1];

    uint32_t       t0, t1;
    const uint32_t cs  = conv2d_hmx_chunk_range(hc, hc->unpack_chunk, hc->unpack_tpt, ith, &t0, &t1);
    const uint32_t nct = MIN(cs + kp->ct_chunk, hc->t_end) - cs;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // one output DMA per step: dummies start the pipeline, then pop(N - 2), unpack(N), push(N)
    conv2d_dma_push(dmaq, dma_make_data(ostage[0], ostage[0]), 0, 0, 0, 0);
    conv2d_dma_push(dmaq, dma_make_data(ostage[1], ostage[1]), 0, 0, 0, 0);
    uint32_t b = 0;

    for (uint32_t t = t0; t < t1;) {
        const uint32_t row = fastdiv(t, &kp->tiled.div_ow_tiles);
        const uint32_t owt = t - row * n_owt;
        const uint32_t G   = MIN(MIN(kp->tiled.seg_tiles, t1 - t), n_owt - owt);
        const uint32_t n   = fastdiv(row, &kp->tiled.div_oh);
        const uint32_t oh  = row - n * OH;
        const uint32_t ow0 = owt * 32;
        const uint32_t nv  = MIN(G * 32, OW - ow0);

        for (uint32_t rt = 0; rt < hc->n_rt; rt++) {
            const uint32_t co0 = (hc->rt0 + rt) * 32;
            const uint32_t nco = MIN(32, Cout - co0);
            const HVX_Vector restore = hvx_vec_splat_f32(conv2d_hmx_rt_restore(hc->scales, rt));

            conv2d_dma_pop(dmaq, 1);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) t);
            float * ob = ostage[b];
            for (uint32_t g = 0; g < G; g++) {
                const HVX_Vector * tv = (const HVX_Vector *) (out + ((size_t) rt * nct + (t - cs + g)) * HMX_FP16_TILE_N_ELMS);
                for (uint32_t r = 0; r < 16; r++) {
                    const HVX_VectorPair pr = hvx_vec_f16_to_f32_shuff(tv[r]);
                    *(HVX_Vector *) (ob + (size_t) (2 * r + 0) * OS + g * 32) = hvx_vec_mul_f32_f32(Q6_V_lo_W(pr), restore);
                    *(HVX_Vector *) (ob + (size_t) (2 * r + 1) * OS + g * 32) = hvx_vec_mul_f32_f32(Q6_V_hi_W(pr), restore);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) t);

            const dma_addr_t ddr = dst->data + (size_t) n * dst->nb[3] + (size_t) oh * dst->nb[1] + (size_t) co0 * dst->nb[2] + (size_t) ow0 * sizeof(float);
            conv2d_dma_push(dmaq, dma_make_data(ddr, ob), dst->nb[2], (size_t) OS * sizeof(float), (size_t) nv * sizeof(float), nco);
            b ^= 1;
        }
        t += G;
    }
}

static void conv2d_hmx_phase_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;

    if (hc->unpack_chunk >= 0) {
        conv2d_hmx_unpack(hc, ith);
    }
    // weights of this rt block are packed before its first chunk
    if (hc->pack_chunk == 0) {
        const uint32_t n_kt = hc->kp->tiled.n_k_tiles;
        for (uint32_t rt = ith; rt < hc->n_rt; rt += nth) {
            conv2d_hmx_rt_scale(hc->scales + rt * CONV2D_HMX_SCALES_N, hc->w_tiles + (size_t) rt * n_kt * HMX_FP16_TILE_N_ELMS, n_kt,
                                hc->kp->tiled.k_size);
        }
    }
    if (hc->pack_chunk >= 0) {
        conv2d_hmx_pack(hc, ith);
    }
    dma_queue_flush(hc->octx->ctx->dma[ith]);
}

static inline uint32_t conv2d_hmx_tpt(const struct htp_conv2d_hmx_context * hc, int32_t c, uint32_t nth) {
    if (c < 0) {
        return 0;
    }
    const uint32_t cs  = hc->t_base + (uint32_t) c * hc->kp->ct_chunk;
    const uint32_t nct = MIN(hc->kp->ct_chunk, hc->t_end - cs);
    return (nct + nth - 1) / nth;
}

static void conv2d_hmx_run_phase(struct htp_conv2d_hmx_context * hc, int32_t pack, int32_t unpack, uint32_t nth) {
    hc->pack_chunk   = pack;
    hc->unpack_chunk = unpack;
    hc->pack_tpt     = conv2d_hmx_tpt(hc, pack, nth);
    hc->unpack_tpt   = conv2d_hmx_tpt(hc, unpack, nth);
    work_queue_run(hc->octx->ctx->work_queue, conv2d_hmx_phase_thread, hc, nth);
}

static int conv2d_hmx_tiled(struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp) {
    if (!octx->ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (kp->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const struct htp_tensor * dst = octx->dst;
    const uint32_t OH = dst->ne[1];
    const uint32_t N  = dst->ne[3];

    uint32_t row_base = 0;
    uint32_t n_rows   = N * OH;
    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        if (htp_tensor_mdev_data_aligned(dst) && (dst->nb[2] % HTP_TENSOR_MDEV_LINE_SIZE) == 0) {
            rows_per_chunk = HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(dst->nb[1], HTP_TENSOR_MDEV_LINE_SIZE);
            if (N > 1 && OH % rows_per_chunk != 0) {
                rows_per_chunk = 0;
            }
        }
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            n_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        n_rows   = range.count;
    }
    if (n_rows == 0) {
        return HTP_STATUS_OK;
    }

    uint8_t * base = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_conv2d_hmx_context hc = {
        .octx     = octx,
        .kp       = kp,
        .scales   = (__fp16 *) base,
        .w_tiles  = VTCM_LAYOUT_PTR(__fp16, base, kp->off_w),
        .act      = { VTCM_LAYOUT_PTR(__fp16, base, kp->off_act), VTCM_LAYOUT_PTR(__fp16, base, kp->off_act + kp->act_size) },
        .out      = { VTCM_LAYOUT_PTR(__fp16, base, kp->off_out), VTCM_LAYOUT_PTR(__fp16, base, kp->off_out + kp->out_size) },
        .thr_base = VTCM_LAYOUT_PTR(uint8_t, base, kp->off_thr),
        .t_base   = row_base * kp->tiled.n_ow_tiles,
        .t_end    = (row_base + n_rows) * kp->tiled.n_ow_tiles,
    };

    const uint32_t nth      = kp->n_threads;
    const int32_t  n_chunks = (int32_t) ((hc.t_end - hc.t_base + kp->ct_chunk - 1) / kp->ct_chunk);

    struct htp_conv2d_hmx_job job;

    for (uint32_t rt0 = 0; rt0 < kp->n_row_tiles; rt0 += kp->rt_block) {
        hc.rt0  = rt0;
        hc.n_rt = MIN(kp->rt_block, kp->n_row_tiles - rt0);
        work_queue_run(octx->ctx->work_queue, kp->w_kmajor ? conv2d_hmx_wpack_kmajor_thread : conv2d_hmx_wpack_thread, &hc, nth);

        conv2d_hmx_run_phase(&hc, 0, -1, nth);

        for (int32_t c = 0; c < n_chunks; c++) {
            const uint32_t cs = hc.t_base + (uint32_t) c * kp->ct_chunk;

            job.output       = hc.out[c & 1];
            job.kernel_tiles = hc.w_tiles;
            job.patch_tiles  = hc.act[c & 1];
            job.scales       = hc.scales;
            job.n_row_tiles  = hc.n_rt;
            job.n_col_tiles  = MIN(kp->ct_chunk, hc.t_end - cs);
            job.n_dot_tiles  = kp->tiled.n_k_tiles;
            hmx_queue_push(octx->ctx->hmx_queue, hmx_queue_make_desc(conv2d_hmx_worker, &job));

            const int32_t pack   = c + 1 < n_chunks ? c + 1 : -1;
            const int32_t unpack = c - 1;
            if (pack >= 0 || unpack >= 0) {
                conv2d_hmx_run_phase(&hc, pack, unpack, nth);
            }

            hmx_queue_pop(octx->ctx->hmx_queue);
        }

        conv2d_hmx_run_phase(&hc, -1, n_chunks - 1, nth);
    }

    return HTP_STATUS_OK;
}

// ---- HMX_WIN: implicit GEMM over sliding crouton windows ----

#define CONV2D_WIN_RT_FULL 0x7FE // spatial mask 11111, channel stop 31
#define CONV2D_WIN_SCAT_MAX 4    // scatter croutons for Cin <= 2 * CONV2D_WIN_SCAT_MAX

// whcn small Cin: crouton byte offset of word j of a [c | c + 1] pixel pair vector
static const int32_t conv2d_hmx_win_scat_off[32] __attribute__((aligned(VLEN))) = {
    0 * 128, 1 * 128, 2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,
    8 * 128, 9 * 128, 10 * 128, 11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128,
    0 * 128 + 4, 1 * 128 + 4, 2 * 128 + 4,  3 * 128 + 4,  4 * 128 + 4,  5 * 128 + 4,  6 * 128 + 4,  7 * 128 + 4,
    8 * 128 + 4, 9 * 128 + 4, 10 * 128 + 4, 11 * 128 + 4, 12 * 128 + 4, 13 * 128 + 4, 14 * 128 + 4, 15 * 128 + 4,
};

struct htp_conv2d_win_context {
    struct htp_ops_context *             octx;
    const struct htp_conv2d_kernel_params * kp;
    __fp16 *                             scales;  // [rt] HMX scales
    __fp16 *                             w_tiles; // [rt][tap][cc] weight tiles
    __fp16 *                             act[2];  // [cc][b] croutons of a chunk
    __fp16 *                             out[2];  // [rt][tile] output tiles of a chunk
    uint8_t *                            thr_base;
    uint32_t                             t_base;  // device col tile range
    uint32_t                             t_end;
    uint32_t                             vr0;     // device virtual output row range
    uint32_t                             vr1;
    uint32_t                             rt0;
    uint32_t                             n_rt;
    int32_t                              pack_chunk;
    int32_t                              unpack_chunk;
    uint32_t                             n_scat;  // small Cin: scatters per crouton into zeroed croutons, else 0
    uint8_t *                            live[2]; // per chunk: tiles with an output row to write
};

struct htp_conv2d_win_job {
    __fp16 *         out;
    const __fp16 *   w;
    const __fp16 *   act;
    const __fp16 *   scales;
    const uint32_t * taps;       // activation address offset of each tap (crouton + window offset fields)
    uint32_t         n_taps;
    uint32_t         n_rt;
    uint32_t         n_tiles;
    uint32_t         out_stride; // elements between row tiles in out
    uint32_t         n_cc;
    uint32_t         plane;      // bytes between channel chunks in act
    uint32_t         rt_last;
    uint32_t         wr_last;
    const uint8_t *  live;       // tiles with at least one output row to write
};

// Activation address offset of input pixel offset u: crouton u / 32, window offset u % 32 in Rs[10:7] and Rs[1]
static inline uint32_t conv2d_hmx_win_act_off(uint32_t u) {
    return ((u >> 5) << 11) | ((u & 30) << 6) | ((u & 1) << 1);
}

static void conv2d_hmx_win_worker(void * data) {
    const struct htp_conv2d_win_job * j = (const struct htp_conv2d_win_job *) data;

    const uint32_t * taps    = j->taps;
    const uint32_t   T       = j->n_taps;
    const uint32_t   n_cc    = j->n_cc;
    const uint32_t   plane   = j->plane;
    const uint32_t   rt_last = HMX_FP16_TILE_SIZE | j->rt_last;
    const uint32_t   wr_last = j->wr_last;
    const uint32_t   n_tiles = j->n_tiles;
    const uint32_t   n_rt    = j->n_rt;
    const uint32_t   ostride = j->out_stride;
    const uint32_t   act     = (uint32_t) j->act;
    const size_t     wstride = (size_t) T * n_cc * HMX_FP16_TILE_N_ELMS;

    for (uint32_t r = 0; r < n_rt; r++) {
        asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) (j->scales + r * CONV2D_HMX_SCALES_N)));

        __fp16 *       o  = j->out + r * ostride;
        const __fp16 * w0 = j->w + r * wstride;

        for (uint32_t i = 0; i < n_tiles; i++, o += HMX_FP16_TILE_N_ELMS) {
            if (!j->live[i]) {
                continue;
            }
            const uint32_t base = act + i * HMX_FP16_TILE_SIZE;
            const __fp16 * w    = w0;

            asm volatile(HMX_CLRACC_F16());
            if (n_cc == 1) {
                for (uint32_t t = 0; t < T; t++) {
                    asm volatile("{ activation.hf = mxmem(%0, %1):single\n weight.hf = mxmem(%2, %3) }"
                                 :: "r"(base + taps[t]), "r"(rt_last), "r"(w), "r"(wr_last));
                    w += HMX_FP16_TILE_N_ELMS;
                }
            } else {
                for (uint32_t t = 0; t < T; t++) {
                    uint32_t rs = base + taps[t];
                    for (uint32_t cc = 1; cc < n_cc; cc++) {
                        asm volatile("{ activation.hf = mxmem(%0, %1):single\n weight.hf = mxmem(%2, %3) }"
                                     :: "r"(rs), "r"(HMX_FP16_TILE_SIZE | CONV2D_WIN_RT_FULL), "r"(w), "r"(HMX_FP16_TILE_SIZE - 1));
                        rs += plane;
                        w  += HMX_FP16_TILE_N_ELMS;
                    }
                    asm volatile("{ activation.hf = mxmem(%0, %1):single\n weight.hf = mxmem(%2, %3) }"
                                 :: "r"(rs), "r"(rt_last), "r"(w), "r"(wr_last));
                    w += HMX_FP16_TILE_N_ELMS;
                }
            }
            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(o), "r"(0) : "memory");
        }
    }
}

static inline void conv2d_vdeal_w(HVX_Vector * a, HVX_Vector * b) {
    const HVX_VectorPair p = Q6_W_vdeal_VVR(*b, *a, -4);
    *a = Q6_V_lo_W(p);
    *b = Q6_V_hi_W(p);
}

#define CONV2D_XPOSE_STAGE(v, m, op)                      \
    _Pragma("unroll") for (uint32_t i = 0; i < 16; i++) { \
        if (!(i & (m))) {                                 \
            op(&(v)[i], &(v)[i | (m)]);                   \
        }                                                 \
    }

// 16x32 word transpose: in[i] = [rows 2i, 2i + 1 of 16 words each], out[slot(v)] word c = row c word v
static inline void conv2d_xpose_w16(HVX_Vector * restrict v) {
    CONV2D_XPOSE_STAGE(v, 8, conv2d_vshuff_w);
    CONV2D_XPOSE_STAGE(v, 4, conv2d_vshuff_w);
    CONV2D_XPOSE_STAGE(v, 2, conv2d_vshuff_w);
    CONV2D_XPOSE_STAGE(v, 1, conv2d_vshuff_w);
    CONV2D_XPOSE_STAGE(v, 8, conv2d_vshuff_w);
}

// Inverse of conv2d_xpose_w16
static inline void conv2d_unxpose_w16(HVX_Vector * restrict v) {
    CONV2D_XPOSE_STAGE(v, 8, conv2d_vdeal_w);
    CONV2D_XPOSE_STAGE(v, 1, conv2d_vdeal_w);
    CONV2D_XPOSE_STAGE(v, 2, conv2d_vdeal_w);
    CONV2D_XPOSE_STAGE(v, 4, conv2d_vdeal_w);
    CONV2D_XPOSE_STAGE(v, 8, conv2d_vdeal_w);
}

// conv2d_xpose_w16 output position of row v
static inline uint32_t conv2d_xpose_slot(uint32_t v) {
    return ((v & 1) << 3) | (v >> 1);
}

static inline uint32_t conv2d_hmx_win_chunk_tiles(const struct htp_conv2d_win_context * wc, uint32_t c, uint32_t * ts) {
    *ts = wc->t_base + c * wc->kp->ct_chunk;
    return MIN(wc->kp->ct_chunk, wc->t_end - *ts);
}

// Cout-contiguous kernel: DMA w[t][c][32 cout] rows, tile vector p = (row 2p, row 2p+1) interleaved per cout
static void conv2d_hmx_win_wpack_kmajor(const struct htp_conv2d_win_context * wc, unsigned int nth, unsigned int ith) {
    struct htp_ops_context *             octx   = wc->octx;
    const struct htp_conv2d_kernel_params * kp     = wc->kp;
    const struct htp_tensor *            kernel = octx->src[0];

    const uint32_t KW     = kernel->ne[0];
    const uint32_t KH     = kernel->ne[1];
    const uint32_t Cin    = kernel->ne[2];
    const uint32_t Cout   = kernel->ne[3];
    const uint32_t T      = KW * KH;
    const uint32_t n_cc   = kp->window.n_cc;
    const uint32_t TG     = kp->window.w_cs;
    const bool     is_f16 = kernel->type == HTP_TYPE_F16;
    const uint32_t es     = is_f16 ? sizeof(__fp16) : sizeof(float);

    dma_queue *  dmaq = octx->ctx->dma[ith];
    uint8_t *    raw  = wc->thr_base + (size_t) ith * kp->thr_size;
    const HVX_Vector zero = Q6_V_vzero();

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t n_tg    = (T + TG - 1) / TG;
    const uint32_t n_units = wc->n_rt * n_cc * n_tg;

    for (uint32_t unit = ith; unit < n_units; unit += nth) {
        const uint32_t tg  = unit % n_tg;
        const uint32_t cc  = (unit / n_tg) % n_cc;
        const uint32_t rt  = unit / (n_tg * n_cc);
        const uint32_t co0 = (wc->rt0 + rt) * 32;
        const uint32_t nco = MIN(32, Cout - co0);
        const uint32_t c0  = cc * 32;
        const uint32_t nc  = MIN(32, Cin - c0);
        const uint32_t t0  = tg * TG;
        const uint32_t nt  = MIN(TG, T - t0);

        for (uint32_t t = 0; t < nt; t++) {
            const uint32_t   kh  = (t0 + t) / KW;
            const uint32_t   kw  = (t0 + t) - kh * KW;
            const dma_addr_t src = kernel->data + (size_t) kw * kernel->nb[0] + (size_t) kh * kernel->nb[1] + (size_t) c0 * kernel->nb[2] + (size_t) co0 * es;
            conv2d_dma_push(dmaq, dma_make_data(raw + (size_t) t * 32 * VLEN, src), VLEN, kernel->nb[2], nco * es, nc);
        }
        conv2d_dma_pop(dmaq, nt);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) unit);
        const HVX_VectorPred qco = Q6_Q_vsetq2_R(nco * es);
        if (kp->window.fold) {
            // crouton channel kw * Cin + c is row c of tap kw of this kh
            const HVX_Vector * rows = (const HVX_Vector *) raw;
            HVX_Vector *       tile = (HVX_Vector *) (wc->w_tiles + ((size_t) (rt * KH + tg) * n_cc + cc) * HMX_FP16_TILE_N_ELMS);
            uint32_t           kw   = 0;
            uint32_t           c    = 0;
            for (uint32_t p = 0; p < 16; p++) {
                const HVX_Vector a = kw < KW ? Q6_V_vmux_QVV(qco, rows[kw * 32 + c], zero) : zero;
                if (++c == Cin) {
                    c = 0;
                    kw++;
                }
                const HVX_Vector b = kw < KW ? Q6_V_vmux_QVV(qco, rows[kw * 32 + c], zero) : zero;
                if (++c == Cin) {
                    c = 0;
                    kw++;
                }
                tile[p] = is_f16 ? Q6_V_lo_W(Q6_W_vshuff_VVR(b, a, -2)) : conv2d_f32_to_f16_shuff(a, b);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) unit);
            continue;
        }
        for (uint32_t t = 0; t < nt; t++) {
            const HVX_Vector * rows = (const HVX_Vector *) (raw + (size_t) t * 32 * VLEN);
            HVX_Vector *       tile = (HVX_Vector *) (wc->w_tiles + ((size_t) (rt * T + t0 + t) * n_cc + cc) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t p = 0; p < 16; p++) {
                const HVX_Vector a = 2 * p + 0 < nc ? Q6_V_vmux_QVV(qco, rows[2 * p + 0], zero) : zero;
                const HVX_Vector b = 2 * p + 1 < nc ? Q6_V_vmux_QVV(qco, rows[2 * p + 1], zero) : zero;
                tile[p] = is_f16 ? Q6_V_lo_W(Q6_W_vshuff_VVR(b, a, -2)) : conv2d_f32_to_f16_shuff(a, b);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) unit);
    }
}

// K-contiguous kernel: DMA 32 cout rows of the w_cs cin of a packing unit, a dummy DMA if the unit has nothing to load
static void conv2d_hmx_win_wrows_dma(const struct htp_conv2d_win_context * wc, dma_queue * dmaq, uint8_t * raw, uint32_t unit, uint32_t n_cs) {
    const struct htp_conv2d_kernel_params * kp     = wc->kp;
    const struct htp_tensor *            kernel = wc->octx->src[0];

    const uint32_t T   = kernel->ne[0] * kernel->ne[1];
    const uint32_t Cin = kernel->ne[2];
    const uint32_t CS  = kp->window.w_cs;
    const uint32_t es  = kernel->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t kpw = VLEN / es;

    const uint32_t cs  = unit % n_cs;
    const uint32_t cc  = (unit / n_cs) % kp->window.n_cc;
    const uint32_t rt  = unit / (n_cs * kp->window.n_cc);
    const uint32_t co0 = (wc->rt0 + rt) * 32;
    const uint32_t nco = MIN(32, (uint32_t) kernel->ne[3] - co0);
    const uint32_t c0  = cc * 32 + cs * CS;
    const uint32_t nk  = c0 < Cin ? MIN(CS, Cin - c0) * T : 0;
    if (nk == 0 || (kp->window.fold && cs > 0)) {
        conv2d_dma_push(dmaq, dma_make_data(raw, raw), 0, 0, 0, 0);
        return;
    }
    conv2d_dma_push(dmaq, dma_make_data(raw, kernel->data + (size_t) co0 * kernel->nb[3] + (size_t) c0 * kernel->nb[2]),
                    (size_t) ((nk + kpw - 1) / kpw) * VLEN, kernel->nb[3], (size_t) nk * es, nco);
}

// K-contiguous kernel (k = cin * T + t): transpose the 32 cout rows of a packing unit to k rows of 32 cout, then into tiles
static void conv2d_hmx_win_wrows_pack(const struct htp_conv2d_win_context * wc, const uint8_t * raw, uint8_t * xr, HVX_Vector * vin,
                                      HVX_Vector * scratch, uint32_t unit, uint32_t n_cs) {
    const struct htp_conv2d_kernel_params * kp     = wc->kp;
    const struct htp_tensor *            kernel = wc->octx->src[0];

    const uint32_t KW     = kernel->ne[0];
    const uint32_t KH     = kernel->ne[1];
    const uint32_t Cin    = kernel->ne[2];
    const uint32_t Cout   = kernel->ne[3];
    const uint32_t T      = KW * KH;
    const uint32_t n_cc   = kp->window.n_cc;
    const uint32_t CS     = kp->window.w_cs;
    const bool     is_f16 = kernel->type == HTP_TYPE_F16;
    const uint32_t kpw    = is_f16 ? 64 : 32; // k per vector
    const uint32_t rstep  = is_f16 ? 64 : 128; // bytes per transposed k row
    const HVX_Vector zero = Q6_V_vzero();

    const uint32_t cs  = unit % n_cs;
    const uint32_t cc  = (unit / n_cs) % n_cc;
    const uint32_t rt  = unit / (n_cs * n_cc);
    const uint32_t co0 = (wc->rt0 + rt) * 32;
    const uint32_t nco = MIN(32, Cout - co0);
    const uint32_t c0  = cc * 32 + cs * CS;
    const uint32_t nc   = c0 < Cin ? MIN(CS, Cin - c0) : 0;
    const uint32_t nk   = nc * T;
    const uint32_t nvec = (nk + kpw - 1) / kpw;
    const size_t   rs   = (size_t) nvec * VLEN;

    if (kp->window.fold && cs > 0) {
        return;
    }

    for (uint32_t b = 0; b < nvec; b++) {
        for (uint32_t o = 0; o < 32; o++) {
            vin[o] = o < nco ? *(const HVX_Vector *) (raw + o * rs + (size_t) b * VLEN) : zero;
        }
        HVX_Vector * xo = (HVX_Vector *) (xr + (size_t) b * 32 * VLEN);
        conv2d_transpose_w32(vin, 1, scratch, xo, xo + 16);
        if (is_f16) {
            for (uint32_t i = 0; i < 32; i++) {
                xo[i] = Q6_Vh_vdeal_Vh(xo[i]);
            }
        }
    }

    if (kp->window.fold) {
        // tile kh, crouton channel kw * Cin + c is k row c * T + kh * KW + kw
        for (uint32_t kh = 0; kh < KH; kh++) {
            HVX_Vector * tile = (HVX_Vector *) (wc->w_tiles + ((size_t) (rt * KH + kh) * n_cc + cc) * HMX_FP16_TILE_N_ELMS);
            uint32_t     kw   = 0;
            uint32_t     c    = 0;
            for (uint32_t p = 0; p < 16; p++) {
                const HVX_Vector a = kw < KW ? hvx_vmemu(xr + (size_t) (c * T + kh * KW + kw) * rstep) : zero;
                if (++c == Cin) {
                    c = 0;
                    kw++;
                }
                const HVX_Vector b = kw < KW ? hvx_vmemu(xr + (size_t) (c * T + kh * KW + kw) * rstep) : zero;
                if (++c == Cin) {
                    c = 0;
                    kw++;
                }
                tile[p] = is_f16 ? Q6_V_lo_W(Q6_W_vshuff_VVR(b, a, -2)) : conv2d_f32_to_f16_shuff(a, b);
            }
        }
        return;
    }

    // tile (t, cc) vector p takes cin rows 2p, 2p + 1 of this step
    for (uint32_t t = 0; t < T; t++) {
        HVX_Vector * tile = (HVX_Vector *) (wc->w_tiles + ((size_t) (rt * T + t) * n_cc + cc) * HMX_FP16_TILE_N_ELMS);
        for (uint32_t pc = 0; pc < CS; pc += 2) {
            const uint32_t   c = cs * CS + pc;
            const uint32_t   p = c / 2;
            const HVX_Vector a = pc + 0 < nc ? hvx_vmemu(xr + (size_t) ((pc + 0) * T + t) * rstep) : zero;
            const HVX_Vector b = pc + 1 < nc ? hvx_vmemu(xr + (size_t) ((pc + 1) * T + t) * rstep) : zero;
            tile[p] = is_f16 ? Q6_V_lo_W(Q6_W_vshuff_VVR(b, a, -2)) : conv2d_f32_to_f16_shuff(a, b);
        }
    }
}

static void conv2d_hmx_win_wpack_rows(const struct htp_conv2d_win_context * wc, unsigned int nth, unsigned int ith) {
    struct htp_ops_context *             octx   = wc->octx;
    const struct htp_conv2d_kernel_params * kp     = wc->kp;
    const struct htp_tensor *            kernel = octx->src[0];

    const uint32_t T   = kernel->ne[0] * kernel->ne[1];
    const uint32_t CS  = kp->window.w_cs;
    const uint32_t kpw = kernel->type == HTP_TYPE_F16 ? 64 : 32; // k per vector

    const size_t   wraw    = (size_t) 32 * ((CS * T + kpw - 1) / kpw) * VLEN;
    dma_queue *    dmaq    = octx->ctx->dma[ith];
    uint8_t *      raws[2] = { wc->thr_base + (size_t) ith * kp->thr_size, wc->thr_base + (size_t) ith * kp->thr_size + wraw };
    uint8_t *      xr      = raws[1] + wraw;
    HVX_Vector *   vin     = (HVX_Vector *) (xr + wraw);
    HVX_Vector *   scratch = vin + 32;

    const uint32_t n_cs    = (32 + CS - 1) / CS;
    const uint32_t n_units = wc->n_rt * kp->window.n_cc * n_cs;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // one DMA per unit: push(N), push(N + 1), pop(N), pack(N), push(N + 2), ...
    for (uint32_t k = 0, unit = ith; k < 2 && unit < n_units; k++, unit += nth) {
        conv2d_hmx_win_wrows_dma(wc, dmaq, raws[k], unit, n_cs);
    }
    for (uint32_t unit = ith, k = 0; unit < n_units; unit += nth, k++) {
        uint8_t * raw = raws[k & 1];

        conv2d_dma_pop(dmaq, 1);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) unit);
        conv2d_hmx_win_wrows_pack(wc, raw, xr, vin, scratch, unit, n_cs);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) unit);

        if (unit + 2 * nth < n_units) {
            conv2d_hmx_win_wrows_dma(wc, dmaq, raw, unit + 2 * nth, n_cs);
        }
    }
}

static void conv2d_hmx_win_wpack_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_win_context * wc = (const struct htp_conv2d_win_context *) data;

    if (wc->kp->w_kmajor) {
        conv2d_hmx_win_wpack_kmajor(wc, nth, ith);
    } else {
        conv2d_hmx_win_wpack_rows(wc, nth, ith);
    }
}

// Fill stage with input pixels [ua, ub): whcn stage[c][u - ua], cwhn stage[u - ua][c]. Zero outside the image.
// Returns the number of DMAs pushed.
static uint32_t conv2d_hmx_win_stage(const struct htp_conv2d_win_context * wc, dma_queue * dmaq, float * stage, uint32_t ua, uint32_t ub) {
    const struct htp_ops_context *       octx  = wc->octx;
    const struct htp_conv2d_kernel_params * kp    = wc->kp;
    const struct htp_tensor *            input = octx->src[1];

    const uint32_t IW   = input->ne[0];
    const uint32_t IH   = input->ne[1];
    const uint32_t Cin  = input->ne[2];
    const uint32_t N    = input->ne[3];
    const uint32_t SR   = kp->window.sr;
    const uint32_t IHS  = kp->window.ihs;
    const uint32_t p0   = octx->op_params[2];
    const uint32_t p1   = octx->op_params[3];
    const uint32_t span = kp->window.seg_b * 32;
    const bool     cwhn = kp->cwhn;

    // rows of one image are contiguous when there is no column padding
    const bool rows_contig = p0 == 0 && input->nb[1] == (size_t) IW * input->nb[0];

    uint32_t nd = 0;
    uint32_t u  = ua;
    while (u < ub) {
        const uint32_t vr  = fastdiv(u, &kp->window.div_sr);
        const uint32_t col = u - vr * SR;
        const uint32_t n   = fastdiv(vr, &kp->window.div_ihs);
        const uint32_t r   = vr - n * IHS;
        const int32_t  ih  = (int32_t) r - (int32_t) p1;

        uint32_t end = MIN(ub, (vr + 1) * SR);
        if (n >= N || ih < 0 || ih >= (int32_t) IH) {
            const uint32_t len = end - u;
            if (cwhn) {
                hvx_splat_f32_u(stage + (size_t) (u - ua) * Cin, 0.0f, len * Cin);
            } else {
                for (uint32_t c = 0; c < Cin; c++) {
                    hvx_splat_f32_u(stage + (size_t) c * span + (u - ua), 0.0f, len);
                }
            }
            u = end;
            continue;
        }

        if (rows_contig) {
            end = MIN(ub, (vr + (IH - (uint32_t) ih)) * SR);
        }

        // valid columns [a, b) of the first row, all columns of the following rows
        const uint32_t a   = MAX(col, p0);
        const uint32_t len = end - u;
        uint32_t       b   = MIN(col + len, p0 + IW);
        if (rows_contig) {
            b = col + len;
        }

        if (a > col) {
            const uint32_t z = MIN(a, col + len) - col;
            if (cwhn) {
                hvx_splat_f32_u(stage + (size_t) (u - ua) * Cin, 0.0f, z * Cin);
            } else {
                for (uint32_t c = 0; c < Cin; c++) {
                    hvx_splat_f32_u(stage + (size_t) c * span + (u - ua), 0.0f, z);
                }
            }
        }
        if (b < col + len) {
            const uint32_t z0 = MAX(a, b);
            const uint32_t z  = col + len - z0;
            if (cwhn) {
                hvx_splat_f32_u(stage + (size_t) (u - ua + z0 - col) * Cin, 0.0f, z * Cin);
            } else {
                for (uint32_t c = 0; c < Cin; c++) {
                    hvx_splat_f32_u(stage + (size_t) c * span + (u - ua + z0 - col), 0.0f, z);
                }
            }
        }
        if (b > a) {
            const dma_addr_t src = input->data + (size_t) n * input->nb[3] + (size_t) ih * input->nb[1] + (size_t) (a - p0) * input->nb[0];
            const uint32_t   off = u - ua + (a - col);
            if (cwhn) {
                const size_t sz = (size_t) (b - a) * Cin * sizeof(float);
                conv2d_dma_push(dmaq, dma_make_data(stage + (size_t) off * Cin, src), sz, sz, sz, 1);
            } else {
                conv2d_dma_push(dmaq, dma_make_data(stage + off, src), (size_t) span * sizeof(float), input->nb[2],
                                (size_t) (b - a) * sizeof(float), Cin);
            }
            nd++;
        }
        u = end;
    }
    return nd;
}

// Fold staging: stage holds the raw input rows of the virtual rows of [ua, ub), after 128 bytes of slack for reads before
// the first row. whcn: raw[c][vr - vr_a][iw], cwhn: raw[vr - vr_a][iw][c].
// Folded pixel (kw * Cin + c, u = vr * OW + ow) is input pixel (vr, ow + kw * d0 - p0), channel c.
static inline float * conv2d_hmx_win_fold_raw(float * stage) {
    return stage + VLEN_FP32;
}

static inline uint32_t conv2d_hmx_win_fold_rplane(const struct htp_conv2d_win_context * wc) {
    const uint32_t IW = wc->octx->src[1]->ne[0];
    return ((wc->kp->window.seg_b * 32 + wc->kp->window.sr - 1) / wc->kp->window.sr + 1) * IW;
}

// DMA the raw input rows of [ua, ub), returns the number of DMAs pushed
static uint32_t conv2d_hmx_win_stage_fold(const struct htp_conv2d_win_context * wc, dma_queue * dmaq, float * stage, uint32_t ua, uint32_t ub) {
    const struct htp_ops_context *       octx  = wc->octx;
    const struct htp_conv2d_kernel_params * kp    = wc->kp;
    const struct htp_tensor *            input = octx->src[1];

    const uint32_t IW   = input->ne[0];
    const int32_t  IH   = input->ne[1];
    const uint32_t Cin  = input->ne[2];
    const uint32_t N    = input->ne[3];
    const uint32_t IHS  = kp->window.ihs;
    const int32_t  p1   = octx->op_params[3];
    const uint32_t RP   = conv2d_hmx_win_fold_rplane(wc);
    const bool     cont = input->nb[1] == (size_t) IW * input->nb[0];
    float *        raw  = conv2d_hmx_win_fold_raw(stage);

    const uint32_t vr_a = fastdiv(ua, &kp->window.div_sr);
    const uint32_t vr_b = fastdiv(ub - 1, &kp->window.div_sr) + 1;

    uint32_t nd = 0;
    for (uint32_t vr = vr_a; vr < vr_b;) {
        const uint32_t n  = fastdiv(vr, &kp->window.div_ihs);
        const int32_t  ih = (int32_t) (vr - n * IHS) - p1;
        if (n >= N || ih < 0 || ih >= IH) {
            vr++;
            continue;
        }
        // rows of one image are contiguous in DDR
        const uint32_t nr = cont ? MIN(vr_b - vr, (uint32_t) (IH - ih)) : 1;
        const dma_addr_t src = input->data + (size_t) n * input->nb[3] + (size_t) ih * input->nb[1];
        if (kp->cwhn) {
            // raw[vr - vr_a][iw][c]
            const size_t sz = (size_t) nr * IW * Cin * sizeof(float);
            conv2d_dma_push(dmaq, dma_make_data(raw + (size_t) (vr - vr_a) * IW * Cin, src), sz, sz, sz, 1);
        } else {
            conv2d_dma_push(dmaq, dma_make_data(raw + (size_t) (vr - vr_a) * IW, src), (size_t) RP * sizeof(float), input->nb[2],
                            (size_t) nr * IW * sizeof(float), Cin);
        }
        nd++;
        vr += nr;
    }
    return nd;
}

// Build croutons [ba, ba + nb) (relative to the chunk crouton base b0) from a filled stage
static void conv2d_hmx_win_croutons(const struct htp_conv2d_win_context * wc, const float * stage, __fp16 * act, uint32_t bo, uint32_t nb) {
    const struct htp_conv2d_kernel_params * kp   = wc->kp;
    const uint32_t                       Cin  = wc->octx->src[1]->ne[2] * (kp->window.fold ? wc->octx->src[0]->ne[0] : 1);
    const uint32_t                       n_cc = kp->window.n_cc;
    const uint32_t                       span = kp->window.seg_b * 32;
    const size_t                         pl   = (size_t) kp->window.n_bc * HMX_FP16_TILE_N_ELMS;
    const HVX_Vector                     zero = Q6_V_vzero();

    if (wc->n_scat) {
        // word j of the [c = 2k | c = 2k + 1] pixel pairs goes to (j % 16) * 128 + (2k + j / 16) * 4
        const HVX_Vector soff = *(const HVX_Vector *) conv2d_hmx_win_scat_off;
        for (uint32_t i = 0; i < nb; i++) {
            const size_t dst = (size_t) (act + (size_t) (bo + i) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t k = 0; k < wc->n_scat; k++) {
                const HVX_Vector     a = *(const HVX_Vector *) (stage + (size_t) (2 * k) * span + i * 32);
                const HVX_Vector     b = 2 * k + 1 < Cin ? *(const HVX_Vector *) (stage + (size_t) (2 * k + 1) * span + i * 32) : zero;
                const HVX_VectorPred q = Q6_Q_vsetq2_R(2 * k + 1 < Cin ? VLEN : VLEN / 2);
                Q6_vscatter_QRMVwV(q, dst, HMX_FP16_TILE_SIZE - 1, Q6_Vw_vadd_VwVw(soff, Q6_V_vsplat_R(8 * k)), conv2d_f32_to_f16(a, b));
            }
        }
        if (nb > 0) {
            hvx_gather_sync(act + (size_t) (bo + nb - 1) * HMX_FP16_TILE_N_ELMS);
        }
        return;
    }

    if (kp->cwhn) {
        for (uint32_t cc = 0; cc < n_cc; cc++) {
            const uint32_t       nc = MIN(32, Cin - cc * 32);
            const HVX_VectorPred qc = Q6_Q_vsetq2_R(nc * sizeof(float));
            for (uint32_t i = 0; i < nb; i++) {
                HVX_Vector *  dst = (HVX_Vector *) (act + cc * pl + (size_t) (bo + i) * HMX_FP16_TILE_N_ELMS);
                const float * src = stage + (size_t) i * 32 * Cin + cc * 32;
                for (uint32_t v = 0; v < 16; v++) {
                    const HVX_Vector a = Q6_V_vmux_QVV(qc, hvx_vmemu(src + (size_t) (2 * v + 0) * Cin), zero);
                    const HVX_Vector b = Q6_V_vmux_QVV(qc, hvx_vmemu(src + (size_t) (2 * v + 1) * Cin), zero);
                    dst[v] = conv2d_f32_to_f16_shuff(a, b);
                }
            }
        }
        return;
    }

    HVX_Vector y[16];
    for (uint32_t cc = 0; cc < n_cc; cc++) {
        for (uint32_t i = 0; i < nb; i++) {
            // channel rows are span apart, walk them with one pointer
            const uint32_t     nv = MIN(32, Cin - cc * 32);
            const HVX_Vector * p  = (const HVX_Vector *) (stage + (size_t) cc * 32 * span + i * 32);
            const uint32_t     sv = span / VLEN_FP32;
            if (nv == 32) {
                #pragma unroll
                for (uint32_t k = 0; k < 16; k++) {
                    const HVX_Vector a = p[0];
                    const HVX_Vector b = p[sv];
                    p += 2 * sv;
                    y[k] = conv2d_f32_to_f16(a, b);
                }
            } else {
                #pragma unroll
                for (uint32_t k = 0; k < 16; k++) {
                    const HVX_Vector a = 2 * k + 0 < nv ? p[0] : zero;
                    const HVX_Vector b = 2 * k + 1 < nv ? p[sv] : zero;
                    p += 2 * sv;
                    y[k] = conv2d_f32_to_f16(a, b);
                }
            }
            conv2d_xpose_w16(y);
            HVX_Vector * dst = (HVX_Vector *) (act + cc * pl + (size_t) (bo + i) * HMX_FP16_TILE_N_ELMS);
            #pragma unroll
            for (uint32_t v = 0; v < 16; v++) {
                dst[v] = y[conv2d_xpose_slot(v)];
            }
        }
    }
}

// Fold: one crouton channel vector (32 pixels u0..u0+31), assembled from the row pieces of the block
struct conv2d_hmx_win_fold_piece {
    int32_t off; // raw offset of lane 0 for kw = 0, c = 0
    int32_t l0;  // lanes [l0, l1) of this row
    int32_t l1;
    int32_t lb;  // lane of iw = 0 for kw = 0
};

// Lanes of a row piece that read inside the image for column shift sh
static inline HVX_VectorPred conv2d_hmx_win_fold_q(const struct conv2d_hmx_win_fold_piece * pc, int32_t sh, int32_t IW) {
    const int32_t lo = MAX(pc->l0, pc->lb - sh);
    const int32_t hi = MIN(pc->l1, pc->lb - sh + IW);
    if (hi <= lo) {
        return Q6_Q_vsetq_R(0);
    }
    return Q6_Q_and_QQn(Q6_Q_vsetq2_R(hi * sizeof(float)), Q6_Q_vsetq_R(lo * sizeof(float)));
}

static void conv2d_hmx_win_croutons_fold(const struct htp_conv2d_win_context * wc, const float * stage, __fp16 * act, uint32_t bo,
                                     uint32_t nb, uint32_t ua) {
    const struct htp_ops_context *       octx  = wc->octx;
    const struct htp_conv2d_kernel_params * kp    = wc->kp;
    const struct htp_tensor *            input = octx->src[1];

    const int32_t  IW  = input->ne[0];
    const int32_t  IH  = input->ne[1];
    const uint32_t Cin = input->ne[2];
    const uint32_t N   = input->ne[3];
    const uint32_t KW  = octx->src[0]->ne[0];
    const uint32_t nch = KW * Cin;
    const uint32_t SR  = kp->window.sr;
    const uint32_t IHS = kp->window.ihs;
    const int32_t  p0  = octx->op_params[2];
    const int32_t  p1  = octx->op_params[3];
    const int32_t  d0  = octx->op_params[4];
    const uint32_t RP  = conv2d_hmx_win_fold_rplane(wc);
    const float *  raw = conv2d_hmx_win_fold_raw((float *) stage);

    const uint32_t   vr_a = fastdiv(ua, &kp->window.div_sr);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector soff = *(const HVX_Vector *) conv2d_hmx_win_scat_off;

    for (uint32_t i = 0; i < nb; i++) {
        const uint32_t u0 = ua + 32 * i;

        struct conv2d_hmx_win_fold_piece pc[4];
        uint32_t                     np = 0;
        for (uint32_t vr = fastdiv(u0, &kp->window.div_sr); vr * SR < u0 + 32 && np < 4; vr++) {
            const uint32_t n  = fastdiv(vr, &kp->window.div_ihs);
            const int32_t  ih = (int32_t) (vr - n * IHS) - p1;
            if (n >= N || ih < 0 || ih >= IH) {
                continue;
            }
            const int32_t l0  = (int32_t) (MAX(u0, vr * SR) - u0);
            const int32_t ow0 = (int32_t) (MAX(u0, vr * SR) - vr * SR);
            pc[np].l0  = l0;
            pc[np].l1  = (int32_t) (MIN(u0 + 32, (vr + 1) * SR) - u0);
            pc[np].lb  = l0 - ow0 + p0;
            pc[np].off = (int32_t) ((vr - vr_a) * IW) + ow0 - l0 - p0;
            np++;
        }

        __fp16 * dst = act + (size_t) (bo + i) * HMX_FP16_TILE_N_ELMS;
        if (np == 0) {
            HVX_Vector * d = (HVX_Vector *) dst;
            for (uint32_t v = 0; v < 16; v++) {
                d[v] = zero;
            }
            continue;
        }

        // channel ch = kw * Cin + c. Channel pairs go to the crouton memory first, the transpose runs on the reloaded rows.
        // Lanes and offsets of the row pieces depend on kw only.
        HVX_Vector * d    = (HVX_Vector *) dst;
        const bool   scat = wc->n_scat != 0;
        HVX_Vector   pa   = zero;
        uint32_t     ch   = 0;
        for (uint32_t kw = 0; kw < KW; kw++) {
            const int32_t  sh = (int32_t) kw * d0;
            const HVX_VectorPred q0 = conv2d_hmx_win_fold_q(&pc[0], sh, IW);
            const HVX_VectorPred q1 = np > 1 ? conv2d_hmx_win_fold_q(&pc[1], sh, IW) : q0;
            const HVX_VectorPred q2 = np > 2 ? conv2d_hmx_win_fold_q(&pc[2], sh, IW) : q0;
            const HVX_VectorPred q3 = np > 3 ? conv2d_hmx_win_fold_q(&pc[3], sh, IW) : q0;
            const int32_t        f0 = pc[0].off + sh;
            const int32_t        f1 = np > 1 ? pc[1].off + sh : f0;
            const int32_t        f2 = np > 2 ? pc[2].off + sh : f0;
            const int32_t        f3 = np > 3 ? pc[3].off + sh : f0;

            for (uint32_t c = 0; c < Cin; c++, ch++) {
                const float * r = raw + (size_t) c * RP;
                HVX_Vector    x = Q6_V_vmux_QVV(q0, hvx_vmemu(r + f0), zero);
                if (np > 1) {
                    x = Q6_V_vmux_QVV(q1, hvx_vmemu(r + f1), x);
                }
                if (np > 2) {
                    x = Q6_V_vmux_QVV(q2, hvx_vmemu(r + f2), x);
                }
                if (np > 3) {
                    x = Q6_V_vmux_QVV(q3, hvx_vmemu(r + f3), x);
                }
                if (!(ch & 1)) {
                    pa = x;
                    continue;
                }
                const HVX_Vector y = conv2d_f32_to_f16(pa, x);
                if (scat) {
                    Q6_vscatter_QRMVwV(Q6_Q_vsetq2_R(VLEN), (size_t) dst, HMX_FP16_TILE_SIZE - 1,
                                       Q6_Vw_vadd_VwVw(soff, Q6_V_vsplat_R(4 * (ch - 1))), y);
                } else {
                    d[ch / 2] = y;
                }
            }
        }
        if (ch & 1) {
            const HVX_Vector y = conv2d_f32_to_f16(pa, zero);
            if (scat) {
                Q6_vscatter_QRMVwV(Q6_Q_vsetq2_R(VLEN / 2), (size_t) dst, HMX_FP16_TILE_SIZE - 1,
                                   Q6_Vw_vadd_VwVw(soff, Q6_V_vsplat_R(4 * (ch - 1))), y);
            } else {
                d[ch / 2] = y;
            }
            ch++;
        }
        if (!scat) {
            for (uint32_t k = ch / 2; k < 16; k++) {
                d[k] = zero;
            }
        }
        if (scat) {
            continue;
        }

        HVX_Vector y[16];
        #pragma unroll
        for (uint32_t k = 0; k < 16; k++) {
            y[k] = d[k];
        }
        conv2d_xpose_w16(y);
        #pragma unroll
        for (uint32_t v = 0; v < 16; v++) {
            d[v] = y[conv2d_xpose_slot(v)];
        }
    }
    if (wc->n_scat && nb > 0) {
        hvx_gather_sync(act + (size_t) (bo + nb - 1) * HMX_FP16_TILE_N_ELMS);
    }
}

// Fold, cwhn with d0 = 1: the KW * Cin folded channels of a pixel are contiguous in the raw row
static void conv2d_hmx_win_croutons_fold_cwhn(const struct htp_conv2d_win_context * wc, const float * stage, __fp16 * act, uint32_t bo,
                                          uint32_t nb, uint32_t ua) {
    const struct htp_ops_context *       octx  = wc->octx;
    const struct htp_conv2d_kernel_params * kp    = wc->kp;
    const struct htp_tensor *            input = octx->src[1];

    const int32_t  IW  = input->ne[0];
    const int32_t  IH  = input->ne[1];
    const int32_t  Cin = input->ne[2];
    const uint32_t N   = input->ne[3];
    const int32_t  KW  = octx->src[0]->ne[0];
    const uint32_t SR  = kp->window.sr;
    const uint32_t IHS = kp->window.ihs;
    const int32_t  p0  = octx->op_params[2];
    const int32_t  p1  = octx->op_params[3];
    const float *  raw = conv2d_hmx_win_fold_raw((float *) stage);

    const uint32_t   vr_a = fastdiv(ua, &kp->window.div_sr);
    const HVX_Vector zero = Q6_V_vzero();

    uint32_t vr = vr_a;
    uint32_t ow = ua - vr * SR;
    bool     ok = false;
    for (uint32_t i = 0; i < nb; i++) {
        HVX_Vector * dst = (HVX_Vector *) (act + (size_t) (bo + i) * HMX_FP16_TILE_N_ELMS);
        HVX_Vector   pa  = zero;
        for (uint32_t j = 0; j < 32; j++) {
            if (ow == 0 || (i == 0 && j == 0)) {
                const uint32_t n  = fastdiv(vr, &kp->window.div_ihs);
                const int32_t  ih = (int32_t) (vr - n * IHS) - p1;
                ok                = n < N && ih >= 0 && ih < IH;
            }
            HVX_Vector x = zero;
            if (ok) {
                // folded channels kw * Cin + c read input column ow + kw - p0, valid for kw in [p0 - ow, p0 + IW - ow)
                const int32_t lo = MAX(0, p0 - (int32_t) ow) * Cin;
                const int32_t hi = MIN(KW, p0 + IW - (int32_t) ow) * Cin;
                if (hi > lo) {
                    const float *        src = raw + ((int32_t) ((vr - vr_a) * IW + ow) - p0) * Cin;
                    const HVX_VectorPred q   = Q6_Q_and_QQn(Q6_Q_vsetq2_R(hi * sizeof(float)), Q6_Q_vsetq_R(lo * sizeof(float)));
                    x = Q6_V_vmux_QVV(q, hvx_vmemu(src), zero);
                }
            }
            if (j & 1) {
                dst[j / 2] = conv2d_f32_to_f16_shuff(pa, x);
            } else {
                pa = x;
            }
            if (++ow == SR) {
                ow = 0;
                vr++;
            }
        }
    }
}

static void conv2d_hmx_win_pack(const struct htp_conv2d_win_context * wc, unsigned int nth, unsigned int ith) {
    struct htp_ops_context *             octx = wc->octx;
    const struct htp_conv2d_kernel_params * kp   = wc->kp;

    uint32_t       ts;
    const uint32_t nt  = conv2d_hmx_win_chunk_tiles(wc, wc->pack_chunk, &ts);
    const uint32_t ncb = MIN(kp->window.n_bc, nt + (kp->window.max_off >> 5) + 1);
    const uint32_t bpt = (ncb + nth - 1) / nth;
    const uint32_t b0  = MIN(ith * bpt, ncb);
    const uint32_t b1  = MIN(b0 + bpt, ncb);

    dma_queue * dmaq      = octx->ctx->dma[ith];
    uint8_t *   thr       = wc->thr_base + (size_t) ith * kp->thr_size;
    float *     stages[2] = { (float *) thr, (float *) (thr + kp->stage_size) };
    __fp16 *    act       = wc->act[wc->pack_chunk & 1];

    uint32_t (*stage_fn)(const struct htp_conv2d_win_context *, dma_queue *, float *, uint32_t, uint32_t) =
        kp->window.fold ? conv2d_hmx_win_stage_fold : conv2d_hmx_win_stage;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t SB = kp->window.seg_b;

    // push(N), push(N + 1), pop(N), croutons(N), push(N + 2), ...
    // The first pop also retires the output DMAs left in flight by the unpack of this phase.
    uint32_t nd[2] = { dma_queue_depth(dmaq), 0 };
    for (uint32_t i = 0; i < 2 && b0 + i * SB < b1; i++) {
        const uint32_t b = b0 + i * SB;
        nd[i] += stage_fn(wc, dmaq, stages[i], (ts + b) * 32, (ts + MIN(b + SB, b1)) * 32);
    }

    uint32_t si = 0;
    for (uint32_t b = b0; b < b1; b += SB) {
        const uint32_t nb = MIN(SB, b1 - b);

        conv2d_dma_pop(dmaq, nd[si]);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) b);
        if (kp->window.fold && kp->cwhn) {
            conv2d_hmx_win_croutons_fold_cwhn(wc, stages[si], act, b, nb, (ts + b) * 32);
        } else if (kp->window.fold) {
            conv2d_hmx_win_croutons_fold(wc, stages[si], act, b, nb, (ts + b) * 32);
        } else {
            conv2d_hmx_win_croutons(wc, stages[si], act, b, nb);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) b);

        const uint32_t bn = b + 2 * SB;
        nd[si] = bn < b1 ? stage_fn(wc, dmaq, stages[si], (ts + bn) * 32, (ts + MIN(bn + SB, b1)) * 32) : 0;
        si ^= 1;
    }
}

static void conv2d_hmx_win_unpack(const struct htp_conv2d_win_context * wc, unsigned int nth, unsigned int ith) {
    struct htp_ops_context *             octx = wc->octx;
    const struct htp_conv2d_kernel_params * kp   = wc->kp;
    const struct htp_tensor *            dst  = octx->dst;

    const uint32_t OW   = dst->ne[0];
    const uint32_t OH   = dst->ne[1];
    const uint32_t Cout = dst->ne[2];
    const uint32_t SR   = kp->window.sr;
    const uint32_t IHS  = kp->window.ihs;
    const uint32_t ST   = kp->window.seg_t;
    const uint32_t OS   = ST * 32;

    uint32_t       ts;
    const uint32_t nt  = conv2d_hmx_win_chunk_tiles(wc, wc->unpack_chunk, &ts);
    const uint32_t tpt = (nt + nth - 1) / nth;
    const uint32_t i0  = MIN(ith * tpt, nt);
    const uint32_t i1  = MIN(i0 + tpt, nt);

    dma_queue *    dmaq      = octx->ctx->dma[ith];
    uint8_t *      thr       = wc->thr_base + (size_t) ith * kp->thr_size + 2 * (size_t) kp->stage_size;
    float *        obuf      = (float *) thr;
    const __fp16 * out       = wc->out[wc->unpack_chunk & 1];

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // nd[bi] = DMAs pushed from obuf bi, popped before it is reused HTP_CONV2D_WIN_OBUF steps later
    uint32_t nd[HTP_CONV2D_WIN_OBUF] = { 0 };
    uint32_t bi                      = 0;

    const bool rows_contig = SR == OW && dst->nb[1] == OW * sizeof(float);

    for (uint32_t i = i0; i < i1; i += ST) {
        const uint32_t ni   = MIN(ST, i1 - i);
        const uint32_t q_lo = (ts + i) * 32;
        const uint32_t q_hi = q_lo + ni * 32;

        for (uint32_t rt = 0; rt < wc->n_rt; rt++) {
            const uint32_t co0 = (wc->rt0 + rt) * 32;
            const uint32_t nco = MIN(32, Cout - co0);
            const uint32_t npr = (nco + 1) / 2;
            const HVX_Vector restore = hvx_vec_splat_f32(conv2d_hmx_rt_restore(wc->scales, rt));

            conv2d_dma_pop(dmaq, nd[bi]);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) i);
            float * ob = obuf + (size_t) bi * 32 * OS;
            for (uint32_t g = 0; g < ni; g++) {
                const HVX_Vector * tv = (const HVX_Vector *) (out + ((size_t) rt * kp->ct_chunk + i + g) * HMX_FP16_TILE_N_ELMS);
                HVX_Vector         y[16];
                #pragma unroll
                for (uint32_t v = 0; v < 16; v++) {
                    y[conv2d_xpose_slot(v)] = tv[v];
                }
                conv2d_unxpose_w16(y);
                HVX_Vector * o = (HVX_Vector *) (ob + g * 32);
                #pragma unroll
                for (uint32_t k = 0; k < 16; k++) {
                    if (k < npr) {
                        const HVX_VectorPair pr = hvx_vec_f16_to_f32(y[k]);
                        o[(2 * k + 0) * (OS / 32)] = hvx_vec_mul_f32_f32(Q6_V_lo_W(pr), restore);
                        o[(2 * k + 1) * (OS / 32)] = hvx_vec_mul_f32_f32(Q6_V_hi_W(pr), restore);
                    }
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) i);

            // write the valid part of each virtual row in [q_lo, q_hi)
            nd[bi] = 0;
            uint32_t vr = fastdiv(q_lo, &kp->window.div_sr);
            for (; vr * SR < q_hi; vr++) {
                const uint32_t n  = fastdiv(vr, &kp->window.div_ihs);
                const uint32_t oh = vr - n * IHS;
                if (vr < wc->vr0 || vr >= wc->vr1 || oh >= OH) {
                    continue;
                }
                const uint32_t base = vr * SR;
                const uint32_t a    = q_lo > base ? q_lo - base : 0;
                uint32_t       b    = MIN(OW, q_hi - base);
                if (b <= a) {
                    continue;
                }
                const dma_addr_t ddr = dst->data + (size_t) n * dst->nb[3] + (size_t) oh * dst->nb[1] + (size_t) co0 * dst->nb[2] + (size_t) a * sizeof(float);

                // no gap between rows: extend over the following rows of this image
                if (rows_contig) {
                    uint32_t r = 1;
                    while (oh + r < OH && vr + r < wc->vr1 && (vr + r) * SR < q_hi) {
                        r++;
                    }
                    b  = MIN(q_hi - base, (r - 1) * SR + OW);
                    vr += r - 1;
                }
                conv2d_dma_push(dmaq, dma_make_data(ddr, ob + (base + a - q_lo)), dst->nb[2], (size_t) OS * sizeof(float),
                                (size_t) (b - a) * sizeof(float), nco);
                nd[bi]++;
            }
            bi = bi + 1 == HTP_CONV2D_WIN_OBUF ? 0 : bi + 1;
        }
    }
}

// per-thread callback for one pipeline phase
static void conv2d_hmx_win_phase_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_win_context * wc = (const struct htp_conv2d_win_context *) data;

    if (wc->unpack_chunk >= 0) {
        conv2d_hmx_win_unpack(wc, nth, ith);
    }
    // weights of this rt block are packed before its first chunk
    if (wc->pack_chunk == 0) {
        const struct htp_tensor * kernel = wc->octx->src[0];
        const uint32_t            n_wt   = (wc->kp->window.fold ? kernel->ne[1] : kernel->ne[0] * kernel->ne[1]) * wc->kp->window.n_cc;
        for (uint32_t rt = ith; rt < wc->n_rt; rt += nth) {
            conv2d_hmx_rt_scale(wc->scales + rt * CONV2D_HMX_SCALES_N, wc->w_tiles + (size_t) rt * n_wt * HMX_FP16_TILE_N_ELMS, n_wt,
                                kernel->ne[0] * kernel->ne[1] * kernel->ne[2]);
        }
    }
    if (wc->pack_chunk >= 0) {
        conv2d_hmx_win_pack(wc, nth, ith);
    }
    dma_queue_flush(wc->octx->ctx->dma[ith]);
}

// zero out 2 act buffer regions in VTCM before packing
static void conv2d_hmx_win_zero_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_win_context * wc = (const struct htp_conv2d_win_context *) data;

    const size_t n   = 2 * (size_t) wc->kp->act_size / sizeof(float);
    const size_t per = hex_round_up((n + nth - 1) / nth, VLEN_FP32);
    const size_t i0  = MIN((size_t) ith * per, n);
    const size_t i1  = MIN(i0 + per, n);

    struct htp_thread_trace * tr = &wc->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) ith);
    if (i1 > i0) {
        hvx_splat_f32_a((uint8_t *) wc->act[0] + i0 * sizeof(float), 0.0f, i1 - i0);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) ith);
}

static void conv2d_hmx_win_run_phase(struct htp_conv2d_win_context * wc, int32_t pack, int32_t unpack) {
    wc->pack_chunk   = pack;
    wc->unpack_chunk = unpack;
    work_queue_run(wc->octx->ctx->work_queue, conv2d_hmx_win_phase_thread, wc, wc->kp->n_threads);
}


// live[i] = tile ts + i has a row that is written. Dead tiles lie inside the rows [OH, IHS) of an image.
static void conv2d_hmx_win_live(const struct htp_conv2d_win_context * wc, uint8_t * live, uint32_t ts, uint32_t nt) {
    const struct htp_conv2d_kernel_params * kp = wc->kp;

    const uint32_t SR  = kp->window.sr;
    const uint32_t IHS = kp->window.ihs;
    const uint32_t OH  = wc->octx->dst->ne[1];
    const uint32_t q0  = ts * 32;
    const uint32_t q1  = (ts + nt) * 32;

    memset(live, 1, nt);
    for (uint32_t n = fastdiv(fastdiv(q0, &kp->window.div_sr), &kp->window.div_ihs); n * IHS * SR < q1; n++) {
        const uint32_t g0 = MAX((n * IHS + OH) * SR, q0);
        const uint32_t g1 = MIN((n + 1) * IHS * SR, q1);
        const uint32_t t0 = (g0 + 31) / 32;
        const uint32_t t1 = g1 / 32;
        if (t1 > t0) {
            memset(live + (t0 - ts), 0, t1 - t0);
        }
    }
}

static int conv2d_hmx_win(struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp) {
    if (!octx->ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (kp->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * dst    = octx->dst;

    if (hex_round_up((size_t) kernel->ne[0] * kernel->ne[1] * sizeof(uint32_t), 128) + 2 * hex_round_up(kp->ct_chunk, 128) > octx->ctx->ddr_spad_size) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t KW  = kernel->ne[0];
    const uint32_t KH  = kernel->ne[1];
    const uint32_t OW  = dst->ne[0];
    const uint32_t OH  = dst->ne[1];
    const uint32_t N   = dst->ne[3];
    const uint32_t SR  = kp->window.sr;
    const uint32_t IHS = kp->window.ihs;
    const uint32_t d0  = octx->op_params[4];
    const uint32_t d1  = octx->op_params[5];

    // partition output rows across HTP devices - output rows [row_base, row_base + n_rows) of N * OH
    uint32_t row_base = 0;
    uint32_t n_rows   = N * OH;
    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        if (htp_tensor_mdev_data_aligned(dst) && (dst->nb[2] % HTP_TENSOR_MDEV_LINE_SIZE) == 0) {
            rows_per_chunk = HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(dst->nb[1], HTP_TENSOR_MDEV_LINE_SIZE);
            if (N > 1 && OH % rows_per_chunk != 0) {
                rows_per_chunk = 0;
            }
        }
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            n_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        n_rows   = range.count;
    }
    if (n_rows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t ra  = row_base;
    const uint32_t rb  = row_base + n_rows - 1;
    const uint32_t vr0 = (ra / OH) * IHS + ra % OH;
    const uint32_t vr1 = (rb / OH) * IHS + rb % OH + 1;

    uint8_t * base = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_conv2d_win_context wctx = {
        .octx     = octx,
        .kp       = kp,
        .scales   = (__fp16 *) base,
        .w_tiles  = VTCM_LAYOUT_PTR(__fp16, base, kp->off_w),
        .act      = { VTCM_LAYOUT_PTR(__fp16, base, kp->off_act), VTCM_LAYOUT_PTR(__fp16, base, kp->off_act + kp->act_size) },
        .out      = { VTCM_LAYOUT_PTR(__fp16, base, kp->off_out), VTCM_LAYOUT_PTR(__fp16, base, kp->off_out + kp->out_size) },
        .thr_base = VTCM_LAYOUT_PTR(uint8_t, base, kp->off_thr),
        .t_base   = (vr0 * SR) / 32,
        .t_end    = ((vr1 - 1) * SR + OW + 31) / 32,
        .vr0      = vr0,
        .vr1      = vr1,
    };

    // small Cin: croutons are built by scattering the staged values into zeroed croutons
    const uint32_t n_ch = octx->src[1]->ne[2] * (kp->window.fold ? KW : 1);
    if (!kp->cwhn && n_ch <= 2 * CONV2D_WIN_SCAT_MAX) {
        wctx.n_scat = (n_ch + 1) / 2;
        work_queue_run(octx->ctx->work_queue, conv2d_hmx_win_zero_thread, &wctx, kp->n_threads);
    }

    const int32_t  n_chunks = (int32_t) ((wctx.t_end - wctx.t_base + kp->ct_chunk - 1) / kp->ct_chunk);
    const uint32_t rem      = n_ch - (kp->window.n_cc - 1) * 32;

    // fold: kw is in the channels, taps are kh only
    const uint32_t TW   = kp->window.fold ? 1 : KW;
    uint32_t *     taps = (uint32_t *) octx->ctx->ddr_spad_base;
    for (uint32_t kh = 0; kh < KH; kh++) {
        for (uint32_t kw = 0; kw < TW; kw++) {
            taps[kh * TW + kw] = conv2d_hmx_win_act_off(kh * d1 * SR + kw * d0);
        }
    }

    wctx.live[0] = (uint8_t *) octx->ctx->ddr_spad_base + hex_round_up(KH * KW * sizeof(uint32_t), 128);
    wctx.live[1] = wctx.live[0] + hex_round_up(kp->ct_chunk, 128);

    struct htp_conv2d_win_job job = {
        .scales     = wctx.scales,
        .taps       = taps,
        .n_taps     = TW * KH,
        .out_stride = kp->ct_chunk * HMX_FP16_TILE_N_ELMS,
        .n_cc       = kp->window.n_cc,
        .plane      = kp->window.n_bc * HMX_FP16_TILE_SIZE,
        .rt_last    = kp->window.act_rt_last,
        .wr_last    = (hex_round_up(rem, 8) / 2) * VLEN - 1,
    };

    for (uint32_t rt0 = 0; rt0 < kp->n_row_tiles; rt0 += kp->rt_block) {
        wctx.rt0  = rt0;
        wctx.n_rt = MIN(kp->rt_block, kp->n_row_tiles - rt0);
        work_queue_run(octx->ctx->work_queue, conv2d_hmx_win_wpack_thread, &wctx, kp->n_threads);

        conv2d_hmx_win_run_phase(&wctx, 0, -1);

        for (int32_t c = 0; c < n_chunks; c++) {
            uint32_t ts;
            const uint32_t nt = conv2d_hmx_win_chunk_tiles(&wctx, c, &ts);
            conv2d_hmx_win_live(&wctx, wctx.live[c & 1], ts, nt);
            job.live    = wctx.live[c & 1];
            job.out     = wctx.out[c & 1];
            job.w       = wctx.w_tiles;
            job.act     = wctx.act[c & 1];
            job.n_rt    = wctx.n_rt;
            job.n_tiles = nt;
            hmx_queue_push(octx->ctx->hmx_queue, hmx_queue_make_desc(conv2d_hmx_win_worker, &job));

            const int32_t pack   = c + 1 < n_chunks ? c + 1 : -1;
            const int32_t unpack = c - 1;
            if (pack >= 0 || unpack >= 0) {
                conv2d_hmx_win_run_phase(&wctx, pack, unpack);
            }

            hmx_queue_pop(octx->ctx->hmx_queue);
        }

        conv2d_hmx_win_run_phase(&wctx, -1, n_chunks - 1);
    }

    return HTP_STATUS_OK;
}

int op_conv_2d(struct htp_ops_context * octx) {
    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];
    const struct htp_tensor * dst    = octx->dst;

    if ((kernel->type != HTP_TYPE_F16 && kernel->type != HTP_TYPE_F32) ||
        input->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "conv_2d: only F16/F32 kernel, F32 input and F32 dst are supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_conv2d_kernel_params * kparams =
        (const struct htp_conv2d_kernel_params *) octx->kernel_params;

    if (kparams->total_rows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->kernel_type == HTP_CONV2D_KERNEL_HMX_WINDOW) {
        return conv2d_hmx_win(octx, kparams);
    }
    if (kparams->kernel_type == HTP_CONV2D_KERNEL_HMX_TILED) {
        return conv2d_hmx_tiled(octx, kparams);
    }

    return HTP_STATUS_NO_SUPPORT;
}
