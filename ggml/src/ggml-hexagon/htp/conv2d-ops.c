#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "conv2d-ops.h"
#include "hex-dma.h"
#include "hex-profile.h"
#include "matmul-ops.h"
#include "hmx-mm-kernels-tiled.h"
#include "hmx-utils.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "hvx-sigmoid.h"

static const uint32_t conv2d_gather_rows_32x32[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,
    8 * 128,  9 * 128, 10 * 128, 11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128,
   16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128, 22 * 128, 23 * 128,
   24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

static const uint16_t conv2d_scatter_cols_32x32[64] __attribute__((aligned(VLEN))) = {
     0,    2,  128,  130,  256,  258,  384,  386,
   512,  514,  640,  642,  768,  770,  896,  898,
  1024, 1026, 1152, 1154, 1280, 1282, 1408, 1410,
  1536, 1538, 1664, 1666, 1792, 1794, 1920, 1922,
     0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,
};

#define CONV2D_PAIR_DECLS(prefix) \
    HVX_VectorPair prefix##01, prefix##23, prefix##45, prefix##67; \
    HVX_VectorPair prefix##89, prefix##1011, prefix##1213, prefix##1415; \
    HVX_VectorPair prefix##1617, prefix##1819, prefix##2021, prefix##2223; \
    HVX_VectorPair prefix##2425, prefix##2627, prefix##2829, prefix##3031

#define CONV2D_TRANSPOSE_FINISH(STORE_PAIR) do { \
    CONV2D_PAIR_DECLS(s); \
    s01   = Q6_W_vshuff_VVR(Q6_V_lo_W(d23),   Q6_V_lo_W(d01),   -8); \
    s23   = Q6_W_vshuff_VVR(Q6_V_hi_W(d23),   Q6_V_hi_W(d01),   -8); \
    s45   = Q6_W_vshuff_VVR(Q6_V_lo_W(d67),   Q6_V_lo_W(d45),   -8); \
    s67   = Q6_W_vshuff_VVR(Q6_V_hi_W(d67),   Q6_V_hi_W(d45),   -8); \
    s89   = Q6_W_vshuff_VVR(Q6_V_lo_W(d1011), Q6_V_lo_W(d89),   -8); \
    s1011 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1011), Q6_V_hi_W(d89),   -8); \
    s1213 = Q6_W_vshuff_VVR(Q6_V_lo_W(d1415), Q6_V_lo_W(d1213), -8); \
    s1415 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1415), Q6_V_hi_W(d1213), -8); \
    s1617 = Q6_W_vshuff_VVR(Q6_V_lo_W(d1819), Q6_V_lo_W(d1617), -8); \
    s1819 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1819), Q6_V_hi_W(d1617), -8); \
    s2021 = Q6_W_vshuff_VVR(Q6_V_lo_W(d2223), Q6_V_lo_W(d2021), -8); \
    s2223 = Q6_W_vshuff_VVR(Q6_V_hi_W(d2223), Q6_V_hi_W(d2021), -8); \
    s2425 = Q6_W_vshuff_VVR(Q6_V_lo_W(d2627), Q6_V_lo_W(d2425), -8); \
    s2627 = Q6_W_vshuff_VVR(Q6_V_hi_W(d2627), Q6_V_hi_W(d2425), -8); \
    s2829 = Q6_W_vshuff_VVR(Q6_V_lo_W(d3031), Q6_V_lo_W(d2829), -8); \
    s3031 = Q6_W_vshuff_VVR(Q6_V_hi_W(d3031), Q6_V_hi_W(d2829), -8); \
    d01   = Q6_W_vshuff_VVR(Q6_V_lo_W(s45),   Q6_V_lo_W(s01),   -16); \
    d23   = Q6_W_vshuff_VVR(Q6_V_hi_W(s45),   Q6_V_hi_W(s01),   -16); \
    d45   = Q6_W_vshuff_VVR(Q6_V_lo_W(s67),   Q6_V_lo_W(s23),   -16); \
    d67   = Q6_W_vshuff_VVR(Q6_V_hi_W(s67),   Q6_V_hi_W(s23),   -16); \
    d89   = Q6_W_vshuff_VVR(Q6_V_lo_W(s1213), Q6_V_lo_W(s89),   -16); \
    d1011 = Q6_W_vshuff_VVR(Q6_V_hi_W(s1213), Q6_V_hi_W(s89),   -16); \
    d1213 = Q6_W_vshuff_VVR(Q6_V_lo_W(s1415), Q6_V_lo_W(s1011), -16); \
    d1415 = Q6_W_vshuff_VVR(Q6_V_hi_W(s1415), Q6_V_hi_W(s1011), -16); \
    d1617 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2021), Q6_V_lo_W(s1617), -16); \
    d1819 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2021), Q6_V_hi_W(s1617), -16); \
    d2021 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2223), Q6_V_lo_W(s1819), -16); \
    d2223 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2223), Q6_V_hi_W(s1819), -16); \
    d2425 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2829), Q6_V_lo_W(s2425), -16); \
    d2627 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2829), Q6_V_hi_W(s2425), -16); \
    d2829 = Q6_W_vshuff_VVR(Q6_V_lo_W(s3031), Q6_V_lo_W(s2627), -16); \
    d3031 = Q6_W_vshuff_VVR(Q6_V_hi_W(s3031), Q6_V_hi_W(s2627), -16); \
    s01   = Q6_W_vshuff_VVR(Q6_V_lo_W(d89),   Q6_V_lo_W(d01),   -32); \
    s23   = Q6_W_vshuff_VVR(Q6_V_hi_W(d89),   Q6_V_hi_W(d01),   -32); \
    s45   = Q6_W_vshuff_VVR(Q6_V_lo_W(d1011), Q6_V_lo_W(d23),   -32); \
    s67   = Q6_W_vshuff_VVR(Q6_V_hi_W(d1011), Q6_V_hi_W(d23),   -32); \
    s89   = Q6_W_vshuff_VVR(Q6_V_lo_W(d1213), Q6_V_lo_W(d45),   -32); \
    s1011 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1213), Q6_V_hi_W(d45),   -32); \
    s1213 = Q6_W_vshuff_VVR(Q6_V_lo_W(d1415), Q6_V_lo_W(d67),   -32); \
    s1415 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1415), Q6_V_hi_W(d67),   -32); \
    s1617 = Q6_W_vshuff_VVR(Q6_V_lo_W(d2425), Q6_V_lo_W(d1617), -32); \
    s1819 = Q6_W_vshuff_VVR(Q6_V_hi_W(d2425), Q6_V_hi_W(d1617), -32); \
    s2021 = Q6_W_vshuff_VVR(Q6_V_lo_W(d2627), Q6_V_lo_W(d1819), -32); \
    s2223 = Q6_W_vshuff_VVR(Q6_V_hi_W(d2627), Q6_V_hi_W(d1819), -32); \
    s2425 = Q6_W_vshuff_VVR(Q6_V_lo_W(d2829), Q6_V_lo_W(d2021), -32); \
    s2627 = Q6_W_vshuff_VVR(Q6_V_hi_W(d2829), Q6_V_hi_W(d2021), -32); \
    s2829 = Q6_W_vshuff_VVR(Q6_V_lo_W(d3031), Q6_V_lo_W(d2223), -32); \
    s3031 = Q6_W_vshuff_VVR(Q6_V_hi_W(d3031), Q6_V_hi_W(d2223), -32); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s1617), Q6_V_lo_W(s01), -64); STORE_PAIR( 0, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s1617), Q6_V_hi_W(s01), -64); STORE_PAIR( 1, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s1819), Q6_V_lo_W(s23), -64); STORE_PAIR( 2, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s1819), Q6_V_hi_W(s23), -64); STORE_PAIR( 3, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2021), Q6_V_lo_W(s45), -64); STORE_PAIR( 4, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2021), Q6_V_hi_W(s45), -64); STORE_PAIR( 5, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2223), Q6_V_lo_W(s67), -64); STORE_PAIR( 6, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2223), Q6_V_hi_W(s67), -64); STORE_PAIR( 7, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2425), Q6_V_lo_W(s89), -64); STORE_PAIR( 8, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2425), Q6_V_hi_W(s89), -64); STORE_PAIR( 9, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2627), Q6_V_lo_W(s1011), -64); STORE_PAIR(10, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2627), Q6_V_hi_W(s1011), -64); STORE_PAIR(11, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s2829), Q6_V_lo_W(s1213), -64); STORE_PAIR(12, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s2829), Q6_V_hi_W(s1213), -64); STORE_PAIR(13, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_lo_W(s3031), Q6_V_lo_W(s1415), -64); STORE_PAIR(14, d01); \
    d01 = Q6_W_vshuff_VVR(Q6_V_hi_W(s3031), Q6_V_hi_W(s1415), -64); STORE_PAIR(15, d01); \
} while (0)

static __attribute__((noinline)) void conv2d_pack_tile_f32_to_f16(
        __fp16 * dst,
        const float * src,
        size_t channel_stride) {
    CONV2D_PAIR_DECLS(d);
#define LOAD_CHANNEL(c) hvx_vmemu(src + (size_t) (c) * channel_stride)
    d01   = Q6_W_vshuff_VVR(LOAD_CHANNEL( 1), LOAD_CHANNEL( 0), -4);
    d23   = Q6_W_vshuff_VVR(LOAD_CHANNEL( 3), LOAD_CHANNEL( 2), -4);
    d45   = Q6_W_vshuff_VVR(LOAD_CHANNEL( 5), LOAD_CHANNEL( 4), -4);
    d67   = Q6_W_vshuff_VVR(LOAD_CHANNEL( 7), LOAD_CHANNEL( 6), -4);
    d89   = Q6_W_vshuff_VVR(LOAD_CHANNEL( 9), LOAD_CHANNEL( 8), -4);
    d1011 = Q6_W_vshuff_VVR(LOAD_CHANNEL(11), LOAD_CHANNEL(10), -4);
    d1213 = Q6_W_vshuff_VVR(LOAD_CHANNEL(13), LOAD_CHANNEL(12), -4);
    d1415 = Q6_W_vshuff_VVR(LOAD_CHANNEL(15), LOAD_CHANNEL(14), -4);
    d1617 = Q6_W_vshuff_VVR(LOAD_CHANNEL(17), LOAD_CHANNEL(16), -4);
    d1819 = Q6_W_vshuff_VVR(LOAD_CHANNEL(19), LOAD_CHANNEL(18), -4);
    d2021 = Q6_W_vshuff_VVR(LOAD_CHANNEL(21), LOAD_CHANNEL(20), -4);
    d2223 = Q6_W_vshuff_VVR(LOAD_CHANNEL(23), LOAD_CHANNEL(22), -4);
    d2425 = Q6_W_vshuff_VVR(LOAD_CHANNEL(25), LOAD_CHANNEL(24), -4);
    d2627 = Q6_W_vshuff_VVR(LOAD_CHANNEL(27), LOAD_CHANNEL(26), -4);
    d2829 = Q6_W_vshuff_VVR(LOAD_CHANNEL(29), LOAD_CHANNEL(28), -4);
    d3031 = Q6_W_vshuff_VVR(LOAD_CHANNEL(31), LOAD_CHANNEL(30), -4);
#undef LOAD_CHANNEL
#define PACK_STORE_PAIR(i, p) \
    ((HVX_Vector *) dst)[i] = hvx_vec_f32_to_f16_shuff(Q6_V_lo_W(p), Q6_V_hi_W(p))
    CONV2D_TRANSPOSE_FINISH(PACK_STORE_PAIR);
#undef PACK_STORE_PAIR
}

static __attribute__((noinline)) void conv2d_store_tile_f16_to_f32(
        float * dst,
        const __fp16 * src,
        size_t channel_stride) {
    CONV2D_PAIR_DECLS(d);
    const HVX_Vector * tiles = (const HVX_Vector *) src;
    HVX_VectorPair rows;
#define LOAD_ROW_PAIR(i, dpair) do { \
    rows = hvx_vec_f16_to_f32_shuff(tiles[i]); \
    dpair = Q6_W_vshuff_VVR(Q6_V_hi_W(rows), Q6_V_lo_W(rows), -4); \
} while (0)
    LOAD_ROW_PAIR( 0, d01);
    LOAD_ROW_PAIR( 1, d23);
    LOAD_ROW_PAIR( 2, d45);
    LOAD_ROW_PAIR( 3, d67);
    LOAD_ROW_PAIR( 4, d89);
    LOAD_ROW_PAIR( 5, d1011);
    LOAD_ROW_PAIR( 6, d1213);
    LOAD_ROW_PAIR( 7, d1415);
    LOAD_ROW_PAIR( 8, d1617);
    LOAD_ROW_PAIR( 9, d1819);
    LOAD_ROW_PAIR(10, d2021);
    LOAD_ROW_PAIR(11, d2223);
    LOAD_ROW_PAIR(12, d2425);
    LOAD_ROW_PAIR(13, d2627);
    LOAD_ROW_PAIR(14, d2829);
    LOAD_ROW_PAIR(15, d3031);
#undef LOAD_ROW_PAIR
#define OUTPUT_STORE_PAIR(i, p) do { \
    hvx_vmem(dst + (size_t) (2 * (i) + 0) * channel_stride) = Q6_V_lo_W(p); \
    hvx_vmem(dst + (size_t) (2 * (i) + 1) * channel_stride) = Q6_V_hi_W(p); \
} while (0)
    CONV2D_TRANSPOSE_FINISH(OUTPUT_STORE_PAIR);
#undef OUTPUT_STORE_PAIR
}

struct conv2d_prepare_state {
    struct htp_context * ctx;
    float * x;
    const float * src;
    __fp16 * a;
    uint32_t iw;
    uint32_t ih;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t halo_w;
    uint32_t halo_h;
    uint32_t sx_begin;
    uint32_t sy_begin;
    uint32_t copy_w;
    uint32_t copy_h;
    uint32_t dx;
    uint32_t dy;
    int32_t sx0;
    int32_t sy0;
    uint32_t kh;
    uint32_t kw;
    uint32_t ic_blocks;
    uint32_t n_groups;
    uint32_t group_channels;
    uint32_t k_tiles;
    uint32_t activation_k_tiles;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t x_tiles;
    uint32_t m_tiles;
    bool clear;
    bool wait_weight_dma;
    bool upscale2;
    bool direct_upscale2;
    bool input_silu;
    bool input_f16;
};

static inline HVX_Vector conv2d_silu(HVX_Vector x) {
    const HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(
        x, hvx_vec_splat_f32(1.0f), hvx_vec_splat_f32(87.0f),
        hvx_vec_splat_f32(-87.0f));
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, sigmoid));
}

static void conv2d_silu_inplace(float * data, uint32_t n) {
    HVX_Vector * vectors = (HVX_Vector *) data;
    const uint32_t nvec = n / 32u;
    uint32_t i = 0;
    for (; i + 3u < nvec; i += 4u) {
        const HVX_Vector x0 = vectors[i + 0u];
        const HVX_Vector x1 = vectors[i + 1u];
        const HVX_Vector x2 = vectors[i + 2u];
        const HVX_Vector x3 = vectors[i + 3u];
        vectors[i + 0u] = conv2d_silu(x0);
        vectors[i + 1u] = conv2d_silu(x1);
        vectors[i + 2u] = conv2d_silu(x2);
        vectors[i + 3u] = conv2d_silu(x3);
    }
    for (; i < nvec; ++i) {
        vectors[i] = conv2d_silu(vectors[i]);
    }
    const uint32_t tail = n % 32u;
    if (tail) {
        hvx_vec_store_a(vectors + nvec, tail * sizeof(float),
                        conv2d_silu(vectors[nvec]));
    }
}

static void conv2d_silu_f16_inplace(__fp16 * data, uint32_t n) {
    HVX_Vector * vectors = (HVX_Vector *) data;
    const uint32_t nvec = n / 64u;
    for (uint32_t i = 0; i < nvec; ++i) {
        const HVX_Vector x = vectors[i];
        vectors[i] = hvx_vec_mul_f16_f16(x, hvx_vec_fast_sigmoid_f16(x));
    }
    const uint32_t tail = n % 64u;
    if (tail) {
        const HVX_Vector x = vectors[nvec];
        hvx_vec_store_a(vectors + nvec, tail * sizeof(__fp16),
                        hvx_vec_mul_f16_f16(x, hvx_vec_fast_sigmoid_f16(x)));
    }
}

static inline HVX_Vector conv2d_join_f16_halves(HVX_Vector lo, HVX_Vector hi) {
    const HVX_VectorPred lower_half = Q6_Q_vsetq_R(64);
    return Q6_V_vmux_QVV(lower_half, lo, Q6_V_vror_VR(hi, 64));
}

static inline HVX_Vector conv2d_pack_f16_pair(HVX_Vector row0, HVX_Vector row1) {
    return Q6_Vh_vshuff_Vh(conv2d_join_f16_halves(row0, row1));
}

static inline void conv2d_upscale2_row(float * dst, const float * src,
                                       uint32_t logical_x, uint32_t count) {
    while (count > 0) {
        const uint32_t take = count < 32u ? count : 32u;
        const HVX_Vector input = hvx_vmemu(src + logical_x / 2u);
        const HVX_VectorPair dup = Q6_W_vshuff_VVR(input, input, -4);
        HVX_Vector output = Q6_V_lo_W(dup);
        if (logical_x & 1u) {
            output = Q6_V_valign_VVR(Q6_V_hi_W(dup), output, sizeof(float));
        }
        hvx_vec_store_u(dst, take * sizeof(float), output);
        logical_x += take;
        dst += take;
        count -= take;
    }
}

static inline void conv2d_upscale2_row_f16(__fp16 * dst, const __fp16 * src,
                                           uint32_t logical_x, uint32_t count) {
    while (count > 0) {
        const uint32_t take = count < 64u ? count : 64u;
        const HVX_Vector input = hvx_vmemu(src + logical_x / 2u);
        const HVX_VectorPair dup = Q6_W_vshuff_VVR(input, input, -2);
        HVX_Vector output = Q6_V_lo_W(dup);
        if (logical_x & 1u) {
            output = Q6_V_valign_VVR(Q6_V_hi_W(dup), output, sizeof(__fp16));
        }
        hvx_vec_store_u(dst, take * sizeof(__fp16), output);
        logical_x += take;
        dst += take;
        count -= take;
    }
}

static void conv2d_push_halo_group(
        const struct conv2d_prepare_state * st,
        dma_queue * q,
        uint32_t group,
        void * x_ptr) {
    const size_t channel_elms = (size_t) st->halo_w * st->halo_h;
    if (st->input_f16) {
        __fp16 * x = (__fp16 *) x_ptr;
        if (st->clear) {
            memset(x, 0, st->group_channels * channel_elms * sizeof(__fp16));
        }
        const uint32_t channel_first = group * st->group_channels;
        if (st->upscale2) {
            for (uint32_t c = 0; c < st->group_channels; ++c) {
                __fp16 * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
                const __fp16 * channel = (const __fp16 *) st->src +
                    (size_t) (channel_first + c) * st->src_h * st->src_w;
                for (uint32_t y = 0; y < st->copy_h; ++y) {
                    const uint32_t logical_y = st->sy_begin + y;
                    const __fp16 * s = channel + (size_t) (logical_y / 2u) * st->src_w;
                    conv2d_upscale2_row_f16(d + (size_t) y * st->halo_w, s,
                                            st->sx_begin, st->copy_w);
                }
            }
            return;
        }
        for (uint32_t c = 0; c < st->group_channels; ++c) {
            __fp16 * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
            const __fp16 * s = (const __fp16 *) st->src +
                ((size_t) (channel_first + c) * st->ih + st->sy_begin) * st->iw +
                st->sx_begin;
            while (!dma_queue_push(q, dma_make_ptr(d, s),
                                   st->halo_w * sizeof(__fp16), st->iw * sizeof(__fp16),
                                   st->copy_w * sizeof(__fp16), st->copy_h)) {
                dma_queue_flush(q);
            }
        }
        return;
    }
    float * x = (float *) x_ptr;
    if (st->clear) {
        memset(x, 0, st->group_channels * channel_elms * sizeof(float));
    }
    const uint32_t channel_first = group * st->group_channels;
    if (st->upscale2) {
        for (uint32_t c = 0; c < st->group_channels; ++c) {
            float * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
            const float * channel = st->src +
                (size_t) (channel_first + c) * st->src_h * st->src_w;
            for (uint32_t y = 0; y < st->copy_h; ++y) {
                const uint32_t logical_y = st->sy_begin + y;
                const float * s = channel + (size_t) (logical_y / 2u) * st->src_w;
                conv2d_upscale2_row(d + (size_t) y * st->halo_w, s,
                                    st->sx_begin, st->copy_w);
            }
        }
        return;
    }
    for (uint32_t c = 0; c < st->group_channels; ++c) {
        float * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
        const float * s = st->src + ((size_t) (channel_first + c) * st->ih + st->sy_begin) * st->iw + st->sx_begin;
        while (!dma_queue_push(q, dma_make_ptr(d, s),
                               st->halo_w * sizeof(float), st->iw * sizeof(float),
                               st->copy_w * sizeof(float), st->copy_h)) {
            dma_queue_flush(q);
        }
    }
}

static inline void conv2d_store_activation_triplet(
        const struct conv2d_prepare_state * st,
        uint32_t source_mt,
        uint32_t icb,
        uint32_t cp,
        HVX_Vector p0,
        HVX_Vector p1,
    HVX_Vector p2) {
    HVX_Vector * tile0 = (HVX_Vector *)
        (st->a + ((size_t) source_mt * st->activation_k_tiles +
         0 * st->ic_blocks + icb) * HTP_MM_HMX_TILE_N_ELMS);
    HVX_Vector * tile1 = (HVX_Vector *)
        (st->a + ((size_t) source_mt * st->activation_k_tiles +
         1 * st->ic_blocks + icb) * HTP_MM_HMX_TILE_N_ELMS);
    HVX_Vector * tile2 = (HVX_Vector *)
        (st->a + ((size_t) source_mt * st->activation_k_tiles +
         2 * st->ic_blocks + icb) * HTP_MM_HMX_TILE_N_ELMS);
    tile0[cp] = p0;
    tile1[cp] = p1;
    tile2[cp] = p2;
}

static void conv2d_pack_activation_group_f16(
        const struct conv2d_prepare_state * st,
        uint32_t group,
        __fp16 * x) {
    const size_t channel_elms = (size_t) st->halo_w * st->halo_h;
    const uint32_t channel_first = group * st->group_channels;
    const uint32_t icb = channel_first / 32u;
    const uint32_t cp_first = (channel_first % 32u) / 2u;
    const uint32_t n_pairs = st->group_channels / 2u;
    if (st->input_silu) {
        conv2d_silu_f16_inplace(x, st->group_channels * channel_elms);
    }
    if (st->kw == 3) {
        for (uint32_t cp = 0; cp < n_pairs; ++cp) {
            for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
                const __fp16 * row = x + (size_t) sy * st->halo_w;
                const __fp16 * row0 = row + (2u * cp + 0u) * channel_elms;
                const __fp16 * row1 = row + (2u * cp + 1u) * channel_elms;
                for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                    const uint32_t x0 = tx * HTP_CONV2D_TILE_W;
                    conv2d_store_activation_triplet(
                        st, sy * st->x_tiles + tx, icb, cp_first + cp,
                        conv2d_pack_f16_pair(hvx_vmemu(row0 + x0 + 0u),
                                             hvx_vmemu(row1 + x0 + 0u)),
                        conv2d_pack_f16_pair(hvx_vmemu(row0 + x0 + 1u),
                                             hvx_vmemu(row1 + x0 + 1u)),
                        conv2d_pack_f16_pair(hvx_vmemu(row0 + x0 + 2u),
                                             hvx_vmemu(row1 + x0 + 2u)));
                }
            }
        }
        return;
    }
    for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
        for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
            const uint32_t mt = oy * st->x_tiles + tx;
            HVX_Vector * tile = (HVX_Vector *)
                (st->a + ((size_t) mt * st->k_tiles + icb) *
                 HTP_MM_HMX_TILE_N_ELMS);
            const __fp16 * src = x + (size_t) oy * st->halo_w +
                                  tx * HTP_CONV2D_TILE_W;
            for (uint32_t cp = 0; cp < n_pairs; ++cp) {
                tile[cp_first + cp] = conv2d_pack_f16_pair(
                    hvx_vmemu(src + (2u * cp + 0u) * channel_elms),
                    hvx_vmemu(src + (2u * cp + 1u) * channel_elms));
            }
        }
    }
}

static inline HVX_Vector conv2d_upscale2_shift(
        HVX_Vector lo,
        HVX_Vector hi,
        uint32_t shift_words) {
    return shift_words == 0 ? lo
                            : Q6_V_valign_VVR(hi, lo, shift_words * sizeof(float));
}

static void conv2d_prepare_upscale_group_direct(
        const struct conv2d_prepare_state * st,
        uint32_t group) {
    const uint32_t channel_first = group * st->group_channels;
    const uint32_t icb = channel_first / 32;
    const uint32_t cp_first = (channel_first % 32) / 2;
    const uint32_t n_pairs = st->group_channels / 2;
    const uint32_t x_parity = (uint32_t) st->sx0 & 1u;

    for (uint32_t cp = 0; cp < n_pairs; ++cp) {
        const uint32_t c0 = channel_first + 2 * cp;
        const uint32_t c1 = c0 + 1;
        for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
            const uint32_t iy = (uint32_t) (st->sy0 + (int32_t) sy) / 2u;
            const float * row0 = st->src + ((size_t) c0 * st->src_h + iy) * st->src_w;
            const float * row1 = st->src + ((size_t) c1 * st->src_h + iy) * st->src_w;
            for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                const uint32_t logical_x = (uint32_t) st->sx0 + tx * HTP_CONV2D_TILE_W;
                const uint32_t ix = logical_x / 2u;
                const HVX_Vector in0 = hvx_vmemu(row0 + ix);
                const HVX_Vector in1 = hvx_vmemu(row1 + ix);
                const HVX_VectorPair dup0 = Q6_W_vshuff_VVR(in0, in0, -4);
                const HVX_VectorPair dup1 = Q6_W_vshuff_VVR(in1, in1, -4);
                HVX_Vector x00 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 0u);
                HVX_Vector x01 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 0u);
                HVX_Vector x10 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 1u);
                HVX_Vector x11 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 1u);
                HVX_Vector x20 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 2u);
                HVX_Vector x21 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 2u);
                if (st->input_silu) {
                    x00 = conv2d_silu(x00);
                    x01 = conv2d_silu(x01);
                    x10 = conv2d_silu(x10);
                    x11 = conv2d_silu(x11);
                    x20 = conv2d_silu(x20);
                    x21 = conv2d_silu(x21);
                }
                conv2d_store_activation_triplet(
                    st, sy * st->x_tiles + tx, icb, cp_first + cp,
                    hvx_vec_f32_to_f16_shuff(x00, x01),
                    hvx_vec_f32_to_f16_shuff(x10, x11),
                    hvx_vec_f32_to_f16_shuff(x20, x21));
            }
        }
    }
}

static inline HVX_Vector conv2d_upscale2_shift_f16(
        HVX_Vector lo,
        HVX_Vector hi,
        uint32_t shift) {
    return shift == 0 ? lo : Q6_V_valign_VVR(hi, lo, shift * sizeof(__fp16));
}

static void conv2d_prepare_upscale_group_direct_f16(
        const struct conv2d_prepare_state * st,
        uint32_t group) {
    const uint32_t channel_first = group * st->group_channels;
    const uint32_t icb = channel_first / 32u;
    const uint32_t cp_first = (channel_first % 32u) / 2u;
    const uint32_t n_pairs = st->group_channels / 2u;
    const uint32_t x_parity = (uint32_t) st->sx0 & 1u;
    const __fp16 * src = (const __fp16 *) st->src;

    for (uint32_t cp = 0; cp < n_pairs; ++cp) {
        const uint32_t c0 = channel_first + 2u * cp;
        const uint32_t c1 = c0 + 1u;
        for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
            const uint32_t iy = (uint32_t) (st->sy0 + (int32_t) sy) / 2u;
            const __fp16 * row0 = src + ((size_t) c0 * st->src_h + iy) * st->src_w;
            const __fp16 * row1 = src + ((size_t) c1 * st->src_h + iy) * st->src_w;
            for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                const uint32_t logical_x = (uint32_t) st->sx0 + tx * HTP_CONV2D_TILE_W;
                const uint32_t ix = logical_x / 2u;
                const HVX_Vector in0 = hvx_vmemu(row0 + ix);
                const HVX_Vector in1 = hvx_vmemu(row1 + ix);
                const HVX_VectorPair dup0 = Q6_W_vshuff_VVR(in0, in0, -2);
                const HVX_VectorPair dup1 = Q6_W_vshuff_VVR(in1, in1, -2);
                HVX_Vector x00 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 0u);
                HVX_Vector x01 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 0u);
                HVX_Vector x10 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 1u);
                HVX_Vector x11 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 1u);
                HVX_Vector x20 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 2u);
                HVX_Vector x21 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 2u);
                if (st->input_silu) {
                    x00 = hvx_vec_mul_f16_f16(x00, hvx_vec_fast_sigmoid_f16(x00));
                    x01 = hvx_vec_mul_f16_f16(x01, hvx_vec_fast_sigmoid_f16(x01));
                    x10 = hvx_vec_mul_f16_f16(x10, hvx_vec_fast_sigmoid_f16(x10));
                    x11 = hvx_vec_mul_f16_f16(x11, hvx_vec_fast_sigmoid_f16(x11));
                    x20 = hvx_vec_mul_f16_f16(x20, hvx_vec_fast_sigmoid_f16(x20));
                    x21 = hvx_vec_mul_f16_f16(x21, hvx_vec_fast_sigmoid_f16(x21));
                }
                conv2d_store_activation_triplet(
                    st, sy * st->x_tiles + tx, icb, cp_first + cp,
                    conv2d_pack_f16_pair(x00, x01),
                    conv2d_pack_f16_pair(x10, x11),
                    conv2d_pack_f16_pair(x20, x21));
            }
        }
    }
}

static void conv2d_prepare_activation_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_prepare_state * st = (struct conv2d_prepare_state *) data;
    const uint32_t group_first = fastdiv(st->n_groups * ith, &st->ctx->n_threads_div);
    const uint32_t group_last = fastdiv(st->n_groups * (ith + 1), &st->ctx->n_threads_div);
    const size_t channel_elms = (size_t) st->halo_w * st->halo_h;
    const uint32_t element_size = st->input_f16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t x_slot_first = 2 * ith;
    uint8_t * x_slots = (uint8_t *) st->x +
        (size_t) x_slot_first * st->group_channels * channel_elms * element_size;
    dma_queue * q = st->ctx->dma[ith];
    if (ith == 0 && st->wait_weight_dma) {
        dma_queue_pop(q);
    }
    if (group_first == group_last) {
        return;
    }
    if (st->direct_upscale2) {
        for (uint32_t group = group_first; group < group_last; ++group) {
            if (st->input_f16) {
                conv2d_prepare_upscale_group_direct_f16(st, group);
            } else {
                conv2d_prepare_upscale_group_direct(st, group);
            }
        }
        return;
    }

    if (st->copy_w != 0 && st->copy_h != 0) {
        conv2d_push_halo_group(st, q, group_first, x_slots);
        if (group_first + 1 < group_last) {
            conv2d_push_halo_group(st, q, group_first + 1,
                                   x_slots + st->group_channels * channel_elms * element_size);
        }
    } else if (st->clear) {
        memset(x_slots, 0, (group_last - group_first < 2 ? group_last - group_first : 2) *
                            st->group_channels * channel_elms * element_size);
    }

    for (uint32_t group = group_first; group < group_last; ++group) {
        if (!st->upscale2 && st->copy_w != 0 && st->copy_h != 0) {
            for (uint32_t c = 0; c < st->group_channels; ++c) {
                dma_queue_pop(q);
            }
        }

        uint8_t * x_ptr = x_slots + (size_t) ((group - group_first) & 1u) *
                                   st->group_channels * channel_elms * element_size;
        if (st->input_f16) {
            conv2d_pack_activation_group_f16(st, group, (__fp16 *) x_ptr);
        } else {
            const uint32_t channel_first = group * st->group_channels;
            const uint32_t icb = channel_first / 32;
            const uint32_t cp_first = (channel_first % 32) / 2;
            const uint32_t n_pairs = st->group_channels / 2;
            float * x = (float *) x_ptr;
            if (st->input_silu) {
                conv2d_silu_inplace(x, st->group_channels * channel_elms);
            }
            if (st->kw == 3) {
            for (uint32_t cp = 0; cp < n_pairs; ++cp) {
                for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
                    const float * row = x + (size_t) sy * st->halo_w;
                        const float * row0 = row + (2 * cp + 0) * channel_elms;
                        const float * row1 = row + (2 * cp + 1) * channel_elms;
                        HVX_Vector lo0 = hvx_vmemu(row0);
                        HVX_Vector lo1 = hvx_vmemu(row1);
                        for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                            HVX_Vector hi0 = hvx_vmemu(
                                row0 + (tx + 1) * HTP_CONV2D_TILE_W);
                            HVX_Vector hi1 = hvx_vmemu(
                                row1 + (tx + 1) * HTP_CONV2D_TILE_W);
                            const HVX_Vector p0 = hvx_vec_f32_to_f16_shuff(lo0, lo1);
                            const HVX_Vector p1 = hvx_vec_f32_to_f16_shuff(
                                Q6_V_valign_VVR(hi0, lo0, sizeof(float)),
                                Q6_V_valign_VVR(hi1, lo1, sizeof(float)));
                            const HVX_Vector p2 = hvx_vec_f32_to_f16_shuff(
                                Q6_V_valign_VVR(hi0, lo0, 2 * sizeof(float)),
                                Q6_V_valign_VVR(hi1, lo1, 2 * sizeof(float)));
                            conv2d_store_activation_triplet(
                                st, sy * st->x_tiles + tx, icb,
                                cp_first + cp, p0, p1, p2);
                            lo0 = hi0;
                            lo1 = hi1;
                        }
                    }
            }
            } else {
            for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
                for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                    const uint32_t mt = oy * st->x_tiles + tx;
                    HVX_Vector * tile = (HVX_Vector *)
                        (st->a + ((size_t) mt * st->k_tiles + icb) *
                         HTP_MM_HMX_TILE_N_ELMS);
                    const float * src = x + (size_t) oy * st->halo_w +
                                        tx * HTP_CONV2D_TILE_W;
                    for (uint32_t cp = 0; cp < n_pairs; ++cp) {
                        HVX_Vector row0 =
                            hvx_vmemu(src + (2 * cp + 0) * channel_elms);
                        HVX_Vector row1 =
                            hvx_vmemu(src + (2 * cp + 1) * channel_elms);
                        tile[cp_first + cp] =
                            hvx_vec_f32_to_f16_shuff(row0, row1);
                    }
                }
            }
            }
        }
        if (st->copy_w != 0 && st->copy_h != 0 && group + 2 < group_last) {
            conv2d_push_halo_group(st, q, group + 2, x_ptr);
        }
    }
}

struct conv2d_store_state {
    struct htp_context * ctx;
    const __fp16 * c;
    float * dst;
    const float * bias;
    const float * residual;
    uint32_t ow;
    uint32_t oh;
    uint32_t oc;
    uint32_t n_tiles;
    uint32_t x0;
    uint32_t y0;
    uint32_t tile_w;
    uint32_t valid_w;
    uint32_t tile_h;
    uint32_t x_tiles;
    uint32_t m_tiles;
    bool final_block;
    atomic_uint * next_pair;
    uint32_t flags;
    bool output_f16;
};

static void conv2d_store_output_f16_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_store_state * st = (struct conv2d_store_state *) data;
    const uint32_t n_pairs = (st->oc + 1) / 2;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_VectorPred all = Q6_Q_vcmp_eq_VbVb(zero, zero);
    const uint32_t pair_chunk = 1;
    for (;;) {
        const uint32_t pair_first = atomic_fetch_add_explicit(
            st->next_pair, pair_chunk, memory_order_relaxed);
        if (pair_first >= n_pairs) {
            break;
        }
        const uint32_t pair_last = hex_smin(pair_first + pair_chunk, n_pairs);
        for (uint32_t pair = pair_first; pair < pair_last; ++pair) {
            const uint32_t channel = pair * 2u;
            const uint32_t nt = channel / 32u;
            const uint32_t rp = (channel % 32u) / 2u;
            const HVX_Vector bias0 = hvx_vec_splat_f16(
                (st->flags & HTP_CONV2D_BIAS) ? st->bias[channel] : 0.0f);
            const HVX_Vector bias1 = hvx_vec_splat_f16(
                (st->flags & HTP_CONV2D_BIAS) && channel + 1u < st->oc
                    ? st->bias[channel + 1u] : 0.0f);
            const HVX_Vector packed_bias = Q6_Vh_vshuff_Vh(
                conv2d_join_f16_halves(bias0, bias1));
            for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
                for (uint32_t tx = 0; tx < st->x_tiles; tx += 2u) {
                    HVX_Vector planar[2];
                    const uint32_t count = hex_smin(st->x_tiles - tx, 2u);
                    const uint32_t valid_count = hex_smin(
                        st->valid_w - tx * HTP_CONV2D_TILE_W,
                        count * HTP_CONV2D_TILE_W);
                    for (uint32_t j = 0; j < count; ++j) {
                        const uint32_t mt = oy * st->x_tiles + tx + j;
                        const HVX_Vector * tile = (const HVX_Vector *)
                            (st->c + ((size_t) nt * st->m_tiles + mt) *
                             HTP_MM_HMX_TILE_N_ELMS);
                        HVX_Vector output = hvx_vec_add_f16_f16(tile[rp], packed_bias);
                        __fp16 * dst0 = (__fp16 *) st->dst +
                            ((size_t) channel * st->oh + st->y0 + oy) * st->ow +
                            st->x0 + (tx + j) * HTP_CONV2D_TILE_W;
                        if (st->flags & HTP_CONV2D_RESIDUAL) {
                            const __fp16 * residual0 = (const __fp16 *) st->residual +
                                (dst0 - (__fp16 *) st->dst);
                            const __fp16 * residual1 = residual0 + (size_t) st->oh * st->ow;
                            const HVX_Vector packed_residual = Q6_Vh_vshuff_Vh(
                                conv2d_join_f16_halves(
                                    hvx_vmemu(residual0), hvx_vmemu(residual1)));
                            output = hvx_vec_add_f16_f16(output, packed_residual);
                        }
                        planar[j] = Q6_Vh_vdeal_Vh(output);
                    }
                    __fp16 * dst0 = (__fp16 *) st->dst +
                        ((size_t) channel * st->oh + st->y0 + oy) * st->ow +
                        st->x0 + tx * HTP_CONV2D_TILE_W;
                    const HVX_Vector row0 = count == 2u
                        ? conv2d_join_f16_halves(planar[0], planar[1]) : planar[0];
                    if (valid_count == 2u * HTP_CONV2D_TILE_W &&
                        ((uintptr_t) dst0 & 127u) == 0) {
                        Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst0, row0);
                    } else {
                        hvx_vec_store_u(dst0, valid_count * sizeof(__fp16), row0);
                    }
                    if (channel + 1u < st->oc) {
                        __fp16 * dst1 = dst0 + (size_t) st->oh * st->ow;
                        const HVX_Vector p0 = Q6_V_vror_VR(planar[0], 64);
                        const HVX_Vector p1 = count == 2u
                            ? Q6_V_vror_VR(planar[1], 64) : p0;
                        const HVX_Vector row1 = count == 2u
                            ? conv2d_join_f16_halves(p0, p1) : p0;
                        if (valid_count == 2u * HTP_CONV2D_TILE_W &&
                            ((uintptr_t) dst1 & 127u) == 0) {
                            Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst1, row1);
                        } else {
                            hvx_vec_store_u(dst1, valid_count * sizeof(__fp16), row1);
                        }
                    }
                }
            }
        }
    }
    if (st->final_block) {
        asm volatile("syncht" ::: "memory");
    }
}

static void conv2d_store_output_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_store_state * st = (struct conv2d_store_state *) data;
    if (st->output_f16) {
        conv2d_store_output_f16_worker(nth, ith, data);
        return;
    }
    const uint32_t n_pairs = (st->oc + 1) / 2;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_VectorPred all = Q6_Q_vcmp_eq_VbVb(zero, zero);

    const uint32_t pair_chunk = 4;
    for (;;) {
        const uint32_t pair_first = atomic_fetch_add_explicit(
            st->next_pair, pair_chunk, memory_order_relaxed);
        if (pair_first >= n_pairs) {
            break;
        }
        const uint32_t pair_last = hex_smin(pair_first + pair_chunk, n_pairs);
        for (uint32_t pair = pair_first; pair < pair_last; ++pair) {
            const uint32_t channel = pair * 2;
            const uint32_t nt = channel / 32;
            const uint32_t rp = (channel % 32) / 2;
            const HVX_Vector bias0 = hvx_vec_splat_f32(
                (st->flags & HTP_CONV2D_BIAS) ? st->bias[channel] : 0.0f);
            const HVX_Vector bias1 = hvx_vec_splat_f32(
                (st->flags & HTP_CONV2D_BIAS) && channel + 1 < st->oc
                    ? st->bias[channel + 1] : 0.0f);
            const size_t plane = (size_t) st->oh * st->ow;
            const float * residual0 = (st->flags & HTP_CONV2D_RESIDUAL)
                ? st->residual + (size_t) channel * plane +
                    (size_t) st->y0 * st->ow + st->x0
                : NULL;
            const float * residual1 = residual0 && channel + 1 < st->oc
                ? residual0 + plane : NULL;
            if (residual0) {
                const uint32_t rows = hex_smin(st->tile_h, 4u);
                hex_l2fetch(residual0, st->valid_w * sizeof(float),
                            st->ow * sizeof(float), rows);
                if (residual1) {
                    hex_l2fetch(residual1, st->valid_w * sizeof(float),
                                st->ow * sizeof(float), rows);
                }
            }
            for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
                if (residual0 && (oy & 3u) == 0 && oy + 4u < st->tile_h) {
                    const uint32_t rows = hex_smin(st->tile_h - oy - 4u, 4u);
                    hex_l2fetch(residual0 + (size_t) (oy + 4u) * st->ow,
                                st->valid_w * sizeof(float), st->ow * sizeof(float), rows);
                    if (residual1) {
                        hex_l2fetch(residual1 + (size_t) (oy + 4u) * st->ow,
                                    st->valid_w * sizeof(float), st->ow * sizeof(float), rows);
                    }
                }
                for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                    const uint32_t valid_count = hex_smin(
                        st->valid_w - tx * HTP_CONV2D_TILE_W,
                        HTP_CONV2D_TILE_W);
                    const uint32_t mt = oy * st->x_tiles + tx;
                    const HVX_Vector * tile = (const HVX_Vector *)
                        (st->c + ((size_t) nt * st->m_tiles + mt) *
                         HTP_MM_HMX_TILE_N_ELMS);
                    const HVX_VectorPair rows = hvx_vec_f16_to_f32_shuff(tile[rp]);
                    float * dst = st->dst + ((size_t) channel * st->oh + st->y0 + oy) *
                                  st->ow + st->x0 + tx * HTP_CONV2D_TILE_W;
                    HVX_Vector row0 = hvx_vec_add_f32_f32(Q6_V_lo_W(rows), bias0);
                    if (st->flags & HTP_CONV2D_RESIDUAL) {
                        row0 = hvx_vec_add_f32_f32(
                            row0, hvx_vmemu(st->residual + (dst - st->dst)));
                    }
                    if (valid_count == HTP_CONV2D_TILE_W) {
                        Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst, row0);
                    } else {
                        hvx_vec_store_u(dst, valid_count * sizeof(float), row0);
                    }
                    if (channel + 1 < st->oc) {
                        float * dst1 = dst + (size_t) st->oh * st->ow;
                        HVX_Vector row1 = hvx_vec_add_f32_f32(Q6_V_hi_W(rows), bias1);
                        if (st->flags & HTP_CONV2D_RESIDUAL) {
                            row1 = hvx_vec_add_f32_f32(
                                row1, hvx_vmemu(st->residual + (dst1 - st->dst)));
                        }
                        if (valid_count == HTP_CONV2D_TILE_W) {
                            Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst1, row1);
                        } else {
                            hvx_vec_store_u(dst1, valid_count * sizeof(float), row1);
                        }
                    }
                }
            }
        }
    }
    if (st->final_block) {
        asm volatile("syncht" ::: "memory");
    }
}

struct conv2d_hmx_job {
    __fp16 * c;
    const __fp16 * a;
    const __fp16 * b;
    const __fp16 * scales;
    uint32_t mt;
    uint32_t nt;
    uint32_t kt;
    uint32_t kh;
    uint32_t x_tiles;
    uint32_t tile_h;
    uint32_t ic_blocks;
};

static inline void conv2d_hmx_mpy_tiles(
        const __fp16 ** row,
        const __fp16 ** col,
        uint32_t n_tiles) {
    while (n_tiles >= 32) {
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :
                     : "r"(65535), "r"(*row), "r"(*col));
        *row += 32 * HTP_MM_HMX_TILE_N_ELMS;
        *col += 32 * HTP_MM_HMX_TILE_N_ELMS;
        n_tiles -= 32;
    }
    if (n_tiles > 0) {
        const uint32_t range = 2048u * n_tiles - 1u;
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :
                     : "r"(range), "r"(*row), "r"(*col));
    }
}

static void conv2d_hmx_3x3(const struct conv2d_hmx_job * job) {
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) job->scales));

    const uint32_t activation_k_tiles = 3 * job->ic_blocks;
    const size_t activation_stride =
        (size_t) activation_k_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t weight_row_stride =
        (size_t) 3 * activation_stride;
    const uint32_t m_tiles = job->x_tiles * job->tile_h;

    for (uint32_t r = 0; r < job->mt; ++r) {
        const __fp16 * weight_row = job->a + r * weight_row_stride;
        for (uint32_t oy = 0; oy < job->tile_h; ++oy) {
            for (uint32_t tx = 0; tx < job->x_tiles; ++tx) {
                asm volatile(HMX_CLRACC_F16());
                for (uint32_t ky = 0; ky < 3; ++ky) {
                    const __fp16 * row = weight_row + ky * activation_stride;
                    const __fp16 * col = job->b +
                        ((size_t) (oy + ky) * job->x_tiles + tx) * activation_stride;
                    conv2d_hmx_mpy_tiles(&row, &col, activation_k_tiles);
                }
                __fp16 * out = job->c +
                    ((size_t) r * m_tiles + oy * job->x_tiles + tx) *
                    HTP_MM_HMX_TILE_N_ELMS;
                asm volatile(HMX_STORE_AFTER_F16("%0", "%1") :
                             : "r"(out), "r"(0) : "memory");
            }
        }
    }
}

static void conv2d_hmx_worker(void * data) {
    struct conv2d_hmx_job * job = (struct conv2d_hmx_job *) data;
    if (job->kh == 3) {
        conv2d_hmx_3x3(job);
    } else {
        core_dot_chunk_fp16(job->c, job->a, job->b, job->scales,
                            job->mt, job->nt, job->kt);
    }
}

static void conv2d_init_prepare_state(
        struct htp_ops_context * octx,
        float * x,
        __fp16 * a,
        uint32_t x0,
        uint32_t y0,
        uint32_t tile_w,
        uint32_t tile_h,
        bool wait_weight_dma,
        const struct htp_conv2d_kernel_params * kp,
        struct conv2d_prepare_state * state) {
    const struct htp_tensor * src = octx->src[1];
    const int32_t p0 = octx->op_params[2];
    const int32_t p1 = octx->op_params[3];
    const bool upscale2 = (kp->flags & HTP_CONV2D_UPSCALE2) != 0;
    const uint32_t iw = upscale2 ? octx->dst->ne[0] : src->ne[0];
    const uint32_t ih = upscale2 ? octx->dst->ne[1] : src->ne[1];
    const uint32_t halo_w = tile_w + octx->src[0]->ne[0] - 1;
    const uint32_t halo_h = tile_h + octx->src[0]->ne[1] - 1;
    const int32_t sx0 = (int32_t) x0 - p0;
    const int32_t sy0 = (int32_t) y0 - p1;
    const int32_t sx_begin = sx0 < 0 ? 0 : sx0;
    const int32_t sy_begin = sy0 < 0 ? 0 : sy0;
    const int32_t sx_end = hex_smin((int32_t) iw, sx0 + (int32_t) halo_w);
    const int32_t sy_end = hex_smin((int32_t) ih, sy0 + (int32_t) halo_h);

    *state = (struct conv2d_prepare_state) {
        .ctx = octx->ctx,
        .x = x,
        .src = (const float *) src->data,
        .a = a,
        .iw = iw,
        .ih = ih,
        .src_w = src->ne[0],
        .src_h = src->ne[1],
        .halo_w = halo_w,
        .halo_h = halo_h,
        .sx_begin = sx_begin,
        .sy_begin = sy_begin,
        .copy_w = sx_end > sx_begin ? (uint32_t) (sx_end - sx_begin) : 0,
        .copy_h = sy_end > sy_begin ? (uint32_t) (sy_end - sy_begin) : 0,
        .dx = (uint32_t) (sx_begin - sx0),
        .dy = (uint32_t) (sy_begin - sy0),
        .sx0 = sx0,
        .sy0 = sy0,
        .kh = octx->src[0]->ne[1],
        .kw = octx->src[0]->ne[0],
        .ic_blocks = src->ne[2] / 32,
        .n_groups = src->ne[2] / kp->activation_group_channels,
        .group_channels = kp->activation_group_channels,
        .k_tiles = kp->k_tiles,
        .activation_k_tiles = kp->activation_k_tiles,
        .tile_w = tile_w,
        .tile_h = tile_h,
        .x_tiles = tile_w / HTP_CONV2D_TILE_W,
        .m_tiles = (tile_w / HTP_CONV2D_TILE_W) * tile_h,
        .clear = sx_end - sx_begin != (int32_t) halo_w ||
                 sy_end - sy_begin != (int32_t) halo_h,
        .wait_weight_dma = wait_weight_dma,
        .upscale2 = upscale2,
        .direct_upscale2 = upscale2 && sx0 >= 0 && sy0 >= 0 &&
                           sx0 + (int32_t) halo_w <= (int32_t) iw &&
                           sy0 + (int32_t) halo_h <= (int32_t) ih &&
                           (uint32_t) (sx0 / 2) +
                               (tile_w / HTP_CONV2D_TILE_W - 1u) *
                               (HTP_CONV2D_TILE_W / 2u) +
                               (src->type == HTP_TYPE_F16 ? 64u : 32u) <= src->ne[0],
        .input_silu = (kp->flags & HTP_CONV2D_INPUT_SILU) != 0,
        .input_f16 = src->type == HTP_TYPE_F16,
    };
}

static void conv2d_prepare_activation(
        struct htp_ops_context * octx,
        float * x,
        __fp16 * a,
        uint32_t x0,
        uint32_t y0,
        uint32_t tile_w,
        uint32_t tile_h,
        bool wait_weight_dma,
        const struct htp_conv2d_kernel_params * kp) {
    struct conv2d_prepare_state state;
    conv2d_init_prepare_state(octx, x, a, x0, y0, tile_w, tile_h,
                              wait_weight_dma, kp, &state);
    work_queue_run(octx->ctx->work_queue, conv2d_prepare_activation_worker, &state, octx->n_threads);
}

struct conv2d_pipeline_state {
    struct conv2d_prepare_state prepare;
    struct conv2d_store_state store;
    struct conv2d_hmx_job * next_job;
    atomic_uint prepare_barrier;
    atomic_uint next_store_pair;
    atomic_bool hmx_ready;
    uint32_t block;
    bool has_next;
};

static void conv2d_pipeline_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_pipeline_state * st = (struct conv2d_pipeline_state *) data;
    if (st->has_next) {
        conv2d_prepare_activation_worker(nth, ith, &st->prepare);
    }

    atomic_fetch_sub_explicit(&st->prepare_barrier, 1, memory_order_release);
    if (ith == 0) {
        hmx_queue_pop(st->store.ctx->hmx_queue);
        htp_trace_event_start(&st->store.ctx->trace[0], HTP_TRACE_EVT_HVX_O_PROC, st->block);
        atomic_store_explicit(&st->hmx_ready, true, memory_order_release);

        if (st->has_next) {
            while (atomic_load_explicit(&st->prepare_barrier, memory_order_acquire) != 0) {
                hex_pause();
            }
            htp_trace_event_stop(&st->store.ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, st->block + 1);
            hmx_queue_push(st->store.ctx->hmx_queue,
                           hmx_queue_make_desc(conv2d_hmx_worker, st->next_job));
        }
    } else {
        while (!atomic_load_explicit(&st->hmx_ready, memory_order_acquire)) {
            hex_pause();
        }
    }

    conv2d_store_output_worker(nth, ith, &st->store);
}

int op_conv2d(struct htp_ops_context * octx) {
    const struct htp_tensor * weight = octx->src[0];
    const struct htp_tensor * src = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    const struct htp_conv2d_kernel_params * kp =
        (const struct htp_conv2d_kernel_params *) octx->kernel_params;

    if (weight->type != HTP_TYPE_F16 ||
        (src->type != HTP_TYPE_F32 && src->type != HTP_TYPE_F16) ||
        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) ||
        (src->type == HTP_TYPE_F16 && dst->type != HTP_TYPE_F16) ||
        src->ne[3] != 1 || dst->ne[3] != 1 || kp->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_BIAS) &&
        (!octx->src[2] || octx->src[2]->type != HTP_TYPE_F32 ||
         (uint64_t) octx->src[2]->ne[0] * octx->src[2]->ne[1] *
         octx->src[2]->ne[2] * octx->src[2]->ne[3] != weight->ne[3])) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_RESIDUAL) &&
        (!octx->src[3] ||
         octx->src[3]->ne[0] != dst->ne[0] || octx->src[3]->ne[1] != dst->ne[1] ||
         octx->src[3]->ne[2] != dst->ne[2] || octx->src[3]->ne[3] != dst->ne[3] ||
         octx->src[3]->type != dst->type)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_UPSCALE2) &&
        (dst->ne[0] != 2u * src->ne[0] || dst->ne[1] != 2u * src->ne[1] ||
         dst->ne[2] != weight->ne[3] || src->ne[2] != weight->ne[2])) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t kh = weight->ne[1];
    const uint32_t kw = weight->ne[0];
    const uint32_t ic = weight->ne[2];
    const uint32_t oc = weight->ne[3];
    const uint32_t ow = dst->ne[0];
    const uint32_t oh = dst->ne[1];
    uint8_t * base = (uint8_t *) octx->ctx->vtcm_base;
    __fp16 * b = VTCM_LAYOUT_PTR(__fp16, base, kp->off_weight);
    float * xbuf[2] = {
        VTCM_LAYOUT_PTR(float, base, kp->off_x[0]),
        VTCM_LAYOUT_PTR(float, base, kp->off_x[1]),
    };
    __fp16 * abuf[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_a[0]),
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_a[1]),
    };
    __fp16 * cbuf[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_c[0]),
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_c[1]),
    };
    __fp16 * scales = VTCM_LAYOUT_PTR(__fp16, base, kp->off_scales);

    dma_queue * q = octx->ctx->dma[0];
    if (!dma_queue_push(q, dma_make_ptr(b, (const void *) (uintptr_t) weight->data), HTP_CONV2D_TILE_BYTES,
                        HTP_CONV2D_TILE_BYTES, HTP_CONV2D_TILE_BYTES,
                        kp->k_tiles * kp->n_tiles)) {
        return HTP_STATUS_INTERNAL_ERR;
    }
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

    const uint32_t nx = (ow + kp->tile_w - 1) / kp->tile_w;
    const uint32_t ny = (oh + kp->tile_h - 1) / kp->tile_h;
    const uint32_t nblocks = nx * ny;
    float * dst_data = (float *) dst->data;
    const float * bias = (kp->flags & HTP_CONV2D_BIAS)
                       ? (const float *) octx->src[2]->data : NULL;
    const float * residual = (kp->flags & HTP_CONV2D_RESIDUAL)
                           ? (const float *) octx->src[3]->data : NULL;
    struct conv2d_hmx_job jobs[2];

    const uint32_t first_valid_w = hex_smin(kp->tile_w, ow);
    const uint32_t first_tile_w =
        htp_conv2d_align_up(first_valid_w, HTP_CONV2D_TILE_W);
    const uint32_t first_tile_h = hex_smin(kp->tile_h, oh);
    const uint32_t first_m_tiles = (first_tile_w / HTP_CONV2D_TILE_W) * first_tile_h;
    htp_trace_event_start(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, 0);
    conv2d_prepare_activation(octx, xbuf[0], abuf[0], 0, 0,
                              first_tile_w, first_tile_h, true, kp);
    htp_trace_event_stop(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, 0);

    jobs[0] = (struct conv2d_hmx_job) {
        .c = cbuf[0], .a = b, .b = abuf[0], .scales = scales,
        .mt = kp->n_tiles, .nt = first_m_tiles, .kt = kp->k_tiles,
        .kh = kh, .x_tiles = first_tile_w / HTP_CONV2D_TILE_W,
        .tile_h = first_tile_h, .ic_blocks = ic / 32,
    };
    hmx_queue_push(octx->ctx->hmx_queue, hmx_queue_make_desc(conv2d_hmx_worker, &jobs[0]));

    for (uint32_t block = 0; block < nblocks; ++block) {
        const uint32_t slot = block & 1u;
        const uint32_t bx = block % nx;
        const uint32_t by = block / nx;
        const uint32_t x0 = bx * kp->tile_w;
        const uint32_t y0 = by * kp->tile_h;
        const uint32_t valid_w = hex_smin(kp->tile_w, ow - x0);
        const uint32_t tile_w =
            htp_conv2d_align_up(valid_w, HTP_CONV2D_TILE_W);
        const uint32_t tile_h = hex_smin(kp->tile_h, oh - y0);
        const uint32_t x_tiles = tile_w / HTP_CONV2D_TILE_W;
        const uint32_t m_tiles = x_tiles * tile_h;

        struct conv2d_pipeline_state pipeline = {
            .store = {
                .ctx = octx->ctx, .c = cbuf[slot], .dst = dst_data,
                .bias = bias, .residual = residual,
                .ow = ow, .oh = oh, .oc = oc, .n_tiles = kp->n_tiles,
                .x0 = x0, .y0 = y0, .tile_w = tile_w, .valid_w = valid_w,
                .tile_h = tile_h,
                .x_tiles = x_tiles, .m_tiles = m_tiles,
                .final_block = block + 1 == nblocks,
                .next_pair = NULL,
                .flags = kp->flags,
                .output_f16 = dst->type == HTP_TYPE_F16,
            },
            .next_job = NULL,
            .block = block,
            .has_next = block + 1 < nblocks,
        };

        if (pipeline.has_next) {
            const uint32_t next = block + 1;
            const uint32_t next_slot = next & 1u;
            const uint32_t next_x0 = (next % nx) * kp->tile_w;
            const uint32_t next_y0 = (next / nx) * kp->tile_h;
            const uint32_t next_valid_w = hex_smin(kp->tile_w, ow - next_x0);
            const uint32_t next_tile_w =
                htp_conv2d_align_up(next_valid_w, HTP_CONV2D_TILE_W);
            const uint32_t next_tile_h = hex_smin(kp->tile_h, oh - next_y0);
            const uint32_t next_m_tiles =
                (next_tile_w / HTP_CONV2D_TILE_W) * next_tile_h;

            htp_trace_event_start(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, next);
            conv2d_init_prepare_state(octx, xbuf[next_slot], abuf[next_slot],
                                      next_x0, next_y0, next_tile_w, next_tile_h,
                                      false, kp, &pipeline.prepare);
            jobs[next_slot] = (struct conv2d_hmx_job) {
                .c = cbuf[next_slot], .a = b, .b = abuf[next_slot], .scales = scales,
                .mt = kp->n_tiles, .nt = next_m_tiles, .kt = kp->k_tiles,
                .kh = kh, .x_tiles = next_tile_w / HTP_CONV2D_TILE_W,
                .tile_h = next_tile_h, .ic_blocks = ic / 32,
            };
            pipeline.next_job = &jobs[next_slot];
        }

        atomic_init(&pipeline.prepare_barrier, octx->n_threads);
        atomic_init(&pipeline.next_store_pair, 0);
        atomic_init(&pipeline.hmx_ready, false);
        pipeline.store.next_pair = &pipeline.next_store_pair;
        work_queue_run(octx->ctx->work_queue, conv2d_pipeline_worker, &pipeline, octx->n_threads);
        htp_trace_event_stop(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_O_PROC, block);
    }

    return HTP_STATUS_OK;
}
