//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "wdecomp.hpp"
#include "dequantize.hpp"

// The 8 weights [8l, 8l + 8) of a lane: weight m is s * q[m] + b.
struct wdecomp_lane {
    float s;
    float b;
    int   q[8];
};

#if QK_K == 256

static __dpct_inline__ wdecomp_lane wdecomp_q2_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t i        = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     h        = p / 128;
    const int     j        = (p % 128) / 32;
    const int     s0       = p % 32;

    const uint32_t *   qs     = (const uint32_t *) (base + i * (QK_K / 4) + 32 * h + s0);
    const uint8_t *    scales = base + n_blocks * (QK_K / 4) + i * (QK_K / 16);
    const ggml_half2 * dm     = reinterpret_cast<const ggml_half2 *>(base + n_blocks * (QK_K / 4) + n_blocks * (QK_K / 16) + i * sizeof(ggml_half2));

    const uint8_t sc = scales[8 * h + 2 * j + s0 / 16];
    const float   dall = (*dm)[0];
    const float   dmin = (*dm)[1];

    wdecomp_lane r;
    r.s = dall * (sc & 0xF);
    r.b = -(dmin * (sc >> 4));
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (byte8(qs[0], qs[1], m) >> (2 * j)) & 3;
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q3_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t i        = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     n        = p / 128;
    const int     j        = (p % 128) / 32;
    const int     l0       = p % 32;

    const uint32_t * q      = (const uint32_t *) (base + i * (QK_K / 4) + 32 * n + l0);
    const uint32_t * hm     = (const uint32_t *) (base + n_blocks * (QK_K / 4) + i * (QK_K / 8) + l0);
    const uint8_t *  scales = base + n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + i * 12;
    const float      d_all  = static_cast<float>(*reinterpret_cast<const ggml_half *>(
        base + n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + n_blocks * 12 + i * sizeof(ggml_half)));

    const int     is = 8 * n + 2 * j + l0 / 16;
    const uint8_t mk = 1 << (4 * n + j);
    const uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    wdecomp_lane r;
    r.s = d_all * (us - 32);
    r.b = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (int) ((byte8(q[0], q[1], m) >> (2 * j)) & 3) - ((byte8(hm[0], hm[1], m) & mk) ? 0 : 4);
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q4_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb   = k / QK_K;
    const int64_t i    = l / (QK_K / 8);
    const int     p    = 8 * (l % (QK_K / 8));
    const int     il   = p / 64;
    const int     half = (p % 64) / 32;
    const int     pos  = p % 32;

    const uint32_t * qs         = (const uint32_t *) (base + i * (QK_K / 2) + 32 * il + pos);
    const uint8_t *  scales_ptr = base + nb * (QK_K / 2) + i * K_SCALE_SIZE;
    const ggml_half2 dm         = *reinterpret_cast<const ggml_half2 *>(base + nb * (QK_K / 2) + nb * K_SCALE_SIZE + i * sizeof(ggml_half2));

    uint8_t sc, mn;
    get_scale_min_k4(2 * il + half, scales_ptr, sc, mn);

    wdecomp_lane r;
    r.s = (float) dm.x() * sc;
    r.b = -((float) dm.y() * mn);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = byte8(qs[0] >> (4 * half), qs[1] >> (4 * half), m) & 0xF;
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q5_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t ib       = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     il       = p / 64;
    const int     half     = (p % 64) / 32;
    const int     pos      = p % 32;

    const uint32_t * ql         = (const uint32_t *) (base + ib * (QK_K / 2) + 32 * il + pos);
    const uint32_t * qh         = (const uint32_t *) (base + n_blocks * (QK_K / 2) + ib * (QK_K / 8) + pos);
    const uint8_t *  scales_ptr = base + n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + ib * K_SCALE_SIZE;
    const ggml_half2 dm         = *reinterpret_cast<const ggml_half2 *>(
        base + n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + n_blocks * K_SCALE_SIZE + ib * sizeof(ggml_half2));

    uint8_t sc, mn;
    get_scale_min_k4(2 * il + half, scales_ptr, sc, mn);
    const uint8_t hmk = 1 << (2 * il + half);

    wdecomp_lane r;
    r.s = (float) dm.x() * sc;
    r.b = -((float) dm.y() * mn);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (byte8(ql[0] >> (4 * half), ql[1] >> (4 * half), m) & 0xF) + (byte8(qh[0], qh[1], m) & hmk ? 16 : 0);
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q6_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t ib       = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     ip       = p / 128;
    const int     q4       = (p % 128) / 32;
    const int     il       = p % 32;

    const uint32_t *  ql = (const uint32_t *) (base + ib * (QK_K / 2) + 64 * ip + il + 32 * (q4 & 1));
    const uint32_t *  qh = (const uint32_t *) (base + (QK_K / 2) * n_blocks + (QK_K / 4) * ib + 32 * ip + il);
    const int8_t *    sc = reinterpret_cast<const int8_t *>(base + (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks +
                                                            (QK_K / 16) * ib + 8 * ip + il / 16 + 2 * q4);
    const ggml_half * d  = (const ggml_half *) (base + ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks) + ib;

    const uint32_t la = ql[0] >> (4 * (q4 >> 1));
    const uint32_t lb = ql[1] >> (4 * (q4 >> 1));
    const uint32_t ha = qh[0] >> (2 * q4);
    const uint32_t hb = qh[1] >> (2 * q4);

    wdecomp_lane r;
    r.s = (float) *d * sc[0];
    r.b = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (int) ((byte8(la, lb, m) & 0xF) | ((byte8(ha, hb, m) & 3) << 4)) - 32;
    }
    return r;
}

static __dpct_inline__ void wdecomp_signed_grid(wdecomp_lane & r, const uint8_t * grid, const uint8_t signs) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        r.q[j] = signs & kmask_iq2xs[j] ? -(int) grid[j] : (int) grid[j];
    }
}

static __dpct_inline__ wdecomp_lane wdecomp_iq2_xxs(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t * q2    = base + i * (QK_K / 4) + 8 * ib;
    const uint32_t  aux32 = *reinterpret_cast<const uint32_t *>(q2 + 4);
    const float     dall  = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (0.5f + (aux32 >> 28)) * 0.25f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) (iq2xxs_grid + q2[il]), ksigns_iq2xs[(aux32 >> 7*il) & 127]);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq2_xs(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint16_t q2     = *reinterpret_cast<const uint16_t *>(base + i * (QK_K / 4) + 2 * (4 * ib + il));
    const uint8_t  scales = base[nb * (QK_K / 4) + i * (QK_K / 32) + ib];
    const float    dall   = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 32) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (0.5f + ((scales >> 4*(il/2)) & 0xf)) * 0.25f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) (iq2xs_grid + (q2 & 511)), ksigns_iq2xs[q2 >> 9]);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq2_s(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t   qs    = base[i * (QK_K / 8) + 4 * ib + il];
    const uint8_t   signs = base[nb * (QK_K / 8) + i * (QK_K / 8) + 4 * ib + il];
    const uint8_t * hs    = base + nb * (QK_K / 4) + i * (QK_K / 16);
    const float     dall  = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 16) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (0.5f + ((hs[QK_K / 32 + ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) (iq2s_grid + (qs | ((hs[ib] << (8-2*il)) & 0x300))), signs);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq3_xxs(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t * q3    = base + i * (QK_K / 4) + 8 * ib + 2 * il;
    const uint32_t  aux32 = *reinterpret_cast<const uint32_t *>(base + nb * (QK_K / 4) + i * (QK_K / 8) + 4 * ib);
    const float     dall  = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 8) + i * sizeof(ggml_half));

    const uint32_t g[2] = { iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]] };

    wdecomp_lane r;
    r.s = dall * (0.5f + (aux32 >> 28)) * 0.5f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) g, ksigns_iq2xs[(aux32 >> 7*il) & 127]);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq3_s(const uint8_t * base, int64_t l, int64_t k) {
    constexpr int ss_size = QK_K / 8 + QK_K / 64;

    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t * qs   = base + i * (QK_K / 4) + 8 * ib + 2 * il;
    const uint8_t   qh   = base[nb * (QK_K / 4) + i * (QK_K / 32) + ib];
    const uint8_t * ss   = base + nb * (QK_K / 4 + QK_K / 32) + i * ss_size;
    const float     dall = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 32 + ss_size) + i * sizeof(ggml_half));

    const uint32_t g[2] = { iq3s_grid[qs[0] | ((qh << (8-2*il)) & 256)], iq3s_grid[qs[1] | ((qh << (7-2*il)) & 256)] };

    wdecomp_lane r;
    r.s = dall * (1 + 2*((ss[QK_K / 8 + ib/2] >> 4*(ib%2)) & 0xf));
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) g, ss[4*ib + il]);
    return r;
}

static __dpct_inline__ void wdecomp_iq1_grid(wdecomp_lane & r, const uint32_t g, const bool neg_delta) {
    const int c = neg_delta ? -9 : -7;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        r.q[m + 0] = 8 * (int) ((g >> (8 * m)) & 0xF) + c;
        r.q[m + 4] = 8 * (int) ((g >> (8 * m + 4)) & 0xF) + c;
    }
}

static __dpct_inline__ wdecomp_lane wdecomp_iq1_s(const uint8_t * base, int64_t l, int64_t k) {
    static_assert(IQ1S_DELTA == 0.125f, "IQ1_S integer weights assume IQ1S_DELTA == 1/8");
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t  qs   = base[i * (QK_K / 8) + 4 * ib + il];
    const uint16_t qh   = *reinterpret_cast<const uint16_t *>(base + nb * (QK_K / 8) + i * (QK_K / 16) + 2 * ib);
    const float    dall = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 8 + QK_K / 16) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (2*((qh >> 12) & 7) + 1) * 0.125f;
    r.b = 0.0f;
    wdecomp_iq1_grid(r, iq1s_grid_gpu[qs | (((qh >> 3*il) & 7) << 8)], qh & 0x8000);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq1_m(const uint8_t * base, int64_t l, int64_t k) {
    static_assert(IQ1M_DELTA == 0.125f, "IQ1_M integer weights assume IQ1M_DELTA == 1/8");
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t    qs = base[i * (QK_K / 8) + 4 * ib + il];
    const uint8_t    qh = base[nb * (QK_K / 8) + i * (QK_K / 16) + 2 * ib + il / 2];
    const uint16_t * sc = reinterpret_cast<const uint16_t *>(base + nb * (QK_K / 8 + QK_K / 16) + i * (QK_K / 32));

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int ib16 = 2*ib + il/2;

    wdecomp_lane r;
    r.s = (float) scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1) * 0.125f;
    r.b = 0.0f;
    wdecomp_iq1_grid(r, iq1s_grid_gpu[qs | (((qh >> 4*(il%2)) & 7) << 8)], qh & (0x08 << 4*(il%2)));
    return r;
}

#endif

static __dpct_inline__ void wdecomp_iq4_codes(wdecomp_lane & r, const uint32_t q0, const uint32_t q1) {
    const uint32_t lo = iq4nl_lookup4(q0 & 0x0F0F0F0F);
    const uint32_t hi = iq4nl_lookup4(q1 & 0x0F0F0F0F);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        r.q[m + 0] = (int8_t) (lo >> (8 * m));
        r.q[m + 4] = (int8_t) (hi >> (8 * m));
    }
}

static __dpct_inline__ wdecomp_lane wdecomp_iq4_nl(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t ib    = l / 4;
    const int     j     = 8 * (l % 4);
    const int     shift = j < QK4_NL / 2 ? 0 : 4;

    const uint32_t * q = (const uint32_t *) (base + ib * (QK4_NL / 2) + j % (QK4_NL / 2));

    wdecomp_lane r;
    r.s = *((const sycl::half *) (base + k / 2) + ib);
    r.b = 0.0f;
    wdecomp_iq4_codes(r, q[0] >> shift, q[1] >> shift);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq4_xs(const uint8_t * base, int64_t l, int64_t /* k */) {
    const int64_t i     = l / (QK_K / 8);
    const int     ib    = (l % (QK_K / 8)) / 4;
    const int     j     = 8 * (l % 4);
    const int     shift = j < 16 ? 0 : 4;

    const block_iq4_xs * x = (const block_iq4_xs *) base + i;
    const uint32_t *     q = (const uint32_t *) (x->qs + 16*ib + j%16);

    wdecomp_lane r;
    r.s = (float) x->d * ((((x->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x->scales_h >> 2*ib) & 3) << 4)) - 32);
    r.b = 0.0f;
    wdecomp_iq4_codes(r, q[0] >> shift, q[1] >> shift);
    return r;
}

typedef wdecomp_lane (*wdecomp_decoder_t)(const uint8_t *, int64_t, int64_t);

template <wdecomp_decoder_t decode, ggml_sycl_wdecomp_int wt, int group, bool has_bias>
static void dequantize_to_int_sycl(const void * vx, void * w, sycl::half * scales, sycl::half * bias, int64_t nrows,
                                   int64_t ncols, dpct::queue_ptr stream) {
    constexpr int chunks      = wt == GGML_SYCL_WDECOMP_S8 ? 2 : 4;
    constexpr int lane_elems  = 8 * chunks;
    constexpr int tile_rows   = 16;
    constexpr int tile_lanes  = WARP_SIZE;
    constexpr int tile_groups = tile_lanes * lane_elems / group;
    const int64_t k             = nrows * ncols;
    const int64_t lanes_per_row = ncols / lane_elems;
    const int64_t ngroups       = ncols / group;
    const sycl::range<2> global(ceil_div(nrows, tile_rows) * tile_rows, ceil_div(lanes_per_row, tile_lanes) * tile_lanes);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 2> s_tile(sycl::range<2>(tile_rows, tile_groups + 1), cgh);
        sycl::local_accessor<sycl::half, 2> b_tile(sycl::range<2>(has_bias ? tile_rows : 1, tile_groups + 1), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, sycl::range<2>(tile_rows, tile_lanes)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int     tr  = item.get_local_id(0);
            const int     tl  = item.get_local_id(1);
            const int64_t row = item.get_global_id(0);
            const int64_t lc  = item.get_global_id(1);
            if (row < nrows && lc < lanes_per_row) {
                const uint8_t * base = static_cast<const uint8_t *>(vx);
                const int64_t   lane = row * lanes_per_row + lc;
                uint32_t        v[4];
#pragma unroll
                for (int c = 0; c < chunks; ++c) {
                    const wdecomp_lane r = decode(base, chunks * lane + c, k);
                    if constexpr (wt == GGML_SYCL_WDECOMP_S8) {
                        v[2 * c + 0] = 0;
                        v[2 * c + 1] = 0;
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            v[2 * c + 0] |= (uint32_t) (r.q[m + 0] & 0xFF) << (8 * m);
                            v[2 * c + 1] |= (uint32_t) (r.q[m + 4] & 0xFF) << (8 * m);
                        }
                    } else {
                        v[c] = 0;
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            v[c] |= (uint32_t) (r.q[m] & 0xF) << (4 * m);
                        }
                    }
                    if ((8 * c) % group == 0) {
                        const int gi = (tl * lane_elems + 8 * c) / group;
                        s_tile[tr][gi] = sycl::half(r.s);
                        if constexpr (has_bias) {
                            b_tile[tr][gi] = sycl::half(r.b);
                        }
                    }
                }
                static_cast<sycl::uint4 *>(w)[lane] = sycl::uint4(v[0], v[1], v[2], v[3]);
            }
            sycl::group_barrier(item.get_group());
            const int64_t row0 = item.get_group(0) * tile_rows;
            const int64_t kg0  = item.get_group(1) * tile_groups;
#pragma unroll
            for (int gi = tr; gi < tile_groups; gi += tile_rows) {
                if (row0 + tl < nrows && kg0 + gi < ngroups) {
                    scales[(kg0 + gi) * nrows + row0 + tl] = s_tile[tl][gi];
                    if constexpr (has_bias) {
                        bias[(kg0 + gi) * nrows + row0 + tl] = b_tile[tl][gi];
                    }
                }
            }
        });
    });
}

bool ggml_sycl_wdecomp_format(ggml_type type, bool reordered, ggml_sycl_wdecomp_fmt & fmt) {
#if QK_K == 256
    if (type == GGML_TYPE_IQ4_XS) {
        fmt = { GGML_SYCL_WDECOMP_S8, 32, false };
        return !reordered;
    }
    if (!reordered) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q2_K:    fmt = { GGML_SYCL_WDECOMP_U4, 16, true  }; return true;
        case GGML_TYPE_Q3_K:    fmt = { GGML_SYCL_WDECOMP_S4, 16, false }; return true;
        case GGML_TYPE_Q4_K:    fmt = { GGML_SYCL_WDECOMP_U4, 32, true  }; return true;
        case GGML_TYPE_Q5_K:    fmt = { GGML_SYCL_WDECOMP_S8, 32, true  }; return true;
        case GGML_TYPE_Q6_K:    fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ1_S:   fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ1_M:   fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ2_XXS: fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ2_XS:  fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ2_S:   fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ3_XXS: fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ3_S:   fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ4_NL:  fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        default:                return false;
    }
#else
    GGML_UNUSED(type); GGML_UNUSED(reordered); GGML_UNUSED(fmt);
    return false;
#endif
}

bool ggml_sycl_wdecomp_pays(const ggml_sycl_wdecomp_fmt & fmt, int64_t M, int64_t K) {
    return fmt.has_bias ? M * 256 <= K * fmt.group : 2 * M < K;
}

void ggml_sycl_dequantize_to_int(ggml_type type, bool reordered, const void * vx, void * w, sycl::half * scales,
                                 sycl::half * bias, int64_t nrows, int64_t ncols, dpct::queue_ptr stream) {
    GGML_UNUSED(reordered);
#if QK_K == 256
    switch (type) {
        case GGML_TYPE_Q2_K:    dequantize_to_int_sycl<wdecomp_q2_K,    GGML_SYCL_WDECOMP_U4, 16, true >(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q3_K:    dequantize_to_int_sycl<wdecomp_q3_K,    GGML_SYCL_WDECOMP_S4, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q4_K:    dequantize_to_int_sycl<wdecomp_q4_K,    GGML_SYCL_WDECOMP_U4, 32, true >(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q5_K:    dequantize_to_int_sycl<wdecomp_q5_K,    GGML_SYCL_WDECOMP_S8, 32, true >(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q6_K:    dequantize_to_int_sycl<wdecomp_q6_K,    GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ1_S:   dequantize_to_int_sycl<wdecomp_iq1_s,   GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ1_M:   dequantize_to_int_sycl<wdecomp_iq1_m,   GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ2_XXS: dequantize_to_int_sycl<wdecomp_iq2_xxs, GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ2_XS:  dequantize_to_int_sycl<wdecomp_iq2_xs,  GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ2_S:   dequantize_to_int_sycl<wdecomp_iq2_s,   GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ3_XXS: dequantize_to_int_sycl<wdecomp_iq3_xxs, GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ3_S:   dequantize_to_int_sycl<wdecomp_iq3_s,   GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ4_NL:  dequantize_to_int_sycl<wdecomp_iq4_nl,  GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ4_XS:  dequantize_to_int_sycl<wdecomp_iq4_xs,  GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        default:                GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(type));
    }
#else
    GGML_UNUSED(type); GGML_UNUSED(vx); GGML_UNUSED(w); GGML_UNUSED(scales); GGML_UNUSED(bias);
    GGML_UNUSED(nrows); GGML_UNUSED(ncols); GGML_UNUSED(stream);
    GGML_ABORT("%s: requires QK_K == 256", __func__);
#endif
}

template <int group>
static void f32_to_f16_gsum_sycl(const float * x, sycl::half * y, sycl::half * gsum, int64_t k, dpct::queue_ptr stream) {
    constexpr int wg_size = 256;
    constexpr int lanes   = group / 8;
    const int64_t n_lanes = k / 8;
    const int64_t n_wg    = (n_lanes + wg_size - 1) / wg_size;
    stream->parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        const int64_t l     = item.get_global_id(0);
        const bool    valid = l < n_lanes;
        float         sum   = 0.0f;
        if (valid) {
            const sycl::vec<float, 8> f = reinterpret_cast<const sycl::vec<float, 8> *>(x)[l];
            reinterpret_cast<sycl::vec<sycl::half, 8> *>(y)[l] = f.convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                sum += f[m];
            }
        }
        const auto sg = item.get_sub_group();
#pragma unroll
        for (int o = 1; o < lanes; o <<= 1) {
            sum += sycl::permute_group_by_xor(sg, sum, o);
        }
        if (valid && l % lanes == 0) {
            gsum[l / lanes] = sycl::half(sum);
        }
    });
}

void ggml_sycl_f32_to_f16_gsum(const float * x, sycl::half * y, sycl::half * gsum, int64_t k, int group,
                               dpct::queue_ptr stream) {
    GGML_ASSERT(k % group == 0);
    switch (group) {
        case 16: f32_to_f16_gsum_sycl<16>(x, y, gsum, k, stream); break;
        case 32: f32_to_f16_gsum_sycl<32>(x, y, gsum, k, stream); break;
        default: GGML_ABORT("%s: unsupported group %d", __func__, group);
    }
}
