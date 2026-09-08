
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tiled.h"

#include "ggml-quants.h"
// kvalues table (impl section) for the iq4_xs unpack
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "tiled-kernel.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <mutex>

#define UNUSED GGML_UNUSED

// unpack routines for various quant types src0
static void tiled_unpack_src0(const block_q4_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int r = 0; r < n_rows; r++) {
        const block_q4_K & x = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        uint32_t utmp[4];
        memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * scales = (const uint8_t *) &utmp[0];
        const uint8_t * mins = (const uint8_t *) &utmp[2];
        for (int s = 0; s < NB; s++) {
            tile->scales[r * NB + s] = (int32_t) scales[s];
            tile->mins[r * NB + s] = (int32_t) mins[s];
        }

        // extract the 4-bit codes (low 4 + high 4), same extraction as the reference kernels
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
        tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
        tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
        tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
    }
}

static void tiled_unpack_src0(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int r = 0; r < n_rows; r++) {
        const block_q5_K & x = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        uint32_t utmp[4];
        memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * scales = (const uint8_t *) &utmp[0];
        const uint8_t * mins = (const uint8_t *) &utmp[2];
        for (int s = 0; s < NB; s++) {
            tile->scales[r * NB + s] = (int32_t) scales[s];
            tile->mins[r * NB + s] = (int32_t) mins[s];
        }

        // extract the 5-bit codes (4 low bits + 1 high bit), same as the generic kernels:
        // 64-element chunk j uses qh bits 2j (low 32) and 2j+1 (high 32); OR adds the 5th bit
        // (no overlap with the 4-bit codes, identical to the reference ADD)
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
        tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
        tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
        tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
        tiled_unpk_or<0, 4, 1>(q + 0,   x.qh);
        tiled_unpk_or<1, 4, 1>(q + 32,  x.qh);
        tiled_unpk_or<2, 4, 1>(q + 64,  x.qh);
        tiled_unpk_or<3, 4, 1>(q + 96,  x.qh);
        tiled_unpk_or<4, 4, 1>(q + 128, x.qh);
        tiled_unpk_or<5, 4, 1>(q + 160, x.qh);
        tiled_unpk_or<6, 4, 1>(q + 192, x.qh);
        tiled_unpk_or<7, 4, 1>(q + 224, x.qh);
    }
}

static void tiled_unpack_src0(const block_q6_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 16; // 16-wide subblocks
    for (int r = 0; r < n_rows; r++) {
        const block_q6_K & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d);

        // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
        // per half the lanes are [ql lo(0:32)] [ql lo(32:64)] [ql hi(0:32)] [ql hi(32:64)]
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int half = 0; half < 2; half++) {
            uint8_t * out = q + 128 * half;
            tiled_unpk_nib4(x.ql + 64 * half + 0,  out + 0,  out + 64);
            tiled_unpk_nib4(x.ql + 64 * half + 32, out + 32, out + 96);
            tiled_unpk_or<0, 4, 3>(out + 0,  x.qh + 32 * half);
            tiled_unpk_or<2, 4, 3>(out + 32, x.qh + 32 * half);
            tiled_unpk_or<4, 4, 3>(out + 64, x.qh + 32 * half);
            tiled_unpk_or<6, 4, 3>(out + 96, x.qh + 32 * half);
        }
        // scale is a plain int8 per 16-element subblock (16 per 256-K)
        for (int s = 0; s < NB; s++) { tile->scales[r * NB + s] = (int32_t) (int8_t) x.scales[s]; }
    }
}

static void tiled_unpack_src0(const block_q3_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 16; // 16-wide subblocks
    for (int r = 0; r < n_rows; r++) {
        const block_q3_K & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d);

        // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
        // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
        //   low2 = (qs[half*32 + l] >> 2*group) & 3
        //   high = (hmask[l] >> (half*4 + group)) & 1
        // see ggml_vec_dot_q3_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        const uint8_t * s0 = x.qs;
        const uint8_t * s1 = x.qs + 32;
        uint8_t * o0 = q;
        uint8_t * o1 = q + 128;
        tiled_unpk_2bit<0>(s0, o0);      tiled_unpk_or<0, 2, 1>(o0,      x.hmask);
        tiled_unpk_2bit<2>(s0, o0 + 32); tiled_unpk_or<1, 2, 1>(o0 + 32, x.hmask);
        tiled_unpk_2bit<4>(s0, o0 + 64); tiled_unpk_or<2, 2, 1>(o0 + 64, x.hmask);
        tiled_unpk_2bit<6>(s0, o0 + 96); tiled_unpk_or<3, 2, 1>(o0 + 96, x.hmask);
        tiled_unpk_2bit<0>(s1, o1);      tiled_unpk_or<4, 2, 1>(o1,      x.hmask);
        tiled_unpk_2bit<2>(s1, o1 + 32); tiled_unpk_or<5, 2, 1>(o1 + 32, x.hmask);
        tiled_unpk_2bit<4>(s1, o1 + 64); tiled_unpk_or<6, 2, 1>(o1 + 64, x.hmask);
        tiled_unpk_2bit<6>(s1, o1 + 96); tiled_unpk_or<7, 2, 1>(o1 + 96, x.hmask);
        // 6-bit scale decode (same kmask trick as the reference), stored as (scales - 32)
        static const uint32_t kmask1 = 0x03030303;
        static const uint32_t kmask2 = 0x0f0f0f0f;
        uint32_t auxs[4];
        memcpy(auxs, x.scales, 12);
        const uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t * scales = (const int8_t *) &auxs[0];
        for (int s = 0; s < NB; s++) { tile->scales[r * NB + s] = (int32_t) scales[s] - 32; }
    }
}

static void tiled_unpack_src0(const block_q2_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 16; // 16-wide subblocks
    for (int r = 0; r < n_rows; r++) {
        const block_q2_K & x = rows[r * row_stride];
        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        // 2-bit code: element e -> half=e>>7, el=e&127
        //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
        // see ggml_vec_dot_q2_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int half = 0; half < 2; half++) {
            const uint8_t * s = x.qs + 32 * half;
            uint8_t * out = q + 128 * half;
            tiled_unpk_2bit<0>(s, out + 0);
            tiled_unpk_2bit<2>(s, out + 32);
            tiled_unpk_2bit<4>(s, out + 64);
            tiled_unpk_2bit<6>(s, out + 96);
        }
        // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
        for (int s = 0; s < NB; s++) {
            tile->scales[r * NB + s] = (int32_t) (x.scales[s] & 0xF);
            tile->mins[r * NB + s] = (int32_t) (x.scales[s] >> 4);
        }
    }
}

// iq4_xs: 4-bit codes through the kvalues_iq4nl LUT, 6-bit scales per 32, no min.
// LUT values are stored as kvalues + 128 so the +128 shift the kernel applies to
// the activations cancels against the bsums correction (BIAS = 128); the +128 is
// folded into the table up front so the expansion is a plain lookup
static void tiled_unpack_src0(const block_iq4_xs * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks

    static const uint8_t lut[16] = {
        (uint8_t) (kvalues_iq4nl[0] + 128),  (uint8_t) (kvalues_iq4nl[1] + 128),
        (uint8_t) (kvalues_iq4nl[2] + 128),  (uint8_t) (kvalues_iq4nl[3] + 128),
        (uint8_t) (kvalues_iq4nl[4] + 128),  (uint8_t) (kvalues_iq4nl[5] + 128),
        (uint8_t) (kvalues_iq4nl[6] + 128),  (uint8_t) (kvalues_iq4nl[7] + 128),
        (uint8_t) (kvalues_iq4nl[8] + 128),  (uint8_t) (kvalues_iq4nl[9] + 128),
        (uint8_t) (kvalues_iq4nl[10] + 128), (uint8_t) (kvalues_iq4nl[11] + 128),
        (uint8_t) (kvalues_iq4nl[12] + 128), (uint8_t) (kvalues_iq4nl[13] + 128),
        (uint8_t) (kvalues_iq4nl[14] + 128), (uint8_t) (kvalues_iq4nl[15] + 128),
    };

    for (int r = 0; r < n_rows; r++) {
        const block_iq4_xs & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d);

        // 6-bit scale per 32, stored as (ls - 32); same extraction as dequantize_row_iq4_xs
        for (int s = 0; s < NB; s++) {
            const int ls = ((x.scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x.scales_h >> 2 * s) & 3) << 4);
            tile->scales[r * NB + s] = (int32_t) ls - 32;
        }

        // 4-bit codes through the LUT: low nibbles of a byte pair come first
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        uint8_t lo[32], hi[32];
        for (int u = 0; u < 4; u++) {
            tiled_unpk_nib4(x.qs + 32 * u, lo, hi);
            tiled_lut8(lut, lo + 0,  q + 64 * u + 0);
            tiled_lut8(lut, hi + 0,  q + 64 * u + 16);
            tiled_lut8(lut, lo + 16, q + 64 * u + 32);
            tiled_lut8(lut, hi + 16, q + 64 * u + 48);
        }
    }
}

// iq2_xxs: 2-bit grids through the iq2xxs_grid LUT, 4-bit scale per 32, no min
// codes stored as (value + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq2_xxs * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq2_xxs & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d) * 0.125f;

        uint32_t aux32[2];
        const uint8_t * aux8 = (const uint8_t *) aux32;
        for (int ib32 = 0; ib32 < NB; ib32++) {
            memcpy(aux32, x.qs + 4 * ib32, 2 * sizeof(uint32_t));
            tile->scales[r * NB + ib32] = (int32_t) (2 * (aux32[1] >> 28) + 1);
            for (int l = 0; l < 4; l++) {
                const uint64_t entry = iq2xxs_grid[aux8[l]];
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                tiled_unpk_sign8((const uint8_t *) &entry, signs, &tile->q[r * TILED_TILE_K + 32 * ib32 + 8 * l]);
            }
        }
    }
}

// iq2_xs: 2-bit grids through the iq2xs_grid LUT, 4-bit scale per 16 (two per 32), no min
// codes stored as (value + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq2_xs * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 16; // 16-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq2_xs & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d) * 0.125f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            tile->scales[r * NB + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
            tile->scales[r * NB + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
            const uint16_t * q = x.qs + 4 * ib32;
            for (int l = 0; l < 4; l++) {
                const uint64_t entry = iq2xs_grid[q[l] & 511];
                const uint8_t signs = ksigns_iq2xs[q[l] >> 9];
                tiled_unpk_sign8((const uint8_t *) &entry, signs, &tile->q[r * TILED_TILE_K + 32 * ib32 + 8 * l]);
            }
        }
    }
}

// iq2_s: 2-bit grids through the iq2s_grid LUT, 4-bit scale per 16 (two per 32), no min
// codes stored as (value + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq2_s * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 16; // 16-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq2_s & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d) * 0.125f;

        const uint8_t * qs = x.qs;
        const uint8_t * signs = x.qs + QK_K / 8; // packed sign bytes share the qs array, same as the reference
        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            tile->scales[r * NB + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
            tile->scales[r * NB + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
            for (int l = 0; l < 4; l++) {
                const uint64_t entry = iq2s_grid[qs[l] | (x.qh[ib32] << (8 - 2 * l) & 0x300)];
                tiled_unpk_sign8((const uint8_t *) &entry, signs[l], &tile->q[r * TILED_TILE_K + 32 * ib32 + 8 * l]);
            }
            qs += 4;
            signs += 4;
        }
    }
}

// iq3_xxs: 3-bit grids through the iq3xxs_grid LUT, 4-bit scale per 32, no min
// codes stored as (value + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq3_xxs * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq3_xxs & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d) * 0.25f;

        const uint8_t * qs = x.qs;
        const uint8_t * scales_and_signs = x.qs + QK_K / 4; // 4 bytes per 32: code bits in the top nibble, signs in 7-bit chunks
        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            const uint32_t aux32 = *(const uint32_t *) (scales_and_signs + 4 * ib32);
            tile->scales[r * NB + ib32] = (int32_t) (2 * (aux32 >> 28) + 1);
            for (int l = 0; l < 4; l++) {
                const uint32_t e1 = iq3xxs_grid[qs[2 * l + 0]];
                const uint32_t e2 = iq3xxs_grid[qs[2 * l + 1]];
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                // out[j] = e1 byte j, out[j + 4] = e2 byte j
                uint8_t t[8] = {
                    (uint8_t) e1, (uint8_t) (e1 >> 8), (uint8_t) (e1 >> 16), (uint8_t) (e1 >> 24),
                    (uint8_t) e2, (uint8_t) (e2 >> 8), (uint8_t) (e2 >> 16), (uint8_t) (e2 >> 24),
                };
                tiled_unpk_sign8(t, signs, &tile->q[r * TILED_TILE_K + 32 * ib32 + 8 * l]);
            }
            qs += 8;
        }
    }
}

// iq3_s: 3-bit grids through the iq3s_grid LUT, 4-bit scale per 32 (two per scale byte), no min
// codes stored as (value + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq3_s * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq3_s & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d);

        const uint8_t * qs = x.qs;
        const uint8_t * qh = x.qh;
        const uint8_t * signs = x.signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            tile->scales[r * NB + ib32 + 0] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] & 0xf));
            tile->scales[r * NB + ib32 + 1] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] >> 4));
            for (int h = 0; h < 2; h++) {
                for (int l = 0; l < 4; l++) {
                    const uint32_t e1 = iq3s_grid[qs[2 * l + 0] | ((qh[h] << (8 - 2 * l)) & 256)];
                    const uint32_t e2 = iq3s_grid[qs[2 * l + 1] | ((qh[h] << (7 - 2 * l)) & 256)];
                    // out[j] = e1 byte j, out[j + 4] = e2 byte j
                    uint8_t t[8] = {
                        (uint8_t) e1, (uint8_t) (e1 >> 8), (uint8_t) (e1 >> 16), (uint8_t) (e1 >> 24),
                        (uint8_t) e2, (uint8_t) (e2 >> 8), (uint8_t) (e2 >> 16), (uint8_t) (e2 >> 24),
                    };
                    tiled_unpk_sign8(t, signs[l], &tile->q[r * TILED_TILE_K + (ib32 + h) * 32 + 8 * l]);
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
}

// iq1_s: ternary grid (values +-1/0) scaled by 8 to leave room for the +-1 delta offset,
// the /8 folds into d; 3-bit scale per 32, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq1_s * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 8; // 32-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq1_s & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d) * 0.125f;

        const uint8_t * qs = x.qs;
        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint16_t hw = x.qh[ib];
            tile->scales[r * NB + ib] = (int32_t) (2 * ((hw >> 12) & 7) + 1);
            const int8_t delta = (hw & 0x8000) ? -1 : 1;
            for (int l = 0; l < 4; l++) {
                const uint64_t entry = iq1s_grid[qs[l] | (((hw >> (3 * l)) & 7) << 8)];
                tiled_unpk_tern8((const uint8_t *) &entry, delta, &tile->q[r * TILED_TILE_K + 32 * ib + 8 * l]);
            }
            qs += 4;
        }
    }
}

// iq1_m: like iq1_s but the fp16 scale is packed across the 4 scale bytes (no d field) and the
// delta offset is per 8, giving one scale per 16; 3-bit scale per 16, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
static void tiled_unpack_src0(const block_iq1_m * rows, int64_t row_stride, int n_rows, tiled_tile_src0 * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = 16; // 16-wide subblocks

    for (int r = 0; r < n_rows; r++) {
        const block_iq1_m & x = rows[r * row_stride];

        const uint16_t * sc = (const uint16_t *) x.scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        tile->d[r] = ggml_fp16_to_fp32(scale.f16) * 0.125f;

        const uint8_t * qs = x.qs;
        const uint8_t * qh = x.qh;
        for (int ib = 0; ib < QK_K / 32; ib++) {
            const uint16_t hw = sc[ib / 2];
            const int sh = 6 * (ib % 2);
            tile->scales[r * NB + 2 * ib + 0] = (int32_t) (2 * ((hw >> sh) & 7) + 1);
            tile->scales[r * NB + 2 * ib + 1] = (int32_t) (2 * ((hw >> (sh + 3)) & 7) + 1);

            const uint16_t idx[4] = {
                (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
                (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
                (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
                (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
            };
            const int8_t delta[4] = {
                (int8_t) ((qh[0] & 0x08) ? -1 : 1), (int8_t) ((qh[0] & 0x80) ? -1 : 1),
                (int8_t) ((qh[1] & 0x08) ? -1 : 1), (int8_t) ((qh[1] & 0x80) ? -1 : 1),
            };
            for (int l = 0; l < 4; l++) {
                const uint64_t entry = iq1s_grid[idx[l]];
                tiled_unpk_tern8((const uint8_t *) &entry, delta[l], &tile->q[r * TILED_TILE_K + 32 * ib + 8 * l]);
            }
            qs += 4;
            qh += 2;
        }
    }
}

// unpack src1 tile from q8_K rows
static void tiled_unpack_src1_q8_K(const block_q8_K * const * rows, int n_rows, tiled_tile_src1 * tile,
                                   int kblk) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    // codes: natural [row][k] fill or the VNNI transposed repack, see tiled_repack_src1_codes
    tiled_repack_src1_codes(rows, n_rows, tile, kblk);
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    // d and bsums (ISA-independent)
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & x = rows[r][kblk];
            for (int s = 0; s < TILED_TILE_K / TILED_MICRO; s++) { tile->bsums[s * TILED_TILE_ROWS + r] = (int32_t) x.bsums[s]; }
            tile->d[r] = x.d;
        } else {
            for (int s = 0; s < TILED_TILE_K / TILED_MICRO; s++) { tile->bsums[s * TILED_TILE_ROWS + r] = 0; }
            tile->d[r] = 0.0f;
        }
    }
}

// GGML_CPU_TILED_MM: master switch, on by default. If off, we fast return false and normal vec_dot mul_mat resumes
static bool ggml_tiled_matmul_enabled(void) {
    static bool enabled = true;
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char * env = getenv("GGML_CPU_TILED_MM");
        enabled = env == NULL || atoi(env) != 0;
    });
    return enabled;
}

// GGML_CPU_TILED_MM_FORCE: test/bench only, take the tiled path even when unprofitable
static bool ggml_tiled_matmul_forced(void) {
    static bool forced = false;
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char * env = getenv("GGML_CPU_TILED_MM_FORCE");
        forced = env != NULL && atoi(env) == 1;
    });
    return forced;
}

// hard constraints shared by the MUL_MAT and MUL_MAT_ID entries; the src0 type gate is the
// entries' type switch

#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
static bool ggml_tiled_supported(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1) {
    UNUSED(src0);
    UNUSED(src1);
    return false;
}
#else
static bool ggml_tiled_supported(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1) {
    if (!ggml_tiled_matmul_enabled()) {
        return false;
    }

    // repack-buffer weights hold a repacked layout, let that kernel handle
    if (src0->extra != NULL) {
        return false;
    }
    // Supported quant types for src0
    switch (src0->type) {
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        return false;
    }

    if (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        // We can handle noncontiguous floats because we're repacking to q8_k anyways
        return false;
    }
    return true;
}
#endif

// per-thread workspace size (0 when tiled is disabled or unsupported on this arch)
static size_t ggml_tiled_ws_size(void) {
    if (!ggml_tiled_matmul_enabled()) {
        return 0;
    }
    return sizeof(tiled_ws);
}

size_t ggml_tiled_wdata_size(int n_tasks, struct ggml_tensor * dst) {
    if (! ggml_tiled_supported(dst->src[0], dst->src[1])) {
        return 0; // unsupported, don't allocate
    }
    return 64 + n_tasks * ggml_tiled_ws_size();  // 64 for alignment plus scratch for each thread
}

// the GEMM is profitable at this many src1 rows (MUL_MAT: src1->ne[1], MUL_MAT_ID: the
// expert's routed row count); below it, fall back to optimized vec_dot
#define TILED_MIN_BATCH 32

static bool ggml_tiled_min_batch(int64_t rows) {
    // FORCE (test/bench only) takes the tiled path even when unprofitable
    return ggml_tiled_matmul_forced() || rows >= TILED_MIN_BATCH;
}

// Writeback of the 256x256 window: buf is j-major (row stride buf_stride),
// dst is i-major (column stride dst_stride).
static void tiled_store_window(const float * buf, int n_src0, int n_src1, int buf_stride,
                               float * dst, size_t dst_stride) {
    int ri = 0;
    for (; ri + 16 <= n_src0; ri += 16) {
        int rj = 0;
        for (; rj + 8 <= n_src1; rj += 8) {
            float r[16][8];
            for (int t = 0; t < 16; t++) {
                for (int u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            for (int u = 0; u < 8; u++) {
                for (int t = 0; t < 16; t++) {
                    dst[(ri + t) + (size_t) (rj + u) * dst_stride] = r[t][u];
                }
            }
        }
        // ragged j tail
        for (; rj < n_src1; rj++) {
            for (int t = 0; t < 16; t++) {
                dst[(ri + t) + (size_t) rj * dst_stride] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    for (; ri < n_src0; ri++) {
        for (int j = 0; j < n_src1; j++) {
            dst[ri + (size_t) j * dst_stride] = buf[ri * buf_stride + j];
        }
    }
}

// MUL_MAT_ID (MoE): src1 rows are gathered per output row (expert dispatch) and dst rows are
// scattered back. The expert's cne1 gathered q8_K rows are staged once per expert into
// thread-local scratch (contiguous rows, plus the [k/4][row][4] interleave on VNNI) so the
// unpack and microtile are reused unchanged from mul_mat. Each thread stages the rows itself,
// no cross-thread sync; staging is small vs the gemm.

// src0 rows per g group; finer than the TILED_TILE_K kernel tile so all threads stay busy at small ne01
#define TILED_MMID_GROUP 64

// like tiled_store_window, but the dst columns are not contiguous: col_ptrs[j] points at the
// start of dst column j, whose rows are contiguous (dim 0, nb0 == 4 bytes).
static void tiled_store_window_scatter(const float * buf, int n_src0, int n_src1, int buf_stride,
                                       float * const * col_ptrs) {
    int ri = 0;
    for (; ri + 16 <= n_src0; ri += 16) {
        int rj = 0;
        for (; rj + 8 <= n_src1; rj += 8) {
            float r[16][8];
            for (int t = 0; t < 16; t++) {
                for (int u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            for (int u = 0; u < 8; u++) {
                for (int t = 0; t < 16; t++) {
                    col_ptrs[rj + u][ri + t] = r[t][u];
                }
            }
        }
        // ragged j tail
        for (; rj < n_src1; rj++) {
            for (int t = 0; t < 16; t++) {
                col_ptrs[rj][ri + t] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    for (; ri < n_src0; ri++) {
        for (int j = 0; j < n_src1; j++) {
            col_ptrs[j][ri] = buf[ri * buf_stride + j];
        }
    }
}


// one (g, k) macrotile: zero the acc window, sweep K in 256 element slabs, scatter the result rows
// rows points at this k window's routed src1 rows, row r at its base block
template <typename B, int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_mmid_gemm_window(struct ggml_tensor * dst, const struct ggml_tensor * src0,
                                   const char * src0_cur, int64_t r, int64_t k, int64_t nrows,
                                   const int32_t * expert_rows,
                                   const block_q8_K * const * rows,
                                   tiled_ws * ws) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const int64_t r_end = MIN(r + TILED_MMID_GROUP, ne01);
    const int n_src0 = (int) (r_end - r);

    const size_t src0_bs = ggml_type_size(src0->type);
    const int64_t src0_stride = src0->nb[1] / src0_bs;

    // the window is at most TILED_MMID_GROUP x TILED_TILE_K, so zero only that region of acc
    for (int64_t i = 0; i < n_src0; i++) {
        memset(&ws->acc[i * TILED_TILE_ROWS], 0, nrows * sizeof(float));
    }

    // scattered writeback: column m goes to its routed dst row; r * nb[0] is the window row offset
    float * col_ptrs[TILED_TILE_ROWS];
    for (int64_t m = 0; m < nrows; m++) {
        col_ptrs[m] = (float *) ((char *) dst->data + r * dst->nb[0] + expert_rows[2 * (k + m) + 0] * dst->nb[1] +
                                 expert_rows[2 * (k + m) + 1] * dst->nb[2]);
    }

    // K is stepped in slabs; the slab offset is applied in the unpack
    for (int64_t ib = 0; ib < ne00; ib += TILED_TILE_K) {
        const int kblk = (int) (ib / TILED_TILE_K);
        tiled_unpack_src0((const B *) (src0_cur + r * src0->nb[1] + kblk * src0_bs), src0_stride, n_src0, &ws->src0);
        tiled_unpack_src1_q8_K(rows, nrows, &ws->src1, kblk);
        // 16x16 microtiles sweeping the window; the unpack routines zeropad the
        // macrotiles outside the valid ranges, so the ragged tails are harmless
        for (int64_t ir0 = r; ir0 < r_end; ir0 += TILED_MICRO) {
            for (int64_t ir1 = 0; ir1 < nrows; ir1 += TILED_MICRO) {
                tiled_run_microtile<SUBBLK, HAS_MIN, BIAS>(ws->src0, ws->src1,
                    (int) (ir0 - r), (int) ir1,
                    ws->acc, TILED_TILE_ROWS);
            }
        }
    }

    tiled_store_window_scatter(ws->acc, n_src0, nrows, TILED_TILE_ROWS, col_ptrs);
}

// one expert of MUL_MAT_ID: each k window's dispatched rows are pointed at by a per-window
// row pointer list and swept over the thread's row windows, the dst rows are scattered back
// per window
template <typename B, int SUBBLK, bool HAS_MIN, int BIAS>
static void ggml_compute_forward_mul_mat_id_tiled_one_expert(
        const struct ggml_compute_params * params,
              struct ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        const int32_t *                    expert_rows,
        char *                             scratch) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    const size_t nbw1 = (src1->type == vec_dot_type) ? nb11 : ggml_row_size(vec_dot_type, ne10);
    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    const char * src0_cur = (const char *) src0->data + cur_a * nb02;

    // this thread's workspace
    tiled_ws * ws = (tiled_ws *) (scratch + (size_t) ith * ggml_tiled_ws_size());

    // groups of TILED_MMID_GROUP rows; rounded up, the window tail is clamped in the gemm
    const int64_t ngroups = (ne01 + TILED_MMID_GROUP - 1) / TILED_MMID_GROUP;

    const int64_t g0 = (ngroups * ith) / nth;
    const int64_t g1 = (ngroups * (ith + 1)) / nth;

    // no rows for this thread; nothing to do
    if (g0 >= g1) {
        return;
    }

    // MUL_MAT_ID scatters the src1 rows per routed row: point each k window's rows
    // at the routed src1 rows, then sweep the row windows
    for (int64_t k = 0; k < cne1; k += TILED_TILE_K) {
        const int64_t nrows = MIN(TILED_TILE_K, cne1 - k);

        const block_q8_K * rows[TILED_TILE_ROWS];
        for (int64_t i = 0; i < nrows; i++) {
            const int64_t i11 = expert_rows[2 * (k + i) + 0] % ne11;
            const int64_t i12 = expert_rows[2 * (k + i) + 1];
            rows[i] = (const block_q8_K *) ((const char *) wdata + (i11 + i12 * ne11) * nbw1);
        }

        for (int64_t g = g0; g < g1; g++) {
            const int64_t r = g * TILED_MMID_GROUP;

            tiled_mmid_gemm_window<B, SUBBLK, HAS_MIN, BIAS>(dst, src0, src0_cur, r, k, nrows, expert_rows, rows, ws);
        }
    }
}

template <typename B, int SUBBLK, bool HAS_MIN, int BIAS>
static void ggml_compute_forward_mul_mat_tiled_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    tiled_ws * ws) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const size_t src0_bs  = ggml_type_size(src0->type);
    const size_t src1_bs  = ggml_type_size(vec_dot_type);

    GGML_ASSERT(ne00 % 256 == 0);
    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    const int64_t src0_stride = nb01 / src0_bs;  // blocks between src0 rows
    const int64_t src1_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_size) / src1_bs;

    const int64_t TILE = 256;
    const int64_t MICRO = 16;

    // 256-wide windows over the chunk. The iir1 window is additionally clamped at the
    // src1 batch (ne11) boundary: the tiles require a constant batch index (i12/i13)
    // within a window, so advance by the clamped end, not a fixed 256.
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; ) {
        int64_t iir1_end = MIN(iir1 + TILE, ir1_end);
        const int64_t bnd = (iir1 / ne11 + 1) * ne11;
        if (bnd < iir1_end) {
            iir1_end = bnd;
        }

        const int n_src1 = (int) (iir1_end - iir1);

        // batch coords, constant within the clamped window
        const int64_t i13 = iir1 / (ne12 * ne11);
        const int64_t i12 = (iir1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; dst_col below holds the i12/i13 batch offset, so the store
        // applies i11 * nb1 (not the flattened iir1, which spans all batch dims)
        const int64_t i11 = iir1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat), so the loop batch coords are in
        // src1 space; src0 batches are broadcast over them, map down into src0's batch
        const int64_t i02 = i12 / r2;
        const int64_t i03 = i13 / r3;

        const char * src0_row = (const char *) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3];
        char * dst_col = (char *) dst->data + i12 * nb2 + i13 * nb3;

        // rows[r] is the window's row r base block; the k-slab offset is applied in the unpack
        const block_q8_K * rows[TILED_TILE_ROWS];
        for (int r = 0; r < n_src1; r++) {
            rows[r] = (const block_q8_K *) ((const char *) wdata + (iir1 + r) * src1_stride * src1_bs);
        }

        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += TILE) {
            int64_t iir0_end = MIN(iir0 + TILE, ir0_end);
            const int n_src0 = (int) (iir0_end - iir0);

            // result buffer zeroed once per macrotile
            memset(ws->acc, 0, (size_t)TILED_TILE_ROWS * TILED_TILE_ROWS * sizeof(float));

            // Iterate K dimension by chunks of 256
            for (int64_t ib = 0; ib < ne00; ib += TILE) {
                // Unpack src0 and src1 into macrotiles
                const int kblk = (int) (ib / TILE);
                tiled_unpack_src0((const B *) (src0_row + iir0 * nb01 + kblk * src0_bs), src0_stride, n_src0, &ws->src0);
                tiled_unpack_src1_q8_K(rows, n_src1, &ws->src1, kblk);

                // 16x16 microtiles sweeping the window
                for (int64_t ir0 = iir0; ir0 < iir0_end; ir0 += MICRO) {
                    for (int64_t ir1 = iir1; ir1 < iir1_end; ir1 += MICRO) {
                        tiled_run_microtile<SUBBLK, HAS_MIN, BIAS>(ws->src0, ws->src1,
                            (int) (ir0 - iir0), (int) (ir1 - iir1),
                            ws->acc, TILED_TILE_ROWS);
                    }
                }
            }
            // write acc back out from L2 to main memory
            tiled_store_window(ws->acc, n_src0, n_src1, TILED_TILE_ROWS,
                               (float *) (dst_col + iir0 * nb0 + i11 * nb1), nb1 / nb0);
        }
        iir1 = iir1_end;
    }
}

template <typename B, int SUBBLK, bool HAS_MIN, int BIAS>
static void ggml_compute_forward_mul_mat_tiled_driver(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type      const vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t   const from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (src1->type != vec_dot_type) {
        char * wdata = (char *) params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1] + ne10_block_start*bs*src1->nb[0]),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth. This saves a bit of coordination right at the start.
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 256; //TILED_TILE_ROWS;

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim. CEIL(nr/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // Step down chunk size if too few chunks to saturate cores, minimum is microtile size
    while (nchunk0 * nchunk1 < nth * 4 && chunk_size > 16) {
        chunk_size = chunk_size / 2;
        nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    // per-thread workspace: after any converted src1 data in wdata, aligned to 64
    char * ws_base = (char *) params->wdata;
    if (src1->type != vec_dot_type) {
        ws_base += GGML_PAD(ggml_row_size(vec_dot_type, ggml_nelements(src1)), 64);
    }
    ws_base = (char *) (((uintptr_t) ws_base + 63) & ~(uintptr_t) 63);
    tiled_ws * ws = (tiled_ws *) ws_base + ith;

    // TODO:  if we KNOW we're on a machine where all cores are equal, we could skip the coordination/work-stealing and just assign chunks deterministically
    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_tiled_one_chunk<B, SUBBLK, HAS_MIN, BIAS>(params, dst, ir0_start, ir0_end, ir1_start, ir1_end, ws);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

// src0 type dispatch, shared by the MUL_MAT and MUL_MAT_ID entries: one expert for
// MUL_MAT_ID (expert_rows != NULL), the full op for MUL_MAT
template <typename B, int SUBBLK, bool HAS_MIN, int BIAS>
static bool tiled_matmul_dispatch(const struct ggml_compute_params * params,
                                  struct ggml_tensor * dst,
                                  const int32_t * expert_rows,
                                  int64_t cur_a,
                                  int64_t cne1,
                                  char * scratch) {
    if (expert_rows == NULL) {
        ggml_compute_forward_mul_mat_tiled_driver<B, SUBBLK, HAS_MIN, BIAS>(params, dst);
    } else {
        ggml_compute_forward_mul_mat_id_tiled_one_expert<B, SUBBLK, HAS_MIN, BIAS>(params, dst, cur_a, cne1, expert_rows, scratch);
    }
    return true;
}

// the supported src0 types, one list for both ops; false to fall through to vec_dot
static bool ggml_tiled_matmul_type_dispatch(const struct ggml_compute_params * params,
                                            struct ggml_tensor * dst,
                                            const int32_t * expert_rows = NULL,
                                            int64_t cur_a = 0,
                                            int64_t cne1 = 0,
                                            char * scratch = NULL) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q6_K:
            return tiled_matmul_dispatch<block_q6_K, 16, false, 32>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_Q5_K:
            return tiled_matmul_dispatch<block_q5_K, 32, true,  0>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_Q4_K:
            return tiled_matmul_dispatch<block_q4_K, 32, true,  0>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_Q3_K:
            return tiled_matmul_dispatch<block_q3_K, 16, false,  4>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_Q2_K:
            return tiled_matmul_dispatch<block_q2_K, 16, true,  0>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ4_XS:
            return tiled_matmul_dispatch<block_iq4_xs, 32, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ2_XXS:
            return tiled_matmul_dispatch<block_iq2_xxs, 32, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ2_XS:
            return tiled_matmul_dispatch<block_iq2_xs, 16, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ2_S:
            return tiled_matmul_dispatch<block_iq2_s, 16, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ3_XXS:
            return tiled_matmul_dispatch<block_iq3_xxs, 32, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ3_S:
            return tiled_matmul_dispatch<block_iq3_s, 32, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ1_S:
            return tiled_matmul_dispatch<block_iq1_s, 32, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        case GGML_TYPE_IQ1_M:
            return tiled_matmul_dispatch<block_iq1_m, 16, false, 128>(params, dst, expert_rows, cur_a, cne1, scratch);
        default:
            return false;
    }
}

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    // --use-ref means bail out and go back to vec_dot reference impl
    if (params->use_ref) {
        return false;
    }
    if (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        return false;
    }
    if (!ggml_tiled_min_batch(dst->src[1]->ne[1])) {
        return false;
    }
    return ggml_tiled_matmul_type_dispatch(params, dst);
}

// MUL_MAT_ID (MoE), one expert; returns true if the expert was computed here,
// per expert eligibility (type gate, batch floor) is decided here
bool ggml_compute_forward_mul_mat_id_tiled(
        const struct ggml_compute_params * params,
              struct ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        const int32_t *                    expert_rows,
        char *                             scratch) {
    if (params->use_ref) {
        return false;
    }
    if (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        return false;
    }
    // profitability is per expert: the rows routed to this expert
    if (!ggml_tiled_min_batch(cne1)) {
        return false;
    }
    return ggml_tiled_matmul_type_dispatch(params, dst, expert_rows, cur_a, cne1, scratch);
}
