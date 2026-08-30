
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tiled.h"

#include "ggml-quants.h"
#include "tiled-kernel.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <new>

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

// unpack src1 tile from q8_K rows
static void tiled_unpack_src1_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
#if defined(KERNEL_SRC1_UNPACK)
    // Kernel-defined unpack for VNNI
    tiled_unpack_src1_q8_K_kernel(n_rows, tile, qv, nr1_pad, r_start, kblk);
#else
    // Straight row copy for normal contiguous rows
    UNUSED(qv);
    UNUSED(nr1_pad);
    UNUSED(r_start);
    UNUSED(kblk);
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            memcpy(&tile->q[r * TILED_TILE_K], rows[r * row_stride].qs, TILED_TILE_K);
        } else {
            memset(&tile->q[r * TILED_TILE_K], 0, TILED_TILE_K);
        }
    }
#endif
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & x = rows[r * row_stride];
            for (int s = 0; s < TILED_TILE_K / TILED_MICRO; s++) { tile->bsums[s * TILED_TILE_ROWS + r] = (int32_t) x.bsums[s]; }
            tile->d[r] = x.d;
        } else {
            for (int s = 0; s < TILED_TILE_K / TILED_MICRO; s++) { tile->bsums[s * TILED_TILE_ROWS + r] = 0; }
            tile->d[r] = 0.0f;
        }
    }
}

// All three pieces of per-thread tiled state, lazily allocated.
struct tiled_kernel_ws {
    tiled_tile_src0 * src0 = nullptr;
    tiled_tile_src1 * src1 = nullptr;
    float * acc = nullptr;

    ~tiled_kernel_ws() {
        delete src0;
        delete src1;
        // acc was allocated 64B-aligned (std::align_val_t), so free with the
        // matching aligned delete, not delete[]
        if (acc) {
            ::operator delete(acc, std::align_val_t(64));
        }
    }
};

static thread_local tiled_kernel_ws tiled_ws;

// GGML_CPU_TILED_MM: master switch, on by default. If off, we fast return false and normal vec_dot mul_mat resumes
static bool ggml_tiled_matmul_enabled(void) {
    static bool enabled = true;
    static bool inited  = false;
    if (!inited) {
        const char * env = getenv("GGML_CPU_TILED_MM");
        enabled = env == NULL || atoi(env) != 0;
        inited = true;
    }
    return enabled;
}

// GGML_CPU_TILED_MM_FORCE: test/bench only, take the tiled path even when unprofitable
static bool ggml_tiled_matmul_forced(void) {
    static bool forced = false;
    static bool inited  = false;
    if (!inited) {
        const char * env = getenv("GGML_CPU_TILED_MM_FORCE");
        forced = env != NULL && atoi(env) == 1;
        inited = true;
    }
    return forced;
}

// Called by ggml-cpu.c to increase wdata in the case of VNNI, or other future kernels that need a second scratch space
size_t ggml_tiled_extra_wdata_len(int64_t ne10, int64_t nr1) {
#if defined(KERNEL_SRC1_UNPACK)
    if (ggml_tiled_matmul_enabled()) {
        const int64_t k1_pad = (ne10 + 255) & ~255LL; // the region holds whole slabs
        const int64_t nr1_pad = (nr1 + 15) & ~15LL;
        return (size_t) k1_pad * (size_t) nr1_pad;
    }
#endif
    UNUSED(ne10);
    UNUSED(nr1);
    return 0;
}

// compatibility/profitability gate.  anything not supported here will fall back to the vec_dot path
static bool ggml_tiled_matmul_supported(const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src1) {

#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
    return false;
#else
    if (!ggml_tiled_matmul_enabled()) {
        return false;
    }

    // hard constraints: the kernel is only correct/defined for these

    // repack-buffer weights hold a repacked layout, let that kernel handle
    if (src0->extra != NULL) {
        return false;
    }
    // K-quant weights only for now
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K &&
        src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_Q3_K && src0->type != GGML_TYPE_Q2_K) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        return false;
    }

    if (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        // We can handle noncontiguous floats because we're repacking to q8_k anyways
        return false;
    }

    // If forced, skip profitability check
    if (ggml_tiled_matmul_forced()) {
        return true;
    }

    // We are slightly profitable at 64 rows, unprofitable below, fall back to optimized vec_dot
    if (src1->ne[1] < 64) {
        return false;
    }
    return true;
#endif
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

template <typename B, int SUBBLK, bool HAS_MIN, int BIAS>
static void ggml_compute_forward_mul_mat_tiled_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

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

    // info about the interleaved geometry in the case we have special unpacking for src1 (VNNI interleaving)
#if defined(KERNEL_SRC1_UNPACK)
    const tiled_interleave_geom geom = tiled_get_interleave_geom(params, src1, vec_dot_type, ne10, ne11 * ne12 * ne13);
    const int8_t * qv = geom.qv;
    const int64_t nr1_pad = geom.nr1_pad;
#else
    const int8_t * qv = nullptr;
    const int64_t nr1_pad = 0;
#endif

    if (!tiled_ws.src0) {
        tiled_ws.src0 = new tiled_tile_src0();
    }
    if (!tiled_ws.src1) {
        tiled_ws.src1 = new tiled_tile_src1();
    }
    if (!tiled_ws.acc) {
        // Write buffer, stays in L2 and reduces TLB pressure until we copy/transpose out to main mem at the end.
        tiled_ws.acc = static_cast<float *>(
            ::operator new(sizeof(float) * (size_t) TILED_TILE_ROWS * TILED_TILE_ROWS,
                           std::align_val_t(64)));
    }

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

        const block_q8_K * src1_col = (const block_q8_K *) ((const char *) wdata + iir1 * src1_stride * src1_bs);

        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += TILE) {
            int64_t iir0_end = MIN(iir0 + TILE, ir0_end);
            const int n_src0 = (int) (iir0_end - iir0);

            // result buffer zeroed once per macrotile
            memset(tiled_ws.acc, 0, (size_t)TILED_TILE_ROWS * TILED_TILE_ROWS * sizeof(float));

            // Iterate K dimension by chunks of 256
            for (int64_t ib = 0; ib < ne00; ib += TILE) {
                // Unpack src0 and src1 into macrotiles
                const int kblk = (int) (ib / TILE);
                tiled_unpack_src0((const B *) (src0_row + iir0 * nb01 + kblk * src0_bs), src0_stride, n_src0, tiled_ws.src0);
                tiled_unpack_src1_q8_K(src1_col + kblk, src1_stride, n_src1, tiled_ws.src1,
                                       qv, nr1_pad, iir1, kblk);

                // 16x16 microtiles sweeping the window
                // Unpack routines zeropad our macrotiles outside the valid unpacked ranges, 
                // so invalid vals are harmless for final dotproducts, don't need to special case them
                for (int64_t ir0 = iir0; ir0 < iir0_end; ir0 += MICRO) {
                    for (int64_t ir1 = iir1; ir1 < iir1_end; ir1 += MICRO) {
                        tiled_run_microtile<SUBBLK, HAS_MIN, BIAS>(*tiled_ws.src0, *tiled_ws.src1,
                            (int) (ir0 - iir0), (int) (ir1 - iir1),
                            tiled_ws.acc, TILED_TILE_ROWS);
                    }
                }
            }
            // write acc back out from L2 to main memory
            tiled_store_window(tiled_ws.acc, n_src0, n_src1, TILED_TILE_ROWS,
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

    // interleave the whole tensor's src1 codes once (VNNI builds only)
#if defined(KERNEL_SRC1_UNPACK)
    tiled_prepare_src1_interleave(params, src1, vec_dot_type, ne10, ne11 * ne12 * ne13, ith, nth);
#endif

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
    int chunk_size = TILED_TILE_ROWS;

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

    // TODO:  if we KNOW we're on a machine where all cores are equal, we could skip the coordination/work-stealing and just assign chunks deterministically
    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_tiled_one_chunk<B, SUBBLK, HAS_MIN, BIAS>(params, dst, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

bool ggml_compute_forward_mul_mat_tiled(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    // --use-ref means bail out and go back to vec_dot reference impl
    if (params->use_ref) {
        return false;
    }
    if (!ggml_tiled_matmul_supported(dst->src[0], dst->src[1])) {
        return false;
    }
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q6_K:
            ggml_compute_forward_mul_mat_tiled_driver<block_q6_K, 16, false, 32>(params, dst);
            break;
        case GGML_TYPE_Q5_K:
            ggml_compute_forward_mul_mat_tiled_driver<block_q5_K, 32, true,  0>(params, dst);
            break;
        case GGML_TYPE_Q4_K:
            ggml_compute_forward_mul_mat_tiled_driver<block_q4_K, 32, true,  0>(params, dst);
            break;
        case GGML_TYPE_Q3_K:
            ggml_compute_forward_mul_mat_tiled_driver<block_q3_K, 16, false,  4>(params, dst);
            break;
        case GGML_TYPE_Q2_K:
            ggml_compute_forward_mul_mat_tiled_driver<block_q2_K, 16, true,  0>(params, dst);
            break;
        default:
            return false;
    }
    return true;
}
