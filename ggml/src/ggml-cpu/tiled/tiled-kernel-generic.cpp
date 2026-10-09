// mulmat microtile kernels: generic scalar reference implementation

#include "tiled-kernel.h"

#include "ggml.h"

#include <cstring>

// raw scalar MAC, one 16x16 microtile. NK: 1 = standard single-slab (strides compile-time),
// 0 = runtime-strided (the narrow path). BIAS is corrected in the int domain, so there is no act-bias variant.
template <int SUBBLK, bool HAS_MIN, int BIAS, int NK>
static void tiled_run_microtile_scalar(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                       int i0, int j0, int num_k, int slab, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;    // subblocks per 256-K block
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock

    // num_k/slab: the narrow path holds num_k slabs at row stride num_k*256 (num_k=1, slab=0 = standard)
    // NK: 1 = standard single-slab (offsets 0, strides compile-time); 0 = runtime
    // (num_k/slab from the args). The narrow path is memory-bound, so one runtime
    // version serves all of it.
    const int nkr  = (NK > 0) ? NK : num_k;
    const int seff = (NK > 0) ? 0 : slab;
    const int qk_stride = nkr * TILED_TILE_K;
    const int qk_off = seff * TILED_TILE_K;
    const int nb_stride = NB * nkr;
    const int nb_off = seff * NB;
    const int bs_stride = (nkr == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    const int bs_off = seff * TILED_MICRO;

    float acc[TILED_MICRO][TILED_MICRO];
    memset(acc, 0, sizeof(acc));

    // subdots at subblock granularity over the 256-K block, exact integer math
    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < TILED_MICRO; j++) {
            const int br = j0 + j;
            const int8_t * q1 = &src1.q[br * qk_stride + qk_off + s * SUBBLK];
            int32_t bsum = 0;
            for (int u = 0; u < NS; u++) {
                bsum += src1.bsums[(bs_off + s * NS + u) * bs_stride + br];
            }

            for (int i = 0; i < TILED_MICRO; i++) {
                const int ar = i0 + i;
                const int d_off = seff * TILED_MICRO + ar;
                const uint8_t * q0 = &src0.q[ar * qk_stride + qk_off + s * SUBBLK];

                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[e] * (int32_t) q1[e];
                }

                // BIAS: subtract BIAS*bsum (src1's per-subblock code sum) from the exact int raw
                int32_t corr = raw;
                if constexpr (BIAS != 0) {
                    corr -= BIAS * bsum;
                }
                const int32_t scales_raw = (int32_t) src0.scales[ar * nb_stride + nb_off + s] * corr;
                // d is NOT applied here: it is constant over the s-loop, so we apply it last before write-out
                if constexpr (HAS_MIN) {
                    const int32_t mins_bsum = (int32_t) src0.mins[ar * nb_stride + nb_off + s] * bsum;
                    acc[i][j] += (float) src0.d[d_off] * (float) scales_raw
                              - (float) src0.dmin[d_off] * (float) mins_bsum;
                } else {
                    acc[i][j] += (float) src0.d[d_off] * (float) scales_raw;
                }
            }
        }
    }

    // Apply d and write out to buf
    for (int i = 0; i < TILED_MICRO; i++) {
        for (int j = 0; j < TILED_MICRO; j++) {
            buf[(i0 + i) * buf_stride + (j0 + j)] += src1.d[seff * TILED_MICRO + j0 + j] * acc[i][j];
        }
    }
}

template <int SUBBLK, bool HAS_MIN, int BIAS, bool ACTBIAS>
void tiled_run_microtile_generic(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                 int i0, int j0, int num_k, int slab, float * buf, int buf_stride) {
    const bool standard = (num_k == 1);
    if (standard) {
        tiled_run_microtile_scalar<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, num_k, slab, buf, buf_stride);
    } else {
        tiled_run_microtile_scalar<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, num_k, slab, buf, buf_stride);
    }
}
// explicit instantiations: referenced cross-TU by the x86 #else rung
template void tiled_run_microtile_generic<32, true, 0, false>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
template void tiled_run_microtile_generic<32, false, 128, false>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                  int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
template void tiled_run_microtile_generic<32, false, 128, true>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                  int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
template void tiled_run_microtile_generic<16, false, 128, true>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                 int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
template void tiled_run_microtile_generic<16, false, 32, true>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                 int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
template void tiled_run_microtile_generic<16, false, 4, true>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                 int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
template void tiled_run_microtile_generic<16, true, 0, false>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, int num_k, int slab, float * buf, int buf_stride);

// layout guard for the src1 unpack (tiled.cpp reads block_q8_K directly)
static_assert(sizeof(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 repack");
