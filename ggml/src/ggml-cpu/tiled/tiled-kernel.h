#pragma once

// Tiled matmul kernel API: tile structs, kernel declarations

// Currently only optimized for x86, new architectures should implement:
// tiled_run_microtile:  16x16 microkernel
// tiled_repack_src0: Optional repack/recalculation of src0, per macrotile
// tiled_repack_src1: Optional repack/recalculation of src1, per microtile-band
// unpack primitives: see the body headers selected by tiled-unpk.h

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <stddef.h>
#include <stdint.h>

#include "tiled-unpk.h"

#define TILED_TILE_K    256 // one QK_K block
#define TILED_TILE_ROWS 256 // max window rows, ragged at edges
#define TILED_MICRO     16  // microtile edge (also the bsums code-sum granularity)
#define TILED_WS_SLOT   (512 * 1024) // per-thread workspace slot, a clean 512KB (multiple of 64B)

// src0 tile: weight side, shared by all formats.
// scales/mins are sized for the max subblock count (SUBBLK=16);
// SUBBLK=32 formats index at stride 8 and leave the slack unused.
struct tiled_tile_src0 {
    static constexpr int NB_MAX = TILED_TILE_K / 16; // max subblocks per 256-elem block

    alignas(64) uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]; // unsigned quants, widened to uint8
    float    d[TILED_TILE_ROWS];  // One d from each input block, widened to f32
    float    dmin[TILED_TILE_ROWS]; // dmin from each input block (if applicable), widened to F32
    int32_t   scales[TILED_TILE_ROWS * NB_MAX];  // per-subblock scale, stored as int32_t
    int32_t   mins[TILED_TILE_ROWS * NB_MAX];    // per-subblock min, used when HAS_MIN
};

// src1 tile: built from q8_K (wdata)
struct tiled_tile_src1 {
    // q8 codes, one byte per element. Note for VNNI these are reshaped + transposed to be suitable for dpbusd.
    alignas(64) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];
    // per-16 code sums from q8_k (int16), widened to int32 so the kernels load them directly, no per-use cvt
    alignas(64) int32_t bsums[(TILED_TILE_K / 16) * TILED_TILE_ROWS];
    // f32 (not f16): q8_k stores fp16, the unpack converts once
    float       d[TILED_TILE_ROWS];
};

// per-thread workspace: all tiled state lives here, allocated in wdata (one slot per thread)
struct tiled_ws {
    tiled_tile_src0 src0;
    tiled_tile_src1 src1;
    alignas(64) float acc[TILED_TILE_ROWS * TILED_TILE_ROWS];
};

static_assert(sizeof(tiled_ws) <= TILED_WS_SLOT, "tiled workspace exceeds the 512KB per-thread slot");
// the slot base is 64B-aligned and TILED_WS_SLOT is a multiple of 64B, so each per-thread slot
// is 64B-aligned; these pin the tile fields and the acc buffer at aligned offsets within a slot
static_assert(offsetof(tiled_ws, src0) % 64 == 0, "src0 not 64B-aligned in the workspace");
static_assert(offsetof(tiled_ws, src1) % 64 == 0, "src1 not 64B-aligned in the workspace");
static_assert(offsetof(tiled_ws, acc)  % 64 == 0, "acc not 64B-aligned in the workspace");

// Accumulate one 16x16 microtile (src0 rows [i0, i0+16), src1 cols [j0, j0+16))
// over one 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial.
// SUBBLK/HAS_MIN/BIAS are the src0 format constants (see tiled_tile_src0).
// ACTBIAS (AVX2 only): the activation is pre-biased +128 by tiled_repack_src1,
// num_k = K-blocks per row: the tile holds num_k slabs at row stride num_k*256
// (Default case is num_k=1, 256x256 tiles, we go to longer num_k to improve memory bandwidth when num_rows is small)
template <int SUBBLK, bool HAS_MIN, int BIAS, bool ACTBIAS>
void tiled_run_microtile(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, int num_k, int slab, float * buf, int buf_stride);

// Optional repack, if profitable for the kernel.
// Repacks one 16-row band of src1 codes, called by driver as we reach each 16-row band in outer loop
void tiled_repack_src1(tiled_tile_src1 * src1, int row0, int num_k, bool bias);

// Optional repack, if profitable for the kernel
// Repack the entire src0 panel in place, called by the driver immediately after dequant
template <int SUBBLK>
void tiled_repack_src0(tiled_tile_src0 * tile, int n_rows, int num_k, int BIAS, bool corr);

// generic (scalar) microtile kernel, reference implementation.  Never actually called in production.
// The repacks need no _generic variant - they are no-ops for the scalar tier.
template <int SUBBLK, bool HAS_MIN, int BIAS, bool ACTBIAS>
void tiled_run_microtile_generic(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                 int i0, int j0, int num_k, int slab, float * buf, int buf_stride);
