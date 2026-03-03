// HMX matrix multiplication operations.
// Ported from htp-ops-lib/src/dsp/ops/mat_mul.c.
//
// Changes from the original:
//   - All symbols prefixed with hmx_ to avoid collisions with hexagon HVX headers.
//   - dma_utils.h replaced with hex-dma.h queue-based DMA.
//   - vtcm_manager / hmx-mgr globals replaced with htp_context fields.
//   - hmx_worker_pool replaced with main worker_pool API.
//   - ggml_type enum replaced with int weight_type + HMX_TYPE_* constants.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hmx-utils.h"
#include "hmx-hvx-convert.h"
#include "hmx-hvx-internal.h"
#include "hmx-quants.h"
#include "htp-ctx.h"
#include "worker-pool.h"
#include "hex-dma.h"

#include <HAP_compute_res.h>

// debug & profile
#include <HAP_farf.h>
#include "hmx-profile.h"

#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (1 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (1 * 1024 * 1024)
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)

// ---------- Debug: check fp16/fp32 buffers for NaN / Inf / near-overflow ----------
// Set to 1 to enable runtime anomaly checks (adds overhead, use for debugging only).
#define HMX_DEBUG_CHECK_VALUES 0

// Set to 1 to enable lightweight sample-value tracing via FARF(ALWAYS, ...).
#define HMX_DEBUG_TRACE_VALUES 1

#if HMX_DEBUG_CHECK_VALUES

#include <math.h>

// Scan n_elms fp16 values; report NaN, Inf, and near-overflow (|v| > 60000).
static void hmx_check_fp16(const __fp16 *buf, int n_elms, const char *label,
                            int chunk_mr, int chunk_nc) {
  int n_nan = 0, n_inf = 0, n_large = 0;
  int first_nan = -1, first_inf = -1;
  float max_abs = 0.0f;

  const uint16_t *bits = (const uint16_t *)buf;
  for (int i = 0; i < n_elms; ++i) {
    uint16_t exp  = (bits[i] >> 10) & 0x1F;
    uint16_t mant = bits[i] & 0x3FF;
    if (exp == 0x1F) {
      if (mant != 0) { if (first_nan < 0) first_nan = i; ++n_nan; }
      else            { if (first_inf < 0) first_inf = i; ++n_inf; }
      continue;
    }
    float v = (float)buf[i];
    float a = v < 0 ? -v : v;
    if (a > max_abs) max_abs = a;
    if (a > 60000.0f) ++n_large;
  }

  if (n_nan || n_inf || n_large) {
    FARF(ALWAYS, "HMX CHECK [%s] mr=%d nc=%d : nan=%d(first@%d) inf=%d(first@%d) "
                 "near_ovf=%d max_abs=%.1f  n_elms=%d",
         label, chunk_mr, chunk_nc,
         n_nan, first_nan, n_inf, first_inf, n_large, max_abs, n_elms);
  }
}

// Scan n_elms fp32 values; report NaN and Inf.
static void hmx_check_fp32(const float *buf, int n_elms, int stride,
                            int n_rows, int n_cols, const char *label,
                            int chunk_mr, int chunk_nc) {
  int n_nan = 0, n_inf = 0;
  int first_nan = -1, first_inf = -1;
  float max_abs = 0.0f;

  for (int r = 0; r < n_rows; ++r) {
    const float *row = buf + r * stride;
    for (int c = 0; c < n_cols; ++c) {
      float v = row[c];
      if (isnan(v))      { if (first_nan < 0) first_nan = r * n_cols + c; ++n_nan; }
      else if (isinf(v)) { if (first_inf < 0) first_inf = r * n_cols + c; ++n_inf; }
      else {
        float a = v < 0 ? -v : v;
        if (a > max_abs) max_abs = a;
      }
    }
  }

  if (n_nan || n_inf) {
    FARF(ALWAYS, "HMX CHECK [%s] mr=%d nc=%d : nan=%d(first@%d) inf=%d(first@%d) "
                 "max_abs=%.4f  rows=%d cols=%d",
         label, chunk_mr, chunk_nc,
         n_nan, first_nan, n_inf, first_inf, max_abs, n_rows, n_cols);
  }
}

// Print first few values for visual inspection.
static void hmx_dump_fp16_head(const __fp16 *buf, int n_elms, const char *label,
                                int chunk_mr, int chunk_nc) {
  int n = n_elms < 8 ? n_elms : 8;
  float v[8];
  for (int i = 0; i < n; ++i) v[i] = (float)buf[i];
  FARF(ALWAYS, "HMX DUMP [%s] mr=%d nc=%d first %d: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
       label, chunk_mr, chunk_nc, n,
       n > 0 ? v[0] : 0, n > 1 ? v[1] : 0, n > 2 ? v[2] : 0, n > 3 ? v[3] : 0,
       n > 4 ? v[4] : 0, n > 5 ? v[5] : 0, n > 6 ? v[6] : 0, n > 7 ? v[7] : 0);
}

#define HMX_CHECK_FP16(buf, n, label, mr, nc)       hmx_check_fp16((buf), (n), (label), (mr), (nc))
#define HMX_CHECK_FP32(buf, n, s, nr, ncols, l, mr, nc) hmx_check_fp32((buf), (n), (s), (nr), (ncols), (l), (mr), (nc))
#define HMX_DUMP_FP16(buf, n, label, mr, nc)        hmx_dump_fp16_head((buf), (n), (label), (mr), (nc))


// Dump first n_vecs vectors of a tile in raw memory-contiguous order.
// Each HVX vector = 64 fp16 values; prints first 8 of each.
static void hmx_dump_tile_mem(const __fp16 *tile, const char *label,
                               int tile_row, int tile_col, int n_vecs) {
    if (n_vecs > 16) n_vecs = 16;
    for (int v = 0; v < n_vecs; ++v) {
        const __fp16 *p = tile + v * 64;
        FARF(ALWAYS, "TILE_MEM [%s] tr=%d tc=%d v%02d: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             label, tile_row, tile_col, v,
             (float)p[0], (float)p[1], (float)p[2], (float)p[3],
             (float)p[4], (float)p[5], (float)p[6], (float)p[7]);
    }
}

// Dump fp32 values in row-major layout for a tile region.
// Prints n_rows rows of 8 values starting at column = tile_col * 32.
static void hmx_dump_fp32_tile_region(const float *data, int stride,
                                       const char *label, int tile_row, int tile_col,
                                       int n_rows) {
    int col_off = tile_col * 32;
    if (n_rows > 32) n_rows = 32;
    for (int r = 0; r < n_rows; ++r) {
        const float *p = data + r * stride + col_off;
        FARF(ALWAYS, "FP32_MEM [%s] tr=%d tc=%d row%d: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
             label, tile_row, tile_col, r,
             p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }
}

// Dump first 8 fp16 values from an HVX vector (spill to aligned stack buffer).
// Implemented as macro to avoid HVX_Vector in function signature (IDE-friendly).
#define HMX_DUMP_HVX_FP16(v, label, t, r) do { \
    __fp16 _dbg_h16[64] __attribute__((aligned(128))); \
    *(HVX_Vector *)_dbg_h16 = (v); \
    FARF(ALWAYS, "HVX_FP16 [%s] t=%d r=%d: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", \
         (label), (t), (r), \
         (float)_dbg_h16[0], (float)_dbg_h16[1], (float)_dbg_h16[2], (float)_dbg_h16[3], \
         (float)_dbg_h16[4], (float)_dbg_h16[5], (float)_dbg_h16[6], (float)_dbg_h16[7]); \
} while (0)

// Dump first 8 fp32 values from an HVX vector.
#define HMX_DUMP_HVX_FP32(v, label, t, r) do { \
    float _dbg_f32[32] __attribute__((aligned(128))); \
    *(HVX_Vector *)_dbg_f32 = (v); \
    FARF(ALWAYS, "HVX_FP32 [%s] t=%d r=%d: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", \
         (label), (t), (r), \
         _dbg_f32[0], _dbg_f32[1], _dbg_f32[2], _dbg_f32[3], \
         _dbg_f32[4], _dbg_f32[5], _dbg_f32[6], _dbg_f32[7]); \
} while (0)
#define HMX_DUMP_TILE_MEM(tile, label, tr, tc, nv) hmx_dump_tile_mem((tile), (label), (tr), (tc), (nv))
#define HMX_DUMP_FP32_TILE(data, stride, label, tr, tc, nr) hmx_dump_fp32_tile_region((data), (stride), (label), (tr), (tc), (nr))
#define HMX_DUMP_TILE_ROWS(tile, label, r, c) hmx_dump_tile_rows((tile), (label), (r), (c))

#else

#define HMX_CHECK_FP16(buf, n, label, mr, nc)            ((void)0)
#define HMX_CHECK_FP32(buf, n, s, nr, ncols, l, mr, nc) ((void)0)
#define HMX_DUMP_FP16(buf, n, label, mr, nc)             ((void)0)
#define HMX_DUMP_HVX_FP16(v, label, t, r)               ((void)0)
#define HMX_DUMP_HVX_FP32(v, label, t, r)               ((void)0)
#define HMX_DUMP_TILE_MEM(tile, label, tr, tc, nv)       ((void)0)
#define HMX_DUMP_FP32_TILE(data, stride, label, tr, tc, nr) ((void)0)
#define HMX_DUMP_TILE_ROWS(tile, label, r, c)            ((void)0)

#endif // HMX_DEBUG_CHECK_VALUES
// ---------- End debug utilities ----------

static const __fp16 q4_0_to_fp16_lut[64] __attribute__((aligned(HMX_VLEN))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

static const __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(HMX_VLEN))) = {
  -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
  1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128 in the tile.
// Column offset (n*4) is added at runtime.  Only entries 0..15 are used (masked by predicate).
static const int32_t weight_transpose_scatter_offsets[32] __attribute__((aligned(HMX_VLEN))) = {
    0*128,  1*128,  2*128,  3*128,  4*128,  5*128,  6*128,  7*128,
    8*128,  9*128, 10*128, 11*128, 12*128, 13*128, 14*128, 15*128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static inline void swap_ptr(void **p1, void **p2) {
  void *t = *p1;
  *p1     = *p2;
  *p2     = t;
}

// Compute the byte stride of one row in x4x2 format.
// Numerically equals ggml_row_size(type, k) when k is 256-aligned, because
// x4x2 packing has the same density as block_q4_0 / block_q8_0.
// Layout per row: [quants: nb*128 (Q4) or nb*256 (Q8)][scales: nb*16 bytes]
// Total per row = nb * (128+16) = 144*nb (Q4) or nb * (256+16) = 272*nb (Q8).
// Callers must ensure k is a multiple of 256 (enforced by proc_hmx_matmul_req).
static inline size_t get_x4x2_row_stride(int weight_type, int k) {
  int nb = (k + HMX_QK_Q4x4x2 - 1) / HMX_QK_Q4x4x2;
  switch (weight_type) {
    case HMX_TYPE_Q4_0:
    case HMX_TYPE_IQ4_NL:
      return (size_t)nb * (HMX_QK_Q4x4x2 / 2 + HMX_X4X2_DBLK_SIZE);  // 144 * nb
    case HMX_TYPE_Q8_0:
      return (size_t)nb * (HMX_QK_Q8x4x2 + HMX_X4X2_DBLK_SIZE);      // 272 * nb
    default:
      return 0;
  }
}


static void find_chunk_size(size_t x_max, size_t y_max, size_t xy_max, size_t x_unit, size_t y_unit, size_t *x_out,
                            size_t *y_out) {
  int64_t best_xy = 0;
  size_t  best_x = 0, best_y = 0;

  for (size_t x = x_max; x > 0; x -= x_unit) {
    size_t  y  = hmx_smin(hmx_align_down(xy_max / x, y_unit), y_max);
    int64_t xy = x * y;
    if (best_xy < xy) {
      best_xy = xy;
      best_x = x, best_y = y;
    }
  }
  *x_out = best_x, *y_out = best_y;
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                                   int k_block, int k_stride) {
  assert(k_block % HMX_FP16_TILE_N_COLS == 0 && k_stride % HMX_FP16_TILE_N_COLS == 0);
  assert(HMX_VLEN == 32 * sizeof(float));

  for (int r = 0; r < n_rows; r += 2) {
    int prefetch_row_idx = r + 2;
    if (prefetch_row_idx < n_rows) {
      const float *prefetch_addr = src + prefetch_row_idx * k_stride;
      // NOTE(hzx): prefetch 2 rows at a time
      hmx_l2fetch(prefetch_addr, k_stride * sizeof(float), k_block * sizeof(float), 2, 0);
    }

    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);
    for (int c = 0; c < k_block; c += 32) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

      HVX_Vector v_out = hmx_hvx_wsf_to_vhf(v1, v0);

      // compute output position
      int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
      int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

#if HMX_DEBUG_CHECK_VALUES
      if (r == 0 && (tile_idx == 0 || tile_idx == 7)) {
        FARF(ALWAYS, "=== ACT shuffle tile(%d) r=%d  wsf_to_vhf ===", tile_idx, r);
        HMX_DUMP_HVX_FP32(v0, "act_row0_pre_shuf", tile_idx, r);
        HMX_DUMP_HVX_FP32(v1, "act_row1_pre_shuf", tile_idx, r);
        HMX_DUMP_HVX_FP16(v_out, "act_post_shuf", tile_idx, r);
      }
#endif

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

// Scatter row-major FP16 weight (already in VTCM scratch) directly into transposed [K][N] tiles.
// vtcm_src: [n_cols][k] row-major fp16 in VTCM scratch buffer
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
static void interleave_fp16_weight_chunk_to_tiles(__fp16 *restrict vtcm_dst,
                                                   const __fp16 *restrict vtcm_src,
                                                   int n_cols, int k) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  assert(k % HMX_FP16_TILE_N_COLS == 0);

  const int n_k_tiles = k / HMX_FP16_TILE_N_COLS;
  const HVX_Vector v_scat_base = hmx_vmem(weight_transpose_scatter_offsets);
  const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
  const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

  for (int r = 0; r < n_cols; r += 2) {
    int ct = r / HMX_FP16_TILE_N_ROWS;       // N-dimension tile index
    int local_r = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row index
    const bool next_row_valid = (r + 1) < n_cols;

    // Offset vectors for N-columns local_r and local_r+1, reused across K-tiles.
    HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
    HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

    for (int c = 0; c < k; c += HMX_FP16_TILE_N_COLS) {
      int kt       = c / HMX_FP16_TILE_N_COLS;
      int tile_idx = ct * n_k_tiles + kt;
      __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v0 = hmx_vmemu(vtcm_src + r * k + c);
      HVX_Vector v1 = next_row_valid
          ? hmx_vmemu(vtcm_src + (r + 1) * k + c)
          : Q6_V_vzero();

      Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off0, v0);
      Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off1, v1);
    }
  }
}

// --- x4x2 format dequantizers ---

// Dequantize one x4x2 Q4_0 group (32 elements from 32 packed bytes) → 32 FP16 in first 64 bytes.
// In x4x2, sub-blocks 0..3 use lower nibbles, sub-blocks 4..7 use upper nibbles
// of the same 32 packed bytes.
static inline HVX_Vector dequantize_x4x2_q4_0_group_hvx(
    const uint8_t *packed_32, bool upper_nibbles,
    const __fp16 *scale, const HVX_Vector vlut_cvt) {
  HVX_Vector vq = hmx_vmemu(packed_32);
  const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
  HVX_Vector v_scales = Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)scale));
  // q4x4x2 stores two int4 values per byte. Keep only the selected nibble.
  HVX_Vector v_quants = upper_nibbles ? Q6_Vub_vlsr_VubR(vq, 4) : vq;
  v_quants = Q6_V_vand_VV(v_quants, mask_h4);
  // Use standard vlut16 (not _nomatch) to avoid stale-register NaN.
  // _nomatch retains the previous destination-register value for colliding
  // indices, but the C intrinsic doesn't model the implicit read so the
  // compiler may allocate a register containing garbage/NaN.
  HVX_VectorPair vp = Q6_Wh_vlut16_VbVhR(v_quants, vlut_cvt, 0);
  // vlut16 produces interleaved output: even-indexed byte results in lo,
  // odd-indexed in hi.  Deinterleave with vshuff to restore linear order.
  HVX_Vector v_hf = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -2));

#if HMX_DEBUG_TRACE_VALUES
  // Check for anomalous values AFTER vlut16 (before multiply).
  // Valid LUT outputs are -8..+7 (max exp=0x13 in fp16).  Any exp >= 0x18 is corrupt.
  {
    __attribute__((aligned(128))) uint16_t _vhf_buf[64];
    *(HVX_Vector *)_vhf_buf = v_hf;
    int _bad = 0;
    for (int _j = 0; _j < 32; ++_j) {
      uint16_t _e = (_vhf_buf[_j] >> 10) & 0x1F;
      if (_e >= 0x18) _bad++;
    }
    if (_bad > 0) {
      uint16_t _sc = *(const uint16_t *)scale;
      FARF(ALWAYS, "  DBG DEQUANT-INNER: %d bad values AFTER vlut16 (before vmpy), scale=0x%04x, "
           "vlut[0..7]=%04x %04x %04x %04x %04x %04x %04x %04x",
           _bad, _sc,
           _vhf_buf[0], _vhf_buf[1], _vhf_buf[2], _vhf_buf[3],
           _vhf_buf[4], _vhf_buf[5], _vhf_buf[6], _vhf_buf[7]);
    }
  }
#endif

  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hf, v_scales));
}

// Dequantize one x4x2 Q8_0 group (32 int8 quants) → 32 FP16 in first 64 bytes.
static inline HVX_Vector dequantize_x4x2_q8_0_group_hvx(
    const int8_t *quants_32, const __fp16 *scale) {
  HVX_Vector vq = hmx_vmemu(quants_32);
  HVX_Vector v_scales = Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)scale));
  HVX_Vector v0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vq));
  HVX_Vector v_hf = Q6_Vhf_equals_Vh(v0);
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hf, v_scales));
}

// Dequantize a tile range from x4x2 weight data (already in VTCM) to tile-major FP16.
// Input:  vtcm_src has n_cols rows of x4x2 data, each row_stride bytes.
// Output: vtcm_dst in tile-major FP16 layout.
static void dequantize_x4x2_weight_to_fp16_tiles_task(
    __fp16 *restrict vtcm_dst,
    const uint8_t *restrict vtcm_src,
    int n_cols, int k_block,
    size_t row_stride, int weight_type,
    int start_tile, int end_tile) {

  const int n_k_tiles = k_block / HMX_FP16_TILE_N_COLS;
  const bool is_q4 = (weight_type == HMX_TYPE_Q4_0 || weight_type == HMX_TYPE_IQ4_NL);
  const int qrow_size = is_q4 ? (k_block / 2) : k_block;

  const HVX_Vector vlut_cvt = (weight_type == HMX_TYPE_IQ4_NL)
      ? hmx_vmem(iq4_nl_to_fp16_lut) : hmx_vmem(q4_0_to_fp16_lut);

  // vscatter setup: write dequantized K-values directly to transposed [K][N] tile positions.
  // Each int32 element holds a K-row-pair (2 adjacent fp16 values).  word[i] at offset i*128
  // maps to K-rows 2i and 2i+1.  Column offset (n*4) added per row.
  const HVX_Vector v_scat_base = hmx_vmem(weight_transpose_scatter_offsets);
  const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);  // 4 bytes = 1 column step
  const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);  // first 16 words (64 bytes)

  for (int t = start_tile; t < end_tile; ++t) {
    int ct = t / n_k_tiles;  // column tile index
    int kt = t % n_k_tiles;  // K tile index

    __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

    if (is_q4) {
      int blk_idx  = (kt * 32) / HMX_QK_Q4x4x2;
      int sub_blk  = ((kt * 32) % HMX_QK_Q4x4x2) / 32;
      bool upper   = (sub_blk >= 4);
      int byte_off  = blk_idx * (HMX_QK_Q4x4x2 / 2) + (upper ? (sub_blk - 4) : sub_blk) * 32;
      int scale_off = qrow_size + blk_idx * HMX_X4X2_DBLK_SIZE + sub_blk * (int)sizeof(__fp16);

      HVX_Vector v_off = v_scat_base;  // reset to column 0
      for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
        int row0 = ct * HMX_FP16_TILE_N_COLS + r;
        int row1 = row0 + 1;

        const uint8_t *r0 = vtcm_src + row0 * row_stride;
        const uint8_t *r1 = vtcm_src + row1 * row_stride;

        HVX_Vector v0 = dequantize_x4x2_q4_0_group_hvx(
            r0 + byte_off, upper, (const __fp16 *)(r0 + scale_off), vlut_cvt);
        HVX_Vector v1 = (row1 < n_cols)
            ? dequantize_x4x2_q4_0_group_hvx(
                r1 + byte_off, upper, (const __fp16 *)(r1 + scale_off), vlut_cvt)
            : Q6_V_vzero();

#if HMX_DEBUG_TRACE_VALUES
        // Check dequant output BEFORE scatter — show scale value and offset for DMA correlation
        if (t == start_tile) {
          __attribute__((aligned(128))) uint16_t _dq[64];
          *(HVX_Vector *)_dq = v0;
          int _nq0 = 0;
          for (int _j = 0; _j < 32; ++_j) { if (_dq[_j] == 0xFFFF) _nq0++; }
          if (_nq0 > 0) {
            uint16_t _sc = *(const uint16_t *)(r0 + scale_off);
            FARF(ALWAYS, "  DBG DEQUANT-PRE-SCATTER: tile %d r=%d row0=%d v0 has %d NaN, scale=0x%04x byte_off=%d scale_off=%d blk=%d sub=%d",
                 t, r, (int)(ct * HMX_FP16_TILE_N_COLS + r), _nq0, _sc, byte_off, scale_off, blk_idx, sub_blk);
          }
          *(HVX_Vector *)_dq = v1;
          int _nq1 = 0;
          for (int _j = 0; _j < 32; ++_j) { if (_dq[_j] == 0xFFFF) _nq1++; }
          if (_nq1 > 0) {
            uint16_t _sc = *(const uint16_t *)(r1 + scale_off);
            FARF(ALWAYS, "  DBG DEQUANT-PRE-SCATTER: tile %d r=%d row1=%d v1 has %d NaN, scale=0x%04x byte_off=%d scale_off=%d blk=%d sub=%d",
                 t, r, (int)(ct * HMX_FP16_TILE_N_COLS + r + 1), _nq1, _sc, byte_off, scale_off, blk_idx, sub_blk);
          }
        }
#endif

        // Scatter each row's 32 K-values directly to transposed [K][N] positions.
        // word[i] = {k=2i, k=2i+1} written to K-row-pair i, N-column r / r+1.
        Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v0);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v1);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
      }
      // Drain scatter buffer after each tile to prevent HVX scatter FIFO overflow.
      // 32 scatter ops per tile; without periodic drains the buffer overflows for
      // large matrices (e.g. 128 tiles = 4096 scatters) and early writes are lost.
      (void) *(volatile HVX_Vector *)(tile_base);

#if HMX_DEBUG_TRACE_VALUES
      // Verify tile data immediately after scatter+drain
      if (t == start_tile) {
        const uint16_t *_tr = (const uint16_t *)tile_base;
        int _nan_scat = 0;
        for (int _j = 0; _j < 1024; ++_j) { if (_tr[_j] == 0xFFFF) _nan_scat++; }
        if (_nan_scat > 0) {
          FARF(ALWAYS, "  DBG SCATTER-READBACK: tile %d has %d NaN AFTER scatter+drain, [6..9]=%04x %04x %04x %04x",
               t, _nan_scat, _tr[6], _tr[7], _tr[8], _tr[9]);
        } else {
          FARF(ALWAYS, "  DBG SCATTER-READBACK: tile %d OK (0 NaN after scatter+drain)", t);
        }
      }
#endif
    } else {
      // Q8_0
      int blk_idx  = (kt * 32) / HMX_QK_Q8x4x2;
      int sub_blk  = ((kt * 32) % HMX_QK_Q8x4x2) / 32;
      int byte_off  = blk_idx * HMX_QK_Q8x4x2 + sub_blk * 32;
      int scale_off = qrow_size + blk_idx * HMX_X4X2_DBLK_SIZE + sub_blk * (int)sizeof(__fp16);

      HVX_Vector v_off = v_scat_base;  // reset to column 0
      for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
        int row0 = ct * HMX_FP16_TILE_N_COLS + r;
        int row1 = row0 + 1;

        const uint8_t *r0 = vtcm_src + row0 * row_stride;
        const uint8_t *r1 = vtcm_src + row1 * row_stride;

        HVX_Vector v0 = dequantize_x4x2_q8_0_group_hvx(
            (const int8_t *)(r0 + byte_off), (const __fp16 *)(r0 + scale_off));
        HVX_Vector v1 = (row1 < n_cols)
            ? dequantize_x4x2_q8_0_group_hvx(
                (const int8_t *)(r1 + byte_off), (const __fp16 *)(r1 + scale_off))
            : Q6_V_vzero();

        Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v0);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v1);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
      }
      // Drain scatter buffer after each tile (same reason as Q4_0 path above)
      (void) *(volatile HVX_Vector *)(tile_base);
    }
  }

  // Drain HVX scatter write buffer: a vmem load on the same HW thread retires
  // all pending scatter entries to VTCM.  Without this, the main thread's HMX
  // reads may see stale data because atomic_fetch_sub (release) only orders
  // regular stores, not the HVX scatter buffer.
  if (start_tile < end_tile) {
    (void) *(volatile HVX_Vector *)(vtcm_dst + (end_tile - 1) * HMX_FP16_TILE_N_ELMS);
  }
}

typedef struct {
  __fp16        *dst;
  const uint8_t *src;
  int            n_cols;
  int            k_block;
  size_t         row_stride;
  int            weight_type;
  int            n_tot_tiles;
  int            n_tiles_per_task;
  int            n_tasks;
} x4x2_dequantize_state_t;

static void dequantize_x4x2_worker_loop(unsigned int n, unsigned int i, void *data) {
  x4x2_dequantize_state_t *state = (x4x2_dequantize_state_t *)data;

  for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
    int start = task_id * state->n_tiles_per_task;
    int end   = hmx_smin(start + state->n_tiles_per_task, state->n_tot_tiles);

    dequantize_x4x2_weight_to_fp16_tiles_task(
        state->dst, state->src, state->n_cols, state->k_block,
        state->row_stride, state->weight_type, start, end);
  }
}

static void dequantize_x4x2_weight_chunk_to_fp16_tiles(
    struct htp_context *ctx, __fp16 *vtcm_dst,
    const void *vtcm_src, int n_cols, int k_block,
    size_t row_stride, int weight_type) {

  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  assert(k_block % HMX_FP16_TILE_N_COLS == 0);

  int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
  int n_k_tiles   = k_block / HMX_FP16_TILE_N_COLS;
  int n_tot_tiles = n_col_tiles * n_k_tiles;

  size_t n_tiles_per_task = hmx_ceil_div(n_tot_tiles, ctx->n_threads);

  x4x2_dequantize_state_t state;
  state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
  state.n_tot_tiles      = n_tot_tiles;
  state.n_tiles_per_task = n_tiles_per_task;
  state.dst         = vtcm_dst;
  state.src         = (const uint8_t *)vtcm_src;
  state.n_cols      = n_cols;
  state.k_block     = k_block;
  state.row_stride  = row_stride;
  state.weight_type = weight_type;

  worker_pool_run_func(ctx->worker_pool, dequantize_x4x2_worker_loop, &state, ctx->n_threads);
}

// --- End x4x2 dequantizers ---

#if HMX_DEBUG_CHECK_VALUES
// Reference matmul verifier: computes expected output from tile-major fp16 buffers
// in scalar fp32, then dumps both reference and HMX results row by row via FARF.
//
// Tile interleave layout (activation, weight, output all share the same format):
//   element(row, col) = tile[(row>>1)*64 + col*2 + (row&1)]
//   Each HVX vector (128 B = 64 fp16) holds two interleaved adjacent rows:
//     even fp16 slots → row_r, odd fp16 slots → row_r+1.
static void hmx_dump_ref_vs_hmx(
    const __fp16 *activation,  // [n_row_tiles][n_dot_tiles][TILE_ELMS] tile-major fp16
    const __fp16 *weight,      // [n_col_tiles][n_dot_tiles][TILE_ELMS] tile-major fp16
    const __fp16 *hmx_out,     // [n_row_tiles][n_col_tiles][TILE_ELMS] tile-major fp16
    int n_row_tiles, int n_col_tiles, int n_dot_tiles)
{
  const int TC = HMX_FP16_TILE_N_COLS;  // 32
  const int TR = HMX_FP16_TILE_N_ROWS;  // 32
  const int VL = TC * 2;                 // 64 fp16 per HVX vector (two interleaved rows)

  // Access element (row, col) inside a tile pointer.
#define TELEM(tile, row, col) \
  ((tile)[((row) >> 1) * VL + (col) * 2 + ((row) & 1)])

  FARF(ALWAYS, "=== hmx_dump_ref_vs_hmx n_rt=%d n_ct=%d n_kt=%d ===",
       n_row_tiles, n_col_tiles, n_dot_tiles);

  // --- Pass 1: all REF rows (scalar fp32) ---
  FARF(ALWAYS, "--- REF (scalar fp32) ---");
  for (int rt = 0; rt < n_row_tiles; ++rt) {
    for (int ri = 0; ri < TR; ++ri) {
      int global_row = rt * TR + ri;
      for (int ct = 0; ct < n_col_tiles; ++ct) {
        int c_base = ct * TC;
        char buf[320];
        int p = 0;
        for (int ci = 0; ci < TC; ++ci) {
          float acc = 0.0f;
          for (int kt = 0; kt < n_dot_tiles; ++kt) {
            const __fp16 *at =
                activation + (rt * n_dot_tiles + kt) * HMX_FP16_TILE_N_ELMS;
            const __fp16 *wt =
                weight + (ct * n_dot_tiles + kt) * HMX_FP16_TILE_N_ELMS;
            for (int ki = 0; ki < TC; ++ki) {
              // Weight tile is now [K][N]: ki=K-row, ci=N-col.
              acc += (float)TELEM(at, ri, ki) * (float)TELEM(wt, ki, ci);
            }
          }
          p += snprintf(buf + p, (int)sizeof(buf) - p, " %7.3f", acc);
        }
        FARF(ALWAYS, "REF[%d][c%d-%d]:%s", global_row, c_base, c_base + TC - 1, buf);
      }
    }
  }

  // --- Pass 2: all HMX rows (core_dot_chunk_fp16 output) ---
  FARF(ALWAYS, "--- HMX (core_dot_chunk_fp16) ---");
  for (int rt = 0; rt < n_row_tiles; ++rt) {
    for (int ri = 0; ri < TR; ++ri) {
      int global_row = rt * TR + ri;
      for (int ct = 0; ct < n_col_tiles; ++ct) {
        const __fp16 *out_tile =
            hmx_out + (rt * n_col_tiles + ct) * HMX_FP16_TILE_N_ELMS;
        int c_base = ct * TC;
        char buf[320];
        int p = 0;
        for (int ci = 0; ci < TC; ++ci) {
          p += snprintf(buf + p, (int)sizeof(buf) - p,
                        " %7.3f", (float)TELEM(out_tile, ri, ci));
        }
        FARF(ALWAYS, "HMX[%d][c%d-%d]:%s", global_row, c_base, c_base + TC - 1, buf);
      }
    }
  }

#undef TELEM
}
#endif // HMX_DEBUG_CHECK_VALUES

static void core_dot_chunk_fp16(__fp16 *output, const __fp16 *activation, const __fp16 *weight, const __fp16 *scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles, uint32_t vtcm_rctx) {
  HAP_compute_res_hmx_lock(vtcm_rctx);

  hmx_set_output_scales(scales);

  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < n_col_tiles; ++c) {
      // asm volatile("mxclracc.hf");
      Q6_mxclracc_hf();

      const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < n_dot_tiles; ++k) {
        int offset = k * HMX_FP16_TILE_N_ELMS;
        hmx_load_tile_pair_fp16(row_tiles + offset, col_tiles + offset);
      }

      __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(out_tile);
    }
  }

  HAP_compute_res_hmx_unlock(vtcm_rctx);
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows,
                                               int n_cols, int n) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;
    int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      int c0 = c / HMX_FP16_TILE_N_COLS;

      const __fp16 *tile = vtcm_src + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v_src = ((const HVX_Vector *) tile)[r1 / 2];

      HVX_VectorPair vp = hmx_hvx_vhf_to_wsf(v_src);

#if HMX_DEBUG_CHECK_VALUES
      if (r == 0 && (c0 == 0 || c0 == 7)) {
        FARF(ALWAYS, "=== OUT de-shuffle tile(0,%d) r=%d  vhf_to_wsf ===", c0, r);
        HMX_DUMP_HVX_FP16(v_src, "out_pre_deshuf", c0, r);
        HMX_DUMP_HVX_FP32(Q6_V_lo_W(vp), "out_row0_post_deshuf", c0, r);
        HMX_DUMP_HVX_FP32(Q6_V_hi_W(vp), "out_row1_post_deshuf", c0, r);
      }
#endif

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + (r * n + c + 0));
      HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r * n + c + n));  // next row in global memory

      *pv_out0 = Q6_V_lo_W(vp);
      if (r + 1 < n_rows) {
        *pv_out1 = Q6_V_hi_W(vp);
      }
    }
  }
}

int hmx_mat_mul_permuted_w16a32(struct htp_context *ctx, float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict permuted_weight, int m, int k, int n) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!hmx_is_aligned(dst, HMX_VLEN) || !hmx_is_aligned(activation, HMX_VLEN) || !hmx_is_aligned(permuted_weight, HMX_VLEN)) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = OUTPUT_AREA_SIZE;
  const size_t scratch_area_size    = SCRATCH_AREA_SIZE;

  // VTCM layout: weight | activation | output | scratch0 | scratch1 | scales
  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  assert((size_t)(vtcm_ptr - ctx->vtcm_base) <= ctx->vtcm_scratch_size);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = hmx_align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = hmx_align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  TIMER_DEFINE(activation_load);
  TIMER_DEFINE(weight_load);
  TIMER_DEFINE(hmx_core);
  TIMER_DEFINE(output_store);

  TIMER_DEFINE(total);
  TIMER_START(total);

  for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
    // transfer activation matrix chunk into VTCM
    size_t n_rows = hmx_smin(m - mr, m_chunk_n_rows);

    TIMER_START(activation_load);
    {
      const float *activation_chunk = activation + mr * k;
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
    }
    TIMER_STOP(activation_load);

    const size_t fp16_row_stride = k * sizeof(__fp16);

    void *buf_curr = vtcm_scratch0;
    void *buf_next = vtcm_scratch1;

    // issue async DMA for the first weight chunk
    // NOTE: use 2D DMA (n_cols rows × fp16_row_stride) to avoid 16-bit roiwidth overflow.
    {
      const size_t n_cols_first = hmx_smin(n, n_chunk_n_cols);

      dma_queue_push(ctx->dma[0], dma_make_ptr(buf_curr, permuted_weight),
                     fp16_row_stride, fp16_row_stride, fp16_row_stride, n_cols_first);
    }

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = hmx_smin(n - nc, n_chunk_n_cols);

      TIMER_START(weight_load);
      {
        dma_queue_pop(ctx->dma[0]);  // wait until current weight chunk is ready

        // issue async DMA for the next weight chunk (double buffering)
        const size_t nc_next = nc + n_chunk_n_cols;
        if (nc_next < n) {
          const size_t n_cols_next       = hmx_smin(n - nc_next, n_chunk_n_cols);
          const __fp16 *next_weight_chunk = permuted_weight + nc_next * k;

          dma_queue_push(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk),
                         fp16_row_stride, fp16_row_stride, fp16_row_stride, n_cols_next);
        }

        // interleave row-major fp16 from scratch into tile-major in vtcm_weight
        interleave_fp16_weight_chunk_to_tiles(vtcm_weight, (const __fp16 *)buf_curr, n_cols, k);

        swap_ptr(&buf_curr, &buf_next);
      }
      TIMER_STOP(weight_load);

      TIMER_START(hmx_core);
      {
        const int n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        const int n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32, ctx->vtcm_rctx);
      }
      TIMER_STOP(hmx_core);

#if HMX_DEBUG_CHECK_VALUES
      {
        int _nrt = hmx_ceil_div((int)n_rows, HMX_FP16_TILE_N_ROWS);
        int _nct = hmx_ceil_div((int)n_cols, HMX_FP16_TILE_N_COLS);
        hmx_dump_ref_vs_hmx(vtcm_activation, vtcm_weight, vtcm_output,
                            _nrt, _nct, (int)(k / 32));
      }
#endif

      TIMER_START(output_store);
      {
        float *output = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
      }
      TIMER_STOP(output_store);
    }
  }

  TIMER_STOP(total);
  FARF(ALWAYS, "%s: %lld us, m=%d k=%d n=%d", __func__, TIMER_US(total), m, k, n);

#if defined(ENABLE_PROFILE_TIMERS)
  FARF(ALWAYS, "  activation_load: %lld us, weight_load: %lld us, hmx_core: %lld us, output_store: %lld us",
       TIMER_US(activation_load), TIMER_US(weight_load), TIMER_US(hmx_core), TIMER_US(output_store));
  {
    size_t weight_size = (size_t)k * n * sizeof(__fp16);
    float  bandwidth   = 1e-3f * weight_size / (float)TIMER_US(weight_load);
    FARF(ALWAYS, "  weight load bandwidth: %.2f GB/s", bandwidth);
  }
#endif

  return 0;
}

typedef struct {
  __fp16                *c;
  const __fp16          *a, *b, *s;
  int                    n_row_tiles, n_col_tiles, n_dot_tiles;
  uint32_t               vtcm_rctx;
} core_dot_fp16_task_state_t;

static void core_dot_fp16_hmx_worker_fn(unsigned int n, unsigned int i, void *data) {
  (void) n; (void) i;
  core_dot_fp16_task_state_t *st = (core_dot_fp16_task_state_t *) data;

  core_dot_chunk_fp16(st->c, st->a, st->b, st->s, st->n_row_tiles, st->n_col_tiles, st->n_dot_tiles, st->vtcm_rctx);
}

int mat_mul_qk_0_d16a32_out_stationary(struct htp_context *ctx, float *restrict out, const float *restrict x, const uint8_t *restrict w, int m,
                                       int k, int n, int w_type);

int hmx_mat_mul_permuted_qk_0_d16a32(struct htp_context *ctx, float *restrict dst, const float *restrict activation,
                                     const uint8_t *restrict permuted_weight, int m, int k, int n,
                                     int weight_type) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!hmx_is_aligned(dst, HMX_VLEN) || !hmx_is_aligned(activation, HMX_VLEN) || !hmx_is_aligned(permuted_weight, HMX_VLEN)) {
    return -1;
  }

  // for large m, k (e.g. prefill FFN Down), use out-stationary version
  if (m >= 128 && k > n && n > 1024) {
    return mat_mul_qk_0_d16a32_out_stationary(ctx, dst, activation, permuted_weight, m, k, n, weight_type);
  }

  size_t row_stride = get_x4x2_row_stride(weight_type, k);
  if (row_stride == 0) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = OUTPUT_AREA_SIZE;
  const size_t scratch_area_size    = SCRATCH_AREA_SIZE;

  // VTCM layout: weight | activation | output | scratch (x4x2 DMA) | scales
  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch2   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  assert((size_t)(vtcm_ptr - ctx->vtcm_base) <= ctx->vtcm_scratch_size);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = hmx_align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = hmx_align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

#if HMX_DEBUG_TRACE_VALUES
  FARF(ALWAYS, "%s: m=%d k=%d n=%d wtype=%d m_chunk=%zu n_chunk=%zu vec_dot_sz=%zu row_stride=%zu",
       __func__, m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vec_dot_size, row_stride);
#endif

  const bool use_pipeline = (m >= 128) && (k <= n);
  // const bool use_pipeline = false;

  TIMER_DEFINE(activation_load);
  TIMER_DEFINE(weight_load);
  TIMER_DEFINE(hmx_core);
  TIMER_DEFINE(output_store);

  TIMER_DEFINE(total);
  TIMER_START(total);

  if (!use_pipeline) {
    // NOTE(hzx): In this simple implementation, load-matmul-store are executed sequentially
    // only DMA load and dequantization process are overlapped during the load stage

    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      // transfer activation matrix chunk into VTCM
      size_t n_rows = hmx_smin(m - mr, m_chunk_n_rows);

      TIMER_START(activation_load);
      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }
      TIMER_STOP(activation_load);

#if HMX_DEBUG_TRACE_VALUES
      // DBG: sample activation src (fp32 DDR) and tile (fp16 VTCM) after first chunk
      if (mr == 0) {
        const float *act_src = activation;
        FARF(ALWAYS, "  DBG act_src[0..3] = %.6f %.6f %.6f %.6f  act_src[%d] = %.6f",
             act_src[0], act_src[1], act_src[2], act_src[3], k - 1, act_src[k - 1]);
        // Sample first tile raw fp16 (interleaved pair rows 0,1 cols 0,1)
        FARF(ALWAYS, "  DBG vtcm_act[0..3] = %.6f %.6f %.6f %.6f",
             (float)vtcm_activation[0], (float)vtcm_activation[1],
             (float)vtcm_activation[2], (float)vtcm_activation[3]);
      }
#endif

      // checkpoint 1: activation tiles (tile-major fp16 in VTCM)
      {
        int act_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        int act_col_tiles = (int)(k / HMX_FP16_TILE_N_COLS);
        int act_n_tiles   = act_row_tiles * act_col_tiles;
        (void)act_n_tiles;
        // HMX_CHECK_FP16(vtcm_activation, act_n_tiles * HMX_FP16_TILE_N_ELMS,
        //                 "act_tile_fp16", (int)mr, 0);
        // HMX_DUMP_FP16(vtcm_activation, act_n_tiles * HMX_FP16_TILE_N_ELMS,
        //                "act_tile_fp16", (int)mr, 0);
      }

#if HMX_DEBUG_CHECK_VALUES
      if (mr == 0) {
        const float *act_raw = activation;
        int n_k_tiles_act = k / HMX_FP16_TILE_N_COLS;
        FARF(ALWAYS, "=== ACT tile(0,0) pre-shuffle (fp32 row-major) ===");
        HMX_DUMP_FP32_TILE(act_raw, k, "act_pre_shuf", 0, 0, 8);
        FARF(ALWAYS, "=== ACT tile(0,0) post-shuffle (tile fp16 interleaved) ===");
        HMX_DUMP_TILE_MEM(vtcm_activation, "act_post_shuf", 0, 0, 16);
        if (n_k_tiles_act > 7) {
          FARF(ALWAYS, "=== ACT tile(0,7) pre-shuffle (fp32 row-major) ===");
          HMX_DUMP_FP32_TILE(act_raw, k, "act_pre_shuf", 0, 7, 8);
          FARF(ALWAYS, "=== ACT tile(0,7) post-shuffle (tile fp16 interleaved) ===");
          HMX_DUMP_TILE_MEM(vtcm_activation + 7 * HMX_FP16_TILE_N_ELMS, "act_post_shuf", 0, 7, 16);
        }
      }
#endif

      void *buf_curr = vtcm_scratch0;
      void *buf_next = vtcm_scratch1;

      // issue async DDR data transfer for the first weight chunk
      // NOTE: use 2D DMA (n_cols rows × row_stride bytes) instead of 1D
      // because UDMA roiwidth is 16-bit and total size can exceed 65535.
      {
        const size_t n_cols_first = hmx_smin(n, n_chunk_n_cols);

        dma_queue_push(ctx->dma[0], dma_make_ptr(buf_curr, permuted_weight), row_stride, row_stride, row_stride, n_cols_first);
      }

      for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
        size_t n_cols = hmx_smin(n - nc, n_chunk_n_cols);

        TIMER_START(weight_load);
        {
          dma_queue_pop(ctx->dma[0]);  // wait until current weight chunk become ready

          const size_t nc_next = nc + n_chunk_n_cols;
          if (nc_next < n) {
            const size_t n_cols_next = hmx_smin(n - nc_next, n_chunk_n_cols);

            const uint8_t *next_weight_chunk = permuted_weight + nc_next * row_stride;

            dma_queue_push(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk), row_stride, row_stride, row_stride, n_cols_next);
          }

#if HMX_DEBUG_TRACE_VALUES
          // DBG: DMA integrity check — compare VTCM buffer with DDR source at scale region.
          // This determines whether DMA corrupts the data or the DDR source is already wrong.
          if (mr == 0 && nc == 0) {
            const bool _is_q4 = (weight_type == HMX_TYPE_Q4_0 || weight_type == HMX_TYPE_IQ4_NL);
            const int _qrow_sz = _is_q4 ? ((int)k / 2) : (int)k;
            const uint8_t *_vtcm_buf = (const uint8_t *)buf_curr;
            const uint8_t *_ddr_buf  = permuted_weight;  // DDR source (nc==0, so no offset)

            // Check scale region for rows 0, 13, 26 (row 26 was where NaN was seen before)
            int _check_rows[] = { 0, 13, 26 };
            for (int _ri = 0; _ri < 3; _ri++) {
              int _row = _check_rows[_ri];
              if (_row >= (int)n_cols) break;

              const uint8_t *_vtcm_row = _vtcm_buf + _row * row_stride;
              const uint8_t *_ddr_row  = _ddr_buf  + _row * row_stride;

              // Dump first 8 scale bytes (4 fp16 scales = first half of block 0's scales)
              const uint16_t *_vs = (const uint16_t *)(_vtcm_row + _qrow_sz);  // VTCM scale region
              const uint16_t *_ds = (const uint16_t *)(_ddr_row  + _qrow_sz);  // DDR scale region
              FARF(ALWAYS, "  DBG DMA-CHECK row %d: VTCM scales[0..3]=0x%04x %04x %04x %04x  DDR scales[0..3]=0x%04x %04x %04x %04x",
                   _row, _vs[0], _vs[1], _vs[2], _vs[3], _ds[0], _ds[1], _ds[2], _ds[3]);

              // Also compare packed data region (first 8 bytes) to verify DMA works at all
              const uint16_t *_vq = (const uint16_t *)(_vtcm_row);
              const uint16_t *_dq = (const uint16_t *)(_ddr_row);
              int _scale_match = (_vs[0]==_ds[0] && _vs[1]==_ds[1] && _vs[2]==_ds[2] && _vs[3]==_ds[3]);
              int _quant_match = (_vq[0]==_dq[0] && _vq[1]==_dq[1] && _vq[2]==_dq[2] && _vq[3]==_dq[3]);
              FARF(ALWAYS, "  DBG DMA-CHECK row %d: quants %s, scales %s | VTCM quants[0..3]=0x%04x %04x %04x %04x  DDR quants[0..3]=0x%04x %04x %04x %04x",
                   _row, _quant_match ? "MATCH" : "MISMATCH", _scale_match ? "MATCH" : "MISMATCH",
                   _vq[0], _vq[1], _vq[2], _vq[3], _dq[0], _dq[1], _dq[2], _dq[3]);

              // Full byte-by-byte comparison of scale region for this row
              int _nbytes_scales = (int)row_stride - _qrow_sz;
              int _n_mismatch = 0;
              int _first_mismatch_off = -1;
              for (int _bi = 0; _bi < _nbytes_scales; _bi++) {
                if (_vtcm_row[_qrow_sz + _bi] != _ddr_row[_qrow_sz + _bi]) {
                  if (_first_mismatch_off < 0) _first_mismatch_off = _bi;
                  _n_mismatch++;
                }
              }
              if (_n_mismatch > 0) {
                FARF(ALWAYS, "  DBG DMA-CHECK row %d: %d/%d scale bytes MISMATCH (first at offset +%d)",
                     _row, _n_mismatch, _nbytes_scales, _first_mismatch_off);
              } else {
                FARF(ALWAYS, "  DBG DMA-CHECK row %d: all %d scale bytes MATCH between VTCM and DDR", _row, _nbytes_scales);
              }
            }
          }
#endif
          // Dequant + vscatter writes directly to [K, N] transposed tiles.
          // HMX computes C = A × B, where A=[M,K] activation, B=[K,N] weight.
          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight, buf_curr,
                                                      n_cols, k, row_stride, weight_type);

#if HMX_DEBUG_CHECK_VALUES
          if (mr == 0 && nc == 0) {
            int n_wt_tiles_dbg = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS) * (k / HMX_FP16_TILE_N_COLS);
            FARF(ALWAYS, "=== WT tile(0,0) dequant n_wt_tiles=%d ===", n_wt_tiles_dbg);
            HMX_DUMP_TILE_MEM(vtcm_weight, "wt_dequant", 0, 0, 16);
            if (n_wt_tiles_dbg > 7) {
              FARF(ALWAYS, "=== WT tile(0,7) dequant ===");
              HMX_DUMP_TILE_MEM(vtcm_weight + 7 * HMX_FP16_TILE_N_ELMS, "wt_dequant", 0, 7, 16);
            }
          }
#endif

          swap_ptr(&buf_curr, &buf_next);
        }
        TIMER_STOP(weight_load);

#if HMX_DEBUG_TRACE_VALUES
        // DBG: sample weight tile (fp16 VTCM) after first dequant
        // Tile layout: elem(k_row, n_col) = tile[(k_row>>1)*64 + n_col*2 + (k_row&1)]
        // tile[0..7] = (k0,n0)(k1,n0)(k0,n1)(k1,n1)(k0,n2)(k1,n2)(k0,n3)(k1,n3)
        if (mr == 0 && nc == 0) {
          const uint16_t *wt_raw = (const uint16_t *)vtcm_weight;
          FARF(ALWAYS, "  DBG vtcm_wt[0..7] fp16 = %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f",
               (float)vtcm_weight[0], (float)vtcm_weight[1],
               (float)vtcm_weight[2], (float)vtcm_weight[3],
               (float)vtcm_weight[4], (float)vtcm_weight[5],
               (float)vtcm_weight[6], (float)vtcm_weight[7]);
          FARF(ALWAYS, "  DBG vtcm_wt[0..7] raw  = %04x %04x %04x %04x | %04x %04x %04x %04x",
               wt_raw[0], wt_raw[1], wt_raw[2], wt_raw[3],
               wt_raw[4], wt_raw[5], wt_raw[6], wt_raw[7]);
          // Also check weight tile 1 (K-tile 1, same N-cols) at offset HMX_FP16_TILE_N_ELMS
          const uint16_t *wt1_raw = (const uint16_t *)(vtcm_weight + HMX_FP16_TILE_N_ELMS);
          FARF(ALWAYS, "  DBG vtcm_wt_tile1[0..7] raw = %04x %04x %04x %04x | %04x %04x %04x %04x",
               wt1_raw[0], wt1_raw[1], wt1_raw[2], wt1_raw[3],
               wt1_raw[4], wt1_raw[5], wt1_raw[6], wt1_raw[7]);
          // Check N-col 16 area: tile[(0>>1)*64 + 16*2 + 0] = tile[32]
          FARF(ALWAYS, "  DBG vtcm_wt[32..39] raw (n16-n19) = %04x %04x %04x %04x | %04x %04x %04x %04x",
               wt_raw[32], wt_raw[33], wt_raw[34], wt_raw[35],
               wt_raw[36], wt_raw[37], wt_raw[38], wt_raw[39]);
          // Count how many 0xFFFF (NaN) in first tile AFTER dequant
          int nan_post = 0;
          int n_wt_tiles_total = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS) * (k / HMX_FP16_TILE_N_COLS);
          for (int _i = 0; _i < 1024; ++_i) {
            if (wt_raw[_i] == 0xFFFF) nan_post++;
          }
          FARF(ALWAYS, "  DBG vtcm_wt POST-dequant: %d/1024 values are 0xFFFF in tile 0 (total tiles=%d)",
               nan_post, n_wt_tiles_total);
          // Also check last tile for stale data
          if (n_wt_tiles_total > 1) {
            const uint16_t *last_tile = (const uint16_t *)(vtcm_weight + (n_wt_tiles_total - 1) * HMX_FP16_TILE_N_ELMS);
            int nan_last = 0;
            for (int _i = 0; _i < 1024; ++_i) {
              if (last_tile[_i] == 0xFFFF) nan_last++;
            }
            FARF(ALWAYS, "  DBG vtcm_wt POST-dequant: %d/1024 values are 0xFFFF in last tile (%d)",
                 nan_last, n_wt_tiles_total - 1);
          }

          // === SCALAR REFERENCE DEQUANT: manually dequant a few positions ===
          // Find first NaN position in tile 0 and trace back to DDR input data
          {
            // For kt=0, ct=0: blk_idx=0, sub_blk=0, upper=false, byte_off=0, scale_off=qrow_size
            int _qrow_size = k / 2;  // 2048 for k=4096
            // Find first NaN in tile 0
            int nan_pos = -1;
            for (int _i = 0; _i < 1024; ++_i) {
              if (wt_raw[_i] == 0xFFFF) { nan_pos = _i; break; }
            }
            if (nan_pos >= 0) {
              // Decode tile position: elem(k_row, n_col) = tile[(k_row>>1)*64 + n_col*2 + (k_row&1)]
              int _vec_idx = nan_pos / 64;  // k_row >> 1
              int _within  = nan_pos % 64;
              int _n_col   = _within / 2;
              int _k_parity = _within & 1;
              int _k_row   = _vec_idx * 2 + _k_parity;
              FARF(ALWAYS, "  DBG SCALAR: first NaN at tile[%d] = k_row=%d n_col=%d", nan_pos, _k_row, _n_col);

              // The NaN was written by vscatter for weight column _n_col at K-row _k_row.
              // In the dequant loop: r = _n_col & ~1, row0 = _n_col & ~1, row1 = row0+1
              // The specific row is _n_col (for N-col mapping).
              // Input data: vtcm_src + _n_col * row_stride + byte_off (=0 for kt=0)
              // Scale: vtcm_src + _n_col * row_stride + _qrow_size (for kt=0, sub_blk=0)
              // The byte at position _k_row/2 in the 32-byte group contains the nibble.

              // Read from the ORIGINAL weight data in DDR (not DMA buffer)
              const uint8_t *wt_row = permuted_weight + _n_col * row_stride;
              // K-row i maps to byte[i] in the 32-byte group (byte_off=0 for kt=0)
              uint8_t raw_byte = wt_row[_k_row];
              int nibble = raw_byte & 0x0F;  // upper=false for sub_blk=0
              __fp16 scale_val = *(const __fp16 *)(wt_row + _qrow_size);
              uint16_t scale_raw = *(const uint16_t *)(wt_row + _qrow_size);

              // Scalar dequant: q4_0_to_fp16_lut layout: entries at even positions
              // LUT[nibble*2] = the fp16 integer value
              __fp16 lut_val = q4_0_to_fp16_lut[nibble * 2];
              __fp16 scalar_result = (__fp16)((float)lut_val * (float)scale_val);
              uint16_t scalar_raw = *(const uint16_t *)&scalar_result;

              FARF(ALWAYS, "  DBG SCALAR: row=%d byte[%d]=0x%02x nibble=%d lut=%.1f scale=%.6f(0x%04x) result=%.4f(0x%04x)",
                   _n_col, _k_row, raw_byte, nibble,
                   (float)lut_val, (float)scale_val, scale_raw,
                   (float)scalar_result, scalar_raw);

              // Also check the paired K-row element
              int pair_pos = (_k_parity == 0) ? nan_pos + 1 : nan_pos - 1;
              if (pair_pos >= 0 && pair_pos < 1024) {
                uint16_t pair_val = wt_raw[pair_pos];
                int _k_row2 = _k_parity == 0 ? _k_row + 1 : _k_row - 1;
                uint8_t raw_byte2 = wt_row[_k_row2];
                int nibble2 = raw_byte2 & 0x0F;
                __fp16 lut_val2 = q4_0_to_fp16_lut[nibble2 * 2];
                __fp16 scalar_res2 = (__fp16)((float)lut_val2 * (float)scale_val);
                uint16_t scalar_raw2 = *(const uint16_t *)&scalar_res2;
                FARF(ALWAYS, "  DBG SCALAR: pair tile[%d]=0x%04x k_row=%d byte=0x%02x nib=%d lut=%.1f scalar=%.4f(0x%04x)",
                     pair_pos, pair_val, _k_row2, raw_byte2, nibble2,
                     (float)lut_val2, (float)scalar_res2, scalar_raw2);
              }

              // Print first 4 raw bytes and scale for rows 0..3 (N-cols 0..3)
              for (int _r = 0; _r < 4 && _r < (int)n_cols; ++_r) {
                const uint8_t *_wr = permuted_weight + _r * row_stride;
                uint16_t _sc_raw = *(const uint16_t *)(_wr + _qrow_size);
                FARF(ALWAYS, "  DBG SCALAR: N-col %d: bytes[0..3]=0x%02x 0x%02x 0x%02x 0x%02x scale=0x%04x",
                     _r, _wr[0], _wr[1], _wr[2], _wr[3], _sc_raw);
              }
            } else {
              FARF(ALWAYS, "  DBG SCALAR: no NaN found in tile 0");
            }
          }
        }
#endif

        TIMER_START(hmx_core);
        {
          const int n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          const int n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32, ctx->vtcm_rctx);
        }
        TIMER_STOP(hmx_core);

#if HMX_DEBUG_TRACE_VALUES
        // DBG: sample HMX output tile (fp16 VTCM) after first compute
        // Tile layout: elem(row, col) = tile[(row>>1)*64 + col*2 + (row&1)]
        // So tile[0]=e(0,0) tile[1]=e(1,0) tile[2]=e(0,1) tile[3]=e(1,1) tile[4]=e(0,2) ...
        if (mr == 0 && nc == 0) {
          const uint16_t *raw = (const uint16_t *)vtcm_output;
          FARF(ALWAYS, "  DBG vtcm_out[0..7] fp16 = %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f",
               (float)vtcm_output[0], (float)vtcm_output[1],
               (float)vtcm_output[2], (float)vtcm_output[3],
               (float)vtcm_output[4], (float)vtcm_output[5],
               (float)vtcm_output[6], (float)vtcm_output[7]);
          FARF(ALWAYS, "  DBG vtcm_out[0..7] raw  = %04x %04x %04x %04x | %04x %04x %04x %04x",
               raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
          // Print more columns: [8..15] = cols 4-7
          FARF(ALWAYS, "  DBG vtcm_out[8..15] raw = %04x %04x %04x %04x | %04x %04x %04x %04x",
               raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]);
          // Count NaN in entire output tile
          int out_nan = 0;
          for (int _i = 0; _i < 1024; ++_i) {
            if ((raw[_i] & 0x7C00) == 0x7C00 && (raw[_i] & 0x03FF) != 0) out_nan++;
          }
          FARF(ALWAYS, "  DBG vtcm_out: %d/1024 NaN values in output tile 0", out_nan);
        }
#endif

#if HMX_DEBUG_CHECK_VALUES
        {
          int _nrt = hmx_ceil_div((int)n_rows, HMX_FP16_TILE_N_ROWS);
          int _nct = hmx_ceil_div((int)n_cols, HMX_FP16_TILE_N_COLS);
          hmx_dump_ref_vs_hmx(vtcm_activation, vtcm_weight, vtcm_output,
                              _nrt, _nct, (int)(k / 32));
        }
        {
          int out_n_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS)
                          * hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          HMX_CHECK_FP16(vtcm_output, out_n_tiles * HMX_FP16_TILE_N_ELMS,
                          "hmx_out_fp16", (int)mr, (int)nc);
          HMX_DUMP_FP16(vtcm_output, out_n_tiles * HMX_FP16_TILE_N_ELMS,
                         "hmx_out_fp16", (int)mr, (int)nc);
        }
#endif

#if HMX_DEBUG_CHECK_VALUES
        if (mr == 0 && nc == 0) {
          int n_out_ct = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          FARF(ALWAYS, "=== OUT tile(0,0) pre-deshuffle (tile fp16) n_out_ct=%d ===", n_out_ct);
          HMX_DUMP_TILE_MEM(vtcm_output, "out_pre_shuf", 0, 0, 16);
          if (n_out_ct > 7) {
            FARF(ALWAYS, "=== OUT tile(0,7) pre-deshuffle (tile fp16) ===");
            HMX_DUMP_TILE_MEM(vtcm_output + 7 * HMX_FP16_TILE_N_ELMS, "out_pre_shuf", 0, 7, 16);
          }
        }
#endif

        TIMER_START(output_store);
        {
          float *output = dst + (mr * n + nc);
          transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);

#if HMX_DEBUG_TRACE_VALUES
          // DBG: sample DDR output (fp32) after first store
          if (mr == 0 && nc == 0) {
            const uint32_t *raw32 = (const uint32_t *)output;
            FARF(ALWAYS, "  DBG ddr_out[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                 output[0], output[1], output[2], output[3],
                 output[4], output[5], output[6], output[7]);
            FARF(ALWAYS, "  DBG ddr_out[0..7] raw = %08x %08x %08x %08x %08x %08x %08x %08x",
                 raw32[0], raw32[1], raw32[2], raw32[3],
                 raw32[4], raw32[5], raw32[6], raw32[7]);
            // Also print row 1 (starts at output[n])
            FARF(ALWAYS, "  DBG ddr_row1[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                 output[n], output[n+1], output[n+2], output[n+3],
                 output[n+4], output[n+5], output[n+6], output[n+7]);
          }
#endif

#if HMX_DEBUG_CHECK_VALUES
          HMX_CHECK_FP32(output, n_rows * n_cols, n, n_rows, n_cols,
                          "out_fp32", (int)mr, (int)nc);

          if (mr == 0 && nc == 0) {
            int n_out_ct2 = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
            FARF(ALWAYS, "=== OUT tile(0,0) post-deshuffle (fp32 row-major) ===");
            HMX_DUMP_FP32_TILE(output, n, "out_post_shuf", 0, 0, 8);
            if (n_out_ct2 > 7) {
              FARF(ALWAYS, "=== OUT tile(0,7) post-deshuffle (fp32 row-major) ===");
              HMX_DUMP_FP32_TILE(output, n, "out_post_shuf", 0, 7, 8);
            }
          }
#endif

        }
        TIMER_STOP(output_store);
      }
    }
  } else {
    // 4-stage pipeline: DMA load (A), dequantize (B), HMX matmul (C), store (D)
    // stage B and D (dequantize and store) are expected to be on the critical path

    // A --> B: vtcm_qweight, 1 buffer
    // B --> C: vtcm_weight0/vtcm_weight1, 2 buffers
    // C --> D: vtcm_output0/vtcm_output1, 2 buffers

    //
    // LD ||A3|  | B3 ||
    // MM ||    C2    ||
    // ST || D1 |     ||

    static core_dot_fp16_task_state_t mm_task_state;

    int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);
    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      const size_t n_rows = hmx_smin(m - mr, m_chunk_n_rows);

      void *vtcm_qweight        = vtcm_weight;
      void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
      void *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

      // prologue: A0
      const size_t n_cols_A0 = hmx_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
      {
        const size_t chunk_size_A0 = n_cols_A0 * row_stride;

        const uint8_t *qweight_chunk_A0 = permuted_weight;
        dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_qweight, qweight_chunk_A0), chunk_size_A0, chunk_size_A0, chunk_size_A0, 1);
      }

      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }

      // prologue: B0, A1, C0, B1
      {
        // B0
        dma_queue_pop(ctx->dma[0]);
        dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight_bufs[0], vtcm_qweight,
                                                    n_cols_A0, k, row_stride, weight_type);

        // A1
        const size_t n_cols_A1 = hmx_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
        if (1 < n_chunk_cnt) {
          const size_t chunk_size_A1 = n_cols_A1 * row_stride;

          const uint8_t *qweight_chunk_A1 = permuted_weight + n_chunk_n_cols * row_stride;
          dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_qweight, qweight_chunk_A1), chunk_size_A1, chunk_size_A1, chunk_size_A1, 1);
        }

        // C0
        {
          core_dot_fp16_task_state_t *s = &mm_task_state;

          s->c = (__fp16 *) vtcm_output_bufs[0];
          s->a = (__fp16 *) vtcm_activation;
          s->b = (__fp16 *) vtcm_weight_bufs[0];
          s->s = vtcm_scales;

          s->n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          s->n_col_tiles = hmx_ceil_div(n_cols_A0, HMX_FP16_TILE_N_COLS);
          s->n_dot_tiles = k / HMX_FP16_TILE_N_ROWS;
          s->vtcm_rctx   = ctx->vtcm_rctx;

          worker_pool_run_func(ctx->worker_pool, core_dot_fp16_hmx_worker_fn, s, 1);
        }

        // B1
        if (1 < n_chunk_cnt) {
          dma_queue_pop(ctx->dma[0]);
          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight_bufs[1], vtcm_qweight,
                                                      n_cols_A1, k, row_stride, weight_type);
        }
      }

      // main loop
      for (int i = 0; i < n_chunk_cnt; ++i) {
        const size_t nc    = i * n_chunk_n_cols;
        const size_t nc_p1 = nc + 1 * n_chunk_n_cols;
        const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

        const size_t n_cols    = hmx_smin(n - nc, n_chunk_n_cols);
        const size_t n_cols_p1 = hmx_smin(n - nc_p1, n_chunk_n_cols);
        const size_t n_cols_p2 = hmx_smin(n - nc_p2, n_chunk_n_cols);

        // issue A_{i+2}
        if (i + 2 < n_chunk_cnt) {
          const size_t   chunk_size_p2    = n_cols_p2 * row_stride;
          const uint8_t *qweight_chunk_p2 = permuted_weight + nc_p2 * row_stride;
          dma_queue_push(ctx->dma[0], dma_make_ptr(vtcm_qweight, qweight_chunk_p2), chunk_size_p2, chunk_size_p2, chunk_size_p2, 1);
        }

        // wait for HMX (C_{i}) — worker_pool_run_func already returned, so C_{i} is done

        // result of B_{i+1} (input of C_{i+1}) should be ready now

        // issue C_{i+1}
        if (i + 1 < n_chunk_cnt) {
          core_dot_fp16_task_state_t *s = &mm_task_state;

          s->c = (__fp16 *) vtcm_output_bufs[(i + 1) % 2];
          s->a = (__fp16 *) vtcm_activation;
          s->b = (__fp16 *) vtcm_weight_bufs[(i + 1) % 2];
          s->s = vtcm_scales;

          s->n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          s->n_col_tiles = hmx_ceil_div(n_cols_p1, HMX_FP16_TILE_N_COLS);
          s->n_dot_tiles = k / HMX_FP16_TILE_N_ROWS;
          s->vtcm_rctx   = ctx->vtcm_rctx;

          worker_pool_run_func(ctx->worker_pool, core_dot_fp16_hmx_worker_fn, s, 1);
        }

        // compute D_{i}
        float *output_chunk = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output_chunk, vtcm_output_bufs[i % 2], n_rows, n_cols, n);

        // wait for DMA (A_{i+2}), compute B_{i+2}
        if (i + 2 < n_chunk_cnt) {
          dma_queue_pop(ctx->dma[0]);
          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight_bufs[(i + 2) % 2], vtcm_qweight,
                                                      n_cols_p2, k, row_stride, weight_type);
        }
      }
    }
  }

  TIMER_STOP(total);

#if HMX_DEBUG_TRACE_VALUES
  // DBG: sample final output from DDR
  FARF(ALWAYS, "  DBG final dst[0..3] = %.6f %.6f %.6f %.6f  dst[%d] = %.6f",
       dst[0], dst[1], dst[2], dst[3], m * n - 1, dst[m * n - 1]);
#endif

  FARF(ALWAYS, "%s: %lld us, m=%d k=%d n=%d pipeline=%d", __func__, TIMER_US(total), m, k, n, use_pipeline);

#if defined(ENABLE_PROFILE_TIMERS)
  if (!use_pipeline) {
    FARF(ALWAYS, "  activation_load: %lld us, weight_load: %lld us, hmx_core: %lld us, output_store: %lld us",
         TIMER_US(activation_load), TIMER_US(weight_load), TIMER_US(hmx_core), TIMER_US(output_store));
    size_t weight_size = (size_t)n * row_stride;
    float  bandwidth   = 1e-3f * weight_size / (float)TIMER_US(weight_load);
    FARF(ALWAYS, "  weight load bandwidth: %.2f GB/s", bandwidth);
  }
#endif

  return 0;
}

// C += AB
void core_mma_chunk_fp16(__fp16 *c, const __fp16 *a, const __fp16 *b, const __fp16 *col_scales, const __fp16 *eye_tile,
                         int n_row_tiles, int n_col_tiles, int n_dot_tiles, bool zero_init, uint32_t vtcm_rctx) {
  HAP_compute_res_hmx_lock(vtcm_rctx);

  hmx_set_output_scales(col_scales);

  for (int i = 0; i < n_row_tiles; ++i) {
    for (int j = 0; j < n_col_tiles; ++j) {
      asm volatile("mxclracc.hf");

      const __fp16 *row_tiles = a + i * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = b + j * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      __fp16 *accum_tile = c + (i * n_col_tiles + j) * HMX_FP16_TILE_N_ELMS;
      if (!zero_init) {
        hmx_load_tile_pair_fp16(accum_tile, eye_tile);
      }

      for (int k = 0; k < n_dot_tiles; ++k) {
        int offset = k * HMX_FP16_TILE_N_ELMS;
        hmx_load_tile_pair_fp16(row_tiles + offset, col_tiles + offset);
      }

      hmx_consume_accumulator_fp16(accum_tile);
    }
  }

  HAP_compute_res_hmx_unlock(vtcm_rctx);
}

typedef struct {
  uint8_t              *dst;
  const uint8_t        *src;
  size_t                n_rows;
  size_t                src_stride;   // DDR row stride (full row_stride)
  size_t                dst_stride;   // VTCM sub-block row stride
  size_t                quant_off;    // quant byte offset in each DDR row
  size_t                quant_width;  // quant bytes to copy per row
  size_t                scale_off;    // scale byte offset in each DDR row
  size_t                scale_width;  // scale bytes to copy per row
  dma_queue            *dma;
} qweight_fetch_task_state_t;

static void qweight_fetch_worker_fn(unsigned int n, unsigned int i, void *data) {
  (void) n; (void) i;
  qweight_fetch_task_state_t *st = (qweight_fetch_task_state_t *) data;

  // 2D DMA: quants sub-range
  dma_queue_push(st->dma,
                 dma_make_ptr(st->dst, st->src + st->quant_off),
                 st->dst_stride, st->src_stride, st->quant_width, st->n_rows);
  dma_queue_pop(st->dma);
  // 2D DMA: scales sub-range
  dma_queue_push(st->dma,
                 dma_make_ptr(st->dst + st->quant_width, st->src + st->scale_off),
                 st->dst_stride, st->src_stride, st->scale_width, st->n_rows);
  dma_queue_pop(st->dma);
}

// Only slightly faster than the common version (with L2 prefetch enabled) when doing VTCM to VTCM transfer
void transfer_activation_chunk_no_prefetch(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                           int k_block, int k_stride) {
  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);
    for (int c = 0; c < k_block; c += 32) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

      HVX_Vector v_out = hmx_hvx_wsf_to_vhf(v1, v0);

      // compute output position
      int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
      int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  int                    n_tasks;
  int                    n_tot_chunks;
  int                    n_chunks_per_task;
  __fp16      *dst;
  const float *src;
  int          k_block, k_stride;
} activation_transfer_task_state_t;

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
  activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

  for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
    // one chunk: one row
    int    chunk_idx  = task_id * st->n_chunks_per_task;
    size_t chunk_size = hmx_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

    __fp16      *dst = st->dst + chunk_idx * st->k_block;
    const float *src = st->src + chunk_idx * st->k_stride;
    transfer_activation_chunk_no_prefetch(dst, src, chunk_size, st->k_block, st->k_stride);
  }
}

void transfer_activation_chunk_multithread(struct htp_context *ctx, __fp16 *dst, const float *src, int n_rows, int k_block, int k_stride) {
  size_t n_tot_chunks      = n_rows;
  size_t n_chunks_per_task = 32;  // NOTE(hzx): must be multiple of 32 to ensure correct destination address

  activation_transfer_task_state_t state;
  state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task;
  state.n_tot_chunks      = n_tot_chunks;
  state.n_chunks_per_task = n_chunks_per_task;
  state.dst      = dst;
  state.src      = src;
  state.k_block  = k_block;
  state.k_stride = k_stride;

  worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, ctx->n_threads);
}

int mat_mul_qk_0_d16a32_out_stationary(struct htp_context *ctx, float *restrict out, const float *restrict x, const uint8_t *restrict w, int m,
                                       int k, int n, int weight_type) {
  // NOTE(hzx): this constraint on k originates from 2D DMA, consider alternative ways to load activation
  assert(k < 16384);
  // assume k % 32 == 0 && n % 32 == 0
  const size_t row_stride = get_x4x2_row_stride(weight_type, k);
  if (row_stride == 0) {
    return -1;
  }

  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, WEIGHT_AREA_SIZE);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, ACTIVATION_AREA_SIZE);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, OUTPUT_AREA_SIZE);
  uint8_t *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, SCRATCH_AREA_SIZE);
  uint8_t *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, SCRATCH_AREA_SIZE * 2);
  __fp16  *vtcm_eye_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  assert((size_t)(vtcm_ptr - ctx->vtcm_base) <= ctx->vtcm_scratch_size);

  // initialize eye tile (32x32 identity matrix)
  {
    HVX_Vector v;
    v = Q6_V_vzero();
    v = Q6_Vw_vinsert_VwR(v, 0x3c000000);
    v = Q6_V_vror_VR(v, HMX_VLEN - 4);
    v = Q6_Vw_vinsert_VwR(v, 0x00003c00);
    for (int i = 0; i < 16; ++i) {
      ((HVX_Vector *) vtcm_eye_tile)[i] = v;

      v = Q6_V_vror_VR(v, HMX_VLEN - 8);
    }
  }
  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  // 704, 512
  const size_t M_BLOCK_SIZE = 512;
  const size_t N_BLOCK_SIZE = 512;
  const size_t K_BLOCK_SIZE = 512;

  qweight_fetch_task_state_t fetch_task_state;

  TIMER_DEFINE(fetch);
  TIMER_DEFINE(load);
  TIMER_DEFINE(core);

  for (size_t mr = 0; mr < m; mr += M_BLOCK_SIZE) {
    size_t m_blk_sz = hmx_smin(m - mr, M_BLOCK_SIZE);
    for (size_t nc = 0; nc < n; nc += N_BLOCK_SIZE) {
      size_t n_blk_sz = hmx_smin(n - nc, N_BLOCK_SIZE);

      const int n_row_tiles = hmx_ceil_div(m_blk_sz, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = hmx_ceil_div(n_blk_sz, HMX_FP16_TILE_N_COLS);

      // TODO(hzx): fully pipelined loop
      for (size_t kk = 0; kk < k; kk += K_BLOCK_SIZE) {
        size_t k_blk_sz = hmx_smin(k - kk, K_BLOCK_SIZE);

        TIMER_START(fetch);
        // fetch activation block into VTCM
        {
          const float *activation_block = x + mr * k + kk;

          dma_queue_push(ctx->dma[0],
                         dma_make_ptr(vtcm_scratch1, activation_block),
                         k_blk_sz * sizeof(float),
                         k * sizeof(float),
                         k_blk_sz * sizeof(float),
                         m_blk_sz);
          dma_queue_pop(ctx->dma[0]);
        }

        // fetch weight block into VTCM (x4x2 sub-block: quants + scales)
        {
          qweight_fetch_task_state_t *s = &fetch_task_state;

          const bool is_q4 = (weight_type == HMX_TYPE_Q4_0 || weight_type == HMX_TYPE_IQ4_NL);
          const int blk_start = kk / HMX_QK_Q4x4x2;
          const int nb_sub = (k_blk_sz + HMX_QK_Q4x4x2 - 1) / HMX_QK_Q4x4x2;
          const int full_qrow = is_q4 ? (k / 2) : k;
          const size_t sub_row_stride = get_x4x2_row_stride(weight_type, k_blk_sz);

          s->dst         = vtcm_scratch0;
          s->src         = w + nc * row_stride;
          s->n_rows      = n_blk_sz;
          s->src_stride  = row_stride;
          s->dst_stride  = sub_row_stride;
          s->quant_off   = is_q4 ? (blk_start * (HMX_QK_Q4x4x2 / 2)) : (blk_start * HMX_QK_Q8x4x2);
          s->quant_width = is_q4 ? (nb_sub * (HMX_QK_Q4x4x2 / 2)) : (nb_sub * HMX_QK_Q8x4x2);
          s->scale_off   = full_qrow + blk_start * HMX_X4X2_DBLK_SIZE;
          s->scale_width = nb_sub * HMX_X4X2_DBLK_SIZE;
          s->dma         = ctx->dma[1];

          worker_pool_run_func(ctx->worker_pool, qweight_fetch_worker_fn, s, 1);
        }
        TIMER_STOP(fetch);

        TIMER_START(load);
        // load activation block
        {
          // const float *activation_block = x + mr * k + kk;
          // transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_block, m_blk_sz, k_blk_sz, k);
          // transfer_activation_chunk_multithread(vtcm_activation, activation_block, m_blk_sz, k_blk_sz, k);

          // NOTE(hzx): This code assumes that the activation block already resides in VTCM
          // transfer_activation_chunk_no_prefetch(vtcm_activation, (float *) vtcm_scratch1, m_blk_sz, k_blk_sz, k_blk_sz);
          transfer_activation_chunk_multithread(ctx, vtcm_activation, (float *) vtcm_scratch1, m_blk_sz, k_blk_sz, k_blk_sz);
        }

        // dequantize weight block
        {
          // vtcm_scratch0 is used to store the qweight chunk
          // worker_pool_run_func already returned, so fetch is done
          const size_t sub_row_stride = get_x4x2_row_stride(weight_type, k_blk_sz);
          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight, vtcm_scratch0,
                                                      n_blk_sz, k_blk_sz, sub_row_stride, weight_type);
        }
        TIMER_STOP(load);

        // core mma
        TIMER_START(core);
        {
          core_mma_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, vtcm_eye_tile, n_row_tiles,
                              n_col_tiles, k_blk_sz / HMX_FP16_TILE_N_COLS, kk == 0, ctx->vtcm_rctx);
        }
        TIMER_STOP(core);
      }

      // store output block
      {
        float *output_block = out + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output_block, vtcm_output, m_blk_sz, n_blk_sz, n);
      }
    }
  }

  FARF(ALWAYS, "fetch: %lld us, load: %lld us, core: %lld us", TIMER_US(fetch),
       TIMER_US(load), TIMER_US(core));

  return 0;
}
