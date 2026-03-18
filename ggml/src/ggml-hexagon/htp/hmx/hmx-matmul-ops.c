#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <HAP_farf.h>
#include <HAP_compute_res.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hmx-utils.h"
#include "hmx-hvx-convert.h"
#include "hmx-hvx-internal.h"
#include "hmx-ops.h"
#include "hmx-quants.h"
#include "htp-ctx.h"
#include "worker-pool.h"
#include "hex-dma.h"

#include "hmx-profile.h"

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

// ---------------------------------------------------------------------------
// Overflow-safe DMA push: all UDMA type1 descriptor fields (roiwidth,
// roiheight, srcstride, dststride) are 16-bit, max 65535.  This helper
// transparently handles values that exceed the 16-bit limit.
//
// Case 1 (fast path): all params fit in 16 bits -> direct dma_queue_push.
// Case 2 (contiguous block): width == srcstride == dststride.  Reshape the
//   flat transfer into a 2D descriptor with sub_width <= 65535.  Produces a
//   single descriptor, preserving async DMA behavior.
// Case 3 (stride overflow): srcstride or dststride > 65535.  Issue rows
//   one at a time.  The first N-1 rows are pushed+popped synchronously;
//   the last row is left async so the caller can pop it.
// ---------------------------------------------------------------------------
#define UDMA_MAX_FIELD_VAL 65535u

static inline bool hmx_dma_push_safe(dma_queue *q,
                                     dma_ptr    dptr,
                                     size_t     dst_stride,
                                     size_t     src_stride,
                                     size_t     width,
                                     size_t     nrows) {
  // Fast path: everything fits in 16 bits.
  if (__builtin_expect(
        width     <= UDMA_MAX_FIELD_VAL &&
        nrows     <= UDMA_MAX_FIELD_VAL &&
        src_stride <= UDMA_MAX_FIELD_VAL &&
        dst_stride <= UDMA_MAX_FIELD_VAL, 1)) {
    return dma_queue_push(q, dptr, dst_stride, src_stride, width, nrows);
  }

  // Case 2: contiguous block (width == src_stride == dst_stride).
  // Reshape total bytes into sub_width * sub_nrows where sub_width <= 65535.
  if (width == src_stride && width == dst_stride) {
    size_t total = width * nrows;

    // Pick the largest 128-byte-aligned sub_width that divides total evenly.
    size_t sub_width = UDMA_MAX_FIELD_VAL & ~(size_t)127;  // 65408
    while (sub_width > 0 && total % sub_width != 0) {
      sub_width -= 128;
    }
    if (sub_width == 0) {
      // Fallback: use original width (must fit) with adjusted nrows.
      // This shouldn't happen for 128-aligned DMA sizes.
      sub_width = width;
    }
    size_t sub_nrows = total / sub_width;

    // Handle sub_nrows > 65535 by issuing chunked descriptors.
    const uint8_t *src = (const uint8_t *)dptr.src;
    uint8_t       *dst = (uint8_t *)dptr.dst;
    size_t rows_done = 0;
    while (rows_done < sub_nrows) {
      size_t chunk = sub_nrows - rows_done;
      if (chunk > UDMA_MAX_FIELD_VAL) chunk = UDMA_MAX_FIELD_VAL;

      dma_ptr p = dma_make_ptr(dst + rows_done * sub_width,
                               src + rows_done * sub_width);
      if (!dma_queue_push(q, p, sub_width, sub_width, sub_width, chunk))
        return false;

      rows_done += chunk;
      // Complete all chunks without waiting except the last one, so the
      // caller's single dma_queue_pop drains the final descriptor.
      if (rows_done < sub_nrows)
        dma_queue_pop_nowait(q);
    }
    return true;
  }

  // Case 3: stride overflow — fall back to row-by-row.
  {
    const uint8_t *src = (const uint8_t *)dptr.src;
    uint8_t       *dst = (uint8_t *)dptr.dst;
    for (size_t r = 0; r < nrows; ++r) {
      dma_ptr p = dma_make_ptr(dst + r * dst_stride,
                               src + r * src_stride);
      if (!dma_queue_push(q, p, 0, 0, width, 1))
        return false;
      if (r + 1 < nrows)
        dma_queue_pop_nowait(q);
    }
    return true;
  }
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


// --- Overflow-safe arithmetic for VTCM budget calculation ---

static inline bool hmx_mul_overflow(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static inline bool hmx_add_overflow(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return true;
    *out = a + b;
    return false;
}

// Search for optimal (mc, nc) chunk sizes that maximize mc * nc within VTCM budget.
//
// Cost model: total = nc * per_n_cost + mc * per_m_cost + mc * nc * per_mn_cost + overhead
//   per_n_cost:  bytes per nc column (weight + scratch buffers)
//   per_m_cost:  bytes per mc row (activation)
//   per_mn_cost: bytes per mc*nc element (output)
//   overhead:    fixed bytes (scales 256B, eye_tile 2048B, etc.)
//
// Algorithm: nc sweeps from n_max down by 32, analytically solving for mc_max.
// Returns 0 on success, -1 if VTCM is insufficient.
static int hmx_compute_chunks(
    size_t vtcm_total, size_t overhead,
    size_t per_n_cost, size_t per_m_cost, size_t per_mn_cost,
    int m, int n,
    size_t *m_chunk_out, size_t *n_chunk_out,
    size_t *total_out)
{
    if (m <= 0 || n <= 0) return -1;
    if (vtcm_total <= overhead) return -1;
    if (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) return -1;

    const size_t usable = vtcm_total - overhead;
    size_t best_mn = 0, best_m = 0, best_n = 0;

    const size_t n_max = hmx_align_down((size_t)n, HMX_FP16_TILE_N_COLS);
    for (size_t nc = n_max; nc >= HMX_FP16_TILE_N_COLS; nc -= HMX_FP16_TILE_N_COLS) {
        // Early exit: if nc * m_max cannot beat best, smaller nc won't either
        if (nc * hmx_align_down((size_t)m, HMX_FP16_TILE_N_ROWS) <= best_mn)
            break;

        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        if (hmx_mul_overflow(nc, per_n_cost, &n_fixed)) continue;
        if (n_fixed >= usable) goto next_nc;

        if (hmx_mul_overflow(nc, per_mn_cost, &ncmn)) goto next_nc;
        if (hmx_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) goto next_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hmx_align_down(mc, HMX_FP16_TILE_N_ROWS);
            mc = hmx_smin(mc, (size_t)m);

            if (mc > 0 && mc * nc > best_mn) {
                best_mn = mc * nc;
                best_m  = mc;
                best_n  = nc;
            }
        }

next_nc:
        if (nc == HMX_FP16_TILE_N_COLS) break;  // avoid size_t underflow
    }

    if (best_m == 0 || best_n == 0) return -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, total = 0;
    if (hmx_mul_overflow(best_n, per_n_cost, &t0)) return -1;
    if (hmx_mul_overflow(best_m, per_m_cost, &t1)) return -1;
    if (hmx_mul_overflow(best_m, best_n, &mn)) return -1;
    if (hmx_mul_overflow(mn, per_mn_cost, &t2)) return -1;
    if (hmx_add_overflow(t0, t1, &total)) return -1;
    if (hmx_add_overflow(total, t2, &total)) return -1;
    if (hmx_add_overflow(total, overhead, &total)) return -1;

    *m_chunk_out = best_m;
    *n_chunk_out = best_n;
    *total_out   = total;
    return 0;
}

// forward declaration – defined after transfer_activation_chunk_fp32_to_fp16
void transfer_activation_chunk_multithread(struct htp_context *ctx, __fp16 *dst, const float *src, int n_rows, int k_block, int k_stride);

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

// Dequantize one x4x2 Q4_0 group (32 elements from 32 packed bytes) -> 32 FP16 in first 64 bytes.
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

  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hf, v_scales));
}

// Batch-dequantize 4 contiguous x4x2 Q4_0 groups (4x32 = 128 packed bytes) using
// full HVX vector width.  One vmemu + one vlut16 replaces 4 separate calls.
// Output: out[0..3] each hold 32 FP16 values in the first 64 bytes.
static inline void dequantize_x4x2_q4_0_x4groups_hvx(
    const uint8_t *packed_128, bool upper_nibbles,
    const __fp16 *scales_4, const HVX_Vector vlut_cvt,
    HVX_Vector out[4]) {
  // Load all 128 packed bytes (4 contiguous 32-byte groups)
  HVX_Vector vq = hmx_vmemu(packed_128);
  const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
  HVX_Vector v_quants = upper_nibbles ? Q6_Vub_vlsr_VubR(vq, 4) : vq;
  v_quants = Q6_V_vand_VV(v_quants, mask_h4);

  // Full-width vlut16: 128 byte lookups -> 128 fp16 results in a VectorPair
  HVX_VectorPair vp = Q6_Wh_vlut16_VbVhR(v_quants, vlut_cvt, 0);
  // Deinterleave to linear order: lo = bytes 0..63, hi = bytes 64..127
  HVX_VectorPair vp_lin = Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -2);
  HVX_Vector v_lo = Q6_V_lo_W(vp_lin);  // [group0: 32 fp16 | group1: 32 fp16]
  HVX_Vector v_hi = Q6_V_hi_W(vp_lin);  // [group2: 32 fp16 | group3: 32 fp16]

  // Build per-group scale vectors: first 64 bytes use scale_a, last 64 use scale_b
  HVX_VectorPred q64 = Q6_Q_vsetq_R(64);
  HVX_Vector v_sc01 = Q6_V_vmux_QVV(q64,
      Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)&scales_4[0])),
      Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)&scales_4[1])));
  HVX_Vector v_sc23 = Q6_V_vmux_QVV(q64,
      Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)&scales_4[2])),
      Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)&scales_4[3])));

  v_lo = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_lo, v_sc01));
  v_hi = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hi, v_sc23));

  // Extract individual groups: scatter uses q_mask64 so only first 64 bytes matter
  out[0] = v_lo;                      // group0 already in [0:63]
  out[1] = Q6_V_vror_VR(v_lo, 64);    // group1 rotated to [0:63]
  out[2] = v_hi;                      // group2 already in [0:63]
  out[3] = Q6_V_vror_VR(v_hi, 64);    // group3 rotated to [0:63]
}

// Dequantize one x4x2 Q8_0 group (32 int8 quants) -> 32 FP16 in first 64 bytes.
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

  for (int t = start_tile; t < end_tile; ) {
    int ct = t / n_k_tiles;  // column tile index
    int kt = t % n_k_tiles;  // K tile index

    // --- Batch-4 fast path for Q4: process 4 contiguous K-tiles with one vlut16 per row ---
    if (is_q4 && (kt % 4 == 0) && (t + 4 <= end_tile) && ((t + 3) / n_k_tiles == ct)) {
      int blk_idx      = (kt * 32) / HMX_QK_Q4x4x2;
      int sub_blk_base = ((kt * 32) % HMX_QK_Q4x4x2) / 32;  // 0 or 4
      bool upper       = (sub_blk_base >= 4);
      int packed_off   = blk_idx * (HMX_QK_Q4x4x2 / 2);     // 128 contiguous packed bytes
      int scale_off    = qrow_size + blk_idx * HMX_X4X2_DBLK_SIZE
                       + sub_blk_base * (int)sizeof(__fp16);   // 4 consecutive scales

      __fp16 *tile_bases[4];
      for (int g = 0; g < 4; g++)
        tile_bases[g] = vtcm_dst + (t + g) * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v_off = v_scat_base;
      for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
        int row0 = ct * HMX_FP16_TILE_N_COLS + r;
        int row1 = row0 + 1;
        const uint8_t *r0 = vtcm_src + row0 * row_stride;
        const uint8_t *r1 = vtcm_src + row1 * row_stride;

        HVX_Vector v0[4], v1[4];
        dequantize_x4x2_q4_0_x4groups_hvx(
            r0 + packed_off, upper, (const __fp16 *)(r0 + scale_off), vlut_cvt, v0);
        if (row1 < n_cols) {
          dequantize_x4x2_q4_0_x4groups_hvx(
              r1 + packed_off, upper, (const __fp16 *)(r1 + scale_off), vlut_cvt, v1);
        } else {
          v1[0] = v1[1] = v1[2] = v1[3] = Q6_V_vzero();
        }

        for (int g = 0; g < 4; g++)
          Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_bases[g], HMX_FP16_TILE_SIZE - 1, v_off, v0[g]);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        for (int g = 0; g < 4; g++)
          Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_bases[g], HMX_FP16_TILE_SIZE - 1, v_off, v1[g]);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
      }

      for (int g = 0; g < 4; g++)
        (void) *(volatile HVX_Vector *)(tile_bases[g]);

      t += 4;
      continue;
    }

    // --- Single-tile fallback ---
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

        Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v0);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v1);
        v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
      }
      (void) *(volatile HVX_Vector *)(tile_base);
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
      (void) *(volatile HVX_Vector *)(tile_base);
    }
    ++t;
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

// requires external HMX lock
static void core_dot_chunk_fp16(__fp16 *output, const __fp16 *activation, const __fp16 *weight, const __fp16 *scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
  hmx_set_output_scales(scales);

  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < n_col_tiles; ++c) {
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
}

static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows, int n_cols, int n) {
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

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + (r * n + c + 0));
      HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r * n + c + n));  // next row in global memory

      *pv_out0 = Q6_V_lo_W(vp);
      if (r + 1 < n_rows) {
        *pv_out1 = Q6_V_hi_W(vp);
      }
    }
  }
}

typedef struct {
  int            n_tasks;
  int            n_tot_chunks;
  int            n_chunks_per_task;
  float         *dst;
  const __fp16  *vtcm_src;
  int            n_cols;
  int            n;  // DDR row stride (total output columns)
} output_transfer_task_state_t;

static void transfer_output_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
  output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

  for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
    int    chunk_idx  = task_id * st->n_chunks_per_task;
    size_t chunk_size = hmx_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

    float        *dst     = st->dst     + chunk_idx * st->n;
    const __fp16 *vtcm_src = st->vtcm_src + chunk_idx * st->n_cols;
    transfer_output_chunk_fp16_to_fp32(dst, vtcm_src, chunk_size, st->n_cols, st->n);
  }
}

static void transfer_output_chunk_multithread(struct htp_context *ctx, float *dst, const __fp16 *vtcm_src,
                                              int n_rows, int n_cols, int n) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  size_t n_tot_chunks      = n_rows;
  size_t n_chunks_per_task = 32;  // must be multiple of HMX_FP16_TILE_N_ROWS (32)

  output_transfer_task_state_t state;
  state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task;
  state.n_tot_chunks      = n_tot_chunks;
  state.n_chunks_per_task = n_chunks_per_task;
  state.dst               = dst;
  state.vtcm_src          = vtcm_src;
  state.n_cols            = n_cols;
  state.n                 = n;

  worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, ctx->n_threads);
}

static inline int hmx_matmul_batch_r2(const hmx_matmul_w16a32_batched_params_t *params) {
  return params->ne02 > 0 ? params->ne12 / params->ne02 : 1;
}

static inline int hmx_matmul_batch_r3(const hmx_matmul_w16a32_batched_params_t *params) {
  return params->ne03 > 0 ? params->ne13 / params->ne03 : 1;
}

static inline const __fp16 *hmx_matmul_weight_batch_ptr(const hmx_matmul_w16a32_batched_params_t *params,
                                                        int dst_b2, int dst_b3) {
  const int r2 = hmx_matmul_batch_r2(params);
  const int r3 = hmx_matmul_batch_r3(params);
  return (const __fp16 *) ((const uint8_t *) params->permuted_weight +
                           (size_t) (dst_b2 / r2) * params->src0_nb2 +
                           (size_t) (dst_b3 / r3) * params->src0_nb3);
}

static inline const float *hmx_matmul_activation_batch_ptr(const hmx_matmul_w16a32_batched_params_t *params,
                                                           int dst_b2, int dst_b3) {
  return (const float *) ((const uint8_t *) params->activation +
                          (size_t) dst_b2 * params->src1_nb2 +
                          (size_t) dst_b3 * params->src1_nb3);
}

static inline float *hmx_matmul_dst_batch_ptr(const hmx_matmul_w16a32_batched_params_t *params,
                                              int dst_b2, int dst_b3) {
  return (float *) ((uint8_t *) params->dst +
                    (size_t) dst_b2 * params->dst_nb2 +
                    (size_t) dst_b3 * params->dst_nb3);
}

static int hmx_mat_mul_permuted_w16a32_batched_legacy(struct htp_context *ctx,
                                                      const hmx_matmul_w16a32_batched_params_t *params) {
  int ret = 0;
  for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
    for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
      ret = hmx_mat_mul_permuted_w16a32(ctx,
                                        hmx_matmul_dst_batch_ptr(params, b2, b3),
                                        hmx_matmul_activation_batch_ptr(params, b2, b3),
                                        hmx_matmul_weight_batch_ptr(params, b2, b3),
                                        params->m, params->k, params->n,
                                        params->act_stride, params->weight_stride);
    }
  }
  return ret;
}

int hmx_mat_mul_permuted_w16a32_batched(struct htp_context *ctx,
                                        const hmx_matmul_w16a32_batched_params_t *params) {
  if (!ctx || !params || !params->dst || !params->activation || !params->permuted_weight) {
    return -1;
  }
  if (!params->m || !params->k || !params->n) {
    return -1;
  }
  if (params->act_stride < params->k || params->weight_stride < params->k || params->dst_stride < params->n) {
    return -1;
  }
  if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) {
    return -1;
  }
  if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) {
    return -1;
  }
  if (params->k % 32 != 0 || params->n % 32 != 0) {
    return -1;
  }
  if (!hmx_is_aligned(params->dst, HMX_VLEN) ||
      !hmx_is_aligned(params->activation, HMX_VLEN) ||
      !hmx_is_aligned(params->permuted_weight, HMX_VLEN)) {
    return -1;
  }

  const int group_size = hmx_matmul_batch_r2(params);

  if (group_size <= 1) {
    FARF(MEDIUM, "%s: no dim2 GQA reuse (group=%d), using legacy batched loop", __func__, group_size);
    return hmx_mat_mul_permuted_w16a32_batched_legacy(ctx, params);
  }

  // Grouped path: reuse interleaved weight across all q_heads sharing a
  // kv_head.  Each q_head gets its own activation buffer in VTCM (so
  // activation is loaded once per m_chunk and reused across all n_chunks),
  // and each q_head is computed individually to avoid tile-major packing
  // issues.  m_chunk_n_rows is always a multiple of 32 (from
  // hmx_compute_chunks), so per-head tile arrays don't overlap.
  const size_t vtcm_budget  = ctx->vtcm_scratch_size;
  const size_t vec_dot_size = params->k * sizeof(__fp16);

  // When the activation has a large stride (e.g. permuted Q tensor with
  // act_stride >> k), HVX vector loads from strided DDR thrash L2 cache.
  // Allocate an F32 scratch buffer in VTCM and use 2D DMA to gather
  // strided rows into a contiguous block before the F32->F16 conversion.
  const bool use_dma_activation = (params->act_stride > params->k);
  const size_t f32_scratch_per_m = use_dma_activation ? (size_t) params->k * sizeof(float) : 0;

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0, vtcm_used = 0;
  if (hmx_compute_chunks(vtcm_budget, /*overhead=*/256,
                         /*per_n=*/3 * vec_dot_size,
                         /*per_m=*/group_size * vec_dot_size + f32_scratch_per_m,
                         /*per_mn=*/sizeof(__fp16),
                         params->m, params->n,
                         &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used) != 0) {
    FARF(HIGH, "%s: grouped path does not fit VTCM, falling back to legacy batched loop", __func__);
    return hmx_mat_mul_permuted_w16a32_batched_legacy(ctx, params);
  }

  const size_t act_head_stride      = m_chunk_n_rows * (size_t) params->k;  // fp16 elements between heads
  const size_t weight_area_size     = hmx_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t activation_area_size = hmx_align_up(group_size * m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t output_area_size     = hmx_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);
  const size_t scratch_area_size    = hmx_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t f32_scratch_size     = use_dma_activation
      ? hmx_align_up(m_chunk_n_rows * (size_t) params->k * sizeof(float), HMX_FP16_TILE_SIZE) : 0;

  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  float   *vtcm_f32_act    = use_dma_activation
      ? (float *) vtcm_seq_alloc(&vtcm_ptr, f32_scratch_size) : NULL;
  if ((size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base) > vtcm_budget) {
    FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to legacy batched loop", __func__);
    return hmx_mat_mul_permuted_w16a32_batched_legacy(ctx, params);
  }

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  FARF(MEDIUM, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
       __func__, params->m, params->k, params->n, group_size, params->ne13,
       m_chunk_n_rows, n_chunk_n_cols,
       (size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base), vtcm_budget);

  TIMER_DEFINE(activation_load);
  TIMER_DEFINE(weight_load);
  TIMER_DEFINE(hmx_core);
  TIMER_DEFINE(output_store);
  TIMER_DEFINE(total);

  TIMER_START(total);

  const size_t fp16_row_bytes   = (size_t) params->k * sizeof(__fp16);
  const size_t weight_row_bytes = (size_t) params->weight_stride * sizeof(__fp16);

  for (int b3 = 0; b3 < params->ne13; ++b3) {
    for (int b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
      const __fp16 *weight_group = hmx_matmul_weight_batch_ptr(params, b2_base, b3);

      for (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
        const size_t n_rows = hmx_smin((size_t) params->m - mr, m_chunk_n_rows);

        // Pre-load activations for all heads in the group (once per m_chunk).
        // When the source is strided (permuted Q), use 2D DMA to gather
        // contiguous rows into a VTCM scratch buffer first, then HVX
        // converts from the contiguous VTCM buffer.  This avoids L2 cache
        // thrashing from HVX loads at large strides.
        TIMER_START(activation_load);
        for (int g = 0; g < group_size; ++g) {
          const float *activation_chunk = hmx_matmul_activation_batch_ptr(params, b2_base + g, b3) +
                                          mr * params->act_stride;
          __fp16 *vtcm_act_g = vtcm_activation + (size_t) g * act_head_stride;
          if (use_dma_activation) {
            const size_t row_bytes    = (size_t) params->k * sizeof(float);
            const size_t stride_bytes = (size_t) params->act_stride * sizeof(float);
            hmx_dma_push_safe(ctx->dma[0],
                              dma_make_ptr(vtcm_f32_act, activation_chunk),
                              row_bytes, stride_bytes, row_bytes, n_rows);
            dma_queue_pop(ctx->dma[0]);
            transfer_activation_chunk_multithread(ctx, vtcm_act_g,
                                                  vtcm_f32_act, (int) n_rows,
                                                  params->k, params->k);
          } else {
            transfer_activation_chunk_multithread(ctx, vtcm_act_g,
                                                  activation_chunk, (int) n_rows,
                                                  params->k, params->act_stride);
          }
        }
        TIMER_STOP(activation_load);

        void *buf_curr = vtcm_scratch0;
        void *buf_next = vtcm_scratch1;

        {
          const size_t n_cols_first = hmx_smin((size_t) params->n, n_chunk_n_cols);
          hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(buf_curr, weight_group),
                            fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
        }

	HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

        for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
          const size_t n_cols = hmx_smin((size_t) params->n - nc, n_chunk_n_cols);

          TIMER_START(weight_load);
          {
            dma_queue_pop(ctx->dma[0]);

            const size_t nc_next = nc + n_chunk_n_cols;
            if (nc_next < (size_t) params->n) {
              const size_t n_cols_next = hmx_smin((size_t) params->n - nc_next, n_chunk_n_cols);
              const __fp16 *next_weight_chunk = weight_group + nc_next * params->weight_stride;

              hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk),
                                fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
            }

            interleave_fp16_weight_chunk_to_tiles(vtcm_weight, (const __fp16 *) buf_curr, n_cols, params->k);
            swap_ptr(&buf_curr, &buf_next);
          }
          TIMER_STOP(weight_load);

          // Reuse the interleaved weight for every q_head in this GQA group
          for (int g = 0; g < group_size; ++g) {
            TIMER_START(hmx_core);
            {
              const __fp16 *vtcm_act_g = vtcm_activation + (size_t) g * act_head_stride;
              const int n_row_tiles = hmx_ceil_div((int) n_rows, HMX_FP16_TILE_N_ROWS);
              const int n_col_tiles = hmx_ceil_div((int) n_cols, HMX_FP16_TILE_N_COLS);
              core_dot_chunk_fp16(vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales,
                                  n_row_tiles, n_col_tiles, params->k / 32);
            }
            TIMER_STOP(hmx_core);

            TIMER_START(output_store);
            {
              float *output = hmx_matmul_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
              transfer_output_chunk_multithread(ctx, output, vtcm_output, (int) n_rows, (int) n_cols, params->dst_stride);
            }
            TIMER_STOP(output_store);
          }
        }

	HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
      }
    }
  }

  TIMER_STOP(total);

#if defined(ENABLE_PROFILE_TIMERS)
  FARF(HIGH, "%s: %lld us, m=%d k=%d n=%d group=%d", __func__, TIMER_US(total),
       params->m, params->k, params->n, group_size);
  FARF(HIGH, "  activation_load: %lld us, weight_load: %lld us, hmx_core: %lld us, output_store: %lld us",
       TIMER_US(activation_load), TIMER_US(weight_load), TIMER_US(hmx_core), TIMER_US(output_store));
#endif

  return 0;
}

int hmx_mat_mul_permuted_w16a32(struct htp_context *ctx, float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict permuted_weight, int m, int k, int n,
                                int act_stride, int weight_stride) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (act_stride < k || weight_stride < k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    return -1;
  }
  if (!hmx_is_aligned(dst, HMX_VLEN) || !hmx_is_aligned(activation, HMX_VLEN) || !hmx_is_aligned(permuted_weight, HMX_VLEN)) {
    return -1;
  }

  // --- Dynamic VTCM layout ---
  const size_t vtcm_budget  = ctx->vtcm_scratch_size;
  const size_t vec_dot_size = k * sizeof(__fp16);

  // DMA-based activation gather for strided tensors (see batched path comment).
  const bool use_dma_activation = (act_stride > k);
  const size_t f32_scratch_per_m = use_dma_activation ? (size_t) k * sizeof(float) : 0;

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0, vtcm_used = 0;
  if (hmx_compute_chunks(vtcm_budget, /*overhead=*/256,
                          /*per_n=*/3 * vec_dot_size,  // W + S0 + S1
                          /*per_m=*/vec_dot_size + f32_scratch_per_m,  // A + optional F32 scratch
                          /*per_mn=*/sizeof(__fp16),     // O
                          m, n,
                          &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used) != 0) {
    FARF(HIGH, "%s: VTCM too small (m=%d k=%d n=%d budget=%zu)", __func__, m, k, n, vtcm_budget);
    return -1;
  }

  const size_t weight_area_size     = hmx_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t activation_area_size = hmx_align_up(m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t output_area_size     = hmx_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);
  const size_t scratch_area_size    = hmx_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t f32_scratch_size     = use_dma_activation
      ? hmx_align_up(m_chunk_n_rows * (size_t) k * sizeof(float), HMX_FP16_TILE_SIZE) : 0;

  // VTCM layout: weight | activation | output | scratch0 | scratch1 | scales | [f32_scratch]
  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  float   *vtcm_f32_act    = use_dma_activation ? (float *) vtcm_seq_alloc(&vtcm_ptr, f32_scratch_size) : NULL;
  if ((size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base) > vtcm_budget) {
    FARF(ERROR, "%s: vtcm overflow: used=%zu limit=%zu", __func__,
         (size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base), vtcm_budget);
    return -1;
  }

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  FARF(MEDIUM, "%s: m=%d k=%d n=%d mc=%zu nc=%zu vtcm=%zu/%zu",
       __func__, m, k, n, m_chunk_n_rows, n_chunk_n_cols,
       (size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base), vtcm_budget);

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
      const float *activation_chunk = activation + mr * act_stride;
      if (use_dma_activation) {
        const size_t row_bytes    = (size_t) k * sizeof(float);
        const size_t stride_bytes = (size_t) act_stride * sizeof(float);
        hmx_dma_push_safe(ctx->dma[0],
                          dma_make_ptr(vtcm_f32_act, activation_chunk),
                          row_bytes, stride_bytes, row_bytes, n_rows);
        dma_queue_pop(ctx->dma[0]);
        transfer_activation_chunk_multithread(ctx, vtcm_activation,
                                              vtcm_f32_act, n_rows, k, k);
      } else {
        transfer_activation_chunk_multithread(ctx, vtcm_activation,
                                              activation_chunk, n_rows, k, act_stride);
      }
    }
    TIMER_STOP(activation_load);

    const size_t fp16_row_bytes    = (size_t) k * sizeof(__fp16);
    const size_t weight_row_bytes  = (size_t) weight_stride * sizeof(__fp16);

    void *buf_curr = vtcm_scratch0;
    void *buf_next = vtcm_scratch1;

    // issue async DMA for the first weight chunk
    // NOTE: use 2D DMA (n_cols rows x fp16_row_bytes) to avoid 16-bit roiwidth overflow.
    // The source rows can be strided (e.g. KV-cache K after ggml_permute).
    {
      const size_t n_cols_first = hmx_smin(n, n_chunk_n_cols);

      hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(buf_curr, permuted_weight),
                        fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
    }

    HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = hmx_smin(n - nc, n_chunk_n_cols);

      TIMER_START(weight_load);
      {
        dma_queue_pop(ctx->dma[0]);  // wait until current weight chunk is ready

        // issue async DMA for the next weight chunk (double buffering)
        const size_t nc_next = nc + n_chunk_n_cols;
        if (nc_next < n) {
          const size_t n_cols_next       = hmx_smin(n - nc_next, n_chunk_n_cols);
          const __fp16 *next_weight_chunk = permuted_weight + nc_next * weight_stride;

          hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk),
                            fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
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
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
      }
      TIMER_STOP(hmx_core);

      TIMER_START(output_store);
      {
        float *output = dst + (mr * n + nc);
        transfer_output_chunk_multithread(ctx, output, vtcm_output, n_rows, n_cols, n);
      }
      TIMER_STOP(output_store);
    }

    HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
  }

  TIMER_STOP(total);

#if defined(ENABLE_PROFILE_TIMERS)
  FARF(HIGH, "%s: %lld us, m=%d k=%d n=%d", __func__, TIMER_US(total), m, k, n);
  FARF(HIGH, "  activation_load: %lld us, weight_load: %lld us, hmx_core: %lld us, output_store: %lld us",
       TIMER_US(activation_load), TIMER_US(weight_load), TIMER_US(hmx_core), TIMER_US(output_store));
  {
    size_t weight_size = (size_t)k * n * sizeof(__fp16);
    float  bandwidth   = 1e-3f * weight_size / (float)TIMER_US(weight_load);
    FARF(HIGH, "  weight load bandwidth: %.2f GB/s", bandwidth);
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

  HAP_compute_res_hmx_lock(st->vtcm_rctx);

  core_dot_chunk_fp16(st->c, st->a, st->b, st->s, st->n_row_tiles, st->n_col_tiles, st->n_dot_tiles);

  HAP_compute_res_hmx_unlock(st->vtcm_rctx);
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
    return -1;
  }
  if (!hmx_is_aligned(dst, HMX_VLEN) || !hmx_is_aligned(activation, HMX_VLEN) || !hmx_is_aligned(permuted_weight, HMX_VLEN)) {
    return -1;
  }

  // for large m, k (e.g. prefill FFN Down), use out-stationary version
  if (m >= 128 && k > n && n > 1024) {
    FARF(MEDIUM, "hmx_matmul_qk: OUT-STATIONARY path m=%d k=%d n=%d type=%d (K_BLOCK=512, %d K-iters with fp16 intermediate)",
         m, k, n, weight_type, (k + 511) / 512);
    return mat_mul_qk_0_d16a32_out_stationary(ctx, dst, activation, permuted_weight, m, k, n, weight_type);
  }

  size_t row_stride = get_x4x2_row_stride(weight_type, k);
  if (row_stride == 0) {
    return -1;
  }

  FARF(MEDIUM, "hmx_matmul_qk: STANDARD path m=%d k=%d n=%d type=%d", m, k, n, weight_type);

  // --- Dynamic VTCM layout ---
  const size_t vtcm_budget   = ctx->vtcm_scratch_size;
  const size_t vec_dot_size  = k * sizeof(__fp16);
  const bool   use_pipeline  = (m >= 128) && (k <= n);

  // Select cost parameters based on execution path
  size_t per_n_cost, per_mn_cost;
  if (use_pipeline) {
    per_n_cost  = row_stride + 2 * vec_dot_size;  // Q + S0 + S1 (dequant bufs)
    per_mn_cost = 2 * sizeof(__fp16);              // O x 2 (output double buffer)
  } else {
    per_n_cost  = vec_dot_size + 2 * row_stride;   // W + S0 + S1 (x4x2 DMA bufs)
    per_mn_cost = sizeof(__fp16);                   // O x 1
  }

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0, vtcm_used = 0;
  if (hmx_compute_chunks(vtcm_budget, /*overhead=*/256,
                          per_n_cost, /*per_m=*/vec_dot_size, per_mn_cost,
                          m, n, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used) != 0) {
    FARF(HIGH, "%s: VTCM too small (m=%d k=%d n=%d pipe=%d budget=%zu)",
         __func__, m, k, n, use_pipeline, vtcm_budget);
    return -1;
  }

  // Compute precise buffer sizes per execution path
  const size_t weight_area_size = hmx_align_up(
      n_chunk_n_cols * (use_pipeline ? row_stride : vec_dot_size), HMX_FP16_TILE_SIZE);
  const size_t activation_area_size = hmx_align_up(m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
  const size_t output_area_size = hmx_align_up(
      m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);

  size_t scratch0_size, scratch1_size, scratch2_size;
  if (use_pipeline) {
    scratch0_size = hmx_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);  // dequant buf 0
    scratch1_size = scratch0_size;                                                    // dequant buf 1
    scratch2_size = output_area_size;                                                 // output buf 1
  } else {
    scratch0_size = hmx_align_up(n_chunk_n_cols * row_stride, HMX_FP16_TILE_SIZE);    // x4x2 DMA buf 0
    scratch1_size = scratch0_size;                                                    // x4x2 DMA buf 1
    scratch2_size = 0;                                                                // unused
  }

  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch1_size);
  void    *vtcm_scratch2   = scratch2_size ? vtcm_seq_alloc(&vtcm_ptr, scratch2_size) : NULL;
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  if ((size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base) > vtcm_budget) {
    FARF(ERROR, "%s: vtcm overflow: used=%zu limit=%zu", __func__,
         (size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base), vtcm_budget);
    return -1;
  }

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  FARF(MEDIUM, "%s: m=%d k=%d n=%d wtype=%d pipe=%d mc=%zu nc=%zu vtcm=%zu/%zu",
       __func__, m, k, n, weight_type, use_pipeline,
       m_chunk_n_rows, n_chunk_n_cols,
       (size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base), vtcm_budget);

  TIMER_DEFINE(activation_load);
  TIMER_DEFINE(weight_load);
  TIMER_DEFINE(hmx_core);
  TIMER_DEFINE(output_store);

  TIMER_DEFINE(total);
  TIMER_START(total);

  FARF(MEDIUM, "hmx_matmul_qk: %s mc=%zu nc=%zu vtcm=%zu/%zu",
       use_pipeline ? "PIPELINE" : "SEQUENTIAL", m_chunk_n_rows, n_chunk_n_cols,
       (size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base), vtcm_budget);

  if (!use_pipeline) {
    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      // transfer activation matrix chunk into VTCM
      size_t n_rows = hmx_smin(m - mr, m_chunk_n_rows);

      TIMER_START(activation_load);
      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_multithread(ctx, vtcm_activation, activation_chunk, n_rows, k, k);
      }
      TIMER_STOP(activation_load);

      void *buf_curr = vtcm_scratch0;
      void *buf_next = vtcm_scratch1;

      // issue async DDR data transfer for the first weight chunk
      // NOTE: use 2D DMA (n_cols rows x row_stride bytes) instead of 1D
      // because UDMA roiwidth is 16-bit and total size can exceed 65535.
      {
        const size_t n_cols_first = hmx_smin(n, n_chunk_n_cols);

        hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(buf_curr, permuted_weight), row_stride, row_stride, row_stride, n_cols_first);
      }

      HAP_compute_res_hmx_lock(ctx->vtcm_rctx);

      for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
        size_t n_cols = hmx_smin(n - nc, n_chunk_n_cols);

        TIMER_START(weight_load);
        {
          dma_queue_pop(ctx->dma[0]);  // wait until current weight chunk become ready

          const size_t nc_next = nc + n_chunk_n_cols;
          if (nc_next < n) {
            const size_t n_cols_next = hmx_smin(n - nc_next, n_chunk_n_cols);

            const uint8_t *next_weight_chunk = permuted_weight + nc_next * row_stride;

            hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk), row_stride, row_stride, row_stride, n_cols_next);
          }

          // Dequant + vscatter writes directly to [K, N] transposed tiles.
          // HMX computes C = A x B, where A=[M,K] activation, B=[K,N] weight.
          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight, buf_curr,
                                                      n_cols, k, row_stride, weight_type);

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

        TIMER_START(output_store);
        {
          float *output = dst + (mr * n + nc);
          transfer_output_chunk_multithread(ctx, output, vtcm_output, n_rows, n_cols, n);
        }
        TIMER_STOP(output_store);
      }

      HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
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
        // Use 2D DMA (n_cols rows x row_stride) to avoid 16-bit roiwidth overflow.
        const uint8_t *qweight_chunk_A0 = permuted_weight;
        hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(vtcm_qweight, qweight_chunk_A0), row_stride, row_stride, row_stride, n_cols_A0);
      }

      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_multithread(ctx, vtcm_activation, activation_chunk, n_rows, k, k);
      }

      // prologue: B0, A1, C0, B1
      {
        // B0
        dma_queue_pop(ctx->dma[0]);
        dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight_bufs[0], vtcm_qweight, n_cols_A0, k, row_stride, weight_type);

        // A1
        const size_t n_cols_A1 = hmx_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
        if (1 < n_chunk_cnt) {
          const uint8_t *qweight_chunk_A1 = permuted_weight + n_chunk_n_cols * row_stride;
          hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(vtcm_qweight, qweight_chunk_A1), row_stride, row_stride, row_stride, n_cols_A1);
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
          const uint8_t *qweight_chunk_p2 = permuted_weight + nc_p2 * row_stride;
          hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(vtcm_qweight, qweight_chunk_p2), row_stride, row_stride, row_stride, n_cols_p2);
        }

        // wait for HMX (C_{i}) -- worker_pool_run_func already returned, so C_{i} is done

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
        transfer_output_chunk_multithread(ctx, output_chunk, vtcm_output_bufs[i % 2], n_rows, n_cols, n);

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

#if defined(ENABLE_PROFILE_TIMERS)
  FARF(HIGH, "%s: %lld us, m=%d k=%d n=%d pipeline=%d", __func__, TIMER_US(total), m, k, n, use_pipeline);
  if (!use_pipeline) {
    FARF(HIGH, "  activation_load: %lld us, weight_load: %lld us, hmx_core: %lld us, output_store: %lld us",
         TIMER_US(activation_load), TIMER_US(weight_load), TIMER_US(hmx_core), TIMER_US(output_store));
    size_t weight_size = (size_t)n * row_stride;
    float  bandwidth   = 1e-3f * weight_size / (float)TIMER_US(weight_load);
    FARF(HIGH, "  weight load bandwidth: %.2f GB/s", bandwidth);
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
      Q6_mxclracc_hf();

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

static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
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
  int          n_tasks;
  int          n_tot_chunks;
  int          n_chunks_per_task;
  __fp16      *dst;
  const float *src;
  int          k_block;
  int          k_stride;
} activation_transfer_task_state_t;

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
  activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

  for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
    // one chunk: one row
    int    chunk_idx  = task_id * st->n_chunks_per_task;
    size_t chunk_size = hmx_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

    __fp16      *dst = st->dst + chunk_idx * st->k_block;
    const float *src = st->src + chunk_idx * st->k_stride;
    transfer_activation_chunk_fp32_to_fp16(dst, src, chunk_size, st->k_block, st->k_stride);
  }
}

void transfer_activation_chunk_multithread(struct htp_context *ctx, __fp16 *dst, const float *src, int n_rows, int k_block, int k_stride) {
  assert(k_block % HMX_FP16_TILE_N_COLS == 0 && k_stride % HMX_FP16_TILE_N_COLS == 0);
  assert(HMX_VLEN == 32 * sizeof(float));

  size_t n_tot_chunks      = n_rows;
  size_t n_chunks_per_task = 32;  // must be multiple of 32 to ensure correct destination address

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
  // Runtime check -- k >= 16384 exceeds 2D DMA limit
  if (k >= 16384) {
    FARF(HIGH, "%s: k=%d exceeds 2D DMA limit", __func__, k);
    return -1;
  }
  // assume k % 32 == 0 && n % 32 == 0
  const size_t row_stride = get_x4x2_row_stride(weight_type, k);
  if (row_stride == 0) {
    return -1;
  }

  const size_t vtcm_budget = ctx->vtcm_scratch_size;

  const size_t M_BLOCK_SIZE = 512;
  const size_t N_BLOCK_SIZE = 512;
  const size_t K_BLOCK_SIZE = 512;

  // Compute precise buffer sizes
  const size_t sub_row_stride_alloc = get_x4x2_row_stride(weight_type, K_BLOCK_SIZE);
  const size_t weight_size  = hmx_align_up(N_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(__fp16), HMX_FP16_TILE_SIZE);
  const size_t act_size     = hmx_align_up(M_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(__fp16), HMX_FP16_TILE_SIZE);
  const size_t out_size     = hmx_align_up(M_BLOCK_SIZE * N_BLOCK_SIZE * sizeof(__fp16), HMX_FP16_TILE_SIZE);
  const size_t scratch0_sz  = hmx_align_up(N_BLOCK_SIZE * sub_row_stride_alloc, HMX_FP16_TILE_SIZE);
  const size_t scratch1_sz  = hmx_align_up(M_BLOCK_SIZE * K_BLOCK_SIZE * sizeof(float), HMX_FP16_TILE_SIZE);

  const size_t total_vtcm = weight_size + act_size + out_size
                           + scratch0_sz + scratch1_sz
                           + HMX_FP16_TILE_SIZE + 256;
  if (total_vtcm > vtcm_budget) {
    FARF(HIGH, "%s: VTCM too small: need %zu have %zu (m=%d k=%d n=%d)",
         __func__, total_vtcm, vtcm_budget, m, k, n);
    return -1;
  }

  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, out_size);
  uint8_t *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch0_sz);
  uint8_t *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch1_sz);
  __fp16  *vtcm_eye_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  assert((size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base) <= vtcm_budget);

  FARF(MEDIUM, "%s: m=%d k=%d n=%d wtype=%d vtcm=%zu/%zu",
       __func__, m, k, n, weight_type,
       (size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base), vtcm_budget);

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

  TIMER_DEFINE(fetch);
  TIMER_DEFINE(act_load);
  TIMER_DEFINE(wt_dequant);
  TIMER_DEFINE(core);

  for (size_t mr = 0; mr < m; mr += M_BLOCK_SIZE) {
    size_t m_blk_sz = hmx_smin(m - mr, M_BLOCK_SIZE);
    for (size_t nc = 0; nc < n; nc += N_BLOCK_SIZE) {
      size_t n_blk_sz = hmx_smin(n - nc, N_BLOCK_SIZE);

      const int n_row_tiles = hmx_ceil_div(m_blk_sz, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = hmx_ceil_div(n_blk_sz, HMX_FP16_TILE_N_COLS);

      for (size_t kk = 0; kk < k; kk += K_BLOCK_SIZE) {
        size_t k_blk_sz = hmx_smin(k - kk, K_BLOCK_SIZE);

        TIMER_START(fetch);
        // fetch activation block into VTCM
        {
          const float *activation_block = x + mr * k + kk;

          hmx_dma_push_safe(ctx->dma[0],
                           dma_make_ptr(vtcm_scratch1, activation_block),
                           k_blk_sz * sizeof(float),
                           k * sizeof(float),
                           k_blk_sz * sizeof(float),
                           m_blk_sz);
        }

        // fetch weight block into VTCM (x4x2 sub-block: quants + scales)
        {
          qweight_fetch_task_state_t s;

          const bool is_q4 = (weight_type == HMX_TYPE_Q4_0 || weight_type == HMX_TYPE_IQ4_NL);
          const int blk_start = kk / HMX_QK_Q4x4x2;
          const int nb_sub = (k_blk_sz + HMX_QK_Q4x4x2 - 1) / HMX_QK_Q4x4x2;
          const int full_qrow = is_q4 ? (k / 2) : k;
          const size_t sub_row_stride = get_x4x2_row_stride(weight_type, k_blk_sz);

          s.dst         = vtcm_scratch0;
          s.src         = w + nc * row_stride;
          s.n_rows      = n_blk_sz;
          s.src_stride  = row_stride;
          s.dst_stride  = sub_row_stride;
          s.quant_off   = is_q4 ? (blk_start * (HMX_QK_Q4x4x2 / 2)) : (blk_start * HMX_QK_Q8x4x2);
          s.quant_width = is_q4 ? (nb_sub    * (HMX_QK_Q4x4x2 / 2)) : (nb_sub * HMX_QK_Q8x4x2);
          s.scale_off   = full_qrow + blk_start * HMX_X4X2_DBLK_SIZE;
          s.scale_width = nb_sub * HMX_X4X2_DBLK_SIZE;

          // 2D DMA: quants sub-range
          hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(s.dst, s.src + s.quant_off),
                            s.dst_stride, s.src_stride, s.quant_width, s.n_rows);
          // 2D DMA: scales sub-range
          hmx_dma_push_safe(ctx->dma[0], dma_make_ptr(s.dst + s.quant_width, s.src + s.scale_off),
                            s.dst_stride, s.src_stride, s.scale_width, s.n_rows);
        }
        TIMER_STOP(fetch);

        TIMER_START(act_load);
        // load activation block
        {
          dma_queue_pop(ctx->dma[0]); // wait for act DNA
          transfer_activation_chunk_multithread(ctx, vtcm_activation, (float *) vtcm_scratch1, m_blk_sz, k_blk_sz, k_blk_sz);
        }
        TIMER_STOP(act_load);

        TIMER_START(wt_dequant);
        // dequantize weight block
        {
          dma_queue_pop(ctx->dma[0]);
          dma_queue_pop(ctx->dma[0]);
          // vtcm_scratch0 is used to store the qweight chunk
          // worker_pool_run_func already returned, so fetch is done
          const size_t sub_row_stride = get_x4x2_row_stride(weight_type, k_blk_sz);
          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight, vtcm_scratch0,
                                                      n_blk_sz, k_blk_sz, sub_row_stride, weight_type);
        }
        TIMER_STOP(wt_dequant);

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
        transfer_output_chunk_multithread(ctx, output_block, vtcm_output, m_blk_sz, n_blk_sz, n);
      }
    }
  }

#if defined(ENABLE_PROFILE_TIMERS)
  FARF(HIGH, "fetch: %lld us, act_load: %lld us, wt_dequant: %lld us, core: %lld us",
       TIMER_US(fetch), TIMER_US(act_load), TIMER_US(wt_dequant), TIMER_US(core));
#endif

  return 0;
}
