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
#include <HAP_perf.h>

#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (1 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (1 * 1024 * 1024)
#define SCRATCH_AREA_SIZE    (1 * 1024 * 1024)

static const __fp16 q4_0_to_fp16_lut[64] __attribute__((aligned(HMX_VLEN))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

static const __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(HMX_VLEN))) = {
  -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
  1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
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

static inline void hmx_dma_load_sync(dma_queue *dma, void *vtcm_dst, const void *src, size_t size) {
    dma_queue_push(dma, dma_make_ptr(vtcm_dst, src), size, size, size, 1);
    dma_queue_pop(dma);
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

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  int                    n_tasks;
  int                    n_tot_chunks;
  int                    n_chunks_per_task;
  int           k;
  __fp16       *dst;
  const __fp16 *src;
} permuted_weight_transfer_fp16_task_state_t;

static void transfer_permuted_weight_fp16_task(__fp16 *restrict vtcm_dst, const __fp16 *restrict src, int k,
                                               int n_col_tiles) {
  // transfer logical K*(32*n_col_tiles) weight block, direct copy, no extra computation
  size_t size   = k * n_col_tiles * HMX_FP16_TILE_N_COLS * sizeof(__fp16);
  int    n_vecs = size / HMX_VLEN;

  const size_t PREFETCH_SIZE   = 4096;
  const int    PREFETCH_N_VECS = PREFETCH_SIZE / HMX_VLEN;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        size_t prefetch_n_vecs = hmx_smin(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        hmx_l2fetch(pv_in + PREFETCH_N_VECS, HMX_VLEN, HMX_VLEN, prefetch_n_vecs, 0);
      }
    }

    *pv_out++ = *pv_in++;
  }
}

static void transfer_permuted_weight_fp16_worker_loop(unsigned int n, unsigned int i, void *data) {
  permuted_weight_transfer_fp16_task_state_t *state = (permuted_weight_transfer_fp16_task_state_t *) data;

  int k = state->k;

  for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = hmx_smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    int           c        = chunk_idx * HMX_FP16_TILE_N_COLS;
    __fp16       *vtcm_dst = state->dst + c * k;
    const __fp16 *src      = state->src + c * k;
    transfer_permuted_weight_fp16_task(vtcm_dst, src, k, chunk_size);
  }
}

static void transfer_permuted_weight_chunk_fp16(struct htp_context *ctx, __fp16 *vtcm_dst, const __fp16 *src, int n_cols, int k) {
  // NOTE(hzx): weight matrix is already transposed. n_cols actually turns into n_rows
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const bool use_dma = true;

  if (use_dma) {
    size_t size = n_cols * k * sizeof(__fp16);

    hmx_dma_load_sync(ctx->dma[0], vtcm_dst, src, size);

    return;
  }

  size_t n_tot_chunks      = n_cols / HMX_FP16_TILE_N_COLS;
  size_t n_chunks_per_task = hmx_ceil_div(n_tot_chunks, ctx->n_threads);

  permuted_weight_transfer_fp16_task_state_t state;
  state.n_tasks           = (n_tot_chunks + n_chunks_per_task - 1) / n_chunks_per_task;
  state.n_tot_chunks      = n_tot_chunks;
  state.n_chunks_per_task = n_chunks_per_task;
  state.k   = k;
  state.dst = vtcm_dst;
  state.src = src;

  worker_pool_run_func(ctx->worker_pool, transfer_permuted_weight_fp16_worker_loop, &state, ctx->n_threads);
}

// --- x4x2 format dequantizers ---

// Dequantize one x4x2 Q4_0 group (32 elements from 32 packed bytes) → 32 FP16 in first 64 bytes.
// In x4x2, sub-blocks 0..3 use lower nibbles, sub-blocks 4..7 use upper nibbles
// of the same 32 packed bytes.
static inline HVX_Vector dequantize_x4x2_q4_0_group_hvx(
    const uint8_t *packed_32, bool upper_nibbles,
    const __fp16 *scale, const HVX_Vector vlut_cvt) {
  HVX_Vector vq = hmx_vmemu(packed_32);
  HVX_Vector v_scales = Q6_Vh_vsplat_R(hmx_fp16_to_bits((__fp16 *)scale));
  HVX_Vector v_quants = upper_nibbles ? Q6_Vub_vlsr_VubR(vq, 4) : vq;
  HVX_VectorPair vp = Q6_Wh_vlut16_VbVhR_nomatch(v_quants, vlut_cvt, 0);
  // vlut16 set 0 uses even-halfword LUT entries (h[0],h[2],...,h[30]).
  // Output for input bytes 0..31 goes to vp.lo.h[0..31] — already contiguous.
  // No vshuff needed; the previous interleave with vp.hi was polluting odd
  // halfword positions with lookups from unrelated bytes 64..95.
  HVX_Vector v_hf = Q6_V_lo_W(vp);
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

  for (int t = start_tile; t < end_tile; ++t) {
    int ct = t / n_k_tiles;  // column tile index
    int kt = t % n_k_tiles;  // K tile index

    HVX_Vector *pv_out = (HVX_Vector *)(vtcm_dst + t * HMX_FP16_TILE_N_ELMS);

    if (is_q4) {
      int blk_idx  = (kt * 32) / HMX_QK_Q4x4x2;
      int sub_blk  = ((kt * 32) % HMX_QK_Q4x4x2) / 32;
      bool upper   = (sub_blk >= 4);
      int byte_off  = blk_idx * (HMX_QK_Q4x4x2 / 2) + (upper ? (sub_blk - 4) : sub_blk) * 32;
      int scale_off = qrow_size + blk_idx * HMX_X4X2_DBLK_SIZE + sub_blk * (int)sizeof(__fp16);

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

        *pv_out++ = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -2));
      }
    } else {
      // Q8_0
      int blk_idx  = (kt * 32) / HMX_QK_Q8x4x2;
      int sub_blk  = ((kt * 32) % HMX_QK_Q8x4x2) / 32;
      int byte_off  = blk_idx * HMX_QK_Q8x4x2 + sub_blk * 32;
      int scale_off = qrow_size + blk_idx * HMX_X4X2_DBLK_SIZE + sub_blk * (int)sizeof(__fp16);

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

        *pv_out++ = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -2));
      }
    }
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

      for (int k = 0; k < n_dot_tiles; k += 32) {
        int    offset  = k * HMX_FP16_TILE_N_ELMS;
        size_t n_tiles = hmx_smin(n_dot_tiles - k, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
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

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) ctx->vtcm_base;
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
  assert((size_t)(vtcm_ptr - ctx->vtcm_base) <= ctx->vtcm_scratch_size);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = hmx_align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = hmx_align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
    // transfer activation matrix chunk into VTCM
    size_t n_rows = hmx_smin(m - mr, m_chunk_n_rows);

    // int64_t act_t0 = HAP_perf_get_qtimer_count();
    {
      const float *activation_chunk = activation + mr * k;
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
    }
    // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

    // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = hmx_smin(n - nc, n_chunk_n_cols);

      // int64_t wei_t0 = HAP_perf_get_qtimer_count();
      {
        const __fp16 *permuted_weight_chunk = permuted_weight + nc * k;
        transfer_permuted_weight_chunk_fp16(ctx, vtcm_weight, permuted_weight_chunk, n_cols, k);
      }
      // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

      // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

      // int64_t core_t0 = HAP_perf_get_qtimer_count();
      {
        const int n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        const int n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32, ctx->vtcm_rctx);
      }
      // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

      // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

      // int64_t out_t0 = HAP_perf_get_qtimer_count();
      {
        float *output = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
      }
      // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

      // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
    }
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n * sizeof(__fp16);
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

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

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  const bool use_pipeline = (m >= 128) && (k <= n);
  // const bool use_pipeline = false;

  if (!use_pipeline) {
    // NOTE(hzx): In this simple implementation, load-matmul-store are executed sequentially
    // only DMA load and dequantization process are overlapped during the load stage

    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      // transfer activation matrix chunk into VTCM
      size_t n_rows = hmx_smin(m - mr, m_chunk_n_rows);

      // int64_t act_t0 = HAP_perf_get_qtimer_count();
      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }
      // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

      // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

      void *buf_curr = vtcm_scratch0;
      void *buf_next = vtcm_scratch1;

      // issue async DDR data transfer for the first weight chunk
      {
        const size_t n_cols_first            = hmx_smin(n, n_chunk_n_cols);
        const size_t first_weight_chunk_size = n_cols_first * row_stride;

        dma_queue_push(ctx->dma[0], dma_make_ptr(buf_curr, permuted_weight), first_weight_chunk_size, first_weight_chunk_size, first_weight_chunk_size, 1);
      }

      for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
        size_t n_cols = hmx_smin(n - nc, n_chunk_n_cols);

        // int64_t wei_t0 = HAP_perf_get_qtimer_count();
        {
          dma_queue_pop(ctx->dma[0]);  // wait until current weight chunk become ready

          const size_t nc_next = nc + n_chunk_n_cols;
          if (nc_next < n) {
            const size_t n_cols_next = hmx_smin(n - nc_next, n_chunk_n_cols);

            const size_t   next_weight_chunk_size = n_cols_next * row_stride;
            const uint8_t *next_weight_chunk      = permuted_weight + nc_next * row_stride;

            dma_queue_push(ctx->dma[0], dma_make_ptr(buf_next, next_weight_chunk), next_weight_chunk_size, next_weight_chunk_size, next_weight_chunk_size, 1);
          }

          dequantize_x4x2_weight_chunk_to_fp16_tiles(ctx, vtcm_weight, buf_curr,
                                                      n_cols, k, row_stride, weight_type);

          swap_ptr(&buf_curr, &buf_next);
        }
        // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

        // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

        // int64_t core_t0 = HAP_perf_get_qtimer_count();
        {
          const int n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          const int n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32, ctx->vtcm_rctx);
        }
        // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

        // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

        // int64_t out_t0 = HAP_perf_get_qtimer_count();
        {
          float *output = dst + (mr * n + nc);
          transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
        }
        // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

        // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
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

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

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

  int64_t t_a, t_b, t_c;
  t_a = t_b = t_c = 0;

  for (size_t mr = 0; mr < m; mr += M_BLOCK_SIZE) {
    size_t m_blk_sz = hmx_smin(m - mr, M_BLOCK_SIZE);
    for (size_t nc = 0; nc < n; nc += N_BLOCK_SIZE) {
      size_t n_blk_sz = hmx_smin(n - nc, N_BLOCK_SIZE);

      const int n_row_tiles = hmx_ceil_div(m_blk_sz, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = hmx_ceil_div(n_blk_sz, HMX_FP16_TILE_N_COLS);

      // TODO(hzx): fully pipelined loop
      for (size_t kk = 0; kk < k; kk += K_BLOCK_SIZE) {
        size_t k_blk_sz = hmx_smin(k - kk, K_BLOCK_SIZE);

        int64_t t0 = HAP_perf_get_qtimer_count();
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
        t_a += HAP_perf_get_qtimer_count() - t0;

        int64_t t1 = HAP_perf_get_qtimer_count();
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
        t_b += HAP_perf_get_qtimer_count() - t1;

        // core mma
        int64_t t2 = HAP_perf_get_qtimer_count();
        {
          core_mma_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, vtcm_eye_tile, n_row_tiles,
                              n_col_tiles, k_blk_sz / HMX_FP16_TILE_N_COLS, kk == 0, ctx->vtcm_rctx);
        }
        t_c += HAP_perf_get_qtimer_count() - t2;
      }

      // store output block
      {
        float *output_block = out + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output_block, vtcm_output, m_blk_sz, n_blk_sz, n);
      }
    }
  }

  FARF(ALWAYS, "t_a: %lld us, t_b: %lld us, t_c: %lld us", HAP_perf_qtimer_count_to_us(t_a),
       HAP_perf_qtimer_count_to_us(t_b), HAP_perf_qtimer_count_to_us(t_c));

  return 0;
}
