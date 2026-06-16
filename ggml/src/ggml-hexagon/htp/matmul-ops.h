#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#include <stdint.h>
#include <stddef.h>
#include "htp-ops.h"
#include "hex-fastdiv.h"
#include "hex-common.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- HMX Tile Constraints ---
#ifndef HMX_FP16_TILE_N_COLS
#define HMX_FP16_TILE_N_COLS 32
#endif
#ifndef HMX_FP16_TILE_N_ROWS
#define HMX_FP16_TILE_N_ROWS 32
#endif
#ifndef HMX_FP16_TILE_SIZE
#define HMX_FP16_TILE_SIZE   (32 * 32 * sizeof(__fp16)) // 2048 bytes
#endif
#ifndef HMX_FP16_TILE_N_ELMS
#define HMX_FP16_TILE_N_ELMS 1024
#endif

// Op-specific struct for precomputed matmul params
struct htp_matmul_kernel_params {
    int32_t  use_hmx;            // 1 = use HMX, 0 = use HVX
    int32_t  use_pipeline;       // 1 = pipelined execution, 0 = standard
    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  num_threads;        // Number of threads to spawn
    int32_t  tile_size;          // Weight tile size
    int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
    int32_t  src1_row_size;      // Row size for quantized activation
    int32_t  spad_size;          // Total required scratchpad size in VTCM

    // Precomputed division values
    struct fastdiv_values div_ne12_ne1;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_r2;
    struct fastdiv_values div_r3;
    struct fastdiv_values div_ne11;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_matmul_kernel_params) <= 96, "htp_matmul_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_matmul_kernel_params) <= 96, "htp_matmul_kernel_params is too large for kernel_params blob");
#endif

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
static inline int hmx_compute_chunks(size_t   vtcm_total,
                              size_t   overhead,
                              size_t   per_n_cost,
                              size_t   per_m_cost,
                              size_t   per_mn_cost,
                              int      m,
                              int      n,
                              size_t   m_block_cost,
                              size_t   n_block_cost,
                              size_t * m_chunk_out,
                              size_t * n_chunk_out,
                              size_t * total_out) {
    if (m <= 0 || n <= 0) return -1;
    if (vtcm_total <= overhead) return -1;
    if (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) return -1;

    const size_t usable = vtcm_total - overhead;

    size_t best_cost = SIZE_MAX;
    size_t best_mn   = 0;
    size_t best_m = 0, best_n = 0;

    const size_t n_max = hex_align_down((size_t)n, HMX_FP16_TILE_N_COLS);
    for (size_t nc = n_max; nc >= HMX_FP16_TILE_N_COLS; nc -= HMX_FP16_TILE_N_COLS) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        if (hmx_mul_overflow(nc, per_n_cost, &n_fixed)) continue;
        if (n_fixed >= usable) goto next_nc;

        if (hmx_mul_overflow(nc, per_mn_cost, &ncmn)) goto next_nc;
        if (hmx_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) goto next_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_down(mc, HMX_FP16_TILE_N_ROWS);
            mc = hex_smin(mc, (size_t)m);

            if (mc == 0) {
                goto next_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t cost    = mblocks * m_block_cost + nblocks * n_block_cost;
            size_t mn      = mc * nc;
            if (cost < best_cost || (cost == best_cost && mn > best_mn)) {
                best_cost = cost;
                best_mn   = mn;
                best_m    = mc;
                best_n    = nc;
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
    if (hmx_mul_overflow(best_m, best_n, &mn))     return -1;
    if (hmx_mul_overflow(mn, per_mn_cost, &t2))    return -1;
    if (hmx_add_overflow(t0, t1, &total))          return -1;
    if (hmx_add_overflow(total, t2, &total))       return -1;
    if (hmx_add_overflow(total, overhead, &total)) return -1;

    *m_chunk_out = best_m;
    *n_chunk_out = best_n;
    *total_out   = total;
    return 0;
}

// --- Weight Repacked Tile Sizes ---
#define HTP_WEIGHT_TILE_SIZE_Q4_0   576
#define HTP_WEIGHT_TILE_SIZE_Q4_1   640
#define HTP_WEIGHT_TILE_SIZE_Q8_0   1088
#define HTP_WEIGHT_TILE_SIZE_IQ4_NL 576
#define HTP_WEIGHT_TILE_SIZE_MXFP4  544

// --- Weight Repacked Aligned Tile Sizes ---
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_0   640
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_1   640
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_Q8_0   1152
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_IQ4_NL 640
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_MXFP4  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_ACT_TILE_SIZE_Q8_0      1152
#define HTP_ACT_TILE_SIZE_Q8_1      1280

// --- Tile Size Helpers ---
static inline uint32_t htp_get_weight_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_WEIGHT_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_WEIGHT_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_WEIGHT_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_WEIGHT_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

static inline uint32_t htp_get_weight_aligned_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

// --- Activation/Row Size Helpers ---
static inline size_t htp_q8_0_tiled_row_size(uint32_t ne) {
    const uint32_t nb_32 = (ne + 31) / 32;
    return nb_32 * HTP_ACT_TILE_SIZE_Q8_0;
}

static inline size_t htp_q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t nb_32 = (ne + 31) / 32;
    return nb_32 * HTP_ACT_TILE_SIZE_Q8_1;
}

static inline size_t htp_get_tiled_row_stride(int weight_type, int k) {
    int nb = (k + QK_Q4_0_TILED - 1) / QK_Q4_0_TILED;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q8_0:
        case HTP_TYPE_MXFP4:
            return (size_t) nb * htp_get_weight_tile_size(weight_type);
        case HTP_TYPE_F16:
            return (size_t) k * sizeof(__fp16);
        case HTP_TYPE_F32:
            return (size_t) k * sizeof(float);
        default:
            return 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
