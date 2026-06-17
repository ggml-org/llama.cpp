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

#ifndef MM_SPAD_SRC0_NROWS
#define MM_SPAD_SRC0_NROWS 16
#endif
#ifndef MM_SPAD_SRC1_NROWS
#define MM_SPAD_SRC1_NROWS 16
#endif
#ifndef MM_SPAD_DST_NROWS
#define MM_SPAD_DST_NROWS  2
#endif

enum htp_matmul_kernel_type {
    HTP_MATMUL_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MATMUL_KERNEL_HMX_2D,
    HTP_MATMUL_KERNEL_HMX_F16_BATCHED,

    // HVX floating-point paths
    HTP_MATMUL_KERNEL_HVX_F16_F16_VTCM,
    HTP_MATMUL_KERNEL_HVX_F16_F16_DDR,
    HTP_MATMUL_KERNEL_HVX_F16_F32_DDR,

    HTP_MATMUL_KERNEL_HVX_F32_F32_VTCM,
    HTP_MATMUL_KERNEL_HVX_F32_F32_DDR,
    HTP_MATMUL_KERNEL_HVX_F32_F16_DDR,

    // HVX quantized paths
    HTP_MATMUL_KERNEL_HVX_QUANT_ROW,   // standard row-wise parallel quantization
    HTP_MATMUL_KERNEL_HVX_QUANT_BLOCK, // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
struct htp_matmul_kernel_params {
    int32_t  kernel_type;        // enum htp_matmul_kernel_type
    int32_t  n_hmx;              // 1 = use HMX, 0 = use HVX
    int32_t  use_pipeline;       // 1 = pipelined execution, 0 = standard
    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  num_threads;        // Number of threads to spawn
    int32_t  tile_size;          // Weight tile size
    int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
    int32_t  src1_row_size;      // Row size for quantized activation
    int32_t  spad_size;          // Total required scratchpad size in VTCM
    int32_t  src0_spad_size;     // src0 scratchpad size in VTCM
    int32_t  src1_spad_size;     // src1 scratchpad size in VTCM
    int32_t  dst_spad_size;      // dst scratchpad size in VTCM
    int32_t  act_threads;        // Number of threads for activation preparation

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
    const uint32_t ne_padded = ((ne + 127) / 128) * 128;
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_ACT_TILE_SIZE_Q8_0;
}

static inline size_t htp_q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = ((ne + 127) / 128) * 128;
    const uint32_t nb_32 = ne_padded / 32;
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

#ifndef DMA_DEPTH
#define DMA_DEPTH 2
#endif

static inline size_t hex_round_up_matmul(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

static inline size_t hex_align_up_matmul(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

static inline bool hmx_use_pipeline(int m) {
    return m > 32;
}

static inline void hmx_get_2d_chunk_costs(
    int wtype, int k, bool use_pipeline, int aligned_tile_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    const bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32);
    const size_t row_stride = htp_get_tiled_row_stride(wtype, k);
    const size_t vec_dot_size = k * sizeof(uint16_t);
    const int n_k_tiles = k / HMX_FP16_TILE_N_COLS;
    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    *size_per_n_out = (use_pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) + 
                      (use_pipeline ? 2 * vec_dot_size : vec_dot_size);
    *size_per_m_out = vec_dot_size;
    *size_per_mn_out = (use_pipeline ? 2 : 1) * sizeof(uint16_t);
}

static inline void hmx_get_batched_chunk_costs(
    int k, int group_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    const size_t vec_dot_size = k * sizeof(uint16_t);
    *size_per_n_out = 3 * vec_dot_size;
    *size_per_m_out = group_size * vec_dot_size;
    *size_per_mn_out = sizeof(uint16_t);
}

static inline size_t hmx_get_2d_vtcm_size(
    int wtype, int k, size_t mc, size_t nc, bool use_pipeline, int act_threads, int aligned_tile_size
) {
    const int n_k_tiles = k / HMX_FP16_TILE_N_COLS;
    const bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32);
    const size_t row_stride = htp_get_tiled_row_stride(wtype, k);
    const size_t vec_dot_size = k * sizeof(uint16_t);

    const size_t act_f32_size = hex_align_up_matmul(act_threads * 4 * k * sizeof(float), HMX_FP16_TILE_SIZE);
    size_t weight_area_size = is_quant
        ? hex_align_up_matmul((nc / 32) * n_k_tiles * aligned_tile_size, HMX_FP16_TILE_SIZE)
        : hex_align_up_matmul(nc * row_stride, HMX_FP16_TILE_SIZE);
    if (use_pipeline) {
        weight_area_size *= 2;
    }
    const size_t act_area_size    = hex_align_up_matmul(mc * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size = hex_align_up_matmul(mc * nc * sizeof(uint16_t), HMX_FP16_TILE_SIZE);

    size_t scratch0_size = hex_align_up_matmul(nc * vec_dot_size, HMX_FP16_TILE_SIZE);
    size_t scratch1_size = use_pipeline ? scratch0_size : 0;
    size_t scratch2_size = use_pipeline ? output_area_size : 0;

    return weight_area_size + act_area_size + act_f32_size + output_area_size + 
           scratch0_size + scratch1_size + scratch2_size + 256;
}

static inline size_t hmx_get_batched_vtcm_size(
    int wtype, int k, size_t mc, size_t nc, int group_size, bool use_dma_activation, bool use_pipeline, int act_threads
) {
    (void)wtype;
    (void)use_pipeline;
    const size_t vec_dot_size = k * sizeof(uint16_t);
    const size_t f32_scratch_size = use_dma_activation
        ? hex_align_up_matmul(act_threads * 4 * k * sizeof(float), HMX_FP16_TILE_SIZE) : 0;

    const size_t act_head_stride  = mc * k;
    const size_t weight_area_size = hex_align_up_matmul(nc * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t act_area_size    = hex_align_up_matmul(group_size * act_head_stride * sizeof(uint16_t), HMX_FP16_TILE_SIZE);
    const size_t output_area_size = hex_align_up_matmul(group_size * mc * nc * sizeof(uint16_t), HMX_FP16_TILE_SIZE);
    const size_t scratch_area_size = hex_align_up_matmul(nc * vec_dot_size, HMX_FP16_TILE_SIZE);

    return weight_area_size + act_area_size + output_area_size + 
           2 * scratch_area_size + 256 + f32_scratch_size;
}

static inline size_t hvx_get_spad_sizes(
    int kernel_type,
    int wtype,
    int ne10, // k
    int src1_nrows, // m_total (or act_nrows)
    int n_threads,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t src1_row_size,
    size_t * src0_spad_size_out,
    size_t * src1_spad_size_out,
    size_t * dst_spad_size_out
) {
    size_t src0_spad_size = 0;
    size_t src1_spad_size = 0;
    size_t dst_spad_size  = 0;

    const bool is_repack = (wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 ||
                            wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL ||
                            wtype == HTP_TYPE_MXFP4);

    const size_t src0_row_size_padded = hex_round_up_matmul(src0_row_size, 128);

    switch (kernel_type) {
        case HTP_MATMUL_KERNEL_HVX_F16_F16_VTCM: {
            size_t f16_src1_row_size = hex_round_up_matmul(ne10 * 2, 128);
            src1_spad_size = hex_round_up_matmul(f16_src1_row_size * src1_nrows, 256);
            src0_spad_size = hex_round_up_matmul(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256) * n_threads;
            dst_spad_size  = hex_round_up_matmul(MM_SPAD_DST_NROWS * dst_row_size, 256) * n_threads;
            break;
        }
        case HTP_MATMUL_KERNEL_HVX_F16_F32_DDR:
        case HTP_MATMUL_KERNEL_HVX_F16_F16_DDR:
        case HTP_MATMUL_KERNEL_HVX_F32_F32_DDR:
        case HTP_MATMUL_KERNEL_HVX_F32_F16_DDR: {
            src0_spad_size = hex_round_up_matmul(MM_SPAD_SRC0_NROWS * src0_row_size, 256) * n_threads;
            src1_spad_size = hex_round_up_matmul(MM_SPAD_SRC1_NROWS * src1_row_size, 256) * n_threads;
            dst_spad_size  = hex_round_up_matmul(MM_SPAD_DST_NROWS * dst_row_size, 256) * n_threads;
            break;
        }
        case HTP_MATMUL_KERNEL_HVX_F32_F32_VTCM: {
            size_t f32_src1_row_size = hex_round_up_matmul(ne10 * 4, 128);
            src1_spad_size = hex_round_up_matmul(f32_src1_row_size * src1_nrows, 256);
            src0_spad_size = hex_round_up_matmul(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256) * n_threads;
            dst_spad_size  = hex_round_up_matmul(MM_SPAD_DST_NROWS * dst_row_size, 256) * n_threads;
            break;
        }
        case HTP_MATMUL_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MATMUL_KERNEL_HVX_QUANT_ROW: {
            size_t q_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_q8_1_tiled_row_size(ne10) : htp_q8_0_tiled_row_size(ne10);
            
            dst_spad_size  = hex_round_up_matmul(MM_SPAD_DST_NROWS * dst_row_size, 256);
            src0_spad_size = hex_round_up_matmul(MM_SPAD_SRC0_NROWS * src0_row_size_padded, 256);
            src1_spad_size = hex_round_up_matmul(q_src1_row_size * src1_nrows, 256);

            // src0 spad is also used in dynamic quantizer to store padded src1 rows
            size_t src1_row_size_padded = hex_round_up_matmul(q_src1_row_size, QK_Q8_0_TILED * sizeof(float));
            if (src0_spad_size < src1_row_size_padded) {
                src0_spad_size = src1_row_size_padded;
            }

            src0_spad_size = src0_spad_size * n_threads;
            dst_spad_size  = dst_spad_size * n_threads;

            if (is_repack) {
                uint32_t aligned_tile_size = htp_get_weight_aligned_tile_size(wtype);
                uint32_t n_k_tiles = ne10 / 32;
                uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
                size_t repacked_spad_size = hex_round_up_matmul(DMA_DEPTH * tile_row_size, 256);
                if (repacked_spad_size < src1_row_size_padded) {
                    repacked_spad_size = src1_row_size_padded;
                }
                src0_spad_size = repacked_spad_size * n_threads;
            }
            break;
        }
        default:
            break;
    }

    *src0_spad_size_out = src0_spad_size;
    *src1_spad_size_out = src1_spad_size;
    *dst_spad_size_out  = dst_spad_size;

    return src0_spad_size + src1_spad_size + dst_spad_size;
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
