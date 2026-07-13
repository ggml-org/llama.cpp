#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#include <stdint.h>
#include <stddef.h>
#include "htp-ops.h"
#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- HMX Tile Constraints ---
#define HTP_MM_HMX_TILE_N_COLS 32
#define HTP_MM_HMX_TILE_N_ROWS 32
#define HTP_MM_HMX_TILE_SIZE   (32 * 32 * sizeof(__fp16)) // 2048 bytes
#define HTP_MM_HMX_TILE_N_ELMS 1024
#define HTP_MM_HMX_MIN_NROWS   4

// --- Weight Repacked Tile Sizes ---
#define HTP_MM_WEIGHT_TILE_SIZE_Q4_0   576
#define HTP_MM_WEIGHT_TILE_SIZE_Q4_1   640
#define HTP_MM_WEIGHT_TILE_SIZE_Q8_0   1088
#define HTP_MM_WEIGHT_TILE_SIZE_IQ4_NL 576
#define HTP_MM_WEIGHT_TILE_SIZE_MXFP4  544

// --- Weight Repacked Aligned Tile Sizes ---
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0   640
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_1   640
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0   1152
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_IQ4_NL 640
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_MXFP4  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_MM_ACT_TILE_SIZE_Q8_0      1152
#define HTP_MM_ACT_TILE_SIZE_Q8_1      1280

#define HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#define HTP_MM_HMX_COST_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#define HTP_MM_HMX_COST_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#define HTP_MM_DMA_ACT_ROWS_PER_STEP 2
#define HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ROWS_PER_STEP)

enum htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MM_KERNEL_HMX_2D,
    HTP_MM_KERNEL_HMX_F16_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F16_F16_DDR,
    HTP_MM_KERNEL_HVX_F16_F32_DDR,

    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_DDR,
    HTP_MM_KERNEL_HVX_F32_F16_DDR,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_ROW,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_BLOCK,    // parallel block-wise quantization
    HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT, // row-wise fallback flat quantization
};

// Op-specific struct for precomputed matmul params
struct htp_mm_kernel_params {
    int32_t  kernel_type;        // enum htp_mm_kernel_type
    int32_t  pipeline;           // 1 = pipelined execution, 0 = standard
    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  n_threads;          // Number of threads to spawn
    int32_t  n_act_threads;      // Number of threads for activation preparation
    int32_t  n_hmx;              // 1 = use HMX, 0 = use HVX
    int32_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    int32_t  tile_size;          // Weight tile size
    int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
    int32_t  src1_row_size;      // Row size for quantized activation
    int32_t  vtcm_size;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_size;     // src0 scratchpad size in VTCM
    int32_t  vtcm_src1_size;     // src1 scratchpad size in VTCM
    int32_t  vtcm_src2_size;     // src2 scratchpad size in VTCM (fused only)
    int32_t  vtcm_src3_size;     // src3 scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_size;      // dst scratchpad size in VTCM

    // Precomputed division values
    struct fastdiv_values div_ne12_ne1;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_r2;
    struct fastdiv_values div_r3;
    struct fastdiv_values div_ne11;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#endif

struct mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
static inline int htp_mm_hmx_compute_chunks(size_t   vtcm_total,
                              size_t   overhead,
                              size_t   per_n_cost,
                              size_t   per_m_cost,
                              size_t   per_mn_cost,
                              size_t   m,
                              size_t   n,
                              size_t   m_block_cost,
                              size_t   n_block_cost,
                              size_t * m_chunk_out,
                              size_t * n_chunk_out,
                              size_t * total_out) {
    if (m == 0 || n == 0) return -1;
    if (vtcm_total <= overhead) return -1;
    if (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) return -1;

    const size_t usable = vtcm_total - overhead;

    size_t best_cost = SIZE_MAX;
    size_t best_mn   = 0;
    size_t best_m = 0, best_n = 0;

    const size_t n_max = hex_align_down((size_t)n, HTP_MM_HMX_TILE_N_COLS);
    for (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_COLS; nc -= HTP_MM_HMX_TILE_N_COLS) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        if (hex_mul_overflow(nc, per_n_cost, &n_fixed)) continue;
        if (n_fixed >= usable) goto next_nc;

        if (hex_mul_overflow(nc, per_mn_cost, &ncmn)) goto next_nc;
        if (hex_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) goto next_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_down(mc, HTP_MM_HMX_TILE_N_ROWS);
            mc = hex_smin(mc, m);

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
        if (nc == HTP_MM_HMX_TILE_N_COLS) break;  // avoid size_t underflow
    }

    if (best_m == 0 || best_n == 0) return -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, total = 0;
    if (hex_mul_overflow(best_n, per_n_cost, &t0)) return -1;
    if (hex_mul_overflow(best_m, per_m_cost, &t1)) return -1;
    if (hex_mul_overflow(best_m, best_n, &mn))     return -1;
    if (hex_mul_overflow(mn, per_mn_cost, &t2))    return -1;
    if (hex_add_overflow(t0, t1, &total))          return -1;
    if (hex_add_overflow(total, t2, &total))       return -1;
    if (hex_add_overflow(total, overhead, &total)) return -1;

    *m_chunk_out = best_m;
    *n_chunk_out = best_n;
    *total_out   = total;
    return 0;
}

// --- Tile Size Helpers ---
static inline uint32_t htp_mm_get_weight_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

static inline uint32_t htp_mm_get_weight_aligned_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

// --- Activation/Row Size Helpers ---
static inline size_t htp_mm_q8_0_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = ((ne + 127) / 128) * 128;
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_0;
}

static inline size_t htp_mm_q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = ((ne + 127) / 128) * 128;
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_1;
}

static inline size_t htp_mm_q8_0_flat_row_size(uint32_t ne) {
    const uint32_t quants_size = hex_align_up(ne, 128);
    const uint32_t num_scales = (ne + 31) / 32;
    const uint32_t scales_size = hex_align_up(num_scales * 2, 128);
    return quants_size + scales_size;
}

static inline size_t htp_mm_q8_1_flat_row_size(uint32_t ne) {
    const uint32_t quants_size = hex_align_up(ne, 128);
    const uint32_t num_scales = (ne + 31) / 32;
    const uint32_t scales_size = hex_align_up(num_scales * 4, 128);
    return quants_size + scales_size;
}

static inline size_t htp_mm_get_tiled_row_stride(int weight_type, uint32_t k) {
    uint32_t nb = (k + QK_Q4_0_TILED - 1) / QK_Q4_0_TILED;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q8_0:
        case HTP_TYPE_MXFP4:
            return (size_t) nb * htp_mm_get_weight_tile_size(weight_type);
        case HTP_TYPE_F16:
            return (size_t) k * sizeof(__fp16);
        case HTP_TYPE_F32:
            return (size_t) k * sizeof(float);
        default:
            return 0;
    }
}

static inline size_t htp_mm_round_up(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

static inline bool htp_mm_hmx_pipeline(uint32_t m) {
    return m > 32;
}

static inline void htp_mm_hmx_get_2d_chunk_costs(
    int wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    const bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32);
    const size_t row_stride = htp_mm_get_tiled_row_stride(wtype, k);
    const size_t vec_dot_size = k * sizeof(uint16_t);
    const uint32_t n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    *size_per_n_out = (pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) +
                      (pipeline ? 2 * vec_dot_size : vec_dot_size);
    *size_per_m_out = vec_dot_size;
    *size_per_mn_out = (pipeline ? 2 : 1) * sizeof(uint16_t);
}

static inline void htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t group_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    const size_t vec_dot_size = k * sizeof(uint16_t);
    *size_per_n_out = 3 * vec_dot_size;
    *size_per_m_out = group_size * vec_dot_size;
    *size_per_mn_out = sizeof(uint16_t);
}

struct htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_weight[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)

    // Cached sizes of regions for HMX kernel use
    size_t weight_area_bytes;
    size_t act_area_bytes;
    size_t act_f32_bytes;
    size_t output_area_bytes;
    size_t scratch_bytes[2];
    size_t act_head_stride;

    size_t total_bytes;
};

struct htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_src1;          // vtcm_src1 (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_src2;          // vtcm_src2 (Wq / fused only)
    size_t off_src3;          // vtcm_src3 (Wv / fused only)
    size_t off_dst;           // vtcm_dst (output scratch)

    // Cached sizes
    size_t src0_bytes;
    size_t src1_bytes;
    size_t src2_bytes;
    size_t src3_bytes;
    size_t dst_bytes;

    size_t total_bytes;
};

static inline void htp_mm_hmx_vtcm_layout_build(
    struct htp_mm_hmx_vtcm_layout * L,
    int kernel_type,
    int wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t group_size,
    bool use_dma_activation,
    bool pipeline,
    uint32_t act_threads,
    uint32_t aligned_tile_size
) {
    size_t off = 0;

    if (kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
        const size_t vec_dot_size     = k * sizeof(uint16_t);
        const size_t act_head_stride   = mc * k;
        const size_t weight_area_size  = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        const size_t activation_area_size = hex_align_up(group_size * act_head_stride * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
        const size_t output_area_size  = hex_align_up(group_size * mc * nc * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
        const size_t scratch_area_size = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        const size_t min_f32_size = use_dma_activation
            ? hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), 128) : 0;

        // Group A: Permanent activation tiles and scales
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, activation_area_size);
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_size, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_size, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch_area_size);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[1], scratch_area_size);

        const size_t group_b_size = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        const size_t max_f32_size = act_threads * 64 * k * sizeof(float);
        const size_t act_f32_size = use_dma_activation
            ? hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128) : 0;
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_f32, act_f32_size, use_dma_activation);

        const size_t group_c_size = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_size;
        L->act_area_bytes    = activation_area_size;
        L->act_f32_bytes     = act_f32_size;
        L->output_area_bytes = output_area_size;
        L->scratch_bytes[0]  = scratch_area_size;
        L->scratch_bytes[1]  = scratch_area_size;
        L->act_head_stride   = act_head_stride;

        off = off_group_a + hex_smax(group_b_size, group_c_size);
    } else {
        // HTP_MM_KERNEL_HMX_2D
        const bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32);
        const size_t row_stride = htp_mm_get_tiled_row_stride(wtype, k);
        const size_t vec_dot_size = k * sizeof(uint16_t);
        const uint32_t n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;

        const size_t min_f32_size = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), 128);
        const size_t weight_area_size = is_quant
            ? hex_align_up((nc / 32) * n_k_tiles * aligned_tile_size, HTP_MM_HMX_TILE_SIZE)
            : hex_align_up(nc * row_stride, HTP_MM_HMX_TILE_SIZE);
        const size_t act_area_size    = hex_align_up(mc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        const size_t output_area_size = hex_align_up(mc * nc * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);

        const size_t scratch0_size = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        const size_t scratch1_size = pipeline ? scratch0_size : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_size);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_size, pipeline);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_size);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch0_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_scratch[1], scratch0_size, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_size, pipeline);

        const size_t group_b_size = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        const size_t max_f32_size = act_threads * 64 * k * sizeof(float);
        const size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);

        const size_t group_c_size = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_size;
        L->act_area_bytes    = act_area_size;
        L->act_f32_bytes     = act_f32_size;
        L->output_area_bytes = output_area_size;
        L->scratch_bytes[0]  = scratch0_size;
        L->scratch_bytes[1]  = scratch1_size;
        L->act_head_stride   = 0;

        off = off_group_a + hex_smax(group_b_size, group_c_size);
    }

    L->total_bytes = off;
}

static inline void htp_mm_hvx_vtcm_layout_build(
    struct htp_mm_hvx_vtcm_layout * L,
    int kernel_type,
    int wtype,
    uint32_t ne10,       // k
    uint32_t src1_nrows, // m_total
    uint32_t n_threads,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t src1_row_size,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_qkv,
    bool is_fused_ffn
) {
    size_t src0_sz = 0;
    size_t src1_sz = 0;
    size_t src2_sz = 0;
    size_t src3_sz = 0;
    size_t dst_sz  = 0;

    const bool is_repack = (wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 ||
                            wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL ||
                            wtype == HTP_TYPE_MXFP4);

    if (is_fused_qkv || is_fused_ffn) {
        const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
        const size_t quant_scratch_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float)) * n_threads;

        size_t src0_sz_per_thread = 0;
        size_t src2_sz_per_thread = 0;
        size_t src3_sz_per_thread = 0;

        if (is_repack) {
            uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
            uint32_t n_k_tiles = hex_round_up(ne10, 32) / 32;
            uint32_t tile_row_size = n_k_tiles * aligned_tile_size;

            src0_sz_per_thread = hex_round_up(n_prefetch * tile_row_size, 128);
            src2_sz_per_thread = hex_round_up(n_prefetch * tile_row_size, 128);
            if (is_fused_qkv) {
                src3_sz_per_thread = hex_round_up(n_prefetch * tile_row_size, 128);
            }
        } else {
            src0_sz_per_thread = hex_round_up(n_prefetch * src0_row_size_padded, 128);
            src2_sz_per_thread = hex_round_up(n_prefetch * src0_row_size_padded, 128);
            if (is_fused_qkv) {
                src3_sz_per_thread = hex_round_up(n_prefetch * src0_row_size_padded, 128);
            }
        }

        size_t flat_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
        size_t tiled_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

        if (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
            src1_sz = hex_round_up(flat_src1_row_size * src1_nrows, 128);
        } else {
            src1_sz = hex_round_up(tiled_src1_row_size * src1_nrows, 128);
        }

        src0_sz = src0_sz_per_thread * n_threads;
        src2_sz = src2_sz_per_thread * n_threads;
        src3_sz = src3_sz_per_thread * n_threads;
        dst_sz  = quant_scratch_size;
    } else if (is_matmul_id) {
        const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
        const size_t src1_row_size_tiled = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10)
                                                                    : htp_mm_q8_0_tiled_row_size(ne10);

        size_t src0_sz_per_thread = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
        src1_sz                   = htp_mm_round_up(src1_row_size_tiled * src1_nrows, 256);

        if (is_repack) {
            const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
            const uint32_t n_k_tiles         = ne10 / 32;
            const uint32_t tile_row_size     = n_k_tiles * aligned_tile_size;
            size_t repacked_vtcm_size        = htp_mm_round_up(n_prefetch * tile_row_size, 256);
            src0_sz_per_thread               = repacked_vtcm_size;
        }

        src0_sz = src0_sz_per_thread * n_threads;
        dst_sz  = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float)) * n_threads;
    } else {
        const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
        const size_t dst_nrows = (src1_nrows > 1) ? 0 : 1;

        switch (kernel_type) {
            case HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_src1_row_size = htp_mm_round_up(ne10 * 2, 128);
                src1_sz = htp_mm_round_up(f16_src1_row_size * src1_nrows, 256);
                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_F16_F32_DDR:
            case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            case HTP_MM_KERNEL_HVX_F32_F16_DDR: {
                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size, 256) * n_threads;
                src1_sz = htp_mm_round_up(n_prefetch * src1_row_size, 256) * n_threads;
                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_F32_F32_VTCM: {
                size_t f32_src1_row_size = htp_mm_round_up(ne10 * 4, 128);
                src1_sz = htp_mm_round_up(f32_src1_row_size * src1_nrows, 256);
                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
            case HTP_MM_KERNEL_HVX_QUANT_ROW: {
                size_t q_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
                src1_sz = htp_mm_round_up(q_src1_row_size * src1_nrows, 256);

                src0_sz = src0_sz * n_threads;

                if (is_repack) {
                    uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
                    uint32_t n_k_tiles = ne10 / 32;
                    uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
                    size_t repacked_vtcm_size = htp_mm_round_up(n_prefetch * tile_row_size, 256);
                    src0_sz = repacked_vtcm_size * n_threads;
                }

                size_t quant_scratch_size_per_thread = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
                size_t dst_size_per_thread = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) : 0;
                if (dst_size_per_thread < quant_scratch_size_per_thread) {
                    dst_size_per_thread = quant_scratch_size_per_thread;
                }
                dst_sz = dst_size_per_thread * n_threads;
                break;
            }
            case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT: {
                size_t q_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
                src1_sz = htp_mm_round_up(q_src1_row_size * src1_nrows, 256);

                src0_sz = src0_sz * n_threads;

                if (is_repack) {
                    uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
                    uint32_t n_k_tiles = ne10 / 32;
                    uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
                    size_t repacked_vtcm_size = htp_mm_round_up(n_prefetch * tile_row_size, 256);
                    src0_sz = repacked_vtcm_size * n_threads;
                }

                size_t quant_scratch_size_per_thread = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
                size_t dst_size_per_thread = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) : 0;
                if (dst_size_per_thread < quant_scratch_size_per_thread) {
                    dst_size_per_thread = quant_scratch_size_per_thread;
                }
                dst_sz = dst_size_per_thread * n_threads;
                break;
            }
            default:
                break;
        }
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_sz);
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_sz);
    VTCM_LAYOUT_ALLOC(off, off_src3, src3_sz);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_sz);

    L->src0_bytes = src0_sz;
    L->src1_bytes = src1_sz;
    L->src2_bytes = src2_sz;
    L->src3_bytes = src3_sz;
    L->dst_bytes  = dst_sz;
    L->total_bytes = off;
}

static inline size_t htp_mm_hmx_get_2d_vtcm_size(
    int wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_size
) {
    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, false, pipeline, act_threads, aligned_tile_size);
    return L.total_bytes;
}

static inline size_t htp_mm_hmx_get_batched_vtcm_size(
    int wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool use_dma_activation, bool pipeline, uint32_t act_threads) {
    (void)pipeline;
    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, wtype, k, mc, nc, group_size, use_dma_activation, false, act_threads, 0);
    return L.total_bytes;
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
