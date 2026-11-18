#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#include <stdint.h>
#include <stddef.h>
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- HMX Tile Constraints ---
#define HTP_MM_HMX_TILE_N_COLS        32
#define HTP_MM_HMX_TILE_N_ROWS        32
#define HTP_MM_HMX_TILE_SIZE          (32 * 32 * sizeof(__fp16)) // 2048 bytes
#define HTP_MM_HMX_TILE_N_ELMS        1024
#define HTP_MM_HMX_MIN_NROWS          4
#define HTP_MM_HMX_WEIGHT_B_TILE_SIZE 1024
#define HTP_MM_HMX_BIAS_BANK_SIZE     256

#define HTP_MM_WEIGHT_TILE_K_Q2_0             64
#define HTP_MM_WEIGHT_TILE_SIZE_Q2_0          HTP_TENSOR_TILE_SIZE_Q2_0
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_MM_ACT_BLOCK_K             128
#define HTP_MM_ACT_TILE_SIZE_Q8_0      1152
#define HTP_MM_ACT_TILE_SIZE_Q8_1      1280
#define HTP_MM_ACT_TILE_SIZE_F16       2048

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
    HTP_MM_KERNEL_HMX_BATCHED,
    HTP_MM_KERNEL_HMX_W8A8_2D,
    HTP_MM_KERNEL_HMX_W8A8_BATCHED,
    HTP_MM_KERNEL_HMX_W8A16_2D,
    HTP_MM_KERNEL_HMX_W8A16_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_BF16_F32_VTCM,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_ROW,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_BLOCK,    // parallel block-wise quantization
};

static inline bool htp_mm_kernel_is_hmx_w8a8(int32_t kt) {
    return kt == HTP_MM_KERNEL_HMX_W8A8_2D || kt == HTP_MM_KERNEL_HMX_W8A8_BATCHED;
}

static inline bool htp_mm_kernel_is_hmx_w8a16(int32_t kt) {
    return kt == HTP_MM_KERNEL_HMX_W8A16_2D || kt == HTP_MM_KERNEL_HMX_W8A16_BATCHED;
}

static inline bool htp_mm_kernel_is_hmx_int(int32_t kt) {
    return htp_mm_kernel_is_hmx_w8a8(kt) || htp_mm_kernel_is_hmx_w8a16(kt);
}

static inline bool htp_mm_kernel_is_hmx_2d(int32_t kt) {
    return kt == HTP_MM_KERNEL_HMX_2D || kt == HTP_MM_KERNEL_HMX_W8A8_2D || kt == HTP_MM_KERNEL_HMX_W8A16_2D;
}

static inline bool htp_mm_kernel_is_hmx_batched(int32_t kt) {
    return kt == HTP_MM_KERNEL_HMX_BATCHED || kt == HTP_MM_KERNEL_HMX_W8A8_BATCHED || kt == HTP_MM_KERNEL_HMX_W8A16_BATCHED;
}

static inline bool htp_is_int_hmx_weight_type(int wtype) {
    return wtype == HTP_TYPE_Q4_0    || wtype == HTP_TYPE_Q8_0  ||
           wtype == HTP_TYPE_IQ4_NL  || wtype == HTP_TYPE_MXFP4 ||
           wtype == HTP_TYPE_Q5_0    || wtype == HTP_TYPE_Q1_0  ||
           wtype == HTP_TYPE_Q2_0    || wtype == HTP_TYPE_NVFP4 ||
           wtype == HTP_TYPE_Q3_K    || wtype == HTP_TYPE_Q6_K  ||
           wtype == HTP_TYPE_IQ2_XXS || wtype == HTP_TYPE_Q4_1  ||
           wtype == HTP_TYPE_Q4_K    || wtype == HTP_TYPE_Q5_1  ||
           wtype == HTP_TYPE_Q5_K    || wtype == HTP_TYPE_Q2_K;
}

// Op-specific struct for precomputed matmul params
struct htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX
    uint8_t  n_hvx;              // Number of HVX threads

    uint8_t  n_act_hvx;          // Number of HVX for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX
    uint8_t  reserved0;

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_size;          // Weight tile size
    int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
    int32_t  act_row_size;       // Row size for activation scratchpad
    int32_t  vtcm_size;          // Total required scratchpad size in VTCM
    int32_t  vtcm_bias_size;     // bias scratchpad size in VTCM (fused only)
    int32_t  hmx_kgroup;         // K superblock size for INT HMX kernels (0 for FP16)
    int32_t  hmx_n_groups;       // Number of K superblocks for INT HMX kernels (1 for FP16)

    // Precomputed division values
    struct fastdiv_values div_ne12_ne1;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_r2;
    struct fastdiv_values div_r3;
    struct fastdiv_values div_ne12;
    struct fastdiv_values div_n_act_hvx;
    struct fastdiv_values div_ne00_padded;
    struct fastdiv_values div_hmx_n_groups;
    struct fastdiv_values div_n_k_tiles;
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
                              size_t   m_align,
                              size_t   m_block_cost,
                              size_t   n_block_cost,
                              size_t * m_chunk_out,
                              size_t * n_chunk_out,
                              size_t * total_out) {
    if (m == 0 || n == 0 || m_align == 0) return -1;
    if (vtcm_total <= overhead) return -1;
    if (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) return -1;

    const size_t usable = vtcm_total - overhead;

    size_t best_cost = SIZE_MAX;
    size_t best_tail_waste = SIZE_MAX;
    size_t best_mn   = 0;
    size_t best_m = 0, best_n = 0;

    const size_t max_nc_budget = (usable / per_n_cost);
    const size_t n_max = hex_align_down(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_COLS);
    for (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_COLS; nc -= HTP_MM_HMX_TILE_N_COLS) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        if (hex_mul_overflow(nc, per_n_cost, &n_fixed)) continue;
        if (n_fixed >= usable) goto next_nc;

        if (hex_mul_overflow(nc, per_mn_cost, &ncmn)) goto next_nc;
        if (hex_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) goto next_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_down(mc, m_align);
            mc = hex_smin(mc, m);

            if (mc == 0) {
                goto next_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t cost    = mblocks * m_block_cost + nblocks * n_block_cost;
            size_t rem     = n % nc;
            size_t tail_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            if (cost < best_cost ||
                (cost == best_cost && tail_waste < best_tail_waste) ||
                (cost == best_cost && tail_waste == best_tail_waste && mn > best_mn)) {
                best_cost       = cost;
                best_tail_waste = tail_waste;
                best_mn         = mn;
                best_m          = mc;
                best_n          = nc;
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

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
static inline bool htp_mm_weight_has_offset(int weight_type) {
    return weight_type == HTP_TYPE_Q4_1 || weight_type == HTP_TYPE_Q4_K || weight_type == HTP_TYPE_Q5_K ||
           weight_type == HTP_TYPE_Q2_K || weight_type == HTP_TYPE_Q2_0 || weight_type == HTP_TYPE_Q5_1;
}

// --- Activation/Row Size Helpers ---
static inline size_t htp_mm_q8_0_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = hex_round_up(ne, HTP_MM_ACT_BLOCK_K);
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_0;
}

static inline size_t htp_mm_q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = hex_round_up(ne, HTP_MM_ACT_BLOCK_K);
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_1;
}

static inline size_t htp_mm_f16_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = hex_round_up(ne, HTP_MM_ACT_BLOCK_K);
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_F16;
}

static inline size_t htp_mm_tiled_act_row_size(int weight_type, uint32_t ne) {
    if (weight_type == HTP_TYPE_F16) {
        return htp_mm_f16_tiled_row_size(ne);
    }
    return htp_mm_weight_has_offset(weight_type) ? htp_mm_q8_1_tiled_row_size(ne) : htp_mm_q8_0_tiled_row_size(ne);
}

static inline size_t htp_mm_act_raw_row_size(uint32_t ne, size_t act_elem_size) {
    return hex_round_up(ne, HTP_MM_ACT_BLOCK_K) * act_elem_size;
}

static inline size_t htp_mm_round_up(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

static inline bool htp_mm_hmx_pipeline(uint32_t m) {
    return m > 32;
}

static inline void htp_mm_hmx_get_2d_chunk_costs_ex(
    int kernel_type, int wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_size, uint32_t k_group,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    const bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    const size_t row_stride = htp_tensor_get_row_size(wtype, k);
    const uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);
    const size_t qweight_row_stride = is_quant ? (size_t)(n_weight_tiles * aligned_tile_size) / 32 : 0;

    if (htp_mm_kernel_is_hmx_int(kernel_type)) {
        const bool is_w8a8 = htp_mm_kernel_is_hmx_w8a8(kernel_type);
        const uint32_t eff_kgroup = (k_group > 0 && k_group < k) ? k_group : k;
        const size_t n_groups = (k + eff_kgroup - 1) / eff_kgroup;
        const size_t w_unpacked_per_n = (size_t) k * sizeof(uint8_t) +
                                        n_groups * (is_w8a8 ? 20 : 12);
        const size_t act_elem_bytes = is_w8a8 ? sizeof(uint8_t) : sizeof(uint16_t);

        *size_per_n_out  = (pipeline ? 2 : 1) * ((is_quant ? qweight_row_stride : row_stride) + w_unpacked_per_n);
        *size_per_m_out  = (size_t) k * act_elem_bytes + n_groups * sizeof(float);
        *size_per_mn_out = (pipeline ? 2 : 1) * n_groups * sizeof(uint16_t);
        return;
    }

    const size_t vec_dot_size = k * sizeof(uint16_t);
    *size_per_n_out = (pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) +
                      (pipeline ? 2 * vec_dot_size : vec_dot_size);
    *size_per_m_out = vec_dot_size;
    *size_per_mn_out = (pipeline ? 2 : 1) * sizeof(uint16_t);
}

static inline void htp_mm_hmx_get_2d_chunk_costs(
    int wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    htp_mm_hmx_get_2d_chunk_costs_ex(HTP_MM_KERNEL_HMX_2D, wtype, k, pipeline, aligned_tile_size, 0,
                                     size_per_n_out, size_per_m_out, size_per_mn_out);
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

static inline size_t htp_mm_hmx_lut_bytes(int wtype) {
    return (wtype == HTP_TYPE_IQ2_XXS) ? hex_align_up(HTP_IQ2XXS_LUT_BYTES, HTP_MM_HMX_TILE_SIZE) : 0;
}

static inline size_t htp_mm_hmx_get_2d_overhead(int wtype, bool pipeline, bool is_matmul_id) {
    size_t num_regions = pipeline ? 8 : (is_matmul_id ? 4 : 6);
    return num_regions * HTP_MM_HMX_TILE_SIZE + 256 + htp_mm_hmx_lut_bytes(wtype);
}

static inline size_t htp_mm_hmx_get_batched_overhead(void) {
    return 5 * HTP_MM_HMX_TILE_SIZE + 256;
}

struct htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_weight[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_scales;    // INT HMX activation group scales d_a
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM
    size_t off_lut;           // IQ2_XXS grid + sign tables (2D only)

    // Cached sizes of regions for HMX kernel use
    size_t weight_area_bytes;
    size_t act_area_bytes;
    size_t act_scales_bytes;
    size_t act_f32_bytes;
    size_t output_area_bytes;
    size_t scratch_bytes[2];
    size_t weight_tiles_bytes;
    size_t weight_bias_offset;
    size_t weight_scales_offset;
    size_t act_head_stride;
    size_t bias_bytes;
    size_t lut_bytes;

    size_t total_bytes;
};

struct htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)
    size_t off_lut;           // vtcm_lut (IQ2_XXS grid + sign tables)

    // Cached sizes
    size_t src0_bytes;
    size_t act_bytes;
    size_t bias_bytes;
    size_t dst_bytes;
    size_t act_raw_bytes;
    size_t lut_bytes;

    size_t total_bytes;
};

static inline void htp_mm_hmx_vtcm_layout_build_ex(
    struct htp_mm_hmx_vtcm_layout * L,
    int kernel_type,
    int wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t group_size,
    bool pipeline,
    uint32_t act_hvx,
    uint32_t aligned_tile_size,
    size_t bias_size,
    uint32_t k_group
) {
    size_t off = 0;
    L->lut_bytes            = 0;
    L->off_lut              = 0;
    L->off_act_scales       = 0;
    L->act_scales_bytes     = 0;
    L->weight_tiles_bytes   = 0;
    L->weight_bias_offset   = 0;
    L->weight_scales_offset = 0;

    if (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        const size_t vec_dot_size     = k * sizeof(uint16_t);
        const size_t act_head_stride   = mc * k;
        const size_t weight_area_size  = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        const size_t activation_area_size = hex_align_up(group_size * act_head_stride * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
        const size_t output_area_size  = hex_align_up(group_size * mc * nc * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
        const size_t scratch_area_size = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        const size_t min_f32_size = hex_align_up(act_hvx * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, activation_area_size);
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE), bias_size > 0);

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
        const size_t max_f32_size = act_hvx * 64 * k * sizeof(float);
        const size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);

        const size_t group_c_size = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_size;
        L->act_area_bytes    = activation_area_size;
        L->act_f32_bytes     = act_f32_size;
        L->output_area_bytes = output_area_size;
        L->scratch_bytes[0]  = scratch_area_size;
        L->scratch_bytes[1]  = scratch_area_size;
        L->act_head_stride   = act_head_stride;
        L->bias_bytes        = bias_size;

        off = off_group_a + hex_smax(group_b_size, group_c_size);
    } else {
        // HTP_MM_KERNEL_HMX_2D / W8A8 / W8A16
        const bool is_int_hmx = htp_mm_kernel_is_hmx_int(kernel_type);
        const bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        const size_t row_stride = htp_tensor_get_row_size(wtype, k);
        const uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);

        const uint32_t dma_act_mult = htp_mm_kernel_is_hmx_w8a8(kernel_type) ? 8u : HTP_MM_DMA_ACT_MULTIPLIER;
        const size_t min_f32_size = hex_align_up((size_t) act_hvx * dma_act_mult * k * sizeof(float), 128);
        const size_t weight_area_size = is_quant
            ? hex_align_up((nc / 32) * n_weight_tiles * aligned_tile_size, HTP_MM_HMX_TILE_SIZE)
            : hex_align_up(nc * row_stride, HTP_MM_HMX_TILE_SIZE);

        size_t act_area_size    = 0;
        size_t act_scales_size  = 0;
        size_t output_area_size = 0;
        size_t scratch0_size    = 0;

        if (is_int_hmx) {
            const bool is_w8a8        = htp_mm_kernel_is_hmx_w8a8(kernel_type);
            const uint32_t eff_kgroup = (k_group > 0 && k_group < k) ? k_group : k;
            const size_t n_groups     = (k + eff_kgroup - 1) / eff_kgroup;
            const size_t mc_pad       = hex_align_up(mc, is_w8a8 ? 64 : 32);
            const size_t nc_pad       = hex_align_up(nc, 32);
            const size_t act_elem_sz  = is_w8a8 ? sizeof(uint8_t) : sizeof(uint16_t);

            act_area_size    = hex_align_up(mc_pad * k * act_elem_sz, HTP_MM_HMX_TILE_SIZE);
            act_scales_size  = hex_align_up(n_groups * mc_pad * sizeof(float), HTP_MM_HMX_TILE_SIZE);
            output_area_size = hex_align_up(n_groups * mc_pad * nc_pad * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);

            const size_t w_tiles_bytes  = hex_align_up(nc_pad * k * sizeof(uint8_t), 256);
            const size_t w_bias_bytes   = hex_align_up((nc_pad / 32) * n_groups * (is_w8a8 ? 512 : 256), 256);
            const size_t w_scales_bytes = hex_align_up(nc_pad * n_groups * sizeof(float), 256);

            L->weight_tiles_bytes   = w_tiles_bytes;
            L->weight_bias_offset   = w_tiles_bytes;
            L->weight_scales_offset = w_tiles_bytes + w_bias_bytes;
            scratch0_size           = hex_align_up(w_tiles_bytes + w_bias_bytes + w_scales_bytes, HTP_MM_HMX_TILE_SIZE);
        } else {
            const size_t vec_dot_size = k * sizeof(uint16_t);
            act_area_size    = hex_align_up(mc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
            output_area_size = hex_align_up(mc * nc * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);
            scratch0_size    = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        }

        const size_t scratch1_size = pipeline ? scratch0_size : 0;

        // Group A: Scales and activation tiles (must not overlap with Group B or C)
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE, !is_int_hmx);
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_act_scales, act_scales_size, act_scales_size > 0);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE), bias_size > 0);

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
        const size_t max_f32_size = act_hvx * 64 * k * sizeof(float);
        const size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);

        const size_t group_c_size = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_size;
        L->act_area_bytes    = act_area_size;
        L->act_scales_bytes  = act_scales_size;
        L->act_f32_bytes     = act_f32_size;
        L->output_area_bytes = output_area_size;
        L->scratch_bytes[0]  = scratch0_size;
        L->scratch_bytes[1]  = scratch1_size;
        L->act_head_stride   = 0;
        L->bias_bytes        = bias_size;

        off = off_group_a + hex_smax(group_b_size, group_c_size);

        L->lut_bytes = htp_mm_hmx_lut_bytes(wtype);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_lut, L->lut_bytes, L->lut_bytes > 0);
    }

    L->total_bytes = off;
}

static inline void htp_mm_hmx_vtcm_layout_build(
    struct htp_mm_hmx_vtcm_layout * L,
    int kernel_type,
    int wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t group_size,
    bool pipeline,
    uint32_t act_hvx,
    uint32_t aligned_tile_size,
    size_t bias_size
) {
    htp_mm_hmx_vtcm_layout_build_ex(L, kernel_type, wtype, k, mc, nc, group_size, pipeline, act_hvx, aligned_tile_size, bias_size, 0);
}

static inline void htp_mm_hvx_vtcm_layout_build(
    struct htp_mm_hvx_vtcm_layout * L,
    int kernel_type,
    int wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_hvx,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t act_elem_size,
    size_t bias_row_size,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    const size_t raw_elem_size = (act_elem_size == sizeof(__fp16)) ? sizeof(__fp16) : sizeof(float);
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_row_size > 0 ? htp_mm_round_up(bias_row_size, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;
    const size_t lut_sz = (wtype == HTP_TYPE_IQ2_XXS) ? HTP_IQ2XXS_LUT_BYTES : 0;

    if (is_fused_nx) {
        uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t weight_sz_per_hvx = hex_round_up(n_prefetch * tile_row_size, 128);

        size_t tiled_act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        size_t q_act_sz = hex_round_up(tiled_act_row_size * act_nrows, 128);
        size_t raw_row_size = htp_mm_act_raw_row_size(ne10, raw_elem_size);

        src0_sz    = weight_sz_per_hvx * n_hvx; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
    } else if (is_matmul_id) {
        const size_t act_row_size_tiled = htp_mm_tiled_act_row_size(wtype, ne10);
        const uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
        const uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        const uint32_t tile_row_size     = n_k_tiles * aligned_tile_size;
        size_t src0_sz_per_hvx           = htp_mm_round_up(n_prefetch * tile_row_size, 256);

        act_sz = htp_mm_round_up(act_row_size_tiled * act_nrows, 256);

        size_t raw_row_size = htp_mm_act_raw_row_size(ne10, raw_elem_size);

        src0_sz    = src0_sz_per_hvx * n_hvx;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
    } else {
        const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
        const size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        switch (kernel_type) {
            case HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_row_size = htp_mm_round_up(ne10 * 2, 128);
                act_sz     = htp_mm_round_up(f16_act_row_size * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_hvx;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_hvx : 0;
                act_raw_sz = hex_round_up(hex_round_up(ne10 * raw_elem_size, 128) * act_nrows, 128);
                break;
            }
            case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            case HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_row_size = htp_mm_round_up(ne10 * 4, 128);
                act_sz     = htp_mm_round_up(f32_act_row_size * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_hvx;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_hvx : 0;
                act_raw_sz = (raw_elem_size == sizeof(__fp16))
                           ? hex_round_up(hex_round_up(ne10 * sizeof(__fp16), 128) * act_nrows, 128)
                           : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
            case HTP_MM_KERNEL_HVX_QUANT_ROW: {
                size_t q_act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
                uint32_t n_weight_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_row_size = n_weight_tiles * aligned_tile_size;
                size_t tiled_vtcm_size = htp_mm_round_up(n_prefetch * tile_row_size, 256);

                src0_sz = tiled_vtcm_size * n_hvx;
                act_sz  = htp_mm_round_up(q_act_row_size * act_nrows, 256);

                size_t dst_slice_per_hvx = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_round_up((dst_row_size + n_hvx - 1) / n_hvx, 128) : 0;
                dst_sz = dst_slice_per_hvx * n_hvx;
                size_t raw_row_size = htp_mm_act_raw_row_size(ne10, raw_elem_size);
                act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
                break;
            }
            default:
                break;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_group_a = 0;
    VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, bias_sz, bias_sz > 0);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_lut, lut_sz, lut_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_group_b = off_group_a;
    VTCM_LAYOUT_ALLOC(off_group_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst, dst_sz, dst_sz > 0);
    const size_t group_b_size = off_group_b - off_group_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_group_c = off_group_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    const size_t group_c_size = off_group_c - off_group_a;

    L->src0_bytes    = src0_sz;
    L->act_bytes     = act_sz;
    L->bias_bytes    = bias_sz;
    L->dst_bytes     = dst_sz;
    L->act_raw_bytes = act_raw_sz;
    L->lut_bytes     = lut_sz;
    L->total_bytes   = off_group_a + hex_smax(group_b_size, group_c_size);
}

static inline bool htp_mm_hvx_solve_vtcm_params(
    int kernel_type,
    int wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_hvx,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t act_elem_size,
    size_t bias_row_size,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    struct htp_mm_hvx_vtcm_layout * L_out,
    uint32_t * m_chunk_out
) {
    const size_t raw_elem_size = (act_elem_size == sizeof(__fp16)) ? sizeof(__fp16) : sizeof(float);
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_hvx,
        dst_row_size, src0_row_size, raw_elem_size, bias_row_size, n_prefetch, false, false
    );

    if (L.total_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = act_nrows;
        return true;
    }

    const size_t fixed_bytes = L.src0_bytes + L.bias_bytes + L.dst_bytes + L.lut_bytes;
    if (vtcm_budget <= fixed_bytes) {
        return false;
    }

    const size_t avail_act = vtcm_budget - fixed_bytes;
    size_t row_size = 0;
    if (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        row_size = htp_mm_tiled_act_row_size(wtype, ne10);
    } else if (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        row_size = hex_round_up(ne10 * 2, 128);
    } else {
        row_size = hex_round_up(ne10 * 4, 128);
    }
    if (row_size == 0) {
        return false;
    }

    size_t eff_row_size = row_size;
    if (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        eff_row_size += htp_mm_act_raw_row_size(ne10, raw_elem_size);
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_row_size);
    if (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    if (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    if (m_chunk < 1) {
        return false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_hvx,
        dst_row_size, src0_row_size, raw_elem_size, bias_row_size, n_prefetch, false, false
    );

    while (m_chunk > 2 && L.total_bytes > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_hvx,
            dst_row_size, src0_row_size, raw_elem_size, bias_row_size, n_prefetch, false, false
        );
    }

    if (m_chunk > 1 && L.total_bytes > vtcm_budget) {
        m_chunk = 1;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_hvx,
            dst_row_size, src0_row_size, raw_elem_size, bias_row_size, n_prefetch, false, false
        );
    }

    if (L.total_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = m_chunk;
        return true;
    }

    return false;
}

static inline size_t htp_mm_hmx_get_2d_vtcm_size_ex(
    int kernel_type, int wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_hvx, uint32_t aligned_tile_size, size_t bias_size, uint32_t k_group
) {
    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build_ex(&L, kernel_type, wtype, k, mc, nc, 1, pipeline, act_hvx, aligned_tile_size, bias_size, k_group);
    return L.total_bytes;
}

static inline size_t htp_mm_hmx_get_2d_vtcm_size(
    int wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_hvx, uint32_t aligned_tile_size, size_t bias_size
) {
    return htp_mm_hmx_get_2d_vtcm_size_ex(HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, pipeline, act_hvx, aligned_tile_size, bias_size, 0);
}

static inline size_t htp_mm_hmx_get_batched_vtcm_size(
    int wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool pipeline, uint32_t act_hvx, size_t bias_size) {
    (void)pipeline;
    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, group_size, false, act_hvx, 0, bias_size);
    return L.total_bytes;
}

static inline bool htp_mm_hmx_solve_batched_params(
    int wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t group_size,
    int n_hvx,
    bool pipeline,
    size_t bias_size,
    size_t vtcm_budget,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    int * act_hvx_out,
    size_t * vtcm_size_out
) {
    size_t best_mblocks = SIZE_MAX;
    int best_act_hvx = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_size = 0;

    int act_hvx = n_hvx;
    while (act_hvx >= 1) {
        size_t group_overhead = htp_mm_hmx_get_batched_overhead() + (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0);
        size_t group_size_per_n, group_size_per_m, group_size_per_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, group_size, &group_size_per_n, &group_size_per_m, &group_size_per_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_size_candidate = 0;

        if (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_size_per_n, group_size_per_m, group_size_per_mn, hex_align_up(ne11, 32), ne01_padded, HTP_MM_HMX_TILE_N_ROWS,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
            size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, group_size, pipeline, act_hvx, bias_size);
            if (exact_size <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                if (mblocks < best_mblocks || (mblocks == best_mblocks && act_hvx > best_act_hvx)) {
                    best_mblocks = mblocks;
                    best_act_hvx = act_hvx;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_size = exact_size;
                }
            }
        }
        if (act_hvx == 1) {
            act_hvx = 0;
        } else {
            act_hvx /= 2;
        }
    }

    if (best_act_hvx > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_size_out = best_vtcm_size;
        *act_hvx_out = best_act_hvx;
        return true;
    }
    return false;
}

static inline bool htp_mm_int_hmx_solve_2d_params(
    int kernel_type,
    int wtype,
    uint32_t k,
    uint32_t m_id_rows,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    int n_hvx,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_size,
    size_t bias_size,
    size_t vtcm_budget,
    uint32_t k_group,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    int * act_hvx_out,
    size_t * vtcm_size_out
) {
    const bool is_w8a8 = htp_mm_kernel_is_hmx_w8a8(kernel_type);
    const size_t m_align = is_w8a8 ? 64u : 32u;
    const size_t n_total = hex_align_up((size_t) ne01_padded, HTP_MM_HMX_TILE_N_COLS);
    const size_t m_target = is_matmul_id ? hex_align_up((size_t) m_id_rows, m_align)
                                         : hex_align_up((size_t) ne11_padded, m_align);
    const size_t m_rows = (m_for_cost > 0) ? (size_t) m_for_cost : m_target;
    if (m_target == 0 || n_total == 0) {
        return false;
    }

    size_t per_n_approx = 0, per_m_approx = 0, per_mn_approx = 0;
    htp_mm_hmx_get_2d_chunk_costs_ex(kernel_type, wtype, k, pipeline, aligned_tile_size, k_group,
                                     &per_n_approx, &per_m_approx, &per_mn_approx);
    const size_t base_overhead = htp_mm_hmx_get_2d_overhead(wtype, pipeline, is_matmul_id) +
                                 HTP_MM_HMX_TILE_SIZE + (pipeline ? 2u : 1u) * 768u +
                                 (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0u);
    if (vtcm_budget <= base_overhead || per_n_approx == 0 || per_m_approx == 0 || per_mn_approx == 0) {
        return false;
    }
    const size_t usable = vtcm_budget - base_overhead;

    const size_t min_par_nc = (size_t) (n_hvx > 0 ? n_hvx : 1) * HTP_MM_HMX_TILE_N_COLS;
    size_t nc_cap = n_total;
    if (pipeline) {
        if (n_total >= 4u * min_par_nc) {
            nc_cap = hex_align_down(n_total / 4u, HTP_MM_HMX_TILE_N_COLS);
        } else if (n_total >= 2u * min_par_nc) {
            nc_cap = hex_align_down(n_total / 2u, HTP_MM_HMX_TILE_N_COLS);
        }
        if (nc_cap > 512u && min_par_nc <= 512u) {
            nc_cap = 512u;
        }
    }
    const size_t nc_hi = hex_align_down(hex_smin(nc_cap, usable / per_n_approx), HTP_MM_HMX_TILE_N_COLS);
    if (nc_hi < HTP_MM_HMX_TILE_N_COLS) {
        return false;
    }

    size_t best_mblocks   = SIZE_MAX;
    size_t best_ncost     = SIZE_MAX;
    int    best_act_hvx   = 0;
    size_t best_mc        = 0;
    size_t best_nc        = 0;
    size_t best_vtcm_size = 0;

    for (int act_hvx = n_hvx; act_hvx >= 1; act_hvx = (act_hvx == 1) ? 0 : (act_hvx / 2)) {
        const uint32_t layout_act_hvx = (uint32_t) act_hvx;

        for (size_t nc = nc_hi; nc >= HTP_MM_HMX_TILE_N_COLS; nc -= HTP_MM_HMX_TILE_N_COLS) {
            size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
            if (hex_mul_overflow(nc, per_n_approx, &n_fixed) || n_fixed >= usable) {
                if (nc == HTP_MM_HMX_TILE_N_COLS) break;
                continue;
            }
            if (hex_mul_overflow(nc, per_mn_approx, &ncmn) ||
                hex_add_overflow(per_m_approx, ncmn, &mc_denom) || mc_denom == 0) {
                if (nc == HTP_MM_HMX_TILE_N_COLS) break;
                continue;
            }

            size_t mc = hex_align_down((usable - n_fixed) / mc_denom, m_align);
            mc = hex_smin(mc, m_target);
            while (mc >= m_align) {
                size_t exact = htp_mm_hmx_get_2d_vtcm_size_ex(
                    kernel_type, wtype, k, mc, nc, pipeline, layout_act_hvx, aligned_tile_size, bias_size, k_group);
                if (exact <= vtcm_budget) {
                    const size_t mblocks = (m_rows + mc - 1u) / mc;
                    size_t mc_tight = (mblocks > 1u)
                        ? hex_smin(mc, hex_align_up((m_rows + mblocks - 1u) / mblocks, m_align))
                        : mc;
                    if (mc_tight != mc) {
                        size_t exact_tight = htp_mm_hmx_get_2d_vtcm_size_ex(
                            kernel_type, wtype, k, mc_tight, nc, pipeline, layout_act_hvx, aligned_tile_size, bias_size, k_group);
                        if (exact_tight <= vtcm_budget) {
                            mc    = mc_tight;
                            exact = exact_tight;
                        }
                    }

                    const size_t nblocks    = (n_total + nc - 1u) / nc;
                    const size_t col_tiles  = nc / HTP_MM_HMX_TILE_N_COLS;
                    const size_t hvx_cnt    = (n_hvx > 0) ? (size_t) n_hvx : 1u;
                    const size_t eff_nc     = (col_tiles >= hvx_cnt)
                        ? (((col_tiles + hvx_cnt - 1u) / hvx_cnt) * hvx_cnt * HTP_MM_HMX_TILE_N_COLS)
                        : nc;
                    const size_t pipe_tail  = pipeline ? ((nblocks == 1u) ? (2u * eff_nc) : eff_nc) : 0u;
                    const size_t ncost      = nblocks * eff_nc + pipe_tail + nblocks * HTP_MM_HMX_TILE_N_COLS;

                    const bool better =
                        (mblocks < best_mblocks) ||
                        (mblocks == best_mblocks && ncost < best_ncost) ||
                        (mblocks == best_mblocks && ncost == best_ncost && act_hvx > best_act_hvx) ||
                        (mblocks == best_mblocks && ncost == best_ncost && act_hvx == best_act_hvx && nc < best_nc);

                    if (better) {
                        best_mblocks   = mblocks;
                        best_ncost     = ncost;
                        best_act_hvx   = act_hvx;
                        best_mc        = mc;
                        best_nc        = nc;
                        best_vtcm_size = exact;
                    }
                    break;
                }
                mc -= m_align;
            }

            if ((best_mblocks == 1u && best_nc >= n_total) || nc == HTP_MM_HMX_TILE_N_COLS) break;
        }
        if (best_mblocks == 1u && (best_nc >= n_total || best_nc >= 128u)) break;
    }

    if (best_act_hvx > 0) {
        *m_chunk_out   = best_mc;
        *n_chunk_out   = best_nc;
        *act_hvx_out   = best_act_hvx;
        *vtcm_size_out = best_vtcm_size;
        return true;
    }
    return false;
}

static inline bool htp_mm_hmx_solve_2d_params_ex(
    int kernel_type,
    int wtype,
    uint32_t k,
    uint32_t m_id_rows,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    int n_hvx,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_size,
    size_t bias_size,
    size_t vtcm_budget,
    uint32_t k_group,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    int * act_hvx_out,
    size_t * vtcm_size_out
) {
    if (htp_mm_kernel_is_hmx_int(kernel_type)) {
        return htp_mm_int_hmx_solve_2d_params(
            kernel_type, wtype, k, m_id_rows, ne01_padded, ne11_padded, m_for_cost,
            n_hvx, pipeline, is_matmul_id, aligned_tile_size, bias_size, vtcm_budget, k_group,
            m_chunk_out, n_chunk_out, act_hvx_out, vtcm_size_out);
    }

    size_t best_mblocks = SIZE_MAX;
    int best_act_hvx = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_size = 0;

    const bool is_w8a8 = htp_mm_kernel_is_hmx_w8a8(kernel_type);
    const size_t m_align = is_w8a8 ? 64 : 32;
    const size_t m_for_chunks = is_matmul_id ? hex_align_up(m_id_rows, m_align) : hex_align_up(ne11_padded, m_align);

    int act_hvx = n_hvx;
    while (act_hvx >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(wtype, pipeline, is_matmul_id) + (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0);
        size_t simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn;
        htp_mm_hmx_get_2d_chunk_costs_ex(kernel_type, wtype, k, pipeline, aligned_tile_size, k_group,
                                         &simple_2d_size_per_n, &simple_2d_size_per_m, &simple_2d_size_per_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_size_candidate = 0;

        if (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn, m_for_chunks, ne01_padded, m_align,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
            size_t exact_size = htp_mm_hmx_get_2d_vtcm_size_ex(kernel_type, wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_hvx, aligned_tile_size, bias_size, k_group);
            if (exact_size <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                if (mblocks < best_mblocks || (mblocks == best_mblocks && act_hvx > best_act_hvx)) {
                    best_mblocks = mblocks;
                    best_act_hvx = act_hvx;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_size = exact_size;
                }
            }
        }
        if (act_hvx == 1) {
            act_hvx = 0;
        } else {
            act_hvx /= 2;
        }
    }

    if (best_act_hvx > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_size_out = best_vtcm_size;
        *act_hvx_out = best_act_hvx;
        return true;
    }
    return false;
}

static inline bool htp_mm_hmx_solve_2d_params(
    int wtype,
    uint32_t k,
    uint32_t m_id_rows,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    int n_hvx,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_size,
    size_t bias_size,
    size_t vtcm_budget,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    int * act_hvx_out,
    size_t * vtcm_size_out
) {
    return htp_mm_hmx_solve_2d_params_ex(HTP_MM_KERNEL_HMX_2D, wtype, k, m_id_rows, ne01_padded, ne11_padded, m_for_cost,
                                         n_hvx, pipeline, is_matmul_id, aligned_tile_size, bias_size, vtcm_budget,
                                         0, m_chunk_out, n_chunk_out, act_hvx_out, vtcm_size_out);
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
