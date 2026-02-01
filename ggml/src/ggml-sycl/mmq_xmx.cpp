//
// XMX-accelerated quantized matrix multiplication for Intel Arc GPUs
// Uses Intel Xe Matrix eXtensions (XMX) via SYCL joint_matrix API
//

#include "mmq_xmx.hpp"

#include "common.hpp"

#include <sycl/sycl.hpp>

// Check for joint_matrix support
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define SYCL_XMX_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#    define SYCL_XMX_AVAILABLE 0
#endif

#if SYCL_XMX_AVAILABLE

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// XMX Int8 GEMM Configuration for Intel Arc (Xe-HPG)
// =============================================================================
constexpr int XMX_M       = 8;   // Tile rows
constexpr int XMX_N       = 16;  // Tile columns
constexpr int XMX_K       = 32;  // Reduction dimension (matches QK8_0 and QK4_0!)
constexpr int XMX_SG_SIZE = 16;  // Sub-group size for Intel XMX

// =============================================================================
// Single-Tile Configuration (optimized for XMX)
// =============================================================================
// Each work-group = one sub-group processes one XMX tile (8×16 output)
// Optimization: Use vectorized loads for better memory bandwidth

// Work-group output dimensions = one XMX tile
constexpr int WG_M = XMX_M;  // 8 rows per work-group
constexpr int WG_N = XMX_N;  // 16 cols per work-group

// =============================================================================
// Q4_0 x Q8_1 XMX GEMM Kernel - Single-Tile with Vectorized Loads
// Computes: C[M,N] = A[M,K] * B[K,N] where A is Q4_0 weights, B is Q8_1 activations
//
// Design: Each work-group = one sub-group processes one XMX tile (8×16 output)
// Optimizations:
// - Vectorized memory loads using sycl::vec<uint8_t, 8> for Q4_0 nibbles
// - Vectorized memory loads using sycl::vec<int8_t, 16> for Q8_1 values
// - Efficient SLM usage with minimal barriers
// =============================================================================

// SLM size constants for single-tile
constexpr int SLM_A_SIZE        = WG_M * XMX_K;   // 8*32 = 256 int8
constexpr int SLM_B_SIZE        = XMX_K * WG_N;   // 32*16 = 512 int8
constexpr int SLM_C_SIZE        = XMX_M * XMX_N;  // 8*16 = 128 int32
constexpr int SLM_SCALES_A_SIZE = WG_M;           // 8 floats
constexpr int SLM_SCALES_B_SIZE = WG_N;           // 16 floats
constexpr int SLM_SUMS_B_SIZE   = WG_N;           // 16 floats (for Q8_1 sum field)

// Double-buffer SLM sizes (2× for A, B, scales, and sums)
constexpr int SLM_A_SIZE_DB        = SLM_A_SIZE * 2;         // 512 int8
constexpr int SLM_B_SIZE_DB        = SLM_B_SIZE * 2;         // 1024 int8
constexpr int SLM_SCALES_A_SIZE_DB = SLM_SCALES_A_SIZE * 2;  // 16 floats
constexpr int SLM_SCALES_B_SIZE_DB = SLM_SCALES_B_SIZE * 2;  // 32 floats
constexpr int SLM_SUMS_B_SIZE_DB   = SLM_SUMS_B_SIZE * 2;    // 32 floats

static inline int64_t xmx_gemm_weight_block_index(int row,
                                                  int block_idx,
                                                  int blocks_per_row,
                                                  int tile_m,
                                                  bool use_tiled) {
    if (!use_tiled) {
        return static_cast<int64_t>(row) * blocks_per_row + block_idx;
    }
    const int tile        = row / tile_m;
    const int row_in_tile = row - tile * tile_m;
    return static_cast<int64_t>(tile) * static_cast<int64_t>(blocks_per_row) * tile_m +
           static_cast<int64_t>(block_idx) * tile_m + row_in_tile;
}

// =============================================================================
// Single-Tile XMX GEMM Kernel with Vectorized Loads
// Each work-group = one sub-group processes one 8×16 output tile
// =============================================================================
void mul_mat_q4_0_q8_1_xmx_kernel(
    const void * __restrict__ vx,       // Q4_0 weights [nrows_x, ncols_x/32 blocks]
    const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks] - AOS layout
    float * __restrict__ dst,           // Output [nrows_x, ncols_y]
    const int ncols_x,                  // K dimension (must be multiple of 32)
    const int nrows_x,                  // M dimension (rows of weights to process)
    const int ncols_y,                  // N dimension (batch size / columns of output)
    const int nrows_dst,                // Output row stride
    int8_t * __restrict__ slm_A,        // SLM for A tile [8 × 32]
    int8_t * __restrict__ slm_B,        // SLM for B tile [32 × 16]
    int32_t * __restrict__ slm_C,       // SLM for C tile [8 × 16]
    float * __restrict__ slm_scales_A,  // SLM for A scales [8]
    float * __restrict__ slm_scales_B,  // SLM for B scales [16]
    float * __restrict__ slm_sums_B,    // SLM for B sums [16] - for Q8_1 sum field
    const int tile_m,
    const bool use_tiled,
    sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    // Work-group position (one XMX tile per work-group)
    const int row_base = item.get_group(0) * XMX_M;  // 8 rows
    const int col_base = item.get_group(1) * XMX_N;  // 16 cols

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    // Block pointers
    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;  // sizeof(block_q4_0) = 2 + 16
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 2 + 2 + 32

    // Private accumulator - 8 elements per lane (128 total / 16 lanes)
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // ============== Load A: 8 rows × 32 values ==============
        // Q4_0: store RAW nibbles (0-15), correction applied later via sum term
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;

                // Load scale
                sycl::half d          = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                // Unpack 16 bytes → 32 int8 values (RAW, no -8 offset!)
                // Q4_0 storage: byte[j] has low nibble = K position j, high nibble = K position j+16
                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                int             base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed           = nibbles[j];
                    slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);  // K position j
                    slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);    // K position j+16
                }
            } else {
                // Zero padding for out-of-bounds rows
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // ============== Load B: 16 columns × 32 values ==============
        // Q8_1: AOS layout - each block is [half d][half sum][int8 qs[32]] = 36 bytes
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);  // sum field
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);  // s = d * sum(qs), precomputed

// Store column-major for XMX B matrix (stride = XMX_K = 32)
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                // Zero padding for out-of-bounds columns
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        // Barrier to ensure all loading is complete
        item.barrier(sycl::access::fence_space::local_space);

        // ============== XMX Multiply ==============
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// ============== Apply scales with sum correction ==============
// Formula: result = d_A * (C * d_B - 8 * s_B)
// Where C = dot(raw_q4_0, q8_1), s_B = d_B * sum(q8_1)
// This correctly accounts for the -8 offset in Q4_0 values
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i    = idx / XMX_N;  // row within tile
            int   j    = idx % XMX_N;  // col within tile
            float d_A  = slm_scales_A[i];
            float d_B  = slm_scales_B[j];
            float s_B  = slm_sums_B[j];
            float C_ij = float(slm_C[idx]);
            // Match standard formula: d_A * (C * d_B - 8 * s_B)
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
        }

        // Barrier before next K block (SLM will be overwritten)
        item.barrier(sycl::access::fence_space::local_space);
    }

// ============== Write output ==============
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            // Output is column-major (matches MMQ: dst[col*nrows_dst + row])
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Q4_0 x Q8_1 XMX GEMM Kernel - Double-Buffered Version
// Overlaps memory loads with computation for better performance
// Uses ping-pong buffers: while computing on buffer 0, load into buffer 1
// =============================================================================
void mul_mat_q4_0_q8_1_xmx_doublebuf_kernel(
    const void * __restrict__ vx,       // Q4_0 weights [nrows_x, ncols_x/32 blocks]
    const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks]
    float * __restrict__ dst,           // Output [nrows_x, ncols_y]
    const int ncols_x,                  // K dimension (must be multiple of 32)
    const int nrows_x,                  // M dimension (rows of weights to process)
    const int ncols_y,                  // N dimension (batch size / columns of output)
    const int nrows_dst,                // Output row stride
    int8_t * __restrict__ slm_A,        // SLM for A tiles [2 × 8 × 32]
    int8_t * __restrict__ slm_B,        // SLM for B tiles [2 × 32 × 16]
    int32_t * __restrict__ slm_C,       // SLM for C tile [8 × 16]
    float * __restrict__ slm_scales_A,  // SLM for A scales [2 × 8]
    float * __restrict__ slm_scales_B,  // SLM for B scales [2 × 16]
    float * __restrict__ slm_sums_B,    // SLM for B sums [2 × 16]
    const int tile_m,
    const bool use_tiled,
    sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    // Work-group position (one XMX tile per work-group)
    const int row_base = item.get_group(0) * XMX_M;  // 8 rows
    const int col_base = item.get_group(1) * XMX_N;  // 16 cols

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    // Block pointers
    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;  // sizeof(block_q4_0) = 2 + 16
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 2 + 2 + 32

    // Private accumulator - 8 elements per lane (128 total / 16 lanes)
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Buffer indices for double buffering
    int buf_load = 0;  // Buffer to load into
    int buf_comp = 0;  // Buffer to compute from

    // ============== Pre-load first K block into buffer 0 ==============
    if (num_k_blocks > 0) {
        // Load A: 8 rows × 32 values (vectorized Q4_0 nibble load)
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, 0, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = float(d);

                // Load and unpack Q4_0 nibbles (simple scalar - memory bound anyway)
                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                int             base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed           = nibbles[j];
                    slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                    slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                }
            } else {
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                int base_idx                                         = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B: 16 columns × 32 values (vectorized Q8_1 load)
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr = src1 + (col * num_k_blocks + 0) * Q8_1_BLOCK_SIZE;
                sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half   s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = float(d);
                slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = float(s);

                // Load Q8_1 values (simple scalar - memory bound anyway)
                const int8_t * qs       = reinterpret_cast<const int8_t *>(block_ptr + 4);
                int            base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = 0.0f;
                int base_idx                                         = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    // Process K blocks with double buffering
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        buf_comp = k_block & 1;        // Current buffer for compute
        buf_load = (k_block + 1) & 1;  // Next buffer for loading

        // ============== Load NEXT K block (if not last) ==============
        if (k_block + 1 < num_k_blocks) {
            int next_k = k_block + 1;

            // Load A for next K block (vectorized)
            if (lane_id < XMX_M) {
                int row = row_base + lane_id;
                if (row < nrows_x) {
                    const int64_t block_idx = xmx_gemm_weight_block_index(row, next_k, num_k_blocks, tile_m, use_tiled);
                    const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                    sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = float(d);

                    // Load and unpack Q4_0 nibbles
                    const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                    int             base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#    pragma unroll
                    for (int j = 0; j < 16; j++) {
                        uint8_t packed           = nibbles[j];
                        slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                        slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                    }
                } else {
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                    int base_idx                                         = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#    pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = 0;
                    }
                }
            }

            // Load B for next K block (vectorized)
            {
                int col = col_base + lane_id;
                if (col < ncols_y) {
                    const char * block_ptr = src1 + (col * num_k_blocks + next_k) * Q8_1_BLOCK_SIZE;
                    sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                    sycl::half   s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = float(d);
                    slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = float(s);

                    // Load Q8_1 values
                    const int8_t * qs       = reinterpret_cast<const int8_t *>(block_ptr + 4);
                    int            base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#    pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = qs[k];
                    }
                } else {
                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                    slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = 0.0f;
                    int base_idx                                         = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#    pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = 0;
                    }
                }
            }
        }

        // ============== XMX Multiply using CURRENT buffer ==============
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
            slm_A + buf_comp * SLM_A_SIZE);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
            slm_B + buf_comp * SLM_B_SIZE);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// ============== Apply scales with sum correction ==============
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i    = idx / XMX_N;
            int   j    = idx % XMX_N;
            float d_A  = slm_scales_A[buf_comp * SLM_SCALES_A_SIZE + i];
            float d_B  = slm_scales_B[buf_comp * SLM_SCALES_B_SIZE + j];
            float s_B  = slm_sums_B[buf_comp * SLM_SUMS_B_SIZE + j];
            float C_ij = float(slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
        }

        // Barrier to sync load and compute phases
        item.barrier(sycl::access::fence_space::local_space);
    }

// ============== Write output ==============
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Q4_0 x Q8_1 XMX GEMM Kernel - Multi-Tile Version
// Uses 4 sub-groups per work-group to process 4 tiles (8×64 output)
// Key optimization: A tile loaded once, shared across all sub-groups
// =============================================================================

// Multi-tile configuration - VERTICAL tiling (multiple rows)
// Each work-group processes 4 row tiles × all columns (32 rows × 16 columns)
// 4 tiles is optimal - more tiles causes diminishing returns due to sub-group overhead
constexpr int MT_TILES_M     = 4;                          // 4 tiles vertically per work-group
constexpr int MT_SG_COUNT    = MT_TILES_M;                 // 4 sub-groups per work-group
constexpr int MT_WG_SIZE     = MT_SG_COUNT * XMX_SG_SIZE;  // 64 threads per work-group
constexpr int MT_OUTPUT_ROWS = MT_TILES_M * XMX_M;         // 32 rows per work-group

// Multi-tile SLM sizes - each sub-group has its own A tile (different rows)
constexpr int MT_SLM_A_SIZE        = XMX_M * XMX_K * MT_TILES_M;  // 8×32×4 = 1024 (per sub-group, different rows)
constexpr int MT_SLM_B_SIZE        = XMX_K * XMX_N;               // 32×16 = 512 (shared B tile)
constexpr int MT_SLM_C_SIZE        = XMX_M * XMX_N;               // 8×16 = 128 (per sub-group)
constexpr int MT_SLM_SCALES_A_SIZE = XMX_M * MT_TILES_M;          // 8×4 = 32 (per sub-group)
constexpr int MT_SLM_SCALES_B_SIZE = XMX_N;                       // 16 (shared)
constexpr int MT_SLM_SUMS_B_SIZE   = XMX_N;                       // 16 (shared)

void mul_mat_q4_0_q8_1_xmx_multitile_kernel(
    const void * __restrict__ vx,       // Q4_0 weights [nrows_x, ncols_x/32 blocks]
    const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks]
    float * __restrict__ dst,           // Output [nrows_x, ncols_y]
    const int ncols_x,                  // K dimension (must be multiple of 32)
    const int nrows_x,                  // M dimension
    const int ncols_y,                  // N dimension (batch size)
    const int nrows_dst,                // Output row stride
    int8_t * __restrict__ slm_A,        // SLM for A tiles [2 × 8 × 32] - per sub-group
    int8_t * __restrict__ slm_B,        // SLM for shared B tile [32 × 16]
    int32_t * __restrict__ slm_C,       // SLM for C tiles [2 × 8 × 16]
    float * __restrict__ slm_scales_A,  // SLM for A scales [2 × 8]
    float * __restrict__ slm_scales_B,  // SLM for shared B scales [16]
    float * __restrict__ slm_sums_B,    // SLM for shared B sums [16]
    const int tile_m,
    const bool use_tiled,
    sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = sg.get_group_id()[0];  // Which sub-group (0 or 1)
    const int  lane_id = sg.get_local_id()[0];  // Lane within sub-group (0-15)

    // Work-group position - vertical tiling (2 row tiles per work-group)
    const int row_base = item.get_group(0) * MT_OUTPUT_ROWS + sg_id * XMX_M;  // Each sub-group handles different rows
    const int col_base = item.get_group(1) * XMX_N;  // 16 cols per work-group (single column tile)

    if (row_base >= nrows_x) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    // Block pointers
    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int Q8_1_BLOCK_SIZE = 36;

    // Per-sub-group SLM pointers for A (each sub-group has different rows)
    int8_t *  my_slm_A        = slm_A + sg_id * (XMX_M * XMX_K);
    int32_t * my_slm_C        = slm_C + sg_id * (XMX_M * XMX_N);
    float *   my_slm_scales_A = slm_scales_A + sg_id * XMX_M;
    // B is shared across both sub-groups (same columns)

    // Private accumulator
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // ============== Phase 1: Load A tiles (each sub-group loads different rows) ==============
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;

                // Load scale
                sycl::half d             = *reinterpret_cast<const sycl::half *>(block_ptr);
                my_slm_scales_A[lane_id] = float(d);

                // Unpack 16 bytes → 32 int8 values
                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                int             base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed              = nibbles[j];
                    my_slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                    my_slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                }
            } else {
                my_slm_scales_A[lane_id] = 0.0f;
                int base_idx             = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    my_slm_A[base_idx + j] = 0;
                }
            }
        }

        // ============== Phase 2: Load shared B tile (sub-group 0 loads, both use) ==============
        if (sg_id == 0) {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        // Barrier to ensure all data is loaded
        item.barrier(sycl::access::fence_space::local_space);

        // ============== Phase 3: XMX Compute (each sub-group uses own A, shared B) ==============
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_A);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
            slm_B);  // Shared
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);  // Both sub-groups read same B
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply scales with sum correction
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i    = idx / XMX_N;
            int   j    = idx % XMX_N;
            float d_A  = my_slm_scales_A[i];
            float d_B  = slm_scales_B[j];  // Shared
            float s_B  = slm_sums_B[j];    // Shared
            float C_ij = float(my_slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
        }

        // Barrier before next K block
        item.barrier(sycl::access::fence_space::local_space);
    }

// ============== Write output ==============
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Q4_0 x Q8_1 XMX GEMM Kernel - Large Tile Version (256 threads)
// Uses 16 sub-groups (4×4) to process 64×64 output tile = 4096 outputs
// Key optimization: Much larger tile size amortizes work-group launch overhead
// =============================================================================

// Large-tile configuration: 4×4 sub-groups = 16 sub-groups = 256 threads
constexpr int LT_TILES_M     = 8;                          // 8 row tiles per work-group
constexpr int LT_TILES_N     = 4;                          // 4 column tiles per work-group
constexpr int LT_SG_COUNT    = LT_TILES_M * LT_TILES_N;    // 32 sub-groups per work-group
constexpr int LT_WG_SIZE     = LT_SG_COUNT * XMX_SG_SIZE;  // 512 threads per work-group
constexpr int LT_OUTPUT_ROWS = LT_TILES_M * XMX_M;         // 64 rows per work-group
constexpr int LT_OUTPUT_COLS = LT_TILES_N * XMX_N;         // 64 cols per work-group

// Large-tile SLM sizes (C is accumulated in registers, not SLM)
constexpr int LT_SLM_A_SIZE        = XMX_M * XMX_K * LT_TILES_M;  // 8×32×8 = 2048 bytes (shared by col tiles)
constexpr int LT_SLM_B_SIZE        = XMX_K * XMX_N * LT_TILES_N;  // 32×16×4 = 2048 bytes (shared by row tiles)
constexpr int LT_SLM_SCALES_A_SIZE = XMX_M * LT_TILES_M;          // 8×8 = 64 floats = 256 bytes
constexpr int LT_SLM_SCALES_B_SIZE = XMX_N * LT_TILES_N;          // 16×4 = 64 floats = 256 bytes
constexpr int LT_SLM_SUMS_B_SIZE   = XMX_N * LT_TILES_N;          // 16×4 = 64 floats = 256 bytes

// Total SLM: ~5KB (reduced from ~21KB by using register accumulation)

void mul_mat_q4_0_q8_1_xmx_largetile_kernel(const void * __restrict__ vx,       // Q4_0 weights
                                            const void * __restrict__ vy,       // Q8_1 activations
                                            float * __restrict__ dst,           // Output
                                            const int ncols_x,                  // K dimension
                                            const int nrows_x,                  // M dimension
                                            const int ncols_y,                  // N dimension (batch size)
                                            const int nrows_dst,                // Output row stride
                                            int8_t * __restrict__ slm_A,        // SLM for A tiles [8 × 8 × 32]
                                            int8_t * __restrict__ slm_B,        // SLM for B tiles [4 × 32 × 16]
                                            float * __restrict__ slm_scales_A,  // SLM for A scales [8 × 8]
                                            float * __restrict__ slm_scales_B,  // SLM for B scales [4 × 16]
                                            float * __restrict__ slm_sums_B,    // SLM for B sums [4 × 16]
                                            const int tile_m,
                                            const bool use_tiled,
                                            sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = sg.get_group_id()[0];  // Which sub-group (0-31)
    const int  lane_id = sg.get_local_id()[0];  // Lane within sub-group (0-15)

    // Map sub-group ID to 2D tile position
    const int sg_row = sg_id / LT_TILES_N;  // Row tile within work-group (0-7)
    const int sg_col = sg_id % LT_TILES_N;  // Col tile within work-group (0-3)

    // Work-group position
    const int wg_row_base = item.get_group(0) * LT_OUTPUT_ROWS;
    const int wg_col_base = item.get_group(1) * LT_OUTPUT_COLS;

    // This sub-group's tile position
    const int row_base = wg_row_base + sg_row * XMX_M;
    const int col_base = wg_col_base + sg_col * XMX_N;

    if (wg_row_base >= nrows_x) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int Q8_1_BLOCK_SIZE = 36;

    // Pointers to this sub-group's A and B tiles in SLM
    // Row tiles share the same A data, col tiles share the same B data
    int8_t * my_slm_A        = slm_A + sg_row * (XMX_M * XMX_K);
    int8_t * my_slm_B        = slm_B + sg_col * (XMX_K * XMX_N);
    float *  my_slm_scales_A = slm_scales_A + sg_row * XMX_M;
    float *  my_slm_scales_B = slm_scales_B + sg_col * XMX_N;
    float *  my_slm_sums_B   = slm_sums_B + sg_col * XMX_N;

    // Private accumulator (8 elements per lane)
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // ============== Phase 1: Load A tiles (only first col tile per row loads) ==============
        if (sg_col == 0 && lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                sycl::half   d           = *reinterpret_cast<const sycl::half *>(block_ptr);
                my_slm_scales_A[lane_id] = float(d);

                // Vectorized nibble unpacking: load 16 bytes as 4x uint32, unpack to 32 int8
                const uint32_t * nibbles_u32 = reinterpret_cast<const uint32_t *>(block_ptr + 2);
                int              base_idx    = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t packed4                    = nibbles_u32[j];
                    // Extract low nibbles (bytes 0-3 -> positions 0,1,2,3)
                    my_slm_A[base_idx + j * 4 + 0]      = (int8_t) ((packed4 >> 0) & 0x0F);
                    my_slm_A[base_idx + j * 4 + 1]      = (int8_t) ((packed4 >> 8) & 0x0F);
                    my_slm_A[base_idx + j * 4 + 2]      = (int8_t) ((packed4 >> 16) & 0x0F);
                    my_slm_A[base_idx + j * 4 + 3]      = (int8_t) ((packed4 >> 24) & 0x0F);
                    // Extract high nibbles (bytes 0-3 -> positions 16,17,18,19)
                    my_slm_A[base_idx + 16 + j * 4 + 0] = (int8_t) ((packed4 >> 4) & 0x0F);
                    my_slm_A[base_idx + 16 + j * 4 + 1] = (int8_t) ((packed4 >> 12) & 0x0F);
                    my_slm_A[base_idx + 16 + j * 4 + 2] = (int8_t) ((packed4 >> 20) & 0x0F);
                    my_slm_A[base_idx + 16 + j * 4 + 3] = (int8_t) ((packed4 >> 28) & 0x0F);
                }
            } else {
                my_slm_scales_A[lane_id] = 0.0f;
                int base_idx             = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    my_slm_A[base_idx + j] = 0;
                }
            }
        }

        // ============== Phase 2: Load B tiles (only first row tile per col loads) ==============
        if (sg_row == 0) {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                my_slm_scales_B[lane_id] = float(d);
                my_slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                my_slm_scales_B[lane_id] = 0.0f;
                my_slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        // Barrier to ensure all data is loaded
        item.barrier(sycl::access::fence_space::local_space);

        // ============== Phase 3: XMX Compute (each sub-group uses its row's A and col's B) ==============
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_B);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);

        // Apply scales using get_wi_data to access matrix elements directly
        // For Intel XMX 8x16 accumulator with 16-wide sub-group:
        // - Each work-item owns 1 column (lane_id) with 8 row elements
        // - wi_data[i] corresponds to row i of the column owned by this work-item
        auto wi_data = sycl::ext::oneapi::detail::get_wi_data(sg, matC);

        // Get this work-item's column's scales (same for all 8 rows)
        float d_B = my_slm_scales_B[lane_id];
        float s_B = my_slm_sums_B[lane_id];

#    pragma unroll
        for (int row = 0; row < XMX_M; row++) {
            int32_t C_val = wi_data[row];
            float   d_A   = my_slm_scales_A[row];
            // Q4_0 x Q8_1 formula: d_A * (C * d_B - 8 * s_B)
            // where 8 is the zero-point offset for Q4_0 (values 0-15 centered at 8)
            acc[row] += d_A * (float(C_val) * d_B - 8.0f * s_B);
        }

        // No barrier needed here - each sub-group works independently on its own data
        // The barrier at start of next iteration ensures A/B data is loaded before use
    }

    // ============== Write output ==============
    // Each work-item owns column lane_id of the 8x16 tile
    // acc[row] holds the value for position (row, lane_id)
    int col = col_base + lane_id;  // This work-item's column in output
    if (col < ncols_y) {
#    pragma unroll
        for (int row_offset = 0; row_offset < XMX_M; row_offset++) {
            int row = row_base + row_offset;
            if (row < nrows_x) {
                dst[col * nrows_dst + row] = acc[row_offset];
            }
        }
    }
}

// =============================================================================
// Q4_0 x Q8_1 XMX GEMM Kernel - Multi-Tile Double-Buffered Version
// Overlaps memory loads with computation using ping-pong buffers
// This reduces barrier overhead by 50% and hides memory latency
// =============================================================================

// Double-buffer SLM sizes for multi-tile kernel
constexpr int MT_DB_SLM_A_SIZE        = XMX_M * XMX_K * MT_TILES_M * 2;            // 2× for ping-pong
constexpr int MT_DB_SLM_B_SIZE        = XMX_K * XMX_N * 2;                         // 2× for ping-pong
constexpr int MT_DB_SLM_SCALES_A_SIZE = XMX_M * MT_TILES_M * 2;                    // 2× for ping-pong
constexpr int MT_DB_SLM_SCALES_B_SIZE = XMX_N * 2;                                 // 2× for ping-pong
constexpr int MT_DB_SLM_SUMS_B_SIZE   = XMX_N * 2;                                 // 2× for ping-pong

void mul_mat_q4_0_q8_1_xmx_multitile_db_kernel(const void * __restrict__ vx,       // Q4_0 weights
                                               const void * __restrict__ vy,       // Q8_1 activations
                                               float * __restrict__ dst,           // Output
                                               const int ncols_x,                  // K dimension
                                               const int nrows_x,                  // M dimension
                                               const int ncols_y,                  // N dimension (batch size)
                                               const int nrows_dst,                // Output row stride
                                               int8_t * __restrict__ slm_A,        // SLM for A tiles [2 × 4 × 8 × 32]
                                               int8_t * __restrict__ slm_B,        // SLM for B tiles [2 × 32 × 16]
                                               int32_t * __restrict__ slm_C,       // SLM for C tiles [4 × 8 × 16]
                                               float * __restrict__ slm_scales_A,  // SLM for A scales [2 × 4 × 8]
                                               float * __restrict__ slm_scales_B,  // SLM for B scales [2 × 16]
                                               float * __restrict__ slm_sums_B,    // SLM for B sums [2 × 16]
                                               const int tile_m,
                                               const bool use_tiled,
                                               sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = sg.get_group_id()[0];
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * MT_OUTPUT_ROWS + sg_id * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int Q8_1_BLOCK_SIZE = 36;

    // Per-sub-group C tile (not double-buffered, used for intermediate results)
    int32_t * my_slm_C = slm_C + sg_id * (XMX_M * XMX_N);

    // Private accumulator
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Lambda to load A tile into specified buffer
    auto load_A = [&](int k_block, int buf) {
        int8_t * buf_A        = slm_A + buf * (XMX_M * XMX_K * MT_TILES_M) + sg_id * (XMX_M * XMX_K);
        float *  buf_scales_A = slm_scales_A + buf * (XMX_M * MT_TILES_M) + sg_id * XMX_M;

        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                buf_scales_A[lane_id]  = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                int             base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed           = nibbles[j];
                    buf_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                    buf_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                }
            } else {
                buf_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    buf_A[base_idx + j] = 0;
                }
            }
        }
    };

    // Lambda to load B tile into specified buffer (only sub-group 0)
    auto load_B = [&](int k_block, int buf) {
        if (sg_id == 0) {
            int8_t * buf_B        = slm_B + buf * (XMX_K * XMX_N);
            float *  buf_scales_B = slm_scales_B + buf * XMX_N;
            float *  buf_sums_B   = slm_sums_B + buf * XMX_N;

            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                buf_scales_B[lane_id] = float(d);
                buf_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    buf_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                buf_scales_B[lane_id] = 0.0f;
                buf_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    buf_B[k + lane_id * XMX_K] = 0;
                }
            }
        }
    };

    // Lambda to compute using specified buffer
    auto compute = [&](int buf) {
        int8_t * buf_A        = slm_A + buf * (XMX_M * XMX_K * MT_TILES_M) + sg_id * (XMX_M * XMX_K);
        int8_t * buf_B        = slm_B + buf * (XMX_K * XMX_N);
        float *  buf_scales_A = slm_scales_A + buf * (XMX_M * MT_TILES_M) + sg_id * XMX_M;
        float *  buf_scales_B = slm_scales_B + buf * XMX_N;
        float *  buf_sums_B   = slm_sums_B + buf * XMX_N;

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(buf_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(buf_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i    = idx / XMX_N;
            int   j    = idx % XMX_N;
            float d_A  = buf_scales_A[i];
            float d_B  = buf_scales_B[j];
            float s_B  = buf_sums_B[j];
            float C_ij = float(my_slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
        }
    };

    // Prefetch first K block into buffer 0
    load_A(0, 0);
    load_B(0, 0);
    item.barrier(sycl::access::fence_space::local_space);

    // Main loop with double buffering
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int cur_buf  = k_block & 1;
        int next_buf = (k_block + 1) & 1;

        // Start loading next K block (if not last iteration)
        if (k_block + 1 < num_k_blocks) {
            load_A(k_block + 1, next_buf);
            load_B(k_block + 1, next_buf);
        }

        // Compute current K block
        compute(cur_buf);

        // Single barrier (loads and compute must finish before next iteration)
        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Q4_0 x Q8_1 XMX GEMM Kernel - Wide Tile Version
// Key optimization: Process 2 column tiles with 2 sub-groups in PARALLEL
// Each sub-group processes 8 rows × 16 cols, sharing the same loaded A tile
// This halves memory traffic for A while maintaining XMX parallelism
// =============================================================================

// Configuration for wide-tile kernel
// Process 2 column tiles (32 columns) per work-group with 2 parallel sub-groups
constexpr int CF_MAX_COL_TILES = 2;  // Process 2 column tiles (32 columns) per work-group
constexpr int CF_SG_COUNT      = 2;  // 2 sub-groups for parallel compute

void mul_mat_q4_0_q8_1_xmx_colfused_kernel(const void * __restrict__ vx,       // Q4_0 weights
                                           const void * __restrict__ vy,       // Q8_1 activations
                                           float * __restrict__ dst,           // Output
                                           const int ncols_x,                  // K dimension
                                           const int nrows_x,                  // M dimension
                                           const int ncols_y,                  // N dimension (batch size)
                                           const int nrows_dst,                // Output row stride
                                           int8_t * __restrict__ slm_A,        // SLM for A tile [8 × 32] - SHARED
                                           int8_t * __restrict__ slm_B,        // SLM for B tiles [2 × 32 × 16]
                                           int32_t * __restrict__ slm_C,       // SLM for C tiles [2 × 8 × 16]
                                           float * __restrict__ slm_scales_A,  // SLM for A scales [8] - SHARED
                                           float * __restrict__ slm_scales_B,  // SLM for B scales [2 × 16]
                                           float * __restrict__ slm_sums_B,    // SLM for B sums [2 × 16]
                                           const int tile_m,
                                           const bool use_tiled,
                                           sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = item.get_local_id(0);  // Which sub-group (0-1)
    const int  lane_id = sg.get_local_id()[0];  // Lane within sub-group (0-15)

    // Each work-group processes one row tile (8 rows) × 2 column tiles (32 cols)
    // Work-groups are tiled over row and column dimensions
    // Sub-group 0: column tile 0, Sub-group 1: column tile 1 within this work-group
    const int row_base  = item.get_group(0) * XMX_M;
    const int col_group = item.get_group(1);  // Which 32-column group
    const int col_base  = col_group * (CF_MAX_COL_TILES * XMX_N) + sg_id * XMX_N;

    if (row_base >= nrows_x) {
        return;
    }

    const int  num_k_blocks = ncols_x / XMX_K;
    const bool active_sg    = (col_base < ncols_y);

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int Q8_1_BLOCK_SIZE = 36;

    // Private accumulator for this sub-group's column tile
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Pointers to this sub-group's B tile and C tile in SLM
    int8_t *  my_B_tile   = slm_B + sg_id * (XMX_K * XMX_N);
    int32_t * my_C_tile   = slm_C + sg_id * (XMX_M * XMX_N);
    float *   my_scales_B = slm_scales_B + sg_id * XMX_N;
    float *   my_sums_B   = slm_sums_B + sg_id * XMX_N;

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // === COOPERATIVE LOAD PHASE ===
        // Sub-group 0 loads A tile (8 rows × 32 elements)
        // Both sub-groups load their own B tiles in parallel

        if (sg_id == 0 && lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id]  = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                int             base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed           = nibbles[j];
                    slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                    slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Each sub-group loads its own B tile
        if (active_sg) {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                my_scales_B[lane_id] = float(d);
                my_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                my_scales_B[lane_id] = 0.0f;
                my_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // === PARALLEL COMPUTE PHASE ===
        // Both sub-groups compute their column tiles using SHARED A
        if (active_sg) {
            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_B_tile);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_C_tile);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                       matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Accumulate with scale correction
#    pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                int   i    = idx / XMX_N;
                int   j    = idx % XMX_N;
                float d_A  = slm_scales_A[i];
                float d_B  = my_scales_B[j];
                float s_B  = my_sums_B[j];
                float C_ij = float(my_C_tile[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    // Write outputs - each sub-group writes its column tile
    if (active_sg) {
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int i   = idx / XMX_N;
            int j   = idx % XMX_N;
            int row = row_base + i;
            int col = col_base + j;

            if (row < nrows_x && col < ncols_y) {
                dst[col * nrows_dst + row] = acc[acc_idx];
            }
        }
    }
}

// =============================================================================
// Column-Fused Kernel Launcher
// Uses 2 sub-groups for parallel compute, each handling one column tile
// Both sub-groups share the same A tile to reduce cache misses
// =============================================================================
static void ggml_mul_mat_q4_0_q8_1_xmx_colfused_sycl(const void *    vx,
                                                     const void *    vy,
                                                     float *         dst,
                                                     const int       ncols_x,
                                                     const int       nrows_x,
                                                     const int       ncols_y,
                                                     const int       nrows_dst,
                                                     const int       tile_m,
                                                     const bool      use_tiled,
                                                     dpct::queue_ptr stream) {
    // Grid: work-groups for row tiles × column tile groups
    // Each work-group handles 8 rows × 32 columns (2 sub-groups × 16 cols each)
    const int num_row_tiles  = (nrows_x + XMX_M - 1) / XMX_M;
    const int cols_per_wg    = CF_MAX_COL_TILES * XMX_N;  // 32 columns per work-group
    const int num_col_groups = (ncols_y + cols_per_wg - 1) / cols_per_wg;

    // 2D block: (2 sub-groups, 16 threads per sub-group) = 32 threads
    sycl::range<2> grid(num_row_tiles, num_col_groups);
    sycl::range<2> block(CF_SG_COUNT, XMX_SG_SIZE);  // 2 × 16 = 32 threads

    GGML_SYCL_DEBUG("[XMX-CF] Launching column-fused kernel: grid=(%d,%d), block=(%d,%d), M=%d, N=%d, K=%d\n",
                    num_row_tiles, num_col_groups, CF_SG_COUNT, XMX_SG_SIZE, nrows_x, ncols_y, ncols_x);

    // SLM layout for column-fused kernel (2 sub-groups, each with own B/C tiles, sharing A):
    // - A tile: 8×32 = 256 bytes (SHARED between sub-groups)
    // - B tiles: 2×32×16 = 1024 bytes (one per sub-group)
    // - C tiles: 2×8×16×4 = 1024 bytes (one per sub-group)
    // - A scales: 8×4 = 32 bytes (SHARED)
    // - B scales: 2×16×4 = 128 bytes
    // - B sums: 2×16×4 = 128 bytes
    constexpr int CF_SLM_A_SIZE        = XMX_M * XMX_K;                           // 256 bytes
    constexpr int CF_SLM_B_SIZE        = CF_MAX_COL_TILES * XMX_K * XMX_N;        // 1024 bytes
    constexpr int CF_SLM_C_SIZE        = CF_MAX_COL_TILES * XMX_M * XMX_N * 4;    // 1024 bytes
    constexpr int CF_SLM_SCALES_A_SIZE = XMX_M * 4;                               // 32 bytes
    constexpr int CF_SLM_SCALES_B_SIZE = CF_MAX_COL_TILES * XMX_N * 4;            // 128 bytes
    constexpr int CF_SLM_SUMS_B_SIZE   = CF_MAX_COL_TILES * XMX_N * 4;            // 128 bytes
    constexpr int CF_TOTAL_SLM_SIZE    = CF_SLM_A_SIZE + CF_SLM_B_SIZE + CF_SLM_C_SIZE + CF_SLM_SCALES_A_SIZE +
                                      CF_SLM_SCALES_B_SIZE + CF_SLM_SUMS_B_SIZE;  // ~2.6KB

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(CF_TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * slm_A  = reinterpret_cast<int8_t *>(shared + offset);
                           offset += CF_SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += CF_SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += CF_SLM_C_SIZE;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += CF_SLM_SCALES_A_SIZE;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += CF_SLM_SCALES_B_SIZE;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_0_q8_1_xmx_colfused_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst,
                                                                 slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                                 slm_sums_B, tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Column-Fused 4-Tile Sequential Kernel
// Single sub-group processes 4 column tiles sequentially, maximizing A tile reuse
// Better for batch 33-64 where 2-tile parallel has poor efficiency
// =============================================================================
constexpr int CF4_MAX_COL_TILES = 4;  // Process 4 column tiles (64 columns) per work-group

void mul_mat_q4_0_q8_1_xmx_colfused_4tile_kernel(const void * __restrict__ vx,
                                                 const void * __restrict__ vy,
                                                 float * __restrict__ dst,
                                                 const int ncols_x,
                                                 const int nrows_x,
                                                 const int ncols_y,
                                                 const int nrows_dst,
                                                 int8_t * __restrict__ slm_A,
                                                 int8_t * __restrict__ slm_B,
                                                 int32_t * __restrict__ slm_C,
                                                 float * __restrict__ slm_scales_A,
                                                 float * __restrict__ slm_scales_B,
                                                 float * __restrict__ slm_sums_B,
                                                 const int tile_m,
                                                 const bool use_tiled,
                                                 sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base       = item.get_group(0) * XMX_M;
    const int col_group      = item.get_group(1);
    const int col_group_base = col_group * (CF4_MAX_COL_TILES * XMX_N);

    if (row_base >= nrows_x) {
        return;
    }

    const int num_k_blocks             = ncols_x / XMX_K;
    const int num_col_tiles_this_group = sycl::min(CF4_MAX_COL_TILES, (ncols_y - col_group_base + XMX_N - 1) / XMX_N);

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_0_BLOCK_SIZE = 18;
    constexpr int Q8_1_BLOCK_SIZE = 36;

    // Private accumulators for all column tiles
    float acc[CF4_MAX_COL_TILES][8];
#    pragma unroll
    for (int t = 0; t < CF4_MAX_COL_TILES; t++) {
#    pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[t][i] = 0.0f;
        }
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // Load A tile (shared across all column tiles)
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                sycl::half   d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id]  = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                int             base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed           = nibbles[j];
                    slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                    slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        sycl::group_barrier(sg);

        // Process all column tiles sequentially
        for (int col_tile = 0; col_tile < num_col_tiles_this_group; col_tile++) {
            int col_base = col_group_base + col_tile * XMX_N;
            int col      = col_base + lane_id;

            // Load B tile
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }

            sycl::group_barrier(sg);

            // XMX compute
            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                       matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Accumulate with scale correction
#    pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                int   i    = idx / XMX_N;
                int   j    = idx % XMX_N;
                float d_A  = slm_scales_A[i];
                float d_B  = slm_scales_B[j];
                float s_B  = slm_sums_B[j];
                float C_ij = float(slm_C[idx]);
                acc[col_tile][acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
            }
        }

        sycl::group_barrier(sg);
    }

    // Write outputs for all column tiles
    for (int col_tile = 0; col_tile < num_col_tiles_this_group; col_tile++) {
        int col_base = col_group_base + col_tile * XMX_N;

#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int i   = idx / XMX_N;
            int j   = idx % XMX_N;
            int row = row_base + i;
            int col = col_base + j;

            if (row < nrows_x && col < ncols_y) {
                dst[col * nrows_dst + row] = acc[col_tile][acc_idx];
            }
        }
    }
}

// 4-tile sequential launcher
static void ggml_mul_mat_q4_0_q8_1_xmx_colfused_4tile_sycl(const void *    vx,
                                                          const void *    vy,
                                                          float *         dst,
                                                          const int       ncols_x,
                                                          const int       nrows_x,
                                                          const int       ncols_y,
                                                          const int       nrows_dst,
                                                          const int       tile_m,
                                                          const bool      use_tiled,
                                                          dpct::queue_ptr stream) {
    const int num_row_tiles  = (nrows_x + XMX_M - 1) / XMX_M;
    const int cols_per_wg    = CF4_MAX_COL_TILES * XMX_N;  // 64 columns per work-group
    const int num_col_groups = (ncols_y + cols_per_wg - 1) / cols_per_wg;

    sycl::range<2> grid(num_row_tiles, num_col_groups);
    sycl::range<2> block(1, XMX_SG_SIZE);  // Single sub-group

    GGML_SYCL_DEBUG("[XMX-CF4] Launching 4-tile sequential kernel: grid=(%d,%d), block=(1,%d), M=%d, N=%d, K=%d\n",
                    num_row_tiles, num_col_groups, XMX_SG_SIZE, nrows_x, ncols_y, ncols_x);

    // SLM for single-tile processing (tiles reused sequentially)
    constexpr int CF4_SLM_A_SIZE        = XMX_M * XMX_K;      // 256 bytes
    constexpr int CF4_SLM_B_SIZE        = XMX_K * XMX_N;      // 512 bytes
    constexpr int CF4_SLM_C_SIZE        = XMX_M * XMX_N * 4;  // 512 bytes
    constexpr int CF4_SLM_SCALES_A_SIZE = XMX_M * 4;          // 32 bytes
    constexpr int CF4_SLM_SCALES_B_SIZE = XMX_N * 4;          // 64 bytes
    constexpr int CF4_SLM_SUMS_B_SIZE   = XMX_N * 4;          // 64 bytes
    constexpr int CF4_TOTAL_SLM_SIZE    = CF4_SLM_A_SIZE + CF4_SLM_B_SIZE + CF4_SLM_C_SIZE + CF4_SLM_SCALES_A_SIZE +
                                       CF4_SLM_SCALES_B_SIZE + CF4_SLM_SUMS_B_SIZE;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(CF4_TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * slm_A  = reinterpret_cast<int8_t *>(shared + offset);
                           offset += CF4_SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += CF4_SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += CF4_SLM_C_SIZE;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += CF4_SLM_SCALES_A_SIZE;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += CF4_SLM_SCALES_B_SIZE;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_0_q8_1_xmx_colfused_4tile_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y,
                                                                       nrows_dst, slm_A, slm_B, slm_C, slm_scales_A,
                                                                       slm_scales_B, slm_sums_B, tile_m, use_tiled,
                                                                       item);
                       });
    });
}

// =============================================================================
// Q8_0 x Q8_1 XMX GEMM Kernel - Single-Tile
// Simpler than Q4_0: int8 values stored directly, no unpacking needed
// =============================================================================
void mul_mat_q8_0_q8_1_xmx_kernel(const void * __restrict__ vx,       // Q8_0 weights [nrows_x, ncols_x/32 blocks]
                                  const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,           // Output [nrows_x, ncols_y]
                                  const int ncols_x,                  // K dimension (must be multiple of 32)
                                  const int nrows_x,                  // M dimension (rows of weights to process)
                                  const int ncols_y,                  // N dimension (batch size / columns of output)
                                  const int nrows_dst,                // Output row stride
                                  int8_t * __restrict__ slm_A,        // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,        // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,       // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    // Work-group position (one XMX tile per work-group)
    const int row_base = item.get_group(0) * XMX_M;  // 8 rows
    const int col_base = item.get_group(1) * XMX_N;  // 16 cols

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    // Block pointers
    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q8_0_BLOCK_SIZE = 34;  // sizeof(block_q8_0) = 2 + 32
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 2 + 2 + 32

    // Private accumulator - 8 elements per lane (128 total / 16 lanes)
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // ============== Load A: 8 rows × 32 int8 values (no unpacking!) ==============
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;

                // Load scale
                sycl::half d          = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                // Load int8 values directly (no unpacking needed!)
                const int8_t * qs       = reinterpret_cast<const int8_t *>(block_ptr + 2);
                int            base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                // Zero padding for out-of-bounds rows
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // ============== Load B: 16 columns × 32 int8 values ==============
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);

// Store column-major for XMX B matrix
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        // Barrier to ensure all loading is complete
        item.barrier(sycl::access::fence_space::local_space);

        // ============== XMX Multiply ==============
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// ============== Apply scales and accumulate ==============
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int i = idx / XMX_N;
            int j = idx % XMX_N;
            acc[acc_idx] += slm_scales_A[i] * slm_scales_B[j] * float(slm_C[idx]);
        }

        // Barrier before next K block
        item.barrier(sycl::access::fence_space::local_space);
    }

// ============== Write output ==============
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Q8_0 x Q8_1 XMX GEMM Kernel - Large Tile Version (512 threads)
// Uses 32 sub-groups (8×4) to process 64×64 output tile = 4096 outputs
// Key advantage: No nibble unpacking needed - direct int8 values!
// =============================================================================

// Large-tile configuration for Q8_0 (same as Q4_0)
constexpr int Q8_LT_TILES_M     = 8;                              // 8 row tiles per work-group
constexpr int Q8_LT_TILES_N     = 4;                              // 4 column tiles per work-group
constexpr int Q8_LT_SG_COUNT    = Q8_LT_TILES_M * Q8_LT_TILES_N;  // 32 sub-groups per work-group
constexpr int Q8_LT_WG_SIZE     = Q8_LT_SG_COUNT * XMX_SG_SIZE;   // 512 threads per work-group
constexpr int Q8_LT_OUTPUT_ROWS = Q8_LT_TILES_M * XMX_M;          // 64 rows per work-group
constexpr int Q8_LT_OUTPUT_COLS = Q8_LT_TILES_N * XMX_N;          // 64 cols per work-group

// Large-tile SLM sizes for Q8_0
constexpr int Q8_LT_SLM_A_SIZE        = XMX_M * XMX_K * Q8_LT_TILES_M;   // 8×32×8 = 2048 bytes
constexpr int Q8_LT_SLM_B_SIZE        = XMX_K * XMX_N * Q8_LT_TILES_N;   // 32×16×4 = 2048 bytes
constexpr int Q8_LT_SLM_SCALES_A_SIZE = XMX_M * Q8_LT_TILES_M;           // 8×8 = 64 floats = 256 bytes
constexpr int Q8_LT_SLM_SCALES_B_SIZE = XMX_N * Q8_LT_TILES_N;           // 16×4 = 64 floats = 256 bytes
constexpr int Q8_LT_SLM_C_SIZE        = XMX_M * XMX_N * Q8_LT_SG_COUNT;  // 8×16×32 = 4096 int32 = 16384 bytes

// Total SLM: ~21KB

void mul_mat_q8_0_q8_1_xmx_largetile_kernel(const void * __restrict__ vx,       // Q8_0 weights
                                            const void * __restrict__ vy,       // Q8_1 activations
                                            float * __restrict__ dst,           // Output
                                            const int ncols_x,                  // K dimension
                                            const int nrows_x,                  // M dimension
                                            const int ncols_y,                  // N dimension (batch size)
                                            const int nrows_dst,                // Output row stride
                                            int8_t * __restrict__ slm_A,        // SLM for A tiles [8 × 8 × 32]
                                            int8_t * __restrict__ slm_B,        // SLM for B tiles [4 × 32 × 16]
                                            int32_t * __restrict__ slm_C,       // SLM for C tiles [32 × 8 × 16]
                                            float * __restrict__ slm_scales_A,  // SLM for A scales [8 × 8]
                                            float * __restrict__ slm_scales_B,  // SLM for B scales [4 × 16]
                                            const int tile_m,
                                            const bool use_tiled,
                                            sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = sg.get_group_id()[0];  // Which sub-group (0-31)
    const int  lane_id = sg.get_local_id()[0];  // Lane within sub-group (0-15)

    // Map sub-group ID to 2D tile position
    const int sg_row = sg_id / Q8_LT_TILES_N;  // Row tile within work-group (0-7)
    const int sg_col = sg_id % Q8_LT_TILES_N;  // Col tile within work-group (0-3)

    // Work-group position
    const int wg_row_base = item.get_group(0) * Q8_LT_OUTPUT_ROWS;
    const int wg_col_base = item.get_group(1) * Q8_LT_OUTPUT_COLS;

    // This sub-group's tile position
    const int row_base = wg_row_base + sg_row * XMX_M;
    const int col_base = wg_col_base + sg_col * XMX_N;

    if (wg_row_base >= nrows_x) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q8_0_BLOCK_SIZE = 34;  // sizeof(block_q8_0) = 2 + 32
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 2 + 2 + 32

    // Pointers to this sub-group's tiles in SLM
    int8_t *  my_slm_A        = slm_A + sg_row * (XMX_M * XMX_K);
    int8_t *  my_slm_B        = slm_B + sg_col * (XMX_K * XMX_N);
    int32_t * my_slm_C        = slm_C + sg_id * (XMX_M * XMX_N);  // Each sub-group has its own 8×16 C tile
    float *   my_slm_scales_A = slm_scales_A + sg_row * XMX_M;
    float *   my_slm_scales_B = slm_scales_B + sg_col * XMX_N;

    // Private accumulator (8 elements per lane)
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Process each K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // ============== Phase 1: Load A tiles (only first col tile per row loads) ==============
        // A tiles are shared by all sub-groups in the same row tile
        // Only sg_col == 0 and lane_id < 8 loads (matching Q4_0 pattern)
        if (sg_col == 0 && lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;
                sycl::half   d           = *reinterpret_cast<const sycl::half *>(block_ptr);
                my_slm_scales_A[lane_id] = float(d);

                // Load 32 int8 values directly (no unpacking needed for Q8_0!)
                const int8_t * qs       = reinterpret_cast<const int8_t *>(block_ptr + 2);
                int            base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    my_slm_A[base_idx + j] = qs[j];
                }
            } else {
                my_slm_scales_A[lane_id] = 0.0f;
                int base_idx             = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    my_slm_A[base_idx + j] = 0;
                }
            }
        }

        // ============== Phase 2: Load B tiles (only first row tile per col loads) ==============
        // B tiles are shared by all sub-groups in the same col tile
        // Only sg_row == 0 loads to avoid race conditions
        if (sg_row == 0) {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr   = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half   d           = *reinterpret_cast<const sycl::half *>(block_ptr);
                my_slm_scales_B[lane_id] = float(d);

                // Load 32 int8 values directly
                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                my_slm_scales_B[lane_id] = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        // Barrier to ensure all data is loaded
        item.barrier(sycl::access::fence_space::local_space);

        // ============== Phase 3: XMX Compute ==============
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(my_slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// ============== Phase 4: Apply scales and accumulate ==============
// Use correct indexed pattern matching single-tile kernel
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int i = idx / XMX_N;  // row within tile
            int j = idx % XMX_N;  // col within tile
            // Q8_0 x Q8_1: just multiply scales (no zero-point offset like Q4_0)
            acc[acc_idx] += my_slm_scales_A[i] * my_slm_scales_B[j] * float(my_slm_C[idx]);
        }

        // Barrier before next K block
        item.barrier(sycl::access::fence_space::local_space);
    }

// ============== Write output ==============
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Q4_1 x Q8_1 XMX GEMM Kernel - Single-Tile
// Similar to Q4_0 but with min offset: x = d * q + m
// =============================================================================
void mul_mat_q4_1_q8_1_xmx_kernel(const void * __restrict__ vx,       // Q4_1 weights [nrows_x, ncols_x/32 blocks]
                                  const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,           // Output [nrows_x, ncols_y]
                                  const int ncols_x,                  // K dimension (must be multiple of 32)
                                  const int nrows_x,                  // M dimension (rows of weights to process)
                                  const int ncols_y,                  // N dimension (batch size / columns of output)
                                  const int nrows_dst,                // Output row stride
                                  int8_t * __restrict__ slm_A,        // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,        // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,       // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  float * __restrict__ slm_mins_A,    // SLM for A mins [8]
                                  float * __restrict__ slm_sums_B,    // SLM for B sums [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q4_1_BLOCK_SIZE = 20;  // sizeof(block_q4_1) = 4 + 16
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 4 + 32

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // Load A (Q4_1): unpack nibbles to [0,15] range, store as int8
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q4_1_BLOCK_SIZE;

                sycl::half d          = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half m          = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                slm_scales_A[lane_id] = float(d);
                slm_mins_A[lane_id]   = float(m);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 4);
                int             base_idx = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed           = nibbles[j];
                    // Q4_1: values are unsigned [0,15], keep as-is
                    // Q4_1 storage: byte[j] low nibble = K pos j, high nibble = K pos j+16
                    slm_A[base_idx + j]      = (int8_t) (packed & 0x0F);
                    slm_A[base_idx + j + 16] = (int8_t) (packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                slm_mins_A[lane_id]   = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1): int8 values + scale + sum
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);  // s = d * sum(qs)

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply scales with Q4_1 formula: result = d_A * d_B * dot + m_A * s_B
// where s_B = d_B * sum(qs_B)
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i   = idx / XMX_N;
            int   j   = idx % XMX_N;
            float d_A = slm_scales_A[i];
            float d_B = slm_scales_B[j];
            float m_A = slm_mins_A[i];
            float s_B = slm_sums_B[j];
            acc[acc_idx] += d_A * d_B * float(slm_C[idx]) + m_A * s_B;
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Launcher for Q4_1 x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q4_1_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q4_1 kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    // Additional SLM for mins and sums
    constexpr int TOTAL_SLM_SIZE = SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
                                   SLM_SCALES_B_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
                                   SLM_SCALES_B_SIZE * 4;  // mins_A, sums_B

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * A_mins = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_sums = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_1_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, A_mins, B_sums, tile_m,
                                                        use_tiled, item);
                       });
    });
}

// =============================================================================
// Q5_0 x Q8_1 XMX GEMM Kernel - Single-Tile
// Q5_0: 5-bit quantization with 4-bit nibbles + 5th bit in qh array
// Block: 22 bytes = 2 (d) + 4 (qh) + 16 (qs)
// Dequant: x = d * (q - 16) where q is 5-bit [0,31]
// =============================================================================
void mul_mat_q5_0_q8_1_xmx_kernel(const void * __restrict__ vx,       // Q5_0 weights [nrows_x, ncols_x/32 blocks]
                                  const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,           // Output [nrows_x, ncols_y]
                                  const int ncols_x,                  // K dimension (must be multiple of 32)
                                  const int nrows_x,                  // M dimension
                                  const int ncols_y,                  // N dimension
                                  const int nrows_dst,                // Output row stride
                                  int8_t * __restrict__ slm_A,        // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,        // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,       // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q5_0_BLOCK_SIZE = 22;  // sizeof(block_q5_0) = 2 + 4 + 16
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 4 + 32

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // Load A (Q5_0): unpack 5-bit values to signed int8 [-16, 15]
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q5_0_BLOCK_SIZE;

                sycl::half d          = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                // Load 5th bits as uint32
                uint32_t qh;
                memcpy(&qh, block_ptr + 2, sizeof(qh));

                const uint8_t * qs       = reinterpret_cast<const uint8_t *>(block_ptr + 6);
                int             base_idx = lane_id * XMX_K;

// Unpack: nibbles + 5th bit, then subtract 16 to center
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed = qs[j];

                    // Lower nibble (elements 0-15): 5th bit at qh bit j
                    int xh_0 = ((qh >> j) & 1) << 4;
                    int q0   = (packed & 0x0F) | xh_0;

                    // Upper nibble (elements 16-31): 5th bit at qh bit (j+16)
                    int xh_1 = ((qh >> (j + 16)) & 1) << 4;
                    int q1   = (packed >> 4) | xh_1;

                    // Subtract 16 to center: [0,31] → [-16,15]
                    slm_A[base_idx + j]      = (int8_t) (q0 - 16);
                    slm_A[base_idx + j + 16] = (int8_t) (q1 - 16);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1): int8 values + scale
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply Q5_0 scales: result = d_A * d_B * int_result
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i   = idx / XMX_N;
            int   j   = idx % XMX_N;
            float d_A = slm_scales_A[i];
            float d_B = slm_scales_B[j];
            acc[acc_idx] += d_A * d_B * float(slm_C[idx]);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Launcher for Q5_0 x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q5_0_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q5_0 kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q5_0_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Q5_1 x Q8_1 XMX GEMM Kernel - Single-Tile
// Q5_1: 5-bit quantization with min offset
// Block: 24 bytes = 2 (d) + 2 (m) + 4 (qh) + 16 (qs)
// Dequant: x = d * q + m where q is 5-bit [0,31]
// Formula: result = d_A * d_B * dot + m_A * s_B
// =============================================================================
void mul_mat_q5_1_q8_1_xmx_kernel(const void * __restrict__ vx,       // Q5_1 weights [nrows_x, ncols_x/32 blocks]
                                  const void * __restrict__ vy,       // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,           // Output [nrows_x, ncols_y]
                                  const int ncols_x,                  // K dimension (must be multiple of 32)
                                  const int nrows_x,                  // M dimension
                                  const int ncols_y,                  // N dimension
                                  const int nrows_dst,                // Output row stride
                                  int8_t * __restrict__ slm_A,        // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,        // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,       // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  float * __restrict__ slm_mins_A,    // SLM for A mins [8]
                                  float * __restrict__ slm_sums_B,    // SLM for B sums [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char *  src0            = (const char *) vx;
    const char *  src1            = (const char *) vy;
    constexpr int Q5_1_BLOCK_SIZE = 24;  // sizeof(block_q5_1) = 2 + 2 + 4 + 16
    constexpr int Q8_1_BLOCK_SIZE = 36;  // sizeof(block_q8_1) = 4 + 32

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // Load A (Q5_1): unpack 5-bit values to unsigned int8 [0,31]
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, k_block, num_k_blocks, tile_m, use_tiled);
                const char * block_ptr = src0 + block_idx * Q5_1_BLOCK_SIZE;

                sycl::half d          = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half m          = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                slm_scales_A[lane_id] = float(d);
                slm_mins_A[lane_id]   = float(m);

                // Load 5th bits as uint32
                uint32_t qh;
                memcpy(&qh, block_ptr + 4, sizeof(qh));

                const uint8_t * qs       = reinterpret_cast<const uint8_t *>(block_ptr + 8);
                int             base_idx = lane_id * XMX_K;

// Unpack: nibbles + 5th bit (unsigned, no centering for Q5_1)
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    uint8_t packed = qs[j];

                    // Lower nibble (elements 0-15): 5th bit at qh bit j
                    int xh_0 = ((qh >> j) & 1) << 4;
                    int q0   = (packed & 0x0F) | xh_0;

                    // Upper nibble (elements 16-31): 5th bit at qh bit (j+16)
                    int xh_1 = ((qh >> (j + 16)) & 1) << 4;
                    int q1   = (packed >> 4) | xh_1;

                    // Store unsigned [0,31] for Q5_1
                    slm_A[base_idx + j]      = (int8_t) q0;
                    slm_A[base_idx + j + 16] = (int8_t) q1;
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                slm_mins_A[lane_id]   = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1): int8 values + scale + sum
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply Q5_1 formula: result = d_A * d_B * dot + m_A * s_B
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i   = idx / XMX_N;
            int   j   = idx % XMX_N;
            float d_A = slm_scales_A[i];
            float d_B = slm_scales_B[j];
            float m_A = slm_mins_A[i];
            float s_B = slm_sums_B[j];
            acc[acc_idx] += d_A * d_B * float(slm_C[idx]) + m_A * s_B;
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output (column-major: dst[col * nrows_dst + row])
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Launcher for Q5_1 x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q5_1_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q5_1 kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    constexpr int TOTAL_SLM_SIZE = SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
                                   SLM_SCALES_B_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * A_mins = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_sums = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q5_1_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, A_mins, B_sums, tile_m,
                                                        use_tiled, item);
                       });
    });
}

// =============================================================================
// Q3_K x Q8_1 XMX GEMM Kernel - Single-Tile
// Super-block: 256 elements = 8 blocks of 32 elements each
// 3-bit quantization with complex bit packing
// =============================================================================
void mul_mat_q3_K_q8_1_xmx_kernel(const void * __restrict__ vx,  // Q3_K weights [nrows_x, ncols_x/256 super-blocks]
                                  const void * __restrict__ vy,  // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,      // Output [nrows_x, ncols_y]
                                  const int ncols_x,             // K dimension (must be multiple of 256)
                                  const int nrows_x,             // M dimension
                                  const int ncols_y,             // N dimension
                                  const int nrows_dst,           // Output row stride
                                  int8_t * __restrict__ slm_A,   // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,   // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,  // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    // Q3_K super-block: 256 elements
    // Layout: ggml_half d (2) + qs[64] + hmask[32] + scales[12]
    constexpr int Q3_K_SB_SIZE    = 256;
    constexpr int Q3_K_BLOCK_SIZE = 2 + Q3_K_SB_SIZE / 4 + Q3_K_SB_SIZE / 8 + 12;  // 110 bytes
    constexpr int Q8_1_BLOCK_SIZE = 36;

    const int num_super_blocks = ncols_x / Q3_K_SB_SIZE;
    const int num_k_blocks     = ncols_x / XMX_K;

    const char * src0 = (const char *) vx;
    const char * src1 = (const char *) vy;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int super_block_idx    = k_block / 8;
        const int internal_block_idx = k_block % 8;

        // Load A (Q3_K)
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, super_block_idx, num_super_blocks, tile_m, use_tiled);
                const char * sb_ptr = src0 + block_idx * Q3_K_BLOCK_SIZE;

                // Super-block scale
                sycl::half d   = *reinterpret_cast<const sycl::half *>(sb_ptr);
                float      d_f = float(d);

                // qs: 2-bit values packed (64 bytes for 256 elements, 4 elements per byte)
                const uint8_t * qs     = reinterpret_cast<const uint8_t *>(sb_ptr + 2);
                // hmask: high bits (32 bytes for 256 elements)
                const uint8_t * hmask  = reinterpret_cast<const uint8_t *>(sb_ptr + 2 + 64);
                // scales: 12 bytes packed 6-bit scales
                const uint8_t * scales = reinterpret_cast<const uint8_t *>(sb_ptr + 2 + 64 + 32);

                // Q3_K has 16 sub-blocks (16 elements each), 16 6-bit scales packed in 12 bytes
                // For XMX K=32, each internal block covers 2 sub-blocks
                // We extract both scales and average them
                int n       = internal_block_idx / 4;  // which 128-element half (0 or 1)
                int j_inner = internal_block_idx % 4;  // which 32-element group (0-3)

                // Sub-block indices for this internal block: is0 and is1
                // Formula: is = 8*n + 2*j + is0 where is0 is 0 or 1
                int is0 = 8 * n + 2 * j_inner;      // first 16 elements
                int is1 = 8 * n + 2 * j_inner + 1;  // second 16 elements

                // Lambda to extract 6-bit scale from packed 12-byte array
                auto extract_scale = [&](int is) -> int8_t {
                    if (is < 4) {
                        return (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4);
                    } else if (is < 8) {
                        return (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4);
                    } else if (is < 12) {
                        return (scales[is - 8] >> 4) | (((scales[is] >> 4) & 3) << 4);
                    } else {
                        return (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
                    }
                };

                int8_t us0 = extract_scale(is0);
                int8_t us1 = extract_scale(is1);

                // Average the two scales (this introduces some error but simplifies XMX usage)
                float dl0             = d_f * (float) (us0 - 32);
                float dl1             = d_f * (float) (us1 - 32);
                float dl_avg          = (dl0 + dl1) * 0.5f;
                slm_scales_A[lane_id] = dl_avg;

                // Unpack 3-bit values for this internal block
                // Q3_K layout from dequantize_block_q3_K:
                // - n = tid / 4 = which 128-element half
                // - j = tid - 4*n = which 32-element group within the half
                // - q = qs + 32*n (32 bytes of qs for this half)
                // - shift = 2*j (which 2-bit pair in each byte)
                // - hm = hmask (32 bytes)
                // - m = 1 << (4*n + j) = 1 << internal_block_idx

                int base_idx = lane_id * XMX_K;
                int qs_base  = 32 * n;       // 0 or 32
                int shift    = 2 * j_inner;  // 0, 2, 4, or 6

#    pragma unroll
                for (int k = 0; k < 32; k++) {
                    // Extract 2-bit value from qs
                    // For element k, read from qs[qs_base + k], shift by 'shift', mask 2 bits
                    int q2 = (qs[qs_base + k] >> shift) & 3;

                    // Extract hmask bit for this element
                    // m = 1 << internal_block_idx, check if hmask[k] & m
                    int hmask_bit = (hmask[k] >> internal_block_idx) & 1;

                    // Apply hmask: if bit is 1, value = q2; if bit is 0, value = q2 - 4
                    // Range: hmask=1 -> [0,3], hmask=0 -> [-4,-1]
                    int value           = hmask_bit ? q2 : (q2 - 4);
                    slm_A[base_idx + k] = (int8_t) value;
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1)
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply scales
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i       = idx / XMX_N;
            int   j       = idx % XMX_N;
            float scale_A = slm_scales_A[i];
            float d_B     = slm_scales_B[j];
            acc[acc_idx] += scale_A * d_B * float(slm_C[idx]);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Launcher for Q3_K x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q3_K_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q3_K kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q3_K_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Q5_K x Q8_1 XMX GEMM Kernel - Single-Tile
// Super-block: 256 elements = 8 blocks of 32 elements each
// 5-bit quantization with nibbles + high bits + scales/mins
// =============================================================================
void mul_mat_q5_K_q8_1_xmx_kernel(const void * __restrict__ vx,  // Q5_K weights [nrows_x, ncols_x/256 super-blocks]
                                  const void * __restrict__ vy,  // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,      // Output [nrows_x, ncols_y]
                                  const int ncols_x,             // K dimension (must be multiple of 256)
                                  const int nrows_x,             // M dimension
                                  const int ncols_y,             // N dimension
                                  const int nrows_dst,           // Output row stride
                                  int8_t * __restrict__ slm_A,   // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,   // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,  // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  float * __restrict__ slm_mins_A,    // SLM for A mins [8]
                                  float * __restrict__ slm_sums_B,    // SLM for B sums [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    // Q5_K super-block: 256 elements
    // Layout: ggml_half2 dm (4) + scales[12] + qh[32] + qs[128]
    constexpr int Q5_K_SB_SIZE     = 256;
    constexpr int Q5_K_SCALE_BYTES = 12;
    constexpr int Q5_K_BLOCK_SIZE  = 4 + Q5_K_SCALE_BYTES + Q5_K_SB_SIZE / 8 + Q5_K_SB_SIZE / 2;  // 176 bytes
    constexpr int Q8_1_BLOCK_SIZE  = 36;

    const int num_super_blocks = ncols_x / Q5_K_SB_SIZE;
    const int num_k_blocks     = ncols_x / XMX_K;

    const char * src0 = (const char *) vx;
    const char * src1 = (const char *) vy;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int super_block_idx    = k_block / 8;
        const int internal_block_idx = k_block % 8;

        // Load A (Q5_K)
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, super_block_idx, num_super_blocks, tile_m, use_tiled);
                const char * sb_ptr = src0 + block_idx * Q5_K_BLOCK_SIZE;

                // Super-block scales (d and dmin)
                sycl::half d      = *reinterpret_cast<const sycl::half *>(sb_ptr);
                sycl::half dmin   = *reinterpret_cast<const sycl::half *>(sb_ptr + 2);
                float      d_f    = float(d);
                float      dmin_f = float(dmin);

                // Packed scales (same as Q4_K)
                const uint8_t * scales = reinterpret_cast<const uint8_t *>(sb_ptr + 4);

                // Extract 6-bit scale and min
                int j = internal_block_idx;
                int sc, m;
                if (j < 4) {
                    sc = scales[j] & 63;
                    m  = scales[j + 4] & 63;
                } else {
                    sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                    m  = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }

                slm_scales_A[lane_id] = d_f * (float) sc;
                slm_mins_A[lane_id]   = dmin_f * (float) m;

                // qh: 5th bits (32 bytes)
                const uint8_t * qh = reinterpret_cast<const uint8_t *>(sb_ptr + 4 + Q5_K_SCALE_BYTES);
                // qs: 4-bit nibbles (128 bytes)
                const uint8_t * qs = reinterpret_cast<const uint8_t *>(sb_ptr + 4 + Q5_K_SCALE_BYTES + 32);

                // Similar layout to Q4_K but with 5th bits in qh
                int il          = internal_block_idx / 2;
                int use_upper   = internal_block_idx % 2;
                int byte_offset = 32 * il;
                int base_idx    = lane_id * XMX_K;

#    pragma unroll
                for (int k = 0; k < 32; k++) {
                    uint8_t packed = qs[byte_offset + k];

                    // Get 5th bit from qh
                    int qh_byte_idx   = il * 8 + k / 4;
                    int qh_bit_offset = (k % 4) * 2 + use_upper;
                    int h             = (qh[qh_byte_idx] >> qh_bit_offset) & 1;

                    int q;
                    if (use_upper) {
                        q = (packed >> 4) | (h << 4);
                    } else {
                        q = (packed & 0x0F) | (h << 4);
                    }

                    // Store as unsigned [0,31] for XMX
                    slm_A[base_idx + k] = (int8_t) q;
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                slm_mins_A[lane_id]   = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1)
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply Q5_K formula: result = scale_A * d_B * dot - min_A * s_B
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i       = idx / XMX_N;
            int   j       = idx % XMX_N;
            float scale_A = slm_scales_A[i];
            float d_B     = slm_scales_B[j];
            float min_A   = slm_mins_A[i];
            float s_B     = slm_sums_B[j];
            acc[acc_idx] += scale_A * d_B * float(slm_C[idx]) - min_A * s_B;
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Launcher for Q5_K x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q5_K_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q5_K kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    constexpr int TOTAL_SLM_SIZE_Q5K = SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
                                       SLM_SCALES_B_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE_Q5K), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * A_mins = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_sums = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q5_K_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, A_mins, B_sums, tile_m,
                                                        use_tiled, item);
                       });
    });
}

// =============================================================================
// Q4_K x Q8_1 XMX GEMM Kernel - Single-Tile
// Super-block: 256 elements = 8 blocks of 32 elements each
// Each 32-element block has its own 6-bit scale and min
// =============================================================================
void mul_mat_q4_K_q8_1_xmx_kernel(const void * __restrict__ vx,  // Q4_K weights [nrows_x, ncols_x/256 super-blocks]
                                  const void * __restrict__ vy,  // Q8_1 activations [ncols_y, ncols_x/32 blocks]
                                  float * __restrict__ dst,      // Output [nrows_x, ncols_y]
                                  const int ncols_x,             // K dimension (must be multiple of 256)
                                  const int nrows_x,             // M dimension
                                  const int ncols_y,             // N dimension
                                  const int nrows_dst,           // Output row stride
                                  int8_t * __restrict__ slm_A,   // SLM for A tile [8 × 32]
                                  int8_t * __restrict__ slm_B,   // SLM for B tile [32 × 16]
                                  int32_t * __restrict__ slm_C,  // SLM for C tile [8 × 16]
                                  float * __restrict__ slm_scales_A,  // SLM for A scales [8]
                                  float * __restrict__ slm_scales_B,  // SLM for B scales [16]
                                  float * __restrict__ slm_mins_A,    // SLM for A mins [8]
                                  float * __restrict__ slm_sums_B,    // SLM for B sums [16]
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    // Q4_K has 256-element super-blocks, but XMX processes 32 elements at a time
    // We'll iterate over 8 internal blocks per super-block
    constexpr int Q4_K_SB_SIZE     = 256;                                      // Super-block size (QK_K)
    constexpr int Q4_K_SCALE_BYTES = 12;                                       // K_SCALE_SIZE
    constexpr int Q4_K_BLOCK_SIZE  = 4 + Q4_K_SCALE_BYTES + Q4_K_SB_SIZE / 2;  // 144 bytes
    constexpr int Q8_1_BLOCK_SIZE  = 36;

    const int num_super_blocks = ncols_x / Q4_K_SB_SIZE;
    const int num_k_blocks     = ncols_x / XMX_K;  // Total 32-element blocks

    const char * src0 = (const char *) vx;
    const char * src1 = (const char *) vy;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Process each 32-element K block
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // Determine which super-block and internal block
        const int super_block_idx    = k_block / 8;
        const int internal_block_idx = k_block % 8;

        // Load A (Q4_K): extract scale/min for this internal block
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, super_block_idx, num_super_blocks, tile_m, use_tiled);
                const char * sb_ptr = src0 + block_idx * Q4_K_BLOCK_SIZE;

                // Super-block scales
                sycl::half d      = *reinterpret_cast<const sycl::half *>(sb_ptr);
                sycl::half dmin   = *reinterpret_cast<const sycl::half *>(sb_ptr + 2);
                float      d_f    = float(d);
                float      dmin_f = float(dmin);

                // Extract 6-bit scale and min for this internal block from packed scales
                const uint8_t * scales = reinterpret_cast<const uint8_t *>(sb_ptr + 4);

                // Extract 6-bit scale and min using get_scale_min_k4 logic
                // K_SCALE_SIZE = 12 bytes packs 8 scales and 8 mins
                int j = internal_block_idx;
                int sc, m;
                if (j < 4) {
                    sc = scales[j] & 63;
                    m  = scales[j + 4] & 63;
                } else {
                    // For j >= 4: use nibbles from scales[j+4] with upper bits from scales[j-4]/scales[j]
                    sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                    m  = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }

                slm_scales_A[lane_id] = d_f * (float) sc;
                slm_mins_A[lane_id]   = dmin_f * (float) m;

                // Unpack nibbles for this internal block
                // Q4_K nibble layout: 32 bytes contain 64 elements (interleaved)
                // - Elements 0-31: lower nibbles of bytes 0-31
                // - Elements 32-63: upper nibbles of bytes 0-31
                const uint8_t * qs          = reinterpret_cast<const uint8_t *>(sb_ptr + 4 + Q4_K_SCALE_BYTES);
                int             il          = internal_block_idx / 2;  // Group index (0-3)
                int             use_upper   = internal_block_idx % 2;  // 0 = lower nibbles, 1 = upper nibbles
                int             byte_offset = 32 * il;                 // 32 bytes per 64-element group
                int             base_idx    = lane_id * XMX_K;
#    pragma unroll
                for (int k = 0; k < 32; k++) {
                    uint8_t packed = qs[byte_offset + k];
                    if (use_upper) {
                        slm_A[base_idx + k] = (int8_t) (packed >> 4);
                    } else {
                        slm_A[base_idx + k] = (int8_t) (packed & 0x0F);
                    }
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                slm_mins_A[lane_id]   = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1)
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs        = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply Q4_K formula: result = scale_A * d_B * dot - min_A * s_B
// where scale_A = d * sc, min_A = dmin * m, s_B = d_B * sum(qs_B)
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i       = idx / XMX_N;
            int   j       = idx % XMX_N;
            float scale_A = slm_scales_A[i];
            float d_B     = slm_scales_B[j];
            float min_A   = slm_mins_A[i];
            float s_B     = slm_sums_B[j];
            acc[acc_idx] += scale_A * d_B * float(slm_C[idx]) - min_A * s_B;
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

// Write output
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// =============================================================================
// Launcher for Q4_K x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q4_K_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q4_K kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    constexpr int TOTAL_SLM_SIZE = SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
                                   SLM_SCALES_B_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * A_mins = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_sums = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_K_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, A_mins, B_sums, tile_m,
                                                        use_tiled, item);
                       });
    });
}

// =============================================================================
// Q2_K x Q8_1 XMX GEMM Kernel
// Super-block: 256 elements, 2-bit values with scale and min per 16 elements
// =============================================================================
void mul_mat_q2_K_q8_1_xmx_kernel(const void * __restrict__ vx,
                                  const void * __restrict__ vy,
                                  float * __restrict__ dst,
                                  const int ncols_x,
                                  const int nrows_x,
                                  const int ncols_y,
                                  const int nrows_dst,
                                  int8_t * __restrict__ slm_A,
                                  int8_t * __restrict__ slm_B,
                                  int32_t * __restrict__ slm_C,
                                  float * __restrict__ slm_scales_A,
                                  float * __restrict__ slm_scales_B,
                                  float * __restrict__ slm_mins_A,
                                  float * __restrict__ slm_sums_B,
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    // Q2_K: 256 elements per super-block
    // Layout: dm[2] (4 bytes) + scales[16] + qs[64] = 84 bytes
    constexpr int Q2_K_SB_SIZE    = 256;
    constexpr int Q2_K_BLOCK_SIZE = 4 + 16 + 64;  // 84 bytes
    constexpr int Q8_1_BLOCK_SIZE = 36;

    const int num_super_blocks = ncols_x / Q2_K_SB_SIZE;
    const int num_k_blocks     = ncols_x / XMX_K;

    const char * src0 = (const char *) vx;
    const char * src1 = (const char *) vy;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int super_block_idx    = k_block / 8;
        const int internal_block_idx = k_block % 8;

        // Load A (Q2_K)
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, super_block_idx, num_super_blocks, tile_m, use_tiled);
                const char * sb_ptr = src0 + block_idx * Q2_K_BLOCK_SIZE;

                // dm[0] = dall, dm[1] = dmin
                sycl::half dall_h = *reinterpret_cast<const sycl::half *>(sb_ptr);
                sycl::half dmin_h = *reinterpret_cast<const sycl::half *>(sb_ptr + 2);
                float      dall   = float(dall_h);
                float      dmin   = float(dmin_h);

                const uint8_t * scales = reinterpret_cast<const uint8_t *>(sb_ptr + 4);
                const uint8_t * qs     = reinterpret_cast<const uint8_t *>(sb_ptr + 4 + 16);

                // Q2_K layout: each byte in qs has 4 2-bit values for offsets 0, 32, 64, 96
                // For internal block: n = internal_block_idx / 4, determines which 128-element half
                // Internal block 0-3 use qs[0:31], blocks 4-7 use qs[32:63]
                int n     = internal_block_idx / 4;        // 0 or 1
                int shift = (internal_block_idx % 4) * 2;  // 0, 2, 4, or 6

                // Scale index: is = 8*n + (internal_block_idx % 4) * 2 + is0
                // Each 32-element block uses 2 scales (for 16 elements each)
                int     is_base = 8 * n + (internal_block_idx % 4) * 2;
                uint8_t sc0     = scales[is_base];
                uint8_t sc1     = scales[is_base + 1];

                // For XMX, we use average scale (elements 0-15 and 16-31 of internal block)
                float scale_avg = dall * (((sc0 & 0xF) + (sc1 & 0xF)) * 0.5f);
                float min_avg   = dmin * (((sc0 >> 4) + (sc1 >> 4)) * 0.5f);

                slm_scales_A[lane_id] = scale_avg;
                slm_mins_A[lane_id]   = min_avg;

                int base_idx = lane_id * XMX_K;
                int qs_base  = 32 * n;

#    pragma unroll
                for (int k = 0; k < 32; k++) {
                    int q2              = (qs[qs_base + k] >> shift) & 3;
                    slm_A[base_idx + k] = (int8_t) q2;
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                slm_mins_A[lane_id]   = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1)
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                sycl::half     s         = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                const int8_t * qs_b      = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs_b[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply scales and mins (Q2_K has min offset)
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int   i       = idx / XMX_N;
            int   j       = idx % XMX_N;
            float scale_a = slm_scales_A[i];
            float scale_b = slm_scales_B[j];
            float min_a   = slm_mins_A[i];
            float sum_b   = slm_sums_B[j];
            acc[acc_idx] += scale_a * scale_b * float(slm_C[idx]) - min_a * sum_b;
        }
    }

// Write output (column-major: dst[col * nrows_dst + row])
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// Launcher for Q2_K
static void ggml_mul_mat_q2_K_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    constexpr int TOTAL_SLM_SIZE = SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
                                   SLM_SCALES_B_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * A_mins = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * B_sums = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q2_K_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, A_mins, B_sums, tile_m,
                                                        use_tiled, item);
                       });
    });
}

// =============================================================================
// Q6_K x Q8_1 XMX GEMM Kernel
// Super-block: 256 elements, 6-bit values (4 bits in ql, 2 bits in qh)
// =============================================================================
void mul_mat_q6_K_q8_1_xmx_kernel(const void * __restrict__ vx,
                                  const void * __restrict__ vy,
                                  float * __restrict__ dst,
                                  const int ncols_x,
                                  const int nrows_x,
                                  const int ncols_y,
                                  const int nrows_dst,
                                  int8_t * __restrict__ slm_A,
                                  int8_t * __restrict__ slm_B,
                                  int32_t * __restrict__ slm_C,
                                  float * __restrict__ slm_scales_A,
                                  float * __restrict__ slm_scales_B,
                                  const int tile_m,
                                  const bool use_tiled,
                                  sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= nrows_x || col_base >= ncols_y) {
        return;
    }

    // Q6_K: 256 elements per super-block
    // Layout: ql[128] + qh[64] + scales[16] + d (2 bytes) = 210 bytes
    constexpr int Q6_K_SB_SIZE    = 256;
    constexpr int Q6_K_BLOCK_SIZE = 128 + 64 + 16 + 2;  // 210 bytes
    constexpr int Q8_1_BLOCK_SIZE = 36;

    const int num_super_blocks = ncols_x / Q6_K_SB_SIZE;
    const int num_k_blocks     = ncols_x / XMX_K;

    const char * src0 = (const char *) vx;
    const char * src1 = (const char *) vy;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int super_block_idx    = k_block / 8;
        const int internal_block_idx = k_block % 8;

        // Load A (Q6_K)
        if (lane_id < XMX_M) {
            int row = row_base + lane_id;
            if (row < nrows_x) {
                const int64_t block_idx = xmx_gemm_weight_block_index(row, super_block_idx, num_super_blocks, tile_m, use_tiled);
                const char * sb_ptr = src0 + block_idx * Q6_K_BLOCK_SIZE;

                const uint8_t * ql     = reinterpret_cast<const uint8_t *>(sb_ptr);
                const uint8_t * qh     = reinterpret_cast<const uint8_t *>(sb_ptr + 128);
                const int8_t *  scales = reinterpret_cast<const int8_t *>(sb_ptr + 128 + 64);
                sycl::half      d_h    = *reinterpret_cast<const sycl::half *>(sb_ptr + 128 + 64 + 16);
                float           d      = float(d_h);

                // Q6_K layout from dequantize:
                // ip = internal_block_idx / 4 (0 or 1 - which 128-element half)
                // Positions within 128-element half based on internal_block_idx % 4
                int ip       = internal_block_idx / 4;  // 0 or 1
                int il_group = internal_block_idx % 4;  // 0-3

                // Scale index: is = 8*ip + (il_group / 2) * 2 + il_group % 2 = 8*ip + il_group
                // But actually 2 scales per 32 elements (16 each)
                int   is_base         = 8 * ip + il_group * 2;
                float scale_avg       = d * ((float) scales[is_base] + (float) scales[is_base + 1]) * 0.5f;
                slm_scales_A[lane_id] = scale_avg;

                int base_idx = lane_id * XMX_K;

                // 4 internal blocks per 128-element half
                // Each internal block of 32 elements spans:
                // - For blocks 0,1: ql[64*ip : 64*ip+31], using lower nibble for 0, upper for 1
                // - For blocks 2,3: ql[64*ip+32 : 64*ip+63], using lower nibble for 2, upper for 3
                // - qh[32*ip : 32*ip+31] provides upper 2 bits

                int ql_base          = 64 * ip + (il_group / 2) * 32;
                int qh_base          = 32 * ip;
                int use_upper_nibble = il_group % 2;
                int qh_shift         = (il_group / 2) * 4 + use_upper_nibble * 2;  // 0, 2, 4, or 6

#    pragma unroll
                for (int k = 0; k < 32; k++) {
                    uint8_t ql_val = ql[ql_base + k];
                    uint8_t qh_val = qh[qh_base + k];

                    int q_low  = use_upper_nibble ? (ql_val >> 4) : (ql_val & 0xF);
                    int q_high = (qh_val >> qh_shift) & 3;
                    int q6     = q_low | (q_high << 4);

                    slm_A[base_idx + k] = (int8_t) (q6 - 32);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                int base_idx          = lane_id * XMX_K;
#    pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        // Load B (Q8_1)
        {
            int col = col_base + lane_id;
            if (col < ncols_y) {
                const char *   block_ptr = src1 + (col * num_k_blocks + k_block) * Q8_1_BLOCK_SIZE;
                sycl::half     d         = *reinterpret_cast<const sycl::half *>(block_ptr);
                const int8_t * qs_b      = reinterpret_cast<const int8_t *>(block_ptr + 4);

                slm_scales_B[lane_id] = float(d);

#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs_b[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
#    pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // XMX Multiply
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                                                                                                   matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto B_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);
        auto C_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);

        sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
        sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
        sycl_xmx::joint_matrix_fill(sg, matC, 0);
        sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
        sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

// Apply scales
#    pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            int i = idx / XMX_N;
            int j = idx % XMX_N;
            acc[acc_idx] += slm_scales_A[i] * slm_scales_B[j] * float(slm_C[idx]);
        }
    }

// Write output (column-major: dst[col * nrows_dst + row])
#    pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        int i   = idx / XMX_N;
        int j   = idx % XMX_N;
        int row = row_base + i;
        int col = col_base + j;
        if (row < nrows_x && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

// Launcher for Q6_K
static void ggml_mul_mat_q6_K_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int8_t *  A_int8  = reinterpret_cast<int8_t *>(shared);
                           int8_t *  B_int8  = reinterpret_cast<int8_t *>(shared + SLM_A_SIZE);
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + SLM_A_SIZE + SLM_B_SIZE);
                           float *   A_scales =
                               reinterpret_cast<float *>(shared + SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4);
                           float * B_scales = reinterpret_cast<float *>(shared + SLM_A_SIZE + SLM_B_SIZE +
                                                                        SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4);

                           mul_mat_q6_K_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Launcher for Q8_0 x Q8_1 XMX GEMM
// =============================================================================
static void ggml_mul_mat_q8_0_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG("[XMX] Launching Q8_0 kernel: grid=(%d,%d), M=%d, N=%d, K=%d\n", num_row_tiles, num_col_tiles,
                    nrows_x, ncols_y, ncols_x);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int8_t *  A_int8  = reinterpret_cast<int8_t *>(shared);
                           int8_t *  B_int8  = reinterpret_cast<int8_t *>(shared + SLM_A_SIZE);
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + SLM_A_SIZE + SLM_B_SIZE);
                           float *   A_scales =
                               reinterpret_cast<float *>(shared + SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4);
                           float * B_scales = reinterpret_cast<float *>(shared + SLM_A_SIZE + SLM_B_SIZE +
                                                                        SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4);

                           mul_mat_q8_0_q8_1_xmx_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, A_int8,
                                                        B_int8, C_int32, A_scales, B_scales, tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Launcher for Q8_0 x Q8_1 XMX GEMM - Large Tile (64×64 per work-group)
// Uses 32 sub-groups (8×4) = 512 threads per work-group
// Key advantage: No nibble unpacking - should be faster than Q4_0!
// =============================================================================
static void ggml_mul_mat_q8_0_q8_1_xmx_largetile_sycl(const void *    vx,
                                                      const void *    vy,
                                                      float *         dst,
                                                      const int       ncols_x,
                                                      const int       nrows_x,
                                                      const int       ncols_y,
                                                      const int       nrows_dst,
                                                      const int       tile_m,
                                                      const bool      use_tiled,
                                                      dpct::queue_ptr stream) {
    // Grid: one work-group per 64×64 output tile
    const int num_row_groups = (nrows_x + Q8_LT_OUTPUT_ROWS - 1) / Q8_LT_OUTPUT_ROWS;
    const int num_col_groups = (ncols_y + Q8_LT_OUTPUT_COLS - 1) / Q8_LT_OUTPUT_COLS;

    sycl::range<2> grid(num_row_groups, num_col_groups);
    sycl::range<2> block(Q8_LT_SG_COUNT, XMX_SG_SIZE);  // 32 sub-groups × 16 threads = 512 threads

    GGML_SYCL_DEBUG("[XMX-Q8-LT] Launching Q8_0 large-tile kernel: grid=(%d,%d), block=(%d,%d), M=%d, N=%d, K=%d\n",
                    num_row_groups, num_col_groups, Q8_LT_SG_COUNT, XMX_SG_SIZE, nrows_x, ncols_y, ncols_x);

    // SLM layout for Q8_0 large-tile:
    // - A tiles: 8 row tiles × 8×32 int8 = 2048 bytes
    // - B tiles: 4 col tiles × 32×16 int8 = 2048 bytes
    // - C tiles: 32 sub-groups × 8×16 int32 = 16384 bytes
    // - A scales: 8 row tiles × 8 floats = 256 bytes
    // - B scales: 4 col tiles × 16 floats = 256 bytes
    constexpr int Q8_LT_TOTAL_SLM_SIZE = Q8_LT_SLM_A_SIZE +             // 2048
                                         Q8_LT_SLM_B_SIZE +             // 2048
                                         Q8_LT_SLM_C_SIZE * 4 +         // 16384
                                         Q8_LT_SLM_SCALES_A_SIZE * 4 +  // 256
                                         Q8_LT_SLM_SCALES_B_SIZE * 4;   // 256
                                                                        // Total: ~21KB

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(Q8_LT_TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * slm_A  = reinterpret_cast<int8_t *>(shared + offset);
                           offset += Q8_LT_SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += Q8_LT_SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += Q8_LT_SLM_C_SIZE * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += Q8_LT_SLM_SCALES_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q8_0_q8_1_xmx_largetile_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst,
                                                                  slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                                  tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Launcher for Q4_0 x Q8_1 XMX GEMM - Large Tile (64×64 per work-group)
// Uses 32 sub-groups (8×4) = 512 threads per work-group
// Best for large batch sizes (≥64) to amortize launch overhead
// =============================================================================
static void ggml_mul_mat_q4_0_q8_1_xmx_largetile_sycl(const void *    vx,
                                                      const void *    vy,
                                                      float *         dst,
                                                      const int       ncols_x,
                                                      const int       ncols_x_padded,
                                                      const int       nrows_x,
                                                      const int       ncols_y,
                                                      const int       nrows_dst,
                                                      const int       tile_m,
                                                      const bool      use_tiled,
                                                      dpct::queue_ptr stream) {
    // Grid: one work-group per 64×64 output tile
    const int num_row_groups = (nrows_x + LT_OUTPUT_ROWS - 1) / LT_OUTPUT_ROWS;
    const int num_col_groups = (ncols_y + LT_OUTPUT_COLS - 1) / LT_OUTPUT_COLS;

    sycl::range<2> grid(num_row_groups, num_col_groups);
    sycl::range<2> block(LT_SG_COUNT, XMX_SG_SIZE);  // 32 sub-groups × 16 threads = 512

    GGML_SYCL_DEBUG("[XMX-LT] Launching large-tile kernel: grid=(%d,%d), block=(%d,%d), M=%d, N=%d, K=%d\n",
                    num_row_groups, num_col_groups, LT_SG_COUNT, XMX_SG_SIZE, nrows_x, ncols_y, ncols_x);

    // SLM layout for large-tile (C accumulated in registers, not SLM):
    // - A tiles: 8 row tiles × 8×32 int8 = 2048 bytes
    // - B tiles: 4 col tiles × 32×16 int8 = 2048 bytes
    // - A scales: 8 row tiles × 8 floats = 256 bytes
    // - B scales: 4 col tiles × 16 floats = 256 bytes
    // - B sums: 4 col tiles × 16 floats = 256 bytes
    constexpr int LT_TOTAL_SLM_SIZE = LT_SLM_A_SIZE +             // 2048
                                      LT_SLM_B_SIZE +             // 2048
                                      LT_SLM_SCALES_A_SIZE * 4 +  // 256
                                      LT_SLM_SCALES_B_SIZE * 4 +  // 256
                                      LT_SLM_SUMS_B_SIZE * 4;     // 256
                                                                  // Total: ~5KB (reduced from 21KB)

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(LT_TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * slm_A  = reinterpret_cast<int8_t *>(shared + offset);
                           offset += LT_SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += LT_SLM_B_SIZE;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += LT_SLM_SCALES_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += LT_SLM_SCALES_B_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_0_q8_1_xmx_largetile_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst,
                                                                  slm_A, slm_B, slm_scales_A, slm_scales_B, slm_sums_B,
                                                                  tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Launcher for Q4_0 x Q8_1 XMX GEMM (Double-Buffered Version)
// Uses ping-pong buffers to overlap memory loads with computation
// =============================================================================
// Multi-tile launcher - VERTICAL tiling (16 rows × 16 cols per work-group)
static void ggml_mul_mat_q4_0_q8_1_xmx_multitile_sycl(const void *    vx,
                                                      const void *    vy,
                                                      float *         dst,
                                                      const int       ncols_x,
                                                      const int       ncols_x_padded,
                                                      const int       nrows_x,
                                                      const int       ncols_y,
                                                      const int       nrows_dst,
                                                      const int       tile_m,
                                                      const bool      use_tiled,
                                                      dpct::queue_ptr stream) {
    // Grid: one work-group per 16×16 output tile (2 vertical XMX tiles)
    const int num_row_tiles = (nrows_x + MT_OUTPUT_ROWS - 1) / MT_OUTPUT_ROWS;  // 16 rows per work-group
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;                    // 16 cols per work-group

    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(MT_SG_COUNT, XMX_SG_SIZE);  // 2 sub-groups × 16 threads = 32

    GGML_SYCL_DEBUG(
        "[XMX-MT-DB] Launching double-buffered vertical multi-tile kernel: grid=(%d,%d), block=(%d,%d), M=%d, N=%d, "
        "K=%d\n",
        num_row_tiles, num_col_tiles, MT_SG_COUNT, XMX_SG_SIZE, nrows_x, ncols_y, ncols_x);

    // SLM layout for double-buffered vertical multi-tile:
    // - Per-sub-group A tiles: 2 buffers × 4 subgroups × 8×32 int8 = 2048 bytes
    // - Shared B tile: 2 buffers × 32×16 int8 = 1024 bytes
    // - Per-sub-group C tiles: 4 subgroups × 8×16 int32 = 2048 bytes (not double-buffered)
    // - Per-sub-group A scales: 2 buffers × 4 subgroups × 8 floats = 256 bytes
    // - Shared B scales: 2 buffers × 16 floats = 128 bytes
    // - Shared B sums: 2 buffers × 16 floats = 128 bytes
    constexpr int MT_DB_TOTAL_SLM_SIZE = MT_DB_SLM_A_SIZE +                // 2048
                                         MT_DB_SLM_B_SIZE +                // 1024
                                         MT_SLM_C_SIZE * MT_TILES_M * 4 +  // 2048
                                         MT_DB_SLM_SCALES_A_SIZE * 4 +     // 256
                                         MT_DB_SLM_SCALES_B_SIZE * 4 +     // 128
                                         MT_DB_SLM_SUMS_B_SIZE * 4;        // 128
                                                                           // Total: ~5.6KB

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(MT_DB_TOTAL_SLM_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * slm_A  = reinterpret_cast<int8_t *>(shared + offset);
                           offset += MT_DB_SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += MT_DB_SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += MT_SLM_C_SIZE * MT_TILES_M * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += MT_DB_SLM_SCALES_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += MT_DB_SLM_SCALES_B_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_0_q8_1_xmx_multitile_db_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst,
                                                                     slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                                     slm_sums_B, tile_m, use_tiled, item);
                       });
    });
}

// Single-tile launcher for small batch sizes (batch <= 16)
static void ggml_mul_mat_q4_0_q8_1_xmx_sycl(const void *    vx,
                                            const void *    vy,
                                            float *         dst,
                                            const int       ncols_x,
                                            const int       ncols_x_padded,
                                            const int       nrows_x,
                                            const int       ncols_y,
                                            const int       nrows_dst,
                                            const int       tile_m,
                                            const bool      use_tiled,
                                            dpct::queue_ptr stream) {
    // Adaptive dispatch based on batch size for optimal performance:
    // - batch >= 64: large-tile (512 threads, 64×64 output per work-group)
    // - batch 33-63: 4-tile sequential (better A tile reuse, efficient for 3-4 column tiles)
    // - batch 17-32: 2-tile parallel (2 sub-groups sharing A tile)
    // - batch <= 16: single-tile double-buffered
    if (ncols_y >= 64) {
        ggml_mul_mat_q4_0_q8_1_xmx_largetile_sycl(vx, vy, dst, ncols_x, ncols_x_padded, nrows_x, ncols_y, nrows_dst,
                                                  tile_m, use_tiled, stream);
        return;
    }
    if (ncols_y > 32) {
        // 4-tile sequential is better for batch 33-64 (handles 3-4 column tiles efficiently)
        ggml_mul_mat_q4_0_q8_1_xmx_colfused_4tile_sycl(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, tile_m,
                                                      use_tiled, stream);
        return;
    }
    if (ncols_y > 16) {
        // 2-tile parallel is better for batch 17-32 (exactly 2 column tiles)
        ggml_mul_mat_q4_0_q8_1_xmx_colfused_sycl(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst, tile_m, use_tiled,
                                                 stream);
        return;
    }

    // Grid: one work-group per 8×16 output tile
    const int num_row_tiles = (nrows_x + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (ncols_y + XMX_N - 1) / XMX_N;

    // Block: one sub-group (16 threads)
    sycl::range<2> grid(num_row_tiles, num_col_tiles);
    sycl::range<2> block(1, XMX_SG_SIZE);

    GGML_SYCL_DEBUG(
        "[XMX] Launching double-buffered kernel: grid=(%d,%d), block=(1,%d), M=%d, N=%d, K=%d (padded=%d)\n",
        num_row_tiles, num_col_tiles, XMX_SG_SIZE, nrows_x, ncols_y, ncols_x, ncols_x_padded);

    // Double-buffered XMX kernel with 2× SLM for A, B, scales, and sums
    constexpr int TOTAL_SLM_SIZE_DB = SLM_A_SIZE_DB + SLM_B_SIZE_DB + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE_DB * 4 +
                                      SLM_SCALES_B_SIZE_DB * 4 + SLM_SUMS_B_SIZE_DB * 4;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE_DB), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           int      offset = 0;
                           int8_t * A_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE_DB;
                           int8_t * B_int8 = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE_DB;
                           int32_t * C_int32 = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * A_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE_DB * 4;
                           float * B_scales = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE_DB * 4;
                           float * B_sums = reinterpret_cast<float *>(shared + offset);

                           mul_mat_q4_0_q8_1_xmx_doublebuf_kernel(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_dst,
                                                                  A_int8, B_int8, C_int32, A_scales, B_scales, B_sums,
                                                                  tile_m, use_tiled, item);
                       });
    });
}

// =============================================================================
// Public XMX Mul Mat Operation
// =============================================================================

void ggml_sycl_op_mul_mat_q_xmx(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                ggml_tensor *               dst,
                                const char *                src0_dd_i,
                                const float *               src1_ddf_i,
                                const char *                src1_ddq_i,
                                float *                     dst_dd_i,
                                const int64_t               row_low,
                                const int64_t               row_high,
                                const int64_t               src1_ncols,
                                const int64_t               src1_padded_row_size,
                                const dpct::queue_ptr &     stream) {
    // Check tiered dispatch for weight tensor
    if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool                   in_inventory = false;
        void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            GGML_SYCL_DEBUG("[SYCL] mmq_xmx tiered hit: %s (tier=%d)\n", src0->name, static_cast<int>(tier));
        } else if (in_inventory) {
            GGML_SYCL_DEBUG("[SYCL] mmq_xmx tiered pending: %s\n", src0->name);
        }
    }

    const int64_t ne00     = src0->ne[0];         // K dimension
    const int64_t row_diff = row_high - row_low;  // M dimension (rows to process)

    int device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));

    const auto & caps = ggml_sycl_info().devices[device_id].xmx_caps;
    int          tile_m = static_cast<int>(caps.M);
    bool         use_tiled = false;
    if (const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra)) {
        use_tiled = get_effective_layout_mode(extra) == GGML_LAYOUT_XMX_GEMM_TILED;
    }
    if (use_tiled) {
        GGML_ASSERT(caps.supported && caps.supports_int8 && caps.M > 0 && caps.N > 0 && caps.K > 0);
    }

    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = device_id == ctx.device ? dst->ne[0] : row_diff;

    GGML_SYCL_DEBUG("[XMX] mul_mat: type=%d, M=%ld, N=%ld, K=%ld (padded=%ld), nrows_dst=%ld\n", src0->type, row_diff,
                    src1_ncols, ne00, src1_padded_row_size, nrows_dst);
    // Debug print to stderr to see values when debug flag is set
    static int debug_count = 0;
    if (debug_count == 0 && getenv("GGML_SYCL_DEBUG_XMX_GEMM")) {
        fprintf(stderr, "[XMX DEBUG] K=%ld, K_padded=%ld, difference=%ld\n", (long) ne00, (long) src1_padded_row_size,
                (long) (src1_padded_row_size - ne00));
    }
    debug_count++;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, src1_padded_row_size, row_diff,
                                            src1_ncols, nrows_dst, tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q8_0:
            // Use optimized large-tile kernel for Q8_0
            ggml_mul_mat_q8_0_q8_1_xmx_largetile_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols,
                                                      nrows_dst, tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, nrows_dst,
                                            tile_m, use_tiled, stream);
            break;
        default:
            GGML_LOG_WARN("[XMX] Unsupported type %d\n", src0->type);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(src1_ddf_i);
}

// =============================================================================
// XMX Support Checks
// =============================================================================

bool ggml_sycl_xmx_supports_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        case GGML_TYPE_Q4_1:
            return true;
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q5_0:
            return true;  // 5-bit, 1.52x speedup at pp
        case GGML_TYPE_Q5_1:
            return true;  // 5-bit with min offset
        case GGML_TYPE_Q3_K:
            return true;  // Re-enabled after fixing bit unpacking and scale extraction
        case GGML_TYPE_Q4_K:
            return true;  // 1.55x speedup at pp16 (256-element super-blocks)
        case GGML_TYPE_Q5_K:
            return true;  // 1.66x speedup at pp16
        case GGML_TYPE_Q2_K:
            return true;  // 2-bit K-quant
        case GGML_TYPE_Q6_K:
            return true;  // 6-bit K-quant
        default:
            return false;
    }
}

// =============================================================================
// Test Function (verified working)
// =============================================================================

void xmx_test_kernel(sycl::queue & q) {
    constexpr int M = 8, N = 16, K = 32;
    int8_t *      d_A = ggml_sycl_malloc_device_tracked_t<int8_t>(M * K, q, "mmq_xmx_test");
    int8_t *      d_B = ggml_sycl_malloc_device_tracked_t<int8_t>(K * N, q, "mmq_xmx_test");
    int32_t *     d_C = ggml_sycl_malloc_device_tracked_t<int32_t>(M * N, q, "mmq_xmx_test");

    std::vector<int8_t> h_A(M * K, 1);
    std::vector<int8_t> h_B(K * N, 2);
    q.memcpy(d_A, h_A.data(), M * K * sizeof(int8_t));
    q.memcpy(d_B, h_B.data(), K * N * sizeof(int8_t));
    q.wait();

    q.submit([&](sycl::handler & h) {
         h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
             auto sg = item.get_sub_group();

             sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, M, K, sycl_xmx::layout::row_major> matA;
             sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, K, N, sycl_xmx::layout::col_major> matB;
             sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, M, N>                   matC;

             sycl_xmx::joint_matrix_load(
                 sg, matA,
                 sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_A),
                 K);
             sycl_xmx::joint_matrix_load(
                 sg, matB,
                 sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_B),
                 K);
             sycl_xmx::joint_matrix_fill(sg, matC, 0);

             sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);

             sycl_xmx::joint_matrix_store(
                 sg, matC,
                 sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_C),
                 N, sycl_xmx::layout::row_major);
         });
     }).wait();

    std::vector<int32_t> h_C(M * N);
    q.memcpy(h_C.data(), d_C, M * N * sizeof(int32_t)).wait();

    bool pass     = true;
    int  expected = 1 * 2 * K;
    for (int i = 0; i < M * N; i++) {
        if (h_C[i] != expected) {
            pass = false;
            break;
        }
    }

    if (pass) {
        GGML_LOG_INFO("XMX Int8 test PASSED: all %d elements = %d\n", M * N, expected);
    } else {
        GGML_LOG_WARN("XMX Int8 test FAILED: got %d, expected %d\n", h_C[0], expected);
    }

    ggml_sycl_free_device_tracked_t(d_A, M * K, q);
    ggml_sycl_free_device_tracked_t(d_B, K * N, q);
    ggml_sycl_free_device_tracked_t(d_C, M * N, q);
}

void ggml_sycl_xmx_test_func(ggml_backend_sycl_context & ctx) {
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    xmx_test_kernel(*main_stream);
}

bool ggml_sycl_xmx_available() {
    return true;
}

void ggml_sycl_xmx_get_tile_dims(int * m, int * n, int * k) {
    if (!m || !n || !k) {
        return;
    }
    int device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));
    if (device_id < 0 || device_id >= ggml_sycl_info().device_count) {
        *m = 0;
        *n = 0;
        *k = 0;
        return;
    }
    const auto & caps = ggml_sycl_info().devices[device_id].xmx_caps;
    if (!caps.supported || !caps.supports_int8 || caps.M == 0 || caps.N == 0 || caps.K == 0) {
        *m = 0;
        *n = 0;
        *k = 0;
        return;
    }
    *m = static_cast<int>(caps.M);
    *n = static_cast<int>(caps.N);
    *k = static_cast<int>(caps.K);
}

#else   // !SYCL_XMX_AVAILABLE

void ggml_sycl_op_mul_mat_q_xmx(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                ggml_tensor *               dst,
                                const char *                src0_dd_i,
                                const float *               src1_ddf_i,
                                const char *                src1_ddq_i,
                                float *                     dst_dd_i,
                                const int64_t               row_low,
                                const int64_t               row_high,
                                const int64_t               src1_ncols,
                                const int64_t               src1_padded_row_size,
                                const dpct::queue_ptr &     stream) {
    GGML_LOG_WARN("XMX not available\n");
    (void) ctx;
    (void) src0;
    (void) src1;
    (void) dst;
    (void) src0_dd_i;
    (void) src1_ddf_i;
    (void) src1_ddq_i;
    (void) dst_dd_i;
    (void) row_low;
    (void) row_high;
    (void) src1_ncols;
    (void) src1_padded_row_size;
    (void) stream;
}

bool ggml_sycl_xmx_supports_type(ggml_type type) {
    (void) type;
    return false;
}

void ggml_sycl_xmx_test_func(ggml_backend_sycl_context & ctx) {
    GGML_LOG_WARN("XMX not available\n");
    (void) ctx;
}

bool ggml_sycl_xmx_available() {
    return false;
}

void ggml_sycl_xmx_get_tile_dims(int * m, int * n, int * k) {
    *m = 0;
    *n = 0;
    *k = 0;
}

#endif  // SYCL_XMX_AVAILABLE
