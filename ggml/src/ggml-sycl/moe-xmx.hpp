// moe-xmx.hpp - XMX-accelerated MoE GEMM kernel
#pragma once

#include "common.hpp"
#include "moe-sort.hpp"

#include <sycl/sycl.hpp>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define SYCL_XMX_MOE_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#    define SYCL_XMX_MOE_AVAILABLE 0
#endif

#if SYCL_XMX_MOE_AVAILABLE

namespace moe_xmx {

using namespace sycl::ext::oneapi::experimental::matrix;

// Device function: binary search to find expert from work-group ID
// Given wg_id and expert_tile_offsets[], returns the expert index
// such that expert_tile_offsets[expert] <= wg_id < expert_tile_offsets[expert + 1]
inline int find_expert_for_workgroup(
    int             wg_id,
    const int32_t * expert_tile_offsets,
    int             n_experts) {
    int lo = 0, hi = n_experts;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (expert_tile_offsets[mid + 1] <= wg_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Compute local tile index within expert's tile range
inline int get_local_tile_index(
    int             wg_id,
    int             expert_idx,
    const int32_t * expert_tile_offsets) {
    return wg_id - expert_tile_offsets[expert_idx];
}

// Configuration for XMX MoE kernel
struct MoEXMXConfig {
    // Hardware parameters (from XMXCapabilities)
    int M = 8;   // Tile rows
    int N = 16;  // Tile cols
    int K = 32;  // Reduction dim

    // Tunable parameters
    int tiles_m = 4;  // Tiles per WG in M dimension
    int tiles_n = 4;  // Tiles per WG in N dimension
    int wg_size = 256;

    // SLM allocation
    int slm_weight_bytes = 16 * 1024;  // 16KB for weight double-buffer
    int slm_token_bytes  = 4 * 1024;   // 4KB for token tile

    static MoEXMXConfig from_capabilities(const XMXCapabilities & caps) {
        MoEXMXConfig cfg;
        if (caps.M > 0) {
            cfg.M = static_cast<int>(caps.M);
        }
        if (caps.N > 0) {
            cfg.N = static_cast<int>(caps.N);
        }
        if (caps.K > 0) {
            cfg.K = static_cast<int>(caps.K);
        }
        cfg.tiles_m = caps.optimal_tiles_m;
        cfg.tiles_n = caps.optimal_tiles_n;
        return cfg;
    }
};

// Pre-quantize fp16 tokens to int8 with per-block scales
// Output: q_tokens[batch * in_dim] int8, scales[batch * (in_dim/32)] fp16
void preprocess_tokens_q8(const sycl::half * tokens,    // [batch, in_dim] fp16 input
                          int8_t *           q_tokens,  // [batch, in_dim] int8 output
                          sycl::half *       scales,    // [batch, in_dim/32] per-block scales
                          int64_t            batch,
                          int64_t            in_dim,
                          sycl::queue &      queue);

inline void preprocess_tokens_q8(const sycl::half * tokens,
                                 int8_t *           q_tokens,
                                 sycl::half *       scales,
                                 int64_t            batch,
                                 int64_t            in_dim,
                                 sycl::queue &      queue) {
    constexpr int SG_SIZE = 16;

    int64_t num_blocks = batch * (in_dim / QK8_0);

    queue.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_blocks * SG_SIZE, SG_SIZE),
                         [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                             auto    sg       = item.get_sub_group();
                             int64_t block_id = item.get_group(0);
                             int64_t row      = block_id / (in_dim / QK8_0);
                             int64_t k_block  = block_id % (in_dim / QK8_0);

                             int lane = sg.get_local_linear_id();

                             // Each lane loads 2 values (32 total per sub-group)
                             int64_t base = row * in_dim + k_block * QK8_0;
                             float   v0   = static_cast<float>(tokens[base + lane * 2]);
                             float   v1   = static_cast<float>(tokens[base + lane * 2 + 1]);

                             // Find max absolute value via sub-group reduction
                             float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
                             float amax      = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                             // Compute scale and inverse scale
                             float scale     = amax / 127.0f;
                             float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;

                             // Quantize values
                             int8_t q0 = static_cast<int8_t>(sycl::round(v0 * inv_scale));
                             int8_t q1 = static_cast<int8_t>(sycl::round(v1 * inv_scale));

                             // Store quantized values
                             q_tokens[base + lane * 2]     = q0;
                             q_tokens[base + lane * 2 + 1] = q1;

                             // Store scale (one per block, lane 0 only)
                             if (lane == 0) {
                                 scales[row * (in_dim / QK8_0) + k_block] = sycl::half(scale);
                             }
                         });
    });
}

// Extract fp16 scales from Q8_0 weight blocks
// Q8_0 block layout: [2 bytes fp16 scale][32 int8 values] = 34 bytes per block
// Output: scales[out_dim * (in_dim/32)] in row-major order
inline void extract_q8_0_scales(const void *  weights_qs,  // [out_dim, in_dim] Q8_0 packed
                                sycl::half *  scales,      // [out_dim, in_dim/32] output scales
                                int64_t       out_dim,
                                int64_t       in_dim,
                                sycl::queue & queue) {
    constexpr int Q8_0_BLOCK_SIZE = 34;  // 32 int8 + 2 bytes fp16 scale

    int64_t num_blocks_per_row = in_dim / QK8_0;
    int64_t total_blocks       = out_dim * num_blocks_per_row;

    const uint8_t * w_ptr = static_cast<const uint8_t *>(weights_qs);

    queue
        .parallel_for(sycl::range<1>(total_blocks),
                      [=](sycl::id<1> idx) {
                          // Q8_0 block layout: first 2 bytes are fp16 scale (little-endian)
                          // then 32 bytes of int8 values
                          int64_t block_offset = idx * Q8_0_BLOCK_SIZE;

                          // Load fp16 scale (stored at start of block in GGML Q8_0 format)
                          uint16_t scale_bits =
                              w_ptr[block_offset] | (static_cast<uint16_t>(w_ptr[block_offset + 1]) << 8);

                          // Reinterpret as fp16
                          sycl::half scale;
                          std::memcpy(&scale, &scale_bits, sizeof(sycl::half));

                          scales[idx] = scale;
                      })
        .wait();
}

// Q8_0 XMX GEMM for a single expert's token batch
// Computes: output[batch, out_dim] = q_tokens[batch, in_dim] @ weights[in_dim, out_dim]
//
// Q8_0 block format (34 bytes per 32 elements):
//   - d: 2 bytes fp16 scale (at offset 0)
//   - qs[32]: 32 int8 values (at offset 2)
//
// Note: Token quantization is done externally via preprocess_tokens_q8()
//
template <int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(const void *       weights_qs,  // [in_dim/32, out_dim] int8 quantized (coalesced layout)
                              const sycl::half * weights_d,   // [in_dim/32, out_dim] Q8_0 scales (coalesced layout)
                              const int8_t *     q_tokens,    // [batch, in_dim] pre-quantized int8
                              const sycl::half * token_scales,  // [batch, in_dim/32] token scales
                              sycl::half *       output,        // [batch, out_dim]
                              int64_t            batch,
                              int64_t            out_dim,
                              int64_t            in_dim,
                              const MoEXMXConfig & cfg,
                              sycl::queue &        queue) {
    constexpr int XMX_M           = 8;
    constexpr int XMX_N           = 16;
    constexpr int XMX_K           = 32;
    constexpr int SG_SIZE         = 16;
    constexpr int Q8_0_BLOCK_SIZE = 34;  // 2 bytes fp16 scale + 32 int8 values
    constexpr int Q8_0_SCALE_SIZE = 2;   // fp16 scale at start of block

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{ static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
                           static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    // Compute num_sgs for per-sub-group SLM allocation
    const int num_sgs = cfg.wg_size / SG_SIZE;

    queue
        .submit([&](sycl::handler & cgh) {
            // SLM for unpacked weight tiles
            // One K-block of weights for TILES_N output columns: TILES_N * XMX_N * XMX_K int8 values
            // = 4 * 16 * 32 = 2048 bytes
            constexpr int                   slm_weights_size = TILES_N * XMX_N * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

            // Separate SLM for accumulator extraction - PER SUB-GROUP to avoid race conditions
            // Size: num_sgs * XMX_M * XMX_N * sizeof(int32_t) bytes
            // Each sub-group gets its own 512-byte section to prevent concurrent writes
            constexpr int                   slm_acc_per_sg = XMX_M * XMX_N * sizeof(int32_t);
            const int                       slm_acc_size   = num_sgs * slm_acc_per_sg;
            sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

            // SLM for token tiles (TILES_M * XMX_M * XMX_K int8 = 1024 bytes)
            sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(TILES_M * XMX_M * XMX_K), cgh);

            // SLM for scales (token + weight scales for current K-block)
            // Token scales: one per row (TILES_M * XMX_M = 32 rows)
            sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);
            // Weight scales: one per output column (TILES_N * XMX_N = 64 columns)
            sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                    auto sg    = item.get_sub_group();
                    int  sg_id = sg.get_group_linear_id();

                    int wg_row = item.get_group(0) * wg_out_rows;
                    int wg_col = item.get_group(1) * wg_out_cols;

                    // Bounds check
                    if (wg_row >= batch) {
                        return;
                    }

                    // Initialize accumulators
                    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];

                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tm][tn], 0);
                        }
                    }

                    // Float accumulators for precision across K-blocks
                    float float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };

                    // K-dimension reduction loop
                    // Each iteration processes XMX_K=32 elements along the reduction dimension
                    // For Q8_0: 32 elements = 1 block = 34 bytes (2 byte scale + 32 int8)
                    const uint8_t * w_ptr        = static_cast<const uint8_t *>(weights_qs);
                    const int64_t   num_k_blocks = in_dim / XMX_K;
                    int             lane         = sg.get_local_linear_id();
                    int             num_sgs      = cfg.wg_size / SG_SIZE;

                    for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                        int64_t k = k_block * XMX_K;

                        // === Cooperative token loading to SLM ===
                        // Load pre-quantized int8 tokens
                        constexpr int slm_tokens_size = TILES_M * XMX_M * XMX_K;
                        int           items_per_sg    = slm_tokens_size / num_sgs;
                        int           sg_offset       = sg_id * items_per_sg;
                        for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                            int idx = sg_offset + i + lane;
                            if (idx < slm_tokens_size) {
                                int     tile_row   = idx / XMX_K;
                                int     tile_k     = idx % XMX_K;
                                int     global_row = wg_row + tile_row;
                                int64_t global_k   = k + tile_k;
                                if (global_row < batch && global_k < in_dim) {
                                    slm_tokens[idx] = q_tokens[global_row * in_dim + global_k];
                                } else {
                                    slm_tokens[idx] = 0;
                                }
                            }
                        }

                        // === Load token scales for this K-block ===
                        if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                            int row_idx = sg_id * SG_SIZE + lane;
                            if (row_idx < TILES_M * XMX_M) {
                                int global_row = wg_row + row_idx;
                                if (global_row < batch) {
                                    slm_token_scales[row_idx] =
                                        static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                                } else {
                                    slm_token_scales[row_idx] = 0.0f;
                                }
                            }
                        }

                        // === Q8_0 Weight Unpacking to SLM ===
                        // Each sub-group handles part of the weight tile
                        // Q8_0 block layout: [2 bytes fp16 scale][32 int8 values] = 34 bytes
                        // Weight layout: [in_dim/32, out_dim] blocks (coalesced layout)
                        // For TILES_N * XMX_N output columns, we need to unpack
                        // TILES_N * XMX_N blocks (one block per column for this K-block)
                        //
                        // Total elements to unpack: TILES_N * XMX_N * XMX_K = 64 * 32 = 2048
                        int weights_per_sg = slm_weights_size / num_sgs;
                        int w_sg_offset    = sg_id * weights_per_sg;

                        for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                            int idx = w_sg_offset + i + lane;
                            if (idx < slm_weights_size) {
                                // Decode which output column and K-element this is
                                int out_col_local = idx / XMX_K;  // 0..63 (within TILES_N * XMX_N)
                                int k_elem        = idx % XMX_K;  // 0..31

                                int global_col = wg_col + out_col_local;

                                int8_t unpacked_val = 0;
                                if (global_col < out_dim) {
                                    // Calculate Q8_0 block address for this (out_col, k_block)
                                    // Weight layout: [in_dim, out_dim] - ggml tensors are row-major with ne[0] fastest
                                    // For Q8_0: in_dim elements quantized to in_dim/32 blocks per output row
                                    // Block at (out_col, k_block) = data[out_col * num_k_blocks + k_block]
                                    int64_t         block_offset = global_col * num_k_blocks + k_block;
                                    const uint8_t * block_ptr    = w_ptr + block_offset * Q8_0_BLOCK_SIZE;

                                    // Q8_0 unpacking: skip 2-byte scale header, read int8 directly
                                    unpacked_val = static_cast<int8_t>(block_ptr[Q8_0_SCALE_SIZE + k_elem]);
                                }
                                // Store in column-major order for XMX mat_b (col_major layout)
                                // col_major[K,N]: element (k, n) stored at [k + n * K]
                                slm_weights[k_elem + out_col_local * XMX_K] = unpacked_val;
                            }
                        }

                        // === Load Q8_0 scales for weight columns ===
                        // One scale per output column (TILES_N * XMX_N = 64 scales)
                        if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                            int col_idx = sg_id * SG_SIZE + lane;
                            if (col_idx < TILES_N * XMX_N) {
                                int global_col = wg_col + col_idx;
                                if (global_col < out_dim) {
                                    // Scale is stored at weights_d which is pre-extracted
                                    // Scale layout: [out_dim, num_k_blocks] - row-major, one scale per block
                                    slm_weight_scales[col_idx] =
                                        static_cast<float>(weights_d[global_col * num_k_blocks + k_block]);
                                } else {
                                    slm_weight_scales[col_idx] = 0.0f;
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        // === XMX Computation (only sub-group 0 computes) ===
                        // FIX: Only sg_id == 0 performs computation to avoid race conditions
                        // where all sub-groups would write to same output locations.
                        // Other sub-groups helped with cooperative data loading above.
                        if (sg_id == 0) {
                            // Declare joint matrices for this K-tile
                            // mat_a: row_major [M, K] - tokens stored as [row * K + k]
                            // mat_b: col_major [K, N] - weights stored as [col * K + k] in SLM
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            // Compute tiles - iterate over M and N tile positions
                            for (int tm = 0; tm < TILES_M; tm++) {
                                // Load mat_a from SLM tokens
                                auto slm_tokens_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                               sycl::access::decorated::no>(
                                    &slm_tokens[tm * XMX_M * XMX_K]);
                                joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                                for (int tn = 0; tn < TILES_N; tn++) {
                                    int row = wg_row + tm * XMX_M;
                                    int col = wg_col + tn * XMX_N;

                                    if (row < batch && col < out_dim) {
                                        // Load mat_b from SLM unpacked weights
                                        // Weight tile layout: [TILES_N * XMX_N, XMX_K] stored as
                                        // [out_col_local * XMX_K + k_elem]
                                        // For joint_matrix_load, we need contiguous K dimension
                                        auto slm_weights_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(
                                                &slm_weights[tn * XMX_N * XMX_K]);
                                        joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                        // XMX multiply-accumulate: acc += mat_a * mat_b
                                        joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                        // Store accumulator to SLM section to extract values
                                        constexpr int acc_elements = XMX_M * XMX_N;
                                        int32_t *     acc_slm_raw =
                                            reinterpret_cast<int32_t *>(&slm_acc[0]);  // sg_id == 0, use offset 0
                                        auto acc_slm_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(acc_slm_raw);
                                        joint_matrix_store(sg, acc[tm][tn], acc_slm_ptr, XMX_N, layout::row_major);

                                        // Sub-group barrier is sufficient
                                        sycl::group_barrier(sg);

                                        // Apply scales and accumulate in float
                                        // Note: slm_weight_scales[tn * XMX_N + tile_col] for per-column scales
                                        for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                            int tile_row   = i / XMX_N;
                                            int tile_col   = i % XMX_N;
                                            int global_row = row + tile_row;
                                            int global_col = col + tile_col;
                                            if (global_row < batch && global_col < out_dim) {
                                                float   t_scale = slm_token_scales[tm * XMX_M + tile_row];
                                                float   w_scale = slm_weight_scales[tn * XMX_N + tile_col];
                                                int32_t raw     = acc_slm_raw[i];
                                                float_acc[tm][tn][i] += raw * t_scale * w_scale;
                                            }
                                        }

                                        // Reset accumulator for next K-block
                                        joint_matrix_fill(sg, acc[tm][tn], 0);
                                    }
                                }
                            }
                        }

                        // Barrier before next K-iteration
                        item.barrier(sycl::access::fence_space::local_space);
                    }  // end K-loop

                    // === Final output store (only sub-group 0 writes) ===
                    // MoE scatter expects: sorted_output[pair_idx * output_dim + dim]
                    // This differs from MMQ which outputs column-major to ggml tensors
                    if (sg_id == 0) {
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                int row = wg_row + tm * XMX_M;
                                int col = wg_col + tn * XMX_N;

                                if (row < batch && col < out_dim) {
                                    for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                        int tile_row = i / XMX_N;
                                        int tile_col = i % XMX_N;
                                        if (row + tile_row < batch && col + tile_col < out_dim) {
                                            // Row-major: output[row * out_dim + col]
                                            output[(row + tile_row) * out_dim + col + tile_col] =
                                                sycl::half(float_acc[tm][tn][i]);
                                        }
                                    }
                                }

                                // Suppress unused accumulator warning (used during K-loop for XMX MAD)
                                (void) acc[tm][tn];
                            }
                        }
                    }
                });
        })
        .wait();

}  // end launch_xmx_moe_gemm_q8_0

// Q8_0 SoA XMX GEMM for a single expert's token batch
// Same as launch_xmx_moe_gemm_q8_0 but reads from SoA-formatted weights
//
// SoA layout (per expert):
//   - qs: [nblocks * 32] int8 values contiguously
//   - d:  [nblocks * 2] fp16 scales contiguously
// Where nblocks = out_dim * (in_dim / 32)
//
// Block indexing: block_idx = out_col * num_k_blocks + k_block
//   - qs offset: block_idx * 32
//   - d offset: block_idx
//
template <int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0_soa(const int8_t *       weights_qs,    // [out_dim * in_dim] SoA quantized values
                                  const sycl::half *   weights_d,     // [out_dim * in_dim/32] SoA scales
                                  const int8_t *       q_tokens,      // [batch, in_dim] pre-quantized int8
                                  const sycl::half *   token_scales,  // [batch, in_dim/32] token scales
                                  sycl::half *         output,        // [batch, out_dim]
                                  int64_t              batch,
                                  int64_t              out_dim,
                                  int64_t              in_dim,
                                  const MoEXMXConfig & cfg,
                                  sycl::queue &        queue) {
    constexpr int XMX_M   = 8;
    constexpr int XMX_N   = 16;
    constexpr int XMX_K   = 32;
    constexpr int SG_SIZE = 16;

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{ static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
                           static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    // Compute num_sgs for per-sub-group SLM allocation
    const int num_sgs = cfg.wg_size / SG_SIZE;

    queue
        .submit([&](sycl::handler & cgh) {
            // SLM for unpacked weight tiles
            constexpr int                   slm_weights_size = TILES_N * XMX_N * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

            // Per-sub-group SLM for accumulator extraction
            constexpr int                   slm_acc_per_sg = XMX_M * XMX_N * sizeof(int32_t);
            const int                       slm_acc_size   = num_sgs * slm_acc_per_sg;
            sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

            // SLM for token tiles
            sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(TILES_M * XMX_M * XMX_K), cgh);

            // SLM for scales
            sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);
            sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                    auto sg    = item.get_sub_group();
                    int  sg_id = sg.get_group_linear_id();

                    int wg_row = item.get_group(0) * wg_out_rows;
                    int wg_col = item.get_group(1) * wg_out_cols;

                    if (wg_row >= batch) {
                        return;
                    }

                    // Initialize accumulators
                    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];

                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tm][tn], 0);
                        }
                    }

                    float float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };

                    const int64_t num_k_blocks  = in_dim / XMX_K;
                    int           lane          = sg.get_local_linear_id();
                    int           num_sgs_local = cfg.wg_size / SG_SIZE;

                    for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                        int64_t k = k_block * XMX_K;

                        // === Cooperative token loading to SLM ===
                        constexpr int slm_tokens_size = TILES_M * XMX_M * XMX_K;
                        int           items_per_sg    = slm_tokens_size / num_sgs_local;
                        int           sg_offset       = sg_id * items_per_sg;
                        for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                            int idx = sg_offset + i + lane;
                            if (idx < slm_tokens_size) {
                                int     tile_row   = idx / XMX_K;
                                int     tile_k     = idx % XMX_K;
                                int     global_row = wg_row + tile_row;
                                int64_t global_k   = k + tile_k;
                                if (global_row < batch && global_k < in_dim) {
                                    slm_tokens[idx] = q_tokens[global_row * in_dim + global_k];
                                } else {
                                    slm_tokens[idx] = 0;
                                }
                            }
                        }

                        // === Load token scales for this K-block ===
                        if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                            int row_idx = sg_id * SG_SIZE + lane;
                            if (row_idx < TILES_M * XMX_M) {
                                int global_row = wg_row + row_idx;
                                if (global_row < batch) {
                                    slm_token_scales[row_idx] =
                                        static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                                } else {
                                    slm_token_scales[row_idx] = 0.0f;
                                }
                            }
                        }

                        // === SoA Weight Loading to SLM ===
                        // SoA layout: qs values are contiguous, 32 bytes per block
                        // Block index = out_col * num_k_blocks + k_block
                        int weights_per_sg = slm_weights_size / num_sgs_local;
                        int w_sg_offset    = sg_id * weights_per_sg;

                        for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                            int idx = w_sg_offset + i + lane;
                            if (idx < slm_weights_size) {
                                int out_col_local = idx / XMX_K;
                                int k_elem        = idx % XMX_K;

                                int global_col = wg_col + out_col_local;

                                int8_t val = 0;
                                if (global_col < out_dim) {
                                    // SoA: block_idx * 32 + k_elem
                                    int64_t block_idx = global_col * num_k_blocks + k_block;
                                    val               = weights_qs[block_idx * QK8_0 + k_elem];
                                }
                                slm_weights[k_elem + out_col_local * XMX_K] = val;
                            }
                        }

                        // === Load SoA scales for weight columns ===
                        if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                            int col_idx = sg_id * SG_SIZE + lane;
                            if (col_idx < TILES_N * XMX_N) {
                                int global_col = wg_col + col_idx;
                                if (global_col < out_dim) {
                                    // SoA: scales are at block_idx directly
                                    int64_t block_idx          = global_col * num_k_blocks + k_block;
                                    slm_weight_scales[col_idx] = static_cast<float>(weights_d[block_idx]);
                                } else {
                                    slm_weight_scales[col_idx] = 0.0f;
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        // === XMX Computation (only sub-group 0 computes) ===
                        // FIX: Only sg_id == 0 performs computation to avoid race conditions
                        // where all sub-groups would write to same output locations.
                        // Other sub-groups helped with cooperative data loading above.
                        if (sg_id == 0) {
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            for (int tm = 0; tm < TILES_M; tm++) {
                                auto slm_tokens_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                               sycl::access::decorated::no>(
                                    &slm_tokens[tm * XMX_M * XMX_K]);
                                joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                                for (int tn = 0; tn < TILES_N; tn++) {
                                    int row = wg_row + tm * XMX_M;
                                    int col = wg_col + tn * XMX_N;

                                    if (row < batch && col < out_dim) {
                                        auto slm_weights_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(
                                                &slm_weights[tn * XMX_N * XMX_K]);
                                        joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                        joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                        constexpr int acc_elements = XMX_M * XMX_N;
                                        int32_t *     acc_slm_raw =
                                            reinterpret_cast<int32_t *>(&slm_acc[0]);  // sg_id == 0, use offset 0
                                        auto acc_slm_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(acc_slm_raw);
                                        joint_matrix_store(sg, acc[tm][tn], acc_slm_ptr, XMX_N, layout::row_major);

                                        sycl::group_barrier(sg);

                                        for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                            int tile_row   = i / XMX_N;
                                            int tile_col   = i % XMX_N;
                                            int global_row = row + tile_row;
                                            int global_col = col + tile_col;
                                            if (global_row < batch && global_col < out_dim) {
                                                float   t_scale = slm_token_scales[tm * XMX_M + tile_row];
                                                float   w_scale = slm_weight_scales[tn * XMX_N + tile_col];
                                                int32_t raw     = acc_slm_raw[i];
                                                float_acc[tm][tn][i] += raw * t_scale * w_scale;
                                            }
                                        }

                                        joint_matrix_fill(sg, acc[tm][tn], 0);
                                    }
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }  // end K-loop

                    // === Final output store (only sub-group 0 writes) ===
                    if (sg_id == 0) {
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                int row = wg_row + tm * XMX_M;
                                int col = wg_col + tn * XMX_N;

                                if (row < batch && col < out_dim) {
                                    for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                        int tile_row = i / XMX_N;
                                        int tile_col = i % XMX_N;
                                        if (row + tile_row < batch && col + tile_col < out_dim) {
                                            output[(row + tile_row) * out_dim + col + tile_col] =
                                                sycl::half(float_acc[tm][tn][i]);
                                        }
                                    }
                                }

                                (void) acc[tm][tn];
                            }
                        }
                    }
                });
        })
        .wait();

}  // end launch_xmx_moe_gemm_q8_0_soa

// Q8_0 Coalesced XMX GEMM for a single expert's token batch
// Same as launch_xmx_moe_gemm_q8_0_soa but reads from coalesced-formatted weights
//
// Coalesced layout (per expert):
//   Weights are organized in tiles of TILE_BLOCKS=16 blocks.
//   Within each tile, bytes are reordered to word-major format:
//   - Word 0 (bytes 0-3): [B0.qs[0:3], B1.qs[0:3], ..., B15.qs[0:3]] = 64 bytes
//   - Word 1 (bytes 4-7): [B0.qs[4:7], B1.qs[4:7], ..., B15.qs[4:7]] = 64 bytes
//   ...
//   - Word 7 (bytes 28-31): [B0.qs[28:31], ..., B15.qs[28:31]] = 64 bytes
//   - Scales: stored contiguously after all quant tiles
//
// This layout enables coalesced memory access when warps load consecutive elements.
//
template <int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0_coalesced(
    const int8_t *       weights_qs,    // [tiles * TILE_BLOCKS * 32] coalesced quantized values
    const sycl::half *   weights_d,     // [out_dim * in_dim/32] coalesced scales
    const int8_t *       q_tokens,      // [batch, in_dim] pre-quantized int8
    const sycl::half *   token_scales,  // [batch, in_dim/32] token scales
    sycl::half *         output,        // [batch, out_dim]
    int64_t              batch,
    int64_t              out_dim,
    int64_t              in_dim,
    const MoEXMXConfig & cfg,
    sycl::queue &        queue) {
    constexpr int XMX_M             = 8;
    constexpr int XMX_N             = 16;
    constexpr int XMX_K             = 32;
    constexpr int SG_SIZE           = 16;
    constexpr int TILE_BLOCKS       = 16;               // Coalesced tile size
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;  // 64 bytes per word plane

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{ static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
                           static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    // Compute num_sgs for per-sub-group SLM allocation
    const int num_sgs = cfg.wg_size / SG_SIZE;

    queue
        .submit([&](sycl::handler & cgh) {
            // SLM for unpacked weight tiles
            constexpr int                   slm_weights_size = TILES_N * XMX_N * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

            // Per-sub-group SLM for accumulator extraction
            constexpr int                   slm_acc_per_sg = XMX_M * XMX_N * sizeof(int32_t);
            const int                       slm_acc_size   = num_sgs * slm_acc_per_sg;
            sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

            // SLM for token tiles
            sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(TILES_M * XMX_M * XMX_K), cgh);

            // SLM for scales
            sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);
            sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                    auto sg    = item.get_sub_group();
                    int  sg_id = sg.get_group_linear_id();

                    int wg_row = item.get_group(0) * wg_out_rows;
                    int wg_col = item.get_group(1) * wg_out_cols;

                    if (wg_row >= batch) {
                        return;
                    }

                    // Initialize accumulators
                    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];

                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tm][tn], 0);
                        }
                    }

                    float float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };

                    const int64_t num_k_blocks  = in_dim / XMX_K;
                    int           lane          = sg.get_local_linear_id();
                    int           num_sgs_local = cfg.wg_size / SG_SIZE;

                    for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                        int64_t k = k_block * XMX_K;

                        // === Cooperative token loading to SLM ===
                        constexpr int slm_tokens_size = TILES_M * XMX_M * XMX_K;
                        int           items_per_sg    = slm_tokens_size / num_sgs_local;
                        int           sg_offset       = sg_id * items_per_sg;
                        for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                            int idx = sg_offset + i + lane;
                            if (idx < slm_tokens_size) {
                                int     tile_row   = idx / XMX_K;
                                int     tile_k     = idx % XMX_K;
                                int     global_row = wg_row + tile_row;
                                int64_t global_k   = k + tile_k;
                                if (global_row < batch && global_k < in_dim) {
                                    slm_tokens[idx] = q_tokens[global_row * in_dim + global_k];
                                } else {
                                    slm_tokens[idx] = 0;
                                }
                            }
                        }

                        // === Load token scales for this K-block ===
                        if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                            int row_idx = sg_id * SG_SIZE + lane;
                            if (row_idx < TILES_M * XMX_M) {
                                int global_row = wg_row + row_idx;
                                if (global_row < batch) {
                                    slm_token_scales[row_idx] =
                                        static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                                } else {
                                    slm_token_scales[row_idx] = 0.0f;
                                }
                            }
                        }

                        // === Coalesced Weight Loading to SLM ===
                        // Coalesced layout: word-major within tiles of TILE_BLOCKS=16 blocks
                        // Address = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4 + byte
                        int weights_per_sg = slm_weights_size / num_sgs_local;
                        int w_sg_offset    = sg_id * weights_per_sg;

                        for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                            int idx = w_sg_offset + i + lane;
                            if (idx < slm_weights_size) {
                                int out_col_local = idx / XMX_K;
                                int k_elem        = idx % XMX_K;

                                int global_col = wg_col + out_col_local;

                                int8_t val = 0;
                                if (global_col < out_dim) {
                                    // Compute block index in linear order
                                    int64_t block_idx = global_col * num_k_blocks + k_block;

                                    // Coalesced addressing:
                                    int64_t tile_idx      = block_idx / TILE_BLOCKS;
                                    int     block_in_tile = block_idx % TILE_BLOCKS;
                                    int     word          = k_elem / 4;
                                    int     byte          = k_elem % 4;

                                    int64_t tile_base = tile_idx * (TILE_BLOCKS * QK8_0);  // 512 bytes per tile
                                    int64_t qs_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4 + byte;
                                    val               = weights_qs[qs_offset];
                                }
                                slm_weights[k_elem + out_col_local * XMX_K] = val;
                            }
                        }

                        // === Load coalesced scales for weight columns ===
                        // Scales are stored tile-major: after all quant data
                        if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                            int col_idx = sg_id * SG_SIZE + lane;
                            if (col_idx < TILES_N * XMX_N) {
                                int global_col = wg_col + col_idx;
                                if (global_col < out_dim) {
                                    // For coalesced, scales maintain the same linear indexing
                                    int64_t block_idx          = global_col * num_k_blocks + k_block;
                                    slm_weight_scales[col_idx] = static_cast<float>(weights_d[block_idx]);
                                } else {
                                    slm_weight_scales[col_idx] = 0.0f;
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        // === XMX Computation (only sub-group 0 computes) ===
                        // FIX: Only sg_id == 0 performs computation to avoid race conditions
                        // where all sub-groups would write to same output locations.
                        // Other sub-groups helped with cooperative data loading above.
                        if (sg_id == 0) {
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            for (int tm = 0; tm < TILES_M; tm++) {
                                auto slm_tokens_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                               sycl::access::decorated::no>(
                                    &slm_tokens[tm * XMX_M * XMX_K]);
                                joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                                for (int tn = 0; tn < TILES_N; tn++) {
                                    int row = wg_row + tm * XMX_M;
                                    int col = wg_col + tn * XMX_N;

                                    if (row < batch && col < out_dim) {
                                        auto slm_weights_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(
                                                &slm_weights[tn * XMX_N * XMX_K]);
                                        joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                        joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                        constexpr int acc_elements = XMX_M * XMX_N;
                                        int32_t *     acc_slm_raw =
                                            reinterpret_cast<int32_t *>(&slm_acc[0]);  // sg_id == 0, use offset 0
                                        auto acc_slm =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(acc_slm_raw);

                                        joint_matrix_store(sg, acc[tm][tn], acc_slm, XMX_N, layout::row_major);
                                        sg.barrier();

                                        for (int elem = lane; elem < acc_elements; elem += SG_SIZE) {
                                            int   m       = elem / XMX_N;
                                            int   n       = elem % XMX_N;
                                            float t_scale = slm_token_scales[tm * XMX_M + m];
                                            float w_scale = slm_weight_scales[tn * XMX_N + n];
                                            float_acc[tm][tn][elem] +=
                                                static_cast<float>(acc_slm[elem]) * t_scale * w_scale;
                                        }

                                        sg.barrier();
                                        joint_matrix_fill(sg, acc[tm][tn], 0);
                                    }
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // === Store output (only sub-group 0 writes) ===
                    if (sg_id == 0) {
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                int row = wg_row + tm * XMX_M;
                                int col = wg_col + tn * XMX_N;

                                if (row < batch && col < out_dim) {
                                    constexpr int acc_elements = XMX_M * XMX_N;
                                    for (int elem = lane; elem < acc_elements; elem += SG_SIZE) {
                                        int m       = elem / XMX_N;
                                        int n       = elem % XMX_N;
                                        int out_row = row + m;
                                        int out_col = col + n;
                                        if (out_row < batch && out_col < out_dim) {
                                            output[out_row * out_dim + out_col] =
                                                static_cast<sycl::half>(float_acc[tm][tn][elem]);
                                        }
                                    }
                                }

                                // Suppress unused variable warnings
                                (void) acc[tm][tn];
                            }
                        }
                    }
                });
        })
        .wait();

}  // end launch_xmx_moe_gemm_q8_0_coalesced

// MXFP4 XMX GEMM for a single expert's token batch
// Computes: output[batch, out_dim] = q_tokens[batch, in_dim] @ weights[in_dim, out_dim]
//
// SKELETON STATUS: Weight unpacking infrastructure implemented. GEMM logic pending.
// The following must be implemented before production use:
//   1. [DONE] MXFP4 unpacking: 4-bit E2M1 values -> int8 for XMX via kvalues_mxfp4 LUT
//   2. [DONE] Token input: Pre-quantized int8 tokens with per-block scales
//   3. [DONE] E8M0 exponent loading to SLM scales
//   4. [TODO] Proper joint_matrix_load for mat_a from SLM tokens
//   5. [TODO] Proper joint_matrix_load for mat_b from SLM unpacked weights
//   6. [TODO] Scale application during output: out = (int32_acc * token_scale * weight_scale)
//   7. [TODO] Conversion from scaled float to fp16 for storage
//
// MXFP4 block format (17 bytes per 32 elements):
//   - qs[16]: 32 4-bit values packed (2 per byte)
//   - e: 1-byte E8M0 shared exponent
//
// MXFP4 unpacking key insight:
//   kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
//   Values fit in int8, perfect for XMX int8 operands!
//   Low nibble (& 0xF) -> elements 0-15
//   High nibble (>> 4) -> elements 16-31
//
template <int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_mxfp4(const void * weights_qs,  // [in_dim/32, out_dim] MXFP4 packed (17 bytes per 32 elements)
                               const int8_t *       q_tokens,      // [batch, in_dim] pre-quantized int8
                               const sycl::half *   token_scales,  // [batch, in_dim/32] token scales
                               sycl::half *         output,        // [batch, out_dim]
                               int64_t              batch,
                               int64_t              out_dim,
                               int64_t              in_dim,
                               const MoEXMXConfig & cfg,
                               sycl::queue &        queue) {
    constexpr int XMX_M              = 8;
    constexpr int XMX_N              = 16;
    constexpr int XMX_K              = 32;
    constexpr int SG_SIZE            = 16;
    constexpr int MXFP4_PACKED_BYTES = 16;  // 16 packed bytes per 32 elements
    constexpr int MXFP4_BLOCK_STRIDE = 17;  // 16 bytes packed + 1 byte E8M0 exponent

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{ static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
                           static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    queue
        .submit([&](sycl::handler & cgh) {
            // SLM for unpacked weight tiles
            // One K-block of weights for TILES_N output columns: TILES_N * XMX_N * XMX_K int8 values
            // = 4 * 16 * 32 = 2048 bytes
            constexpr int                   slm_weights_size = TILES_N * XMX_N * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

            // Separate SLM for accumulator extraction - PER SUB-GROUP to avoid race conditions
            // Size: num_sgs * XMX_M * XMX_N * sizeof(int32_t) bytes
            // Each sub-group gets its own 512-byte section to prevent concurrent writes
            const int                       num_sgs_mxfp4        = cfg.wg_size / SG_SIZE;
            constexpr int                   slm_acc_per_sg_mxfp4 = XMX_M * XMX_N * sizeof(int32_t);
            const int                       slm_acc_size_mxfp4   = num_sgs_mxfp4 * slm_acc_per_sg_mxfp4;
            sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size_mxfp4), cgh);

            // SLM for token tiles (TILES_M * XMX_M rows * XMX_K cols)
            // = 4 * 8 * 32 = 1024 bytes
            constexpr int                   slm_tokens_size = TILES_M * XMX_M * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(slm_tokens_size), cgh);

            // SLM for E8M0 weight scales (one per output column per K-block)
            // TILES_N * XMX_N = 64 floats = 256 bytes
            sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);

            // SLM for token scales (one per row)
            // TILES_M * XMX_M = 32 floats = 128 bytes
            sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);

            // SLM for kvalues_mxfp4 LUT (16 int8 values)
            sycl::local_accessor<int8_t, 1> slm_kvalues(sycl::range<1>(16), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                    auto sg      = item.get_sub_group();
                    int  sg_id   = sg.get_group_linear_id();
                    int  lane    = sg.get_local_linear_id();
                    int  num_sgs = cfg.wg_size / SG_SIZE;

                    int wg_row = item.get_group(0) * wg_out_rows;
                    int wg_col = item.get_group(1) * wg_out_cols;

                    // Bounds check
                    if (wg_row >= batch) {
                        return;
                    }

                    // === Load kvalues_mxfp4 LUT into SLM (once per work-group) ===
                    if (sg_id == 0 && lane < 16) {
                        slm_kvalues[lane] = kvalues_mxfp4[lane];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Initialize accumulators
                    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];

                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tm][tn], 0);
                        }
                    }

                    // Float accumulators for precision across K-blocks
                    float float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };

                    // K-dimension reduction loop
                    // Each iteration processes XMX_K=32 elements along the reduction dimension
                    // For MXFP4: 32 elements = 1 block = 17 bytes (16 packed + 1 E8M0)
                    const uint8_t * w_ptr        = static_cast<const uint8_t *>(weights_qs);
                    const int64_t   num_k_blocks = in_dim / XMX_K;

                    for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                        int64_t k = k_block * XMX_K;

                        // === Cooperative token loading to SLM ===
                        // Load pre-quantized int8 tokens (same as Q8_0 kernel)
                        int items_per_sg = slm_tokens_size / num_sgs;
                        int sg_offset    = sg_id * items_per_sg;
                        for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                            int idx = sg_offset + i + lane;
                            if (idx < slm_tokens_size) {
                                int     tile_row   = idx / XMX_K;
                                int     tile_k     = idx % XMX_K;
                                int     global_row = wg_row + tile_row;
                                int64_t global_k   = k + tile_k;
                                if (global_row < batch && global_k < in_dim) {
                                    slm_tokens[idx] = q_tokens[global_row * in_dim + global_k];
                                } else {
                                    slm_tokens[idx] = 0;
                                }
                            }
                        }

                        // === Load token scales for this K-block ===
                        if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                            int row_idx = sg_id * SG_SIZE + lane;
                            if (row_idx < TILES_M * XMX_M) {
                                int global_row = wg_row + row_idx;
                                if (global_row < batch) {
                                    slm_token_scales[row_idx] =
                                        static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                                } else {
                                    slm_token_scales[row_idx] = 0.0f;
                                }
                            }
                        }

                        // === MXFP4 Weight Unpacking to SLM ===
                        // Each sub-group handles part of the weight tile
                        // Weight layout: [in_dim, out_dim] with MXFP4 blocks along in_dim
                        // Each block = 17 bytes (16 packed nibbles + 1 E8M0 exponent)
                        // For TILES_N * XMX_N output columns, we need to unpack
                        // TILES_N * XMX_N blocks (one block per column for this K-block)
                        //
                        // Total elements to unpack: TILES_N * XMX_N * XMX_K = 64 * 32 = 2048
                        // Each work item unpacks multiple elements
                        int weights_per_sg = slm_weights_size / num_sgs;
                        int w_sg_offset    = sg_id * weights_per_sg;

                        for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                            int idx = w_sg_offset + i + lane;
                            if (idx < slm_weights_size) {
                                // Decode which output column and K-element this is
                                int out_col_local = idx / XMX_K;  // 0..63 (within TILES_N * XMX_N)
                                int k_elem        = idx % XMX_K;  // 0..31

                                int global_col = wg_col + out_col_local;

                                int8_t unpacked_val = 0;
                                if (global_col < out_dim) {
                                    // Calculate MXFP4 block address for this (out_col, k_block)
                                    // Weight layout: [in_dim, out_dim] - ggml tensors are row-major with ne[0] fastest
                                    // For MXFP4: in_dim elements quantized to in_dim/32 blocks per output row
                                    // Block at (out_col, k_block) = data[out_col * num_k_blocks + k_block]
                                    int64_t         block_offset = global_col * num_k_blocks + k_block;
                                    const uint8_t * block_ptr    = w_ptr + block_offset * MXFP4_BLOCK_STRIDE;

                                    // MXFP4 unpacking:
                                    // block_mxfp4 layout: [e:1 byte][qs:16 bytes]
                                    // So qs starts at offset 1 from block_ptr
                                    //
                                    // Each qs byte contains 2 nibbles:
                                    //   - Low nibble (& 0xF) of qs[i] -> element i (for i in 0..15)
                                    //   - High nibble (>> 4) of qs[i] -> element i+16 (for i in 0..15)
                                    // So for k_elem in [0..31]:
                                    //   - byte_idx = k_elem % 16
                                    //   - use low nibble for k_elem < 16, high nibble for k_elem >= 16
                                    int     byte_idx    = k_elem % MXFP4_PACKED_BYTES;  // k_elem % 16
                                    uint8_t packed_byte = block_ptr[1 + byte_idx];      // qs starts at offset 1
                                    uint8_t nibble =
                                        (k_elem < MXFP4_PACKED_BYTES) ? (packed_byte & 0xF) : (packed_byte >> 4);

                                    // LUT lookup for int8 value
                                    unpacked_val = slm_kvalues[nibble];
                                }
                                // Store in column-major order for XMX mat_b (col_major layout)
                                // col_major[K,N]: element (k, n) stored at [k + n * K]
                                slm_weights[k_elem + out_col_local * XMX_K] = unpacked_val;
                            }
                        }

                        // === Load E8M0 scales for weight columns ===
                        // One scale per output column (TILES_N * XMX_N = 64 scales)
                        if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                            int col_idx = sg_id * SG_SIZE + lane;
                            if (col_idx < TILES_N * XMX_N) {
                                int global_col = wg_col + col_idx;
                                if (global_col < out_dim) {
                                    // Weight layout: [in_dim, out_dim] - ggml tensors are row-major with ne[0] fastest
                                    // Block at (out_col, k_block) = data[out_col * num_k_blocks + k_block]
                                    // block_mxfp4 layout: [e:1 byte][qs:16 bytes], so e is at offset 0
                                    int64_t         block_offset = global_col * num_k_blocks + k_block;
                                    const uint8_t * block_ptr    = w_ptr + block_offset * MXFP4_BLOCK_STRIDE;
                                    uint8_t         e8m0         = block_ptr[0];  // E8M0 is FIRST byte (offset 0)

                                    // sycl_e8m0_to_fp32_half includes the 0.5 factor for kvalues
                                    slm_weight_scales[col_idx] = sycl_e8m0_to_fp32_half(e8m0);
                                } else {
                                    slm_weight_scales[col_idx] = 0.0f;
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        // === XMX Computation (only sub-group 0 computes) ===
                        // FIX: Only sg_id == 0 performs computation to avoid race conditions
                        // where all sub-groups would write to same output locations.
                        // Other sub-groups helped with cooperative data loading above.
                        if (sg_id == 0) {
                            // Declare joint matrices for this K-tile
                            // mat_a: row_major [M, K] - tokens stored as [row * K + k]
                            // mat_b: col_major [K, N] - weights stored as [col * K + k] in SLM
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            // Compute tiles
                            for (int tm = 0; tm < TILES_M; tm++) {
                                // Load mat_a from SLM tokens
                                auto slm_tokens_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                               sycl::access::decorated::no>(
                                    &slm_tokens[tm * XMX_M * XMX_K]);
                                joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                                for (int tn = 0; tn < TILES_N; tn++) {
                                    int row = wg_row + tm * XMX_M;
                                    int col = wg_col + tn * XMX_N;

                                    if (row < batch && col < out_dim) {
                                        // Load mat_b from SLM unpacked weights
                                        // Weight tile layout: [TILES_N * XMX_N, XMX_K] in col_major
                                        // slm_weights[col * XMX_K + k] for col_major access
                                        auto slm_weights_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(
                                                &slm_weights[tn * XMX_N * XMX_K]);
                                        joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                        // XMX multiply-accumulate: acc += mat_a * mat_b
                                        joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                        // Store accumulator to SLM section to extract values
                                        constexpr int acc_elements = XMX_M * XMX_N;
                                        int32_t *     acc_slm_raw =
                                            reinterpret_cast<int32_t *>(&slm_acc[0]);  // sg_id == 0, use offset 0
                                        auto acc_slm_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(acc_slm_raw);
                                        joint_matrix_store(sg, acc[tm][tn], acc_slm_ptr, XMX_N, layout::row_major);

                                        // Sub-group barrier is sufficient
                                        sycl::group_barrier(sg);

                                        // Apply scales and accumulate in float
                                        // Note: slm_weight_scales[tn * XMX_N + tile_col] for per-column scales
                                        for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                            int tile_row   = i / XMX_N;
                                            int tile_col   = i % XMX_N;
                                            int global_row = row + tile_row;
                                            int global_col = col + tile_col;
                                            if (global_row < batch && global_col < out_dim) {
                                                float   t_scale = slm_token_scales[tm * XMX_M + tile_row];
                                                float   w_scale = slm_weight_scales[tn * XMX_N + tile_col];
                                                int32_t raw     = acc_slm_raw[i];
                                                float_acc[tm][tn][i] += raw * t_scale * w_scale;
                                            }
                                        }

                                        // Reset accumulator for next K-block
                                        joint_matrix_fill(sg, acc[tm][tn], 0);
                                    }
                                }
                            }
                        }

                        // Barrier before next K-iteration
                        item.barrier(sycl::access::fence_space::local_space);
                    }  // end K-loop

                    // === Final output store (only sub-group 0 writes) ===
                    // MoE scatter expects: sorted_output[pair_idx * output_dim + dim]
                    // This differs from MMQ which outputs column-major to ggml tensors
                    if (sg_id == 0) {
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                int row = wg_row + tm * XMX_M;
                                int col = wg_col + tn * XMX_N;

                                if (row < batch && col < out_dim) {
                                    for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                        int tile_row = i / XMX_N;
                                        int tile_col = i % XMX_N;
                                        if (row + tile_row < batch && col + tile_col < out_dim) {
                                            // Row-major: output[row * out_dim + col]
                                            output[(row + tile_row) * out_dim + col + tile_col] =
                                                sycl::half(float_acc[tm][tn][i]);
                                        }
                                    }
                                }

                                // Suppress unused accumulator warning
                                (void) acc[tm][tn];
                            }
                        }
                    }
                });
        })
        .wait();
}

// MXFP4 SoA XMX GEMM for a single expert's token batch
// Same as launch_xmx_moe_gemm_mxfp4 but reads from SoA-formatted weights
//
// SoA layout (per expert):
//   - qs: [nblocks * 16] packed nibble bytes contiguously
//   - e:  [nblocks] E8M0 exponents contiguously
// Where nblocks = out_dim * (in_dim / 32)
//
// Block indexing: block_idx = out_col * num_k_blocks + k_block
//   - qs offset: block_idx * 16
//   - e offset: block_idx
//
template <int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_mxfp4_soa(const uint8_t *      weights_qs,    // [nblocks * 16] SoA packed nibbles
                                   const uint8_t *      weights_e,     // [nblocks] SoA E8M0 exponents
                                   const int8_t *       q_tokens,      // [batch, in_dim] pre-quantized int8
                                   const sycl::half *   token_scales,  // [batch, in_dim/32] token scales
                                   sycl::half *         output,        // [batch, out_dim]
                                   int64_t              batch,
                                   int64_t              out_dim,
                                   int64_t              in_dim,
                                   const MoEXMXConfig & cfg,
                                   sycl::queue &        queue) {
    constexpr int XMX_M              = 8;
    constexpr int XMX_N              = 16;
    constexpr int XMX_K              = 32;
    constexpr int SG_SIZE            = 16;
    constexpr int MXFP4_PACKED_BYTES = 16;  // 16 packed bytes per 32 elements

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;
    int wg_out_cols = TILES_N * XMX_N;

    sycl::range<2> global{ static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
                           static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    const int num_sgs = cfg.wg_size / SG_SIZE;

    queue
        .submit([&](sycl::handler & cgh) {
            constexpr int                   slm_weights_size = TILES_N * XMX_N * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

            constexpr int                   slm_acc_per_sg = XMX_M * XMX_N * sizeof(int32_t);
            const int                       slm_acc_size   = num_sgs * slm_acc_per_sg;
            sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

            constexpr int                   slm_tokens_size = TILES_M * XMX_M * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(slm_tokens_size), cgh);

            sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);
            sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);

            sycl::local_accessor<int8_t, 1> slm_kvalues(sycl::range<1>(16), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(
                                                                   SG_SIZE)]] {
                auto sg            = item.get_sub_group();
                int  sg_id         = sg.get_group_linear_id();
                int  lane          = sg.get_local_linear_id();
                int  num_sgs_local = cfg.wg_size / SG_SIZE;

                int wg_row = item.get_group(0) * wg_out_rows;
                int wg_col = item.get_group(1) * wg_out_cols;

                if (wg_row >= batch) {
                    return;
                }

                // Load kvalues LUT
                if (sg_id == 0 && lane < 16) {
                    slm_kvalues[lane] = kvalues_mxfp4[lane];
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Initialize accumulators
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];

                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        joint_matrix_fill(sg, acc[tm][tn], 0);
                    }
                }

                float         float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };
                const int64_t num_k_blocks                               = in_dim / XMX_K;

                for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                    int64_t k = k_block * XMX_K;

                    // === Cooperative token loading (SoA kernel) ===
                    int items_per_sg = slm_tokens_size / num_sgs_local;
                    int sg_offset    = sg_id * items_per_sg;
                    for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                        int idx = sg_offset + i + lane;
                        if (idx < slm_tokens_size) {
                            int     tile_row   = idx / XMX_K;
                            int     tile_k     = idx % XMX_K;
                            int     global_row = wg_row + tile_row;
                            int64_t global_k   = k + tile_k;
                            if (global_row < batch && global_k < in_dim) {
                                int8_t tok_val  = q_tokens[global_row * in_dim + global_k];
                                slm_tokens[idx] = tok_val;
                            } else {
                                slm_tokens[idx] = 0;
                            }
                        }
                    }

                    // === Token scales (SoA kernel) ===
                    if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                        int row_idx = sg_id * SG_SIZE + lane;
                        if (row_idx < TILES_M * XMX_M) {
                            int global_row = wg_row + row_idx;
                            if (global_row < batch) {
                                float t_scale = static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                                slm_token_scales[row_idx] = t_scale;
                            } else {
                                slm_token_scales[row_idx] = 0.0f;
                            }
                        }
                    }

                    // === SoA MXFP4 Weight Unpacking ===
                    // SoA: qs at block_idx * 16, exponents at weights_e[block_idx]
                    int weights_per_sg = slm_weights_size / num_sgs_local;
                    int w_sg_offset    = sg_id * weights_per_sg;

                    for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                        int idx = w_sg_offset + i + lane;
                        if (idx < slm_weights_size) {
                            int out_col_local = idx / XMX_K;
                            int k_elem        = idx % XMX_K;

                            int global_col = wg_col + out_col_local;

                            int8_t unpacked_val = 0;
                            if (global_col < out_dim) {
                                int64_t block_idx = global_col * num_k_blocks + k_block;

                                // SoA: qs are contiguous, 16 bytes per block
                                int64_t qs_offset   = block_idx * MXFP4_PACKED_BYTES;
                                int     byte_idx    = k_elem % MXFP4_PACKED_BYTES;
                                uint8_t packed_byte = weights_qs[qs_offset + byte_idx];
                                uint8_t nibble =
                                    (k_elem < MXFP4_PACKED_BYTES) ? (packed_byte & 0xF) : (packed_byte >> 4);

                                unpacked_val = slm_kvalues[nibble];
                            }
                            slm_weights[k_elem + out_col_local * XMX_K] = unpacked_val;
                        }
                    }

                    // === SoA E8M0 scales ===
                    if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                        int col_idx = sg_id * SG_SIZE + lane;
                        if (col_idx < TILES_N * XMX_N) {
                            int global_col = wg_col + col_idx;
                            if (global_col < out_dim) {
                                int64_t block_idx          = global_col * num_k_blocks + k_block;
                                // SoA: exponents are contiguous
                                uint8_t e8m0               = weights_e[block_idx];
                                float   w_scale            = sycl_e8m0_to_fp32_half(e8m0);
                                slm_weight_scales[col_idx] = w_scale;
                            } else {
                                slm_weight_scales[col_idx] = 0.0f;
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    // === XMX Computation (only sub-group 0 computes) ===
                    // FIX: Only sg_id == 0 performs computation to avoid race conditions
                    // where all sub-groups would write to same output locations.
                    // Other sub-groups helped with cooperative data loading above.
                    if (sg_id == 0) {
                        joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                        joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                        for (int tm = 0; tm < TILES_M; tm++) {
                            auto slm_tokens_ptr =
                                sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::no>(&slm_tokens[tm * XMX_M * XMX_K]);
                            joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                int row = wg_row + tm * XMX_M;
                                int col = wg_col + tn * XMX_N;

                                if (row < batch && col < out_dim) {
                                    auto slm_weights_ptr =
                                        sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                 sycl::access::decorated::no>(
                                            &slm_weights[tn * XMX_N * XMX_K]);
                                    joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                    joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                    constexpr int acc_elements = XMX_M * XMX_N;
                                    int32_t *     acc_slm_raw =
                                        reinterpret_cast<int32_t *>(&slm_acc[0]);  // sg_id == 0, use offset 0
                                    auto acc_slm = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                            sycl::access::decorated::no>(acc_slm_raw);

                                    joint_matrix_store(sg, acc[tm][tn], acc_slm, XMX_N, layout::row_major);
                                    sg.barrier();

                                    for (int elem = lane; elem < acc_elements; elem += SG_SIZE) {
                                        int     m       = elem / XMX_N;
                                        int     n       = elem % XMX_N;
                                        float   t_scale = slm_token_scales[tm * XMX_M + m];
                                        float   w_scale = slm_weight_scales[tn * XMX_N + n];
                                        int32_t raw_acc = acc_slm[elem];
                                        float   scaled  = static_cast<float>(raw_acc) * t_scale * w_scale;
                                        float_acc[tm][tn][elem] += scaled;
                                    }

                                    sg.barrier();
                                    joint_matrix_fill(sg, acc[tm][tn], 0);
                                }
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                // === Store output (only sub-group 0 writes) ===
                if (sg_id == 0) {
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int row = wg_row + tm * XMX_M;
                            int col = wg_col + tn * XMX_N;

                            if (row < batch && col < out_dim) {
                                constexpr int acc_elements = XMX_M * XMX_N;
                                for (int elem = lane; elem < acc_elements; elem += SG_SIZE) {
                                    int m       = elem / XMX_N;
                                    int n       = elem % XMX_N;
                                    int out_row = row + m;
                                    int out_col = col + n;
                                    if (out_row < batch && out_col < out_dim) {
                                        output[out_row * out_dim + out_col] =
                                            static_cast<sycl::half>(float_acc[tm][tn][elem]);
                                    }
                                }
                            }
                            (void) acc[tm][tn];
                        }
                    }
                }
            });
        })
        .wait();
}  // end launch_xmx_moe_gemm_mxfp4_soa

// MXFP4 Coalesced XMX GEMM for a single expert's token batch
// Same as launch_xmx_moe_gemm_mxfp4_soa but reads from coalesced-formatted weights
//
// Coalesced layout (per expert):
//   Weights are organized in tiles of TILE_BLOCKS=16 blocks.
//   Within each tile, packed nibble bytes are reordered to word-major format.
//   Exponents are stored contiguously after all packed data.
//
// For MXFP4 with 16 bytes per block:
//   - Word plane stride = TILE_BLOCKS * 4 = 64 bytes (only 4 "words" since 16 bytes = 4 × 4)
//   - Layout: [W0:B0..B15][W1:B0..B15][W2:B0..B15][W3:B0..B15] per tile
//
template <int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_mxfp4_coalesced(
    const uint8_t *      weights_qs,    // [tiles * TILE_BLOCKS * 16] coalesced packed nibbles
    const uint8_t *      weights_e,     // [nblocks] coalesced E8M0 exponents
    const int8_t *       q_tokens,      // [batch, in_dim] pre-quantized int8
    const sycl::half *   token_scales,  // [batch, in_dim/32] token scales
    sycl::half *         output,        // [batch, out_dim]
    int64_t              batch,
    int64_t              out_dim,
    int64_t              in_dim,
    const MoEXMXConfig & cfg,
    sycl::queue &        queue) {
    constexpr int XMX_M              = 8;
    constexpr int XMX_N              = 16;
    constexpr int XMX_K              = 32;
    constexpr int SG_SIZE            = 16;
    constexpr int MXFP4_PACKED_BYTES = 16;
    constexpr int TILE_BLOCKS        = 16;
    constexpr int WORD_PLANE_STRIDE  = TILE_BLOCKS * 4;  // 64 bytes per word plane

    int wg_out_rows = TILES_M * XMX_M;
    int wg_out_cols = TILES_N * XMX_N;

    sycl::range<2> global{ static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
                           static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    const int num_sgs = cfg.wg_size / SG_SIZE;

    queue
        .submit([&](sycl::handler & cgh) {
            constexpr int                   slm_weights_size = TILES_N * XMX_N * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

            constexpr int                   slm_acc_per_sg = XMX_M * XMX_N * sizeof(int32_t);
            const int                       slm_acc_size   = num_sgs * slm_acc_per_sg;
            sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

            constexpr int                   slm_tokens_size = TILES_M * XMX_M * XMX_K;
            sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(slm_tokens_size), cgh);

            sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);
            sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);

            sycl::local_accessor<int8_t, 1> slm_kvalues(sycl::range<1>(16), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                    auto sg            = item.get_sub_group();
                    int  sg_id         = sg.get_group_linear_id();
                    int  lane          = sg.get_local_linear_id();
                    int  num_sgs_local = cfg.wg_size / SG_SIZE;

                    int wg_row = item.get_group(0) * wg_out_rows;
                    int wg_col = item.get_group(1) * wg_out_cols;

                    if (wg_row >= batch) {
                        return;
                    }

                    // Load kvalues LUT
                    if (sg_id == 0 && lane < 16) {
                        slm_kvalues[lane] = kvalues_mxfp4[lane];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Initialize accumulators
                    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];

                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tm][tn], 0);
                        }
                    }

                    float         float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };
                    const int64_t num_k_blocks                               = in_dim / XMX_K;

                    for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                        int64_t k = k_block * XMX_K;

                        // === Cooperative token loading ===
                        int items_per_sg = slm_tokens_size / num_sgs_local;
                        int sg_offset    = sg_id * items_per_sg;
                        for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                            int idx = sg_offset + i + lane;
                            if (idx < slm_tokens_size) {
                                int     tile_row   = idx / XMX_K;
                                int     tile_k     = idx % XMX_K;
                                int     global_row = wg_row + tile_row;
                                int64_t global_k   = k + tile_k;
                                if (global_row < batch && global_k < in_dim) {
                                    slm_tokens[idx] = q_tokens[global_row * in_dim + global_k];
                                } else {
                                    slm_tokens[idx] = 0;
                                }
                            }
                        }

                        // === Token scales ===
                        if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                            int row_idx = sg_id * SG_SIZE + lane;
                            if (row_idx < TILES_M * XMX_M) {
                                int global_row = wg_row + row_idx;
                                if (global_row < batch) {
                                    slm_token_scales[row_idx] =
                                        static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                                } else {
                                    slm_token_scales[row_idx] = 0.0f;
                                }
                            }
                        }

                        // === Coalesced MXFP4 Weight Unpacking ===
                        // Coalesced: word-major within tiles of TILE_BLOCKS=16 blocks
                        // MXFP4 has only 16 bytes per block = 4 "words" of 4 bytes each
                        int weights_per_sg = slm_weights_size / num_sgs_local;
                        int w_sg_offset    = sg_id * weights_per_sg;

                        for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                            int idx = w_sg_offset + i + lane;
                            if (idx < slm_weights_size) {
                                int out_col_local = idx / XMX_K;
                                int k_elem        = idx % XMX_K;

                                int global_col = wg_col + out_col_local;

                                int8_t unpacked_val = 0;
                                if (global_col < out_dim) {
                                    int64_t block_idx = global_col * num_k_blocks + k_block;

                                    // Which packed byte contains this k_elem?
                                    // k_elem 0-15 -> low nibbles of bytes 0-15
                                    // k_elem 16-31 -> high nibbles of bytes 0-15
                                    int byte_idx = k_elem % MXFP4_PACKED_BYTES;

                                    // Coalesced addressing for packed bytes:
                                    int64_t tile_idx      = block_idx / TILE_BLOCKS;
                                    int     block_in_tile = block_idx % TILE_BLOCKS;
                                    int     word          = byte_idx / 4;
                                    int     byte_in_word  = byte_idx % 4;

                                    int64_t tile_base = tile_idx * (TILE_BLOCKS * MXFP4_PACKED_BYTES);
                                    int64_t qs_offset =
                                        tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word;

                                    uint8_t packed_byte = weights_qs[qs_offset];
                                    uint8_t nibble =
                                        (k_elem < MXFP4_PACKED_BYTES) ? (packed_byte & 0xF) : (packed_byte >> 4);

                                    unpacked_val = slm_kvalues[nibble];
                                }
                                slm_weights[k_elem + out_col_local * XMX_K] = unpacked_val;
                            }
                        }

                        // === Coalesced E8M0 scales ===
                        // For simplicity, exponents maintain linear indexing (same as SoA)
                        if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                            int col_idx = sg_id * SG_SIZE + lane;
                            if (col_idx < TILES_N * XMX_N) {
                                int global_col = wg_col + col_idx;
                                if (global_col < out_dim) {
                                    int64_t block_idx          = global_col * num_k_blocks + k_block;
                                    uint8_t e8m0               = weights_e[block_idx];
                                    slm_weight_scales[col_idx] = sycl_e8m0_to_fp32_half(e8m0);
                                } else {
                                    slm_weight_scales[col_idx] = 0.0f;
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        // === XMX Computation (only sub-group 0 computes) ===
                        // FIX: Only sg_id == 0 performs computation to avoid race conditions
                        // where all sub-groups would write to same output locations.
                        // Other sub-groups helped with cooperative data loading above.
                        if (sg_id == 0) {
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            for (int tm = 0; tm < TILES_M; tm++) {
                                auto slm_tokens_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                               sycl::access::decorated::no>(
                                    &slm_tokens[tm * XMX_M * XMX_K]);
                                joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                                for (int tn = 0; tn < TILES_N; tn++) {
                                    int row = wg_row + tm * XMX_M;
                                    int col = wg_col + tn * XMX_N;

                                    if (row < batch && col < out_dim) {
                                        auto slm_weights_ptr =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(
                                                &slm_weights[tn * XMX_N * XMX_K]);
                                        joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                        joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                        constexpr int acc_elements = XMX_M * XMX_N;
                                        int32_t *     acc_slm_raw =
                                            reinterpret_cast<int32_t *>(&slm_acc[0]);  // sg_id == 0, use offset 0
                                        auto acc_slm =
                                            sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                     sycl::access::decorated::no>(acc_slm_raw);

                                        joint_matrix_store(sg, acc[tm][tn], acc_slm, XMX_N, layout::row_major);
                                        sg.barrier();

                                        for (int elem = lane; elem < acc_elements; elem += SG_SIZE) {
                                            int   m       = elem / XMX_N;
                                            int   n       = elem % XMX_N;
                                            float t_scale = slm_token_scales[tm * XMX_M + m];
                                            float w_scale = slm_weight_scales[tn * XMX_N + n];
                                            float_acc[tm][tn][elem] +=
                                                static_cast<float>(acc_slm[elem]) * t_scale * w_scale;
                                        }

                                        sg.barrier();
                                        joint_matrix_fill(sg, acc[tm][tn], 0);
                                    }
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // === Store output (only sub-group 0 writes) ===
                    if (sg_id == 0) {
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                int row = wg_row + tm * XMX_M;
                                int col = wg_col + tn * XMX_N;

                                if (row < batch && col < out_dim) {
                                    constexpr int acc_elements = XMX_M * XMX_N;
                                    for (int elem = lane; elem < acc_elements; elem += SG_SIZE) {
                                        int m       = elem / XMX_N;
                                        int n       = elem % XMX_N;
                                        int out_row = row + m;
                                        int out_col = col + n;
                                        if (out_row < batch && out_col < out_dim) {
                                            output[out_row * out_dim + out_col] =
                                                static_cast<sycl::half>(float_acc[tm][tn][elem]);
                                        }
                                    }
                                }
                                (void) acc[tm][tn];
                            }
                        }
                    }
                });
        })
        .wait();
}  // end launch_xmx_moe_gemm_mxfp4_coalesced

}  // namespace moe_xmx

#endif  // SYCL_XMX_MOE_AVAILABLE
