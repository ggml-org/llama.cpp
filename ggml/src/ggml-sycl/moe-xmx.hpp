// moe-xmx.hpp - XMX-accelerated MoE GEMM kernel
#pragma once

#include "common.hpp"
#include "moe-sort.hpp"
#include <sycl/sycl.hpp>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#define SYCL_XMX_MOE_AVAILABLE 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#define SYCL_XMX_MOE_AVAILABLE 0
#endif

#if SYCL_XMX_MOE_AVAILABLE

namespace moe_xmx {

using namespace sycl::ext::oneapi::experimental::matrix;

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
    int slm_token_bytes = 4 * 1024;    // 4KB for token tile

    static MoEXMXConfig from_capabilities(const XMXCapabilities& caps) {
        MoEXMXConfig cfg;
        if (caps.M > 0) cfg.M = static_cast<int>(caps.M);
        if (caps.N > 0) cfg.N = static_cast<int>(caps.N);
        if (caps.K > 0) cfg.K = static_cast<int>(caps.K);
        cfg.tiles_m = caps.optimal_tiles_m;
        cfg.tiles_n = caps.optimal_tiles_n;
        return cfg;
    }
};

// Pre-quantize fp16 tokens to int8 with per-block scales
// Output: q_tokens[batch * in_dim] int8, scales[batch * (in_dim/32)] fp16
void preprocess_tokens_q8(
    const sycl::half* tokens,   // [batch, in_dim] fp16 input
    int8_t* q_tokens,           // [batch, in_dim] int8 output
    sycl::half* scales,         // [batch, in_dim/32] per-block scales
    int64_t batch,
    int64_t in_dim,
    sycl::queue& queue);

inline void preprocess_tokens_q8(
    const sycl::half* tokens,
    int8_t* q_tokens,
    sycl::half* scales,
    int64_t batch,
    int64_t in_dim,
    sycl::queue& queue)
{
    constexpr int QK = 32;      // Quantization block size (matches XMX_K)
    constexpr int SG_SIZE = 16;

    int64_t num_blocks = batch * (in_dim / QK);

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_blocks * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int64_t block_id = item.get_group(0);
                int64_t row = block_id / (in_dim / QK);
                int64_t k_block = block_id % (in_dim / QK);

                int lane = sg.get_local_linear_id();

                // Each lane loads 2 values (32 total per sub-group)
                int64_t base = row * in_dim + k_block * QK;
                float v0 = static_cast<float>(tokens[base + lane * 2]);
                float v1 = static_cast<float>(tokens[base + lane * 2 + 1]);

                // Find max absolute value via sub-group reduction
                float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
                float amax = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                // Compute scale and inverse scale
                float scale = amax / 127.0f;
                float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;

                // Quantize values
                int8_t q0 = static_cast<int8_t>(sycl::round(v0 * inv_scale));
                int8_t q1 = static_cast<int8_t>(sycl::round(v1 * inv_scale));

                // Store quantized values
                q_tokens[base + lane * 2] = q0;
                q_tokens[base + lane * 2 + 1] = q1;

                // Store scale (one per block, lane 0 only)
                if (lane == 0) {
                    scales[row * (in_dim / QK) + k_block] = sycl::half(scale);
                }
            });
    });
}

// Q8_0 XMX GEMM for a single expert's token batch
// Computes: output[batch, out_dim] = tokens[batch, in_dim] @ weights[out_dim, in_dim]^T
//
// SKELETON STATUS: This kernel compiles but produces INCORRECT output.
// The following must be implemented before production use:
//   1. Token quantization (fp16 -> int8) with scale computation
//   2. Proper joint_matrix_load for mat_a from quantized tokens
//   3. Scale application during output: out = (int32_acc * token_scale * weight_scale)
//   4. Conversion from scaled float to fp16 for storage
//
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(
    const void* weights_qs,       // [out_dim, in_dim] int8 quantized
    const sycl::half* weights_d,  // [out_dim, in_dim/32] Q8_0 scales (1 per 32 elements)
    const sycl::half* tokens,     // [batch, in_dim] fp16 activations
    sycl::half* output,           // [batch, out_dim]
    int64_t batch,
    int64_t out_dim,
    int64_t in_dim,
    const MoEXMXConfig& cfg,
    sycl::queue& queue)
{
    constexpr int XMX_M = 8;
    constexpr int XMX_N = 16;
    constexpr int XMX_K = 32;
    constexpr int SG_SIZE = 16;

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{
        static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
        static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols)
    };
    sycl::range<2> local{static_cast<size_t>(cfg.wg_size), 1};

    queue.submit([&](sycl::handler& cgh) {
        // SLM for weight tiles (double-buffered)
        sycl::local_accessor<int8_t, 1> slm_weights(
            sycl::range<1>(cfg.slm_weight_bytes), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int sg_id = sg.get_group_linear_id();

                int wg_row = item.get_group(0) * wg_out_rows;
                int wg_col = item.get_group(1) * wg_out_cols;

                // Bounds check
                if (wg_row >= batch) return;

                // TODO: Use sg_id for sub-group cooperative loading of weight tiles into SLM
                // TODO: Use slm_weights for double-buffered weight prefetching
                (void)sg_id;
                (void)slm_weights;

                // Initialize accumulators
                joint_matrix<sycl::sub_group, int32_t, use::accumulator,
                             XMX_M, XMX_N> acc[TILES_M][TILES_N];

                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        joint_matrix_fill(sg, acc[tm][tn], 0);
                    }
                }

                // K-dimension reduction loop
                // Each iteration processes XMX_K=32 elements along the reduction dimension
                const int8_t* w_ptr = static_cast<const int8_t*>(weights_qs);

                for (int64_t k = 0; k < in_dim; k += XMX_K) {
                    // Declare joint matrices for this K-tile
                    joint_matrix<sycl::sub_group, int8_t, use::a,
                                 XMX_M, XMX_K, layout::row_major> mat_a;
                    joint_matrix<sycl::sub_group, int8_t, use::b,
                                 XMX_K, XMX_N, layout::row_major> mat_b;

                    // TODO: Load mat_a from tokens
                    // Current tokens are fp16, but XMX requires int8. Options:
                    //   Option A: Pre-quantize tokens to int8 before kernel launch
                    //   Option B: Quantize on-the-fly in SLM (expensive but flexible)
                    // For now, fill with zeros as placeholder (produces zero output)
                    joint_matrix_fill(sg, mat_a, static_cast<int8_t>(0));

                    // Compute tiles - iterate over M and N tile positions
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int row = wg_row + tm * XMX_M;
                            int col = wg_col + tn * XMX_N;

                            if (row < batch && col < out_dim) {
                                // Load B (weights) from global memory
                                // weights_qs layout: [out_dim, in_dim] row-major int8
                                auto w_offset = w_ptr + col * in_dim + k;
                                sycl::multi_ptr<const int8_t, sycl::access::address_space::global_space> w_mptr(w_offset);
                                joint_matrix_load(sg, mat_b, w_mptr, static_cast<size_t>(in_dim));

                                // XMX multiply-accumulate: acc += mat_a * mat_b
                                joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
                            }
                        }
                    }

                    // Note: No barrier needed here - each sub-group works independently
                    // TODO: Add barrier when implementing cooperative SLM weight loading
                }

                // TODO: Store results with scale application
                // The accumulator contains int32 dot products. To get fp16 output:
                //   1. Extract int32 values from accumulator
                //   2. For each K-block, multiply by (token_scale[k/32] * weight_scale[k/32])
                //   3. Sum scaled values across K dimension
                //   4. Convert final float to fp16 and store
                //
                // For now, write zeros as placeholder (safe, predictable output)
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        int row = wg_row + tm * XMX_M;
                        int col = wg_col + tn * XMX_N;

                        if (row < batch && col < out_dim) {
                            // Placeholder: write zeros instead of unscaled int32 garbage
                            // Each sub-group lane writes its portion of the tile
                            int lane = sg.get_local_linear_id();
                            for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                int tile_row = i / XMX_N;
                                int tile_col = i % XMX_N;
                                if (row + tile_row < batch && col + tile_col < out_dim) {
                                    output[(row + tile_row) * out_dim + col + tile_col] =
                                        sycl::half(0.0f);
                                }
                            }
                        }

                        // Suppress unused accumulator warning (will be used when scaling is implemented)
                        (void)acc[tm][tn];
                    }
                }
            });
    }).wait();

    // TODO: tokens parameter will be used for:
    //   - Loading fp16 activations to quantize into mat_a
    //   - Computing per-block token scales for dequantization
    (void)tokens;

    // TODO: weights_d parameter will be used for:
    //   - Loading Q8_0 weight scales (1 scale per 32 elements)
    //   - Multiplying with token scales during output dequantization
    (void)weights_d;
}

// MXFP4 XMX GEMM for a single expert's token batch
// Computes: output[batch, out_dim] = tokens[batch, in_dim] @ weights[out_dim, in_dim]^T
//
// SKELETON STATUS: This kernel compiles but produces INCORRECT output.
// The following must be implemented before production use:
//   1. MXFP4 unpacking: 4-bit E2M1 values -> fp16 or int8 for XMX
//   2. Token quantization (fp16 -> int8) with scale computation
//   3. Proper joint_matrix_load for mat_a and mat_b
//   4. E8M0 exponent handling during output dequantization
//   5. Conversion from scaled float to fp16 for storage
//
// MXFP4 block format (17 bytes per 32 elements):
//   - qs[16]: 32 4-bit values packed (2 per byte)
//   - e: 1-byte E8M0 shared exponent
//
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_mxfp4(
    const void* weights_qs,       // [out_dim, in_dim] MXFP4 quantized (17 bytes per 32 elements)
    const sycl::half* tokens,     // [batch, in_dim] fp16 activations
    sycl::half* output,           // [batch, out_dim]
    int64_t batch,
    int64_t out_dim,
    int64_t in_dim,
    const MoEXMXConfig& cfg,
    sycl::queue& queue)
{
    constexpr int XMX_M = 8;
    constexpr int XMX_N = 16;
    constexpr int XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int MXFP4_BLOCK_SIZE = 17;  // 16 bytes packed + 1 byte E8M0 exponent

    // Work-group grid
    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    sycl::range<2> global{
        static_cast<size_t>((batch + wg_out_rows - 1) / wg_out_rows * cfg.wg_size),
        static_cast<size_t>((out_dim + wg_out_cols - 1) / wg_out_cols)
    };
    sycl::range<2> local{static_cast<size_t>(cfg.wg_size), 1};

    queue.submit([&](sycl::handler& cgh) {
        // SLM for unpacked weight tiles (double-buffered)
        // After unpacking: 32 int8 values per block (vs 17 bytes packed)
        sycl::local_accessor<int8_t, 1> slm_weights(
            sycl::range<1>(cfg.slm_weight_bytes), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int sg_id = sg.get_group_linear_id();

                int wg_row = item.get_group(0) * wg_out_rows;
                int wg_col = item.get_group(1) * wg_out_cols;

                // Bounds check
                if (wg_row >= batch) return;

                // TODO: Use sg_id for sub-group cooperative loading of weight tiles into SLM
                // TODO: Unpack MXFP4 4-bit values to int8 in SLM for XMX consumption
                (void)sg_id;
                (void)slm_weights;
                (void)MXFP4_BLOCK_SIZE;

                // Initialize accumulators
                joint_matrix<sycl::sub_group, int32_t, use::accumulator,
                             XMX_M, XMX_N> acc[TILES_M][TILES_N];

                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        joint_matrix_fill(sg, acc[tm][tn], 0);
                    }
                }

                // K-dimension reduction loop
                // Each iteration processes XMX_K=32 elements along the reduction dimension
                // For MXFP4: 32 elements = 1 block = 17 bytes (16 packed + 1 E8M0)
                const uint8_t* w_ptr = static_cast<const uint8_t*>(weights_qs);

                for (int64_t k = 0; k < in_dim; k += XMX_K) {
                    // Declare joint matrices for this K-tile
                    joint_matrix<sycl::sub_group, int8_t, use::a,
                                 XMX_M, XMX_K, layout::row_major> mat_a;
                    joint_matrix<sycl::sub_group, int8_t, use::b,
                                 XMX_K, XMX_N, layout::row_major> mat_b;

                    // TODO: Load mat_a from tokens (fp16 -> int8 quantization)
                    // TODO: Load mat_b from MXFP4 weights (4-bit -> int8 unpacking)
                    //
                    // MXFP4 unpacking requires:
                    //   1. Extract 4-bit nibbles from packed bytes
                    //   2. Use kvalues_mxfp4 LUT to get E2M1 mantissa values
                    //   3. Apply E8M0 shared exponent
                    //   4. Quantize result to int8 for XMX
                    //
                    // For now, fill with zeros as placeholder (produces zero output)
                    joint_matrix_fill(sg, mat_a, static_cast<int8_t>(0));
                    joint_matrix_fill(sg, mat_b, static_cast<int8_t>(0));

                    // Compute tiles
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int row = wg_row + tm * XMX_M;
                            int col = wg_col + tn * XMX_N;

                            if (row < batch && col < out_dim) {
                                // XMX multiply-accumulate: acc += mat_a * mat_b
                                joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
                            }
                        }
                    }
                }

                // Store results
                // TODO: Apply MXFP4 dequantization scales and convert to fp16
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        int row = wg_row + tm * XMX_M;
                        int col = wg_col + tn * XMX_N;

                        if (row < batch && col < out_dim) {
                            // Placeholder: write zeros instead of unscaled int32 garbage
                            int lane = sg.get_local_linear_id();
                            for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                int tile_row = i / XMX_N;
                                int tile_col = i % XMX_N;
                                if (row + tile_row < batch && col + tile_col < out_dim) {
                                    output[(row + tile_row) * out_dim + col + tile_col] =
                                        sycl::half(0.0f);
                                }
                            }
                        }

                        // Suppress unused accumulator warning
                        (void)acc[tm][tn];
                    }
                }

                // Suppress unused pointer warning
                (void)w_ptr;
            });
    }).wait();

    // TODO: tokens parameter will be used for quantization
    (void)tokens;
}

} // namespace moe_xmx

#endif // SYCL_XMX_MOE_AVAILABLE
