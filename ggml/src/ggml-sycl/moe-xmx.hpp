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

// Q8_0 XMX GEMM for a single expert's token batch
// Computes: output[batch, out_dim] = tokens[batch, in_dim] @ weights[out_dim, in_dim]^T
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(
    const void* weights_qs,       // [out_dim, in_dim] int8 quantized
    const sycl::half* weights_d,  // [out_dim, in_dim/32] scales
    const sycl::half* tokens,     // [batch, in_dim]
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

                // Suppress unused variable warning
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

                // K-dimension reduction
                const int8_t* w_ptr = static_cast<const int8_t*>(weights_qs);

                for (int64_t k = 0; k < in_dim; k += XMX_K) {
                    // Load weight and token tiles
                    joint_matrix<sycl::sub_group, int8_t, use::a,
                                 XMX_M, XMX_K, layout::row_major> mat_a;
                    joint_matrix<sycl::sub_group, int8_t, use::b,
                                 XMX_K, XMX_N, layout::row_major> mat_b;

                    // Compute tiles
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int row = wg_row + tm * XMX_M;
                            int col = wg_col + tn * XMX_N;

                            if (row < batch && col < out_dim) {
                                // Load A (tokens) - need to quantize on the fly or use pre-quantized
                                // Load B (weights)
                                joint_matrix_load(sg, mat_b,
                                    w_ptr + col * in_dim + k,
                                    static_cast<size_t>(in_dim));

                                // XMX multiply-accumulate
                                joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
                            }
                        }
                    }

                    sycl::group_barrier(item.get_group());
                }

                // Store results with scale application
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        int row = wg_row + tm * XMX_M;
                        int col = wg_col + tn * XMX_N;

                        if (row < batch && col < out_dim) {
                            // TODO: Apply Q8_0 scales and store as fp16
                            joint_matrix_store(sg, acc[tm][tn],
                                reinterpret_cast<int32_t*>(output + row * out_dim + col),
                                static_cast<size_t>(out_dim),
                                layout::row_major);
                        }
                    }
                }
            });
    }).wait();

    // Suppress unused parameter warnings
    (void)weights_d;
    (void)tokens;
}

} // namespace moe_xmx

#endif // SYCL_XMX_MOE_AVAILABLE
