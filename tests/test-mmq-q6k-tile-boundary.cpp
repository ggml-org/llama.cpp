// Q6_K MMQ Tile Boundary Test
// Tests multi-tile iteration to isolate where SoA kernel diverges from AoS
//
// Key insight from test-mmq-q6k-gpu failures:
//   - Batch=2: Token 1 fails
//   - Batch=8: Tokens 3,5 fail (scattered pattern)
//   - Component tests (tile loading, vec_dot) pass
// This suggests the bug is in multi-tile coordination, not individual tile ops.
//
// Build: cmake --build build --target test-mmq-q6k-tile-boundary
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-q6k-tile-boundary

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"

#define QK_K 256
#define MMQ_Y 128  // Rows per Y-tile (from mmq.cpp)
#define MMQ_X 64   // Tokens per X-tile

// Test configuration
struct TileTestConfig {
    int n_rows;      // Total rows (should span multiple Y-tiles)
    int n_cols;      // Columns (K dimension, must be multiple of QK_K)
    int n_tokens;    // Batch size
    const char* name;
    bool use_random; // Use random data like test-mmq-q6k-gpu
};

static bool test_tile_boundary(const TileTestConfig& cfg, bool verbose) {
    printf("\n=== %s ===\n", cfg.name);
    printf("  Dimensions: rows=%d (%.1f Y-tiles), cols=%d, tokens=%d\n",
           cfg.n_rows, (float)cfg.n_rows / MMQ_Y, cfg.n_cols, cfg.n_tokens);

    // How many Y-tiles does this span?
    int n_y_tiles = (cfg.n_rows + MMQ_Y - 1) / MMQ_Y;
    printf("  Y-tiles: %d (MMQ_Y=%d)\n", n_y_tiles, MMQ_Y);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Weight tensor [n_cols, n_rows] in Q6_K
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, cfg.n_cols, cfg.n_rows);
    ggml_set_name(weight, "weight");

    // Input tensor [n_cols, n_tokens] in F32
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_cols, cfg.n_tokens);
    ggml_set_name(input, "input");

    // Output: MUL_MAT(weight, input) -> [n_rows, n_tokens]
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "output");

    // Allocate weight buffer with SoA reordering
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate weight data
    const int blocks_per_row = cfg.n_cols / QK_K;
    const int total_blocks = cfg.n_rows * blocks_per_row;
    std::vector<block_q6_K> weight_data(total_blocks);

    std::mt19937 rng(42);  // Same seed as test-mmq-q6k-gpu

    if (cfg.use_random) {
        // Random data like test-mmq-q6k-gpu
        printf("  Data mode: RANDOM (like test-mmq-q6k-gpu)\n");
        for (int i = 0; i < total_blocks; i++) {
            for (int j = 0; j < QK_K/2; j++) weight_data[i].ql[j] = rng() & 0xFF;
            for (int j = 0; j < QK_K/4; j++) weight_data[i].qh[j] = rng() & 0x3F;
            for (int j = 0; j < QK_K/16; j++) weight_data[i].scales[j] = (rng() % 64) - 32;
            weight_data[i].d = ggml_fp32_to_fp16((rng() % 100) / 100.0f);
        }
    } else {
        // Predictable pattern per tile
        printf("  Data mode: PREDICTABLE (tile-based pattern)\n");
        for (int row = 0; row < cfg.n_rows; row++) {
            int tile_id = row / MMQ_Y;
            int row_in_tile = row % MMQ_Y;

            for (int blk = 0; blk < blocks_per_row; blk++) {
                block_q6_K& b = weight_data[row * blocks_per_row + blk];
                for (int i = 0; i < QK_K/2; i++) {
                    b.ql[i] = (uint8_t)((tile_id * 17 + row_in_tile + i) & 0xFF);
                }
                for (int i = 0; i < QK_K/4; i++) {
                    b.qh[i] = (uint8_t)((tile_id * 13 + row_in_tile + i) & 0x3F);
                }
                for (int i = 0; i < QK_K/16; i++) {
                    b.scales[i] = (int8_t)((tile_id * 7 + i) % 64 - 32);
                }
                b.d = ggml_fp32_to_fp16(1.0f + 0.01f * tile_id);
            }
        }
    }

    // Upload weights (triggers SoA transformation)
    ggml_backend_tensor_set(weight, weight_data.data(), 0, total_blocks * sizeof(block_q6_K));

    // Generate input
    std::vector<float> input_data(cfg.n_cols * cfg.n_tokens);
    if (cfg.use_random) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : input_data) v = dist(rng);
    } else {
        for (int t = 0; t < cfg.n_tokens; t++) {
            for (int k = 0; k < cfg.n_cols; k++) {
                input_data[t * cfg.n_cols + k] = 1.0f + 0.001f * (t + k);
            }
        }
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Build and execute graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_graph_compute(backend, graph);

    // Get GPU output
    std::vector<float> gpu_output(cfg.n_rows * cfg.n_tokens);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    // CPU reference
    std::vector<float> x_f32(cfg.n_cols);
    std::vector<float> cpu_output(cfg.n_rows * cfg.n_tokens);

    for (int row = 0; row < cfg.n_rows; row++) {
        dequantize_row_q6_K(&weight_data[row * blocks_per_row], x_f32.data(), cfg.n_cols);

        for (int tok = 0; tok < cfg.n_tokens; tok++) {
            float sum = 0.0f;
            for (int k = 0; k < cfg.n_cols; k++) {
                sum += x_f32[k] * input_data[tok * cfg.n_cols + k];
            }
            cpu_output[tok * cfg.n_rows + row] = sum;
        }
    }

    // Analyze errors per Y-tile
    bool all_pass = true;
    printf("\n  Per-tile analysis:\n");

    for (int tile = 0; tile < n_y_tiles; tile++) {
        int row_start = tile * MMQ_Y;
        int row_end = std::min(row_start + MMQ_Y, cfg.n_rows);
        int tile_rows = row_end - row_start;

        int tile_mismatches = 0;
        float tile_max_err = 0.0f;

        for (int tok = 0; tok < cfg.n_tokens; tok++) {
            for (int row = row_start; row < row_end; row++) {
                int idx = tok * cfg.n_rows + row;
                float diff = std::fabs(gpu_output[idx] - cpu_output[idx]);
                float rel_err = std::fabs(cpu_output[idx]) > 1e-6 ?
                    diff / std::fabs(cpu_output[idx]) : diff;

                if (diff > 0.1f && rel_err > 0.15f) {
                    tile_mismatches++;
                    tile_max_err = std::max(tile_max_err, diff);
                }
            }
        }

        int tile_total = tile_rows * cfg.n_tokens;
        bool tile_ok = (tile_mismatches == 0);

        printf("    Y-tile %d (rows %d-%d): %s (%d/%d mismatches, max_err=%.2f)\n",
               tile, row_start, row_end - 1,
               tile_ok ? "PASS" : "FAIL",
               tile_mismatches, tile_total, tile_max_err);

        if (!tile_ok) {
            all_pass = false;

            // Show first few mismatches in this tile
            if (verbose) {
                int shown = 0;
                for (int tok = 0; tok < cfg.n_tokens && shown < 3; tok++) {
                    for (int row = row_start; row < row_end && shown < 3; row++) {
                        int idx = tok * cfg.n_rows + row;
                        float diff = std::fabs(gpu_output[idx] - cpu_output[idx]);
                        float rel_err = std::fabs(cpu_output[idx]) > 1e-6 ?
                            diff / std::fabs(cpu_output[idx]) : diff;
                        if (diff > 0.1f && rel_err > 0.15f) {
                            printf("      tok=%d row=%d: GPU=%.4f CPU=%.4f diff=%.4f\n",
                                   tok, row, gpu_output[idx], cpu_output[idx], diff);
                            shown++;
                        }
                    }
                }
            }
        }
    }

    // Also analyze per-token
    printf("\n  Per-token analysis:\n");
    for (int tok = 0; tok < cfg.n_tokens; tok++) {
        int tok_mismatches = 0;
        float tok_max_err = 0.0f;
        int first_bad_row = -1;

        for (int row = 0; row < cfg.n_rows; row++) {
            int idx = tok * cfg.n_rows + row;
            float diff = std::fabs(gpu_output[idx] - cpu_output[idx]);
            float rel_err = std::fabs(cpu_output[idx]) > 1e-6 ?
                diff / std::fabs(cpu_output[idx]) : diff;

            if (diff > 0.1f && rel_err > 0.15f) {
                if (first_bad_row < 0) first_bad_row = row;
                tok_mismatches++;
                tok_max_err = std::max(tok_max_err, diff);
            }
        }

        bool tok_ok = (tok_mismatches == 0);
        printf("    Token %d: %s (%d/%d mismatches",
               tok, tok_ok ? "PASS" : "FAIL", tok_mismatches, cfg.n_rows);
        if (!tok_ok && first_bad_row >= 0) {
            printf(", first_bad_row=%d (tile %d)", first_bad_row, first_bad_row / MMQ_Y);
        }
        printf(")\n");
    }

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("\n  Result: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

int main() {
    printf("Q6_K MMQ Tile Boundary Test\n");
    printf("===========================\n");
    printf("MMQ_Y=%d (rows per tile), MMQ_X=%d (tokens per tile)\n", MMQ_Y, MMQ_X);

    int passed = 0, failed = 0;

    // Test 1: Single Y-tile (baseline) - predictable
    TileTestConfig cfg1 = {
        .n_rows = 128, .n_cols = 512, .n_tokens = 4,
        .name = "Test 1: Single Y-tile (128 rows, predictable)",
        .use_random = false
    };
    if (test_tile_boundary(cfg1, true)) passed++; else failed++;

    // Test 2: Two Y-tiles - predictable
    TileTestConfig cfg2 = {
        .n_rows = 256, .n_cols = 512, .n_tokens = 4,
        .name = "Test 2: Two Y-tiles (256 rows, predictable)",
        .use_random = false
    };
    if (test_tile_boundary(cfg2, true)) passed++; else failed++;

    // Test 3: Match test-mmq-q6k-gpu dimensions - predictable
    TileTestConfig cfg3 = {
        .n_rows = 2048, .n_cols = 1024, .n_tokens = 8,
        .name = "Test 3: test-mmq-q6k-gpu dims (2048x1024, predictable)",
        .use_random = false
    };
    if (test_tile_boundary(cfg3, true)) passed++; else failed++;

    // Test 4: Match test-mmq-q6k-gpu dimensions - RANDOM (should fail if bug is data-dependent)
    TileTestConfig cfg4 = {
        .n_rows = 2048, .n_cols = 1024, .n_tokens = 8,
        .name = "Test 4: test-mmq-q6k-gpu dims (2048x1024, RANDOM)",
        .use_random = true
    };
    if (test_tile_boundary(cfg4, true)) passed++; else failed++;

    // Test 5: Large with random data
    TileTestConfig cfg5 = {
        .n_rows = 4096, .n_cols = 1024, .n_tokens = 4,
        .name = "Test 5: Large tensor (4096 rows, RANDOM)",
        .use_random = true
    };
    if (test_tile_boundary(cfg5, true)) passed++; else failed++;

    // Test 6: Small with random data
    TileTestConfig cfg6 = {
        .n_rows = 256, .n_cols = 512, .n_tokens = 4,
        .name = "Test 6: Small tensor (256 rows, RANDOM)",
        .use_random = true
    };
    if (test_tile_boundary(cfg6, true)) passed++; else failed++;

    printf("\n=================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
