// tests/test-dmmv-q4-0-coalesced.cpp
// Test that Q4_0 DMMV with coalesced layout produces same output as SoA
//
// This is a TDD test - written to fail until the coalesced DMMV kernel
// is properly integrated and produces correct results.
//
// The coalesced layout reorganizes SoA data into tile-based format for
// better cache line utilization during DMMV operations.
//
// Build: cmake --build build --target test-dmmv-q4-0-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-dmmv-q4-0-coalesced

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Q4_0 block structure (must match ggml-common.h)
#define QK4_0 32
#define QR4_0 2

// Coalesced tile configuration (must match common.hpp)
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
constexpr int WARP_SIZE = GGML_SYCL_WARP_SIZE;
constexpr int MMVQ_COALESCED_TILE_BLOCKS     = WARP_SIZE;
constexpr int MMVQ_COALESCED_TILE_BYTES_Q4_0 = MMVQ_COALESCED_TILE_BLOCKS * 16;

typedef struct {
    ggml_fp16_t d;
    uint8_t     qs[QK4_0 / 2];
} block_q4_0_test;

static_assert(sizeof(block_q4_0_test) == 18, "block_q4_0 size mismatch");

// =============================================================================
// CPU Reference Implementation
// =============================================================================

static void dequantize_block_q4_0_cpu(const block_q4_0_test * block, float * out) {
    const float d = ggml_fp16_to_fp32(block->d);
    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        const int     lo   = (byte & 0xF);
        const int     hi   = (byte >> 4);
        out[i]             = (lo - 8) * d;
        out[i + QK4_0 / 2] = (hi - 8) * d;
    }
}

static void dmmv_q4_0_cpu_reference(const block_q4_0_test * x, const float * y, float * dst, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0_test * block = &x[row * blocks_per_row + b];
            float                   dequant[QK4_0];
            dequantize_block_q4_0_cpu(block, dequant);

            for (int i = 0; i < QK4_0; i++) {
                sum += dequant[i] * y[b * QK4_0 + i];
            }
        }
        dst[row] = sum;
    }
}

static void dmmv_q4_0_cpu_from_coalesced(const uint8_t * coalesced, const float * y, float * dst,
                                         int ncols, int nrows) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int word_stride = TILE_BLOCKS * 4;
    const int row_bytes = ncols / 2;

    const uint8_t * x_qs = coalesced;
    const ggml_fp16_t * x_d = reinterpret_cast<const ggml_fp16_t *>(coalesced + nrows * row_bytes);

    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;

        for (int tile = 0; tile < tiles_per_row; ++tile) {
            const int tile_base = row * row_bytes + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int block_in_tile = 0; block_in_tile < TILE_BLOCKS; ++block_in_tile) {
                const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                const float d = ggml_fp16_to_fp32(x_d[block_idx]);
                const int y_base = (tile * TILE_BLOCKS + block_in_tile) * QK4_0;

                uint8_t qs[QK4_0 / 2];
                for (int word = 0; word < 4; ++word) {
                    const int word_offset = word * word_stride + block_in_tile * 4;
                    const uint8_t * word_ptr = x_qs + tile_base + word_offset;
                    memcpy(qs + word * 4, word_ptr, 4);
                }

                float block_sum = 0.0f;
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    const float v0 = (float)((byte & 0xF) - 8);
                    const float v1 = (float)((byte >> 4) - 8);
                    block_sum += v0 * y[y_base + j];
                    block_sum += v1 * y[y_base + j + 16];
                }

                sum += d * block_sum;
            }
        }

        dst[row] = sum;
    }
}

static void dmmv_q4_0_cpu_from_coalesced_q8_0(const uint8_t * coalesced, const block_q8_0 * y, float * dst,
                                              int ncols, int nrows) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int word_stride = TILE_BLOCKS * 4;
    const int row_bytes = ncols / 2;

    const uint8_t * x_qs = coalesced;
    const ggml_fp16_t * x_d = reinterpret_cast<const ggml_fp16_t *>(coalesced + nrows * row_bytes);

    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;

        for (int tile = 0; tile < tiles_per_row; ++tile) {
            const int tile_base = row * row_bytes + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int block_in_tile = 0; block_in_tile < TILE_BLOCKS; ++block_in_tile) {
                const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                const float d = ggml_fp16_to_fp32(x_d[block_idx]);

                uint8_t qs[QK4_0 / 2];
                for (int word = 0; word < 4; ++word) {
                    const int word_offset = word * word_stride + block_in_tile * 4;
                    const uint8_t * word_ptr = x_qs + tile_base + word_offset;
                    memcpy(qs + word * 4, word_ptr, 4);
                }

                const block_q8_0 * y_blk = y + (tile * TILE_BLOCKS + block_in_tile);
                int sumi = 0;
                for (int j = 0; j < QK4_0 / 2; ++j) {
                    const uint8_t byte = qs[j];
                    const int v0 = (byte & 0xF) - 8;
                    const int v1 = (byte >> 4) - 8;
                    sumi += v0 * y_blk->qs[j];
                    sumi += v1 * y_blk->qs[j + QK4_0 / 2];
                }

                sum += (float)sumi * d * ggml_fp16_to_fp32(y_blk->d);
            }
        }

        dst[row] = sum;
    }
}

static void reorder_q4_0_aos_bytes_to_coalesced(const uint8_t * aos, uint8_t * coalesced,
                                                int ncols, int nrows) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int WORDS_PER_BLOCK = 4;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks = nrows * blocks_per_row;
    const int row_quants_bytes = ncols / 2;
    const int total_quants_bytes = nrows * row_quants_bytes;

    for (int ib = 0; ib < total_blocks; ++ib) {
        const int row = ib / blocks_per_row;
        const int col_block = ib % blocks_per_row;
        const int tile = col_block / TILE_BLOCKS;
        const int block_in_tile = col_block % TILE_BLOCKS;

        const int64_t tile_base = (int64_t)row * row_quants_bytes +
                                  (int64_t)tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        const uint8_t * src_qs = aos + (int64_t)ib * sizeof(block_q4_0_test) + sizeof(ggml_fp16_t);
        for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
            const int64_t dst_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coalesced + dst_offset, src_qs + word * 4, 4);
        }

        const uint8_t * src_d = aos + (int64_t)ib * sizeof(block_q4_0_test);
        memcpy(coalesced + total_quants_bytes + (int64_t)ib * sizeof(ggml_fp16_t), src_d, sizeof(ggml_fp16_t));
    }
}

static void compare_device_coalesced_layout(const uint8_t * aos, const uint8_t * device_coalesced,
                                            int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks = nrows * blocks_per_row;
    const size_t row_quants_bytes = ncols / 2;
    const size_t total_quants_bytes = (size_t)nrows * row_quants_bytes;
    const size_t expected_bytes = total_quants_bytes + (size_t)total_blocks * sizeof(ggml_fp16_t);

    std::vector<uint8_t> expected(expected_bytes);
    reorder_q4_0_aos_bytes_to_coalesced(aos, expected.data(), ncols, nrows);

    int errors = 0;
    size_t first_bad = 0;
    for (size_t i = 0; i < expected_bytes; ++i) {
        if (expected[i] != device_coalesced[i]) {
            if (errors == 0) {
                first_bad = i;
            }
            errors++;
            if (errors > 10) {
                break;
            }
        }
    }

    if (errors == 0) {
        printf("  Coalesced layout check: PASSED (bytes=%zu)\n", expected_bytes);
    } else {
        printf("  Coalesced layout check: FAILED (errors=%d, first_bad=%zu expected=0x%02x got=0x%02x)\n",
               errors, first_bad, expected[first_bad], device_coalesced[first_bad]);
    }
}

// =============================================================================
// GGML Backend Helpers (DMMV path)
// =============================================================================

static bool run_mul_mat_backend(ggml_backend_t backend, ggml_type weight_type,
                                const void * weight_data, size_t weight_size,
                                const float * input_data, int n_embd, int n_rows,
                                std::vector<float> & output,
                                std::vector<uint8_t> * weight_out = nullptr) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_rows);
    ggml_set_name(weight, "weight");

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_name(input, "input");

    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out, "output");

    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, out);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, out, base + input_size);

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);
    ggml_backend_tensor_set(input, input_data, 0, n_embd * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool success = (status == GGML_STATUS_SUCCESS);
    if (success) {
        output.resize(n_rows);
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
        if (weight_out) {
            fprintf(stderr, "[LAYOUT-CHECK] readback start: weight_size=%zu ptr=%p\n",
                    weight_size, weight->data);
            weight_out->resize(weight_size);
            ggml_backend_sycl_memcpy_d2h(weight, weight_out->data(), weight_size);
            fprintf(stderr, "[LAYOUT-CHECK] readback done\n");
        }
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return success;
}

static bool run_dmmv_coalesced_case(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend,
                                    int ncols, int nrows, bool verbose) {
    const auto * qfns = ggml_get_type_traits(GGML_TYPE_Q4_0);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    if (!qfns || !qfns_cpu || !qfns_cpu->from_float || qfns->blck_size == 0) {
        printf("  SKIP: Q4_0 quantization functions not available\n");
        return true;
    }

    std::mt19937 rng(1234 + ncols + nrows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_float(nrows * ncols);
    for (float & v : weight_float) v = dist(rng);

    size_t block_size = qfns->blck_size;
    size_t type_size = qfns->type_size;
    size_t n_elements = weight_float.size();
    size_t nblocks = n_elements / block_size;
    size_t weight_bytes = nblocks * type_size;

    std::vector<uint8_t> weight_quant(weight_bytes);
    qfns_cpu->from_float(weight_float.data(), weight_quant.data(), n_elements);

    std::vector<float> input_data(ncols);
    for (float & v : input_data) v = dist(rng);

    std::vector<float> gpu_out;
    std::vector<float> cpu_out;
    std::vector<float> cpu_ref;
    const bool debug_coalesced = std::getenv("GGML_SYCL_COALESCED_DMMV_DEBUG") != nullptr;
    const bool layout_check = std::getenv("GGML_SYCL_COALESCED_LAYOUT_CHECK") != nullptr;
    std::vector<uint8_t> weight_coalesced;
    if (!run_mul_mat_backend(gpu_backend, GGML_TYPE_Q4_0, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, gpu_out,
                             (debug_coalesced || layout_check) ? &weight_coalesced : nullptr)) {
        printf("  FAIL: GPU backend compute failed\n");
        return false;
    }
    if (!run_mul_mat_backend(cpu_backend, GGML_TYPE_Q4_0, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, cpu_out)) {
        printf("  FAIL: CPU backend compute failed\n");
        return false;
    }

    cpu_ref.resize(nrows);
    dmmv_q4_0_cpu_reference(reinterpret_cast<const block_q4_0_test *>(weight_quant.data()),
                            input_data.data(), cpu_ref.data(), ncols, nrows);

    if (debug_coalesced && !weight_coalesced.empty()) {
        float max_diff_ref = 0.0f;
        int errors_ref = 0;
        for (int i = 0; i < nrows; ++i) {
            float diff = fabsf(cpu_ref[i] - cpu_out[i]);
            if (diff > max_diff_ref) max_diff_ref = diff;
            if (diff > 1e-4f) {
                errors_ref++;
            }
        }
        printf("  Debug (manual CPU vs CPU backend, expected mismatch): errors=%d max_diff=%.6f\n",
               errors_ref, max_diff_ref);

        const int blocks_per_row = ncols / QK4_0;
        std::vector<block_q8_0> y_q8(blocks_per_row);
        quantize_row_q8_0_ref(input_data.data(), y_q8.data(), ncols);

        std::vector<float> cpu_from_coalesced(nrows);
        dmmv_q4_0_cpu_from_coalesced_q8_0(weight_coalesced.data(), y_q8.data(), cpu_from_coalesced.data(),
                                          ncols, nrows);

        float max_diff_cpu = 0.0f;
        int errors_cpu = 0;
        for (int i = 0; i < nrows; ++i) {
            float diff = fabsf(cpu_from_coalesced[i] - cpu_out[i]);
            if (diff > max_diff_cpu) max_diff_cpu = diff;
            if (diff > 1e-2f) {
                errors_cpu++;
            }
        }
        printf("  Debug (coalesced CPU vs CPU backend): errors=%d max_diff=%.6f\n", errors_cpu, max_diff_cpu);

        float max_diff = 0.0f;
        int errors = 0;
        for (int i = 0; i < nrows; ++i) {
            float diff = fabsf(cpu_from_coalesced[i] - gpu_out[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-2f) {
                errors++;
            }
        }
        printf("  Debug (coalesced CPU vs GPU): errors=%d max_diff=%.6f\n", errors, max_diff);
    }
    if (layout_check && !weight_coalesced.empty()) {
        compare_device_coalesced_layout(weight_quant.data(), weight_coalesced.data(), ncols, nrows);
    }

    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int errors = 0;
    const std::vector<float> & ref = cpu_out;
    for (int i = 0; i < nrows; ++i) {
        float diff = fabsf(gpu_out[i] - ref[i]);
        float denom = fmaxf(1.0f, fabsf(ref[i]));
        float rel = diff / denom;
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > 1e-2f && rel > 1e-2f) {
            errors++;
        }
    }

    bool pass = (errors == 0);
    if (verbose || !pass) {
        printf("  DMMV coalesced ncols=%d nrows=%d: errors=%d max_diff=%.6f max_rel=%.6f %s\n",
               ncols, nrows, errors, max_diff, max_rel, pass ? "PASS" : "FAIL");
    }
    return pass;
}

// =============================================================================
// SoA Layout Conversion (AoS -> SoA)
// =============================================================================

static void convert_aos_to_soa(const block_q4_0_test * aos_data, uint8_t * soa_data, int total_blocks) {
    // SoA layout: all quants first, then all scales
    const size_t qs_bytes = total_blocks * (QK4_0 / 2);

    // Copy quants
    for (int b = 0; b < total_blocks; b++) {
        for (int i = 0; i < QK4_0 / 2; i++) {
            soa_data[b * (QK4_0 / 2) + i] = aos_data[b].qs[i];
        }
    }

    // Copy scales
    ggml_fp16_t * scales = reinterpret_cast<ggml_fp16_t *>(soa_data + qs_bytes);
    for (int b = 0; b < total_blocks; b++) {
        scales[b] = aos_data[b].d;
    }
}

// =============================================================================
// Coalesced Layout Conversion (SoA -> Coalesced)
// =============================================================================

static void convert_soa_to_coalesced(const uint8_t * soa_data, uint8_t * coalesced_data, int ncols, int nrows) {
    const int    blocks_per_row = ncols / QK4_0;
    const int    total_blocks   = nrows * blocks_per_row;
    const size_t qs_bytes       = total_blocks * (QK4_0 / 2);

    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;                   // 16 bytes quants per Q4_0 block
    constexpr int WORDS_PER_BLOCK = BYTES_PER_BLOCK / 4;         // 4 words per block

    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Coalesced row stride: quants only (scales stored globally after all quants)
    const int coalesced_row_stride = blocks_per_row * BYTES_PER_BLOCK;

    // SoA pointers
    const uint8_t *    soa_qs     = soa_data;
    const ggml_fp16_t * soa_scales = reinterpret_cast<const ggml_fp16_t *>(soa_data + qs_bytes);

    for (int row = 0; row < nrows; row++) {
        uint8_t * row_out = coalesced_data + row * coalesced_row_stride;

        for (int tile = 0; tile < tiles_per_row; tile++) {
            uint8_t * tile_out = row_out + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            // Coalesced layout within tile:
            // Word w of block b is at: word_offset = w * (TILE_BLOCKS * 4) + b * 4
            // So all word-0s are together, then all word-1s, etc.

            for (int b = 0; b < TILE_BLOCKS; b++) {
                int             global_block = row * blocks_per_row + tile * TILE_BLOCKS + b;
                const uint8_t * src_block    = soa_qs + global_block * BYTES_PER_BLOCK;

                // Reorder words within tile
                for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                    int dst_offset = w * (TILE_BLOCKS * 4) + b * 4;
                    memcpy(tile_out + dst_offset, src_block + w * 4, 4);
                }
            }
        }

        // Scales are stored globally after all quants
        ggml_fp16_t * global_scales = reinterpret_cast<ggml_fp16_t *>(coalesced_data + qs_bytes);
        for (int b = 0; b < blocks_per_row; b++) {
            global_scales[row * blocks_per_row + b] = soa_scales[row * blocks_per_row + b];
        }
    }
}

// =============================================================================
// Test: Compare SoA DMMV vs Coalesced DMMV
// =============================================================================

bool test_coalesced_vs_soa_dmmv(sycl::queue & q, int ncols, int nrows) {
    printf("\n=== Test: Coalesced vs SoA DMMV (%d x %d) ===\n", nrows, ncols);

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks   = nrows * blocks_per_row;

    // Check if dimensions are compatible with coalesced layout
    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("SKIP: blocks_per_row=%d not divisible by tile size %d\n", blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;  // Skip, not a failure
    }

    // Generate random test data
    std::mt19937                          rng(42 + ncols + nrows);
    std::uniform_real_distribution<float> scale_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<int>    byte_dist(0, 255);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = ggml_fp32_to_fp16(scale_dist(rng));
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_aos[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
        }
    }

    std::vector<float> h_y(ncols);
    for (int i = 0; i < ncols; i++) {
        h_y[i] = y_dist(rng);
    }

    // CPU reference output
    std::vector<float> cpu_output(nrows);
    dmmv_q4_0_cpu_reference(h_aos.data(), h_y.data(), cpu_output.data(), ncols, nrows);

    // Create SoA layout
    const size_t soa_qs_bytes    = total_blocks * (QK4_0 / 2);
    const size_t soa_d_bytes     = total_blocks * sizeof(ggml_fp16_t);
    const size_t soa_total_bytes = soa_qs_bytes + soa_d_bytes;

    std::vector<uint8_t> h_soa(soa_total_bytes);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), total_blocks);

    // Create coalesced layout
    const int    coalesced_row_stride  = blocks_per_row * (QK4_0 / 2);
    const size_t coalesced_total_bytes = nrows * coalesced_row_stride + total_blocks * sizeof(ggml_fp16_t);

    std::vector<uint8_t> h_coalesced(coalesced_total_bytes);
    convert_soa_to_coalesced(h_soa.data(), h_coalesced.data(), ncols, nrows);

    // TODO: Once the coalesced DMMV kernel is properly exposed via ggml API,
    // we would run both SoA and coalesced versions and compare.
    //
    // For now, this test fails to drive TDD - the kernel exists in dmmv.cpp
    // but needs proper integration testing.

    printf("  CPU reference computed: first 4 values = ");
    for (int i = 0; i < std::min(4, nrows); i++) {
        printf("%.4f ", cpu_output[i]);
    }
    printf("\n");

    printf("  SoA data size: %zu bytes\n", soa_total_bytes);
    printf("  Coalesced data size: %zu bytes\n", coalesced_total_bytes);
    printf("  Coalesced row stride: %d bytes\n", coalesced_row_stride);

    // This test should fail until proper coalesced DMMV is integrated
    fprintf(stderr, "ERROR: Coalesced DMMV kernel integration test not implemented\n");
    return false;
}

// =============================================================================
// Test: Coalesced Layout Data Integrity
// =============================================================================

bool test_coalesced_layout_integrity(int ncols, int nrows) {
    printf("\n=== Test: Coalesced Layout Data Integrity (%d x %d) ===\n", nrows, ncols);

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks   = nrows * blocks_per_row;

    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("SKIP: blocks_per_row=%d not divisible by tile size %d\n", blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;
    }

    // Generate deterministic test data
    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = ggml_fp32_to_fp16(0.01f * (b % 100 + 1));
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_aos[b].qs[i] = static_cast<uint8_t>((b + i) % 256);
        }
    }

    // Create SoA layout
    const size_t soa_qs_bytes    = total_blocks * (QK4_0 / 2);
    const size_t soa_d_bytes     = total_blocks * sizeof(ggml_fp16_t);
    const size_t soa_total_bytes = soa_qs_bytes + soa_d_bytes;

    std::vector<uint8_t> h_soa(soa_total_bytes);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), total_blocks);

    // Create coalesced layout
    const int    coalesced_row_stride  = blocks_per_row * (QK4_0 / 2);
    const size_t coalesced_total_bytes = nrows * coalesced_row_stride + total_blocks * sizeof(ggml_fp16_t);

    std::vector<uint8_t> h_coalesced(coalesced_total_bytes);
    convert_soa_to_coalesced(h_soa.data(), h_coalesced.data(), ncols, nrows);

    // Verify that all data is preserved (we can dequantize and get same values)
    bool passed = true;
    int  errors = 0;

    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int WORDS_PER_BLOCK = BYTES_PER_BLOCK / 4;

    const size_t qs_bytes = total_blocks * (QK4_0 / 2);

    for (int row = 0; row < nrows && errors < 10; row++) {
        const uint8_t *    row_data = h_coalesced.data() + row * coalesced_row_stride;
        const ggml_fp16_t * global_scales =
            reinterpret_cast<const ggml_fp16_t *>(h_coalesced.data() + qs_bytes);

        int tiles_per_row = blocks_per_row / TILE_BLOCKS;

        for (int tile = 0; tile < tiles_per_row && errors < 10; tile++) {
            const uint8_t * tile_data = row_data + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int b = 0; b < TILE_BLOCKS && errors < 10; b++) {
                int global_block = row * blocks_per_row + tile * TILE_BLOCKS + b;

                // Extract quants from coalesced layout
                uint8_t extracted_qs[BYTES_PER_BLOCK];
                for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                    int src_offset = w * (TILE_BLOCKS * 4) + b * 4;
                    memcpy(extracted_qs + w * 4, tile_data + src_offset, 4);
                }

                // Compare with original AoS data
                for (int i = 0; i < BYTES_PER_BLOCK; i++) {
                    if (extracted_qs[i] != h_aos[global_block].qs[i]) {
                        fprintf(stderr,
                            "  FAIL: row=%d tile=%d block=%d byte=%d: "
                            "expected=0x%02x got=0x%02x\n",
                            row, tile, b, i, h_aos[global_block].qs[i], extracted_qs[i]);
                        passed = false;
                        errors++;
                    }
                }

                // Check scale
                const float expected_scale = ggml_fp16_to_fp32(h_aos[global_block].d);
                const float actual_scale = ggml_fp16_to_fp32(global_scales[global_block]);
                if (std::fabs(expected_scale - actual_scale) > 1e-6f) {
                    fprintf(stderr, "  FAIL: row=%d block=%d: scale expected=%.4f got=%.4f\n", row,
                           global_block % blocks_per_row, float(expected_scale), float(actual_scale));
                    passed = false;
                    errors++;
                }
            }
        }
    }

    printf("  Layout integrity: %s (errors=%d)\n", passed ? "PASSED" : "FAILED", errors);
    return passed;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    printf("=== Q4_0 DMMV Coalesced Layout Tests ===\n");
    printf("Testing coalesced memory layout for DMMV kernel\n");
    printf("Tile configuration: %d blocks per tile, %d bytes per tile\n\n", MMVQ_COALESCED_TILE_BLOCKS,
           MMVQ_COALESCED_TILE_BYTES_Q4_0);

    // Test layout integrity first (CPU-only, no GPU needed)
    printf("=== Part 1: Layout Integrity Tests (CPU) ===\n");

    bool layout_ok = true;
    layout_ok &= test_coalesced_layout_integrity(1024, 16);  // 1024/32 = 32 blocks per row
    layout_ok &= test_coalesced_layout_integrity(2048, 32);  // 2048/32 = 64 blocks per row
    layout_ok &= test_coalesced_layout_integrity(4096, 64);  // 4096/32 = 128 blocks per row (64 rows)

    if (!layout_ok) {
        printf("\nLayout integrity tests FAILED\n");
        return 1;
    }
    printf("\nLayout integrity tests PASSED\n");

    // GPU tests (coalesced DMMV via ggml backend)
    printf("\n=== Part 2: GPU DMMV Tests (Coalesced) ===\n");

    setenv("GGML_SYCL_LAYOUT_OVERRIDE", "coalesced", 1);
    setenv("GGML_SYCL_FORCE_DMMV", "1", 1);

    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return 0;
    }
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("SKIP: Could not initialize CPU backend\n");
        ggml_backend_free(gpu_backend);
        return 0;
    }

    bool gpu_ok = true;
    gpu_ok &= run_dmmv_coalesced_case(gpu_backend, cpu_backend, 1024, 64, true);
    gpu_ok &= run_dmmv_coalesced_case(gpu_backend, cpu_backend, 2048, 128, true);
    gpu_ok &= run_dmmv_coalesced_case(gpu_backend, cpu_backend, 4096, 256, true);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("\n=== Summary ===\n");
    printf("Layout tests: %s\n", layout_ok ? "PASSED" : "FAILED");
    printf("GPU tests: %s\n", gpu_ok ? "PASSED" : "FAILED");

    return (layout_ok && gpu_ok) ? 0 : 1;
}
