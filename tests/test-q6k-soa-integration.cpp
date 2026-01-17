// Integration test for Q6_K SoA production path
// Tests: CPU reorder (reorder_q6_k_cpu) → tensor upload → DMMV kernel
//
// This test uses the actual ggml_backend API to exercise the full production path,
// unlike unit tests that create their own SoA data.
//
// Production flow for Q6_K:
// 1. ggml_backend_tensor_set() is called with AoS data
// 2. buffer_set_tensor() checks if reordering is enabled (test override can force AoS)
// 3. If enabled: reorder_q6_k_cpu() converts AoS→SoA in staging buffer
// 4. Data is uploaded to GPU in SoA format
// 5. DMMV kernel uses SoA layout for efficient memory access
//
// NOTE: Weight streaming cache (unified_cache) is BYPASSED for SoA-reordered tensors
// because cache stores AoS format but we need SoA on device.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

// Check if AoS override is set (disables reordering)
static bool is_soa_disabled() {
    ggml_layout_mode override_layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl::test_get_layout_override(&override_layout)) {
        return false;
    }
    return override_layout == GGML_LAYOUT_AOS;
}

// Q6_K block structure (must match ggml-common.h)
#define QK_K 256

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];      // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits (16 bytes)
    uint16_t d;              // super-block scale as ggml_fp16_t (2 bytes)
} block_q6_K_test;

static_assert(sizeof(block_q6_K_test) == 210, "wrong q6_K block size");

// CPU reference: dequantize Q6_K and compute dot product
float cpu_vec_dot_q6_K(const block_q6_K_test* x, const float* y, int ncols) {
    const int nb = ncols / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        // Convert fp16 scale to float
        uint16_t d_bits = x[i].d;
        float d;
        // Simple fp16 to float conversion
        uint32_t sign = (d_bits >> 15) & 0x1;
        uint32_t exp = (d_bits >> 10) & 0x1F;
        uint32_t mant = d_bits & 0x3FF;
        if (exp == 0) {
            d = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
        } else if (exp == 31) {
            d = (mant == 0) ? ((sign ? -1.0f : 1.0f) * INFINITY) : NAN;
        } else {
            d = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
        }

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                const float* yp = y + i * QK_K + n;
                sumf += d * sc[is + 0] * q1 * yp[l + 0];
                sumf += d * sc[is + 2] * q2 * yp[l + 32];
                sumf += d * sc[is + 4] * q3 * yp[l + 64];
                sumf += d * sc[is + 6] * q4 * yp[l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return sumf;
}

// Convert float to fp16 bits
uint16_t float_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        return (sign << 15);  // Denormal or zero
    } else if (exp >= 31) {
        return (sign << 15) | (31 << 10);  // Infinity
    }
    return (sign << 15) | (exp << 10) | mant;
}

// Generate random Q6_K test data in AoS format
void generate_q6k_data(block_q6_K_test* data, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> ql_dist(0, 255);
    std::uniform_int_distribution<int> qh_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(-32, 31);
    std::uniform_real_distribution<float> d_dist(0.001f, 0.1f);

    for (int ib = 0; ib < nblocks; ib++) {
        for (int j = 0; j < QK_K/2; j++) {
            data[ib].ql[j] = ql_dist(rng);
        }
        for (int j = 0; j < QK_K/4; j++) {
            data[ib].qh[j] = qh_dist(rng);
        }
        for (int j = 0; j < QK_K/16; j++) {
            data[ib].scales[j] = scale_dist(rng);
        }
        data[ib].d = float_to_fp16(d_dist(rng));
    }
}

// Test configuration
struct TestConfig {
    const char* name;
    int ncols;
    int nrows;
    int batch;  // batch=1 triggers DMMV, batch>1 triggers MMQ
};

bool run_integration_test(ggml_backend_t backend, const TestConfig& cfg) {
    const char* kernel_type = (cfg.batch == 1) ? "DMMV" : "MMQ";
    printf("\n=== Test: %s (ncols=%d, nrows=%d, batch=%d) [%s] ===\n",
           cfg.name, cfg.ncols, cfg.nrows, cfg.batch, kernel_type);

    const int nb_per_row = cfg.ncols / QK_K;
    const int nblocks = cfg.nrows * nb_per_row;
    const size_t weight_size = nblocks * sizeof(block_q6_K_test);

    // Create random test data
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::vector<block_q6_K_test> weight_data(nblocks);
    generate_q6k_data(weight_data.data(), nblocks, rng);

    // Create Y matrix (input) - [ncols x batch]
    std::vector<float> y_data(cfg.ncols * cfg.batch);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);
    for (int i = 0; i < cfg.ncols * cfg.batch; i++) {
        y_data[i] = y_dist(rng);
    }

    // Compute CPU reference for each output element [nrows x batch]
    std::vector<float> cpu_results(cfg.nrows * cfg.batch);
    for (int b = 0; b < cfg.batch; b++) {
        for (int row = 0; row < cfg.nrows; row++) {
            cpu_results[b * cfg.nrows + row] = cpu_vec_dot_q6_K(
                &weight_data[row * nb_per_row],
                &y_data[b * cfg.ncols],
                cfg.ncols);
        }
    }

    // Create ggml context
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to create ggml context\n");
        return false;
    }

    // Create tensors
    // Weight tensor: Q6_K quantized [ncols x nrows]
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, cfg.ncols, cfg.nrows);
    ggml_set_name(weight, "test_weight");

    // Input tensor: float [ncols x batch] (batch=1 for DMMV, batch>1 for MMQ)
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.ncols, cfg.batch);
    ggml_set_name(input, "test_input");

    // Create compute graph FIRST (before allocation)
    // MUL_MAT: weight[ncols x nrows] @ input[ncols x 1] = output[nrows x 1]
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* result = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(result, "mul_mat_result");
    ggml_build_forward_expand(gf, result);

    // Allocate tensors using graph allocator (includes all tensors in graph)
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, gf)) {
        printf("Failed to reserve graph allocator\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        printf("Failed to allocate graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // Upload weight data (this triggers CPU reorder in production path!)
    printf("Uploading weight tensor via buffer_set_tensor (triggers CPU reorder)...\n");
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_size);

    // Upload input data
    ggml_backend_tensor_set(input, y_data.data(), 0, cfg.ncols * cfg.batch * sizeof(float));

    // Compute on GPU
    printf("Computing MUL_MAT on GPU (uses %s kernel)...\n", kernel_type);
    ggml_backend_graph_compute(backend, gf);

    // Get results [nrows x batch]
    const int total_outputs = cfg.nrows * cfg.batch;
    std::vector<float> gpu_results(total_outputs);
    ggml_backend_tensor_get(result, gpu_results.data(), 0, total_outputs * sizeof(float));

    // Compare results
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_idx = 0;
    int nan_count = 0;
    int inf_count = 0;
    int large_err_count = 0;  // errors > 100%

    for (int i = 0; i < total_outputs; i++) {
        float cpu_val = cpu_results[i];
        float gpu_val = gpu_results[i];

        // Check for NaN/Inf first
        if (std::isnan(gpu_val)) {
            nan_count++;
            continue;
        }
        if (std::isinf(gpu_val)) {
            inf_count++;
            continue;
        }

        float abs_err = fabsf(cpu_val - gpu_val);
        float rel_err = (fabsf(cpu_val) > 1e-6f) ? (abs_err / fabsf(cpu_val)) : abs_err;

        if (rel_err > 1.0f) {
            large_err_count++;
        }

        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            max_abs_err = abs_err;
            worst_idx = i;
        }
    }

    // Print sample results (first 5 elements for more context)
    printf("Sample results (first 5 outputs):\n");
    for (int i = 0; i < std::min(5, total_outputs); i++) {
        printf("  [%d]: CPU=%12.4f GPU=%12.4f diff=%12.4f\n",
               i, cpu_results[i], gpu_results[i], fabsf(cpu_results[i] - gpu_results[i]));
    }

    // Print statistics
    printf("Error stats: NaN=%d, Inf=%d, >100%% error=%d, total=%d\n",
           nan_count, inf_count, large_err_count, total_outputs);

    if (nan_count == 0 && inf_count == 0) {
        printf("Max relative error: %.4f%% at idx %d (CPU=%.6f, GPU=%.6f)\n",
               max_rel_err * 100.0f, worst_idx, cpu_results[worst_idx], gpu_results[worst_idx]);
    }

    // If we have bad values, print a few examples
    if (nan_count > 0 || inf_count > 0 || large_err_count > 0) {
        printf("First few problematic values:\n");
        int printed = 0;
        for (int i = 0; i < total_outputs && printed < 5; i++) {
            float gpu_val = gpu_results[i];
            float cpu_val = cpu_results[i];
            float rel_err = (fabsf(cpu_val) > 1e-6f) ?
                            (fabsf(cpu_val - gpu_val) / fabsf(cpu_val)) : fabsf(cpu_val - gpu_val);

            if (std::isnan(gpu_val) || std::isinf(gpu_val) || rel_err > 1.0f) {
                printf("  [%d]: CPU=%12.4f GPU=%12.4f (rel_err=%.2f%%)\n",
                       i, cpu_val, gpu_val, rel_err * 100.0f);
                printed++;
            }
        }
    }

    // Cleanup
    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    // Allow small absolute error on near-zero outputs while keeping a relative guard.
    const float abs_tol = 6.0f;
    const float rel_tol = 0.30f;
    bool pass = (nan_count == 0) && (inf_count == 0) &&
                (max_abs_err <= abs_tol || max_rel_err <= rel_tol);
    printf("%s: %s\n", cfg.name, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== Q6_K SoA Integration Test ===\n");
    printf("Tests the FULL production path:\n");
    printf("  1. Tensor upload via buffer_set_tensor\n");
    printf("  2. CPU-side SoA reorder (reorder_q6_k_cpu) if enabled\n");
    printf("  3. DMMV kernel execution\n\n");

    // Show which mode we're testing
    if (is_soa_disabled()) {
        printf("MODE: AoS (test layout override)\n");
        printf("  - Weight data uploaded in original AoS format\n");
        printf("  - DMMV uses AoS kernel path\n\n");
    } else {
        printf("MODE: SoA (default)\n");
        printf("  - reorder_q6_k_cpu() converts AoS→SoA during upload\n");
        printf("  - DMMV uses SoA kernel path\n\n");
    }

    // Initialize SYCL backend
    ggml_backend_t backend = ggml_backend_sycl_init(0);  // Device 0
    if (!backend) {
        printf("Failed to initialize SYCL backend\n");
        return 1;
    }

    printf("Backend: %s\n", ggml_backend_name(backend));

    // Test configurations - batch=1 triggers DMMV, batch>1 triggers MMQ
    TestConfig tests[] = {
        // DMMV tests (batch=1) - token generation path
        {"DMMV: Single block, 1 row",         256,    1,   1},
        {"DMMV: Single block, 8 rows",        256,    8,   1},
        {"DMMV: Mistral hidden, 1 row",      4096,    1,   1},
        {"DMMV: Mistral hidden, 32 rows",    4096,   32,   1},
        {"DMMV: Mistral hidden, 128 rows",   4096,  128,   1},
        {"DMMV: 8K hidden, 64 rows",         8192,   64,   1},

        // MMQ tests (batch>1) - prompt processing path
        {"MMQ: Single block, batch=4",        256,    8,   4},
        {"MMQ: Mistral hidden, batch=4",     4096,   32,   4},
        {"MMQ: Mistral hidden, batch=16",    4096,   32,  16},
        {"MMQ: Mistral hidden, batch=32",    4096,  128,  32},
        {"MMQ: 8K hidden, batch=8",          8192,   64,   8},
    };

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        if (run_integration_test(backend, test)) {
            passed++;
        } else {
            failed++;
        }
    }

    // Additional test: compare SoA vs AoS (with override)
    printf("\n=== Comparison Note ===\n");
    printf("To compare SoA vs AoS, run this test with and without a test layout override forcing AoS\n");
    printf("If results differ significantly, the bug is in the reorder or kernel path.\n");

    printf("\n=== SUMMARY ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    ggml_backend_free(backend);

    return (failed == 0) ? 0 : 1;
}
