/*
 * Q8_0 SoA Production Test - AoS vs SoA Comparison
 *
 * Tests Q8_0 quantized matrix multiplication by comparing:
 * 1. GPU AoS (GGML_SYCL_LAYOUT_OVERRIDE=aos) vs CPU reference
 * 2. GPU SoA (GGML_SYCL_LAYOUT_OVERRIDE=soa) vs CPU reference
 * 3. GPU SoA vs GPU AoS
 *
 * This helps isolate whether the bug is in SoA reordering or kernel access.
 *
 * Build: cmake --build build --target test-q8-0-production-kernels
 * Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-q8-0-production-kernels
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// Use actual ggml headers
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"

// Q8_0 constants
#define QK8_0 32

// Q8_0 block structure
typedef struct {
    uint16_t d;           // delta (fp16 as uint16_t)
    int8_t qs[QK8_0];     // quants
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

// Helper to convert float to fp16 (as uint16_t)
static inline uint16_t fp32_to_fp16(float f) {
    union {
        float f;
        uint32_t u;
    } fu = {f};

    uint32_t sign = (fu.u >> 16) & 0x8000;
    int32_t exponent = ((fu.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (fu.u >> 13) & 0x3FF;

    if (exponent <= 0) {
        return (uint16_t)sign;
    } else if (exponent >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }

    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

// Convert fp16 (uint16_t) to float
static inline float fp16_to_fp32(uint16_t h) {
    union {
        float f;
        uint32_t u;
    } fu;

    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            fu.u = sign;
        } else {
            // Denormalized
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            fu.u = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        fu.u = sign | 0x7F800000 | (mantissa << 13);
    } else {
        fu.u = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return fu.f;
}

// Quantize float array to Q8_0
static void quantize_to_q8_0(const float* src, block_q8_0_test* dst, int n) {
    int nblocks = n / QK8_0;

    for (int ib = 0; ib < nblocks; ib++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(src[ib * QK8_0 + j]);
            if (v > amax) amax = v;
        }

        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

        dst[ib].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; j++) {
            float v = src[ib * QK8_0 + j] * id;
            dst[ib].qs[j] = (int8_t)roundf(v);
        }
    }
}

// CPU reference using ggml backend (same as inference)
static bool run_cpu_reference(ggml_backend_t cpu_backend,
                              const block_q8_0_test* weight_data, int nblocks,
                              const float* input_data,
                              float* output_data,
                              int ncols, int nrows, int batch) {
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate CPU buffer
    size_t total_size = ggml_nbytes(weight) + ggml_nbytes(input) + ggml_nbytes(output) + 4096;
    ggml_backend_buffer_t cpu_buffer = ggml_backend_buft_alloc_buffer(cpu_buft, total_size);
    if (!cpu_buffer) {
        ggml_free(ctx);
        return false;
    }

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(cpu_buffer);
    ggml_backend_tensor_alloc(cpu_buffer, weight, base);
    ggml_backend_tensor_alloc(cpu_buffer, input, base + ggml_nbytes(weight) + 256);
    ggml_backend_tensor_alloc(cpu_buffer, output, base + ggml_nbytes(weight) + ggml_nbytes(input) + 512);

    // Set data
    ggml_backend_tensor_set(weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(input, input_data, 0, ncols * batch * sizeof(float));

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    enum ggml_status status = ggml_backend_graph_compute(cpu_backend, graph);

    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, output_data, 0, nrows * batch * sizeof(float));
    }

    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(ctx);

    return status == GGML_STATUS_SUCCESS;
}

// Test configuration
struct TestConfig {
    int ncols;
    int nrows;
    int batch;
    const char* name;
};

// Run GPU computation with specified mode
static bool run_gpu_compute(ggml_backend_t gpu_backend,
                            const block_q8_0_test* weight_data, int nblocks,
                            const float* input_data,
                            float* output_data,
                            int ncols, int nrows, int batch,
                            bool force_aos) {
    // Set environment for this run
    if (force_aos) {
        setenv("GGML_SYCL_LAYOUT_OVERRIDE", "aos", 1);
    } else {
        setenv("GGML_SYCL_LAYOUT_OVERRIDE", "soa", 1);
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t weight_size = ggml_nbytes(weight) * 2 + 4096;
    ggml_backend_buffer_t gpu_weight_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, weight_size);
    if (!gpu_weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(gpu_weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_alloc(gpu_weight_buffer, weight,
                               (void*)ggml_backend_buffer_get_base(gpu_weight_buffer));

    size_t compute_size = ggml_nbytes(input) + ggml_nbytes(output) + 4096;
    ggml_backend_buffer_t gpu_compute_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, compute_size);
    if (!gpu_compute_buffer) {
        ggml_backend_buffer_free(gpu_weight_buffer);
        ggml_free(ctx);
        return false;
    }

    uint8_t* compute_base = (uint8_t*)ggml_backend_buffer_get_base(gpu_compute_buffer);
    ggml_backend_tensor_alloc(gpu_compute_buffer, input, compute_base);
    size_t input_alloc = ggml_backend_buft_get_alloc_size(gpu_buft, input);
    ggml_backend_tensor_alloc(gpu_compute_buffer, output, compute_base + input_alloc + 512);

    // Set data
    ggml_backend_tensor_set(weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(input, input_data, 0, ncols * batch * sizeof(float));

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    enum ggml_status status = ggml_backend_graph_compute(gpu_backend, graph);

    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, output_data, 0, nrows * batch * sizeof(float));
    }

    ggml_backend_buffer_free(gpu_compute_buffer);
    ggml_backend_buffer_free(gpu_weight_buffer);
    ggml_free(ctx);

    return status == GGML_STATUS_SUCCESS;
}

// Debug: Read back GPU weight tensor and compare to original
static void debug_gpu_weight_layout(ggml_backend_t gpu_backend,
                                    const block_q8_0_test* original_data,
                                    int ncols, int nrows, bool force_aos) {
    if (force_aos) {
        setenv("GGML_SYCL_LAYOUT_OVERRIDE", "aos", 1);
    } else {
        setenv("GGML_SYCL_LAYOUT_OVERRIDE", "soa", 1);
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return;

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);

    size_t weight_size = ggml_nbytes(weight) * 2 + 4096;
    ggml_backend_buffer_t gpu_weight_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, weight_size);
    if (!gpu_weight_buffer) {
        ggml_free(ctx);
        return;
    }
    ggml_backend_buffer_set_usage(gpu_weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_alloc(gpu_weight_buffer, weight,
                               (void*)ggml_backend_buffer_get_base(gpu_weight_buffer));

    int nblocks = nrows * (ncols / QK8_0);
    ggml_backend_tensor_set(weight, original_data, 0, nblocks * sizeof(block_q8_0_test));

    // Read back - we can only read ggml_nbytes(tensor), not the full alloc_size
    size_t alloc_size = ggml_backend_buft_get_alloc_size(gpu_buft, weight);
    size_t tensor_size = ggml_nbytes(weight);
    std::vector<uint8_t> gpu_data(tensor_size);
    ggml_backend_tensor_get(weight, gpu_data.data(), 0, tensor_size);

    printf("    --- GPU Weight Layout Debug (%s) ---\n", force_aos ? "AoS" : "SoA");
    printf("    Original size: %zu bytes, tensor_size: %zu, alloc_size: %zu bytes\n",
           nblocks * sizeof(block_q8_0_test), tensor_size, alloc_size);

    // Show first few bytes
    printf("    First 64 bytes from GPU:\n    ");
    for (int i = 0; i < 64 && i < (int)gpu_data.size(); i++) {
        printf("%02x ", gpu_data[i]);
        if ((i + 1) % 16 == 0) printf("\n    ");
    }
    printf("\n");

    // Show first block from original
    printf("    Original block 0: d=%04x, qs[0..7]=",
           original_data[0].d);
    for (int i = 0; i < 8; i++) {
        printf("%02x ", (uint8_t)original_data[0].qs[i]);
    }
    printf("\n");

    // For SoA, show where d values should be
    // Note: The tensor data read back may be in original AoS format since
    // the backend may re-serialize it. We should check the actual GPU layout
    // by observing kernel behavior instead.
    if (!force_aos) {
        size_t d_offset = (size_t)nblocks * QK8_0;  // All qs bytes first
        printf("    Expected d_offset (SoA): %zu bytes (but read back may be AoS)\n", d_offset);
        if (d_offset + 8 <= gpu_data.size()) {
            printf("    Bytes at d_offset: ");
            for (int i = 0; i < 8; i++) {
                printf("%02x ", gpu_data[d_offset + i]);
            }
            printf("\n");

            // Interpret as fp16 d values
            uint16_t* d_ptr = (uint16_t*)(gpu_data.data() + d_offset);
            printf("    First 4 d values (fp16->fp32): ");
            for (int i = 0; i < 4; i++) {
                printf("%.4f ", fp16_to_fp32(d_ptr[i]));
            }
            printf("\n");

            // Compare to original d values
            printf("    Original first 4 d values:     ");
            for (int i = 0; i < 4; i++) {
                printf("%.4f ", fp16_to_fp32(original_data[i].d));
            }
            printf("\n");
        }
    }

    ggml_backend_buffer_free(gpu_weight_buffer);
    ggml_free(ctx);
}

// Run a single test with AoS vs SoA comparison
static bool run_test(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, const TestConfig& cfg) {
    printf("  Testing: %s (ncols=%d, nrows=%d, batch=%d)\n",
           cfg.name, cfg.ncols, cfg.nrows, cfg.batch);

    bool passed = true;

    // Generate random weight data
    int nblocks = cfg.nrows * (cfg.ncols / QK8_0);
    std::vector<float> weight_float(cfg.nrows * cfg.ncols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < weight_float.size(); i++) {
        weight_float[i] = dist(rng);
    }

    // Quantize weights
    std::vector<block_q8_0_test> weight_q8(nblocks);
    quantize_to_q8_0(weight_float.data(), weight_q8.data(), (int)weight_float.size());

    // Generate random input
    std::vector<float> input_data(cfg.ncols * cfg.batch);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = dist(rng);
    }

    // Compute CPU reference using ggml backend
    std::vector<float> cpu_results(cfg.nrows * cfg.batch);
    if (!run_cpu_reference(cpu_backend, weight_q8.data(), nblocks,
                           input_data.data(), cpu_results.data(),
                           cfg.ncols, cfg.nrows, cfg.batch)) {
        printf("    FAIL: CPU reference compute failed\n");
        return false;
    }

    // Debug weight layouts (for small tests only)
    if (cfg.ncols <= 64) {
        debug_gpu_weight_layout(gpu_backend, weight_q8.data(), cfg.ncols, cfg.nrows, true);  // AoS
        debug_gpu_weight_layout(gpu_backend, weight_q8.data(), cfg.ncols, cfg.nrows, false); // SoA
    }

    // Compute GPU AoS
    std::vector<float> gpu_aos_results(cfg.nrows * cfg.batch);
    if (!run_gpu_compute(gpu_backend, weight_q8.data(), nblocks,
                         input_data.data(), gpu_aos_results.data(),
                         cfg.ncols, cfg.nrows, cfg.batch, true)) {
        printf("    FAIL: GPU AoS compute failed\n");
        return false;
    }

    // Compute GPU SoA
    std::vector<float> gpu_soa_results(cfg.nrows * cfg.batch);
    if (!run_gpu_compute(gpu_backend, weight_q8.data(), nblocks,
                         input_data.data(), gpu_soa_results.data(),
                         cfg.ncols, cfg.nrows, cfg.batch, false)) {
        printf("    FAIL: GPU SoA compute failed\n");
        return false;
    }

    // Compare results
    const float tolerance = 0.01f;

    // 1. GPU AoS vs CPU
    int aos_errors = 0;
    float aos_max_err = 0;
    for (size_t i = 0; i < cpu_results.size(); i++) {
        float err = fabsf(gpu_aos_results[i] - cpu_results[i]);
        if (err > aos_max_err) aos_max_err = err;
        float rel = (fabsf(cpu_results[i]) > 1e-6f) ? err / fabsf(cpu_results[i]) : err;
        if (rel > tolerance && err > 0.001f) aos_errors++;
    }

    // 2. GPU SoA vs CPU
    int soa_errors = 0;
    float soa_max_err = 0;
    for (size_t i = 0; i < cpu_results.size(); i++) {
        float err = fabsf(gpu_soa_results[i] - cpu_results[i]);
        if (err > soa_max_err) soa_max_err = err;
        float rel = (fabsf(cpu_results[i]) > 1e-6f) ? err / fabsf(cpu_results[i]) : err;
        if (rel > tolerance && err > 0.001f) soa_errors++;
    }

    // 3. GPU SoA vs GPU AoS
    int soa_vs_aos_errors = 0;
    float soa_vs_aos_max = 0;
    int first_diff_idx = -1;
    for (size_t i = 0; i < cpu_results.size(); i++) {
        float err = fabsf(gpu_soa_results[i] - gpu_aos_results[i]);
        if (err > soa_vs_aos_max) soa_vs_aos_max = err;
        if (err > 0.001f) {
            if (first_diff_idx < 0) first_diff_idx = (int)i;
            soa_vs_aos_errors++;
        }
    }

    printf("    GPU AoS vs CPU:  errors=%d/%zu, max_err=%.6f %s\n",
           aos_errors, cpu_results.size(), aos_max_err,
           aos_errors == 0 ? "PASS" : "FAIL");
    printf("    GPU SoA vs CPU:  errors=%d/%zu, max_err=%.6f %s\n",
           soa_errors, cpu_results.size(), soa_max_err,
           soa_errors == 0 ? "PASS" : "FAIL");
    printf("    GPU SoA vs AoS:  errors=%d/%zu, max_err=%.6f %s\n",
           soa_vs_aos_errors, cpu_results.size(), soa_vs_aos_max,
           soa_vs_aos_errors == 0 ? "MATCH" : "DIFFER");

    // Detailed debug for first 16 values
    printf("    --- First 16 values comparison ---\n");
    printf("    Idx |       CPU |   GPU_AoS |   GPU_SoA | AoS_diff | SoA_diff\n");
    printf("    ----|-----------|-----------|-----------|----------|----------\n");
    for (int i = 0; i < 16 && i < (int)cpu_results.size(); i++) {
        float aos_diff = gpu_aos_results[i] - cpu_results[i];
        float soa_diff = gpu_soa_results[i] - cpu_results[i];
        printf("    %3d | %9.4f | %9.4f | %9.4f | %8.5f | %8.5f %s\n",
               i, cpu_results[i], gpu_aos_results[i], gpu_soa_results[i],
               aos_diff, soa_diff,
               (fabsf(soa_diff) > 0.01f && fabsf(aos_diff) < 0.001f) ? "<-- SoA BUG" : "");
    }

    // If there are differences, show where first divergence happens
    if (first_diff_idx >= 0) {
        printf("    First SoA vs AoS difference at index %d:\n", first_diff_idx);
        printf("      CPU=%.6f, AoS=%.6f, SoA=%.6f\n",
               cpu_results[first_diff_idx],
               gpu_aos_results[first_diff_idx],
               gpu_soa_results[first_diff_idx]);

        // Show block-level info for this index
        int row = first_diff_idx % cfg.nrows;
        int block_in_row = 0;  // For batch=1, output is one value per row
        printf("      Row=%d, blocks_per_row=%d\n", row, cfg.ncols / QK8_0);

        // Show values around the error
        printf("    Values around first error (indices %d-%d):\n",
               std::max(0, first_diff_idx - 2),
               std::min((int)cpu_results.size() - 1, first_diff_idx + 2));
        for (int i = std::max(0, first_diff_idx - 2);
             i <= std::min((int)cpu_results.size() - 1, first_diff_idx + 2); i++) {
            printf("      [%d] CPU=%.6f AoS=%.6f SoA=%.6f %s\n",
                   i, cpu_results[i], gpu_aos_results[i], gpu_soa_results[i],
                   i == first_diff_idx ? "<--" : "");
        }
    }

    if (soa_errors > 0) {
        passed = false;
        printf("    RESULT: FAIL (SoA kernel bug detected)\n");
    } else {
        printf("    RESULT: PASS\n");
    }

    return passed;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    printf("=== Q8_0 AoS vs SoA Production Kernel Test ===\n\n");

    // Initialize GPU backend
    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("FAIL: Could not initialize SYCL backend\n");
        printf("Make sure ONEAPI_DEVICE_SELECTOR is set correctly\n");
        return 1;
    }
    printf("GPU Backend: %s\n", ggml_backend_name(gpu_backend));

    // Initialize CPU backend for reference
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(gpu_backend);
        return 1;
    }
    printf("CPU Backend: %s\n\n", ggml_backend_name(cpu_backend));

    // Test configurations - start tiny for debugging
    static const TestConfig test_configs[] = {
        // Tiny: 1 block per row, easy to trace
        { 32, 4, 1, "Tiny 32x4 DMMV (1 block/row)" },

        // Small: multiple blocks per row
        { 64, 4, 1, "Small 64x4 DMMV (2 blocks/row)" },

        // Realistic DMMV
        { 256, 64, 1, "256x64 DMMV (8 blocks/row)" },

        // MMQ test
        { 256, 64, 4, "256x64 MMQ batch=4" },

        // Full scale DMMV
        { 4096, 256, 1, "4096x256 DMMV" },
    };

    int passed = 0;
    int failed = 0;
    int ntests = sizeof(test_configs) / sizeof(test_configs[0]);

    for (int i = 0; i < ntests; i++) {
        if (run_test(gpu_backend, cpu_backend, test_configs[i])) {
            passed++;
        } else {
            failed++;
        }
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("Passed: %d/%d\n", passed, ntests);
    printf("Failed: %d/%d\n", failed, ntests);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    return (failed > 0) ? 1 : 0;
}
