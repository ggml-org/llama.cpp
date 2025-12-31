/*
 * Comprehensive DMMV Q8_0 Unit Test - Production Kernels
 *
 * Tests the Q8_0 DMMV (batch=1) and MMQ (batch>1) kernels using actual
 * ggml production backends for both CPU reference and GPU execution.
 *
 * Key test dimensions:
 * - Various ncols: 32, 64, 128, 256, 512, 1024, 4096
 * - Various nrows: 1, 4, 16, 64, 256
 * - Both DMMV (batch=1) and MMQ (batch=4) paths
 *
 * Build: cmake --build build --target test-dmmv-q8-0
 * Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-dmmv-q8-0
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"

#define QK8_0 32

typedef struct {
    uint16_t d;           // delta (fp16 as uint16_t)
    int8_t qs[QK8_0];     // quants
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } fu = {f};
    uint32_t sign = (fu.u >> 16) & 0x8000;
    int32_t exponent = ((fu.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (fu.u >> 13) & 0x3FF;
    if (exponent <= 0) return (uint16_t)sign;
    if (exponent >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exponent << 10) | mantissa);
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

// Run computation on a specific backend (CPU or GPU)
static bool run_backend_compute(ggml_backend_t backend,
                                 const block_q8_0_test* weight_data, int nblocks,
                                 const float* input_data,
                                 float* output_data,
                                 int ncols, int nrows, int batch) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate weight buffer (marked as weights for potential reordering)
    size_t weight_size = ggml_nbytes(weight) * 2 + 4096;
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight,
                               (void*)ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t compute_size = ggml_nbytes(input) + ggml_nbytes(output) + 4096;
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, compute_size);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }

    uint8_t* compute_base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, compute_base);
    size_t input_alloc = ggml_backend_buft_get_alloc_size(buft, input);
    ggml_backend_tensor_alloc(compute_buffer, output, compute_base + input_alloc + 512);

    // Set data
    ggml_backend_tensor_set(weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(input, input_data, 0, ncols * batch * sizeof(float));

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, output_data, 0, nrows * batch * sizeof(float));
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return status == GGML_STATUS_SUCCESS;
}

struct TestResult {
    int ncols;
    int nrows;
    int batch;
    int errors;
    float max_err;
    float max_rel_err;
    int worst_idx;
    bool passed;
};

// Manual CPU reference computation for debugging
static void cpu_reference_q8_0(const block_q8_0_test* weight, const float* input,
                                float* output, int ncols, int nrows) {
    int blocks_per_row = ncols / QK8_0;
    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int ib = 0; ib < blocks_per_row; ib++) {
            const block_q8_0_test* blk = &weight[row * blocks_per_row + ib];
            // Convert fp16 d to float
            uint16_t d_fp16 = blk->d;
            uint32_t sign = (d_fp16 & 0x8000) << 16;
            int32_t exp = (d_fp16 >> 10) & 0x1F;
            uint32_t mant = d_fp16 & 0x3FF;
            float d;
            if (exp == 0) {
                if (mant == 0) d = (sign ? -0.0f : 0.0f);
                else {
                    exp = 1;
                    while (!(mant & 0x400)) { mant <<= 1; exp--; }
                    mant &= 0x3FF;
                    union { float f; uint32_t u; } fu;
                    fu.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                    d = fu.f;
                }
            } else if (exp == 31) {
                union { float f; uint32_t u; } fu;
                fu.u = sign | 0x7F800000 | (mant << 13);
                d = fu.f;
            } else {
                union { float f; uint32_t u; } fu;
                fu.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                d = fu.f;
            }

            for (int j = 0; j < QK8_0; j++) {
                int col = ib * QK8_0 + j;
                float w = (float)blk->qs[j] * d;
                sum += w * input[col];
            }
        }
        output[row] = sum;
    }
}

static TestResult run_test(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend,
                            int ncols, int nrows, int batch, bool verbose, bool debug) {
    TestResult result = {ncols, nrows, batch, 0, 0, 0, -1, false};

    int nblocks = nrows * (ncols / QK8_0);
    int output_size = nrows * batch;

    // Generate random weight data
    std::vector<float> weight_float(nrows * ncols);
    std::mt19937 rng(42 + ncols * 1000 + nrows * 10 + batch);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < weight_float.size(); i++) {
        weight_float[i] = dist(rng);
    }

    // Quantize weights
    std::vector<block_q8_0_test> weight_q8(nblocks);
    quantize_to_q8_0(weight_float.data(), weight_q8.data(), (int)weight_float.size());

    // Generate random input
    std::vector<float> input_data(ncols * batch);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = dist(rng);
    }

    // Manual CPU reference for debugging
    std::vector<float> manual_output(nrows);
    if (batch == 1) {
        cpu_reference_q8_0(weight_q8.data(), input_data.data(), manual_output.data(), ncols, nrows);
    }

    // CPU reference using production kernel
    std::vector<float> cpu_output(output_size);
    if (!run_backend_compute(cpu_backend, weight_q8.data(), nblocks,
                             input_data.data(), cpu_output.data(), ncols, nrows, batch)) {
        printf("    CPU compute FAILED\n");
        return result;
    }

    // GPU compute using production kernel
    std::vector<float> gpu_output(output_size);
    if (!run_backend_compute(gpu_backend, weight_q8.data(), nblocks,
                             input_data.data(), gpu_output.data(), ncols, nrows, batch)) {
        printf("    GPU compute FAILED\n");
        return result;
    }

    // Debug: compare manual vs CPU backend
    if (debug && batch == 1) {
        printf("\n=== DEBUG: ncols=%d nrows=%d ===\n", ncols, nrows);
        printf("Manual vs CPU Backend vs GPU:\n");
        printf("Row |    Manual |       CPU |       GPU | CPU-Manual | GPU-Manual | GPU-CPU\n");
        printf("----|-----------|-----------|-----------|------------|------------|--------\n");
        // First show rows with largest GPU-CPU difference
        std::vector<std::pair<float, int>> errors;
        for (int i = 0; i < nrows; i++) {
            float gpu_cpu_diff = fabsf(gpu_output[i] - cpu_output[i]);
            errors.push_back({gpu_cpu_diff, i});
        }
        std::sort(errors.begin(), errors.end(), [](auto& a, auto& b) { return a.first > b.first; });

        printf("Top 10 worst errors (sorted by GPU-CPU diff):\n");
        printf("Row# |    Manual |       CPU |       GPU | CPU-Manual | GPU-Manual | GPU-CPU\n");
        printf("-----|-----------|-----------|-----------|------------|------------|--------\n");
        for (int k = 0; k < std::min(10, nrows); k++) {
            int i = errors[k].second;  // actual row index
            float cpu_manual_diff = cpu_output[i] - manual_output[i];
            float gpu_manual_diff = gpu_output[i] - manual_output[i];
            float gpu_cpu_diff = gpu_output[i] - cpu_output[i];
            float rel_err = fabsf(cpu_output[i]) > 1e-6f ? fabsf(gpu_cpu_diff) / fabsf(cpu_output[i]) : 0;
            printf("%4d | %9.4f | %9.4f | %9.4f | %10.6f | %10.6f | %8.4f (rel=%.2f%%)\n",
                   i, manual_output[i], cpu_output[i], gpu_output[i],
                   cpu_manual_diff, gpu_manual_diff, gpu_cpu_diff, rel_err * 100);
        }

        printf("\nFirst 16 rows:\n");
        for (int i = 0; i < std::min(nrows, 16); i++) {
            float cpu_manual_diff = cpu_output[i] - manual_output[i];
            float gpu_manual_diff = gpu_output[i] - manual_output[i];
            float gpu_cpu_diff = gpu_output[i] - cpu_output[i];
            printf("%3d | %9.4f | %9.4f | %9.4f | %10.6f | %10.6f | %10.6f\n",
                   i, manual_output[i], cpu_output[i], gpu_output[i],
                   cpu_manual_diff, gpu_manual_diff, gpu_cpu_diff);
        }

        // Show weight data for first row
        printf("\nFirst row weight blocks:\n");
        int blocks_per_row = ncols / QK8_0;
        for (int ib = 0; ib < std::min(blocks_per_row, 4); ib++) {
            const block_q8_0_test* blk = &weight_q8[ib];
            printf("  Block %d: d=0x%04x, qs[0..7]=", ib, blk->d);
            for (int j = 0; j < 8; j++) printf("%d ", blk->qs[j]);
            printf("\n");
        }

        // Show input data
        printf("\nFirst 32 input values:\n  ");
        for (int i = 0; i < std::min(ncols, 32); i++) {
            printf("%.3f ", input_data[i]);
            if ((i + 1) % 8 == 0) printf("\n  ");
        }
        printf("\n");
    }

    // Compare results
    const float abs_tolerance = 0.001f;
    const float rel_tolerance = 0.01f;  // 1%

    for (int i = 0; i < output_size; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        float rel = (fabsf(cpu_output[i]) > 1e-6f) ? diff / fabsf(cpu_output[i]) : diff;

        if (diff > result.max_err) {
            result.max_err = diff;
            result.worst_idx = i;
        }
        if (rel > result.max_rel_err) {
            result.max_rel_err = rel;
        }

        if (diff > abs_tolerance && rel > rel_tolerance) {
            result.errors++;
        }
    }

    result.passed = (result.errors == 0);

    if (verbose || !result.passed) {
        const char* path = (batch == 1) ? "DMMV" : "MMQ";
        printf("  %s ncols=%4d nrows=%4d batch=%d: errors=%3d/%d max_err=%.6f max_rel=%.4f %s\n",
               path, ncols, nrows, batch, result.errors, output_size,
               result.max_err, result.max_rel_err,
               result.passed ? "PASS" : "FAIL");

        if (!result.passed && result.worst_idx >= 0 && !debug) {
            printf("    Worst at idx %d: CPU=%.6f GPU=%.6f diff=%.6f\n",
                   result.worst_idx, cpu_output[result.worst_idx],
                   gpu_output[result.worst_idx],
                   fabsf(gpu_output[result.worst_idx] - cpu_output[result.worst_idx]));
        }
    }

    return result;
}

int main(int argc, char** argv) {
    bool verbose = false;
    bool debug = false;
    int debug_ncols = 0, debug_nrows = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose = true;
        else if (strcmp(argv[i], "-d") == 0) debug = true;
        else if (strcmp(argv[i], "-ncols") == 0 && i+1 < argc) debug_ncols = atoi(argv[++i]);
        else if (strcmp(argv[i], "-nrows") == 0 && i+1 < argc) debug_nrows = atoi(argv[++i]);
    }

    printf("=== DMMV/MMQ Q8_0 Production Kernel Test ===\n");
    if (debug) printf("DEBUG MODE: comparing manual reference vs CPU backend vs GPU\n");
    printf("\n");

    // Initialize backends
    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("FAIL: Could not initialize SYCL backend\n");
        return 1;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(gpu_backend);
        return 1;
    }

    printf("GPU Backend: %s\n", ggml_backend_name(gpu_backend));
    printf("CPU Backend: %s\n\n", ggml_backend_name(cpu_backend));

    // If specific debug case requested
    if (debug && debug_ncols > 0 && debug_nrows > 0) {
        printf("Running single debug case: ncols=%d nrows=%d\n", debug_ncols, debug_nrows);
        run_test(gpu_backend, cpu_backend, debug_ncols, debug_nrows, 1, true, true);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(gpu_backend);
        return 0;
    }

    // Test matrix configurations
    std::vector<int> test_ncols = {32, 64, 96, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> test_nrows = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    // In debug mode, run smaller subset with debug output
    if (debug) {
        printf("=== Debug mode: Testing first failing cases ===\n");
        // Test cases around the failure boundary
        std::vector<std::pair<int,int>> debug_cases = {
            {32, 4}, {32, 8}, {32, 16},  // boundary around nrows=8
            {64, 4}, {64, 8},
            {128, 4}, {128, 8},
        };
        for (auto& p : debug_cases) {
            run_test(gpu_backend, cpu_backend, p.first, p.second, 1, true, true);
        }
        ggml_backend_free(cpu_backend);
        ggml_backend_free(gpu_backend);
        return 0;
    }

    int total_dmmv = 0, passed_dmmv = 0;
    int total_mmq = 0, passed_mmq = 0;

    printf("=== Testing DMMV (batch=1) ===\n");
    for (int ncols : test_ncols) {
        for (int nrows : test_nrows) {
            TestResult res = run_test(gpu_backend, cpu_backend, ncols, nrows, 1, verbose, false);
            total_dmmv++;
            if (res.passed) passed_dmmv++;
        }
    }

    printf("\n=== Testing MMQ (batch=4) ===\n");
    for (int ncols : test_ncols) {
        for (int nrows : test_nrows) {
            TestResult res = run_test(gpu_backend, cpu_backend, ncols, nrows, 4, verbose, false);
            total_mmq++;
            if (res.passed) passed_mmq++;
        }
    }

    printf("\n=== Summary ===\n");
    printf("DMMV (batch=1): %d/%d passed\n", passed_dmmv, total_dmmv);
    printf("MMQ  (batch=4): %d/%d passed\n", passed_mmq, total_mmq);
    printf("Total: %d/%d passed\n", passed_dmmv + passed_mmq, total_dmmv + total_mmq);

    int failed = (total_dmmv - passed_dmmv) + (total_mmq - passed_mmq);
    if (failed > 0) {
        printf("\n%d tests FAILED\n", failed);
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    return (failed > 0) ? 1 : 0;
}
