// Q6_K MMQ (Matrix-Matrix Quantized) GPU unit test
// Tests the production MMQ kernel path used during prompt processing (batch > 1)
//
// Build: cmake --build build --target test-mmq-q6k-gpu
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-q6k-gpu
//
// This test verifies that the MMQ Q6_K kernel produces correct results
// for multi-token batches (used during prompt processing).

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

// Minimal struct definitions to inspect tensor extra (mirrors ggml-sycl/common.hpp)
// These must match the actual SYCL backend structs exactly!
enum class test_reorder_mode : uint8_t {
    NONE = 0,
    SOA = 1,
    COALESCED = 2,
};

struct test_optimize_feature {
    test_reorder_mode reorder_ = test_reorder_mode::NONE;
    void* data_owner_ = nullptr;  // For view tensors

    test_reorder_mode get_reorder() const {
        if (data_owner_ != nullptr) {
            return ((test_optimize_feature*)data_owner_)->get_reorder();
        }
        return reorder_;
    }
    bool is_soa() const { return get_reorder() == test_reorder_mode::SOA; }
    bool is_coalesced() const { return get_reorder() == test_reorder_mode::COALESCED; }
};

// Use actual values from ggml-sycl.h (redefined to match real struct layout)
#undef GGML_SYCL_MAX_DEVICES  // Avoid redefinition warning
#define GGML_SYCL_MAX_DEVICES 48  // Real value from ggml-sycl.h
#define GGML_SYCL_MAX_STREAMS 8

// Minimal version of ggml_tensor_extra_gpu for inspection
struct test_tensor_extra_gpu {
    void* data_device[GGML_SYCL_MAX_DEVICES];
    void* events[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS];
    test_optimize_feature optimized_feature;
    // ... rest of fields not needed for this check
};

// CPU reference: compute full matmul Q6_K @ F32
static void cpu_mul_mat_q6k_f32(const void* x_data, const float* y,
                                 float* dst, int nrows, int ncols, int n_tokens) {
    const block_q6_K* x = (const block_q6_K*)x_data;
    const int blocks_per_row = ncols / QK_K;

    // Dequantize each row and compute dot product
    std::vector<float> x_f32(ncols);

    for (int row = 0; row < nrows; row++) {
        // Dequantize this row
        dequantize_row_q6_K(&x[row * blocks_per_row], x_f32.data(), ncols);

        // For each token (column of Y)
        for (int tok = 0; tok < n_tokens; tok++) {
            float sum = 0.0f;
            for (int k = 0; k < ncols; k++) {
                sum += x_f32[k] * y[tok * ncols + k];
            }
            // Output is transposed: dst[tok, row]
            dst[tok * nrows + row] = sum;
        }
    }
}

// Test MMQ with specific batch size
static bool test_mmq_q6k_batch(int n_tokens, bool verbose) {
    printf("\n--- MMQ Q6_K batch=%d ---\n", n_tokens);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;  // Skip, not fail
    }
    printf("  Backend: %s\n", ggml_backend_name(backend));

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Use smaller dimensions for faster testing (still exercises MMQ path)
    const int n_embd = 1024;  // Columns (K dimension), must be divisible by QK_K=256
    const int n_ff = 2048;    // Rows (N dimension)

    printf("  Weight: Q6_K [%d x %d], Input: F32 [%d x %d]\n",
           n_embd, n_ff, n_embd, n_tokens);

    // Create context with enough space
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAIL: Could not create ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    // Weight matrix: [n_embd, n_ff] Q6_K
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff);
    ggml_set_name(weight, "test_q6k_weight");

    // Input: [n_embd, n_tokens] F32
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "test_input");

    // Output: [n_ff, n_tokens] F32
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "test_output");

    // Allocate weight buffer (triggers SoA reordering if enabled)
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        printf("  FAIL: Could not allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        printf("  FAIL: Could not allocate compute buffer\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate random test data
    std::mt19937 rng(42 + n_tokens);  // Different seed per batch size
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Create float data for quantization
    const int weight_floats = n_ff * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    // Quantize to Q6_K
    printf("  Quantizing weights to Q6_K...\n");
    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_ff * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_ff, n_embd, nullptr);

    // Upload weight data (this triggers SoA reordering)
    printf("  Uploading weights to GPU...\n");
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    // === DEBUG: Verify d (scale) values ===
    // Read back the GPU tensor data to check SoA layout
    printf("\n  === D (SCALE) VALUE VERIFICATION ===\n");

    // First, print the original AoS d values for first few blocks
    printf("  Original AoS d values (first 8 blocks):\n");
    for (int i = 0; i < 8 && i < total_blocks; i++) {
        // d is stored as ggml_half (2 bytes) at the end of each block
        ggml_half d_half = weight_q6k[i].d;
        float d_float = ggml_fp16_to_fp32(d_half);
        printf("    block[%d].d = 0x%04x (%.6f)\n", i, (unsigned)d_half, d_float);
    }

    // Read back GPU data
    std::vector<uint8_t> gpu_raw_data(total_blocks * sizeof(block_q6_K));
    ggml_backend_tensor_get(weight, gpu_raw_data.data(), 0, total_blocks * sizeof(block_q6_K));

    // Check if SoA reordering happened by checking the tensor extra
    // In SoA layout: ql (nblocks*128) | qh (nblocks*64) | scales (nblocks*16) | d (nblocks*2)
    const int nblocks = total_blocks;
    const size_t ql_offset = 0;
    const size_t qh_offset = nblocks * 128;
    const size_t scales_offset = qh_offset + nblocks * 64;
    const size_t d_offset = scales_offset + nblocks * 16;

    printf("\n  SoA layout offsets for %d blocks:\n", nblocks);
    printf("    ql:     offset=0\n");
    printf("    qh:     offset=%zu\n", qh_offset);
    printf("    scales: offset=%zu\n", scales_offset);
    printf("    d:      offset=%zu\n", d_offset);
    printf("    total:  %zu bytes (vs AoS %zu bytes)\n",
           d_offset + nblocks * 2, nblocks * sizeof(block_q6_K));

    // Interpret GPU data as SoA and read d values
    printf("\n  GPU data interpreted as SoA (d at offset %zu):\n", d_offset);
    for (int i = 0; i < 8 && i < nblocks; i++) {
        size_t d_pos = d_offset + i * 2;  // 2 bytes per half
        if (d_pos + 2 <= gpu_raw_data.size()) {
            ggml_half d_half;
            memcpy(&d_half, &gpu_raw_data[d_pos], 2);
            float d_float = ggml_fp16_to_fp32(d_half);
            printf("    soa_d[%d] at byte %zu = 0x%04x (%.6f)\n",
                   i, d_pos, (unsigned)d_half, d_float);
        }
    }

    // Also interpret GPU data as AoS to see what's there
    printf("\n  GPU data interpreted as AoS (for comparison):\n");
    const block_q6_K* gpu_aos = (const block_q6_K*)gpu_raw_data.data();
    for (int i = 0; i < 8 && i < nblocks; i++) {
        ggml_half d_half = gpu_aos[i].d;
        float d_float = ggml_fp16_to_fp32(d_half);
        printf("    aos_block[%d].d = 0x%04x (%.6f)\n", i, (unsigned)d_half, d_float);
    }

    // Check for zeros or NaN in d values (SoA interpretation)
    int zero_d_count = 0, nan_d_count = 0;
    float d_min = INFINITY, d_max = -INFINITY;
    for (int i = 0; i < nblocks; i++) {
        size_t d_pos = d_offset + i * 2;
        if (d_pos + 2 <= gpu_raw_data.size()) {
            ggml_half d_half;
            memcpy(&d_half, &gpu_raw_data[d_pos], 2);
            float d_float = ggml_fp16_to_fp32(d_half);
            if (std::isnan(d_float)) nan_d_count++;
            else if (d_float == 0.0f) zero_d_count++;
            else {
                d_min = std::min(d_min, d_float);
                d_max = std::max(d_max, d_float);
            }
        }
    }
    printf("\n  SoA d value analysis (%d blocks):\n", nblocks);
    printf("    zeros: %d, NaN: %d\n", zero_d_count, nan_d_count);
    printf("    valid range: [%.6f, %.6f]\n", d_min, d_max);

    // Check if tensor extra is set (required for SoA kernel path)
    printf("\n  Tensor extra check:\n");
    printf("    weight->extra = %p\n", (void*)weight->extra);
    if (!weight->extra) {
        printf("    WARNING: tensor->extra is NULL - SoA path will NOT be used!\n");
    } else {
        // Dump raw bytes to verify struct layout
        const uint8_t* raw = (const uint8_t*)weight->extra;
        printf("    Raw bytes at extra (first 32): ");
        for (int i = 0; i < 32; i++) printf("%02x ", raw[i]);
        printf("\n");

        // Calculate expected offsets based on real struct layout:
        // data_device[48]: 48 * 8 = 384 bytes
        // events[48][8]: dpct::event_ptr is a pointer, so 48 * 8 * 8 = 3072 bytes
        // Total offset to optimized_feature: 384 + 3072 = 3456 bytes
        const size_t opt_offset = 48 * sizeof(void*) + 48 * 8 * sizeof(void*);
        printf("    Expected offset to optimized_feature: %zu bytes\n", opt_offset);
        printf("    Bytes at offset %zu: ", opt_offset);
        for (size_t i = opt_offset; i < opt_offset + 16 && i < 4096; i++) {
            printf("%02x ", raw[i]);
        }
        printf("\n");

        // The first byte at opt_offset should be reorder_mode (0=NONE, 1=SOA, 2=COALESCED)
        uint8_t reorder_byte = raw[opt_offset];
        printf("    reorder_mode byte at offset %zu = %d (0=NONE, 1=SOA, 2=COALESCED)\n",
               opt_offset, reorder_byte);

        // Also try our struct interpretation
        test_tensor_extra_gpu* extra = (test_tensor_extra_gpu*)weight->extra;
        printf("    test_struct: optimized_feature.reorder_ = %d\n",
               (int)extra->optimized_feature.reorder_);
        printf("    test_struct: data_device[0] = %p\n", extra->data_device[0]);

        // Check if the real reorder_mode is SOA
        if (reorder_byte == 1) {
            printf("    Real reorder_mode is SOA - kernel should use SoA path\n");
        } else if (reorder_byte == 0) {
            printf("    WARNING: Real reorder_mode is NONE - AoS kernel will be used despite SoA data!\n");
        } else {
            printf("    Real reorder_mode = %d (unexpected value)\n", reorder_byte);
        }

        // === DATA LAYOUT VERIFICATION ===
        // Verify data actually matches the layout it claims to be in
        printf("\n  === DATA LAYOUT VERIFICATION ===\n");
        printf("    Testing if data layout matches claimed reorder_mode...\n");

        // Get the original d value from first block (from weight_q6k)
        ggml_half expected_d = weight_q6k[0].d;
        printf("    Expected d[0] = 0x%04x (%.6f)\n",
               (unsigned)expected_d, ggml_fp16_to_fp32(expected_d));

        // Read d value assuming AoS layout (offset 208 in each 210-byte block)
        ggml_half aos_d_val;
        memcpy(&aos_d_val, &gpu_raw_data[208], sizeof(ggml_half));  // d offset in block_q6_K
        printf("    AoS interpretation d[0] = 0x%04x (%.6f)\n",
               (unsigned)aos_d_val, ggml_fp16_to_fp32(aos_d_val));

        // Read d value assuming SoA layout (at d_offset)
        ggml_half soa_d_val;
        memcpy(&soa_d_val, &gpu_raw_data[d_offset], sizeof(ggml_half));
        printf("    SoA interpretation d[0] = 0x%04x (%.6f)\n",
               (unsigned)soa_d_val, ggml_fp16_to_fp32(soa_d_val));

        // Determine actual layout from data
        bool aos_matches = (aos_d_val == expected_d);
        bool soa_matches = (soa_d_val == expected_d);

        const char* actual_layout = "UNKNOWN";
        if (aos_matches && !soa_matches) {
            actual_layout = "AoS";
        } else if (soa_matches && !aos_matches) {
            actual_layout = "SoA";
        } else if (aos_matches && soa_matches) {
            actual_layout = "AMBIGUOUS (both match)";
        } else {
            actual_layout = "CORRUPTED (neither matches)";
        }

        const char* claimed_layout = (reorder_byte == 1) ? "SoA" : "AoS";

        printf("    Actual data layout: %s\n", actual_layout);
        printf("    Claimed layout (reorder_mode): %s\n", claimed_layout);

        bool layout_mismatch = false;
        if (soa_matches && reorder_byte == 0) {
            printf("    *** BUG CONFIRMED: Data is SoA but reorder_mode claims AoS! ***\n");
            printf("    *** This causes AoS kernel to run on SoA data → NaN/garbage ***\n");
            layout_mismatch = true;
        } else if (aos_matches && reorder_byte == 1) {
            printf("    *** BUG CONFIRMED: Data is AoS but reorder_mode claims SoA! ***\n");
            printf("    *** This causes SoA kernel to run on AoS data → NaN/garbage ***\n");
            layout_mismatch = true;
        } else if (aos_matches && reorder_byte == 0) {
            printf("    OK: Data is AoS and reorder_mode correctly says AoS\n");
        } else if (soa_matches && reorder_byte == 1) {
            printf("    OK: Data is SoA and reorder_mode correctly says SoA\n");
        }
        printf("  === END DATA LAYOUT VERIFICATION ===\n");
        (void)layout_mismatch;  // suppress unused warning
    }
    printf("  === END D VALUE VERIFICATION ===\n\n");

    // Create input data [n_embd, n_tokens]
    std::vector<float> input_f32(n_embd * n_tokens);
    for (int i = 0; i < n_embd * n_tokens; i++) {
        input_f32[i] = dist(rng);
    }
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * n_tokens * sizeof(float));

    // Build graph
    printf("  Building compute graph...\n");
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph) {
        printf("  FAIL: Could not create graph\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, output);

    // Execute graph
    printf("  Executing MMQ graph...\n");
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Get GPU output [n_ff, n_tokens]
    std::vector<float> gpu_output(n_ff * n_tokens);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_ff * n_tokens * sizeof(float));

    // Compute CPU reference
    printf("  Computing CPU reference...\n");
    std::vector<float> cpu_output(n_ff * n_tokens);
    cpu_mul_mat_q6k_f32(weight_q6k.data(), input_f32.data(),
                        cpu_output.data(), n_ff, n_embd, n_tokens);

    // === DETAILED DEBUG OUTPUT ===
    printf("\n  === DETAILED COMPARISON ===\n");

    // 1. Check for NaN/Inf values
    int nan_count = 0, inf_count = 0;
    float gpu_min = gpu_output[0], gpu_max = gpu_output[0];
    float cpu_min = cpu_output[0], cpu_max = cpu_output[0];

    for (int i = 0; i < n_ff * n_tokens; i++) {
        if (std::isnan(gpu_output[i])) nan_count++;
        if (std::isinf(gpu_output[i])) inf_count++;
        if (!std::isnan(gpu_output[i]) && !std::isinf(gpu_output[i])) {
            gpu_min = std::min(gpu_min, gpu_output[i]);
            gpu_max = std::max(gpu_max, gpu_output[i]);
        }
        cpu_min = std::min(cpu_min, cpu_output[i]);
        cpu_max = std::max(cpu_max, cpu_output[i]);
    }

    printf("  GPU output stats: min=%.6f max=%.6f NaN=%d Inf=%d\n",
           gpu_min, gpu_max, nan_count, inf_count);
    printf("  CPU output stats: min=%.6f max=%.6f\n", cpu_min, cpu_max);

    // 2. Print first 16 values side by side for token 0
    printf("\n  First 16 values (token=0):\n");
    printf("  %6s  %14s  %14s  %14s  %10s\n", "row", "GPU", "CPU", "diff", "rel_err%");
    for (int row = 0; row < std::min(16, n_ff); row++) {
        float gpu_val = gpu_output[row];  // token=0, so idx = row
        float cpu_val = cpu_output[row];
        float diff = gpu_val - cpu_val;
        float rel_err = (std::abs(cpu_val) > 1e-6f) ? std::abs(diff) / std::abs(cpu_val) * 100 : 0;
        printf("  %6d  %14.6f  %14.6f  %14.6f  %10.2f\n", row, gpu_val, cpu_val, diff, rel_err);
    }

    // 3. Print values at different positions to check patterns
    printf("\n  Sample values at different positions:\n");
    printf("  %8s  %14s  %14s  %14s\n", "idx", "GPU", "CPU", "diff");
    int sample_positions[] = {0, n_ff/4, n_ff/2, 3*n_ff/4, n_ff-1,
                              n_ff, n_ff + n_ff/2, n_ff * n_tokens - 1};
    for (int i = 0; i < 8 && sample_positions[i] < n_ff * n_tokens; i++) {
        int idx = sample_positions[i];
        float gpu_val = gpu_output[idx];
        float cpu_val = cpu_output[idx];
        printf("  %8d  %14.6f  %14.6f  %14.6f\n", idx, gpu_val, cpu_val, gpu_val - cpu_val);
    }

    // 4. Per-token error summary
    printf("\n  Per-token error summary:\n");
    for (int tok = 0; tok < n_tokens; tok++) {
        float sum_abs_err = 0, max_abs_err = 0;
        int tok_mismatches = 0;
        for (int row = 0; row < n_ff; row++) {
            int idx = tok * n_ff + row;
            float abs_err = std::abs(gpu_output[idx] - cpu_output[idx]);
            sum_abs_err += abs_err;
            max_abs_err = std::max(max_abs_err, abs_err);
            if (abs_err > 0.1f) tok_mismatches++;
        }
        printf("  Token %d: avg_err=%.6f max_err=%.6f mismatches=%d/%d\n",
               tok, sum_abs_err / n_ff, max_abs_err, tok_mismatches, n_ff);
    }

    // Compare results using hybrid tolerance:
    // - Absolute tolerance: 0.1 (for values near zero where relative error explodes)
    // - Relative tolerance: 15% (normal quantization noise is 5-10%)
    // Values must exceed BOTH thresholds to be a mismatch
    // SoA bug produces errors of 100000%+ so this still catches it
    const float ABS_TOL = 0.1f;
    const float REL_TOL = 0.15f;  // 15%

    int mismatches = 0;
    float max_rel_error = 0.0f;
    float max_abs_error = 0.0f;
    int mismatch_tok = -1, mismatch_row = -1;
    float mismatch_gpu = 0, mismatch_cpu = 0;

    for (int tok = 0; tok < n_tokens; tok++) {
        for (int row = 0; row < n_ff; row++) {
            int idx = tok * n_ff + row;
            float gpu_val = gpu_output[idx];
            float cpu_val = cpu_output[idx];

            float abs_diff = std::abs(gpu_val - cpu_val);
            float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;

            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
                max_abs_error = abs_diff;
                mismatch_tok = tok;
                mismatch_row = row;
                mismatch_gpu = gpu_val;
                mismatch_cpu = cpu_val;
            }

            // Hybrid check: must fail BOTH abs and rel thresholds
            bool abs_fail = abs_diff > ABS_TOL;
            bool rel_fail = rel_error > REL_TOL;
            if (abs_fail && rel_fail) {
                if (verbose && mismatches < 5) {
                    printf("  MISMATCH tok=%d row=%d: GPU=%.6f CPU=%.6f diff=%.6f (%.2f%%)\n",
                           tok, row, gpu_val, cpu_val, abs_diff, rel_error * 100);
                }
                mismatches++;
            }
        }
    }

    printf("\n  Max relative error: %.4f%% at tok=%d row=%d (GPU=%.6f CPU=%.6f)\n",
           max_rel_error * 100, mismatch_tok, mismatch_row, mismatch_gpu, mismatch_cpu);
    printf("  Thresholds: abs>%.2f AND rel>%.0f%%\n", ABS_TOL, REL_TOL * 100);

    // Report SoA status
    const char* disable_opt = getenv("GGML_SYCL_DISABLE_OPT");
    printf("  SoA optimization: %s\n", (disable_opt && strcmp(disable_opt, "1") == 0) ? "disabled (AoS)" : "enabled");

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (mismatches > 0) {
        printf("  FAIL: %d mismatches in %d values (%.2f%%)\n",
               mismatches, n_ff * n_tokens, (float)mismatches / (n_ff * n_tokens) * 100);
        return false;
    }

    printf("  PASS: All values within tolerance (abs>%.2f AND rel>%.0f%%)\n", ABS_TOL, REL_TOL * 100);
    return true;
}

int main() {
    printf("Q6_K MMQ (Matrix-Matrix Quantized) GPU Unit Test\n");
    printf("=================================================\n");
    printf("\nThis tests the MMQ kernel path used for prompt processing (batch > 1)\n");
    printf("MMVQ is disabled (MMVQ_MAX_BATCH_SIZE=0), so all batch>1 goes through MMQ.\n\n");

    int passed = 0;
    int failed = 0;

    // Test 1: Minimal batch (batch=2) - triggers MMQ path
    printf("Test 1: Minimal multi-token batch\n");
    if (test_mmq_q6k_batch(2, true)) passed++; else failed++;

    // Test 2: Typical prompt chunk (batch=8)
    printf("\nTest 2: Typical prompt chunk\n");
    if (test_mmq_q6k_batch(8, true)) passed++; else failed++;

    // Test 3: Larger batch (batch=16)
    printf("\nTest 3: Larger batch\n");
    if (test_mmq_q6k_batch(16, true)) passed++; else failed++;

    printf("\n=================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
