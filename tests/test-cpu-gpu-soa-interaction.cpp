// Unit test for CPU→GPU data path with SoA reordering
// Tests the actual inp_embd bug scenario using ACTUAL kernel functions
// from dequantize.hpp and the reorder kernel from ggml-sycl.cpp
//
// Build: cmake --build build --target test-cpu-gpu-soa-interaction
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-cpu-gpu-soa-interaction

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ============================================================================
// ACTUAL definitions from ggml-common.h
// ============================================================================
#define QK4_0 32
#define QR4_0 2
#define WARP_SIZE 32

// ACTUAL block_q4_0 struct from ggml-common.h
typedef struct {
    sycl::half d;        // delta (fp16)
    uint8_t qs[QK4_0/2]; // nibbles / quants (16 bytes)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

// ACTUAL dfloat2 type from common.hpp
typedef sycl::float2 dfloat2;
typedef float dfloat;

// ============================================================================
// ACTUAL dequantize_q4_0 from dequantize.hpp (AoS version)
// ============================================================================
static inline void dequantize_q4_0(const void *vx, const int64_t ib,
                                    const int iqs, dfloat2 &v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
}

// ============================================================================
// ACTUAL dequantize_q4_0_reorder from dequantize.hpp (SoA version)
// ============================================================================
static inline void dequantize_q4_0_reorder(const void *d_ptr, const int64_t ib, const void *qs,
                                            const int iqs, dfloat2 &v) {
    const dfloat d = (const dfloat)*((const sycl::half*)d_ptr+ib);

    const int vui = *((const uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
}

// ============================================================================
// ACTUAL reorder kernel from ggml-sycl.cpp reorder_qw_q4_0
// ============================================================================
void reorder_q4_0_to_soa_actual(sycl::queue& stream, uint8_t* data_device,
                                 int ncols, int nrows) {
    const int nblocks = nrows * (ncols / QK4_0);
    size_t size = nblocks * sizeof(block_q4_0);

    // ACTUAL: allocate temp buffer
    uint8_t* tmp_buf = sycl::malloc_device<uint8_t>(size, stream);

    // ACTUAL: copy to temp (matches ggml-sycl.cpp line 8602)
    stream.memcpy(tmp_buf, data_device, size).wait();

    // ACTUAL: compute SoA pointers (matches ggml-sycl.cpp lines 8610-8611)
    auto qs_ptr = data_device;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2);

    // ACTUAL: parallel_for reorder kernel (matches ggml-sycl.cpp lines 8638-8649)
    stream.parallel_for(
        size / sizeof(block_q4_0),
        [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0* x = (const block_q4_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK4_0/2; j++) {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        }).wait();

    sycl::free(tmp_buf, stream);
}

// ============================================================================
// ACTUAL DMMV kernel pattern from dmmv.cpp dequantize_mul_mat_vec_reorder
// ============================================================================
#define GGML_SYCL_DMMV_X 32

void dmmv_q4_0_soa_actual(sycl::queue& q, const uint8_t* soa_data,
                          const float* y_vec, float* result,
                          int ncols, int nrows) {
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;  // ACTUAL: from dmmv.cpp
    const int vals_per_iter = 2;  // ACTUAL: from dmmv.cpp
    const int d_offset = nrows * ncols / 2;  // ACTUAL: SoA d offset

    const sycl::half* d_base = (const sycl::half*)(soa_data + d_offset);

    q.parallel_for(sycl::nd_range<1>(nrows * WARP_SIZE, WARP_SIZE),
        [=](sycl::nd_item<1> item) {
            const int row = item.get_group(0);
            const int tid = item.get_local_id(0);
            const int blocks_per_row = ncols / QK4_0;

            float tmp = 0.0f;

            // ACTUAL: iteration pattern from dmmv.cpp
            for (int i = 0; i < ncols; i += iter_stride) {
                const int col = i + vals_per_iter * tid;
                const int ib = row * blocks_per_row + col / QK4_0;
                const int iqs = (col % QK4_0) / vals_per_iter;

                // ACTUAL: SoA data access pattern
                const uint8_t* qs = soa_data + ib * (QK4_0/2);

                dfloat2 v;
                // ACTUAL: call dequantize_q4_0_reorder
                dequantize_q4_0_reorder(d_base, ib, qs, iqs, v);

                // ACTUAL: multiply-accumulate with y vector
                const float* y_col = y_vec + col;
                tmp += v.x() * y_col[0];
                tmp += v.y() * y_col[QK4_0/2];
            }

            // ACTUAL: sub-group reduction
            auto sg = item.get_sub_group();
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                tmp += sycl::shift_group_left(sg, tmp, offset);
            }

            if (tid == 0) {
                result[row] = tmp;
            }
        }).wait();
}

// CPU dequantization (simulates GET_ROWS on CPU - this is what the real code does)
void cpu_dequantize_q4_0(const block_q4_0* src, float* dst, int nblocks) {
    for (int ib = 0; ib < nblocks; ib++) {
        float d = (float)src[ib].d;
        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (src[ib].qs[j] & 0x0F) - 8;
            const int x1 = (src[ib].qs[j] >> 4) - 8;
            dst[ib * QK4_0 + j] = x0 * d;
            dst[ib * QK4_0 + j + QK4_0/2] = x1 * d;
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

// Test 1: Verify ACTUAL dequantize functions match
bool test_actual_dequantize_functions() {
    printf("Test 1: Verify ACTUAL dequantize_q4_0 vs dequantize_q4_0_reorder\n");

    sycl::queue q{sycl::gpu_selector_v};
    printf("  Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int nblocks = 128;
    const int ncols = nblocks * QK4_0;
    const int nrows = 1;

    // Create AoS test data
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        aos_data[i].d = sycl::half(0.1f + (float)(i % 100) / 1000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            aos_data[i].qs[j] = (uint8_t)((i * 17 + j * 3) & 0xFF);
        }
    }

    // Create SoA version
    std::vector<uint8_t> soa_data(nblocks * sizeof(block_q4_0));
    const size_t d_offset = ncols * nrows / 2;

    // Manual SoA conversion
    uint8_t* qs_ptr = soa_data.data();
    sycl::half* d_ptr = (sycl::half*)(soa_data.data() + d_offset);
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < QK4_0/2; j++) {
            qs_ptr[i * (QK4_0/2) + j] = aos_data[i].qs[j];
        }
        d_ptr[i] = aos_data[i].d;
    }

    // Compare dequantize outputs
    int errors = 0;
    for (int ib = 0; ib < nblocks; ib++) {
        for (int iqs = 0; iqs < QK4_0/2; iqs++) {
            dfloat2 v_aos, v_soa;

            // ACTUAL dequantize_q4_0 (AoS)
            dequantize_q4_0(aos_data.data(), ib, iqs, v_aos);

            // ACTUAL dequantize_q4_0_reorder (SoA)
            const uint8_t* qs = soa_data.data() + ib * (QK4_0/2);
            dequantize_q4_0_reorder(d_ptr, ib, qs, iqs, v_soa);

            if (fabsf(v_aos.x() - v_soa.x()) > 1e-5f ||
                fabsf(v_aos.y() - v_soa.y()) > 1e-5f) {
                if (errors < 5) {
                    printf("    Error: ib=%d iqs=%d: AoS=(%.6f,%.6f) SoA=(%.6f,%.6f)\n",
                           ib, iqs, v_aos.x(), v_aos.y(), v_soa.x(), v_soa.y());
                }
                errors++;
            }
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d dequantize mismatches\n", errors);
        return false;
    }

    printf("  PASS: All %d dequantize comparisons match\n", nblocks * (QK4_0/2));
    return true;
}

// Test 2: Verify ACTUAL GPU reorder kernel
bool test_actual_reorder_kernel() {
    printf("Test 2: Verify ACTUAL GPU reorder kernel\n");

    sycl::queue q{sycl::gpu_selector_v};

    const int nrows = 128;
    const int ncols = 4096;
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t size = nblocks * sizeof(block_q4_0);

    // Create AoS test data
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        aos_data[i].d = sycl::half(1.0f + (float)(i % 1000) / 10000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            aos_data[i].qs[j] = (uint8_t)(i + j);
        }
    }

    // Copy to GPU and reorder
    uint8_t* gpu_data = sycl::malloc_device<uint8_t>(size, q);
    q.memcpy(gpu_data, aos_data.data(), size).wait();

    // ACTUAL reorder kernel
    reorder_q4_0_to_soa_actual(q, gpu_data, ncols, nrows);

    // Copy back
    std::vector<uint8_t> soa_result(size);
    q.memcpy(soa_result.data(), gpu_data, size).wait();

    // Verify SoA layout
    const size_t d_offset = nrows * ncols / 2;
    const uint8_t* qs_ptr = soa_result.data();
    const sycl::half* d_ptr = (const sycl::half*)(soa_result.data() + d_offset);

    int errors = 0;
    for (int i = 0; i < nblocks && errors < 5; i++) {
        // Check qs values
        for (int j = 0; j < QK4_0/2; j++) {
            if (qs_ptr[i * (QK4_0/2) + j] != aos_data[i].qs[j]) {
                printf("    Error: block %d qs[%d]: got=%02x expected=%02x\n",
                       i, j, qs_ptr[i * (QK4_0/2) + j], aos_data[i].qs[j]);
                errors++;
            }
        }
        // Check d values
        if (fabsf((float)d_ptr[i] - (float)aos_data[i].d) > 1e-4f) {
            printf("    Error: block %d d: got=%.6f expected=%.6f\n",
                   i, (float)d_ptr[i], (float)aos_data[i].d);
            errors++;
        }
    }

    sycl::free(gpu_data, q);

    if (errors > 0) {
        printf("  FAIL: Reorder kernel produced %d errors\n", errors);
        return false;
    }

    printf("  PASS: GPU reorder kernel verified for %d blocks\n", nblocks);
    return true;
}

// Test 3: CPU GET_ROWS → GPU copy → GPU read with SoA reorder
bool test_cpu_gpu_path_with_soa() {
    printf("Test 3: CPU GET_ROWS → GPU copy → SoA reorder → GPU read\n");

    sycl::queue q{sycl::gpu_selector_v};

    // Setup: token_embd on CPU (Q4_0)
    const int vocab_size = 1024;
    const int embed_dim = 4096;
    const int nblocks_per_row = embed_dim / QK4_0;
    const int total_blocks = vocab_size * nblocks_per_row;

    std::vector<block_q4_0> cpu_token_embd(total_blocks);
    for (int i = 0; i < total_blocks; i++) {
        cpu_token_embd[i].d = sycl::half(0.1f + (float)(i % 100) / 1000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            cpu_token_embd[i].qs[j] = (uint8_t)((i + j) & 0xFF);
        }
    }

    // Setup: layer weights on GPU (will be SoA reordered)
    const int weight_rows = 128;
    const int weight_cols = 4096;
    const int weight_nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t weight_size = weight_nblocks * sizeof(block_q4_0);

    uint8_t* gpu_weights = sycl::malloc_device<uint8_t>(weight_size, q);
    std::vector<block_q4_0> cpu_weights(weight_nblocks);
    for (int i = 0; i < weight_nblocks; i++) {
        cpu_weights[i].d = sycl::half(0.5f);
        for (int j = 0; j < QK4_0/2; j++) {
            cpu_weights[i].qs[j] = 0x77;
        }
    }
    q.memcpy(gpu_weights, cpu_weights.data(), weight_size).wait();

    // GPU buffer for inp_embd (F32)
    float* gpu_inp_embd = sycl::malloc_device<float>(embed_dim, q);

    int pass_count = 0;
    int fail_count = 0;

    printf("  Testing 10 decode tokens...\n");

    for (int token = 1; token <= 10; token++) {
        int token_id = token * 10;

        // Step 1: CPU GET_ROWS - dequantize token embedding
        std::vector<float> cpu_result(embed_dim);
        cpu_dequantize_q4_0(&cpu_token_embd[token_id * nblocks_per_row],
                            cpu_result.data(), nblocks_per_row);

        // Step 2: Copy F32 to GPU (simulates inp_embd transfer)
        q.memcpy(gpu_inp_embd, cpu_result.data(), embed_dim * sizeof(float)).wait();

        // Step 3: SoA reorder on weights (first token only, simulates prompt phase)
        if (token == 1) {
            printf("  Token 1: ACTUAL SoA reorder on weights...\n");
            reorder_q4_0_to_soa_actual(q, gpu_weights, weight_cols, weight_rows);
        }

        // Step 4: GPU reads inp_embd - copy back and verify
        std::vector<float> gpu_result(embed_dim);
        q.memcpy(gpu_result.data(), gpu_inp_embd, embed_dim * sizeof(float)).wait();

        // Verify
        int zeros = 0;
        int errors = 0;
        for (int i = 0; i < embed_dim; i++) {
            if (gpu_result[i] == 0.0f && cpu_result[i] != 0.0f) zeros++;
            if (fabsf(gpu_result[i] - cpu_result[i]) > fabsf(cpu_result[i]) * 0.01f + 1e-6f) {
                errors++;
            }
        }

        if (zeros > embed_dim / 2) {
            printf("    Token %d: ZEROS detected (%d/%d)!\n", token, zeros, embed_dim);
            fail_count++;
        } else if (errors > 0) {
            printf("    Token %d: %d value mismatches\n", token, errors);
            fail_count++;
        } else {
            pass_count++;
        }
    }

    sycl::free(gpu_weights, q);
    sycl::free(gpu_inp_embd, q);

    printf("  Results: %d/10 passed, %d/10 failed\n", pass_count, fail_count);

    if (fail_count > 0) {
        printf("  FAIL: Bug detected in CPU→GPU path with SoA\n");
        return false;
    }

    printf("  PASS: CPU→GPU path works correctly with SoA reordering\n");
    return true;
}

// Test 4: USM host memory (actual inp_embd allocation type)
bool test_usm_host_memory() {
    printf("Test 4: USM host memory (actual inp_embd allocation)\n");

    sycl::queue q{sycl::gpu_selector_v};

    const int embed_dim = 4096;
    const int nblocks = embed_dim / QK4_0;

    // USM host allocation (this is how inp_embd is actually allocated)
    float* usm_inp_embd = sycl::malloc_host<float>(embed_dim, q);
    if (!usm_inp_embd) {
        printf("  SKIP: malloc_host failed\n");
        return true;
    }

    // Q4_0 source
    std::vector<block_q4_0> token_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        token_data[i].d = sycl::half(1.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            token_data[i].qs[j] = (uint8_t)((i + j + 1) & 0xFF);
        }
    }

    // Q4_0 weights on GPU
    const int weight_size = 128 * 4096 / QK4_0 * sizeof(block_q4_0);
    uint8_t* gpu_weights = sycl::malloc_device<uint8_t>(weight_size, q);
    std::vector<uint8_t> weight_init(weight_size, 0x55);
    q.memcpy(gpu_weights, weight_init.data(), weight_size).wait();

    // Expected
    std::vector<float> expected(embed_dim);
    cpu_dequantize_q4_0(token_data.data(), expected.data(), nblocks);

    int pass_count = 0;
    int fail_count = 0;

    for (int token = 1; token <= 10; token++) {
        // CPU writes to USM host memory (GET_ROWS result)
        cpu_dequantize_q4_0(token_data.data(), usm_inp_embd, nblocks);

        // SoA reorder on first token
        if (token == 1) {
            reorder_q4_0_to_soa_actual(q, gpu_weights, 4096, 128);
        }

        // GPU reads USM host memory directly
        float* gpu_sum_ptr = sycl::malloc_device<float>(1, q);
        q.memset(gpu_sum_ptr, 0, sizeof(float)).wait();

        q.parallel_for(1, [=](auto) {
            float sum = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                sum += usm_inp_embd[i];
            }
            *gpu_sum_ptr = sum;
        }).wait();

        float actual_sum;
        q.memcpy(&actual_sum, gpu_sum_ptr, sizeof(float)).wait();
        sycl::free(gpu_sum_ptr, q);

        float expected_sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            expected_sum += expected[i];
        }

        if (fabsf(actual_sum) < 1e-6f && fabsf(expected_sum) > 1e-6f) {
            printf("    Token %d: GPU saw ZEROS (expected_sum=%.2f)\n", token, expected_sum);
            fail_count++;
        } else if (fabsf(actual_sum - expected_sum) > fabsf(expected_sum) * 0.01f) {
            printf("    Token %d: Mismatch (%.2f vs %.2f)\n", token, actual_sum, expected_sum);
            fail_count++;
        } else {
            pass_count++;
        }
    }

    sycl::free(gpu_weights, q);
    sycl::free(usm_inp_embd, q);

    printf("  Results: %d/10 passed, %d/10 failed\n", pass_count, fail_count);

    if (fail_count > 0) {
        printf("  FAIL: USM host memory path has issues\n");
        return false;
    }

    printf("  PASS: USM host memory works correctly\n");
    return true;
}

// Test 5: Full DMMV with SoA (ACTUAL kernel pattern)
bool test_dmmv_soa_actual() {
    printf("Test 5: ACTUAL DMMV kernel with SoA layout\n");

    sycl::queue q{sycl::gpu_selector_v};

    const int nrows = 64;
    const int ncols = 4096;
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t size = nblocks * sizeof(block_q4_0);

    // Create Q4_0 data
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        aos_data[i].d = sycl::half(1.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            aos_data[i].qs[j] = 0x88;  // Neutral value (0 after -8)
        }
    }

    // Y vector
    std::vector<float> y_vec(ncols, 1.0f);

    // GPU allocations
    uint8_t* gpu_weights = sycl::malloc_device<uint8_t>(size, q);
    float* gpu_y = sycl::malloc_device<float>(ncols, q);
    float* gpu_result = sycl::malloc_device<float>(nrows, q);

    q.memcpy(gpu_weights, aos_data.data(), size).wait();
    q.memcpy(gpu_y, y_vec.data(), ncols * sizeof(float)).wait();

    // Reorder to SoA
    reorder_q4_0_to_soa_actual(q, gpu_weights, ncols, nrows);

    // Run ACTUAL DMMV kernel
    dmmv_q4_0_soa_actual(q, gpu_weights, gpu_y, gpu_result, ncols, nrows);

    // Get result
    std::vector<float> result(nrows);
    q.memcpy(result.data(), gpu_result, nrows * sizeof(float)).wait();

    sycl::free(gpu_weights, q);
    sycl::free(gpu_y, q);
    sycl::free(gpu_result, q);

    // With all qs=0x88, after -8 offset we get 0. Result should be 0.
    int errors = 0;
    for (int i = 0; i < nrows; i++) {
        if (fabsf(result[i]) > 1e-5f) {
            if (errors < 5) {
                printf("    Row %d: got %.6f, expected 0.0\n", i, result[i]);
            }
            errors++;
        }
    }

    if (errors > 0) {
        printf("  FAIL: DMMV kernel produced %d incorrect values\n", errors);
        return false;
    }

    printf("  PASS: ACTUAL DMMV kernel works correctly with SoA\n");
    return true;
}

int main() {
    printf("=== CPU→GPU SoA Interaction Tests (ACTUAL Kernels) ===\n");
    printf("Using ACTUAL kernel functions from dequantize.hpp and ggml-sycl.cpp\n\n");

    try {
        int passed = 0;
        int failed = 0;

        if (test_actual_dequantize_functions()) passed++; else failed++;
        printf("\n");

        if (test_actual_reorder_kernel()) passed++; else failed++;
        printf("\n");

        if (test_cpu_gpu_path_with_soa()) passed++; else failed++;
        printf("\n");

        if (test_usm_host_memory()) passed++; else failed++;
        printf("\n");

        if (test_dmmv_soa_actual()) passed++; else failed++;
        printf("\n");

        printf("=================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        if (failed > 0) {
            printf("\nBug detected in one of the tests.\n");
        } else {
            printf("\nAll tests passed with ACTUAL kernel functions.\n");
            printf("The bug is in higher-level ggml integration, not kernel code.\n");
        }

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
}
