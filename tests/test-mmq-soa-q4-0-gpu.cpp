// GPU Test for Q4_0 MMQ SoA vs AoS comparison
// This test runs actual SYCL kernels to compare SoA vs AoS output
//
// Build: cmake --build build --target test-mmq-soa-q4-0-gpu
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-soa-q4-0-gpu

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Q4_0 block structure (matches ggml-common.h)
#define QK4_0 32
typedef struct {
    sycl::half d;        // delta (fp16)
    uint8_t qs[QK4_0/2]; // nibbles / quants (16 bytes)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

// Simplified reorder kernel (matches ggml-sycl.cpp reorder_qw_q4_0)
void reorder_q4_0_gpu(sycl::queue& q, uint8_t* data, int ncols, int nrows) {
    const int nblocks = nrows * (ncols / QK4_0);
    size_t size = nblocks * sizeof(block_q4_0);

    // Allocate temp buffer
    uint8_t* tmp_buf = sycl::malloc_device<uint8_t>(size, q);
    q.memcpy(tmp_buf, data, size).wait();

    // Compute SoA pointers
    uint8_t* qs_ptr = data;
    sycl::half* d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2);

    // Reorder
    q.parallel_for(nblocks, [=](auto i) {
        const block_q4_0* x = (const block_q4_0*)tmp_buf;
        const int ib = i;

        for (int j = 0; j < QK4_0/2; j++) {
            *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
        }
        *(d_ptr + ib) = x[ib].d;
    }).wait();

    sycl::free(tmp_buf, q);
}

// Create test Q4_0 data with controlled values
void create_test_data(block_q4_0* data, int nrows, int ncols, uint32_t seed) {
    std::mt19937 rng(seed);

    const int nblocks = nrows * (ncols / QK4_0);
    for (int i = 0; i < nblocks; i++) {
        // Use consistent scale = 1.0 (fp16)
        data[i].d = sycl::half(1.0f);
        // Fill qs with predictable values
        for (int j = 0; j < QK4_0/2; j++) {
            // Lower nibble = 0-7, upper nibble = 8-15 (gives values -8 to +7 after subtraction)
            data[i].qs[j] = (uint8_t)((j % 16) | ((j % 16) << 4));
        }
    }
}

// Reference dequantization on CPU
void dequantize_q4_0_cpu(const block_q4_0* x, float* y, int nblocks) {
    for (int i = 0; i < nblocks; i++) {
        float d = (float)x[i].d;
        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i*QK4_0 + j] = x0 * d;
            y[i*QK4_0 + j + QK4_0/2] = x1 * d;
        }
    }
}

// GPU dequantization from AoS layout
void dequantize_q4_0_gpu_aos(sycl::queue& q, const block_q4_0* x, float* y, int nblocks) {
    q.parallel_for(nblocks, [=](auto i) {
        float d = (float)x[i].d;
        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i*QK4_0 + j] = x0 * d;
            y[i*QK4_0 + j + QK4_0/2] = x1 * d;
        }
    }).wait();
}

// GPU dequantization from SoA layout (mimics MMQ loader access pattern)
void dequantize_q4_0_gpu_soa(sycl::queue& q, const uint8_t* soa_data, float* y,
                              int nrows, int ncols) {
    const int nblocks = nrows * (ncols / QK4_0);
    const int blocks_per_row = ncols / QK4_0;
    const size_t d_offset = (size_t)nrows * ncols / 2;

    const uint8_t* qs_base = soa_data;
    const sycl::half* d_base = (const sycl::half*)(soa_data + d_offset);

    // Use same access pattern as MMQ loader
    q.parallel_for(sycl::range<2>(nrows, blocks_per_row), [=](sycl::id<2> idx) {
        int row = idx[0];
        int block_in_row = idx[1];
        int global_block = row * blocks_per_row + block_in_row;

        const uint8_t* qs = qs_base + global_block * (QK4_0/2);
        float d = (float)d_base[global_block];

        int base_idx = row * ncols + block_in_row * QK4_0;
        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >> 4) - 8;
            y[base_idx + j] = x0 * d;
            y[base_idx + j + QK4_0/2] = x1 * d;
        }
    }).wait();
}

// Test 1: Verify GPU reorder produces same output as CPU reference
bool test_gpu_reorder() {
    printf("Test 1: GPU reorder correctness\n");

    sycl::queue q{sycl::gpu_selector_v};
    printf("  Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int nrows = 64;
    const int ncols = 2048;
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t size = nblocks * sizeof(block_q4_0);

    // Create test data on host
    std::vector<block_q4_0> aos_host(nblocks);
    create_test_data(aos_host.data(), nrows, ncols, 42);

    // Allocate and copy to device
    uint8_t* data_device = sycl::malloc_device<uint8_t>(size, q);
    q.memcpy(data_device, aos_host.data(), size).wait();

    // Reorder on GPU
    reorder_q4_0_gpu(q, data_device, ncols, nrows);

    // Copy back
    std::vector<uint8_t> soa_host(size);
    q.memcpy(soa_host.data(), data_device, size).wait();

    // Verify SoA layout
    const size_t d_offset = (size_t)nrows * ncols / 2;
    const uint8_t* qs_ptr = soa_host.data();
    const sycl::half* d_ptr = (const sycl::half*)(soa_host.data() + d_offset);

    int errors = 0;
    for (int i = 0; i < nblocks && errors < 5; i++) {
        // Check qs
        for (int j = 0; j < QK4_0/2; j++) {
            if (qs_ptr[i * (QK4_0/2) + j] != aos_host[i].qs[j]) {
                printf("  ERROR: block %d, qs[%d]: GPU=%02x, expected=%02x\n",
                       i, j, qs_ptr[i * (QK4_0/2) + j], aos_host[i].qs[j]);
                errors++;
            }
        }
        // Check d (compare as bits)
        uint16_t d_gpu, d_ref;
        memcpy(&d_gpu, &d_ptr[i], 2);
        memcpy(&d_ref, &aos_host[i].d, 2);
        if (d_gpu != d_ref) {
            printf("  ERROR: block %d, d: GPU=%04x, expected=%04x\n", i, d_gpu, d_ref);
            errors++;
        }
    }

    sycl::free(data_device, q);

    if (errors > 0) {
        printf("  FAIL: GPU reorder has %d errors\n", errors);
        return false;
    }

    printf("  PASS: GPU reorder verified for %d blocks\n", nblocks);
    return true;
}

// Test 2: Compare AoS vs SoA dequantization on GPU
bool test_aos_vs_soa_dequant() {
    printf("Test 2: AoS vs SoA dequantization on GPU\n");

    sycl::queue q{sycl::gpu_selector_v};

    const int nrows = 64;
    const int ncols = 2048;
    const int nblocks = nrows * (ncols / QK4_0);
    const int nelements = nrows * ncols;
    const size_t block_size = nblocks * sizeof(block_q4_0);

    // Create test data
    std::vector<block_q4_0> aos_host(nblocks);
    create_test_data(aos_host.data(), nrows, ncols, 123);

    // CPU reference dequantization
    std::vector<float> cpu_ref(nelements);
    dequantize_q4_0_cpu(aos_host.data(), cpu_ref.data(), nblocks);

    // Allocate device memory
    block_q4_0* aos_device = sycl::malloc_device<block_q4_0>(nblocks, q);
    uint8_t* soa_device = sycl::malloc_device<uint8_t>(block_size, q);
    float* aos_output = sycl::malloc_device<float>(nelements, q);
    float* soa_output = sycl::malloc_device<float>(nelements, q);

    // Copy AoS data to device
    q.memcpy(aos_device, aos_host.data(), block_size).wait();

    // Also copy to SoA buffer and reorder
    q.memcpy(soa_device, aos_host.data(), block_size).wait();
    reorder_q4_0_gpu(q, soa_device, ncols, nrows);

    // Dequantize from both layouts
    dequantize_q4_0_gpu_aos(q, aos_device, aos_output, nblocks);
    dequantize_q4_0_gpu_soa(q, soa_device, soa_output, nrows, ncols);

    // Copy results back
    std::vector<float> aos_result(nelements);
    std::vector<float> soa_result(nelements);
    q.memcpy(aos_result.data(), aos_output, nelements * sizeof(float)).wait();
    q.memcpy(soa_result.data(), soa_output, nelements * sizeof(float)).wait();

    // Compare
    int aos_vs_cpu_errors = 0;
    int soa_vs_cpu_errors = 0;
    int aos_vs_soa_errors = 0;

    for (int i = 0; i < nelements; i++) {
        if (fabsf(aos_result[i] - cpu_ref[i]) > 1e-5f) {
            if (aos_vs_cpu_errors < 3) {
                printf("  AoS vs CPU error at [%d]: AoS=%.6f, CPU=%.6f\n",
                       i, aos_result[i], cpu_ref[i]);
            }
            aos_vs_cpu_errors++;
        }
        if (fabsf(soa_result[i] - cpu_ref[i]) > 1e-5f) {
            if (soa_vs_cpu_errors < 3) {
                printf("  SoA vs CPU error at [%d]: SoA=%.6f, CPU=%.6f\n",
                       i, soa_result[i], cpu_ref[i]);
            }
            soa_vs_cpu_errors++;
        }
        if (fabsf(aos_result[i] - soa_result[i]) > 1e-5f) {
            if (aos_vs_soa_errors < 3) {
                printf("  AoS vs SoA error at [%d]: AoS=%.6f, SoA=%.6f\n",
                       i, aos_result[i], soa_result[i]);
            }
            aos_vs_soa_errors++;
        }
    }

    sycl::free(aos_device, q);
    sycl::free(soa_device, q);
    sycl::free(aos_output, q);
    sycl::free(soa_output, q);

    bool pass = true;
    if (aos_vs_cpu_errors > 0) {
        printf("  FAIL: AoS vs CPU: %d errors\n", aos_vs_cpu_errors);
        pass = false;
    }
    if (soa_vs_cpu_errors > 0) {
        printf("  FAIL: SoA vs CPU: %d errors\n", soa_vs_cpu_errors);
        pass = false;
    }
    if (aos_vs_soa_errors > 0) {
        printf("  FAIL: AoS vs SoA: %d errors\n", aos_vs_soa_errors);
        pass = false;
    }

    if (pass) {
        printf("  PASS: All dequantization outputs match\n");
    }
    return pass;
}

// Test 3: Verify d_offset in reordered data
bool test_d_offset_location() {
    printf("Test 3: d_offset location verification\n");

    sycl::queue q{sycl::gpu_selector_v};

    const int nrows = 128;
    const int ncols = 4096;
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t size = nblocks * sizeof(block_q4_0);

    // Create data with unique scale values
    std::vector<block_q4_0> aos_host(nblocks);
    for (int i = 0; i < nblocks; i++) {
        // Encode block index in scale: 1.0 + (i/1000) to keep it in reasonable fp16 range
        aos_host[i].d = sycl::half(1.0f + (float)(i % 1000) / 10000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            aos_host[i].qs[j] = (uint8_t)(i + j);
        }
    }

    // Allocate and copy to device
    uint8_t* data_device = sycl::malloc_device<uint8_t>(size, q);
    q.memcpy(data_device, aos_host.data(), size).wait();

    // Reorder
    reorder_q4_0_gpu(q, data_device, ncols, nrows);

    // Copy back
    std::vector<uint8_t> soa_host(size);
    q.memcpy(soa_host.data(), data_device, size).wait();

    // Verify d values at expected offset
    const size_t d_offset = (size_t)nrows * ncols / 2;
    const sycl::half* d_ptr = (const sycl::half*)(soa_host.data() + d_offset);

    printf("  d_offset = %zu bytes (expected: %d blocks * %d qs_bytes = %d)\n",
           d_offset, nblocks, QK4_0/2, nblocks * (QK4_0/2));

    int errors = 0;
    for (int i = 0; i < nblocks && errors < 5; i++) {
        float d_soa = (float)d_ptr[i];
        float d_aos = (float)aos_host[i].d;
        if (fabsf(d_soa - d_aos) > 1e-4f) {
            printf("  ERROR: block %d: SoA d=%.6f, AoS d=%.6f\n", i, d_soa, d_aos);
            errors++;
        }
    }

    sycl::free(data_device, q);

    if (errors > 0) {
        printf("  FAIL: d values misplaced\n");
        return false;
    }

    printf("  PASS: All %d d values at correct offset\n", nblocks);
    return true;
}

int main() {
    printf("=== Q4_0 MMQ SoA GPU Tests ===\n\n");

    try {
        int passed = 0;
        int failed = 0;

        if (test_gpu_reorder()) passed++; else failed++;
        printf("\n");

        if (test_aos_vs_soa_dequant()) passed++; else failed++;
        printf("\n");

        if (test_d_offset_location()) passed++; else failed++;
        printf("\n");

        printf("=================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
}
