// Comprehensive unit test for Q6_K DMMV SoA kernel
// Tests both AoS and SoA layouts and verifies they produce matching results
// This exercises the PRODUCTION dequantize_mul_mat_vec_q6_k kernels from dmmv.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#include <sycl/sycl.hpp>

// Forward declare production DMMV kernel dispatch functions from dmmv.cpp
// dpct::queue_ptr is sycl::queue* - use that directly to avoid header dependencies
void dequantize_mul_mat_vec_q6_K_sycl(const void *vx, const float *y,
                                      float *dst, const int ncols,
                                      const int nrows,
                                      sycl::queue* stream);

void dequantize_mul_mat_vec_q6_K_sycl_soa(const void *vx, const float *y,
                                          float *dst, const int ncols,
                                          const int nrows, const int64_t ne01,
                                          const int row_low,
                                          sycl::queue* stream);

// Q6_K block structure (must match ggml)
#define QK_K 256
#define K_QUANTS_PER_ITERATION 2
#define QK_WARP_SIZE 32

// AoS block structure (original layout)
typedef struct {
    uint8_t ql[QK_K/2];     // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];     // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits (16 bytes)
    sycl::half d;           // super-block scale (2 bytes)
} block_q6_K;

static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");

// ========== SoA LAYOUT CONSTANTS (matching quants.hpp) ==========
// SoA layout: [all ql] [all qh] [all scales] [all d]
// ql: 128 bytes per block (QK_K/2)
// qh: 64 bytes per block (QK_K/4)
// scales: 16 bytes per block (QK_K/16)
// d: 2 bytes per block (sizeof(sycl::half))

constexpr int64_t SOA_QL_SIZE_PER_BLOCK = QK_K / 2;     // 128 bytes
constexpr int64_t SOA_QH_SIZE_PER_BLOCK = QK_K / 4;     // 64 bytes
constexpr int64_t SOA_SCALES_SIZE_PER_BLOCK = QK_K / 16; // 16 bytes
constexpr int64_t SOA_D_SIZE_PER_BLOCK = 2;             // sizeof(sycl::half)
constexpr int64_t SOA_TOTAL_SIZE_PER_BLOCK = SOA_QL_SIZE_PER_BLOCK + SOA_QH_SIZE_PER_BLOCK +
                                              SOA_SCALES_SIZE_PER_BLOCK + SOA_D_SIZE_PER_BLOCK; // 210 bytes

// ========== CPU REFERENCE IMPLEMENTATION ==========

float vec_dot_q6_K_cpu(const block_q6_K * x, const float * y, int ncols) {
    const int nb = ncols / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = static_cast<float>(x[i].d);
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                const float * yp = y + i * QK_K + n;
                sumf += d * sc[is + 0] * q1 * yp[l +  0];
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

// ========== AoS TO SoA CONVERSION ==========
// Matches the production reorder function behavior

void convert_aos_to_soa(const block_q6_K * aos_data, void * soa_data,
                         int nrows, int ncols) {
    const int nb_per_row = ncols / QK_K;
    const int64_t nblocks = (int64_t)nrows * nb_per_row;

    // Calculate SoA offsets
    const int64_t ql_offset = 0;
    const int64_t qh_offset = nblocks * SOA_QL_SIZE_PER_BLOCK;
    const int64_t scales_offset = qh_offset + nblocks * SOA_QH_SIZE_PER_BLOCK;
    const int64_t d_offset = scales_offset + nblocks * SOA_SCALES_SIZE_PER_BLOCK;

    uint8_t * soa = (uint8_t *)soa_data;

    for (int row = 0; row < nrows; row++) {
        for (int ib = 0; ib < nb_per_row; ib++) {
            const int block_idx = row * nb_per_row + ib;
            const block_q6_K * src = &aos_data[block_idx];

            // Copy ql (128 bytes)
            memcpy(soa + ql_offset + block_idx * SOA_QL_SIZE_PER_BLOCK,
                   src->ql, SOA_QL_SIZE_PER_BLOCK);

            // Copy qh (64 bytes)
            memcpy(soa + qh_offset + block_idx * SOA_QH_SIZE_PER_BLOCK,
                   src->qh, SOA_QH_SIZE_PER_BLOCK);

            // Copy scales (16 bytes)
            memcpy(soa + scales_offset + block_idx * SOA_SCALES_SIZE_PER_BLOCK,
                   src->scales, SOA_SCALES_SIZE_PER_BLOCK);

            // Copy d (2 bytes)
            memcpy(soa + d_offset + block_idx * SOA_D_SIZE_PER_BLOCK,
                   &src->d, SOA_D_SIZE_PER_BLOCK);
        }
    }
}

// ========== GPU KERNELS: Using PRODUCTION code from dmmv.cpp ==========
// The production dispatch functions are:
// - dequantize_mul_mat_vec_q6_K_sycl() for AoS layout
// - dequantize_mul_mat_vec_q6_K_sycl_soa() for SoA layout
// These are declared in dmmv.hpp and implemented in dmmv.cpp

// ========== QUANTIZATION HELPER ==========

void quantize_row_q6_K(const float * x, block_q6_K * y, int64_t k) {
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float max_abs = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = fabsf(x[i * QK_K + j]);
            if (ax > max_abs) max_abs = ax;
        }

        const float d = max_abs > 0 ? max_abs / 31.0f : 1.0f;
        const float id = 1.0f / d;

        y[i].d = sycl::half(d);

        for (int j = 0; j < 16; j++) {
            y[i].scales[j] = 1;
        }

        for (int n = 0; n < QK_K; n += 128) {
            const float * xp = x + i * QK_K + n;
            uint8_t * ql = y[i].ql + n/2;
            uint8_t * qh = y[i].qh + n/4;

            for (int l = 0; l < 32; ++l) {
                int8_t q1 = (int8_t)roundf(xp[l +  0] * id) + 32;
                int8_t q2 = (int8_t)roundf(xp[l + 32] * id) + 32;
                int8_t q3 = (int8_t)roundf(xp[l + 64] * id) + 32;
                int8_t q4 = (int8_t)roundf(xp[l + 96] * id) + 32;

                q1 = q1 < 0 ? 0 : (q1 > 63 ? 63 : q1);
                q2 = q2 < 0 ? 0 : (q2 > 63 ? 63 : q2);
                q3 = q3 < 0 ? 0 : (q3 > 63 ? 63 : q3);
                q4 = q4 < 0 ? 0 : (q4 > 63 ? 63 : q4);

                ql[l +  0] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);

                qh[l] = ((q1 >> 4) & 0x03) |
                       (((q2 >> 4) & 0x03) << 2) |
                       (((q3 >> 4) & 0x03) << 4) |
                       (((q4 >> 4) & 0x03) << 6);
            }
            ql += 64;
            qh += 32;
        }
    }
}

// ========== TEST RUNNER ==========

struct TestCase {
    int ncols;
    int nrows;
    int ne01;      // ADDED: Full tensor rows (for SoA layout calculation)
    int row_low;   // ADDED: Start row in global layout (for split tensors)
    const char * name;
};

bool run_test(sycl::queue & q, const TestCase & tc, int seed, bool verbose = false) {
    const int ncols = tc.ncols;
    const int nrows = tc.nrows;        // Number of rows in this slice
    const int ne01 = tc.ne01;          // Full tensor rows (for SoA layout)
    const int row_low = tc.row_low;    // Start row in global layout
    const int nb_per_row = ncols / QK_K;

    // Total blocks for the FULL tensor (used for SoA offset calculation)
    const int64_t nblocks_full = (int64_t)ne01 * nb_per_row;
    // Blocks for this slice only (used for AoS testing)
    const int64_t nblocks_slice = (int64_t)nrows * nb_per_row;

    if (verbose) {
        printf("\n=== Test: %s ===\n", tc.name);
        printf("ncols=%d, nrows=%d, ne01=%d, row_low=%d\n", ncols, nrows, ne01, row_low);
        printf("nblocks_full=%lld, nblocks_slice=%lld\n",
               (long long)nblocks_full, (long long)nblocks_slice);
    }

    // Allocate host data for the FULL tensor (needed for correct SoA layout)
    std::vector<float> x_f32_full(ncols * ne01);
    std::vector<float> y_f32(ncols);
    std::vector<block_q6_K> x_aos_full(nblocks_full);
    std::vector<uint8_t> x_soa_full(nblocks_full * SOA_TOTAL_SIZE_PER_BLOCK);
    std::vector<float> dst_cpu(nrows);
    std::vector<float> dst_gpu_aos(nrows);
    std::vector<float> dst_gpu_soa(nrows);

    // Initialize with reproducible random data for FULL tensor
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < ncols * ne01; i++) {
        x_f32_full[i] = dist(rng);
    }
    for (int i = 0; i < ncols; i++) {
        y_f32[i] = dist(rng);
    }

    // Quantize FULL tensor to Q6_K (AoS format)
    quantize_row_q6_K(x_f32_full.data(), x_aos_full.data(), ncols * ne01);

    // Convert FULL AoS to SoA
    convert_aos_to_soa(x_aos_full.data(), x_soa_full.data(), ne01, ncols);

    // CPU reference: compute for this slice only (rows [row_low, row_low+nrows))
    for (int row = 0; row < nrows; row++) {
        int global_row = row_low + row;
        dst_cpu[row] = vec_dot_q6_K_cpu(x_aos_full.data() + global_row * nb_per_row, y_f32.data(), ncols);
    }

    // Calculate SoA offsets using FULL tensor dimensions (matching production!)
    const int64_t qh_offset = nblocks_full * SOA_QL_SIZE_PER_BLOCK;
    const int64_t scales_offset = qh_offset + nblocks_full * SOA_QH_SIZE_PER_BLOCK;
    const int64_t d_offset = scales_offset + nblocks_full * SOA_SCALES_SIZE_PER_BLOCK;

    if (verbose) {
        printf("SoA offsets (based on ne01=%d): qh=%lld, scales=%lld, d=%lld\n",
               ne01, (long long)qh_offset, (long long)scales_offset, (long long)d_offset);
    }

    // Allocate device memory
    // For AoS: only need the slice
    block_q6_K * d_aos = sycl::malloc_device<block_q6_K>(nblocks_slice, q);
    // For SoA: need the FULL tensor (offsets are global)
    void * d_soa = sycl::malloc_device<uint8_t>(nblocks_full * SOA_TOTAL_SIZE_PER_BLOCK, q);
    float * d_y = sycl::malloc_device<float>(ncols, q);
    float * d_dst = sycl::malloc_device<float>(nrows, q);

    // Copy data to device
    // AoS: copy only the slice
    q.memcpy(d_aos, x_aos_full.data() + row_low * nb_per_row,
             nblocks_slice * sizeof(block_q6_K)).wait();
    // SoA: copy the FULL tensor (offsets require global access)
    q.memcpy(d_soa, x_soa_full.data(),
             nblocks_full * SOA_TOTAL_SIZE_PER_BLOCK).wait();
    q.memcpy(d_y, y_f32.data(), ncols * sizeof(float)).wait();

    // ========== TEST AoS KERNEL (using production dispatch function) ==========
    q.memset(d_dst, 0, nrows * sizeof(float)).wait();

    // Call production AoS kernel dispatch function from dmmv.cpp
    dequantize_mul_mat_vec_q6_K_sycl(d_aos, d_y, d_dst, ncols, nrows, &q);
    q.wait();

    q.memcpy(dst_gpu_aos.data(), d_dst, nrows * sizeof(float)).wait();

    // ========== TEST SoA KERNEL (using production dispatch function) ==========
    q.memset(d_dst, 0, nrows * sizeof(float)).wait();

    // Call production SoA kernel dispatch function from dmmv.cpp
    dequantize_mul_mat_vec_q6_K_sycl_soa(d_soa, d_y, d_dst, ncols, nrows, ne01, row_low, &q);
    q.wait();

    q.memcpy(dst_gpu_soa.data(), d_dst, nrows * sizeof(float)).wait();

    // Verify results
    bool all_pass = true;
    float max_diff_aos = 0.0f;
    float max_diff_soa = 0.0f;
    float max_diff_aos_soa = 0.0f;

    for (int row = 0; row < nrows; row++) {
        float diff_aos = fabsf(dst_cpu[row] - dst_gpu_aos[row]);
        float diff_soa = fabsf(dst_cpu[row] - dst_gpu_soa[row]);
        float diff_aos_soa = fabsf(dst_gpu_aos[row] - dst_gpu_soa[row]);

        float rel_aos = diff_aos / (fabsf(dst_cpu[row]) + 1e-10f);
        float rel_soa = diff_soa / (fabsf(dst_cpu[row]) + 1e-10f);
        float rel_aos_soa = diff_aos_soa / (fabsf(dst_gpu_aos[row]) + 1e-10f);

        max_diff_aos = std::max(max_diff_aos, rel_aos);
        max_diff_soa = std::max(max_diff_soa, rel_soa);
        max_diff_aos_soa = std::max(max_diff_aos_soa, rel_aos_soa);

        const float threshold = 0.001f; // 0.1%

        if (rel_aos >= threshold || rel_soa >= threshold) {
            all_pass = false;
            if (verbose || row < 3) {
                printf("Row %d: CPU=%.6f AoS=%.6f SoA=%.6f | diff_aos=%.6f%% diff_soa=%.6f%% %s\n",
                       row, dst_cpu[row], dst_gpu_aos[row], dst_gpu_soa[row],
                       rel_aos * 100.0f, rel_soa * 100.0f,
                       (rel_aos < threshold && rel_soa < threshold) ? "OK" : "FAIL");
            }
        }
    }

    if (verbose) {
        printf("Max relative error: AoS=%.6f%% SoA=%.6f%% AoS-SoA=%.6f%%\n",
               max_diff_aos * 100.0f, max_diff_soa * 100.0f, max_diff_aos_soa * 100.0f);
    }

    printf("%s: %s (max_err: AoS=%.4f%% SoA=%.4f%% delta=%.4f%%)\n",
           tc.name, all_pass ? "PASS" : "FAIL",
           max_diff_aos * 100.0f, max_diff_soa * 100.0f, max_diff_aos_soa * 100.0f);

    // Cleanup
    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_y, q);
    sycl::free(d_dst, q);

    return all_pass;
}

// ========== OFFSET VERIFICATION TEST ==========

bool test_soa_offsets() {
    printf("\n=== SoA Offset Calculation Test ===\n");

    // Test with known values
    const int ncols = 4096;
    const int nrows = 4;
    const int nb_per_row = ncols / QK_K;  // 16
    const int64_t nblocks = nrows * nb_per_row;  // 64

    // Expected offsets based on layout
    const int64_t expected_ql_offset = 0;
    const int64_t expected_qh_offset = nblocks * 128;  // 64 * 128 = 8192
    const int64_t expected_scales_offset = expected_qh_offset + nblocks * 64;  // 8192 + 4096 = 12288
    const int64_t expected_d_offset = expected_scales_offset + nblocks * 16;  // 12288 + 1024 = 13312
    const int64_t expected_total_size = expected_d_offset + nblocks * 2;  // 13312 + 128 = 13440

    // Calculated offsets
    const int64_t qh_offset = nblocks * SOA_QL_SIZE_PER_BLOCK;
    const int64_t scales_offset = qh_offset + nblocks * SOA_QH_SIZE_PER_BLOCK;
    const int64_t d_offset = scales_offset + nblocks * SOA_SCALES_SIZE_PER_BLOCK;
    const int64_t total_size = nblocks * SOA_TOTAL_SIZE_PER_BLOCK;

    printf("Config: ncols=%d, nrows=%d, nblocks=%lld\n", ncols, nrows, (long long)nblocks);
    printf("Expected: ql=0, qh=%lld, scales=%lld, d=%lld, total=%lld\n",
           (long long)expected_qh_offset, (long long)expected_scales_offset,
           (long long)expected_d_offset, (long long)expected_total_size);
    printf("Calculated: ql=0, qh=%lld, scales=%lld, d=%lld, total=%lld\n",
           (long long)qh_offset, (long long)scales_offset,
           (long long)d_offset, (long long)total_size);

    bool pass = (qh_offset == expected_qh_offset) &&
                (scales_offset == expected_scales_offset) &&
                (d_offset == expected_d_offset) &&
                (total_size == expected_total_size);

    printf("Offset test: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ========== DATA LAYOUT VERIFICATION TEST ==========

bool test_aos_to_soa_conversion() {
    printf("\n=== AoS to SoA Conversion Test ===\n");

    const int ncols = 256;  // Single block per row
    const int nrows = 2;
    const int nb_per_row = ncols / QK_K;  // 1
    const int64_t nblocks = nrows * nb_per_row;  // 2

    // Create known AoS data
    std::vector<block_q6_K> aos(nblocks);

    // Fill with recognizable patterns
    for (int b = 0; b < nblocks; b++) {
        for (int i = 0; i < QK_K/2; i++) {
            aos[b].ql[i] = (uint8_t)((b * 10 + i) & 0xFF);
        }
        for (int i = 0; i < QK_K/4; i++) {
            aos[b].qh[i] = (uint8_t)((b * 20 + i) & 0xFF);
        }
        for (int i = 0; i < QK_K/16; i++) {
            aos[b].scales[i] = (int8_t)((b * 5 + i) & 0x7F);
        }
        aos[b].d = sycl::half(1.0f + b * 0.1f);
    }

    // Convert to SoA
    std::vector<uint8_t> soa(nblocks * SOA_TOTAL_SIZE_PER_BLOCK);
    convert_aos_to_soa(aos.data(), soa.data(), nrows, ncols);

    // Calculate offsets
    const int64_t qh_offset = nblocks * SOA_QL_SIZE_PER_BLOCK;
    const int64_t scales_offset = qh_offset + nblocks * SOA_QH_SIZE_PER_BLOCK;
    const int64_t d_offset = scales_offset + nblocks * SOA_SCALES_SIZE_PER_BLOCK;

    // Verify data was placed correctly
    bool pass = true;

    for (int b = 0; b < nblocks; b++) {
        // Check ql
        for (int i = 0; i < SOA_QL_SIZE_PER_BLOCK && pass; i++) {
            uint8_t expected = aos[b].ql[i];
            uint8_t actual = soa[b * SOA_QL_SIZE_PER_BLOCK + i];
            if (expected != actual) {
                printf("FAIL: Block %d ql[%d]: expected %u, got %u\n", b, i, expected, actual);
                pass = false;
            }
        }

        // Check qh
        for (int i = 0; i < SOA_QH_SIZE_PER_BLOCK && pass; i++) {
            uint8_t expected = aos[b].qh[i];
            uint8_t actual = soa[qh_offset + b * SOA_QH_SIZE_PER_BLOCK + i];
            if (expected != actual) {
                printf("FAIL: Block %d qh[%d]: expected %u, got %u\n", b, i, expected, actual);
                pass = false;
            }
        }

        // Check scales
        for (int i = 0; i < SOA_SCALES_SIZE_PER_BLOCK && pass; i++) {
            int8_t expected = aos[b].scales[i];
            int8_t actual = ((int8_t*)&soa[scales_offset])[b * SOA_SCALES_SIZE_PER_BLOCK + i];
            if (expected != actual) {
                printf("FAIL: Block %d scales[%d]: expected %d, got %d\n", b, i, expected, actual);
                pass = false;
            }
        }

        // Check d
        sycl::half expected_d = aos[b].d;
        sycl::half actual_d = *((sycl::half*)&soa[d_offset + b * sizeof(sycl::half)]);
        if (expected_d != actual_d) {
            printf("FAIL: Block %d d: expected %f, got %f\n", b,
                   static_cast<float>(expected_d), static_cast<float>(actual_d));
            pass = false;
        }
    }

    printf("Conversion test: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char ** argv) {
    (void)argc; (void)argv;

    printf("=== Q6_K DMMV SoA Kernel Comprehensive Test ===\n\n");
    printf("Constants: QK_K=%d, K_QUANTS_PER_ITERATION=%d, QK_WARP_SIZE=%d\n",
           QK_K, K_QUANTS_PER_ITERATION, QK_WARP_SIZE);
    printf("Block size: AoS=%zu bytes, SoA total=%lld bytes/block\n\n",
           sizeof(block_q6_K), (long long)SOA_TOTAL_SIZE_PER_BLOCK);

    // Run offset and conversion tests first
    bool offset_pass = test_soa_offsets();
    bool conv_pass = test_aos_to_soa_conversion();

    // Initialize SYCL
    printf("\n=== GPU Kernel Tests ===\n");
    sycl::queue q(sycl::default_selector_v);
    printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Define test cases: {ncols, nrows, ne01, row_low, name}
    // ne01 = full tensor rows, row_low = start row in global layout
    std::vector<TestCase> tests = {
        // Basic tests (no split: ne01=nrows, row_low=0)
        {256, 1, 1, 0, "Single block, single row"},
        {256, 8, 8, 0, "Single block, 8 rows"},
        {4096, 1, 1, 0, "Mistral hidden, single row (DMMV decode)"},
        {4096, 8, 8, 0, "Mistral hidden, 8 rows"},

        // Production-like tests (no split)
        {4096, 32, 32, 0, "Mistral hidden, 32 rows"},
        {4096, 128, 128, 0, "Mistral hidden, 128 rows"},

        // Edge cases (no split)
        {512, 1, 1, 0, "2 blocks, single row"},
        {512, 16, 16, 0, "2 blocks, 16 rows"},
        {1024, 1, 1, 0, "4 blocks, single row"},

        // Larger tests (no split)
        {8192, 1, 1, 0, "8K hidden, single row"},
        {8192, 8, 8, 0, "8K hidden, 8 rows"},

        // ========== SPLIT TENSOR TESTS (row_low > 0) ==========
        // These simulate multi-GPU or CPU/GPU tensor splits

        // Split: middle slice of 128-row tensor
        {4096, 32, 128, 32, "Split: rows 32-63 of 128 (middle)"},
        {4096, 32, 128, 64, "Split: rows 64-95 of 128"},
        {4096, 32, 128, 96, "Split: rows 96-127 of 128 (last)"},

        // Split: single row from different positions
        {4096, 1, 32, 0, "Split: row 0 of 32"},
        {4096, 1, 32, 15, "Split: row 15 of 32 (middle)"},
        {4096, 1, 32, 31, "Split: row 31 of 32 (last)"},

        // Split: 8 rows from 64-row tensor
        {4096, 8, 64, 0, "Split: rows 0-7 of 64 (first)"},
        {4096, 8, 64, 28, "Split: rows 28-35 of 64 (middle)"},
        {4096, 8, 64, 56, "Split: rows 56-63 of 64 (last)"},

        // Split with multiple blocks per row
        {512, 4, 16, 8, "Split: rows 8-11 of 16 (2 blocks/row)"},
        {1024, 4, 32, 16, "Split: rows 16-19 of 32 (4 blocks/row)"},
    };

    int pass_count = 0;
    int fail_count = 0;

    for (const auto & tc : tests) {
        bool pass = run_test(q, tc, 42, false);
        if (pass) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    // Run one test with verbose output for debugging
    printf("\n=== Verbose Test (for debugging) ===\n");
    run_test(q, {4096, 4, 4, 0, "Verbose: 4096x4 (no split)"}, 42, true);

    // Also run a split tensor test with verbose output
    printf("\n=== Verbose Split Test (for debugging) ===\n");
    run_test(q, {4096, 4, 16, 8, "Verbose: split rows 8-11 of 16"}, 42, true);

    // Summary
    printf("\n=== SUMMARY ===\n");
    printf("Offset test: %s\n", offset_pass ? "PASS" : "FAIL");
    printf("Conversion test: %s\n", conv_pass ? "PASS" : "FAIL");
    printf("GPU tests: %d passed, %d failed\n", pass_count, fail_count);

    bool overall = offset_pass && conv_pass && (fail_count == 0);
    printf("\n=== OVERALL: %s ===\n", overall ? "PASS" : "FAIL");

    return overall ? 0 : 1;
}
