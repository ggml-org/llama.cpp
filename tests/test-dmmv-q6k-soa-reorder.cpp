// DMMV Q6_K SoA Reorder Verification Test
// Tests that the SoA layout reordering is correct by verifying kernel reads match expected values
// This is similar to the MMQ vecdot test - verifies intermediate memory access, not just final output
//
// The test:
// 1. Creates Q6_K blocks with known patterns (each byte has unique traceable value)
// 2. Converts AoS to SoA layout
// 3. Simulates what the kernel reads from SoA at each position
// 4. Compares kernel reads to original AoS values

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <sycl/sycl.hpp>

// Q6_K block constants
#define QK_K 256
#define K_QUANTS_PER_ITERATION 2
#define QK_WARP_SIZE 32

// AoS block structure
typedef struct {
    uint8_t ql[QK_K/2];     // 128 bytes
    uint8_t qh[QK_K/4];     // 64 bytes
    int8_t  scales[QK_K/16]; // 16 bytes
    sycl::half d;           // 2 bytes
} block_q6_K;

static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");

// SoA layout sizes
constexpr int64_t SOA_QL_SIZE = QK_K / 2;     // 128 bytes/block
constexpr int64_t SOA_QH_SIZE = QK_K / 4;     // 64 bytes/block
constexpr int64_t SOA_SCALES_SIZE = QK_K / 16; // 16 bytes/block
constexpr int64_t SOA_D_SIZE = sizeof(sycl::half);  // 2 bytes/block

// Convert AoS to SoA (matching production reorder_q6_k_cpu)
void convert_aos_to_soa(const block_q6_K * aos, void * soa, int nrows, int ncols) {
    const int nb_per_row = ncols / QK_K;
    const int64_t nblocks = (int64_t)nrows * nb_per_row;

    uint8_t * dst = (uint8_t *)soa;
    uint8_t * soa_ql = dst;
    uint8_t * soa_qh = dst + nblocks * SOA_QL_SIZE;
    uint8_t * soa_scales = soa_qh + nblocks * SOA_QH_SIZE;
    uint8_t * soa_d = soa_scales + nblocks * SOA_SCALES_SIZE;

    for (int row = 0; row < nrows; row++) {
        for (int ib = 0; ib < nb_per_row; ib++) {
            int block_idx = row * nb_per_row + ib;
            const block_q6_K * src = &aos[block_idx];

            memcpy(soa_ql + block_idx * SOA_QL_SIZE, src->ql, SOA_QL_SIZE);
            memcpy(soa_qh + block_idx * SOA_QH_SIZE, src->qh, SOA_QH_SIZE);
            memcpy(soa_scales + block_idx * SOA_SCALES_SIZE, src->scales, SOA_SCALES_SIZE);
            memcpy(soa_d + block_idx * SOA_D_SIZE, &src->d, SOA_D_SIZE);
        }
    }
}

// Structure to capture what the kernel reads at each position
struct KernelReadValues {
    uint8_t ql[8];    // Up to 8 ql bytes per thread per block
    uint8_t qh[4];    // Up to 4 qh bytes per thread per block
    int8_t scales[8]; // Up to 8 scale bytes per thread per block
    float d;          // d value for this block
    int ql_count;
    int qh_count;
    int scales_count;
};

// Simulate DMMV SoA kernel memory access patterns
// This reproduces the exact indexing from dequantize_mul_mat_vec_q6_k_soa
void simulate_kernel_reads_soa(
    const void * vx,
    int ncols,
    int nrows,
    int ne01,        // Full tensor rows
    int row_low,     // Start row in global layout
    int64_t qh_offset,
    int64_t scales_offset,
    int64_t d_offset,
    int row,         // Row to simulate (local)
    int thread_id,   // Thread ID (0-31)
    std::vector<KernelReadValues> * per_block_reads)
{
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = (row_low + row) * num_blocks_per_row;  // Global block index base

    const uint8_t * ql_base = (const uint8_t *)vx;
    const uint8_t * qh_base = ql_base + qh_offset;
    const int8_t * scales_base = (const int8_t *)(ql_base + scales_offset);
    const sycl::half * d_base = (const sycl::half *)(ql_base + d_offset);

    // Thread indexing (matching kernel)
    const int tid = thread_id / K_QUANTS_PER_ITERATION;
    const int ix = thread_id % K_QUANTS_PER_ITERATION;

    const int step = 16 / K_QUANTS_PER_ITERATION;
    const int im = tid / step;
    const int in = tid - step * im;

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION * in;
    const int is = 0;
#else
    const int l0 = 4 * in;
    const int is = in / 4;
#endif
    const int ql_offset_local = 64 * im + l0;
    const int qh_offset_local = 32 * im + l0;
    const int s_offset = 8 * im + is;

    per_block_reads->clear();
    per_block_reads->resize(num_blocks_per_row);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int block_idx = ib0 + i;

        KernelReadValues & reads = (*per_block_reads)[i];
        reads.ql_count = 0;
        reads.qh_count = 0;
        reads.scales_count = 0;

        // Read d value
        reads.d = static_cast<float>(d_base[block_idx]);

        // Calculate ql, qh, scales addresses
        const uint8_t * ql = ql_base + block_idx * SOA_QL_SIZE + ql_offset_local;
        const uint8_t * qh = qh_base + block_idx * SOA_QH_SIZE + qh_offset_local;
        const int8_t * s = scales_base + block_idx * SOA_SCALES_SIZE + s_offset;

#if K_QUANTS_PER_ITERATION == 1
        // Capture what kernel reads
        reads.ql[reads.ql_count++] = ql[0];
        reads.ql[reads.ql_count++] = ql[16];
        reads.ql[reads.ql_count++] = ql[32];
        reads.ql[reads.ql_count++] = ql[48];
        reads.qh[reads.qh_count++] = qh[0];
        reads.qh[reads.qh_count++] = qh[16];
        for (int k = 0; k < 8; k++) {
            reads.scales[reads.scales_count++] = s[k];
        }
#else
        // K_QUANTS_PER_ITERATION == 2
        for (int l = 0; l < 4; l++) {
            reads.ql[reads.ql_count++] = ql[l + 0];
            reads.ql[reads.ql_count++] = ql[l + 32];
            reads.qh[reads.qh_count++] = qh[l];
        }
        // Scales access pattern for K_QUANTS_PER_ITERATION=2
        reads.scales[reads.scales_count++] = s[0];
        reads.scales[reads.scales_count++] = s[2];
        reads.scales[reads.scales_count++] = s[4];
        reads.scales[reads.scales_count++] = s[6];
#endif
    }
}

// Simulate AoS kernel reads for comparison
void simulate_kernel_reads_aos(
    const block_q6_K * x,
    int ncols,
    int row,
    int thread_id,
    std::vector<KernelReadValues> * per_block_reads)
{
    const int num_blocks_per_row = ncols / QK_K;
    const block_q6_K * row_data = x + row * num_blocks_per_row;

    const int tid = thread_id / K_QUANTS_PER_ITERATION;
    const int ix = thread_id % K_QUANTS_PER_ITERATION;

    const int step = 16 / K_QUANTS_PER_ITERATION;
    const int im = tid / step;
    const int in = tid - step * im;

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION * in;
    const int is = 0;
#else
    const int l0 = 4 * in;
    const int is = in / 4;
#endif
    const int ql_offset_local = 64 * im + l0;
    const int qh_offset_local = 32 * im + l0;
    const int s_offset = 8 * im + is;

    per_block_reads->clear();
    per_block_reads->resize(num_blocks_per_row);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        KernelReadValues & reads = (*per_block_reads)[i];
        reads.ql_count = 0;
        reads.qh_count = 0;
        reads.scales_count = 0;

        const block_q6_K * blk = &row_data[i];
        reads.d = static_cast<float>(blk->d);

        const uint8_t * ql = blk->ql + ql_offset_local;
        const uint8_t * qh = blk->qh + qh_offset_local;
        const int8_t * s = blk->scales + s_offset;

#if K_QUANTS_PER_ITERATION == 1
        reads.ql[reads.ql_count++] = ql[0];
        reads.ql[reads.ql_count++] = ql[16];
        reads.ql[reads.ql_count++] = ql[32];
        reads.ql[reads.ql_count++] = ql[48];
        reads.qh[reads.qh_count++] = qh[0];
        reads.qh[reads.qh_count++] = qh[16];
        for (int k = 0; k < 8; k++) {
            reads.scales[reads.scales_count++] = s[k];
        }
#else
        for (int l = 0; l < 4; l++) {
            reads.ql[reads.ql_count++] = ql[l + 0];
            reads.ql[reads.ql_count++] = ql[l + 32];
            reads.qh[reads.qh_count++] = qh[l];
        }
        reads.scales[reads.scales_count++] = s[0];
        reads.scales[reads.scales_count++] = s[2];
        reads.scales[reads.scales_count++] = s[4];
        reads.scales[reads.scales_count++] = s[6];
#endif
    }
}

// Compare reads from AoS and SoA
bool compare_reads(const std::vector<KernelReadValues> & aos_reads,
                   const std::vector<KernelReadValues> & soa_reads,
                   int row, int thread_id, bool verbose) {
    bool all_match = true;

    for (size_t i = 0; i < aos_reads.size(); i++) {
        const KernelReadValues & aos = aos_reads[i];
        const KernelReadValues & soa = soa_reads[i];

        // Skip blocks not processed by this thread
        if (aos.ql_count == 0 && soa.ql_count == 0) continue;

        // Compare ql values
        for (int j = 0; j < aos.ql_count && j < soa.ql_count; j++) {
            if (aos.ql[j] != soa.ql[j]) {
                if (verbose) {
                    printf("  MISMATCH row=%d thread=%d block=%zu ql[%d]: AoS=0x%02X SoA=0x%02X\n",
                           row, thread_id, i, j, aos.ql[j], soa.ql[j]);
                }
                all_match = false;
            }
        }

        // Compare qh values
        for (int j = 0; j < aos.qh_count && j < soa.qh_count; j++) {
            if (aos.qh[j] != soa.qh[j]) {
                if (verbose) {
                    printf("  MISMATCH row=%d thread=%d block=%zu qh[%d]: AoS=0x%02X SoA=0x%02X\n",
                           row, thread_id, i, j, aos.qh[j], soa.qh[j]);
                }
                all_match = false;
            }
        }

        // Compare scales
        for (int j = 0; j < aos.scales_count && j < soa.scales_count; j++) {
            if (aos.scales[j] != soa.scales[j]) {
                if (verbose) {
                    printf("  MISMATCH row=%d thread=%d block=%zu scales[%d]: AoS=%d SoA=%d\n",
                           row, thread_id, i, j, aos.scales[j], soa.scales[j]);
                }
                all_match = false;
            }
        }

        // Compare d value
        if (fabsf(aos.d - soa.d) > 1e-5f) {
            if (verbose) {
                printf("  MISMATCH row=%d thread=%d block=%zu d: AoS=%f SoA=%f\n",
                       row, thread_id, i, aos.d, soa.d);
            }
            all_match = false;
        }
    }

    return all_match;
}

// Test case structure
struct ReorderTestCase {
    int ncols;
    int nrows;
    int ne01;       // Full tensor rows
    int row_low;    // Start row in global layout
    const char * name;
};

bool run_reorder_test(const ReorderTestCase & tc, bool verbose) {
    const int ncols = tc.ncols;
    const int nrows = tc.nrows;
    const int ne01 = tc.ne01;
    const int row_low = tc.row_low;
    const int nb_per_row = ncols / QK_K;
    const int64_t nblocks_full = (int64_t)ne01 * nb_per_row;

    if (verbose) {
        printf("\n=== Test: %s ===\n", tc.name);
        printf("ncols=%d, nrows=%d, ne01=%d, row_low=%d\n", ncols, nrows, ne01, row_low);
    }

    // Create AoS data with unique traceable values
    // Pattern: each byte encodes (block_idx, component, offset) for debugging
    std::vector<block_q6_K> aos_full(nblocks_full);

    for (int64_t b = 0; b < nblocks_full; b++) {
        // ql: value = (b * 128 + offset) % 256
        for (int i = 0; i < SOA_QL_SIZE; i++) {
            aos_full[b].ql[i] = (uint8_t)((b * 7 + i) & 0xFF);
        }
        // qh: value = (b * 64 + offset) % 256
        for (int i = 0; i < SOA_QH_SIZE; i++) {
            aos_full[b].qh[i] = (uint8_t)((b * 13 + i + 128) & 0xFF);
        }
        // scales: value = (b + offset) % 128 - 64
        for (int i = 0; i < SOA_SCALES_SIZE; i++) {
            aos_full[b].scales[i] = (int8_t)(((b * 3 + i) % 128) - 64);
        }
        // d: unique per block
        aos_full[b].d = sycl::half(1.0f + b * 0.001f);
    }

    // Convert full tensor to SoA
    const int64_t soa_size = nblocks_full * (SOA_QL_SIZE + SOA_QH_SIZE + SOA_SCALES_SIZE + SOA_D_SIZE);
    std::vector<uint8_t> soa_full(soa_size);
    convert_aos_to_soa(aos_full.data(), soa_full.data(), ne01, ncols);

    // Calculate SoA offsets based on FULL tensor
    const int64_t qh_offset = nblocks_full * SOA_QL_SIZE;
    const int64_t scales_offset = qh_offset + nblocks_full * SOA_QH_SIZE;
    const int64_t d_offset = scales_offset + nblocks_full * SOA_SCALES_SIZE;

    // Test all thread IDs for a subset of rows
    bool all_pass = true;
    int total_checks = 0;
    int failed_checks = 0;

    // Test each row in the slice
    for (int row = 0; row < nrows; row++) {
        int global_row = row_low + row;

        // Test all 32 threads
        for (int tid = 0; tid < QK_WARP_SIZE; tid++) {
            std::vector<KernelReadValues> aos_reads, soa_reads;

            // AoS: kernel gets pointer to slice start, uses local row index
            simulate_kernel_reads_aos(
                aos_full.data() + row_low * nb_per_row,  // Pointer to slice start
                ncols,
                row,  // Local row within slice
                tid,
                &aos_reads);

            // SoA: kernel gets full tensor, uses global indexing via row_low
            simulate_kernel_reads_soa(
                soa_full.data(),
                ncols,
                nrows,
                ne01,
                row_low,
                qh_offset,
                scales_offset,
                d_offset,
                row,  // Local row within slice
                tid,
                &soa_reads);

            total_checks++;
            if (!compare_reads(aos_reads, soa_reads, row, tid, verbose && !all_pass)) {
                all_pass = false;
                failed_checks++;
                if (failed_checks == 1 && !verbose) {
                    // Re-run with verbose to show first mismatch
                    compare_reads(aos_reads, soa_reads, row, tid, true);
                }
            }
        }
    }

    if (verbose || !all_pass) {
        printf("%s: %s (%d/%d thread-row combinations matched)\n",
               tc.name, all_pass ? "PASS" : "FAIL",
               total_checks - failed_checks, total_checks);
    } else {
        printf("%s: PASS\n", tc.name);
    }

    return all_pass;
}

// Direct byte-by-byte verification of SoA layout
bool test_soa_layout_bytes() {
    printf("\n=== SoA Layout Byte-by-Byte Verification ===\n");

    const int ncols = 512;  // 2 blocks per row
    const int nrows = 3;
    const int nb_per_row = ncols / QK_K;
    const int64_t nblocks = nrows * nb_per_row;

    // Create AoS with known pattern
    std::vector<block_q6_K> aos(nblocks);
    for (int b = 0; b < nblocks; b++) {
        for (int i = 0; i < SOA_QL_SIZE; i++) {
            aos[b].ql[i] = (b << 4) | (i & 0xF);  // Block ID in upper nibble
        }
        for (int i = 0; i < SOA_QH_SIZE; i++) {
            aos[b].qh[i] = 0x80 | (b << 4) | (i & 0xF);
        }
        for (int i = 0; i < SOA_SCALES_SIZE; i++) {
            aos[b].scales[i] = (int8_t)(b * 10 + i);
        }
        aos[b].d = sycl::half(b + 1.0f);
    }

    // Convert to SoA
    const int64_t soa_size = nblocks * (SOA_QL_SIZE + SOA_QH_SIZE + SOA_SCALES_SIZE + SOA_D_SIZE);
    std::vector<uint8_t> soa(soa_size);
    convert_aos_to_soa(aos.data(), soa.data(), nrows, ncols);

    // Verify layout
    const int64_t qh_offset = nblocks * SOA_QL_SIZE;
    const int64_t scales_offset = qh_offset + nblocks * SOA_QH_SIZE;
    const int64_t d_offset = scales_offset + nblocks * SOA_SCALES_SIZE;

    bool pass = true;

    printf("Layout offsets: ql=0, qh=%lld, scales=%lld, d=%lld\n",
           (long long)qh_offset, (long long)scales_offset, (long long)d_offset);

    // Verify each block's data is at correct SoA location
    for (int b = 0; b < nblocks; b++) {
        // Check ql
        for (int i = 0; i < SOA_QL_SIZE; i++) {
            uint8_t expected = aos[b].ql[i];
            uint8_t actual = soa[b * SOA_QL_SIZE + i];
            if (expected != actual) {
                printf("FAIL: block %d ql[%d] expected 0x%02X got 0x%02X\n",
                       b, i, expected, actual);
                pass = false;
            }
        }

        // Check qh
        for (int i = 0; i < SOA_QH_SIZE; i++) {
            uint8_t expected = aos[b].qh[i];
            uint8_t actual = soa[qh_offset + b * SOA_QH_SIZE + i];
            if (expected != actual) {
                printf("FAIL: block %d qh[%d] expected 0x%02X got 0x%02X\n",
                       b, i, expected, actual);
                pass = false;
            }
        }

        // Check scales
        for (int i = 0; i < SOA_SCALES_SIZE; i++) {
            int8_t expected = aos[b].scales[i];
            int8_t actual = ((int8_t *)&soa[scales_offset])[b * SOA_SCALES_SIZE + i];
            if (expected != actual) {
                printf("FAIL: block %d scales[%d] expected %d got %d\n",
                       b, i, expected, actual);
                pass = false;
            }
        }

        // Check d
        sycl::half expected_d = aos[b].d;
        sycl::half actual_d = *((sycl::half *)&soa[d_offset + b * SOA_D_SIZE]);
        if (fabsf((float)expected_d - (float)actual_d) > 1e-5f) {
            printf("FAIL: block %d d expected %f got %f\n",
                   b, (float)expected_d, (float)actual_d);
            pass = false;
        }
    }

    printf("Layout verification: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Verify that dequantization produces same values from AoS and SoA
bool test_dequant_values() {
    printf("\n=== Dequantization Value Verification ===\n");

    const int ncols = 256;  // 1 block per row
    const int nrows = 4;
    const int nb_per_row = ncols / QK_K;
    const int64_t nblocks = nrows * nb_per_row;

    // Create random-ish but reproducible data
    std::vector<block_q6_K> aos(nblocks);
    for (int b = 0; b < nblocks; b++) {
        for (int i = 0; i < SOA_QL_SIZE; i++) {
            aos[b].ql[i] = (uint8_t)((b * 17 + i * 13) & 0xFF);
        }
        for (int i = 0; i < SOA_QH_SIZE; i++) {
            aos[b].qh[i] = (uint8_t)((b * 23 + i * 7) & 0xFF);
        }
        for (int i = 0; i < SOA_SCALES_SIZE; i++) {
            aos[b].scales[i] = (int8_t)(((b * 5 + i * 3) % 128) - 64);
        }
        aos[b].d = sycl::half(0.1f * (b + 1));
    }

    // Convert to SoA
    const int64_t soa_size = nblocks * (SOA_QL_SIZE + SOA_QH_SIZE + SOA_SCALES_SIZE + SOA_D_SIZE);
    std::vector<uint8_t> soa(soa_size);
    convert_aos_to_soa(aos.data(), soa.data(), nrows, ncols);

    // Dequantize from AoS
    std::vector<float> dequant_aos(ncols * nrows);
    for (int row = 0; row < nrows; row++) {
        const block_q6_K * blk = &aos[row * nb_per_row];
        float * dst = dequant_aos.data() + row * ncols;

        for (int ib = 0; ib < nb_per_row; ib++) {
            float d = static_cast<float>(blk[ib].d);
            const uint8_t * ql = blk[ib].ql;
            const uint8_t * qh = blk[ib].qh;
            const int8_t * sc = blk[ib].scales;

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    dst[ib * QK_K + n + l +  0] = d * sc[is + 0] * q1;
                    dst[ib * QK_K + n + l + 32] = d * sc[is + 2] * q2;
                    dst[ib * QK_K + n + l + 64] = d * sc[is + 4] * q3;
                    dst[ib * QK_K + n + l + 96] = d * sc[is + 6] * q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    // Dequantize from SoA
    const int64_t qh_offset = nblocks * SOA_QL_SIZE;
    const int64_t scales_offset = qh_offset + nblocks * SOA_QH_SIZE;
    const int64_t d_offset = scales_offset + nblocks * SOA_SCALES_SIZE;

    std::vector<float> dequant_soa(ncols * nrows);
    for (int row = 0; row < nrows; row++) {
        float * dst = dequant_soa.data() + row * ncols;

        for (int ib = 0; ib < nb_per_row; ib++) {
            int block_idx = row * nb_per_row + ib;

            float d = static_cast<float>(*((sycl::half *)&soa[d_offset + block_idx * SOA_D_SIZE]));
            const uint8_t * ql = &soa[block_idx * SOA_QL_SIZE];
            const uint8_t * qh = &soa[qh_offset + block_idx * SOA_QH_SIZE];
            const int8_t * sc = (const int8_t *)&soa[scales_offset + block_idx * SOA_SCALES_SIZE];

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    dst[ib * QK_K + n + l +  0] = d * sc[is + 0] * q1;
                    dst[ib * QK_K + n + l + 32] = d * sc[is + 2] * q2;
                    dst[ib * QK_K + n + l + 64] = d * sc[is + 4] * q3;
                    dst[ib * QK_K + n + l + 96] = d * sc[is + 6] * q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    // Compare
    bool pass = true;
    float max_diff = 0.0f;
    int first_mismatch = -1;

    for (int i = 0; i < ncols * nrows; i++) {
        float diff = fabsf(dequant_aos[i] - dequant_soa[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-5f && first_mismatch < 0) {
            first_mismatch = i;
            pass = false;
        }
    }

    if (first_mismatch >= 0) {
        int row = first_mismatch / ncols;
        int col = first_mismatch % ncols;
        printf("First mismatch at row=%d col=%d: AoS=%f SoA=%f\n",
               row, col, dequant_aos[first_mismatch], dequant_soa[first_mismatch]);
    }

    printf("Dequantization verification: %s (max_diff=%e)\n", pass ? "PASS" : "FAIL", max_diff);
    return pass;
}

int main(int argc, char ** argv) {
    (void)argc; (void)argv;

    printf("=== DMMV Q6_K SoA Reorder Verification Test ===\n");
    printf("K_QUANTS_PER_ITERATION = %d\n\n", K_QUANTS_PER_ITERATION);

    // Run fundamental tests first
    bool layout_pass = test_soa_layout_bytes();
    bool dequant_pass = test_dequant_values();

    // Run kernel memory access pattern tests
    printf("\n=== Kernel Memory Access Pattern Tests ===\n");

    std::vector<ReorderTestCase> tests = {
        // Basic tests (no split)
        {256, 1, 1, 0, "1 block, 1 row"},
        {256, 4, 4, 0, "1 block, 4 rows"},
        {512, 1, 1, 0, "2 blocks, 1 row"},
        {512, 4, 4, 0, "2 blocks, 4 rows"},
        {4096, 1, 1, 0, "16 blocks, 1 row"},
        {4096, 4, 4, 0, "16 blocks, 4 rows"},

        // Split tensor tests (row_low > 0)
        {4096, 1, 4, 0, "Split: row 0 of 4"},
        {4096, 1, 4, 1, "Split: row 1 of 4"},
        {4096, 1, 4, 2, "Split: row 2 of 4"},
        {4096, 1, 4, 3, "Split: row 3 of 4"},
        {4096, 2, 8, 0, "Split: rows 0-1 of 8"},
        {4096, 2, 8, 3, "Split: rows 3-4 of 8"},
        {4096, 2, 8, 6, "Split: rows 6-7 of 8"},
        {4096, 4, 16, 4, "Split: rows 4-7 of 16"},
        {4096, 4, 16, 12, "Split: rows 12-15 of 16"},

        // Edge cases
        {256, 1, 8, 7, "Split: last row, 1 block"},
        {512, 1, 8, 4, "Split: middle row, 2 blocks"},
    };

    int pass_count = 0;
    int fail_count = 0;

    for (const auto & tc : tests) {
        if (run_reorder_test(tc, false)) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    // Run one test verbose for debugging
    printf("\n=== Verbose Split Test ===\n");
    run_reorder_test({4096, 2, 8, 3, "Verbose: rows 3-4 of 8"}, true);

    // Summary
    printf("\n=== SUMMARY ===\n");
    printf("Layout verification: %s\n", layout_pass ? "PASS" : "FAIL");
    printf("Dequantization verification: %s\n", dequant_pass ? "PASS" : "FAIL");
    printf("Kernel access pattern tests: %d passed, %d failed\n", pass_count, fail_count);

    bool overall = layout_pass && dequant_pass && (fail_count == 0);
    printf("\n=== OVERALL: %s ===\n", overall ? "PASS" : "FAIL");

    return overall ? 0 : 1;
}
