// Unit test comparing Q6_K AoS vs SoA kernel output step-by-step
// This traces intermediate values to find where SoA diverges from AoS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>

#define QK_K 256
#define QK8_1 32
#define QI6_K 32
#define QR6_K 2
#define QI8_1 (QK8_1 / 4)  // = 8
#define VDR_Q6_K_Q8_1_MMVQ 1
#define WARP_SIZE 16

typedef sycl::half ggml_half;

// AoS block structure
typedef struct {
    uint8_t ql[QK_K/2];      // 128 bytes
    uint8_t qh[QK_K/4];      // 64 bytes
    int8_t  scales[QK_K/16]; // 16 bytes
    ggml_half d;             // 2 bytes
} block_q6_K;

typedef struct {
    ggml_half d;
    ggml_half s;
    int8_t qs[QK8_1];
} block_q8_1;

// Helper functions
static inline int get_int_from_int8_aligned(const int8_t* x8, const int i32) {
    return *((const int*)(x8 + sizeof(int) * i32));
}

static inline int get_int_from_uint8(const uint8_t* x8, const int i32) {
    const uint16_t* x16 = (const uint16_t*)(x8 + sizeof(int) * i32);
    return x16[0] | ((int)x16[1] << 16);
}

// AoS → SoA reorder (same as production code)
static void reorder_q6_k_cpu(void* dst_soa, const void* src_aos, size_t nblocks) {
    const uint8_t* aos = (const uint8_t*)src_aos;
    uint8_t* soa_ql = (uint8_t*)dst_soa;
    uint8_t* soa_qh = soa_ql + nblocks * (QK_K / 2);
    uint8_t* soa_scales = soa_qh + nblocks * (QK_K / 4);
    uint8_t* soa_d = soa_scales + nblocks * (QK_K / 16);

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t* block_aos = aos + ib * sizeof(block_q6_K);
        memcpy(soa_ql + ib * (QK_K / 2), block_aos, QK_K / 2);
        memcpy(soa_qh + ib * (QK_K / 4), block_aos + (QK_K / 2), QK_K / 4);
        memcpy(soa_scales + ib * (QK_K / 16), block_aos + (QK_K / 2) + (QK_K / 4), QK_K / 16);
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos + (QK_K / 2) + (QK_K / 4) + (QK_K / 16), sizeof(ggml_half));
    }
}

// CPU reference for AoS
float cpu_vec_dot_q6_K_aos(const block_q6_K* bq6_K, const block_q8_1* bq8_1, int iqs) {
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t* scales = bq6_K->scales + scale_offset;

    float sum = 0.0f;
    for (int i = 0; i < QR6_K; ++i) {
        const int u = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        const float d8 = float(bq8_1[bq8_offset + 2*i].d);

        // Byte-wise extraction and subtraction
        const int8_t* vl_bytes = (const int8_t*)&vl;
        const int8_t* vh_bytes = (const int8_t*)&vh;
        const int8_t* u_bytes = (const int8_t*)&u;

        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            int q_low = vl_bytes[j] & 0xF;
            int q_high = (vh_bytes[j] & 0x3) << 4;
            int q = (q_low | q_high) - 32;
            sumi += q * u_bytes[j];
        }
        sum += d8 * float(bq6_K->d) * sumi * scales[2*i];
    }
    return sum;
}

// CPU reference for SoA (mimicking GPU kernel's data access)
float cpu_vec_dot_q6_K_soa(const void* soa_data, int block_idx, int nblocks,
                           const block_q8_1* bq8_1, int iqs) {
    const uint8_t* base = (const uint8_t*)soa_data;

    // Calculate SoA offsets (matching get_block_offset and get_d_offset)
    const uint8_t* ql = base + block_idx * (QK_K / 2);
    const uint8_t* qh = base + nblocks * (QK_K / 2) + block_idx * (QK_K / 4);
    const int total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
    const int8_t* scales = (const int8_t*)(base + total_qs_bytes + block_idx * (QK_K / 16));
    const ggml_half* d = (const ggml_half*)(base + total_qs_bytes + nblocks * (QK_K / 16) + block_idx * sizeof(ggml_half));

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(ql, iqs);
    const int vh = get_int_from_uint8(qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t* scs = scales + scale_offset;

    float sum = 0.0f;
    for (int i = 0; i < QR6_K; ++i) {
        const int u = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        const float d8 = float(bq8_1[bq8_offset + 2*i].d);

        const int8_t* vl_bytes = (const int8_t*)&vl;
        const int8_t* vh_bytes = (const int8_t*)&vh;
        const int8_t* u_bytes = (const int8_t*)&u;

        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            int q_low = vl_bytes[j] & 0xF;
            int q_high = (vh_bytes[j] & 0x3) << 4;
            int q = (q_low | q_high) - 32;
            sumi += q * u_bytes[j];
        }
        sum += d8 * float(*d) * sumi * scs[2*i];
    }
    return sum;
}

// Trace offsets to compare AoS vs SoA data access
void trace_q6_K_data_access(const block_q6_K* aos_block, const void* soa_data,
                             int block_idx, int nblocks, int iqs) {
    const uint8_t* aos_base = (const uint8_t*)aos_block;
    const uint8_t* soa_base = (const uint8_t*)soa_data;

    // Calculate offsets
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    printf("\n=== iqs=%d bq8_offset=%d scale_offset=%d vh_shift=%d ===\n",
           iqs, bq8_offset, scale_offset, vh_shift);

    // AoS data
    const uint8_t* aos_ql = aos_block->ql;
    const uint8_t* aos_qh = aos_block->qh;
    const int8_t* aos_scales = aos_block->scales;

    // SoA data
    const uint8_t* soa_ql = soa_base + block_idx * (QK_K / 2);
    const uint8_t* soa_qh = soa_base + nblocks * (QK_K / 2) + block_idx * (QK_K / 4);
    const int total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
    const int8_t* soa_scales = (const int8_t*)(soa_base + total_qs_bytes + block_idx * (QK_K / 16));
    const ggml_half* soa_d = (const ggml_half*)(soa_base + total_qs_bytes + nblocks * (QK_K / 16) + block_idx * sizeof(ggml_half));

    // Read vl (4 bytes at offset iqs*4)
    const int aos_vl = get_int_from_uint8(aos_ql, iqs);
    const int soa_vl = get_int_from_uint8(soa_ql, iqs);
    printf("vl: AoS=0x%08x SoA=0x%08x %s\n", aos_vl, soa_vl, aos_vl == soa_vl ? "OK" : "MISMATCH!");

    // Read vh
    const int qh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
    const int aos_vh = get_int_from_uint8(aos_qh, qh_idx) >> vh_shift;
    const int soa_vh = get_int_from_uint8(soa_qh, qh_idx) >> vh_shift;
    printf("vh: AoS=0x%08x SoA=0x%08x (qh_idx=%d) %s\n", aos_vh, soa_vh, qh_idx, aos_vh == soa_vh ? "OK" : "MISMATCH!");

    // Read scales
    printf("scales: AoS=[%d,%d] SoA=[%d,%d] %s\n",
           aos_scales[scale_offset], aos_scales[scale_offset + 1],
           soa_scales[scale_offset], soa_scales[scale_offset + 1],
           (aos_scales[scale_offset] == soa_scales[scale_offset] &&
            aos_scales[scale_offset + 1] == soa_scales[scale_offset + 1]) ? "OK" : "MISMATCH!");

    // Read d
    printf("d: AoS=%.6f SoA=%.6f %s\n",
           float(aos_block->d), float(*soa_d),
           float(aos_block->d) == float(*soa_d) ? "OK" : "MISMATCH!");
}

int main() {
    printf("Q6_K AoS vs SoA Comparison Test\n");
    printf("================================\n\n");

    const int nblocks = 4;  // Test with multiple blocks
    const int nrows = 1;
    const int ncols = nblocks * QK_K;

    // Create AoS test data
    std::vector<block_q6_K> aos_data(nblocks);
    for (int b = 0; b < nblocks; ++b) {
        // Use distinct patterns for each block to detect index errors
        for (int i = 0; i < QK_K/2; ++i) aos_data[b].ql[i] = (0x55 + b) & 0xFF;
        for (int i = 0; i < QK_K/4; ++i) aos_data[b].qh[i] = (0x11 + b) & 0xFF;
        for (int i = 0; i < QK_K/16; ++i) aos_data[b].scales[i] = (1 + b);
        aos_data[b].d = ggml_half(1.0f / (1 + b));
    }

    // Create SoA data
    const size_t soa_size = nblocks * sizeof(block_q6_K);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_q6_k_cpu(soa_data.data(), aos_data.data(), nblocks);

    // Create Q8_1 Y data
    std::vector<block_q8_1> y_data(ncols / QK8_1);
    for (auto& block : y_data) {
        block.d = ggml_half(1.0f / 127.0f);
        block.s = ggml_half(QK8_1);
        for (int i = 0; i < QK8_1; ++i) block.qs[i] = 127;
    }

    printf("Test setup:\n");
    printf("  nblocks=%d ncols=%d\n", nblocks, ncols);
    printf("  Q6_K block size=%zu bytes\n", sizeof(block_q6_K));
    printf("  SoA total size=%zu bytes\n\n", soa_size);

    // Trace data access for each block
    for (int block = 0; block < nblocks; ++block) {
        printf("\n========== Block %d ==========\n", block);

        // Test a few iqs values
        for (int iqs : {0, 1, 8, 16, 24, 31}) {
            trace_q6_K_data_access(&aos_data[block], soa_data.data(), block, nblocks, iqs);

            // Compare vec_dot results
            float aos_result = cpu_vec_dot_q6_K_aos(&aos_data[block],
                                                     &y_data[block * (QK_K / QK8_1)], iqs);
            float soa_result = cpu_vec_dot_q6_K_soa(soa_data.data(), block, nblocks,
                                                     &y_data[block * (QK_K / QK8_1)], iqs);

            printf("vec_dot result: AoS=%.6f SoA=%.6f diff=%.2e %s\n",
                   aos_result, soa_result, aos_result - soa_result,
                   std::abs(aos_result - soa_result) < 0.01f ? "OK" : "MISMATCH!");
        }
    }

    // Summary: compute full dot product for each block
    printf("\n\n========== Full Block Results ==========\n");
    for (int block = 0; block < nblocks; ++block) {
        float aos_sum = 0.0f, soa_sum = 0.0f;
        for (int iqs = 0; iqs < QI6_K; ++iqs) {
            aos_sum += cpu_vec_dot_q6_K_aos(&aos_data[block],
                                             &y_data[block * (QK_K / QK8_1)], iqs);
            soa_sum += cpu_vec_dot_q6_K_soa(soa_data.data(), block, nblocks,
                                             &y_data[block * (QK_K / QK8_1)], iqs);
        }
        printf("Block %d: AoS=%.4f SoA=%.4f diff=%.2e %s\n",
               block, aos_sum, soa_sum, aos_sum - soa_sum,
               std::abs(aos_sum - soa_sum) < 0.01f ? "OK" : "MISMATCH!");
    }

    return 0;
}
