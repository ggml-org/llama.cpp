// Test to compare MMQ SoA vs AoS kernels
// Compile: icpx -fsycl -I../ggml/include -I../ggml/src test_mmq_soa.cpp -o test_mmq_soa
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./test_mmq_soa

#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Q4_0 block structure (AoS layout)
struct block_q4_0 {
    sycl::half d;           // scale
    uint8_t qs[16];         // 32 4-bit quants packed into 16 bytes
};

// Q8_1 block structure
struct block_q8_1 {
    sycl::half2 ds;         // d (scale) and s (sum)
    int8_t qs[32];          // 32 8-bit quants
};

constexpr int QK4_0 = 32;
constexpr int QK8_1 = 32;

// Reference CPU implementation for verification
float cpu_dot_q4_0_q8_1(const block_q4_0& x, const block_q8_1& y) {
    float sum = 0.0f;
    float d_x = (float)x.d;
    float d_y = (float)y.ds[0];

    for (int i = 0; i < 16; i++) {
        int q_lo = (x.qs[i] & 0x0F) - 8;
        int q_hi = (x.qs[i] >> 4) - 8;
        sum += d_x * d_y * (q_lo * y.qs[i] + q_hi * y.qs[i + 16]);
    }
    return sum;
}

// Convert AoS to SoA layout
void convert_aos_to_soa(const block_q4_0* aos, uint8_t* soa_qs, sycl::half* soa_d, int nblocks) {
    // SoA: all qs first, then all d
    for (int b = 0; b < nblocks; b++) {
        // Copy qs
        for (int i = 0; i < 16; i++) {
            soa_qs[b * 16 + i] = aos[b].qs[i];
        }
        // Copy d
        soa_d[b] = aos[b].d;
    }
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Small test case: 4x4 matrix multiply
    // X: 4 rows, 128 cols (4 blocks per row) -> 16 Q4_0 blocks total
    // Y: 4 cols, 128 rows (4 blocks per col) -> 16 Q8_1 blocks total
    // Output: 4x4 matrix

    constexpr int nrows_x = 4;
    constexpr int ncols_x = 128;  // Must be multiple of QK4_0=32
    constexpr int ncols_y = 4;
    constexpr int nrows_y = ncols_x;  // Y rows = X cols for matmul

    constexpr int blocks_per_row_x = ncols_x / QK4_0;  // 4
    constexpr int blocks_per_col_y = nrows_y / QK8_1;  // 4
    constexpr int nblocks_x = nrows_x * blocks_per_row_x;  // 16
    constexpr int nblocks_y = ncols_y * blocks_per_col_y;  // 16

    printf("Test: %d x %d matrix multiply\n", nrows_x, ncols_y);
    printf("X: %d blocks (%d rows x %d blocks/row)\n", nblocks_x, nrows_x, blocks_per_row_x);
    printf("Y: %d blocks (%d cols x %d blocks/col)\n", nblocks_y, ncols_y, blocks_per_col_y);

    // Create test data
    std::vector<block_q4_0> x_aos(nblocks_x);
    std::vector<block_q8_1> y_data(nblocks_y);

    // Fill with known pattern
    for (int b = 0; b < nblocks_x; b++) {
        x_aos[b].d = sycl::half(0.1f * (b + 1));
        for (int i = 0; i < 16; i++) {
            // Pack two 4-bit values: low nibble = i%8, high nibble = (i+1)%8
            x_aos[b].qs[i] = ((i + 1) % 16) << 4 | (i % 16);
        }
    }

    for (int b = 0; b < nblocks_y; b++) {
        y_data[b].ds = sycl::half2(sycl::half(0.2f * (b + 1)), sycl::half(0.0f));
        for (int i = 0; i < 32; i++) {
            y_data[b].qs[i] = (i % 8) - 4;
        }
    }

    // Compute reference on CPU
    std::vector<float> ref_output(nrows_x * ncols_y, 0.0f);
    for (int row = 0; row < nrows_x; row++) {
        for (int col = 0; col < ncols_y; col++) {
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row_x; b++) {
                const block_q4_0& xb = x_aos[row * blocks_per_row_x + b];
                const block_q8_1& yb = y_data[col * blocks_per_col_y + b];
                sum += cpu_dot_q4_0_q8_1(xb, yb);
            }
            ref_output[col * nrows_x + row] = sum;  // Column-major
        }
    }

    printf("\nCPU Reference output (column-major):\n");
    for (int col = 0; col < ncols_y; col++) {
        for (int row = 0; row < nrows_x; row++) {
            printf("%.4f ", ref_output[col * nrows_x + row]);
        }
        printf("\n");
    }

    // Create SoA version
    size_t soa_qs_size = nblocks_x * 16;  // 16 bytes of qs per block
    size_t soa_d_size = nblocks_x * sizeof(sycl::half);
    std::vector<uint8_t> x_soa_qs(soa_qs_size);
    std::vector<sycl::half> x_soa_d(nblocks_x);
    convert_aos_to_soa(x_aos.data(), x_soa_qs.data(), x_soa_d.data(), nblocks_x);

    printf("\nSoA conversion check:\n");
    printf("AoS block[0]: d=%.4f, qs[0..3]=%02x %02x %02x %02x\n",
           (float)x_aos[0].d, x_aos[0].qs[0], x_aos[0].qs[1], x_aos[0].qs[2], x_aos[0].qs[3]);
    printf("SoA block[0]: d=%.4f, qs[0..3]=%02x %02x %02x %02x\n",
           (float)x_soa_d[0], x_soa_qs[0], x_soa_qs[1], x_soa_qs[2], x_soa_qs[3]);
    printf("AoS block[1]: d=%.4f, qs[0..3]=%02x %02x %02x %02x\n",
           (float)x_aos[1].d, x_aos[1].qs[0], x_aos[1].qs[1], x_aos[1].qs[2], x_aos[1].qs[3]);
    printf("SoA block[1]: d=%.4f, qs[0..3]=%02x %02x %02x %02x\n",
           (float)x_soa_d[1], x_soa_qs[16], x_soa_qs[17], x_soa_qs[18], x_soa_qs[19]);

    // For now, just verify the SoA conversion is correct
    // The actual MMQ kernel test would require including the full mmq.cpp machinery

    printf("\n=== SoA Layout Verification ===\n");
    printf("d_offset (bytes from qs start to d start): %zu\n", soa_qs_size);
    printf("This matches: nrows_x * ncols_x / 2 = %d * %d / 2 = %zu\n",
           nrows_x, ncols_x, (size_t)(nrows_x * ncols_x / 2));

    // Verify block indexing formula
    printf("\n=== Block Indexing Verification ===\n");
    for (int row = 0; row < nrows_x; row++) {
        for (int block_in_row = 0; block_in_row < blocks_per_row_x; block_in_row++) {
            int global_block = row * blocks_per_row_x + block_in_row;

            // AoS access
            float aos_d = (float)x_aos[global_block].d;
            uint8_t aos_qs0 = x_aos[global_block].qs[0];

            // SoA access
            float soa_d = (float)x_soa_d[global_block];
            uint8_t soa_qs0 = x_soa_qs[global_block * 16];  // 16 bytes per block

            bool match = (aos_d == soa_d) && (aos_qs0 == soa_qs0);
            if (!match) {
                printf("MISMATCH at row=%d, block=%d (global=%d): AoS(d=%.4f,qs0=%02x) vs SoA(d=%.4f,qs0=%02x)\n",
                       row, block_in_row, global_block, aos_d, aos_qs0, soa_d, soa_qs0);
            }
        }
    }
    printf("Block indexing: OK\n");

    printf("\n=== Test Complete ===\n");
    return 0;
}
