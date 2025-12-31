// Unit test to compare Q8_0 AoS vs SoA tile loading
// This test runs SYCL kernels to load tiles using both methods and compares results
//
// Build: Requires SYCL, use cmake with GGML_SYCL=ON
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-q8-0-tile-compare

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#ifdef GGML_USE_SYCL
#include <sycl/sycl.hpp>

#define QK8_0 32
#define QI8_0 8
#define WARP_SIZE 32

// Q8_0 block structure
typedef struct {
    sycl::half d;          // delta (scale)
    int8_t qs[QK8_0];      // quants
} block_q8_0;

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

// Helper function from ggml-sycl (reads 4 bytes as int)
static inline int get_int_from_int8(const int8_t* x8, int i) {
    const int* x32 = (const int*)(x8 + 4 * i);
    return *x32;
}

// =============================================================================
// Kernel to load tiles using AoS method (reference)
// =============================================================================
template <int mmq_y, int nwarps>
void load_tiles_aos_kernel(
    const block_q8_0* __restrict__ vx,
    int* __restrict__ result_ql,       // output: tile x_ql values
    float* __restrict__ result_dm,     // output: tile x_dm values
    const int blocks_per_row,
    const int nrows,
    const sycl::nd_item<3>& item_ct1)
{
    const int i_offset = item_ct1.get_local_id(1);  // warp index
    const int k = item_ct1.get_local_id(2);         // lane in warp
    const int i_max = nrows - 1;

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;

    // Load qs values
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (i > i_max) i = i_max;

        const block_q8_0* bxi = vx + i * blocks_per_row + kbx;
        result_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    // Load d values
    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;  // = 4
    const int kbxd = k % blocks_per_tile_x_row;

    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;
        if (i > i_max) i = i_max;

        const block_q8_0* bxi = vx + i * blocks_per_row + kbxd;
        result_dm[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = (float)bxi->d;
    }
}

// =============================================================================
// Kernel to load tiles using SoA method (what we're debugging)
// =============================================================================
template <int mmq_y, int nwarps>
void load_tiles_soa_kernel(
    const int8_t* __restrict__ qs_base,
    const size_t d_offset,
    int* __restrict__ result_ql,       // output: tile x_ql values
    float* __restrict__ result_dm,     // output: tile x_dm values
    const int blocks_per_row,
    const int nrows,
    const int row_low,                 // offset for view support
    const sycl::nd_item<3>& item_ct1)
{
    const int i_offset = item_ct1.get_local_id(1);  // warp index
    const int k = item_ct1.get_local_id(2);         // lane in warp
    const int i_max = nrows - 1;

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;

    // Compute d_base from offset
    const sycl::half* d_base = (const sycl::half*)((const uint8_t*)qs_base + d_offset);

    // row_offset and block_offset are 0 for first iteration
    const int row_offset = 0;
    const int block_offset = 0;

    // Load qs values
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (i > i_max) i = i_max;

        // SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;
        const int8_t* qs_ptr = qs_base + global_block * QK8_0;

        result_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(qs_ptr, kqsx);
    }

    // Load d values
    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;  // = 4
    const int kbxd = k % blocks_per_tile_x_row;

    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;
        if (i > i_max) i = i_max;

        // SoA addressing
        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbxd;
        const float d_val = (float)d_base[global_block];

        result_dm[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = d_val;
    }
}

// =============================================================================
// CPU reorder function (matches ggml-sycl.cpp)
// =============================================================================
static void reorder_q8_0_cpu(void* dst_soa, const void* src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks = blocks_per_row * nrows;

    const uint8_t* aos = (const uint8_t*)src_aos;
    uint8_t* soa_qs = (uint8_t*)dst_soa;
    uint8_t* soa_d = soa_qs + nblocks * QK8_0;

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t* block_aos = aos + ib * sizeof(block_q8_0);
        // Copy qs (32 bytes at offset 2 in AoS block - after d)
        memcpy(soa_qs + ib * QK8_0, block_aos + sizeof(sycl::half), QK8_0);
        // Copy d (2 bytes at offset 0 in AoS block)
        memcpy(soa_d + ib * sizeof(sycl::half), block_aos, sizeof(sycl::half));
    }
}

// =============================================================================
// Main test
// =============================================================================
int main() {
    printf("=== Q8_0 AoS vs SoA Tile Loading Comparison ===\n\n");

    try {
        // Get SYCL device
        sycl::queue q(sycl::gpu_selector_v);
        printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        // Test parameters - match typical MMQ kernel
        constexpr int mmq_y = 64;   // tile height
        constexpr int nwarps = 4;   // number of warps
        const int nrows = 128;      // total rows in matrix
        const int ncols = 128;      // total cols (must be multiple of QK8_0=32)
        const int blocks_per_row = ncols / QK8_0;  // = 4
        const int total_blocks = nrows * blocks_per_row;

        printf("Test config: mmq_y=%d, nwarps=%d, nrows=%d, ncols=%d, blocks_per_row=%d\n",
               mmq_y, nwarps, nrows, ncols, blocks_per_row);

        // Tile sizes
        const int tile_ql_size = mmq_y * (WARP_SIZE + 1);  // with bank conflict padding
        const int tile_dm_size = mmq_y * (WARP_SIZE/QI8_0 + 1);  // scale tile

        printf("Tile sizes: x_ql=%d ints, x_dm=%d floats\n", tile_ql_size, tile_dm_size);

        // =====================================================================
        // Create test data with known pattern
        // =====================================================================
        std::vector<block_q8_0> aos_data(total_blocks);
        for (int b = 0; b < total_blocks; b++) {
            // Use predictable scale values
            float scale = (b + 1) * 0.1f;
            aos_data[b].d = sycl::half(scale);

            // Use predictable qs values
            for (int i = 0; i < QK8_0; i++) {
                aos_data[b].qs[i] = (int8_t)((b * 10 + i) % 256 - 128);
            }
        }

        // Create SoA version
        const size_t qs_bytes = (size_t)total_blocks * QK8_0;
        const size_t d_bytes = (size_t)total_blocks * sizeof(sycl::half);
        const size_t soa_total = qs_bytes + d_bytes;

        std::vector<uint8_t> soa_data(soa_total);
        reorder_q8_0_cpu(soa_data.data(), aos_data.data(), ncols, nrows);

        printf("SoA layout: qs_bytes=%zu, d_bytes=%zu, d_offset=%zu\n",
               qs_bytes, d_bytes, qs_bytes);

        // =====================================================================
        // Allocate device memory
        // =====================================================================
        block_q8_0* d_aos = sycl::malloc_device<block_q8_0>(total_blocks, q);
        uint8_t* d_soa = sycl::malloc_device<uint8_t>(soa_total, q);

        int* d_aos_ql = sycl::malloc_device<int>(tile_ql_size, q);
        float* d_aos_dm = sycl::malloc_device<float>(tile_dm_size, q);

        int* d_soa_ql = sycl::malloc_device<int>(tile_ql_size, q);
        float* d_soa_dm = sycl::malloc_device<float>(tile_dm_size, q);

        // Copy data to device
        q.memcpy(d_aos, aos_data.data(), total_blocks * sizeof(block_q8_0)).wait();
        q.memcpy(d_soa, soa_data.data(), soa_total).wait();

        // Zero output buffers
        q.memset(d_aos_ql, 0, tile_ql_size * sizeof(int)).wait();
        q.memset(d_aos_dm, 0, tile_dm_size * sizeof(float)).wait();
        q.memset(d_soa_ql, 0, tile_ql_size * sizeof(int)).wait();
        q.memset(d_soa_dm, 0, tile_dm_size * sizeof(float)).wait();

        // =====================================================================
        // Launch AoS tile loader kernel
        // =====================================================================
        printf("\nLaunching AoS tile loader...\n");
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, nwarps, WARP_SIZE),
                    sycl::range<3>(1, nwarps, WARP_SIZE)
                ),
                [=](sycl::nd_item<3> item) {
                    load_tiles_aos_kernel<mmq_y, nwarps>(
                        d_aos, d_aos_ql, d_aos_dm, blocks_per_row, nrows, item);
                }
            );
        }).wait();

        // =====================================================================
        // Launch SoA tile loader kernel
        // =====================================================================
        printf("Launching SoA tile loader...\n");
        const size_t d_offset = qs_bytes;
        const int row_low = 0;  // no view offset

        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, nwarps, WARP_SIZE),
                    sycl::range<3>(1, nwarps, WARP_SIZE)
                ),
                [=](sycl::nd_item<3> item) {
                    load_tiles_soa_kernel<mmq_y, nwarps>(
                        (const int8_t*)d_soa, d_offset,
                        d_soa_ql, d_soa_dm, blocks_per_row, nrows, row_low, item);
                }
            );
        }).wait();

        // =====================================================================
        // Copy results back to host
        // =====================================================================
        std::vector<int> h_aos_ql(tile_ql_size);
        std::vector<float> h_aos_dm(tile_dm_size);
        std::vector<int> h_soa_ql(tile_ql_size);
        std::vector<float> h_soa_dm(tile_dm_size);

        q.memcpy(h_aos_ql.data(), d_aos_ql, tile_ql_size * sizeof(int)).wait();
        q.memcpy(h_aos_dm.data(), d_aos_dm, tile_dm_size * sizeof(float)).wait();
        q.memcpy(h_soa_ql.data(), d_soa_ql, tile_ql_size * sizeof(int)).wait();
        q.memcpy(h_soa_dm.data(), d_soa_dm, tile_dm_size * sizeof(float)).wait();

        // =====================================================================
        // Compare results
        // =====================================================================
        printf("\n=== Comparing x_ql (quantized values) ===\n");
        int ql_errors = 0;
        int ql_first_error_idx = -1;
        for (int i = 0; i < tile_ql_size && ql_errors < 20; i++) {
            if (h_aos_ql[i] != h_soa_ql[i]) {
                if (ql_first_error_idx < 0) ql_first_error_idx = i;
                if (ql_errors < 10) {
                    int row = i / (WARP_SIZE + 1);
                    int k = i % (WARP_SIZE + 1);
                    printf("  x_ql[%d] (row=%d k=%d): AoS=0x%08x SoA=0x%08x\n",
                           i, row, k, h_aos_ql[i], h_soa_ql[i]);
                }
                ql_errors++;
            }
        }
        printf("x_ql comparison: %d errors out of %d values\n", ql_errors, tile_ql_size);

        printf("\n=== Comparing x_dm (scale values) ===\n");
        int dm_errors = 0;
        int dm_first_error_idx = -1;
        for (int i = 0; i < tile_dm_size && dm_errors < 20; i++) {
            if (fabsf(h_aos_dm[i] - h_soa_dm[i]) > 0.001f) {
                if (dm_first_error_idx < 0) dm_first_error_idx = i;
                if (dm_errors < 10) {
                    printf("  x_dm[%d]: AoS=%.6f SoA=%.6f diff=%.6f\n",
                           i, h_aos_dm[i], h_soa_dm[i], h_aos_dm[i] - h_soa_dm[i]);
                }
                dm_errors++;
            }
        }
        printf("x_dm comparison: %d errors out of %d values\n", dm_errors, tile_dm_size);

        // Print sample values for debugging
        printf("\n=== Sample values (first 8 of each) ===\n");
        printf("x_ql AoS: ");
        for (int i = 0; i < 8; i++) printf("0x%08x ", h_aos_ql[i]);
        printf("\n");
        printf("x_ql SoA: ");
        for (int i = 0; i < 8; i++) printf("0x%08x ", h_soa_ql[i]);
        printf("\n");

        printf("x_dm AoS: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", h_aos_dm[i]);
        printf("\n");
        printf("x_dm SoA: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", h_soa_dm[i]);
        printf("\n");

        // Cleanup
        sycl::free(d_aos, q);
        sycl::free(d_soa, q);
        sycl::free(d_aos_ql, q);
        sycl::free(d_aos_dm, q);
        sycl::free(d_soa_ql, q);
        sycl::free(d_soa_dm, q);

        // Final result
        printf("\n=== RESULT: %s ===\n",
               (ql_errors == 0 && dm_errors == 0) ? "PASS" : "FAIL");

        return (ql_errors == 0 && dm_errors == 0) ? 0 : 1;

    } catch (sycl::exception& e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }
}

#else
// Non-SYCL build
int main() {
    printf("This test requires SYCL. Build with GGML_SYCL=ON\n");
    return 0;
}
#endif
