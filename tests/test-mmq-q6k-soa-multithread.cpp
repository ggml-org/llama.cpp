// Test 1: Multi-thread X-tile loading test for Q6_K SoA kernel
// Runs load_tiles_q6_K_soa with 32 threads and verifies all tiles are correct
// This isolates whether the bug is in cooperative tile loading

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Constants from mmq.cpp
#define QK_K 256
#define QI6_K 32
#define QR6_K 2
#define WARP_SIZE 32
#define MMQ_TILE_NE_K 32

// Tile dimensions matching actual kernel
constexpr int mmq_y = 128;
constexpr int nwarps = 4;

// Tile sizes (from allocate_tiles_q6_K)
constexpr int tile_x_ql_size = mmq_y * (2 * MMQ_TILE_NE_K) + mmq_y;
constexpr int tile_x_dm_size = mmq_y * (MMQ_TILE_NE_K / QI6_K) + mmq_y / QI6_K;
constexpr int tile_x_sc_size = mmq_y * (MMQ_TILE_NE_K / 8) + mmq_y / 8;

// Helper: get_int_from_uint8
static inline int get_int_from_uint8(const uint8_t *x, int i) {
    return ((const int *)x)[i];
}

// Helper: get_int_from_int8
static inline int get_int_from_int8(const int8_t *x, int i) {
    return ((const int *)x)[i];
}

// SoA layout offsets
struct SoALayout {
    size_t ql_offset;    // = 0
    size_t qh_offset;    // = nblocks * 128
    size_t scales_offset; // = nblocks * 192
    size_t d_offset;     // = nblocks * 208
    int nblocks;
    int blocks_per_row;
};

// The actual load_tiles_q6_K_soa kernel logic (simplified for testing)
template <bool need_check>
void load_tiles_q6_K_soa_kernel(
    const uint8_t *__restrict__ qs_base,
    const size_t qh_offset,
    const size_t scales_offset,
    const size_t d_offset,
    int *__restrict__ x_ql,
    float *__restrict__ x_dmf,
    int *__restrict__ x_sc,
    const int i_offset,
    const int i_max,
    const int k,
    const int blocks_per_row,
    const int row_offset,
    const int block_offset,
    const int row_low)
{
    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    // Compute bases for each section
    const uint8_t *qh_base = qs_base + qh_offset;
    const int8_t *scales_base = (const int8_t *)(qs_base + scales_offset);
    const sycl::half *d_base = (const sycl::half *)(qs_base + d_offset);

    // Load ql and qh
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = std::min(i, i_max);
        }

        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;

        const uint8_t *ql_ptr = qs_base + global_block * (QK_K / 2);
        const uint8_t *qh_ptr = qh_base + global_block * (QK_K / 4);

        const int ky = QR6_K * kqsx;

        const int ql = get_int_from_uint8(ql_ptr, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(qh_ptr, (QI6_K / 4) * (kqsx / (QI6_K / 2)) + kqsx % (QI6_K / 4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K / 2)) / (QI6_K / 4)))) << 4) & 0x30303030;
        const int qh1 = (qh >> (2 * ((kqsx % (QI6_K / 2)) / (QI6_K / 4)))) & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K / 2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K / 2) + (QI6_K / 2);

        // Compute final values (sub_sat not needed for testing - just store raw)
        int val0 = ql0 | qh0;
        int val1 = ql1 | qh1;
        // Apply sub_sat: subtract 32 with saturation
        // For signed bytes: result = max(val - 32, -128)
        int8_t *v0 = (int8_t *)&val0;
        int8_t *v1 = (int8_t *)&val1;
        for (int b = 0; b < 4; b++) {
            v0[b] = (int8_t)(v0[b] - 32);
            v1[b] = (int8_t)(v1[b] - 32);
        }

        x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq0] = val0;
        x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq1] = val1;
    }

    // Load d values
    constexpr int blocks_per_tile_x_row = QI6_K > MMQ_TILE_NE_K ? 1 : MMQ_TILE_NE_K / QI6_K;
    const int kbxd = k % blocks_per_tile_x_row;

    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = std::min(i, i_max);
        }

        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbxd;

        x_dmf[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K + kbxd] = (float)d_base[global_block];
    }

    // Load scales
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (MMQ_TILE_NE_K / 8)) % mmq_y;
        if (need_check) {
            i = std::min(i, i_max);
        }

        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + (k % (MMQ_TILE_NE_K / 8)) / 4;

        const int8_t *sc_ptr = scales_base + global_block * (QK_K / 16);

        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k % (MMQ_TILE_NE_K / 8)] = get_int_from_int8(sc_ptr, k % (QI6_K / 8));
    }
}

// GPU kernel that runs load_tiles with all 32 threads in parallel
void multithread_load_tiles_kernel(
    const uint8_t *__restrict__ qs_base,
    size_t qh_offset, size_t scales_offset, size_t d_offset,
    int *__restrict__ x_ql,
    float *__restrict__ x_dmf,
    int *__restrict__ x_sc,
    int blocks_per_row,
    int row_offset,
    int block_offset,
    int row_low,
    int i_max,
    sycl::nd_item<2> item)
{
    const int i_offset = item.get_local_id(0);  // 0-3 (nwarps)
    const int k = item.get_local_id(1);         // 0-31 (WARP_SIZE)

    // Run the actual tile loading logic
    load_tiles_q6_K_soa_kernel<false>(
        qs_base, qh_offset, scales_offset, d_offset,
        x_ql, x_dmf, x_sc,
        i_offset, i_max, k, blocks_per_row,
        row_offset, block_offset, row_low);
}

int main() {
    printf("=== Test 1: Multi-thread X-tile loading ===\n");
    printf("Tests if 32 threads cooperatively loading tiles produces correct results\n\n");

    // Create test data: single row with one block
    const int nrows = 128;  // mmq_y rows
    const int ncols = QK_K; // One block per row
    const int blocks_per_row = 1;
    const int nblocks = nrows * blocks_per_row;

    // Calculate SoA layout offsets
    SoALayout layout;
    layout.ql_offset = 0;
    layout.qh_offset = nblocks * (QK_K / 2);      // 128 bytes per block
    layout.scales_offset = layout.qh_offset + nblocks * (QK_K / 4);  // 64 bytes per block
    layout.d_offset = layout.scales_offset + nblocks * (QK_K / 16);  // 16 bytes per block
    layout.nblocks = nblocks;
    layout.blocks_per_row = blocks_per_row;

    size_t total_size = layout.d_offset + nblocks * sizeof(sycl::half);
    printf("SoA layout: qh_offset=%zu, scales_offset=%zu, d_offset=%zu, total=%zu\n",
           layout.qh_offset, layout.scales_offset, layout.d_offset, total_size);

    // Allocate and initialize test data with known pattern
    uint8_t *soa_data = (uint8_t *)malloc(total_size);
    memset(soa_data, 0, total_size);

    // Fill with a recognizable pattern
    // ql: block b, byte i = (b * 128 + i) & 0xFF
    for (int b = 0; b < nblocks; b++) {
        uint8_t *ql = soa_data + b * (QK_K / 2);
        for (int i = 0; i < QK_K / 2; i++) {
            ql[i] = (uint8_t)((b * 128 + i) & 0xFF);
        }
    }

    // qh: block b, byte i = (b * 64 + i) & 0x3F (6 bits)
    for (int b = 0; b < nblocks; b++) {
        uint8_t *qh = soa_data + layout.qh_offset + b * (QK_K / 4);
        for (int i = 0; i < QK_K / 4; i++) {
            qh[i] = (uint8_t)((b * 64 + i) & 0x3F);
        }
    }

    // scales: block b, byte i = (b * 16 + i - 8) as signed
    for (int b = 0; b < nblocks; b++) {
        int8_t *sc = (int8_t *)(soa_data + layout.scales_offset + b * (QK_K / 16));
        for (int i = 0; i < QK_K / 16; i++) {
            sc[i] = (int8_t)((b * 16 + i) - 8);
        }
    }

    // d: block b = 0.5 + b * 0.001
    sycl::half *d_data = (sycl::half *)(soa_data + layout.d_offset);
    for (int b = 0; b < nblocks; b++) {
        d_data[b] = sycl::half(0.5f + b * 0.001f);
    }

    // Allocate CPU reference tiles
    int *cpu_x_ql = (int *)malloc(tile_x_ql_size * sizeof(int));
    float *cpu_x_dmf = (float *)malloc(tile_x_dm_size * sizeof(float));
    int *cpu_x_sc = (int *)malloc(tile_x_sc_size * sizeof(int));
    memset(cpu_x_ql, 0, tile_x_ql_size * sizeof(int));
    memset(cpu_x_dmf, 0, tile_x_dm_size * sizeof(float));
    memset(cpu_x_sc, 0, tile_x_sc_size * sizeof(int));

    // Run CPU reference (simulating all threads sequentially)
    printf("Running CPU reference...\n");
    for (int i_offset = 0; i_offset < nwarps; i_offset++) {
        for (int k = 0; k < WARP_SIZE; k++) {
            load_tiles_q6_K_soa_kernel<false>(
                soa_data, layout.qh_offset, layout.scales_offset, layout.d_offset,
                cpu_x_ql, cpu_x_dmf, cpu_x_sc,
                i_offset, mmq_y - 1, k, blocks_per_row,
                0, 0, 0);
        }
    }

    // Count non-zero CPU values
    int cpu_ql_nonzero = 0, cpu_dm_nonzero = 0, cpu_sc_nonzero = 0;
    for (int i = 0; i < tile_x_ql_size; i++) if (cpu_x_ql[i] != 0) cpu_ql_nonzero++;
    for (int i = 0; i < tile_x_dm_size; i++) if (cpu_x_dmf[i] != 0.0f) cpu_dm_nonzero++;
    for (int i = 0; i < tile_x_sc_size; i++) if (cpu_x_sc[i] != 0) cpu_sc_nonzero++;
    printf("CPU tiles: ql_nonzero=%d/%d, dm_nonzero=%d/%d, sc_nonzero=%d/%d\n",
           cpu_ql_nonzero, tile_x_ql_size, cpu_dm_nonzero, tile_x_dm_size, cpu_sc_nonzero, tile_x_sc_size);

    // Run GPU version
    printf("\nRunning GPU version with %d threads...\n", nwarps * WARP_SIZE);

    try {
        sycl::queue q{sycl::gpu_selector_v};
        printf("GPU: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        // Allocate GPU memory
        uint8_t *d_soa = sycl::malloc_device<uint8_t>(total_size, q);
        int *d_x_ql = sycl::malloc_device<int>(tile_x_ql_size, q);
        float *d_x_dmf = sycl::malloc_device<float>(tile_x_dm_size, q);
        int *d_x_sc = sycl::malloc_device<int>(tile_x_sc_size, q);

        // Copy data to GPU
        q.memcpy(d_soa, soa_data, total_size).wait();
        q.memset(d_x_ql, 0, tile_x_ql_size * sizeof(int)).wait();
        q.memset(d_x_dmf, 0, tile_x_dm_size * sizeof(float)).wait();
        q.memset(d_x_sc, 0, tile_x_sc_size * sizeof(int)).wait();

        // Launch kernel with nwarps x WARP_SIZE threads
        q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(nwarps, WARP_SIZE),
                                  sycl::range<2>(nwarps, WARP_SIZE)),
                [=](sycl::nd_item<2> item) {
                    multithread_load_tiles_kernel(
                        d_soa, layout.qh_offset, layout.scales_offset, layout.d_offset,
                        d_x_ql, d_x_dmf, d_x_sc,
                        blocks_per_row, 0, 0, 0, mmq_y - 1, item);
                });
        }).wait();

        // Copy results back
        int *gpu_x_ql = (int *)malloc(tile_x_ql_size * sizeof(int));
        float *gpu_x_dmf = (float *)malloc(tile_x_dm_size * sizeof(float));
        int *gpu_x_sc = (int *)malloc(tile_x_sc_size * sizeof(int));

        q.memcpy(gpu_x_ql, d_x_ql, tile_x_ql_size * sizeof(int)).wait();
        q.memcpy(gpu_x_dmf, d_x_dmf, tile_x_dm_size * sizeof(float)).wait();
        q.memcpy(gpu_x_sc, d_x_sc, tile_x_sc_size * sizeof(int)).wait();

        // Compare CPU vs GPU
        printf("\nComparing CPU vs GPU tiles...\n");

        int ql_mismatches = 0, dm_mismatches = 0, sc_mismatches = 0;
        int first_ql_mismatch = -1, first_dm_mismatch = -1, first_sc_mismatch = -1;

        for (int i = 0; i < tile_x_ql_size; i++) {
            if (cpu_x_ql[i] != gpu_x_ql[i]) {
                if (first_ql_mismatch < 0) first_ql_mismatch = i;
                ql_mismatches++;
            }
        }

        for (int i = 0; i < tile_x_dm_size; i++) {
            if (fabsf(cpu_x_dmf[i] - gpu_x_dmf[i]) > 1e-5f) {
                if (first_dm_mismatch < 0) first_dm_mismatch = i;
                dm_mismatches++;
            }
        }

        for (int i = 0; i < tile_x_sc_size; i++) {
            if (cpu_x_sc[i] != gpu_x_sc[i]) {
                if (first_sc_mismatch < 0) first_sc_mismatch = i;
                sc_mismatches++;
            }
        }

        printf("x_ql mismatches: %d/%d", ql_mismatches, tile_x_ql_size);
        if (first_ql_mismatch >= 0) {
            printf(" (first at [%d]: CPU=0x%08X, GPU=0x%08X)",
                   first_ql_mismatch, cpu_x_ql[first_ql_mismatch], gpu_x_ql[first_ql_mismatch]);
        }
        printf("\n");

        printf("x_dm mismatches: %d/%d", dm_mismatches, tile_x_dm_size);
        if (first_dm_mismatch >= 0) {
            printf(" (first at [%d]: CPU=%f, GPU=%f)",
                   first_dm_mismatch, cpu_x_dmf[first_dm_mismatch], gpu_x_dmf[first_dm_mismatch]);
        }
        printf("\n");

        printf("x_sc mismatches: %d/%d", sc_mismatches, tile_x_sc_size);
        if (first_sc_mismatch >= 0) {
            printf(" (first at [%d]: CPU=0x%08X, GPU=0x%08X)",
                   first_sc_mismatch, cpu_x_sc[first_sc_mismatch], gpu_x_sc[first_sc_mismatch]);
        }
        printf("\n");

        // Check for NaN/Inf in GPU d values
        int nan_count = 0;
        for (int i = 0; i < tile_x_dm_size; i++) {
            if (std::isnan(gpu_x_dmf[i]) || std::isinf(gpu_x_dmf[i])) {
                nan_count++;
            }
        }
        if (nan_count > 0) {
            printf("WARNING: Found %d NaN/Inf values in GPU x_dm!\n", nan_count);
        }

        bool pass = (ql_mismatches == 0 && dm_mismatches == 0 && sc_mismatches == 0 && nan_count == 0);
        printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

        // Cleanup
        sycl::free(d_soa, q);
        sycl::free(d_x_ql, q);
        sycl::free(d_x_dmf, q);
        sycl::free(d_x_sc, q);
        free(gpu_x_ql);
        free(gpu_x_dmf);
        free(gpu_x_sc);

    } catch (sycl::exception &e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }

    free(soa_data);
    free(cpu_x_ql);
    free(cpu_x_dmf);
    free(cpu_x_sc);

    return 0;
}
