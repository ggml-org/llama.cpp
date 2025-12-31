// Test 2: Y-tile loading test for Q6_K SoA kernel
// Verifies Y-tile (activation) loading with phase-based overwriting
// Y tiles use block_q8_1 format (AoS, 36-byte stride)

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Constants
#define QK_K 256
#define QI6_K 32
#define QR6_K 2
#define QK8_1 32
#define QI8_1 8
#define WARP_SIZE 32
#define MMQ_TILE_NE_K 32

// Tile dimensions matching actual kernel
constexpr int mmq_x = 64;
constexpr int mmq_y = 128;
constexpr int nwarps = 4;

// block_q8_1 structure (36 bytes)
struct block_q8_1 {
    sycl::half2 ds;      // 4 bytes: d and sum
    int8_t qs[QK8_1];    // 32 bytes
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 size mismatch");

// Y-tile sizes
constexpr int tile_y_qs_size = mmq_x * WARP_SIZE;
constexpr int tile_y_ds_size = mmq_x * (WARP_SIZE / QI8_1);

// Helper: get_int_from_int8_aligned
static inline int get_int_from_int8_aligned(const int8_t *x, int i) {
    return ((const int *)x)[i];
}

// Y-tile loading kernel (matches mul_mat_q6_K_soa Y-loading pattern)
void load_y_tiles_kernel(
    const block_q8_1 *__restrict__ y,
    int *__restrict__ tile_y_qs,
    float *__restrict__ tile_y_ds,
    const int blocks_per_col_y,
    const int col_y_0,
    const int ncols_y,
    const int ib0,
    const int phase,  // ir = 0 or 1 for Q6_K (phases_per_iter = QR6_K = 2)
    sycl::nd_item<2> item)
{
    const int i_offset = item.get_local_id(0);  // 0-3 (nwarps)
    const int k = item.get_local_id(1);         // 0-31 (WARP_SIZE)

    constexpr int qk = QK_K;

    // Y-tile qs loading
    const int kqs = phase * WARP_SIZE + k;
    const int kbxd = kqs / QI8_1;

    for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = std::min(col_y_0 + i_offset + i, ncols_y - 1);

        const block_q8_1 *by0 = &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + kbxd];

        const int index_y = (i_offset + i) * WARP_SIZE + kqs % WARP_SIZE;
        tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, k % QI8_1);
    }

    // Y-tile ds loading
    for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
        const int ids = (ids0 + i_offset * QI8_1 + k / (WARP_SIZE / QI8_1)) % mmq_x;
        const int kby = k % (WARP_SIZE / QI8_1);
        const int col_y_eff = std::min(col_y_0 + ids, ncols_y - 1);

        const sycl::half2 *dsi_src =
            &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
               phase * (WARP_SIZE / QI8_1) + kby].ds;

        float *dfi_dst = &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
        *dfi_dst = (float)(*dsi_src)[0];  // need_sum=false, extract d only
    }
}

int main() {
    printf("=== Test 2: Y-tile loading test ===\n");
    printf("Tests if Y-tile loading (block_q8_1 format) works correctly\n\n");

    // Create test data
    const int ncols_y = 64;  // mmq_x columns
    const int nrows_y = QK_K;  // One block column
    const int blocks_per_col_y = nrows_y / QK8_1;  // 8 blocks per column

    printf("Y layout: ncols=%d, blocks_per_col=%d\n", ncols_y, blocks_per_col_y);

    // Allocate Y blocks
    int nblocks_y = ncols_y * blocks_per_col_y;
    block_q8_1 *y_data = (block_q8_1 *)malloc(nblocks_y * sizeof(block_q8_1));
    memset(y_data, 0, nblocks_y * sizeof(block_q8_1));

    // Fill with recognizable pattern
    for (int col = 0; col < ncols_y; col++) {
        for (int blk = 0; blk < blocks_per_col_y; blk++) {
            block_q8_1 *b = &y_data[col * blocks_per_col_y + blk];
            // d value: col * 0.1 + blk * 0.01
            float d_val = col * 0.1f + blk * 0.01f;
            b->ds[0] = sycl::half(d_val);
            b->ds[1] = sycl::half(0.0f);
            // qs: (col + blk * 32 + i) as signed byte
            for (int i = 0; i < QK8_1; i++) {
                b->qs[i] = (int8_t)((col + blk * 32 + i) & 0x7F) - 64;
            }
        }
    }

    // CPU reference
    int *cpu_tile_y_qs = (int *)malloc(tile_y_qs_size * sizeof(int));
    float *cpu_tile_y_ds = (float *)malloc(tile_y_ds_size * sizeof(float));
    memset(cpu_tile_y_qs, 0, tile_y_qs_size * sizeof(int));
    memset(cpu_tile_y_ds, 0, tile_y_ds_size * sizeof(float));

    // Run CPU reference for both phases
    printf("Running CPU reference for phases 0 and 1...\n");
    for (int phase = 0; phase < QR6_K; phase++) {
        for (int i_offset = 0; i_offset < nwarps; i_offset++) {
            for (int k = 0; k < WARP_SIZE; k++) {
                constexpr int qk = QK_K;
                const int col_y_0 = 0;
                const int ib0 = 0;

                // Y-tile qs loading
                const int kqs = phase * WARP_SIZE + k;
                const int kbxd = kqs / QI8_1;

                for (int i = 0; i < mmq_x; i += nwarps) {
                    const int col_y_eff = std::min(col_y_0 + i_offset + i, ncols_y - 1);
                    const block_q8_1 *by0 = &y_data[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + kbxd];
                    const int index_y = (i_offset + i) * WARP_SIZE + kqs % WARP_SIZE;
                    cpu_tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, k % QI8_1);
                }

                // Y-tile ds loading
                for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                    const int ids = (ids0 + i_offset * QI8_1 + k / (WARP_SIZE / QI8_1)) % mmq_x;
                    const int kby = k % (WARP_SIZE / QI8_1);
                    const int col_y_eff = std::min(col_y_0 + ids, ncols_y - 1);
                    const sycl::half2 *dsi_src =
                        &y_data[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                               phase * (WARP_SIZE / QI8_1) + kby].ds;
                    float *dfi_dst = &cpu_tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                    *dfi_dst = (float)(*dsi_src)[0];
                }
            }
        }
    }

    // Count non-zero CPU values
    int cpu_qs_nonzero = 0, cpu_ds_nonzero = 0;
    for (int i = 0; i < tile_y_qs_size; i++) if (cpu_tile_y_qs[i] != 0) cpu_qs_nonzero++;
    for (int i = 0; i < tile_y_ds_size; i++) if (cpu_tile_y_ds[i] != 0.0f) cpu_ds_nonzero++;
    printf("CPU tiles: qs_nonzero=%d/%d, ds_nonzero=%d/%d\n",
           cpu_qs_nonzero, tile_y_qs_size, cpu_ds_nonzero, tile_y_ds_size);

    // GPU version
    printf("\nRunning GPU version...\n");

    try {
        sycl::queue q{sycl::gpu_selector_v};
        printf("GPU: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        // Allocate GPU memory
        block_q8_1 *d_y = sycl::malloc_device<block_q8_1>(nblocks_y, q);
        int *d_tile_y_qs = sycl::malloc_device<int>(tile_y_qs_size, q);
        float *d_tile_y_ds = sycl::malloc_device<float>(tile_y_ds_size, q);

        q.memcpy(d_y, y_data, nblocks_y * sizeof(block_q8_1)).wait();
        q.memset(d_tile_y_qs, 0, tile_y_qs_size * sizeof(int)).wait();
        q.memset(d_tile_y_ds, 0, tile_y_ds_size * sizeof(float)).wait();

        // Run both phases
        for (int phase = 0; phase < QR6_K; phase++) {
            q.submit([&](sycl::handler &h) {
                h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>(nwarps, WARP_SIZE),
                                      sycl::range<2>(nwarps, WARP_SIZE)),
                    [=](sycl::nd_item<2> item) {
                        load_y_tiles_kernel(d_y, d_tile_y_qs, d_tile_y_ds,
                                           blocks_per_col_y, 0, ncols_y, 0, phase, item);
                    });
            }).wait();
        }

        // Copy results back
        int *gpu_tile_y_qs = (int *)malloc(tile_y_qs_size * sizeof(int));
        float *gpu_tile_y_ds = (float *)malloc(tile_y_ds_size * sizeof(float));

        q.memcpy(gpu_tile_y_qs, d_tile_y_qs, tile_y_qs_size * sizeof(int)).wait();
        q.memcpy(gpu_tile_y_ds, d_tile_y_ds, tile_y_ds_size * sizeof(float)).wait();

        // Compare
        printf("\nComparing CPU vs GPU...\n");

        int qs_mismatches = 0, ds_mismatches = 0;
        int first_qs_mismatch = -1, first_ds_mismatch = -1;

        for (int i = 0; i < tile_y_qs_size; i++) {
            if (cpu_tile_y_qs[i] != gpu_tile_y_qs[i]) {
                if (first_qs_mismatch < 0) first_qs_mismatch = i;
                qs_mismatches++;
            }
        }

        for (int i = 0; i < tile_y_ds_size; i++) {
            if (fabsf(cpu_tile_y_ds[i] - gpu_tile_y_ds[i]) > 1e-4f) {
                if (first_ds_mismatch < 0) first_ds_mismatch = i;
                ds_mismatches++;
            }
        }

        printf("tile_y_qs mismatches: %d/%d", qs_mismatches, tile_y_qs_size);
        if (first_qs_mismatch >= 0) {
            printf(" (first at [%d]: CPU=0x%08X, GPU=0x%08X)",
                   first_qs_mismatch, cpu_tile_y_qs[first_qs_mismatch], gpu_tile_y_qs[first_qs_mismatch]);
        }
        printf("\n");

        printf("tile_y_ds mismatches: %d/%d", ds_mismatches, tile_y_ds_size);
        if (first_ds_mismatch >= 0) {
            printf(" (first at [%d]: CPU=%f, GPU=%f)",
                   first_ds_mismatch, cpu_tile_y_ds[first_ds_mismatch], gpu_tile_y_ds[first_ds_mismatch]);
        }
        printf("\n");

        // Check for NaN
        int nan_count = 0;
        for (int i = 0; i < tile_y_ds_size; i++) {
            if (std::isnan(gpu_tile_y_ds[i]) || std::isinf(gpu_tile_y_ds[i])) {
                nan_count++;
            }
        }
        if (nan_count > 0) {
            printf("WARNING: Found %d NaN/Inf values in GPU tile_y_ds!\n", nan_count);
        }

        bool pass = (qs_mismatches == 0 && ds_mismatches == 0 && nan_count == 0);
        printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

        // Cleanup
        sycl::free(d_y, q);
        sycl::free(d_tile_y_qs, q);
        sycl::free(d_tile_y_ds, q);
        free(gpu_tile_y_qs);
        free(gpu_tile_y_ds);

    } catch (sycl::exception &e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }

    free(y_data);
    free(cpu_tile_y_qs);
    free(cpu_tile_y_ds);

    return 0;
}
