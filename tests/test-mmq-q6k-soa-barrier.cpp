// Test 5: SLM barrier test for Q6_K SoA kernel
// Checks if work-group barriers are being respected during tile loading
// This simulates the actual kernel's cooperative loading pattern

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
#define VDR_Q6_K_Q8_1_MMQ 1

// Tile dimensions
constexpr int mmq_x = 64;
constexpr int mmq_y = 128;
constexpr int nwarps = 4;

// Tile sizes
constexpr int tile_x_ql_size = mmq_y * (2 * MMQ_TILE_NE_K) + mmq_y;
constexpr int tile_x_dm_size = mmq_y * (MMQ_TILE_NE_K / QI6_K) + mmq_y / QI6_K;
constexpr int tile_x_sc_size = mmq_y * (MMQ_TILE_NE_K / 8) + mmq_y / 8;
constexpr int tile_y_qs_size = mmq_x * WARP_SIZE;
constexpr int tile_y_ds_size = mmq_x * (WARP_SIZE / QI8_1);

// Helper functions
static inline int get_int_from_uint8(const uint8_t *x, int i) {
    return ((const int *)x)[i];
}

static inline int get_int_from_int8(const int8_t *x, int i) {
    return ((const int *)x)[i];
}

static inline int get_int_from_int8_aligned(const int8_t *x, int i) {
    return ((const int *)x)[i];
}

// block_q8_1 structure
struct block_q8_1 {
    sycl::half2 ds;
    int8_t qs[QK8_1];
};

// GPU kernel that mimics the full MMQ kernel structure with barriers
// This specifically tests the barrier synchronization pattern
void barrier_test_kernel(
    const uint8_t *__restrict__ qs_base,
    size_t qh_offset, size_t scales_offset, size_t d_offset,
    const block_q8_1 *__restrict__ y,
    float *__restrict__ results,
    int blocks_per_row_x,
    int blocks_per_col_y,
    int nrows_x,
    int ncols_y,
    int nrows_dst,
    int row_low,
    sycl::nd_item<3> item,
    int *tile_x_ql,      // Shared local memory
    float *tile_x_dmf,
    int *tile_x_sc,
    int *tile_y_qs,
    float *tile_y_df)
{
    const int i_offset = item.get_local_id(1);  // 0-3 (nwarps)
    const int k = item.get_local_id(2);         // 0-31 (WARP_SIZE)

    const int row_dst_0 = item.get_group(2) * mmq_y;
    const int col_dst_0 = item.get_group(1) * mmq_x;

    // Accumulators
    float sum[mmq_y / WARP_SIZE][mmq_x / nwarps] = {{0.0f}};

    // Bases for SoA sections
    const uint8_t *qh_base = qs_base + qh_offset;
    const int8_t *scales_base = (const int8_t *)(qs_base + scales_offset);
    const sycl::half *d_base = (const sycl::half *)(qs_base + d_offset);

    // Main loop over blocks
    constexpr int blocks_per_iter = 1;
    constexpr int phases_per_iter = QR6_K;

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {
        // === Load X tiles ===
        const int kbx = k / QI6_K;
        const int kqsx = k % QI6_K;

        for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
            int i = i0 + i_offset;
            if (i >= nrows_x - row_dst_0) i = nrows_x - row_dst_0 - 1;

            const int global_row = row_low + row_dst_0 + i;
            const int global_block = global_row * blocks_per_row_x + ib0 + kbx;

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

            int val0 = ql0 | qh0;
            int val1 = ql1 | qh1;
            // sub_sat: subtract 32 with saturation
            int8_t *v0 = (int8_t *)&val0;
            int8_t *v1 = (int8_t *)&val1;
            for (int b = 0; b < 4; b++) {
                v0[b] = (int8_t)(v0[b] - 32);
                v1[b] = (int8_t)(v1[b] - 32);
            }

            tile_x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq0] = val0;
            tile_x_ql[i * (2 * MMQ_TILE_NE_K + 1) + kq1] = val1;
        }

        // Load d values
        constexpr int blocks_per_tile_x_row = 1;
        const int kbxd = k % blocks_per_tile_x_row;
        {
            int i = (0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;
            if (i >= nrows_x - row_dst_0) i = nrows_x - row_dst_0 - 1;
            const int global_row = row_low + row_dst_0 + i;
            const int global_block = global_row * blocks_per_row_x + ib0 + kbxd;
            tile_x_dmf[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K + kbxd] = (float)d_base[global_block];
        }

        // Load scales
        {
            int i = (0 + i_offset * 8 + k / (MMQ_TILE_NE_K / 8)) % mmq_y;
            if (i >= nrows_x - row_dst_0) i = nrows_x - row_dst_0 - 1;
            const int global_row = row_low + row_dst_0 + i;
            const int global_block = global_row * blocks_per_row_x + ib0 + (k % (MMQ_TILE_NE_K / 8)) / 4;
            const int8_t *sc_ptr = scales_base + global_block * (QK_K / 16);
            tile_x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k % (MMQ_TILE_NE_K / 8)] = get_int_from_int8(sc_ptr, k % (QI6_K / 8));
        }

        // Y-tile loading + compute phases
        for (int ir = 0; ir < phases_per_iter; ++ir) {
            const int kqs = ir * WARP_SIZE + k;
            const int kbxd = kqs / QI8_1;

            // Load Y qs
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = sycl::min(col_dst_0 + i_offset + i, ncols_y - 1);
                const block_q8_1 *by0 = &y[col_y_eff * blocks_per_col_y + ib0 * (QK_K / QK8_1) + kbxd];
                const int index_y = (i_offset + i) * WARP_SIZE + kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, k % QI8_1);
            }

            // Load Y ds
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + i_offset * QI8_1 + k / (WARP_SIZE / QI8_1)) % mmq_x;
                const int kby = k % (WARP_SIZE / QI8_1);
                const int col_y_eff = sycl::min(col_dst_0 + ids, ncols_y - 1);
                const sycl::half2 *dsi_src = &y[col_y_eff * blocks_per_col_y + ib0 * (QK_K / QK8_1) + ir * (WARP_SIZE / QI8_1) + kby].ds;
                tile_y_df[ids * (WARP_SIZE / QI8_1) + kby] = (float)(*dsi_src)[0];
            }

            // ========== BARRIER ==========
            // This is the critical synchronization point
            item.barrier(sycl::access::fence_space::local_space);

            // Compute dot products
            for (int kk = ir * WARP_SIZE / QR6_K; kk < (ir + 1) * WARP_SIZE / QR6_K; kk += VDR_Q6_K_Q8_1_MMQ) {
                for (int iy = 0; iy < mmq_y / WARP_SIZE; ++iy) {
                    for (int ix = 0; ix < mmq_x / nwarps; ++ix) {
                        int i = k + iy * WARP_SIZE;
                        int j = i_offset + ix * nwarps;

                        const int8_t *sc = ((const int8_t *)&tile_x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + kk / 8]);
                        const int index_x = i * (QR6_K * MMQ_TILE_NE_K + 1) + QR6_K * kk;
                        const int ky = QR6_K * kk;
                        const int index_y = j * WARP_SIZE + ky % WARP_SIZE;
                        const int index_y_ds = j * (WARP_SIZE / QI8_1) + (ky % WARP_SIZE) / QI8_1;

                        const int *vl = &tile_x_ql[index_x];
                        const int *vh = &tile_x_ql[index_x + 4];
                        const int *u = &tile_y_qs[index_y];
                        float d = tile_x_dmf[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K];
                        const float *df = &tile_y_df[index_y_ds];

                        // vec_dot computation
                        float dot_sum = 0.0f;
                        for (int iir = 0; iir < QR6_K; ++iir) {
                            const int sc0 = sc[4 * iir + 0];
                            const int sc1 = sc[4 * iir + 1];
                            const int sc2 = sc[4 * iir + 2];
                            const int sc3 = sc[4 * iir + 3];

                            const int ui_base = iir * 16;

                            auto dp4a = [](int a, int b) {
                                int8_t *ap = (int8_t *)&a;
                                int8_t *bp = (int8_t *)&b;
                                return ap[0] * bp[0] + ap[1] * bp[1] + ap[2] * bp[2] + ap[3] * bp[3];
                            };

                            int sumi =
                                dp4a(vl[4 * iir + 0], u[ui_base + 0]) * sc0 +
                                dp4a(vl[4 * iir + 1], u[ui_base + 1]) * sc1 +
                                dp4a(vl[4 * iir + 2], u[ui_base + 2]) * sc2 +
                                dp4a(vl[4 * iir + 3], u[ui_base + 3]) * sc3 +
                                dp4a(vh[4 * iir + 0], u[ui_base + 4]) * sc0 +
                                dp4a(vh[4 * iir + 1], u[ui_base + 5]) * sc1 +
                                dp4a(vh[4 * iir + 2], u[ui_base + 6]) * sc2 +
                                dp4a(vh[4 * iir + 3], u[ui_base + 7]) * sc3;

                            dot_sum += sumi * df[iir];
                        }
                        sum[iy][ix] += d * dot_sum;
                    }
                }
            }

            // ========== BARRIER ==========
            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    // Store results
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + i_offset;
        if (col_dst >= ncols_y) continue;

        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + k + i;
            if (row_dst >= nrows_dst) continue;

            results[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE][j / nwarps];
        }
    }
}

int main() {
    printf("=== Test 5: SLM barrier test ===\n");
    printf("Tests work-group barrier synchronization during tile load/compute cycles\n\n");

    // Small test case: 128 rows, 64 columns, 1 block per row
    const int nrows = 128;
    const int ncols = QK_K;
    const int ncols_y = 64;
    const int blocks_per_row_x = 1;
    const int nblocks = nrows * blocks_per_row_x;
    const int blocks_per_col_y = ncols / QK8_1;

    // SoA layout
    size_t qh_offset = nblocks * (QK_K / 2);
    size_t scales_offset = qh_offset + nblocks * (QK_K / 4);
    size_t d_offset = scales_offset + nblocks * (QK_K / 16);
    size_t soa_size = d_offset + nblocks * sizeof(sycl::half);

    printf("Test dimensions: rows=%d, cols=%d, ncols_y=%d\n", nrows, ncols, ncols_y);
    printf("SoA size: %zu bytes\n", soa_size);

    // Allocate and initialize data
    uint8_t *soa_data = (uint8_t *)malloc(soa_size);
    memset(soa_data, 0, soa_size);

    // Fill X (SoA) with pattern
    for (int b = 0; b < nblocks; b++) {
        // ql: all 0x11 (gives ql0=0x01010101, ql1=0x01010101)
        memset(soa_data + b * (QK_K / 2), 0x11, QK_K / 2);
        // qh: all 0
        memset(soa_data + qh_offset + b * (QK_K / 4), 0, QK_K / 4);
        // scales: all 1
        memset(soa_data + scales_offset + b * (QK_K / 16), 1, QK_K / 16);
        // d: 1.0
        ((sycl::half *)(soa_data + d_offset))[b] = sycl::half(1.0f);
    }

    // Allocate Y (block_q8_1)
    int nblocks_y = ncols_y * blocks_per_col_y;
    block_q8_1 *y_data = (block_q8_1 *)malloc(nblocks_y * sizeof(block_q8_1));
    for (int b = 0; b < nblocks_y; b++) {
        y_data[b].ds[0] = sycl::half(1.0f);
        y_data[b].ds[1] = sycl::half(0.0f);
        for (int i = 0; i < QK8_1; i++) {
            y_data[b].qs[i] = 1;
        }
    }

    // Result buffer
    float *results = (float *)malloc(nrows * ncols_y * sizeof(float));
    memset(results, 0, nrows * ncols_y * sizeof(float));

    printf("\nRunning GPU kernel with barriers...\n");

    try {
        sycl::queue q{sycl::gpu_selector_v};
        printf("GPU: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        // Allocate GPU memory
        uint8_t *d_soa = sycl::malloc_device<uint8_t>(soa_size, q);
        block_q8_1 *d_y = sycl::malloc_device<block_q8_1>(nblocks_y, q);
        float *d_results = sycl::malloc_device<float>(nrows * ncols_y, q);

        q.memcpy(d_soa, soa_data, soa_size).wait();
        q.memcpy(d_y, y_data, nblocks_y * sizeof(block_q8_1)).wait();
        q.memset(d_results, 0, nrows * ncols_y * sizeof(float)).wait();

        // Launch kernel with proper work-group structure
        // Work-group: (1, nwarps, WARP_SIZE) = (1, 4, 32)
        // Grid: (1, ncols_y/mmq_x, nrows/mmq_y) = (1, 1, 1) for this test size
        sycl::range<3> local_range(1, nwarps, WARP_SIZE);
        sycl::range<3> global_range(1, nwarps, WARP_SIZE);  // Single work-group

        q.submit([&](sycl::handler &h) {
            // Allocate shared local memory
            sycl::local_accessor<int, 1> tile_x_ql_acc(sycl::range<1>(tile_x_ql_size), h);
            sycl::local_accessor<float, 1> tile_x_dmf_acc(sycl::range<1>(tile_x_dm_size), h);
            sycl::local_accessor<int, 1> tile_x_sc_acc(sycl::range<1>(tile_x_sc_size), h);
            sycl::local_accessor<int, 1> tile_y_qs_acc(sycl::range<1>(tile_y_qs_size), h);
            sycl::local_accessor<float, 1> tile_y_df_acc(sycl::range<1>(tile_y_ds_size), h);

            h.parallel_for(
                sycl::nd_range<3>(global_range, local_range),
                [=](sycl::nd_item<3> item) {
                    barrier_test_kernel(
                        d_soa, qh_offset, scales_offset, d_offset,
                        d_y, d_results,
                        blocks_per_row_x, blocks_per_col_y,
                        nrows, ncols_y, nrows, 0, item,
                        tile_x_ql_acc.get_pointer(),
                        tile_x_dmf_acc.get_pointer(),
                        tile_x_sc_acc.get_pointer(),
                        tile_y_qs_acc.get_pointer(),
                        tile_y_df_acc.get_pointer());
                });
        }).wait();

        // Copy results back
        q.memcpy(results, d_results, nrows * ncols_y * sizeof(float)).wait();

        // Analyze results
        printf("\nAnalyzing results...\n");

        int nan_count = 0;
        int inf_count = 0;
        int zero_count = 0;
        float min_val = 1e30f, max_val = -1e30f;

        for (int i = 0; i < nrows * ncols_y; i++) {
            if (std::isnan(results[i])) nan_count++;
            else if (std::isinf(results[i])) inf_count++;
            else if (results[i] == 0.0f) zero_count++;
            else {
                if (results[i] < min_val) min_val = results[i];
                if (results[i] > max_val) max_val = results[i];
            }
        }

        printf("Results: NaN=%d, Inf=%d, Zero=%d, Valid range=[%f, %f]\n",
               nan_count, inf_count, zero_count, min_val, max_val);

        // Sample results
        printf("\nSample results (first 8):\n");
        for (int i = 0; i < std::min(8, nrows * ncols_y); i++) {
            printf("  [%d]: %f\n", i, results[i]);
        }

        bool pass = (nan_count == 0 && inf_count == 0);
        if (!pass) {
            printf("\nFAILURE: Found NaN/Inf values - barrier synchronization may be broken!\n");
        } else if (zero_count == nrows * ncols_y) {
            printf("\nWARNING: All results are zero - computation may not be happening\n");
            pass = false;
        }

        printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

        // Cleanup
        sycl::free(d_soa, q);
        sycl::free(d_y, q);
        sycl::free(d_results, q);

    } catch (sycl::exception &e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }

    free(soa_data);
    free(y_data);
    free(results);

    return 0;
}
