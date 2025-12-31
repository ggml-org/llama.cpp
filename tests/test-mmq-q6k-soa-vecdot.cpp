// Test 3: vec_dot with known tiles test for Q6_K SoA kernel
// Pre-fills X and Y tiles with known values, runs vec_dot, verifies output
// This isolates whether the bug is in the dot product computation

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

// vec_dot_q6_K_q8_1_impl_mmq implementation (from mmq.cpp)
static inline int dpct_dp4a(int a, int b, int c) {
    int8_t *a_ptr = (int8_t *)&a;
    int8_t *b_ptr = (int8_t *)&b;
    c += a_ptr[0] * b_ptr[0];
    c += a_ptr[1] * b_ptr[1];
    c += a_ptr[2] * b_ptr[2];
    c += a_ptr[3] * b_ptr[3];
    return c;
}

static inline float vec_dot_q6_K_q8_1_impl_mmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const int8_t *__restrict__ sc,
    const float &d, const float *__restrict__ df) {

    float sum = 0.0f;

    for (int i = 0; i < QR6_K; ++i) {
        const int sc0 = sc[4 * i + 0];
        const int sc1 = sc[4 * i + 1];
        const int sc2 = sc[4 * i + 2];
        const int sc3 = sc[4 * i + 3];

        const int vl0 = vl[4 * i + 0];
        const int vl1 = vl[4 * i + 1];
        const int vl2 = vl[4 * i + 2];
        const int vl3 = vl[4 * i + 3];

        const int vh0 = vh[4 * i + 0];
        const int vh1 = vh[4 * i + 1];
        const int vh2 = vh[4 * i + 2];
        const int vh3 = vh[4 * i + 3];

        // For QR6_K=2, u indexing: i=0 uses u[0-15], i=1 uses u[16-31]
        const int ui_base = i * 16;
        const int u0 = u[ui_base + 0];
        const int u1 = u[ui_base + 1];
        const int u2 = u[ui_base + 2];
        const int u3 = u[ui_base + 3];
        const int u4 = u[ui_base + 4];
        const int u5 = u[ui_base + 5];
        const int u6 = u[ui_base + 6];
        const int u7 = u[ui_base + 7];

        const int sumi =
            dpct_dp4a(vl0, u0, 0) * sc0 + dpct_dp4a(vl1, u1, 0) * sc1 +
            dpct_dp4a(vl2, u2, 0) * sc2 + dpct_dp4a(vl3, u3, 0) * sc3 +
            dpct_dp4a(vh0, u4, 0) * sc0 + dpct_dp4a(vh1, u5, 0) * sc1 +
            dpct_dp4a(vh2, u6, 0) * sc2 + dpct_dp4a(vh3, u7, 0) * sc3;

        sum += sumi * df[i];
    }

    return d * sum;
}

// vec_dot_q6_K_q8_1_mul_mat (adapted from mmq.cpp)
static float vec_dot_q6_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const float *__restrict__ x_dmf,
    const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const float *__restrict__ y_df,
    const int i, const int j, const int k) {

    const int8_t *sc = ((const int8_t *)&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k / 8]);

    const int index_x = i * (QR6_K * MMQ_TILE_NE_K + 1) + QR6_K * k;

    const int ky = QR6_K * k;
    const int index_y = j * WARP_SIZE + ky % WARP_SIZE;
    const int index_y_ds = j * (WARP_SIZE / QI8_1) + (ky % WARP_SIZE) / QI8_1;

    // For Q6_K, we need 8 int values from x_ql (4 vl + 4 vh per iteration)
    // But the function expects vl and vh separately
    const int *vl = &x_ql[index_x];
    const int *vh = &x_ql[index_x + 4];

    return vec_dot_q6_K_q8_1_impl_mmq(vl, vh, &y_qs[index_y], sc,
                                       x_dmf[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K],
                                       &y_df[index_y_ds]);
}

// GPU kernel that runs vec_dot for specific (i, j, k) values
void vec_dot_test_kernel(
    const int *__restrict__ x_ql,
    const float *__restrict__ x_dmf,
    const int *__restrict__ x_sc,
    const int *__restrict__ y_qs,
    const float *__restrict__ y_df,
    float *__restrict__ results,
    const int num_tests,
    sycl::nd_item<1> item)
{
    int tid = item.get_global_id(0);
    if (tid >= num_tests) return;

    // Test different (i, j, k) combinations
    // i from 0 to mmq_y-1, j from 0 to mmq_x-1, k from 0 to MMQ_TILE_NE_K/QR6_K-1
    const int k_max = MMQ_TILE_NE_K / QR6_K;  // 16

    int i = tid / (mmq_x * k_max);
    int j = (tid / k_max) % mmq_x;
    int k = tid % k_max;

    results[tid] = vec_dot_q6_K_q8_1_mul_mat(x_ql, x_dmf, x_sc, y_qs, y_df, i, j, k);
}

int main() {
    printf("=== Test 3: vec_dot with known tiles ===\n");
    printf("Tests if vec_dot_q6_K_q8_1_mul_mat produces correct results\n\n");

    // Allocate tiles
    int *x_ql = (int *)malloc(tile_x_ql_size * sizeof(int));
    float *x_dmf = (float *)malloc(tile_x_dm_size * sizeof(float));
    int *x_sc = (int *)malloc(tile_x_sc_size * sizeof(int));
    int *y_qs = (int *)malloc(tile_y_qs_size * sizeof(int));
    float *y_df = (float *)malloc(tile_y_ds_size * sizeof(float));

    // Initialize with simple known values for easy verification
    memset(x_ql, 0, tile_x_ql_size * sizeof(int));
    memset(x_dmf, 0, tile_x_dm_size * sizeof(float));
    memset(x_sc, 0, tile_x_sc_size * sizeof(int));
    memset(y_qs, 0, tile_y_qs_size * sizeof(int));
    memset(y_df, 0, tile_y_ds_size * sizeof(float));

    // Fill x_ql with test pattern
    // Each int is 4 signed bytes. Use small values to avoid overflow
    for (int i = 0; i < mmq_y; i++) {
        for (int kq = 0; kq < 2 * MMQ_TILE_NE_K; kq++) {
            int idx = i * (2 * MMQ_TILE_NE_K + 1) + kq;
            // Value: 4 bytes each = (1, 2, 3, 4) for simplicity
            int8_t bytes[4] = {1, 2, 3, 4};
            memcpy(&x_ql[idx], bytes, 4);
        }
    }

    // Fill x_dm (d values): all 1.0
    for (int i = 0; i < tile_x_dm_size; i++) {
        x_dmf[i] = 1.0f;
    }

    // Fill x_sc (scales): pattern that allows easy verification
    for (int i = 0; i < mmq_y; i++) {
        for (int s = 0; s < MMQ_TILE_NE_K / 8; s++) {
            int idx = i * (MMQ_TILE_NE_K / 8) + i / 8 + s;
            // 4 scale bytes, each = 1
            int8_t bytes[4] = {1, 1, 1, 1};
            memcpy(&x_sc[idx], bytes, 4);
        }
    }

    // Fill y_qs: pattern that gives predictable dot products
    for (int j = 0; j < mmq_x; j++) {
        for (int q = 0; q < WARP_SIZE; q++) {
            int idx = j * WARP_SIZE + q;
            // Each int = 4 bytes (1, 1, 1, 1)
            int8_t bytes[4] = {1, 1, 1, 1};
            memcpy(&y_qs[idx], bytes, 4);
        }
    }

    // Fill y_df: all 1.0
    for (int i = 0; i < tile_y_ds_size; i++) {
        y_df[i] = 1.0f;
    }

    // Test a subset of (i, j, k) combinations
    const int num_test_i = 4;
    const int num_test_j = 4;
    const int num_test_k = 4;
    const int num_tests = num_test_i * num_test_j * num_test_k;

    float *cpu_results = (float *)malloc(num_tests * sizeof(float));

    // Run CPU reference
    printf("Running CPU reference...\n");
    int test_idx = 0;
    for (int ti = 0; ti < num_test_i; ti++) {
        int i = ti * (mmq_y / num_test_i);
        for (int tj = 0; tj < num_test_j; tj++) {
            int j = tj * (mmq_x / num_test_j);
            for (int tk = 0; tk < num_test_k; tk++) {
                int k = tk * (MMQ_TILE_NE_K / QR6_K / num_test_k);
                cpu_results[test_idx] = vec_dot_q6_K_q8_1_mul_mat(
                    x_ql, x_dmf, x_sc, y_qs, y_df, i, j, k);
                test_idx++;
            }
        }
    }

    // Print some CPU results
    printf("CPU results sample (first 8):\n");
    for (int i = 0; i < std::min(8, num_tests); i++) {
        printf("  [%d]: %f\n", i, cpu_results[i]);
    }

    // Check for NaN in CPU results
    int cpu_nan = 0;
    for (int i = 0; i < num_tests; i++) {
        if (std::isnan(cpu_results[i]) || std::isinf(cpu_results[i])) cpu_nan++;
    }
    if (cpu_nan > 0) {
        printf("WARNING: CPU produced %d NaN/Inf values!\n", cpu_nan);
    }

    // GPU version
    printf("\nRunning GPU version...\n");

    try {
        sycl::queue q{sycl::gpu_selector_v};
        printf("GPU: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        // Allocate GPU memory
        int *d_x_ql = sycl::malloc_device<int>(tile_x_ql_size, q);
        float *d_x_dmf = sycl::malloc_device<float>(tile_x_dm_size, q);
        int *d_x_sc = sycl::malloc_device<int>(tile_x_sc_size, q);
        int *d_y_qs = sycl::malloc_device<int>(tile_y_qs_size, q);
        float *d_y_df = sycl::malloc_device<float>(tile_y_ds_size, q);
        float *d_results = sycl::malloc_device<float>(num_tests, q);

        // Copy tiles to GPU
        q.memcpy(d_x_ql, x_ql, tile_x_ql_size * sizeof(int)).wait();
        q.memcpy(d_x_dmf, x_dmf, tile_x_dm_size * sizeof(float)).wait();
        q.memcpy(d_x_sc, x_sc, tile_x_sc_size * sizeof(int)).wait();
        q.memcpy(d_y_qs, y_qs, tile_y_qs_size * sizeof(int)).wait();
        q.memcpy(d_y_df, y_df, tile_y_ds_size * sizeof(float)).wait();

        // Create GPU-side test kernel
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(num_tests), [=](sycl::item<1> item) {
                int tid = item.get_id(0);
                if (tid >= num_tests) return;

                // Same (i, j, k) mapping as CPU
                const int k_max = MMQ_TILE_NE_K / QR6_K;
                int ti = tid / (num_test_j * num_test_k);
                int tj = (tid / num_test_k) % num_test_j;
                int tk = tid % num_test_k;

                int i = ti * (mmq_y / num_test_i);
                int j = tj * (mmq_x / num_test_j);
                int k = tk * (k_max / num_test_k);

                const int8_t *sc = ((const int8_t *)&d_x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k / 8]);
                const int index_x = i * (QR6_K * MMQ_TILE_NE_K + 1) + QR6_K * k;
                const int ky = QR6_K * k;
                const int index_y = j * WARP_SIZE + ky % WARP_SIZE;
                const int index_y_ds = j * (WARP_SIZE / QI8_1) + (ky % WARP_SIZE) / QI8_1;

                // Inline vec_dot computation
                const int *vl = &d_x_ql[index_x];
                const int *vh = &d_x_ql[index_x + 4];
                const int *u = &d_y_qs[index_y];
                float d = d_x_dmf[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K];
                const float *df = &d_y_df[index_y_ds];

                float sum = 0.0f;
                for (int ir = 0; ir < QR6_K; ++ir) {
                    const int sc0 = sc[4 * ir + 0];
                    const int sc1 = sc[4 * ir + 1];
                    const int sc2 = sc[4 * ir + 2];
                    const int sc3 = sc[4 * ir + 3];

                    const int ui_base = ir * 16;

                    auto dp4a = [](int a, int b) {
                        int8_t *ap = (int8_t *)&a;
                        int8_t *bp = (int8_t *)&b;
                        return ap[0]*bp[0] + ap[1]*bp[1] + ap[2]*bp[2] + ap[3]*bp[3];
                    };

                    int sumi =
                        dp4a(vl[4*ir+0], u[ui_base+0]) * sc0 +
                        dp4a(vl[4*ir+1], u[ui_base+1]) * sc1 +
                        dp4a(vl[4*ir+2], u[ui_base+2]) * sc2 +
                        dp4a(vl[4*ir+3], u[ui_base+3]) * sc3 +
                        dp4a(vh[4*ir+0], u[ui_base+4]) * sc0 +
                        dp4a(vh[4*ir+1], u[ui_base+5]) * sc1 +
                        dp4a(vh[4*ir+2], u[ui_base+6]) * sc2 +
                        dp4a(vh[4*ir+3], u[ui_base+7]) * sc3;

                    sum += sumi * df[ir];
                }

                d_results[tid] = d * sum;
            });
        }).wait();

        // Copy results back
        float *gpu_results = (float *)malloc(num_tests * sizeof(float));
        q.memcpy(gpu_results, d_results, num_tests * sizeof(float)).wait();

        // Compare
        printf("\nComparing CPU vs GPU...\n");
        int mismatches = 0;
        int first_mismatch = -1;
        float max_error = 0.0f;

        for (int i = 0; i < num_tests; i++) {
            float err = fabsf(cpu_results[i] - gpu_results[i]);
            if (err > max_error) max_error = err;
            if (err > 1e-4f || std::isnan(gpu_results[i])) {
                if (first_mismatch < 0) first_mismatch = i;
                mismatches++;
            }
        }

        printf("Mismatches: %d/%d\n", mismatches, num_tests);
        printf("Max error: %e\n", max_error);

        if (first_mismatch >= 0) {
            printf("First mismatch at [%d]: CPU=%f, GPU=%f\n",
                   first_mismatch, cpu_results[first_mismatch], gpu_results[first_mismatch]);
        }

        // Check for NaN
        int nan_count = 0;
        for (int i = 0; i < num_tests; i++) {
            if (std::isnan(gpu_results[i]) || std::isinf(gpu_results[i])) nan_count++;
        }
        if (nan_count > 0) {
            printf("WARNING: Found %d NaN/Inf values in GPU results!\n", nan_count);
        }

        bool pass = (mismatches == 0 && nan_count == 0);
        printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

        // Cleanup
        sycl::free(d_x_ql, q);
        sycl::free(d_x_dmf, q);
        sycl::free(d_x_sc, q);
        sycl::free(d_y_qs, q);
        sycl::free(d_y_df, q);
        sycl::free(d_results, q);
        free(gpu_results);

    } catch (sycl::exception &e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }

    free(x_ql);
    free(x_dmf);
    free(x_sc);
    free(y_qs);
    free(y_df);
    free(cpu_results);

    return 0;
}
