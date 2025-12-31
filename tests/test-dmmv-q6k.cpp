// Test Q6_K DMMV kernel directly on GPU
// This test exercises the actual dequantize_mul_mat_vec_q6_k kernel
// that is used during token generation (decode phase)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// Q6_K block structure (must match ggml)
#define QK_K 256
#define K_QUANTS_PER_ITERATION 2
#define QK_WARP_SIZE 32

// Use sycl::half for GPU-compatible FP16
typedef struct {
    uint8_t ql[QK_K/2];     // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];     // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits (16 bytes)
    sycl::half d;           // super-block scale (2 bytes)
} block_q6_K;

static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");

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

// ========== GPU KERNEL (copied from dmmv.cpp for direct testing) ==========

static void dequantize_mul_mat_vec_q6_k_kernel(
    const void * __restrict__ vx,
    const float * __restrict__ yy,
    float * __restrict__ dst,
    const int ncols,
    int nrows,
    const sycl::nd_item<3> &item_ct1
) {
    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

    const int tid = item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION;
    const int ix  = item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION;

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;
    const int in = tid - step*im;

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;
    const int is = 0;
#else
    const int l0 = 4 * in;
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = static_cast<float>(x[i].d);

#if K_QUANTS_PER_ITERATION == 1
        int8_t q0 = (int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32;
        int8_t q1 = (int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32;
        int8_t q2 = (int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32;
        int8_t q3 = (int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32;
        int8_t q4 = (int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32;
        int8_t q5 = (int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32;
        int8_t q6 = (int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32;
        int8_t q7 = (int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32;

        float sum = y[ 0] * s[0] * d * q0
                  + y[16] * s[1] * d * q1
                  + y[32] * s[2] * d * q2
                  + y[48] * s[3] * d * q3
                  + y[64] * s[4] * d * q4
                  + y[80] * s[5] * d * q5
                  + y[96] * s[6] * d * q6
                  +y[112] * s[7] * d * q7;

        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            int8_t q0 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q1 = (int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += y[l+ 0] * s[0] * d * q0
                 + y[l+32] * s[2] * d * q1
                 + y[l+64] * s[4] * d * q2
                 + y[l+96] * s[6] * d * q3;
        }
        tmp += sum;
#endif
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

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

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    printf("=== Q6_K DMMV Direct Kernel Test ===\n\n");

    // Test parameters
    const int ncols = 4096;  // Mistral hidden size
    const int nrows = 1;     // Single row for decode (DMMV scenario)
    const int nb = ncols / QK_K;

    printf("Test config: ncols=%d, nrows=%d, nb=%d, block_size=%zu bytes\n",
           ncols, nrows, nb, sizeof(block_q6_K));
    printf("K_QUANTS_PER_ITERATION=%d, QK_WARP_SIZE=%d\n\n", K_QUANTS_PER_ITERATION, QK_WARP_SIZE);

    // Allocate host data
    std::vector<float> x_f32(ncols * nrows);
    std::vector<float> y_f32(ncols);
    std::vector<block_q6_K> x_q6k(nb * nrows);
    std::vector<float> dst_cpu(nrows);
    std::vector<float> dst_gpu(nrows);

    // Initialize with reproducible random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < ncols * nrows; i++) {
        x_f32[i] = dist(rng);
    }
    for (int i = 0; i < ncols; i++) {
        y_f32[i] = dist(rng);
    }

    // Quantize to Q6_K
    printf("Quantizing %d values to Q6_K...\n", ncols * nrows);
    quantize_row_q6_K(x_f32.data(), x_q6k.data(), ncols * nrows);

    // CPU reference computation
    printf("Computing CPU reference...\n");
    for (int row = 0; row < nrows; row++) {
        dst_cpu[row] = vec_dot_q6_K_cpu(x_q6k.data() + row * nb, y_f32.data(), ncols);
    }
    printf("CPU result: %.6f\n\n", dst_cpu[0]);

    // GPU computation using SYCL directly
    printf("Initializing SYCL...\n");
    sycl::queue q(sycl::default_selector_v);
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Allocate device memory
    block_q6_K * d_x = sycl::malloc_device<block_q6_K>(nb * nrows, q);
    float * d_y = sycl::malloc_device<float>(ncols, q);
    float * d_dst = sycl::malloc_device<float>(nrows, q);

    // Copy data to device
    q.memcpy(d_x, x_q6k.data(), nb * nrows * sizeof(block_q6_K)).wait();
    q.memcpy(d_y, y_f32.data(), ncols * sizeof(float)).wait();
    q.memset(d_dst, 0, nrows * sizeof(float)).wait();

    // Launch kernel - matching dequantize_mul_mat_vec_q6_K_sycl dispatch
    printf("Launching Q6_K DMMV kernel...\n");
    const int ny = 2 / K_QUANTS_PER_ITERATION;  // 1 for K_QUANTS_PER_ITERATION=2
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);

    printf("Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           1, 1, block_num_y, 1, ny, QK_WARP_SIZE);

    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
                dequantize_mul_mat_vec_q6_k_kernel(d_x, d_y, d_dst, ncols, nrows, item_ct1);
            });
    }).wait();

    // Copy result back
    q.memcpy(dst_gpu.data(), d_dst, nrows * sizeof(float)).wait();
    printf("GPU result: %.6f\n\n", dst_gpu[0]);

    // Compare results
    float diff = fabsf(dst_cpu[0] - dst_gpu[0]);
    float rel_error = diff / (fabsf(dst_cpu[0]) + 1e-10f);

    printf("=== Results ===\n");
    printf("CPU:  %.6f\n", dst_cpu[0]);
    printf("GPU:  %.6f\n", dst_gpu[0]);
    printf("Diff: %.6f (%.4f%%)\n", diff, rel_error * 100.0f);

    bool pass = rel_error < 0.001f;  // 0.1% tolerance
    printf("\n%s (threshold: 0.1%%)\n", pass ? "PASS" : "FAIL");

    // Test multiple rows
    printf("\n=== Multi-row test (8 rows) ===\n");
    const int test_nrows = 8;
    std::vector<float> x_f32_multi(ncols * test_nrows);
    std::vector<block_q6_K> x_q6k_multi(nb * test_nrows);
    std::vector<float> dst_cpu_multi(test_nrows);
    std::vector<float> dst_gpu_multi(test_nrows);

    for (int i = 0; i < ncols * test_nrows; i++) {
        x_f32_multi[i] = dist(rng);
    }
    quantize_row_q6_K(x_f32_multi.data(), x_q6k_multi.data(), ncols * test_nrows);

    for (int row = 0; row < test_nrows; row++) {
        dst_cpu_multi[row] = vec_dot_q6_K_cpu(x_q6k_multi.data() + row * nb, y_f32.data(), ncols);
    }

    // Reallocate for multi-row
    sycl::free(d_x, q);
    sycl::free(d_dst, q);
    d_x = sycl::malloc_device<block_q6_K>(nb * test_nrows, q);
    d_dst = sycl::malloc_device<float>(test_nrows, q);

    q.memcpy(d_x, x_q6k_multi.data(), nb * test_nrows * sizeof(block_q6_K)).wait();
    q.memset(d_dst, 0, test_nrows * sizeof(float)).wait();

    const int block_num_y_multi = (test_nrows + ny - 1) / ny;
    const sycl::range<3> block_nums_multi(1, 1, block_num_y_multi);

    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums_multi * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
                dequantize_mul_mat_vec_q6_k_kernel(d_x, d_y, d_dst, ncols, test_nrows, item_ct1);
            });
    }).wait();

    q.memcpy(dst_gpu_multi.data(), d_dst, test_nrows * sizeof(float)).wait();

    bool multi_pass = true;
    for (int row = 0; row < test_nrows; row++) {
        float d = fabsf(dst_cpu_multi[row] - dst_gpu_multi[row]);
        float r = d / (fabsf(dst_cpu_multi[row]) + 1e-10f);
        printf("Row %d: CPU=%.4f GPU=%.4f Diff=%.6f (%.4f%%) %s\n",
               row, dst_cpu_multi[row], dst_gpu_multi[row], d, r * 100.0f,
               r < 0.001f ? "OK" : "FAIL");
        if (r >= 0.001f) multi_pass = false;
    }

    printf("\nMulti-row test: %s\n", multi_pass ? "PASS" : "FAIL");

    // Cleanup
    sycl::free(d_x, q);
    sycl::free(d_y, q);
    sycl::free(d_dst, q);

    bool overall = pass && multi_pass;
    printf("\n=== OVERALL: %s ===\n", overall ? "PASS" : "FAIL");
    return overall ? 0 : 1;
}
