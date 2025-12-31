// Unit test for Q6_K MMVQ kernel - compares GPU vs CPU reference
// Build: icpx -fsycl -O2 -I../ggml/include -I../ggml/src tests/test-mmvq-q6k-gpu.cpp -o test-mmvq-q6k-gpu
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./test-mmvq-q6k-gpu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>

// Constants from ggml-common.h
#define QK_K 256
#define QK8_1 32
#define QI6_K 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))  // = 8
#define QR6_K 2
#define VDR_Q6_K_Q8_1_MMVQ 1
#define WARP_SIZE 16
#define GGML_SYCL_MMV_Y 1

// Block structures
typedef sycl::half ggml_half;

typedef struct {
    uint8_t ql[QK_K/2];      // 128 bytes - lower 4 bits
    uint8_t qh[QK_K/4];      // 64 bytes - upper 2 bits
    int8_t  scales[QK_K/16]; // 16 bytes
    ggml_half d;             // 2 bytes
} block_q6_K;

typedef struct {
    ggml_half d;
    ggml_half s;  // sum
    int8_t qs[QK8_1];
} block_q8_1;

// Helper functions (from vecdotq.hpp)
static inline int get_int_from_int8_aligned(const int8_t* x8, const int i32) {
    return *((const int*)(x8 + sizeof(int) * i32));
}

static inline int get_int_from_uint8(const uint8_t* x8, const int i32) {
    const uint16_t* x16 = (const uint16_t*)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0];
    x32 |= (int)x16[1] << 16;
    return x32;
}

// CPU reference implementation of vec_dot_q6_K_q8_1
float cpu_vec_dot_q6_K_q8_1(const block_q6_K* bq6_K, const block_q8_1* bq8_1, int iqs) {
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

        // Extract 4 6-bit values from vl and vh
        const int8_t* vl_bytes = (const int8_t*)&vl;
        const int8_t* vh_bytes = (const int8_t*)&vh;
        const int8_t* u_bytes = (const int8_t*)&u;

        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            // Reconstruct 6-bit value: low 4 bits from vl, high 2 bits from vh
            int q = (vl_bytes[j] & 0xF) | ((vh_bytes[j] & 0x3) << 4);
            q -= 32;  // Q6_K uses signed values centered at 32
            sumi += q * u_bytes[j];
        }

        sum += d8 * float(bq6_K->d) * sumi * scales[2*i];
    }
    return sum;
}

// Full CPU reference for one row dot product
float cpu_row_dot_q6_K(const block_q6_K* x_row, const block_q8_1* y, int ncols) {
    const int blocks_per_row = ncols / QK_K;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const block_q6_K* bx = &x_row[ib];
        const block_q8_1* by = &y[ib * (QK_K / QK8_1)];  // 8 Q8_1 blocks per Q6_K block

        // Sum over all iqs values (0 to QI6_K-1)
        for (int iqs = 0; iqs < QI6_K; ++iqs) {
            sum += cpu_vec_dot_q6_K_q8_1(bx, by, iqs);
        }
    }
    return sum;
}

// Simpler CPU reference - direct dequantization
float cpu_row_dot_q6_K_simple(const block_q6_K* x_row, const float* y, int ncols) {
    const int blocks_per_row = ncols / QK_K;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const block_q6_K* bx = &x_row[ib];
        const float d = float(bx->d);

        for (int k = 0; k < QK_K; ++k) {
            // Decode Q6_K value
            const int ql_idx = k / 2;
            const int qh_idx = k / 4;
            const int scale_idx = k / 16;

            // Get low 4 bits
            int q_low;
            if (k % 2 == 0) {
                q_low = bx->ql[ql_idx] & 0xF;
            } else {
                q_low = bx->ql[ql_idx] >> 4;
            }

            // Get high 2 bits
            int shift = (k % 4) * 2;
            int q_high = (bx->qh[qh_idx] >> shift) & 0x3;

            // Combine to 6-bit value and center
            int q = q_low | (q_high << 4);
            q -= 32;

            // Apply scale and super-block scale
            float w = d * bx->scales[scale_idx] * q;
            sum += w * y[ib * QK_K + k];
        }
    }
    return sum;
}

// GPU kernel - copied from mmvq.cpp mul_mat_vec_q template
template<int qk, int qi, int vdr>
static void kernel_mul_mat_vec_q6_K(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    const block_q6_K* x = (const block_q6_K*)vx;
    const block_q8_1* y = (const block_q8_1*)vy;

    constexpr int qi_div_vdr = qi / vdr;
    const int lane_id = item_ct1.get_local_id(2);
    const int base_iqs = vdr * (lane_id % qi_div_vdr);
    const int row_offset = row * blocks_per_row;

    float tmp = 0.0f;

    for (int i = lane_id / qi_div_vdr; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

        for (int elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;

            // Inline vec_dot_q6_K_q8_1
            const block_q6_K* bq6_K = &x[ibx];
            const block_q8_1* bq8_1 = &y[iby];

            const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
            const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
            const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

            const int vl = get_int_from_uint8(bq6_K->ql, iqs);
            const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

            const int8_t* scales = bq6_K->scales + scale_offset;

            int u[QR6_K];
            float d8[QR6_K];

            for (int j = 0; j < QR6_K; ++j) {
                u[j] = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*j].qs, iqs % QI8_1);
                d8[j] = float(bq8_1[bq8_offset + 2*j].d);
            }

            // vec_dot_q6_K_q8_1_impl_mmvq
            float sum_local = 0.0f;
            for (int j = 0; j < QR6_K; ++j) {
                const int8_t* vl_bytes = (const int8_t*)&vl;
                const int8_t* vh_bytes = (const int8_t*)&vh;
                const int8_t* u_bytes = (const int8_t*)&u[j];

                int sumi = 0;
                for (int k = 0; k < 4; ++k) {
                    int q = (vl_bytes[k] & 0xF) | ((vh_bytes[k] & 0x3) << 4);
                    q -= 32;
                    sumi += q * u_bytes[k];
                }
                sum_local += d8[j] * float(bq6_K->d) * sumi * scales[2*j];
            }
            tmp += sum_local;
        }
    }

    // Warp reduction
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

// Quantize float to Q8_1
void quantize_row_q8_1(const float* x, block_q8_1* y, int k) {
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; ++i) {
        float max_abs = 0.0f;
        float sum = 0.0f;

        for (int j = 0; j < QK8_1; ++j) {
            float v = x[i * QK8_1 + j];
            max_abs = std::max(max_abs, std::abs(v));
            sum += v;
        }

        const float d = max_abs / 127.0f;
        const float id = d > 0 ? 1.0f / d : 0.0f;

        y[i].d = ggml_half(d);
        y[i].s = ggml_half(sum);

        for (int j = 0; j < QK8_1; ++j) {
            y[i].qs[j] = (int8_t)roundf(x[i * QK8_1 + j] * id);
        }
    }
}

int main() {
    printf("Q6_K MMVQ GPU vs CPU Test\n");
    printf("=========================\n\n");

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Small test case for debugging
    const int nrows = 4;
    const int ncols = QK_K;  // Just one Q6_K block per row
    const int blocks_per_row = ncols / QK_K;
    const int total_blocks = nrows * blocks_per_row;
    const int y_blocks = ncols / QK8_1;  // Y is shared across all rows (8 Q8_1 blocks for ncols=256)

    printf("Test config: %d rows, %d cols, %d Q6_K blocks, %d Q8_1 blocks\n\n",
           nrows, ncols, total_blocks, y_blocks);

    // Random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> ql_dist(0, 255);
    std::uniform_int_distribution<int> qh_dist(0, 255);
    std::uniform_int_distribution<int> sc_dist(-16, 15);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

    // Allocate host memory
    std::vector<block_q6_K> h_x(total_blocks);
    std::vector<float> h_y_float(ncols);
    std::vector<block_q8_1> h_y(y_blocks);
    std::vector<float> h_dst_gpu(nrows);
    std::vector<float> h_dst_cpu(nrows);

    // Initialize Q6_K blocks with random data
    for (int b = 0; b < total_blocks; ++b) {
        h_x[b].d = ggml_half(scale_dist(rng));
        for (int i = 0; i < QK_K/2; ++i) h_x[b].ql[i] = ql_dist(rng);
        for (int i = 0; i < QK_K/4; ++i) h_x[b].qh[i] = qh_dist(rng);
        for (int i = 0; i < QK_K/16; ++i) h_x[b].scales[i] = sc_dist(rng);
    }

    // Initialize Y with random floats
    for (int i = 0; i < ncols; ++i) {
        h_y_float[i] = y_dist(rng);
    }

    // Quantize Y to Q8_1
    quantize_row_q8_1(h_y_float.data(), h_y.data(), ncols);

    // Allocate device memory
    auto d_x = sycl::malloc_device<block_q6_K>(total_blocks, q);
    auto d_y = sycl::malloc_device<block_q8_1>(y_blocks, q);
    auto d_dst = sycl::malloc_device<float>(nrows, q);

    // Copy to device
    q.memcpy(d_x, h_x.data(), total_blocks * sizeof(block_q6_K));
    q.memcpy(d_y, h_y.data(), y_blocks * sizeof(block_q8_1));
    q.memset(d_dst, 0, nrows * sizeof(float));
    q.wait();

    // Launch GPU kernel
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    sycl::range<3> grid(1, 1, block_num_y);
    sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                kernel_mul_mat_vec_q6_K<QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ>(
                    d_x, d_y, d_dst, ncols, nrows, item);
            });
    }).wait();

    // Copy result back
    q.memcpy(h_dst_gpu.data(), d_dst, nrows * sizeof(float)).wait();

    // Compute CPU reference using Q8_1 quantized Y (same as GPU)
    printf("Computing CPU reference...\n");
    for (int row = 0; row < nrows; ++row) {
        h_dst_cpu[row] = cpu_row_dot_q6_K(&h_x[row * blocks_per_row], h_y.data(), ncols);
    }

    // Compare results
    printf("\nResults comparison:\n");
    printf("%-6s %-15s %-15s %-12s %-10s\n", "Row", "GPU", "CPU", "Diff", "RelErr%");
    printf("---------------------------------------------------------------\n");

    float max_rel_err = 0.0f;
    int fail_count = 0;

    for (int i = 0; i < nrows; ++i) {
        float diff = h_dst_gpu[i] - h_dst_cpu[i];
        float rel_err = std::abs(h_dst_cpu[i]) > 1e-6f ? std::abs(diff / h_dst_cpu[i]) * 100.0f : 0.0f;
        max_rel_err = std::max(max_rel_err, rel_err);

        const char* status = rel_err > 1.0f ? "FAIL" : "OK";
        if (rel_err > 1.0f) fail_count++;

        printf("%-6d %-15.6f %-15.6f %-12.2e %-10.4f %s\n",
               i, h_dst_gpu[i], h_dst_cpu[i], diff, rel_err, status);
    }

    printf("\nMax relative error: %.4f%%\n", max_rel_err);
    printf("Failed rows: %d / %d\n", fail_count, nrows);

    // Debug: print first Q6_K block details
    printf("\n--- Debug: First Q6_K block ---\n");
    printf("d = %.6f\n", float(h_x[0].d));
    printf("scales[0..3] = %d, %d, %d, %d\n",
           h_x[0].scales[0], h_x[0].scales[1], h_x[0].scales[2], h_x[0].scales[3]);
    printf("ql[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           h_x[0].ql[0], h_x[0].ql[1], h_x[0].ql[2], h_x[0].ql[3],
           h_x[0].ql[4], h_x[0].ql[5], h_x[0].ql[6], h_x[0].ql[7]);
    printf("qh[0..3] = %02x %02x %02x %02x\n",
           h_x[0].qh[0], h_x[0].qh[1], h_x[0].qh[2], h_x[0].qh[3]);

    // Debug: print first Q8_1 block
    printf("\n--- Debug: First Q8_1 block ---\n");
    printf("d = %.6f, s = %.6f\n", float(h_y[0].d), float(h_y[0].s));
    printf("qs[0..7] = %d %d %d %d %d %d %d %d\n",
           h_y[0].qs[0], h_y[0].qs[1], h_y[0].qs[2], h_y[0].qs[3],
           h_y[0].qs[4], h_y[0].qs[5], h_y[0].qs[6], h_y[0].qs[7]);

    // Cleanup
    sycl::free(d_x, q);
    sycl::free(d_y, q);
    sycl::free(d_dst, q);

    return fail_count > 0 ? 1 : 0;
}
