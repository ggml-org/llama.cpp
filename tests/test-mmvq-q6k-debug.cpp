// Detailed debug test for Q6_K MMVQ - traces every step
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>

#define QK_K 256
#define QK8_1 32
#define QI6_K 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))  // = 8
#define QR6_K 2

typedef sycl::half ggml_half;

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

static inline int get_int_from_int8_aligned(const int8_t* x8, int i32) {
    return *((const int*)(x8 + sizeof(int) * i32));
}

static inline int get_int_from_uint8(const uint8_t* x8, int i32) {
    const uint16_t* x16 = (const uint16_t*)(x8 + sizeof(int) * i32);
    return x16[0] | ((int)x16[1] << 16);
}

// Trace a single vec_dot call
void trace_vec_dot_q6_K_q8_1(const block_q6_K* bq6_K, const block_q8_1* bq8_1, int iqs) {
    printf("\n=== iqs=%d ===\n", iqs);

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    printf("  bq8_offset=%d, scale_offset=%d, vh_shift=%d\n", bq8_offset, scale_offset, vh_shift);

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int qh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
    const int vh_raw = get_int_from_uint8(bq6_K->qh, qh_idx);
    const int vh = vh_raw >> vh_shift;

    printf("  vl=0x%08x (from ql[%d*4..%d*4+3])\n", vl, iqs, iqs);
    printf("  qh_idx=%d, vh_raw=0x%08x, vh=0x%08x\n", qh_idx, vh_raw, vh);

    const int8_t* scales = bq6_K->scales + scale_offset;
    printf("  scales at offset %d: [%d, %d, %d, %d]\n",
           scale_offset, scales[0], scales[1], scales[2], scales[3]);

    float total = 0.0f;
    for (int i = 0; i < QR6_K; ++i) {
        const int q8_block = bq8_offset + 2*i;
        printf("  i=%d: accessing bq8_1[%d]\n", i, q8_block);

        const int u = get_int_from_int8_aligned(bq8_1[q8_block].qs, iqs % QI8_1);
        const float d8 = float(bq8_1[q8_block].d);

        printf("    u=0x%08x (from qs[%d*4..]), d8=%.6f\n", u, iqs % QI8_1, d8);

        // Decode and compute
        const int8_t* vl_bytes = (const int8_t*)&vl;
        const int8_t* vh_bytes = (const int8_t*)&vh;
        const int8_t* u_bytes = (const int8_t*)&u;

        int sumi = 0;
        printf("    ");
        for (int j = 0; j < 4; ++j) {
            int q = (vl_bytes[j] & 0xF) | ((vh_bytes[j] & 0x3) << 4);
            q -= 32;
            int prod = q * u_bytes[j];
            sumi += prod;
            printf("q=%d*u=%d=%d  ", q, u_bytes[j], prod);
        }
        printf("\n    sumi=%d, scale=%d\n", sumi, scales[2*i]);

        float contrib = d8 * float(bq6_K->d) * sumi * scales[2*i];
        total += contrib;
        printf("    contrib = d8(%.4f) * d(%.4f) * sumi(%d) * scale(%d) = %.6f\n",
               d8, float(bq6_K->d), sumi, scales[2*i], contrib);
    }
    printf("  iqs=%d total = %.6f\n", iqs, total);
}

// Simpler CPU reference - direct dequantization
float cpu_simple_dot(const block_q6_K* bx, const float* y) {
    const float d = float(bx->d);
    float sum = 0.0f;

    printf("\n=== CPU Simple Dot ===\n");
    printf("super-block d = %.6f\n", d);

    for (int k = 0; k < 16; ++k) {  // Just first 16 elements for debug
        const int ql_idx = k / 2;
        const int qh_idx = k / 4;
        const int scale_idx = k / 16;

        int q_low = (k % 2 == 0) ? (bx->ql[ql_idx] & 0xF) : (bx->ql[ql_idx] >> 4);
        int shift = (k % 4) * 2;
        int q_high = (bx->qh[qh_idx] >> shift) & 0x3;
        int q = (q_low | (q_high << 4)) - 32;

        float w = d * bx->scales[scale_idx] * q;
        sum += w * y[k];

        printf("  k=%2d: ql_idx=%d qh_idx=%d shift=%d | q_low=%d q_high=%d q=%d | scale=%d | w=%.4f y=%.4f prod=%.4f\n",
               k, ql_idx, qh_idx, shift, q_low, q_high, q, bx->scales[scale_idx], w, y[k], w*y[k]);
    }
    printf("Sum (first 16) = %.6f\n", sum);
    return sum;
}

int main() {
    printf("Q6_K MMVQ Detailed Debug Test\n");
    printf("==============================\n\n");

    // Create one simple Q6_K block with known values
    block_q6_K bx;
    memset(&bx, 0, sizeof(bx));

    // Simple test: all ql = 0x55 (low nibbles = 5, high nibbles = 5)
    // This gives q_low = 5 for all elements
    for (int i = 0; i < QK_K/2; ++i) bx.ql[i] = 0x55;

    // qh = 0 -> q_high = 0 for all
    for (int i = 0; i < QK_K/4; ++i) bx.qh[i] = 0x00;

    // So q = 5 - 32 = -27 for all elements
    bx.d = ggml_half(1.0f);  // Super-block scale = 1.0
    for (int i = 0; i < QK_K/16; ++i) bx.scales[i] = 1;  // All sub-block scales = 1

    // Y vector: all 1.0
    std::vector<float> y_float(QK_K, 1.0f);

    // Expected: sum of 256 * (-27 * 1.0 * 1 * 1.0) = -6912
    printf("Expected result: 256 * (-27) = %d\n\n", 256 * (-27));

    // Quantize Y to Q8_1
    std::vector<block_q8_1> y_q8(QK_K / QK8_1);
    for (int b = 0; b < QK_K / QK8_1; ++b) {
        y_q8[b].d = ggml_half(1.0f / 127.0f);  // d such that 127 * d = 1.0
        y_q8[b].s = ggml_half(QK8_1);
        for (int i = 0; i < QK8_1; ++i) {
            y_q8[b].qs[i] = 127;  // All Y values = 127 * d = 1.0
        }
    }

    printf("Q6_K block:\n");
    printf("  d = %.6f\n", float(bx.d));
    printf("  ql[0..3] = %02x %02x %02x %02x\n", bx.ql[0], bx.ql[1], bx.ql[2], bx.ql[3]);
    printf("  qh[0..3] = %02x %02x %02x %02x\n", bx.qh[0], bx.qh[1], bx.qh[2], bx.qh[3]);
    printf("  scales[0..3] = %d %d %d %d\n", bx.scales[0], bx.scales[1], bx.scales[2], bx.scales[3]);

    printf("\nQ8_1 blocks:\n");
    for (int b = 0; b < 2; ++b) {
        printf("  block %d: d=%.6f, s=%.2f, qs[0..3]=%d %d %d %d\n",
               b, float(y_q8[b].d), float(y_q8[b].s),
               y_q8[b].qs[0], y_q8[b].qs[1], y_q8[b].qs[2], y_q8[b].qs[3]);
    }

    // Trace vec_dot for iqs=0
    trace_vec_dot_q6_K_q8_1(&bx, y_q8.data(), 0);
    trace_vec_dot_q6_K_q8_1(&bx, y_q8.data(), 1);
    trace_vec_dot_q6_K_q8_1(&bx, y_q8.data(), 8);
    trace_vec_dot_q6_K_q8_1(&bx, y_q8.data(), 16);

    // CPU simple reference
    cpu_simple_dot(&bx, y_float.data());

    return 0;
}
