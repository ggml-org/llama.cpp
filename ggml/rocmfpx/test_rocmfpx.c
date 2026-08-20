#include "rocmfpx.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void fill_row(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        const float wave = 0.75f*sinf((float) i * 0.37f) + 0.25f*cosf((float) i * 0.13f);
        const float ramp = ((float) (i % 11) - 5.0f) * 0.035f;
        x[i] = wave + ramp;
    }

    x[7]  =  3.25f;
    x[19] = -2.75f;
    x[43] =  1.875f;
}

static float mse(const float * a, const float * b, int n) {
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += d*d;
    }

    return err / (float) n;
}

static float weighted_mse(const float * a, const float * b, const float * w, int n) {
    float err = 0.0f;
    float sum_w = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += w[i]*d*d;
        sum_w += w[i];
    }

    return sum_w > 0.0f ? err / sum_w : 0.0f;
}

static void fill_imatrix_case(float * src, float * imatrix, int n, float base, float outlier) {
    for (int i = 0; i < n; ++i) {
        const float sign = (i % 2) ? 1.0f : -1.0f;
        src[i] = sign * base * (1.0f + 0.03f*(float)(i % 5));
        imatrix[i] = 100.0f;
    }

    src[0] = outlier;
    imatrix[0] = 0.0f;
}

static void check_weighted_imatrix_fp3(void) {
    enum { N = QK_ROCMFP3 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp3 q_plain[N / QK_ROCMFP3];
    block_rocmfp3 q_weighted[N / QK_ROCMFP3];

    fill_imatrix_case(src, imatrix, N, 0.21f, 9.0f);

    rocmfpx_quantize_fp3(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp3(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp3(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp3(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP3 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_weighted_imatrix_fp6(void) {
    enum { N = QK_ROCMFP6 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp6 q_plain[N / QK_ROCMFP6];
    block_rocmfp6 q_weighted[N / QK_ROCMFP6];

    fill_imatrix_case(src, imatrix, N, 0.045f, 6.0f);

    rocmfpx_quantize_fp6(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp6(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp6(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp6(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP6 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_weighted_imatrix_fp8(void) {
    enum { N = QK_ROCMFP8 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp8 q_plain[N / QK_ROCMFP8];
    block_rocmfp8 q_weighted[N / QK_ROCMFP8];

    fill_imatrix_case(src, imatrix, N, 0.008f, 8.0f);

    rocmfpx_quantize_fp8(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp8(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp8(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp8(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP8 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

int main(void) {
    enum { N = 64 };

    float src[N];
    float fp3[N];
    float fp6[N];
    float fp8[N];

    block_rocmfp3 q3[N / QK_ROCMFP3];
    block_rocmfp6 q6[N / QK_ROCMFP6];
    block_rocmfp8 q8[N / QK_ROCMFP8];

    fill_row(src, N);

    rocmfpx_quantize_row_fp3_ref(src, q3, N);
    rocmfpx_quantize_row_fp6_ref(src, q6, N);
    rocmfpx_quantize_row_fp8_ref(src, q8, N);

    assert(rocmfpx_validate_row_data_fp3(q3, sizeof(q3)));
    assert(rocmfpx_validate_row_data_fp6(q6, sizeof(q6)));
    assert(rocmfpx_validate_row_data_fp8(q8, sizeof(q8)));

    rocmfpx_dequantize_row_fp3(q3, fp3, N);
    rocmfpx_dequantize_row_fp6(q6, fp6, N);
    rocmfpx_dequantize_row_fp8(q8, fp8, N);

    const float mse3 = mse(src, fp3, N);
    const float mse6 = mse(src, fp6, N);
    const float mse8 = mse(src, fp8, N);

    printf("ROCmFP3: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp3), rocmfpx_row_size_fp3(N),
            8.0f*(float) sizeof(block_rocmfp3)/(float) QK_ROCMFP3, mse3);
    printf("ROCmFP6: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp6), rocmfpx_row_size_fp6(N),
            8.0f*(float) sizeof(block_rocmfp6)/(float) QK_ROCMFP6, mse6);
    printf("ROCmFP8: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp8), rocmfpx_row_size_fp8(N),
            8.0f*(float) sizeof(block_rocmfp8)/(float) QK_ROCMFP8, mse8);

    assert(isfinite(mse3));
    assert(isfinite(mse6));
    assert(isfinite(mse8));
    assert(mse8 < mse6);
    assert(mse6 < mse3);

    check_weighted_imatrix_fp3();
    check_weighted_imatrix_fp6();
    check_weighted_imatrix_fp8();

    return 0;
}
