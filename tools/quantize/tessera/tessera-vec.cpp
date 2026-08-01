//
// tessera-vec.cpp
//
// Platform dispatch for the tessera-vec.h shim: Accelerate/vDSP on macOS,
// OpenBLAS cblas_s* on Linux, naive scalar loops otherwise. OpenBLAS exposes
// no elementwise or min/max/mean/sum primitives, so those fall back to scalar
// loops even when GGML_USE_OPENBLAS is set.
//

#include "tessera-vec.h"

#if defined(__APPLE__)
// use the non-deprecated CBLAS declarations (legacy ones warn on macOS 13.3+)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

#include <cmath>

// reductions

float ts_vec_meanv(const float * x, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    float r;
    vDSP_meanv(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        s += x[i];
    }
    return s / (float)n;
#endif
}

float ts_vec_measqv(const float * x, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    float r;
    vDSP_measqv(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        s += x[i] * x[i];
    }
    return s / (float)n;
#endif
}

float ts_vec_maxv(const float * x, int64_t n) {
    if (n <= 0) {
        return -INFINITY;
    }
#if defined(__APPLE__)
    float r;
    vDSP_maxv(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    float m = x[0];
    for (int64_t i = 1; i < n; ++i) {
        if (x[i] > m) {
            m = x[i];
        }
    }
    return m;
#endif
}

float ts_vec_minv(const float * x, int64_t n) {
    if (n <= 0) {
        return INFINITY;
    }
#if defined(__APPLE__)
    float r;
    vDSP_minv(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    float m = x[0];
    for (int64_t i = 1; i < n; ++i) {
        if (x[i] < m) {
            m = x[i];
        }
    }
    return m;
#endif
}

float ts_vec_sve(const float * x, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    float r;
    vDSP_sve(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
#endif
}

float ts_vec_dotpr(const float * a, const float * b, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    float r;
    vDSP_dotpr(a, 1, b, 1, &r, (vDSP_Length)n);
    return r;
#elif defined(GGML_USE_OPENBLAS)
    return cblas_sdot((int)n, a, 1, b, 1);
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
#endif
}

float ts_vec_norm2(const float * x, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    float r;
    vDSP_dotpr(x, 1, x, 1, &r, (vDSP_Length)n);
    return std::sqrt(r);
#elif defined(GGML_USE_OPENBLAS)
    return cblas_snrm2((int)n, x, 1);
#else
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        s += x[i] * x[i];
    }
    return std::sqrt(s);
#endif
}

// elementwise

void ts_vec_vsmul(const float * x, float scalar, float * out, int64_t n) {
    if (n <= 0) {
        return;
    }
#if defined(__APPLE__)
    vDSP_vsmul(x, 1, &scalar, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) {
        out[i] = x[i] * scalar;
    }
#endif
}

void ts_vec_vmul(const float * a, const float * b, float * out, int64_t n) {
    if (n <= 0) {
        return;
    }
#if defined(__APPLE__)
    vDSP_vmul(a, 1, b, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#endif
}

void ts_vec_vadd(const float * a, const float * b, float * out, int64_t n) {
    if (n <= 0) {
        return;
    }
#if defined(__APPLE__)
    vDSP_vadd(a, 1, b, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

void ts_vec_vsub(const float * a, const float * b, float * out, int64_t n) {
    if (n <= 0) {
        return;
    }
#if defined(__APPLE__)
    // vDSP_vsub computes C = B - A, so the operands are swapped to yield a - b
    vDSP_vsub(b, 1, a, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#endif
}

void ts_vec_scale(float * x, float scalar, int64_t n) {
    if (n <= 0) {
        return;
    }
#if defined(__APPLE__)
    vDSP_vsmul(x, 1, &scalar, x, 1, (vDSP_Length)n);
#elif defined(GGML_USE_OPENBLAS)
    cblas_sscal((int)n, scalar, x, 1);
#else
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= scalar;
    }
#endif
}

float ts_vec_maxabs(const float * x, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    float r = 0.0f;
    vDSP_maxmgv(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    float r = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        r = std::max(r, std::fabs(x[i]));
    }
    return r;
#endif
}

float ts_vec_meanabs(const float * x, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
#if defined(__APPLE__)
    // mean(|x|) directly via vDSP_meamgv (mean of magnitudes).
    float r = 0.0f;
    vDSP_meamgv(x, 1, &r, (vDSP_Length)n);
    return r;
#else
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += std::fabs((double)x[i]);
    }
    return (float)(sum / (double)n);
#endif
}

// matrix ops

void ts_mat_scale_cols(const float * W, const float * scale, float * out,
                       int64_t rows, int64_t cols) {
    // out[r, c] = W[r, c] * scale[c]. Each row is an elementwise mul against
    // the same `scale` vector of length cols.
    if (rows <= 0 || cols <= 0) {
        return;
    }
#if defined(__APPLE__)
    for (int64_t r = 0; r < rows; ++r) {
        vDSP_vmul(W + r * cols, 1, scale, 1, out + r * cols, 1,
                  (vDSP_Length)cols);
    }
#else
    for (int64_t r = 0; r < rows; ++r) {
        const float * wrow = W + r * cols;
        float * orow = out + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            orow[c] = wrow[c] * scale[c];
        }
    }
#endif
}

void ts_mat_mul(const float * A, const float * B, float * C,
                int64_t M, int64_t K, int64_t N) {
    if (M <= 0 || N <= 0) {
        return;
    }
#if defined(__APPLE__) || defined(GGML_USE_OPENBLAS)
    // C(M x N) = A(M x K) @ B(K x N), row-major
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N, 0.0f, C, (int)N);
#else
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
#endif
}

void ts_mat_mul_at(const float * A, const float * B, float * C,
                   int64_t M, int64_t K, int64_t N) {
    if (M <= 0 || N <= 0) {
        return;
    }
#if defined(__APPLE__) || defined(GGML_USE_OPENBLAS)
    // C(M x N) = A^T @ B, with A stored row-major as (K x M) and B as (K x N)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)M, B, (int)N, 0.0f, C, (int)N);
#else
    // A stored as (K x M): element A[k][m] lives at A[k * M + m]
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                s += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
#endif
}
