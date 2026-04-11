// TurboQuant reference helpers for the CPU path.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-turboq.h"
#include "ggml-turboq-tables.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define TURBOQ_USE_ACCELERATE 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TURBOQ_TLS __thread
#elif defined(_MSC_VER)
#define TURBOQ_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#define TURBOQ_TLS _Thread_local
#else
#define TURBOQ_TLS
#endif

static inline uint64_t splitmix64_next(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void turboq_generate_gaussian(float * out, int64_t n, uint64_t seed) {
    uint64_t state = seed;
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        // Generate two uniform (0,1) variates
        double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double u2 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double r  = sqrt(-2.0 * log(u1));
        double th = 2.0 * 3.14159265358979323846 * u2;
        out[i]     = (float)(r * cos(th));
        out[i + 1] = (float)(r * sin(th));
    }
    if (i < n) {
        double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double u2 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double r  = sqrt(-2.0 * log(u1));
        double th = 2.0 * 3.14159265358979323846 * u2;
        out[i] = (float)(r * cos(th));
    }
}

// ---------------------------------------------------------------------------
// Householder QR decomposition (in-place, no LAPACK dependency)
//
// Input:  A[d*d] stored column-major (A[i + j*d] = A_{i,j})
// Output: Q[d*d] column-major orthogonal matrix, with Haar sign correction
//
// Uses Householder reflections: Q = H_1 * H_2 * ... * H_d where
// H_k = I - 2 * v_k * v_k^T / (v_k^T * v_k)
// ---------------------------------------------------------------------------

// Compute Q from Householder QR of column-major matrix A[d×d].
// A is modified in-place (becomes R on upper triangle, v below diagonal).
// Q is written to Q_out[d×d] column-major.
// Applies Haar sign correction: Q[:,j] *= sign(R[j,j]) so that Q is
// uniformly distributed on O(d) (Haar measure).
static void turboq_householder_qr(float * A, float * Q_out, int64_t d) {
    float * tau = (float *)malloc(d * sizeof(float));
    // Store sign(R[k,k]) = -sign(alpha_k) for Haar correction
    float * r_sign = (float *)malloc(d * sizeof(float));

    for (int64_t k = 0; k < d; k++) {
        // Compute norm of A[k:d, k]
        float norm_sq = 0.0f;
        for (int64_t i = k; i < d; i++) {
            float val = A[i + k * d];
            norm_sq += val * val;
        }
        float norm = sqrtf(norm_sq);

        // Choose sign to avoid cancellation
        float alpha = A[k + k * d];
        float sign_alpha = (alpha >= 0.0f) ? 1.0f : -1.0f;
        float u1 = alpha + sign_alpha * norm;

        // R[k,k] = -sign(alpha) * norm, so sign(R[k,k]) = -sign(alpha)
        r_sign[k] = -sign_alpha;

        // Compute tau = 2 / (v^T v)
        float vtv = u1 * u1 + (norm_sq - alpha * alpha);
        if (vtv < 1e-30f) {
            tau[k] = 0.0f;
            continue;
        }
        tau[k] = 2.0f / vtv;

        // Store v in A[k:d, k]
        A[k + k * d] = u1;

        // Apply H_k to remaining columns A[k:d, k+1:d]
        for (int64_t j = k + 1; j < d; j++) {
            float dot = 0.0f;
            dot += u1 * A[k + j * d];
            for (int64_t i = k + 1; i < d; i++) {
                dot += A[i + k * d] * A[i + j * d];
            }
            dot *= tau[k];
            A[k + j * d] -= dot * u1;
            for (int64_t i = k + 1; i < d; i++) {
                A[i + j * d] -= dot * A[i + k * d];
            }
        }
    }

    // Build Q by back-accumulation: Q = H_1 * H_2 * ... * H_{d-1}
    memset(Q_out, 0, d * d * sizeof(float));
    for (int64_t i = 0; i < d; i++) {
        Q_out[i + i * d] = 1.0f;
    }

    for (int64_t k = d - 1; k >= 0; k--) {
        if (tau[k] == 0.0f) continue;
        float u1 = A[k + k * d];
        for (int64_t j = 0; j < d; j++) {
            float dot = 0.0f;
            dot += u1 * Q_out[k + j * d];
            for (int64_t i = k + 1; i < d; i++) {
                dot += A[i + k * d] * Q_out[i + j * d];
            }
            dot *= tau[k];
            Q_out[k + j * d] -= dot * u1;
            for (int64_t i = k + 1; i < d; i++) {
                Q_out[i + j * d] -= dot * A[i + k * d];
            }
        }
    }

    // Haar sign correction: Q[:,j] *= sign(R[j,j])
    // This ensures Q is uniformly distributed on O(d), not just SO(d).
    // Reference: Mezzadri (2007), "How to Generate Random Matrices from the Classical Compact Groups"
    for (int64_t j = 0; j < d; j++) {
        if (r_sign[j] < 0.0f) {
            for (int64_t i = 0; i < d; i++) {
                Q_out[i + j * d] = -Q_out[i + j * d];
            }
        }
    }

    free(tau);
    free(r_sign);
}

// ---------------------------------------------------------------------------
// Rotation matrix cache
//
// For a given (dimension, seed) pair, generate and cache the d×d orthogonal Q.
// The cache is thread-local to avoid locks. In practice, all rows of a weight
// matrix share the same dimension, so the cache hit rate is ~100%.
// ---------------------------------------------------------------------------

static TURBOQ_TLS float * tl_Q = NULL;
static TURBOQ_TLS float * tl_Q_row = NULL;
static TURBOQ_TLS int64_t tl_Q_dim = 0;
static TURBOQ_TLS uint64_t tl_Q_seed = 0;

static const float * turboq_get_rotation(int64_t d, uint64_t seed) {
    if (tl_Q != NULL && tl_Q_dim == d && tl_Q_seed == seed) {
        return tl_Q;
    }
    // Regenerate
    free(tl_Q);
    free(tl_Q_row);
    tl_Q = (float *)malloc(d * d * sizeof(float));
    tl_Q_row = (float *)malloc(d * d * sizeof(float));
    tl_Q_dim = d;
    tl_Q_seed = seed;

    // Generate d×d Gaussian random matrix (column-major)
    float * A = (float *)malloc(d * d * sizeof(float));
    turboq_generate_gaussian(A, d * d, seed);

    // Compute QR, store Q in tl_Q
    turboq_householder_qr(A, tl_Q, d);

    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            tl_Q_row[i * d + j] = tl_Q[i + j * d];
        }
    }

    // One-shot dump for Metal shader embedding. Set TURBOQ_DUMP_Q=<path> in
    // the environment to dump the Q matrix (and its transpose in row-major
    // form, which is the layout the Metal dequant kernel consumes) to a C
    // header. This lets us bake Q into the shader as a `constant` array and
    // avoid per-dispatch buffer binding. Safe to leave in; only fires when
    // the env var is set and the dimension matches TURBOQ_KV_DIM.
    if (d == 128) {
        const char * dump_path = getenv("TURBOQ_DUMP_Q");
        if (dump_path != NULL && dump_path[0] != '\0') {
            FILE * f = fopen(dump_path, "w");
            if (f != NULL) {
                fprintf(f, "// Auto-generated by ggml-turboq.c (TURBOQ_DUMP_Q=1).\n");
                fprintf(f, "// TurboQuant 128x128 rotation matrices, baked for Metal shaders.\n");
                fprintf(f, "// Seed: 0x%016llx\n\n", (unsigned long long)seed);
                fprintf(f, "#pragma once\n\n");
                // M = Q^T in row-major: M[i*d + j] = Q[j,i]. In the CPU's
                // column-major tl_Q layout, Q[j,i] lives at tl_Q[j + i*d],
                // so M[i*d + j] = tl_Q[j + i*d]. Also note this layout gives
                // each Metal thread i a contiguous d-float row to dot against
                // `rotated` for the inverse-rotation step.
                fprintf(f, "constant float TURBOQ_QT_128[128 * 128] = {\n");
                for (int64_t i = 0; i < d; i++) {
                    for (int64_t j = 0; j < d; j++) {
                        float v = tl_Q[j + i * d];
                        fprintf(f, " %.9ef,", v);
                        if ((j & 3) == 3) fprintf(f, "\n");
                    }
                }
                fprintf(f, "};\n\n");
                // Also emit the forward (Q row-major) matrix for SET_ROWS, so
                // Stage 2 can reuse the same header.
                fprintf(f, "constant float TURBOQ_Q_128[128 * 128] = {\n");
                for (int64_t i = 0; i < d; i++) {
                    for (int64_t j = 0; j < d; j++) {
                        float v = tl_Q_row[i * d + j];
                        fprintf(f, " %.9ef,", v);
                        if ((j & 3) == 3) fprintf(f, "\n");
                    }
                }
                fprintf(f, "};\n");
                fclose(f);
                fprintf(stderr, "turboq: dumped rotation matrices to %s\n", dump_path);
            } else {
                fprintf(stderr, "turboq: TURBOQ_DUMP_Q set but fopen failed\n");
            }
        }
    }

    free(A);
    return tl_Q;
}

static const float * turboq_get_rotation_row(int64_t d, uint64_t seed) {
    turboq_get_rotation(d, seed);
    return tl_Q_row;
}

// ---------------------------------------------------------------------------
// Projection matrix cache (for Q_prod QJL stage)
//
// S is a d×d random Gaussian matrix (NOT orthogonalized), used for QJL:
//   qjl_signs = sign(S · residual)
//   dequant:    sqrt(pi/2)/d · gamma · S^T · signs
// Uses a different seed stream from the rotation matrix Q.
// ---------------------------------------------------------------------------

static TURBOQ_TLS float * tl_S = NULL;
static TURBOQ_TLS float * tl_S_row = NULL;
static TURBOQ_TLS int64_t tl_S_dim = 0;
static TURBOQ_TLS uint64_t tl_S_seed = 0;

static const float * turboq_get_projection(int64_t d, uint64_t seed) {
    // Use a different seed stream for S vs Q
    uint64_t s_seed = seed ^ 0x1234567890abcdefULL;
    if (tl_S != NULL && tl_S_dim == d && tl_S_seed == s_seed) {
        return tl_S;
    }
    free(tl_S);
    free(tl_S_row);
    tl_S = (float *)malloc(d * d * sizeof(float));
    tl_S_row = (float *)malloc(d * d * sizeof(float));
    tl_S_dim = d;
    tl_S_seed = s_seed;

    // Generate d×d Gaussian random matrix (column-major), no QR
    turboq_generate_gaussian(tl_S, d * d, s_seed);

    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            tl_S_row[i * d + j] = tl_S[i + j * d];
        }
    }

    return tl_S;
}

static const float * turboq_get_projection_row(int64_t d, uint64_t seed) {
    turboq_get_projection(d, seed);
    return tl_S_row;
}

// ---------------------------------------------------------------------------
// Dense matrix-vector multiply: y = M * x  (M is d×d column-major)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dense matrix-vector multiply using Accelerate (cblas) on Apple Silicon
// Falls back to scalar/AVX2 on other platforms
// ---------------------------------------------------------------------------

// y = M * x  (M is d×d column-major)
static void matvec(float * y, const float * M, const float * x, int64_t d) {
#if TURBOQ_USE_ACCELERATE
    // cblas_sgemv: y = alpha * M * x + beta * y
    // M is column-major, CblasColMajor, CblasNoTrans
    cblas_sgemv(CblasColMajor, CblasNoTrans, (int)d, (int)d, 1.0f, M, (int)d, x, 1, 0.0f, y, 1);
#else
    for (int64_t i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            sum += M[i + j * d] * x[j];
        }
        y[i] = sum;
    }
#endif
}

// y = M_row * x  (M_row is d×d row-major)
static void matvec_row(float * y, const float * M, const float * x, int64_t d) {
#if TURBOQ_USE_ACCELERATE
    // Row-major M with CblasRowMajor
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)d, (int)d, 1.0f, M, (int)d, x, 1, 0.0f, y, 1);
#else
    for (int64_t i = 0; i < d; ++i) {
        const float * row = M + i * d;
        float sum = 0.0f;
        int64_t j = 0;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; j + 7 < d; j += 8) {
            const __m256 mv = _mm256_loadu_ps(row + j);
            const __m256 xv = _mm256_loadu_ps(x + j);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(mv, xv));
        }
        // horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        s128 = _mm_hadd_ps(s128, s128);
        s128 = _mm_hadd_ps(s128, s128);
        sum += _mm_cvtss_f32(s128);
#endif
        for (; j < d; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
#endif
}

// y = M^T * x  (M is d×d column-major)
static void matvec_t(float * y, const float * M, const float * x, int64_t d) {
#if TURBOQ_USE_ACCELERATE
    // M^T with CblasColMajor, CblasTrans
    cblas_sgemv(CblasColMajor, CblasTrans, (int)d, (int)d, 1.0f, M, (int)d, x, 1, 0.0f, y, 1);
#else
    for (int64_t j = 0; j < d; j++) {
        const float * col = M + j * d;
        float sum = 0.0f;
        int64_t i = 0;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 7 < d; i += 8) {
            const __m256 mv = _mm256_loadu_ps(col + i);
            const __m256 xv = _mm256_loadu_ps(x + i);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(mv, xv));
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        s128 = _mm_hadd_ps(s128, s128);
        s128 = _mm_hadd_ps(s128, s128);
        sum += _mm_cvtss_f32(s128);
#endif
        for (; i < d; ++i) {
            sum += col[i] * x[i];
        }
        y[j] = sum;
    }
#endif
}

// ---------------------------------------------------------------------------
// Public API (kept for compatibility, now wraps dense rotation)
// ---------------------------------------------------------------------------

// The rotation matrix is a global parameter (same for all vectors), per the paper.
// This seed is used to deterministically generate both Q and S matrices.
uint64_t turboq_seed_from_row(int64_t row_idx) {
    (void)row_idx;
    return 0x517cc1b727220a95ULL;
}

// Forward rotation: y = Q · x  (paper Algorithm 1, line 5: y <- Pi . x)
void turboq_rotate_forward(float * y, const float * x, int64_t d, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(d, seed);
    matvec_row(y, Q, x, d);
}

// Inverse rotation: x = Q^T · y  (paper Algorithm 1, line 10: x_tilde <- Pi^T . y_tilde)
void turboq_rotate_inverse(float * x, const float * y, int64_t d, uint64_t seed) {
    const float * Q = turboq_get_rotation(d, seed);
    matvec_t(x, Q, y, d);
}

// ---------------------------------------------------------------------------
// Scratch buffer (thread-local, for temporary vectors)
// ---------------------------------------------------------------------------

static TURBOQ_TLS float * tl_buf = NULL;
static TURBOQ_TLS int64_t tl_buf_size = 0;

static float * turboq_get_scratch(int64_t n) {
    if (n > tl_buf_size) {
        free(tl_buf);
        tl_buf = (float *)malloc(n * sizeof(float));
        tl_buf_size = n;
    }
    return tl_buf;
}

// Second scratch buffer (needed when two temp vectors are required simultaneously,
// e.g. rotated-domain values + original-domain result in dequant)
static TURBOQ_TLS float * tl_buf2 = NULL;
static TURBOQ_TLS int64_t tl_buf2_size = 0;

static float * turboq_get_scratch2(int64_t n) {
    if (n > tl_buf2_size) {
        free(tl_buf2);
        tl_buf2 = (float *)malloc(n * sizeof(float));
        tl_buf2_size = n;
    }
    return tl_buf2;
}

// Third scratch buffer (needed by Q_prod dequant which requires three simultaneous vectors:
// mse_rot, signs_f, and mse_unit)
static TURBOQ_TLS float * tl_buf3 = NULL;
static TURBOQ_TLS int64_t tl_buf3_size = 0;

static float * turboq_get_scratch3(int64_t n) {
    if (n > tl_buf3_size) {
        free(tl_buf3);
        tl_buf3 = (float *)malloc(n * sizeof(float));
        tl_buf3_size = n;
    }
    return tl_buf3;
}

#define TURBOQ_KV_DIM 128

static inline float turboq_block_scale_up(void) {
    return sqrtf((float) QK_K);
}

static inline float turboq_block_scale_down(void) {
    return 1.0f / turboq_block_scale_up();
}

static void turboq_rotate_block_forward(float * y, const float * x, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_row(y + i, Q, x + i, TURBOQ_KV_DIM);
    }
}

static void turboq_rotate_block_inverse(float * x, const float * y, uint64_t seed) {
    const float * Q = turboq_get_rotation(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_t(x + i, Q, y + i, TURBOQ_KV_DIM);
    }
}

static void turboq_project_block(float * y, const float * x, uint64_t seed) {
    const float * S = turboq_get_projection_row(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_row(y + i, S, x + i, TURBOQ_KV_DIM);
    }
}

static void turboq_project_block_inverse(float * x, const float * y, uint64_t seed) {
    const float * S = turboq_get_projection(TURBOQ_KV_DIM, seed);

    for (int64_t i = 0; i < QK_K; i += TURBOQ_KV_DIM) {
        matvec_t(x + i, S, y + i, TURBOQ_KV_DIM);
    }
}

static void turboq_rotate_qk_forward(float * y, const float * x, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(QK_K, seed);
    matvec_row(y, Q, x, QK_K);
}

static void turboq_rotate_qk_inverse(float * x, const float * y, uint64_t seed) {
    const float * Q = turboq_get_rotation(QK_K, seed);
    matvec_t(x, Q, y, QK_K);
}

static void turboq_project_qk(float * y, const float * x, uint64_t seed) {
    const float * S = turboq_get_projection_row(QK_K, seed);
    matvec_row(y, S, x, QK_K);
}

static void turboq_project_qk_inverse(float * x, const float * y, uint64_t seed) {
    const float * S = turboq_get_projection(QK_K, seed);
    matvec_t(x, S, y, QK_K);
}

// ---------------------------------------------------------------------------
// Scalar codebook quantization
// ---------------------------------------------------------------------------

static inline uint8_t quantize_scalar(float val, const float * boundaries, int n_boundaries) {
    for (int i = 0; i < n_boundaries; i++) {
        if (val < boundaries[i]) {
            return (uint8_t)i;
        }
    }
    return (uint8_t)n_boundaries;
}

static inline uint8_t quantize_scalar_3bit(float val) {
    return quantize_scalar(val, turboq_boundaries_3bit, 7);
}

static inline uint8_t quantize_scalar_2bit(float val) {
    return quantize_scalar(val, turboq_boundaries_2bit, 3);
}

static inline uint8_t quantize_scalar_4bit(float val) {
    return quantize_scalar(val, turboq_boundaries_4bit, 15);
}

// ---------------------------------------------------------------------------
// 3-bit packing/unpacking
// ---------------------------------------------------------------------------

static void pack_3bit(uint8_t * dst, const uint8_t * indices, int64_t n) {
    int64_t full_groups = n / 8;
    for (int64_t g = 0; g < full_groups; g++) {
        const uint8_t * idx = indices + g * 8;
        uint32_t bits = 0;
        for (int j = 0; j < 8; j++) {
            bits |= ((uint32_t)(idx[j] & 0x7)) << (j * 3);
        }
        dst[g * 3 + 0] = (uint8_t)(bits & 0xFF);
        dst[g * 3 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        dst[g * 3 + 2] = (uint8_t)((bits >> 16) & 0xFF);
    }
}

static void unpack_3bit(uint8_t * indices, const uint8_t * src, int64_t n) {
    int64_t full_groups = n / 8;
    for (int64_t g = 0; g < full_groups; g++) {
        uint32_t bits = (uint32_t)src[g * 3 + 0]
                     | ((uint32_t)src[g * 3 + 1] << 8)
                     | ((uint32_t)src[g * 3 + 2] << 16);
        for (int j = 0; j < 8; j++) {
            indices[g * 8 + j] = (uint8_t)((bits >> (j * 3)) & 0x7);
        }
    }
}

// ---------------------------------------------------------------------------
// TBQ3_0: TurboQuant 3-bit
// ---------------------------------------------------------------------------

void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float * unit = turboq_get_scratch(QK_K);
    float * rotated = turboq_get_scratch2(QK_K);
    const uint64_t seed = turboq_seed_from_row(0);
    const float scale_up = turboq_block_scale_up();
    uint8_t indices[QK_K];

    for (int64_t b = 0; b < nb; b++) {
        const float * xb = x + b * QK_K;

        float norm_sq = 0.0f;
        for (int64_t j = 0; j < QK_K; ++j) {
            norm_sq += xb[j] * xb[j];
        }

        float norm = sqrtf(norm_sq);
        if (norm < 1e-10f) {
            norm = 1e-10f;
        }

        for (int64_t j = 0; j < QK_K; ++j) {
            unit[j] = xb[j] / norm;
        }

        turboq_rotate_block_forward(rotated, unit, seed);

        for (int64_t j = 0; j < QK_K; j++) {
            float val = rotated[j] * scale_up;
            indices[j] = quantize_scalar_3bit(val);
        }
        pack_3bit(y[b].qs, indices, QK_K);
        y[b].d = GGML_FP32_TO_FP16(norm);
    }
}

void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float * rotated = turboq_get_scratch(QK_K);
    float * unit_approx = turboq_get_scratch2(QK_K);
    const uint64_t seed = turboq_seed_from_row(0);
    const float scale_down = turboq_block_scale_down();
    uint8_t indices[QK_K];

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].d);

        unpack_3bit(indices, x[b].qs, QK_K);
        for (int64_t j = 0; j < QK_K; j++) {
            rotated[j] = turboq_codebook_3bit[indices[j]] * scale_down;
        }

        turboq_rotate_block_inverse(unit_approx, rotated, seed);

        // Norm correction: re-normalize unit_approx to unit length
        float recon_norm_sq = 0.0f;
        for (int64_t j = 0; j < QK_K; j++) {
            recon_norm_sq += unit_approx[j] * unit_approx[j];
        }
        float recon_scale = (recon_norm_sq > 1e-20f) ? (norm / sqrtf(recon_norm_sq)) : norm;

        for (int64_t j = 0; j < QK_K; ++j) {
            y[b * QK_K + j] = unit_approx[j] * recon_scale;
        }
    }
}

size_t quantize_tbq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_K == 0);

    const int64_t nb_per_row = n_per_row / QK_K;
    const size_t row_size = nb_per_row * sizeof(block_tbq3_0);

    for (int64_t row = 0; row < nrows; row++) {
        const float * row_src = src + row * n_per_row;
        block_tbq3_0 * row_dst = (block_tbq3_0 *)((char *)dst + row * row_size);
        quantize_row_tbq3_0_ref(row_src, row_dst, n_per_row);
    }
    return nrows * row_size;
}

// ---------------------------------------------------------------------------
// TBQ4_0: TurboQuant 4-bit
// ---------------------------------------------------------------------------

void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float * unit = turboq_get_scratch(QK_K);
    float * rotated = turboq_get_scratch2(QK_K);
    const uint64_t seed = turboq_seed_from_row(0);
    const float scale_up = turboq_block_scale_up();

    for (int64_t b = 0; b < nb; b++) {
        const float * xb = x + b * QK_K;

        float norm_sq = 0.0f;
        for (int64_t j = 0; j < QK_K; ++j) {
            norm_sq += xb[j] * xb[j];
        }

        float norm = sqrtf(norm_sq);
        if (norm < 1e-10f) {
            norm = 1e-10f;
        }

        for (int64_t j = 0; j < QK_K; ++j) {
            unit[j] = xb[j] / norm;
        }

        turboq_rotate_block_forward(rotated, unit, seed);

        memset(y[b].qs, 0, sizeof(y[b].qs));
        for (int64_t j = 0; j < QK_K; j++) {
            float val = rotated[j] * scale_up;
            uint8_t idx = quantize_scalar_4bit(val);
            if (j % 2 == 0) {
                y[b].qs[j / 2] = idx;
            } else {
                y[b].qs[j / 2] |= (idx << 4);
            }
        }
        y[b].d = GGML_FP32_TO_FP16(norm);
    }
}

void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    float * rotated = turboq_get_scratch(QK_K);
    float * unit_approx = turboq_get_scratch2(QK_K);
    const uint64_t seed = turboq_seed_from_row(0);
    const float scale_down = turboq_block_scale_down();

    for (int64_t b = 0; b < nb; b++) {
        const float norm = GGML_FP16_TO_FP32(x[b].d);

        for (int64_t j = 0; j < QK_K; j++) {
            uint8_t idx;
            if (j % 2 == 0) {
                idx = x[b].qs[j / 2] & 0x0F;
            } else {
                idx = (x[b].qs[j / 2] >> 4) & 0x0F;
            }
            rotated[j] = turboq_codebook_4bit[idx] * scale_down;
        }

        turboq_rotate_block_inverse(unit_approx, rotated, seed);

        // Norm correction: re-normalize unit_approx to unit length
        // This removes magnitude bias from quantization error
        float recon_norm_sq = 0.0f;
        for (int64_t j = 0; j < QK_K; j++) {
            recon_norm_sq += unit_approx[j] * unit_approx[j];
        }
        float recon_scale = (recon_norm_sq > 1e-20f) ? (norm / sqrtf(recon_norm_sq)) : norm;

        for (int64_t j = 0; j < QK_K; ++j) {
            y[b * QK_K + j] = unit_approx[j] * recon_scale;
        }
    }
}

size_t quantize_tbq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_K == 0);

    const int64_t nb_per_row = n_per_row / QK_K;
    const size_t row_size = nb_per_row * sizeof(block_tbq4_0);

    for (int64_t row = 0; row < nrows; row++) {
        const float * row_src = src + row * n_per_row;
        block_tbq4_0 * row_dst = (block_tbq4_0 *)((char *)dst + row * row_size);
        quantize_row_tbq4_0_ref(row_src, row_dst, n_per_row);
    }
    return nrows * row_size;
}
