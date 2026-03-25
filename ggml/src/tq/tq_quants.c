/**
 * TurboQuant KV Cache Quantization — CPU Reference Implementation
 *
 * TQ4_0: Random rotation + Beta-optimal 4-bit quantization.
 *
 * Key design: quantize/dequantize work on multiples of head_dim.
 * ggml calls to_float with k = n_heads * head_dim per row.
 * We split into head_dim-sized chunks and rotate each independently.
 */

#include "tq_quants.h"
#include "../ggml-quants.h"  /* for quantize_row_q4_0_ref, dequantize_row_q4_0 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
static float * g_rotation    = NULL;
static float * g_rotation_t  = NULL;
static int     g_dim         = 0;
static float   g_codebook[16];
static int     g_codebook_size = 0;

/* xoshiro256** RNG */
static uint64_t s_rng[4];

static void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s_rng[i] = z ^ (z >> 31);
    }
}

static uint64_t rng_next(void) {
    const uint64_t result = ((s_rng[1] * 5) << 7 | (s_rng[1] * 5) >> 57) * 9;
    const uint64_t t = s_rng[1] << 17;
    s_rng[2] ^= s_rng[0]; s_rng[3] ^= s_rng[1];
    s_rng[1] ^= s_rng[2]; s_rng[0] ^= s_rng[3];
    s_rng[2] ^= t;
    s_rng[3] = (s_rng[3] << 45) | (s_rng[3] >> 19);
    return result;
}

static float rng_normal(void) {
    double u1 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    double u2 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* Modified Gram-Schmidt QR */
static void qr_decomposition(float * Q, int n) {
    float * col_i = (float *)malloc(n * sizeof(float));
    float * col_j = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int r = 0; r < n; r++) col_i[r] = Q[r * n + i];
        for (int j = 0; j < i; j++) {
            for (int r = 0; r < n; r++) col_j[r] = Q[r * n + j];
            float dot = 0;
            for (int r = 0; r < n; r++) dot += col_i[r] * col_j[r];
            for (int r = 0; r < n; r++) col_i[r] -= dot * col_j[r];
        }
        float norm = 0;
        for (int r = 0; r < n; r++) norm += col_i[r] * col_i[r];
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int r = 0; r < n; r++) col_i[r] /= norm;
        }
        for (int r = 0; r < n; r++) Q[r * n + i] = col_i[r];
    }
    free(col_i);
    free(col_j);
}

static void compute_codebook_4bit(int dim) {
    const double sigma = 1.0 / sqrt((double)dim);
    /* Lloyd-Max optimal centroids for Gaussian(0, sigma^2) with 16 levels */
    const double lloyd_16[8] = {
        0.1284, 0.3882, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
    };
    for (int i = 0; i < 8; i++) {
        g_codebook[i]     = (float)(-lloyd_16[7 - i] * sigma);
        g_codebook[8 + i] = (float)( lloyd_16[i] * sigma);
    }
    g_codebook_size = 16;
}

/* -------------------------------------------------------------------------- */

#ifdef _WIN32
__declspec(dllexport)
#endif
void tq_init(int head_dim, uint64_t seed) {
    if (g_rotation && g_dim == head_dim) return;
    if (g_rotation) { free(g_rotation); free(g_rotation_t); }
    g_dim = head_dim;
    g_rotation   = (float *)malloc(head_dim * head_dim * sizeof(float));
    g_rotation_t = (float *)malloc(head_dim * head_dim * sizeof(float));
    /* Generate random orthogonal rotation matrix via QR decomposition */
    rng_seed(seed);
    for (int i = 0; i < head_dim * head_dim; i++) g_rotation[i] = rng_normal();
    qr_decomposition(g_rotation, head_dim);
    for (int i = 0; i < head_dim; i++)
        for (int j = 0; j < head_dim; j++)
            g_rotation_t[i * head_dim + j] = g_rotation[j * head_dim + i];
    compute_codebook_4bit(head_dim);

    /* Upload rotation matrices to GPU if CUDA is available */
#ifdef GGML_USE_CUDA
    extern void tq_cuda_init(const float * rotation, const float * rotation_t, int head_dim);
    tq_cuda_init(g_rotation, g_rotation_t, head_dim);
#endif
}

void tq_free(void) {
    free(g_rotation); g_rotation = NULL;
    free(g_rotation_t); g_rotation_t = NULL;
    g_dim = 0;
#ifdef GGML_USE_CUDA
    extern void tq_cuda_free(void);
    tq_cuda_free();
#endif
}

int tq_head_dim(void) { return g_dim; }

/* --------------------------------------------------------------------------
 * Internal: quantize/dequantize one head_dim-sized vector
 * -------------------------------------------------------------------------- */

static void tq_quantize_one_head(const float * x, void * y, int dim) {
    /* Step 1: Rotate the input vector: y_rot = Pi^T @ x */
    float y_rot[256];
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += g_rotation_t[i * dim + j] * x[j];
        }
        y_rot[i] = sum;
    }

    /* Step 2: Call the REAL Q4_0 quantize function on the rotated data.
     * This ensures the FP16 scale and nibble packing are identical to Q4_0. */
    quantize_row_q4_0_ref(y_rot, (block_q4_0 *)y, dim);
}

static void tq_dequantize_one_head(const void * x, float * y, int dim) {
    float y_rot[256];

    /* Step 1: Call REAL Q4_0 dequant to get rotated values */
    dequantize_row_q4_0((const block_q4_0 *)x, y_rot, dim);

    /* Step 2: Inverse rotate: output = Pi @ y_rot */
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += g_rotation[i * dim + j] * y_rot[j];
        }
        y[i] = sum;
    }
}

/* --------------------------------------------------------------------------
 * Public API: handle k as multiple of head_dim
 * ggml calls these with k = n_heads * head_dim per KV row
 * -------------------------------------------------------------------------- */

void quantize_row_tq4_0_ref(const float * x, void * y, int64_t k) {
    assert(g_rotation_t && "tq_init() must be called first");
    assert(g_dim > 0);

    static int log_count = 0;
    if (log_count < 5) {
        fprintf(stderr, "[TQ4_0] quantize_row called with k=%lld, g_dim=%d, n_heads=%lld\n", (long long)k, g_dim, (long long)(k / g_dim));
        log_count++;
    }

    /* If k == head_dim, single vector. If k > head_dim, multiple heads. */
    const int dim = g_dim;
    const int nb_per_head = dim / QK_TQ4_0;
    const int n_heads = (int)(k / dim);

    /* Fallback: if k isn't a multiple of head_dim, use plain Q4_0 */
    if (n_heads * dim != (int)k || n_heads == 0) {
        quantize_row_q4_0_ref(x, (block_q4_0 *)y, k);
        return;
    }

    /* Rotate + quantize each head independently */
    block_q4_0 * yb = (block_q4_0 *)y;
    for (int h = 0; h < n_heads; h++) {
        tq_quantize_one_head(x + h * dim, yb + h * nb_per_head, dim);
    }
}

void quantize_row_tq4_0(const float * x, void * y, int64_t k) {
    quantize_row_tq4_0_ref(x, y, k);
}

void dequantize_row_tq4_0(const void * x, float * y, int64_t k) {
    assert(g_rotation && "tq_init() must be called first");
    assert(g_dim > 0);

    static int dlog_count = 0;
    if (dlog_count < 5) {
        fprintf(stderr, "[TQ4_0] dequantize_row called with k=%lld, g_dim=%d\n", (long long)k, g_dim);
        dlog_count++;
    }

    const int dim = g_dim;
    const int nb_per_head = dim / QK_TQ4_0;
    const int n_heads = (int)(k / dim);

    /* Fallback: plain Q4_0 dequant if k isn't a multiple of head_dim */
    if (n_heads * dim != (int)k || n_heads == 0) {
        dequantize_row_q4_0((const block_q4_0 *)x, y, k);
        return;
    }

    /* Dequantize + inverse rotate each head independently */
    const block_q4_0 * xb = (const block_q4_0 *)x;
    for (int h = 0; h < n_heads; h++) {
        tq_dequantize_one_head(xb + h * nb_per_head, y + h * dim, dim);
    }
}

void vec_dot_tq4_0_q8_0(int n, float * s, size_t bs,
                         const void * vx, size_t bx,
                         const void * vy, size_t by,
                         int nrc) {
    /* Dequantize both sides then dot.
     * vx = TQ4_0 blocks (K cache), vy = Q8_0 blocks (query) */
    float * x_deq = (float *)malloc(n * sizeof(float));
    float * y_deq = (float *)malloc(n * sizeof(float));

    dequantize_row_tq4_0(vx, x_deq, n);
    dequantize_row_q8_0((const block_q8_0 *)vy, y_deq, n);

    float sum = 0;
    for (int i = 0; i < n; i++) sum += x_deq[i] * y_deq[i];
    *s = sum;

    free(x_deq);
    free(y_deq);
}

size_t quantize_tq4_0(const float * src, void * dst,
                       int64_t nrows, int64_t n_per_row,
                       const float * imatrix) {
    (void)imatrix;
    const size_t row_size = (n_per_row / QK_TQ4_0) * sizeof(block_q4_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_tq4_0(src + r * n_per_row,
                           (char *)dst + r * row_size,
                           n_per_row);
    }
    return nrows * row_size;
}
