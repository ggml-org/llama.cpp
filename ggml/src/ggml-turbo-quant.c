#include "ggml-turbo-quant.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Simple seeded xoshiro256** PRNG for reproducible rotation generation
typedef struct {
    uint64_t s[4];
} tq_prng_state;

static uint64_t tq_rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t tq_prng_next(tq_prng_state * state) {
    const uint64_t result = tq_rotl(state->s[1] * 5, 7) * 9;
    const uint64_t t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];
    state->s[2] ^= t;
    state->s[3] = tq_rotl(state->s[3], 45);

    return result;
}

static void tq_prng_seed(tq_prng_state * state, uint64_t seed) {
    // SplitMix64 to seed the state
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        state->s[i] = z;
    }
}

// Generate a standard normal random variable using Box-Muller transform
static float tq_randn(tq_prng_state * state) {
    // Generate two uniform random numbers in (0, 1)
    uint64_t u1_bits = tq_prng_next(state);
    uint64_t u2_bits = tq_prng_next(state);

    // Convert to (0, 1) range, avoiding exact 0
    double u1 = ((double)(u1_bits >> 11) + 0.5) / 9007199254740992.0; // 2^53
    double u2 = ((double)(u2_bits >> 11) + 0.5) / 9007199254740992.0;

    // Box-Muller
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;

    return (float)(r * cos(theta));
}

// In-place QR decomposition via modified Gram-Schmidt
// Input: matrix A (dim x dim, row-major)
// Output: Q in A, R discarded (we only need Q)
static void tq_qr_gram_schmidt(float * A, int dim) {
    float * col_i = (float *)malloc(dim * sizeof(float));
    float * col_j = (float *)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        // Extract column i
        for (int r = 0; r < dim; r++) {
            col_i[r] = A[r * dim + i];
        }

        // Orthogonalize against previous columns
        for (int j = 0; j < i; j++) {
            // Extract column j
            for (int r = 0; r < dim; r++) {
                col_j[r] = A[r * dim + j];
            }

            // Compute dot product
            float dot = 0.0f;
            for (int r = 0; r < dim; r++) {
                dot += col_i[r] * col_j[r];
            }

            // Subtract projection
            for (int r = 0; r < dim; r++) {
                col_i[r] -= dot * col_j[r];
            }
        }

        // Normalize
        float norm = 0.0f;
        for (int r = 0; r < dim; r++) {
            norm += col_i[r] * col_i[r];
        }
        norm = sqrtf(norm);

        if (norm > 1e-10f) {
            for (int r = 0; r < dim; r++) {
                col_i[r] /= norm;
            }
        }

        // Write back column i
        for (int r = 0; r < dim; r++) {
            A[r * dim + i] = col_i[r];
        }
    }

    free(col_i);
    free(col_j);
}

turbo_quant_ctx * turbo_quant_init(int dim, uint64_t seed) {
    turbo_quant_ctx * ctx = (turbo_quant_ctx *)malloc(sizeof(turbo_quant_ctx));
    if (!ctx) return NULL;

    ctx->dim  = dim;
    ctx->seed = seed;

    const size_t mat_size = (size_t)dim * dim * sizeof(float);
    ctx->rotation   = (float *)malloc(mat_size);
    ctx->rotation_t = (float *)malloc(mat_size);

    if (!ctx->rotation || !ctx->rotation_t) {
        free(ctx->rotation);
        free(ctx->rotation_t);
        free(ctx);
        return NULL;
    }

    // Generate random Gaussian matrix with deterministic seed
    // Use seed + dim * 7919 to match mlx-vlm reference
    tq_prng_state prng;
    tq_prng_seed(&prng, seed + (uint64_t)dim * 7919ULL);

    for (int i = 0; i < dim * dim; i++) {
        ctx->rotation[i] = tq_randn(&prng);
    }

    // QR decomposition to get orthonormal Q
    tq_qr_gram_schmidt(ctx->rotation, dim);

    // Ensure deterministic signs: multiply each column by sign of diagonal
    for (int i = 0; i < dim; i++) {
        float diag = ctx->rotation[i * dim + i];
        if (diag < 0.0f) {
            for (int r = 0; r < dim; r++) {
                ctx->rotation[r * dim + i] = -ctx->rotation[r * dim + i];
            }
        }
    }

    // Compute transpose
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            ctx->rotation_t[i * dim + j] = ctx->rotation[j * dim + i];
        }
    }

    return ctx;
}

void turbo_quant_free(turbo_quant_ctx * ctx) {
    if (ctx) {
        free(ctx->rotation);
        free(ctx->rotation_t);
        free(ctx);
    }
}

// Global context pointer (TLS breaks with ggml threadpool worker threads)
static turbo_quant_ctx * tq_global_ctx = NULL;

void turbo_quant_set_ctx(turbo_quant_ctx * ctx) {
    tq_global_ctx = ctx;
}

turbo_quant_ctx * turbo_quant_get_ctx(void) {
    return tq_global_ctx;
}
