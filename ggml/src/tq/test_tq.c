/**
 * Test TurboQuant C implementation.
 *
 * Compile: gcc -O2 -lm -o test_tq test_tq.c tq_quants.c
 * Run: ./test_tq
 */

#include "tq_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DIM 128
#define N_VECTORS 100  /* Reduced for stack safety */

static float randn(void) {
    /* Box-Muller */
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2));
}

int main(void) {
    srand(42);

    printf("TurboQuant C Reference Test\n");
    fflush(stdout);
    printf("===========================\n\n");
    fflush(stdout);

    /* Initialize rotation matrix */
    printf("Calling tq_init...\n"); fflush(stdout);
    tq_init(DIM, 42);
    printf("Initialized: head_dim=%d\n", tq_head_dim()); fflush(stdout);

    /* Generate random unit vectors (heap-allocated to avoid stack overflow) */
    float (*vectors)[DIM] = (float (*)[DIM])malloc(N_VECTORS * DIM * sizeof(float));
    float (*reconstructed)[DIM] = (float (*)[DIM])malloc(N_VECTORS * DIM * sizeof(float));
    for (int v = 0; v < N_VECTORS; v++) {
        float norm = 0;
        for (int i = 0; i < DIM; i++) {
            vectors[v][i] = randn();
            norm += vectors[v][i] * vectors[v][i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < DIM; i++) vectors[v][i] /= norm;
    }

    /* Quantize + dequantize all vectors */
    int nb = DIM / QK_TQ4_0;
    block_tq4_0 (*blocks)[DIM / QK_TQ4_0] = (block_tq4_0 (*)[DIM / QK_TQ4_0])malloc(N_VECTORS * nb * sizeof(block_tq4_0));

    clock_t t0 = clock();
    for (int v = 0; v < N_VECTORS; v++) {
        quantize_row_tq4_0_ref(vectors[v], blocks[v], DIM);
    }
    clock_t t1 = clock();
    for (int v = 0; v < N_VECTORS; v++) {
        dequantize_row_tq4_0(blocks[v], reconstructed[v], DIM);
    }
    clock_t t2 = clock();

    /* Compute MSE */
    double total_mse = 0;
    for (int v = 0; v < N_VECTORS; v++) {
        double mse = 0;
        for (int i = 0; i < DIM; i++) {
            double diff = vectors[v][i] - reconstructed[v][i];
            mse += diff * diff;
        }
        total_mse += mse;
    }
    total_mse /= N_VECTORS;

    /* Theoretical bound for 4-bit: sqrt(3)*pi/2 * (1/4^4) = 0.0106 */
    double theoretical = sqrt(3.0) * 3.14159265 / 2.0 * (1.0 / 256.0);

    printf("\nResults (%d vectors, dim=%d, 4-bit):\n", N_VECTORS, DIM);
    printf("  Empirical MSE:     %.6f\n", total_mse);
    printf("  Theoretical bound: %.6f\n", theoretical);
    printf("  Within 2x bound:   %s\n", total_mse < theoretical * 2.0 ? "YES" : "NO");
    printf("  Quantize time:     %.3f ms\n", 1000.0 * (t1 - t0) / CLOCKS_PER_SEC);
    printf("  Dequantize time:   %.3f ms\n", 1000.0 * (t2 - t1) / CLOCKS_PER_SEC);

    /* Memory calculation */
    size_t compressed_bytes = N_VECTORS * nb * sizeof(block_tq4_0);
    size_t fp16_bytes = N_VECTORS * DIM * 2;
    printf("  Compressed:        %zu bytes (%.1f KB)\n", compressed_bytes, compressed_bytes / 1024.0);
    printf("  FP16 equivalent:   %zu bytes (%.1f KB)\n", fp16_bytes, fp16_bytes / 1024.0);
    printf("  Compression ratio: %.1fx\n", (double)fp16_bytes / compressed_bytes);

    /* Test norm preservation */
    double norm_error = 0;
    for (int v = 0; v < N_VECTORS; v++) {
        float orig_norm = 0, recon_norm = 0;
        for (int i = 0; i < DIM; i++) {
            orig_norm += vectors[v][i] * vectors[v][i];
            recon_norm += reconstructed[v][i] * reconstructed[v][i];
        }
        norm_error += fabs(sqrtf(orig_norm) - sqrtf(recon_norm));
    }
    norm_error /= N_VECTORS;
    printf("  Avg norm error:    %.6f\n", norm_error);

    printf("\n%s\n", total_mse < theoretical * 3.0 ? "ALL TESTS PASSED" : "TESTS FAILED");

    tq_free();
    return total_mse < theoretical * 3.0 ? 0 : 1;
}
