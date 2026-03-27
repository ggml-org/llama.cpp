#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// TurboQuant rotation context
// Holds precomputed rotation matrix for a given (dim, seed) pair.
// The rotation matrix is an orthonormal matrix generated via QR decomposition
// of a seeded random Gaussian matrix (per the TurboQuant paper).

typedef struct turbo_quant_ctx {
    int       dim;         // vector dimension (typically head_dim, e.g. 128)
    uint64_t  seed;        // deterministic seed for rotation generation
    float   * rotation;    // dim x dim orthonormal matrix, row-major
    float   * rotation_t;  // dim x dim transpose (for dequantization)
} turbo_quant_ctx;

// Initialize a TurboQuant context for a given dimension and seed.
// Generates the rotation matrix via QR decomposition.
turbo_quant_ctx * turbo_quant_init(int dim, uint64_t seed);

// Free a TurboQuant context.
void turbo_quant_free(turbo_quant_ctx * ctx);

// Set/get the thread-local TurboQuant context.
// Must be set before calling quantize/dequantize functions for TBQ types.
void              turbo_quant_set_ctx(turbo_quant_ctx * ctx);
turbo_quant_ctx * turbo_quant_get_ctx(void);

#ifdef __cplusplus
}
#endif
