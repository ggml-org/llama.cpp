/**
 * TurboQuant KV Cache Quantization for ggml
 *
 * Implements TQ4_0: Random orthogonal rotation + optimal Beta codebook + Q4_0 packing.
 * Same block layout as Q4_0 (18 bytes per 32 values) but better distortion due to
 * rotation-induced coordinate independence.
 *
 * Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
 *        Zandieh et al., ICLR 2026, arXiv:2504.19874
 *
 * Key insight: After rotation by a random orthogonal matrix, each coordinate of a
 * unit vector follows a concentrated Beta distribution. The optimal quantizer for
 * this distribution achieves near-information-theoretic-optimal distortion.
 */

#ifndef TQ_QUANTS_H
#define TQ_QUANTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TQ4_0 uses block_q4_0 from ggml. We forward-declare what we need
 * to avoid include path issues between ggml-base and ggml-cpu. */
#define QK_TQ4_0 32

/* We use void* for block pointers in the API to avoid needing ggml headers.
 * Internally, we cast to block_q4_0*. */

/**
 * Initialize the rotation matrix for a given head dimension.
 * Call once at model init. Thread-safe after init.
 *
 * @param head_dim  Dimension of KV head vectors (typically 128)
 * @param seed      Random seed for rotation matrix generation
 */
__declspec(dllexport) void tq_init(int head_dim, uint64_t seed);

/** Free rotation matrix memory. */
void tq_free(void);

/** Get the head dimension used for init. Returns 0 if not initialized. */
int tq_head_dim(void);

/**
 * Quantize a row of floats using TurboQuant rotation + Beta-optimal codebook.
 *
 * @param x   Input: head_dim floats (one KV head vector)
 * @param y   Output: ceil(head_dim/QK_TQ4_0) blocks of block_tq4_0
 * @param k   Number of floats (must equal head_dim set in tq_init)
 */
void quantize_row_tq4_0_ref(const float * x, void * y, int64_t k);
void quantize_row_tq4_0(const float * x, void * y, int64_t k);

/**
 * Dequantize a row of TQ4_0 blocks back to floats.
 */
void dequantize_row_tq4_0(const void * x, float * y, int64_t k);

/**
 * Dot product: tq4_0 @ q8_0 (for attention score computation).
 */
void vec_dot_tq4_0_q8_0(int n, float * s, size_t bs,
                         const void * vx, size_t bx,
                         const void * vy, size_t by,
                         int nrc);

/**
 * Batch quantize with optional importance matrix.
 */
size_t quantize_tq4_0(const float * src, void * dst,
                       int64_t nrows, int64_t n_per_row,
                       const float * imatrix);

#ifdef __cplusplus
}
#endif

#endif /* TQ_QUANTS_H */
