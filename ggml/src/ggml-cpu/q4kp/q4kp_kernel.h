#ifndef Q125_Q4KP_KERNEL_H
#define Q125_Q4KP_KERNEL_H
#include <stddef.h>

#define Q4KP_PACKED_BLOCK_BYTES 1152
#define Q4KP_LAYOUT_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/* Recode existing Q4_K 8x8 interleaved metadata IN PLACE, once after repacking.
 * Each 12-byte metadata group becomes eight contiguous little-endian six-bit
 * scales (6 bytes), followed by eight six-bit minima (6 bytes). Other bytes and
 * block sizes are unchanged. This is a runtime layout, not a GGUF tensor type.
 *
 * Input must still have the ORIGINAL interleaved metadata encoding. The caller
 * must track layout ownership: calling this twice is NOT idempotent and cannot
 * be detected from arbitrary valid metadata bytes.
 *
 * Returns 0 on success, -1 for invalid pointer/size/address wrap, -3 for any
 * non-finite FP16 d/dmin. Validates all blocks before writing anything. Empty
 * input is a valid no-op. Accepts unaligned byte buffers.
 */
int q4kp_recode(void *packed, size_t packed_bytes);

/* Caller must check AVX2, BMI2, FMA and F16C and own the recoded layout.
 * GEMV: n > 0 divisible by 256, nc > 0 divisible by 8, nr == 1; bs unused.
 * GEMM: same n/nc constraints, nr > 0 divisible by 4, bs >= nc in floats;
 * vy is the original block_q8_Kx4 activation layout, not plain block_q8_K.
 * All pointers must be valid, with disjoint input/output ranges. Invalid basic
 * dimensions or null pointers return without writing. Quant block counts and
 * float output allocations follow the original llama.cpp GEMV/GEMM contracts.
 * Standard Q4_K readers MUST NOT consume recoded metadata, including views.
 */
void q4kp_gemv(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
void q4kp_gemm(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);

#ifdef __cplusplus
}
#endif
#endif
