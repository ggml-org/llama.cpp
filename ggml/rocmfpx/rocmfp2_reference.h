#ifndef ROCMFP2_REFERENCE_H
#define ROCMFP2_REFERENCE_H

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ROCMFP2_P1_BLOCK_WEIGHTS = 32,
    ROCMFP2_P1_GROUP_WEIGHTS = 16,
    ROCMFP2_P1_DATA_BYTES    = 8,
    ROCMFP2_P1_SCALE_BYTES   = 2,
    ROCMFP2_P1_BLOCK_BYTES   = 10,
};

/*
 * Frozen Phase-1 P0_AOS serialization:
 *
 *     d[0] ... d[7], s[0], s[1]
 *
 * Weight 4*j+i occupies bits 2*i+1:2*i of d[j].  Scale s[0]
 * applies to weights 0..15 and s[1] to weights 16..31.  This is a
 * byte-canonical format; host word endianness is not part of serialization.
 */
typedef struct {
    uint8_t d[ROCMFP2_P1_DATA_BYTES];
    uint8_t s[ROCMFP2_P1_SCALE_BYTES];
} rocmfp2_p1_block;

typedef struct {
    uint8_t inner;
    uint8_t outer;
} rocmfp2_p1_codebook;

typedef enum {
    ROCMFP2_P1_MAPPING_MORD = 0,
    ROCMFP2_P1_MAPPING_MSM  = 1,
} rocmfp2_p1_mapping;

typedef enum {
    ROCMFP2_P1_OK = 0,
    ROCMFP2_P1_NONFINITE_SOURCE,
    ROCMFP2_P1_INVALID_ARGUMENT,
    ROCMFP2_P1_INVALID_CODEBOOK,
    ROCMFP2_P1_INVALID_MAPPING,
    ROCMFP2_P1_INVALID_SCALE_METADATA,
    ROCMFP2_P1_INVALID_CODE,
} rocmfp2_p1_status;

#if defined(__cplusplus)
static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
              "ROCmFP2 Phase-1 reference requires IEEE-754 binary64 double");
static_assert(sizeof(rocmfp2_p1_block) == ROCMFP2_P1_BLOCK_BYTES,
              "ROCmFP2 P0 block has padding");
static_assert(offsetof(rocmfp2_p1_block, d) == 0, "ROCmFP2 P0 data offset changed");
static_assert(offsetof(rocmfp2_p1_block, s) == 8, "ROCmFP2 P0 scale offset changed");
#else
_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "ROCmFP2 Phase-1 reference requires IEEE-754 binary64 double");
_Static_assert(sizeof(rocmfp2_p1_block) == ROCMFP2_P1_BLOCK_BYTES,
               "ROCmFP2 P0 block has padding");
_Static_assert(offsetof(rocmfp2_p1_block, d) == 0, "ROCmFP2 P0 data offset changed");
_Static_assert(offsetof(rocmfp2_p1_block, s) == 8, "ROCmFP2 P0 scale offset changed");
#endif

const char * rocmfp2_p1_status_name(rocmfp2_p1_status status);

bool    rocmfp2_p1_scale_is_valid(uint8_t scale_byte);
double  rocmfp2_p1_ue4m3_to_binary64(uint8_t scale_byte);
/* Returns invalid byte 0xff for NaN or a negative target. */
uint8_t rocmfp2_p1_nearest_ue4m3(double target);

rocmfp2_p1_status rocmfp2_p1_decode_code(
        uint8_t code,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        int8_t * value);

rocmfp2_p1_status rocmfp2_p1_decode_packed_byte(
        uint8_t packed,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        int8_t values[4]);

rocmfp2_p1_status rocmfp2_p1_pack_codes(
        const uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS],
        const uint8_t scales[ROCMFP2_P1_SCALE_BYTES],
        rocmfp2_p1_block * block);

void rocmfp2_p1_unpack_codes(
        const rocmfp2_p1_block * block,
        uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS]);

bool rocmfp2_p1_validate_block(const rocmfp2_p1_block * block);
bool rocmfp2_p1_validate_serialized(const void * data, size_t nbytes);

/*
 * The code selectors are exposed so signed-zero and exact-midpoint behavior
 * can be tested independently of scale selection.  The reference selector
 * exhaustively evaluates all four codes.  The optimized selector uses the
 * symmetric sign/magnitude structure.  Both use binary64 arithmetic.
 */
rocmfp2_p1_status rocmfp2_p1_select_code_ref(
        double source,
        uint8_t scale_byte,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        uint8_t * code);

rocmfp2_p1_status rocmfp2_p1_select_code_optimized(
        double source,
        uint8_t scale_byte,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        uint8_t * code);

/*
 * Both quantizers search every legal UE4M3 byte (0x00..0x7e) independently
 * for each 16-weight group and minimize unweighted SSE in binary64.  Exact
 * scale ties select the lower byte.  Outputs are committed only on success.
 */
rocmfp2_p1_status rocmfp2_p1_quantize_block_ref(
        const double source[ROCMFP2_P1_BLOCK_WEIGHTS],
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        rocmfp2_p1_block * block,
        double group_sse[ROCMFP2_P1_SCALE_BYTES]);

rocmfp2_p1_status rocmfp2_p1_quantize_block_optimized(
        const double source[ROCMFP2_P1_BLOCK_WEIGHTS],
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        rocmfp2_p1_block * block,
        double group_sse[ROCMFP2_P1_SCALE_BYTES]);

rocmfp2_p1_status rocmfp2_p1_dequantize_block(
        const rocmfp2_p1_block * block,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        double output[ROCMFP2_P1_BLOCK_WEIGHTS]);

#ifdef __cplusplus
}
#endif

#endif
