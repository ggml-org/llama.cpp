#include "rocmfp2_reference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool rocmfp2_p1_mapping_is_valid(rocmfp2_p1_mapping mapping) {
    return mapping == ROCMFP2_P1_MAPPING_MORD || mapping == ROCMFP2_P1_MAPPING_MSM;
}

static bool rocmfp2_p1_codebook_is_valid(const rocmfp2_p1_codebook * codebook) {
    return codebook != NULL && codebook->inner > 0 && codebook->outer > codebook->inner && codebook->outer <= 127;
}

static double rocmfp2_p1_decode_scale_valid(uint8_t scale_byte) {
    const int exponent = scale_byte >> 3;
    const int mantissa = scale_byte & 7;

    if (exponent == 0) {
        return ldexp((double) mantissa, -10);
    }

    return ldexp((double) (8 + mantissa), exponent - 11);
}

static int8_t rocmfp2_p1_decode_code_valid(
        uint8_t code,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping) {
    const int inner = (int) codebook->inner;
    const int outer = (int) codebook->outer;

    if (mapping == ROCMFP2_P1_MAPPING_MORD) {
        static const int sign[4] = { -1, -1, 1, 1 };
        const int magnitude = (code == 0 || code == 3) ? outer : inner;
        return (int8_t) (sign[code] * magnitude);
    }

    static const int sign[4] = { 1, 1, -1, -1 };
    const int magnitude = (code == 1 || code == 3) ? outer : inner;
    return (int8_t) (sign[code] * magnitude);
}

static uint8_t rocmfp2_p1_encode_semantic_valid(
        bool negative,
        bool outer,
        rocmfp2_p1_mapping mapping) {
    if (mapping == ROCMFP2_P1_MAPPING_MORD) {
        if (negative) {
            return outer ? 0u : 1u;
        }
        return outer ? 3u : 2u;
    }

    if (negative) {
        return outer ? 3u : 2u;
    }
    return outer ? 1u : 0u;
}

/* Lower ranks implement the frozen semantic tie rules. */
static int rocmfp2_p1_code_tie_rank(
        uint8_t code,
        bool source_negative,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping) {
    const int value = (int) rocmfp2_p1_decode_code_valid(code, codebook, mapping);
    const bool code_negative = value < 0;
    const bool outer = abs(value) == (int) codebook->outer;

    /* Inner magnitude wins first; input sign wins a remaining sign tie. */
    return (outer ? 2 : 0) + (code_negative != source_negative ? 1 : 0);
}

static uint8_t rocmfp2_p1_select_code_ref_valid(
        double source,
        double scale,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping) {
    /* Every code reconstructs exact zero; P0 canonicalizes this tie to 00. */
    if (scale == 0.0) {
        return 0;
    }

    const bool source_negative = signbit(source) != 0;
    double best_error = INFINITY;
    int best_rank = 0;
    uint8_t best_code = 0;
    bool have_best = false;

    for (uint8_t code = 0; code < 4; ++code) {
        const double reconstructed = (double) rocmfp2_p1_decode_code_valid(code, codebook, mapping) * scale;
        const double delta = source - reconstructed;
        const double error = delta * delta;
        const int rank = rocmfp2_p1_code_tie_rank(code, source_negative, codebook, mapping);

        if (!have_best || error < best_error || (error == best_error && rank < best_rank)) {
            best_error = error;
            best_rank = rank;
            best_code = code;
            have_best = true;
        }
    }

    return best_code;
}

static uint8_t rocmfp2_p1_select_code_optimized_valid(
        double source,
        double scale,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping) {
    /* Keep serialization identical to the exhaustive reference at scale 0. */
    if (scale == 0.0) {
        return 0;
    }

    const double magnitude = fabs(source);
    const double inner_reconstructed = (double) codebook->inner * scale;
    const double outer_reconstructed = (double) codebook->outer * scale;
    const double inner_delta = magnitude - inner_reconstructed;
    const double outer_delta = magnitude - outer_reconstructed;

    /* Equality is the exact inner/outer midpoint tie and keeps inner. */
    const bool outer = outer_delta * outer_delta < inner_delta * inner_delta;
    return rocmfp2_p1_encode_semantic_valid(signbit(source) != 0, outer, mapping);
}

const char * rocmfp2_p1_status_name(rocmfp2_p1_status status) {
    switch (status) {
        case ROCMFP2_P1_OK:                     return "OK";
        case ROCMFP2_P1_NONFINITE_SOURCE:       return "NONFINITE_SOURCE";
        case ROCMFP2_P1_INVALID_ARGUMENT:       return "INVALID_ARGUMENT";
        case ROCMFP2_P1_INVALID_CODEBOOK:       return "INVALID_CODEBOOK";
        case ROCMFP2_P1_INVALID_MAPPING:        return "INVALID_MAPPING";
        case ROCMFP2_P1_INVALID_SCALE_METADATA: return "INVALID_SCALE_METADATA";
        case ROCMFP2_P1_INVALID_CODE:           return "INVALID_CODE";
    }

    return "UNKNOWN_STATUS";
}

bool rocmfp2_p1_scale_is_valid(uint8_t scale_byte) {
    return scale_byte <= 0x7e;
}

double rocmfp2_p1_ue4m3_to_binary64(uint8_t scale_byte) {
    if (!rocmfp2_p1_scale_is_valid(scale_byte)) {
        return NAN;
    }

    return rocmfp2_p1_decode_scale_valid(scale_byte);
}

uint8_t rocmfp2_p1_nearest_ue4m3(double target) {
    if (isnan(target) || target < 0.0) {
        return 0xff;
    }
    if (target == 0.0) {
        return 0;
    }

    const double maximum = rocmfp2_p1_decode_scale_valid(0x7e);
    if (!isfinite(target) || target >= maximum) {
        return 0x7e;
    }

    for (uint8_t upper_byte = 1; upper_byte <= 0x7e; ++upper_byte) {
        const double upper = rocmfp2_p1_decode_scale_valid(upper_byte);
        if (target <= upper) {
            const uint8_t lower_byte = (uint8_t) (upper_byte - 1);
            const double lower = rocmfp2_p1_decode_scale_valid(lower_byte);

            /* Exact midpoint ties select the smaller scale byte. */
            return target - lower <= upper - target ? lower_byte : upper_byte;
        }
    }

    return 0x7e;
}

rocmfp2_p1_status rocmfp2_p1_decode_code(
        uint8_t code,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        int8_t * value) {
    if (value == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_codebook_is_valid(codebook)) {
        return ROCMFP2_P1_INVALID_CODEBOOK;
    }
    if (!rocmfp2_p1_mapping_is_valid(mapping)) {
        return ROCMFP2_P1_INVALID_MAPPING;
    }
    if (code > 3) {
        return ROCMFP2_P1_INVALID_CODE;
    }

    *value = rocmfp2_p1_decode_code_valid(code, codebook, mapping);
    return ROCMFP2_P1_OK;
}

rocmfp2_p1_status rocmfp2_p1_decode_packed_byte(
        uint8_t packed,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        int8_t values[4]) {
    if (values == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_codebook_is_valid(codebook)) {
        return ROCMFP2_P1_INVALID_CODEBOOK;
    }
    if (!rocmfp2_p1_mapping_is_valid(mapping)) {
        return ROCMFP2_P1_INVALID_MAPPING;
    }

    for (int lane = 0; lane < 4; ++lane) {
        const uint8_t code = (uint8_t) ((packed >> (2 * lane)) & 3u);
        values[lane] = rocmfp2_p1_decode_code_valid(code, codebook, mapping);
    }

    return ROCMFP2_P1_OK;
}

rocmfp2_p1_status rocmfp2_p1_pack_codes(
        const uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS],
        const uint8_t scales[ROCMFP2_P1_SCALE_BYTES],
        rocmfp2_p1_block * block) {
    if (codes == NULL || scales == NULL || block == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_scale_is_valid(scales[0]) || !rocmfp2_p1_scale_is_valid(scales[1])) {
        return ROCMFP2_P1_INVALID_SCALE_METADATA;
    }

    rocmfp2_p1_block temporary;
    for (int packed_index = 0; packed_index < ROCMFP2_P1_DATA_BYTES; ++packed_index) {
        uint8_t packed = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const uint8_t code = codes[4 * packed_index + lane];
            if (code > 3) {
                return ROCMFP2_P1_INVALID_CODE;
            }
            packed |= (uint8_t) (code << (2 * lane));
        }
        temporary.d[packed_index] = packed;
    }
    temporary.s[0] = scales[0];
    temporary.s[1] = scales[1];

    memcpy(block, &temporary, sizeof(temporary));
    return ROCMFP2_P1_OK;
}

void rocmfp2_p1_unpack_codes(
        const rocmfp2_p1_block * block,
        uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS]) {
    for (int packed_index = 0; packed_index < ROCMFP2_P1_DATA_BYTES; ++packed_index) {
        const uint8_t packed = block->d[packed_index];
        for (int lane = 0; lane < 4; ++lane) {
            codes[4 * packed_index + lane] = (uint8_t) ((packed >> (2 * lane)) & 3u);
        }
    }
}

bool rocmfp2_p1_validate_block(const rocmfp2_p1_block * block) {
    return block != NULL && rocmfp2_p1_scale_is_valid(block->s[0]) && rocmfp2_p1_scale_is_valid(block->s[1]);
}

bool rocmfp2_p1_validate_serialized(const void * data, size_t nbytes) {
    if (nbytes == 0) {
        return true;
    }
    if (data == NULL || nbytes % ROCMFP2_P1_BLOCK_BYTES != 0) {
        return false;
    }

    const uint8_t * bytes = (const uint8_t *) data;
    const size_t blocks = nbytes / ROCMFP2_P1_BLOCK_BYTES;
    for (size_t block_index = 0; block_index < blocks; ++block_index) {
        const size_t base = block_index * ROCMFP2_P1_BLOCK_BYTES;
        if (!rocmfp2_p1_scale_is_valid(bytes[base + 8]) || !rocmfp2_p1_scale_is_valid(bytes[base + 9])) {
            return false;
        }
    }

    return true;
}

rocmfp2_p1_status rocmfp2_p1_select_code_ref(
        double source,
        uint8_t scale_byte,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        uint8_t * code) {
    if (code == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_codebook_is_valid(codebook)) {
        return ROCMFP2_P1_INVALID_CODEBOOK;
    }
    if (!rocmfp2_p1_mapping_is_valid(mapping)) {
        return ROCMFP2_P1_INVALID_MAPPING;
    }
    if (!rocmfp2_p1_scale_is_valid(scale_byte)) {
        return ROCMFP2_P1_INVALID_SCALE_METADATA;
    }
    if (!isfinite(source)) {
        return ROCMFP2_P1_NONFINITE_SOURCE;
    }

    *code = rocmfp2_p1_select_code_ref_valid(
            source, rocmfp2_p1_decode_scale_valid(scale_byte), codebook, mapping);
    return ROCMFP2_P1_OK;
}

rocmfp2_p1_status rocmfp2_p1_select_code_optimized(
        double source,
        uint8_t scale_byte,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        uint8_t * code) {
    if (code == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_codebook_is_valid(codebook)) {
        return ROCMFP2_P1_INVALID_CODEBOOK;
    }
    if (!rocmfp2_p1_mapping_is_valid(mapping)) {
        return ROCMFP2_P1_INVALID_MAPPING;
    }
    if (!rocmfp2_p1_scale_is_valid(scale_byte)) {
        return ROCMFP2_P1_INVALID_SCALE_METADATA;
    }
    if (!isfinite(source)) {
        return ROCMFP2_P1_NONFINITE_SOURCE;
    }

    *code = rocmfp2_p1_select_code_optimized_valid(
            source, rocmfp2_p1_decode_scale_valid(scale_byte), codebook, mapping);
    return ROCMFP2_P1_OK;
}

typedef uint8_t (*rocmfp2_p1_selector)(
        double source,
        double scale,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping);

static rocmfp2_p1_status rocmfp2_p1_quantize_block_impl(
        const double source[ROCMFP2_P1_BLOCK_WEIGHTS],
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        rocmfp2_p1_block * block,
        double group_sse[ROCMFP2_P1_SCALE_BYTES],
        rocmfp2_p1_selector selector) {
    if (source == NULL || block == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_codebook_is_valid(codebook)) {
        return ROCMFP2_P1_INVALID_CODEBOOK;
    }
    if (!rocmfp2_p1_mapping_is_valid(mapping)) {
        return ROCMFP2_P1_INVALID_MAPPING;
    }

    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        if (!isfinite(source[i])) {
            return ROCMFP2_P1_NONFINITE_SOURCE;
        }
    }

    uint8_t best_codes[ROCMFP2_P1_BLOCK_WEIGHTS];
    uint8_t best_scales[ROCMFP2_P1_SCALE_BYTES];
    double best_group_sse[ROCMFP2_P1_SCALE_BYTES];

    for (int group = 0; group < ROCMFP2_P1_SCALE_BYTES; ++group) {
        const int source_offset = group * ROCMFP2_P1_GROUP_WEIGHTS;
        double best_sse = INFINITY;
        uint8_t best_scale = 0;
        uint8_t candidate_codes[ROCMFP2_P1_GROUP_WEIGHTS];
        uint8_t saved_codes[ROCMFP2_P1_GROUP_WEIGHTS];
        bool have_best = false;

        /* Exact exhaustive search: every legal finite unsigned UE4M3 byte. */
        for (int scale_index = 0; scale_index <= 0x7e; ++scale_index) {
            const uint8_t scale_byte = (uint8_t) scale_index;
            const double scale = rocmfp2_p1_decode_scale_valid(scale_byte);
            double candidate_sse = 0.0;

            for (int i = 0; i < ROCMFP2_P1_GROUP_WEIGHTS; ++i) {
                const double value = source[source_offset + i];
                const uint8_t code = selector(value, scale, codebook, mapping);
                const double reconstructed = (double) rocmfp2_p1_decode_code_valid(code, codebook, mapping) * scale;
                const double delta = value - reconstructed;

                candidate_codes[i] = code;
                candidate_sse += delta * delta;
            }

            if (!have_best || candidate_sse < best_sse ||
                    (candidate_sse == best_sse && scale_byte < best_scale)) {
                best_sse = candidate_sse;
                best_scale = scale_byte;
                memcpy(saved_codes, candidate_codes, sizeof(saved_codes));
                have_best = true;
            }
        }

        memcpy(best_codes + source_offset, saved_codes, sizeof(saved_codes));
        best_scales[group] = best_scale;
        best_group_sse[group] = best_sse;
    }

    rocmfp2_p1_block temporary;
    const rocmfp2_p1_status pack_status = rocmfp2_p1_pack_codes(best_codes, best_scales, &temporary);
    if (pack_status != ROCMFP2_P1_OK) {
        return pack_status;
    }

    memcpy(block, &temporary, sizeof(temporary));
    if (group_sse != NULL) {
        group_sse[0] = best_group_sse[0];
        group_sse[1] = best_group_sse[1];
    }

    return ROCMFP2_P1_OK;
}

rocmfp2_p1_status rocmfp2_p1_quantize_block_ref(
        const double source[ROCMFP2_P1_BLOCK_WEIGHTS],
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        rocmfp2_p1_block * block,
        double group_sse[ROCMFP2_P1_SCALE_BYTES]) {
    return rocmfp2_p1_quantize_block_impl(
            source, codebook, mapping, block, group_sse, rocmfp2_p1_select_code_ref_valid);
}

rocmfp2_p1_status rocmfp2_p1_quantize_block_optimized(
        const double source[ROCMFP2_P1_BLOCK_WEIGHTS],
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        rocmfp2_p1_block * block,
        double group_sse[ROCMFP2_P1_SCALE_BYTES]) {
    return rocmfp2_p1_quantize_block_impl(
            source, codebook, mapping, block, group_sse, rocmfp2_p1_select_code_optimized_valid);
}

rocmfp2_p1_status rocmfp2_p1_dequantize_block(
        const rocmfp2_p1_block * block,
        const rocmfp2_p1_codebook * codebook,
        rocmfp2_p1_mapping mapping,
        double output[ROCMFP2_P1_BLOCK_WEIGHTS]) {
    if (block == NULL || output == NULL) {
        return ROCMFP2_P1_INVALID_ARGUMENT;
    }
    if (!rocmfp2_p1_codebook_is_valid(codebook)) {
        return ROCMFP2_P1_INVALID_CODEBOOK;
    }
    if (!rocmfp2_p1_mapping_is_valid(mapping)) {
        return ROCMFP2_P1_INVALID_MAPPING;
    }
    if (!rocmfp2_p1_validate_block(block)) {
        return ROCMFP2_P1_INVALID_SCALE_METADATA;
    }

    uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS];
    double temporary[ROCMFP2_P1_BLOCK_WEIGHTS];
    rocmfp2_p1_unpack_codes(block, codes);

    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        const int group = i / ROCMFP2_P1_GROUP_WEIGHTS;
        const double scale = rocmfp2_p1_decode_scale_valid(block->s[group]);
        const int8_t value = rocmfp2_p1_decode_code_valid(codes[i], codebook, mapping);
        temporary[i] = (double) value * scale;
    }

    memcpy(output, temporary, sizeof(temporary));
    return ROCMFP2_P1_OK;
}
