#include "rocmfp2_reference.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static unsigned long long test_checks = 0;

#define CHECK(condition) do {                                                                  \
    ++test_checks;                                                                             \
    if (!(condition)) {                                                                        \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                  \
        return false;                                                                          \
    }                                                                                          \
} while (0)

#define CHECK_STATUS(expression, expected) do {                                                 \
    const rocmfp2_p1_status actual_status_ = (expression);                                      \
    ++test_checks;                                                                             \
    if (actual_status_ != (expected)) {                                                         \
        fprintf(stderr, "FAIL %s:%d: %s returned %s, expected %s\n",                          \
                __FILE__, __LINE__, #expression, rocmfp2_p1_status_name(actual_status_),        \
                rocmfp2_p1_status_name(expected));                                              \
        return false;                                                                          \
    }                                                                                          \
} while (0)

static int8_t expected_code_value(
        uint8_t code,
        rocmfp2_p1_codebook codebook,
        rocmfp2_p1_mapping mapping) {
    if (mapping == ROCMFP2_P1_MAPPING_MORD) {
        const int values[4] = {
            -(int) codebook.outer,
            -(int) codebook.inner,
             (int) codebook.inner,
             (int) codebook.outer,
        };
        return (int8_t) values[code];
    }

    const int values[4] = {
         (int) codebook.inner,
         (int) codebook.outer,
        -(int) codebook.inner,
        -(int) codebook.outer,
    };
    return (int8_t) values[code];
}

static uint8_t expected_semantic_code(bool negative, bool outer, rocmfp2_p1_mapping mapping) {
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

static bool test_layout_and_p0_packing(void) {
    CHECK(sizeof(rocmfp2_p1_block) == 10);
    CHECK(offsetof(rocmfp2_p1_block, d) == 0);
    CHECK(offsetof(rocmfp2_p1_block, s) == 8);

    uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS];
    uint8_t unpacked[ROCMFP2_P1_BLOCK_WEIGHTS];
    const uint8_t scales[2] = { 0x12, 0x7e };
    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        codes[i] = (uint8_t) (i & 3);
    }

    rocmfp2_p1_block block;
    memset(&block, 0xa5, sizeof(block));
    CHECK_STATUS(rocmfp2_p1_pack_codes(codes, scales, &block), ROCMFP2_P1_OK);

    const uint8_t expected_bytes[ROCMFP2_P1_BLOCK_BYTES] = {
        0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0x12, 0x7e,
    };
    CHECK(memcmp(&block, expected_bytes, sizeof(expected_bytes)) == 0);

    rocmfp2_p1_unpack_codes(&block, unpacked);
    CHECK(memcmp(codes, unpacked, sizeof(codes)) == 0);

    rocmfp2_p1_block unchanged;
    rocmfp2_p1_block before;
    memset(&unchanged, 0x5a, sizeof(unchanged));
    memcpy(&before, &unchanged, sizeof(before));
    codes[31] = 4;
    CHECK_STATUS(rocmfp2_p1_pack_codes(codes, scales, &unchanged), ROCMFP2_P1_INVALID_CODE);
    CHECK(memcmp(&unchanged, &before, sizeof(before)) == 0);
    codes[31] = 3;

    const uint8_t invalid_scales[2] = { 0x7f, 0x00 };
    CHECK_STATUS(rocmfp2_p1_pack_codes(codes, invalid_scales, &unchanged),
                 ROCMFP2_P1_INVALID_SCALE_METADATA);
    CHECK(memcmp(&unchanged, &before, sizeof(before)) == 0);

    puts("PASS layout_p0: block=10 data=8 scales=2 bpw=2.50 golden=e4e4e4e4e4e4e4e4127e");
    return true;
}

static bool test_all_packed_bytes(void) {
    const rocmfp2_p1_codebook codebook = { 3, 10 };

    for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
        const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
        for (int packed_value = 0; packed_value <= 0xff; ++packed_value) {
            int8_t decoded[4];
            CHECK_STATUS(rocmfp2_p1_decode_packed_byte(
                    (uint8_t) packed_value, &codebook, mapping, decoded), ROCMFP2_P1_OK);

            for (int lane = 0; lane < 4; ++lane) {
                const uint8_t code = (uint8_t) ((packed_value >> (2 * lane)) & 3);
                CHECK(decoded[lane] == expected_code_value(code, codebook, mapping));

                int8_t scalar = 0;
                CHECK_STATUS(rocmfp2_p1_decode_code(code, &codebook, mapping, &scalar), ROCMFP2_P1_OK);
                CHECK(scalar == decoded[lane]);
            }
        }
    }

    int8_t value = 0;
    CHECK_STATUS(rocmfp2_p1_decode_code(4, &codebook, ROCMFP2_P1_MAPPING_MORD, &value),
                 ROCMFP2_P1_INVALID_CODE);

    puts("PASS packed_byte_decode: mappings=2 bytes=256 lanes=2048");
    return true;
}

static double expected_scale(uint8_t scale_byte) {
    const int exponent = scale_byte >> 3;
    const int mantissa = scale_byte & 7;
    return exponent == 0 ? scalbn((double) mantissa, -10) : scalbn((double) (8 + mantissa), exponent - 11);
}

static bool test_scale_encoding_and_boundaries(void) {
    double previous = -1.0;
    for (int scale_index = 0; scale_index <= 0xff; ++scale_index) {
        const uint8_t scale_byte = (uint8_t) scale_index;
        const bool valid = scale_index <= 0x7e;
        CHECK(rocmfp2_p1_scale_is_valid(scale_byte) == valid);

        const double decoded = rocmfp2_p1_ue4m3_to_binary64(scale_byte);
        if (valid) {
            CHECK(decoded == expected_scale(scale_byte));
            CHECK(decoded > previous);
            CHECK(rocmfp2_p1_nearest_ue4m3(decoded) == scale_byte);
            previous = decoded;
        } else {
            CHECK(isnan(decoded));
        }
    }

    CHECK(rocmfp2_p1_ue4m3_to_binary64(0x00) == 0.0);
    CHECK(rocmfp2_p1_ue4m3_to_binary64(0x01) == 0x1p-10);
    CHECK(rocmfp2_p1_ue4m3_to_binary64(0x07) == 7.0 * 0x1p-10);
    CHECK(rocmfp2_p1_ue4m3_to_binary64(0x08) == 8.0 * 0x1p-10);
    CHECK(rocmfp2_p1_ue4m3_to_binary64(0x7e) == 224.0);

    for (int upper_index = 1; upper_index <= 0x7e; ++upper_index) {
        const uint8_t upper_byte = (uint8_t) upper_index;
        const uint8_t lower_byte = (uint8_t) (upper_index - 1);
        const double lower = expected_scale(lower_byte);
        const double upper = expected_scale(upper_byte);
        const double midpoint = (lower + upper) * 0.5;

        CHECK(rocmfp2_p1_nearest_ue4m3(midpoint) == lower_byte);
        CHECK(rocmfp2_p1_nearest_ue4m3(nextafter(midpoint, -INFINITY)) == lower_byte);
        CHECK(rocmfp2_p1_nearest_ue4m3(nextafter(midpoint, INFINITY)) == upper_byte);
    }

    CHECK(rocmfp2_p1_nearest_ue4m3(-1.0) == 0xff);
    CHECK(rocmfp2_p1_nearest_ue4m3(NAN) == 0xff);
    CHECK(rocmfp2_p1_nearest_ue4m3(nextafter(224.0, INFINITY)) == 0x7e);
    CHECK(rocmfp2_p1_nearest_ue4m3(INFINITY) == 0x7e);

    puts("PASS ue4m3_boundaries: legal=127 invalid=129 adjacent_midpoints=126 max=224");
    return true;
}

static bool test_metadata_validation(void) {
    rocmfp2_p1_block block;
    memset(&block, 0, sizeof(block));

    for (int first = 0; first <= 0x7e; ++first) {
        for (int second = 0; second <= 0x7e; ++second) {
            block.s[0] = (uint8_t) first;
            block.s[1] = (uint8_t) second;
            CHECK(rocmfp2_p1_validate_block(&block));
        }
    }

    for (int invalid = 0x7f; invalid <= 0xff; ++invalid) {
        block.s[0] = (uint8_t) invalid;
        block.s[1] = 0;
        CHECK(!rocmfp2_p1_validate_block(&block));
        block.s[0] = 0;
        block.s[1] = (uint8_t) invalid;
        CHECK(!rocmfp2_p1_validate_block(&block));
    }

    uint8_t serialized[2 * ROCMFP2_P1_BLOCK_BYTES];
    memset(serialized, 0, sizeof(serialized));
    serialized[8] = 0x7e;
    serialized[9] = 0x00;
    serialized[18] = 0x01;
    serialized[19] = 0x7e;
    CHECK(rocmfp2_p1_validate_serialized(serialized, sizeof(serialized)));
    CHECK(rocmfp2_p1_validate_serialized(NULL, 0));
    CHECK(!rocmfp2_p1_validate_serialized(NULL, sizeof(rocmfp2_p1_block)));

    for (size_t length = 1; length < sizeof(serialized); ++length) {
        if (length != ROCMFP2_P1_BLOCK_BYTES) {
            CHECK(!rocmfp2_p1_validate_serialized(serialized, length));
        }
    }

    for (int invalid = 0x7f; invalid <= 0xff; ++invalid) {
        serialized[8] = (uint8_t) invalid;
        CHECK(!rocmfp2_p1_validate_serialized(serialized, sizeof(serialized)));
        serialized[8] = 0x00;
        serialized[19] = (uint8_t) invalid;
        CHECK(!rocmfp2_p1_validate_serialized(serialized, sizeof(serialized)));
        serialized[19] = 0x7e;
    }

    const rocmfp2_p1_codebook codebook = { 3, 10 };
    double output[ROCMFP2_P1_BLOCK_WEIGHTS];
    double before[ROCMFP2_P1_BLOCK_WEIGHTS];
    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        output[i] = 1234.0 + (double) i;
    }
    memcpy(before, output, sizeof(before));
    block.s[0] = 0x7f;
    block.s[1] = 0;
    CHECK_STATUS(rocmfp2_p1_dequantize_block(
            &block, &codebook, ROCMFP2_P1_MAPPING_MORD, output), ROCMFP2_P1_INVALID_SCALE_METADATA);
    CHECK(memcmp(output, before, sizeof(output)) == 0);

    puts("PASS metadata_validation: legal_pairs=16129 invalid_bytes=129 malformed_lengths=18");
    return true;
}

static bool test_zero_midpoint_and_scale_ties(void) {
    const rocmfp2_p1_codebook codebook = { 3, 10 };
    const uint8_t scale_byte = 0x31;
    const double scale = expected_scale(scale_byte);
    const double midpoint = ((double) codebook.inner + (double) codebook.outer) * scale * 0.5;

    for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
        const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
        const double probes[] = {
            0.0,
            -0.0,
            midpoint,
            -midpoint,
            nextafter(midpoint, INFINITY),
            nextafter(-midpoint, -INFINITY),
        };
        const uint8_t expected[] = {
            expected_semantic_code(false, false, mapping),
            expected_semantic_code(true,  false, mapping),
            expected_semantic_code(false, false, mapping),
            expected_semantic_code(true,  false, mapping),
            expected_semantic_code(false, true,  mapping),
            expected_semantic_code(true,  true,  mapping),
        };

        for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
            uint8_t reference_code = 0xff;
            uint8_t optimized_code = 0xff;
            CHECK_STATUS(rocmfp2_p1_select_code_ref(
                    probes[i], scale_byte, &codebook, mapping, &reference_code), ROCMFP2_P1_OK);
            CHECK_STATUS(rocmfp2_p1_select_code_optimized(
                    probes[i], scale_byte, &codebook, mapping, &optimized_code), ROCMFP2_P1_OK);
            CHECK(reference_code == expected[i]);
            CHECK(optimized_code == expected[i]);
        }

        uint8_t zero_scale_positive = 0xff;
        uint8_t zero_scale_negative = 0xff;
        CHECK_STATUS(rocmfp2_p1_select_code_ref(
                1.0, 0, &codebook, mapping, &zero_scale_positive), ROCMFP2_P1_OK);
        CHECK_STATUS(rocmfp2_p1_select_code_optimized(
                -1.0, 0, &codebook, mapping, &zero_scale_negative), ROCMFP2_P1_OK);
        CHECK(zero_scale_positive == 0);
        CHECK(zero_scale_negative == 0);
    }

    const rocmfp2_p1_codebook tie_codebook = { 1, 2 };
    double source[ROCMFP2_P1_BLOCK_WEIGHTS];
    const double group0_value = 2.0 * expected_scale(0x20);
    const double group1_value = 2.0 * expected_scale(0x21);
    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        const double magnitude = i < ROCMFP2_P1_GROUP_WEIGHTS ? group0_value : group1_value;
        source[i] = (i & 1) ? -magnitude : magnitude;
    }

    for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
        const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
        rocmfp2_p1_block reference;
        rocmfp2_p1_block optimized;
        double reference_sse[2];
        double optimized_sse[2];
        CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
                source, &tie_codebook, mapping, &reference, reference_sse), ROCMFP2_P1_OK);
        CHECK_STATUS(rocmfp2_p1_quantize_block_optimized(
                source, &tie_codebook, mapping, &optimized, optimized_sse), ROCMFP2_P1_OK);
        CHECK(memcmp(&reference, &optimized, sizeof(reference)) == 0);
        CHECK(reference.s[0] == 0x20);
        CHECK(reference.s[1] == 0x21);
        CHECK(reference_sse[0] == 0.0 && reference_sse[1] == 0.0);
        CHECK(optimized_sse[0] == 0.0 && optimized_sse[1] == 0.0);
    }

    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        source[i] = (i & 1) ? -0.0 : 0.0;
    }
    for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
        const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
        rocmfp2_p1_block reference;
        rocmfp2_p1_block optimized;
        uint8_t codes[ROCMFP2_P1_BLOCK_WEIGHTS];
        CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
                source, &codebook, mapping, &reference, NULL), ROCMFP2_P1_OK);
        CHECK_STATUS(rocmfp2_p1_quantize_block_optimized(
                source, &codebook, mapping, &optimized, NULL), ROCMFP2_P1_OK);
        CHECK(memcmp(&reference, &optimized, sizeof(reference)) == 0);
        CHECK(reference.s[0] == 0 && reference.s[1] == 0);
        rocmfp2_p1_unpack_codes(&reference, codes);
        for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
            CHECK(codes[i] == 0);
        }
        const uint8_t canonical_zero[ROCMFP2_P1_BLOCK_BYTES] = { 0 };
        CHECK(memcmp(&reference, canonical_zero, sizeof(canonical_zero)) == 0);
    }

    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        source[i] = (i & 1) ? -10000.0 : 10000.0;
    }
    rocmfp2_p1_block saturated;
    CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
            source, &codebook, ROCMFP2_P1_MAPPING_MORD, &saturated, NULL), ROCMFP2_P1_OK);
    CHECK(saturated.s[0] == 0x7e && saturated.s[1] == 0x7e);

    puts("PASS deterministic_ties: signed_zero=2 magnitude_midpoint=4 scale_tuple=2 saturation=0x7e");
    return true;
}

static bool test_python_golden_vectors(void) {
    const rocmfp2_p1_codebook codebook = { 3, 10 };
    double source[ROCMFP2_P1_BLOCK_WEIGHTS];
    static const double values[4] = { -10.0, -3.0, 3.0, 10.0 };
    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        source[i] = values[i & 3] * (i < ROCMFP2_P1_GROUP_WEIGHTS ? 1.0 : 2.0);
    }

    static const uint8_t expected[2][ROCMFP2_P1_BLOCK_BYTES] = {
        { 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0xe4, 0x40, 0x48 },
        { 0x4b, 0x4b, 0x4b, 0x4b, 0x4b, 0x4b, 0x4b, 0x4b, 0x40, 0x48 },
    };

    for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
        const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
        rocmfp2_p1_block reference;
        rocmfp2_p1_block optimized;
        double output[ROCMFP2_P1_BLOCK_WEIGHTS];
        CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
                source, &codebook, mapping, &reference, NULL), ROCMFP2_P1_OK);
        CHECK_STATUS(rocmfp2_p1_quantize_block_optimized(
                source, &codebook, mapping, &optimized, NULL), ROCMFP2_P1_OK);
        CHECK(memcmp(&reference, expected[mapping_index], sizeof(reference)) == 0);
        CHECK(memcmp(&optimized, expected[mapping_index], sizeof(optimized)) == 0);
        CHECK_STATUS(rocmfp2_p1_dequantize_block(
                &reference, &codebook, mapping, output), ROCMFP2_P1_OK);
        CHECK(memcmp(output, source, sizeof(source)) == 0);
    }

    puts("PASS python_golden_vectors: vectors=2 mord=e4x8+4048 msm=4bx8+4048");
    return true;
}

static uint64_t splitmix64_next(uint64_t * state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t fnv1a64(uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void fill_fixed_random_block(double source[ROCMFP2_P1_BLOCK_WEIGHTS], uint64_t * state, int block_index) {
    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        const uint64_t bits = splitmix64_next(state);
        const double unit = (double) (bits >> 11) * 0x1p-53;
        const int exponent = (int) ((bits >> 3) % 18) - 10;
        source[i] = scalbn(2.0 * unit - 1.0, exponent);
    }

    source[(block_index * 7) & 31] = (block_index & 1) ? -0.0 : 0.0;
    source[(block_index * 11 + 3) & 31] *= 13.0;
}

static bool test_fixed_random_reference_optimized_match(void) {
    static const rocmfp2_p1_codebook codebooks[8] = {
        { 1, 2 }, { 2, 5 }, { 1, 3 }, { 3, 10 },
        { 2, 7 }, { 1, 4 }, { 3, 13 }, { 1, 5 },
    };
    enum { BLOCKS_PER_CODEBOOK = 24 };

    uint64_t random_state = UINT64_C(0x726f636d66703231);
    uint64_t fingerprint = UINT64_C(1469598103934665603);

    for (int codebook_index = 0; codebook_index < 8; ++codebook_index) {
        const rocmfp2_p1_codebook * codebook = &codebooks[codebook_index];
        for (int block_index = 0; block_index < BLOCKS_PER_CODEBOOK; ++block_index) {
            double source[ROCMFP2_P1_BLOCK_WEIGHTS];
            rocmfp2_p1_block mapping_blocks[2];
            double mapping_outputs[2][ROCMFP2_P1_BLOCK_WEIGHTS];
            double mapping_sse[2][2];
            fill_fixed_random_block(source, &random_state, block_index);

            for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
                const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
                rocmfp2_p1_block reference;
                rocmfp2_p1_block optimized;
                rocmfp2_p1_block reference_repeat;
                rocmfp2_p1_block optimized_repeat;
                double reference_sse[2];
                double optimized_sse[2];
                double repeat_sse[2];

                CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
                        source, codebook, mapping, &reference, reference_sse), ROCMFP2_P1_OK);
                CHECK_STATUS(rocmfp2_p1_quantize_block_optimized(
                        source, codebook, mapping, &optimized, optimized_sse), ROCMFP2_P1_OK);
                CHECK(memcmp(&reference, &optimized, sizeof(reference)) == 0);
                CHECK(reference_sse[0] == optimized_sse[0]);
                CHECK(reference_sse[1] == optimized_sse[1]);

                CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
                        source, codebook, mapping, &reference_repeat, repeat_sse), ROCMFP2_P1_OK);
                CHECK(memcmp(&reference, &reference_repeat, sizeof(reference)) == 0);
                CHECK(reference_sse[0] == repeat_sse[0] && reference_sse[1] == repeat_sse[1]);

                CHECK_STATUS(rocmfp2_p1_quantize_block_optimized(
                        source, codebook, mapping, &optimized_repeat, repeat_sse), ROCMFP2_P1_OK);
                CHECK(memcmp(&optimized, &optimized_repeat, sizeof(optimized)) == 0);
                CHECK(optimized_sse[0] == repeat_sse[0] && optimized_sse[1] == repeat_sse[1]);
                CHECK(rocmfp2_p1_validate_block(&reference));

                CHECK_STATUS(rocmfp2_p1_dequantize_block(
                        &reference, codebook, mapping, mapping_outputs[mapping_index]), ROCMFP2_P1_OK);
                for (int group = 0; group < 2; ++group) {
                    double recomputed_sse = 0.0;
                    for (int i = 0; i < ROCMFP2_P1_GROUP_WEIGHTS; ++i) {
                        const int index = group * ROCMFP2_P1_GROUP_WEIGHTS + i;
                        const double delta = source[index] - mapping_outputs[mapping_index][index];
                        recomputed_sse += delta * delta;
                    }
                    CHECK(recomputed_sse == reference_sse[group]);
                    mapping_sse[mapping_index][group] = reference_sse[group];
                }

                memcpy(&mapping_blocks[mapping_index], &reference, sizeof(reference));
                fingerprint = fnv1a64(fingerprint, &reference, sizeof(reference));
            }

            CHECK(mapping_blocks[0].s[0] == mapping_blocks[1].s[0]);
            CHECK(mapping_blocks[0].s[1] == mapping_blocks[1].s[1]);
            CHECK(mapping_sse[0][0] == mapping_sse[1][0]);
            CHECK(mapping_sse[0][1] == mapping_sse[1][1]);
            CHECK(memcmp(mapping_outputs[0], mapping_outputs[1], sizeof(mapping_outputs[0])) == 0);
        }
    }

    printf("PASS fixed_random: codebooks=8 blocks_each=%d mappings=2 ref_opt_equal=384 fingerprint=%016" PRIx64 "\n",
           BLOCKS_PER_CODEBOOK, fingerprint);
    return true;
}

static bool test_nonfinite_rejection_and_output_atomicity(void) {
    const rocmfp2_p1_codebook codebook = { 3, 10 };
    double source[ROCMFP2_P1_BLOCK_WEIGHTS];
    for (int i = 0; i < ROCMFP2_P1_BLOCK_WEIGHTS; ++i) {
        source[i] = (double) i * 0.125 - 2.0;
    }

    const double nonfinite_values[3] = { NAN, INFINITY, -INFINITY };
    for (int mapping_index = 0; mapping_index < 2; ++mapping_index) {
        const rocmfp2_p1_mapping mapping = (rocmfp2_p1_mapping) mapping_index;
        for (int position = 0; position < ROCMFP2_P1_BLOCK_WEIGHTS; ++position) {
            const double saved = source[position];
            for (int variant = 0; variant < 3; ++variant) {
                rocmfp2_p1_block reference;
                rocmfp2_p1_block optimized;
                rocmfp2_p1_block before_reference;
                rocmfp2_p1_block before_optimized;
                double reference_sse[2] = { 123.0, 456.0 };
                double optimized_sse[2] = { 789.0, 987.0 };
                const double reference_sse_before[2] = { 123.0, 456.0 };
                const double optimized_sse_before[2] = { 789.0, 987.0 };

                source[position] = nonfinite_values[variant];
                memset(&reference, 0x3c, sizeof(reference));
                memset(&optimized, 0xc3, sizeof(optimized));
                memcpy(&before_reference, &reference, sizeof(reference));
                memcpy(&before_optimized, &optimized, sizeof(optimized));

                CHECK_STATUS(rocmfp2_p1_quantize_block_ref(
                        source, &codebook, mapping, &reference, reference_sse),
                        ROCMFP2_P1_NONFINITE_SOURCE);
                CHECK_STATUS(rocmfp2_p1_quantize_block_optimized(
                        source, &codebook, mapping, &optimized, optimized_sse),
                        ROCMFP2_P1_NONFINITE_SOURCE);
                CHECK(memcmp(&reference, &before_reference, sizeof(reference)) == 0);
                CHECK(memcmp(&optimized, &before_optimized, sizeof(optimized)) == 0);
                CHECK(memcmp(reference_sse, reference_sse_before, sizeof(reference_sse)) == 0);
                CHECK(memcmp(optimized_sse, optimized_sse_before, sizeof(optimized_sse)) == 0);
            }
            source[position] = saved;
        }
    }

    uint8_t code = 0;
    CHECK_STATUS(rocmfp2_p1_select_code_ref(
            NAN, 0x10, &codebook, ROCMFP2_P1_MAPPING_MORD, &code), ROCMFP2_P1_NONFINITE_SOURCE);
    CHECK_STATUS(rocmfp2_p1_select_code_optimized(
            INFINITY, 0x10, &codebook, ROCMFP2_P1_MAPPING_MSM, &code), ROCMFP2_P1_NONFINITE_SOURCE);

    puts("PASS nonfinite_rejection: positions=32 values=3 mappings=2 paths=2 status=NONFINITE_SOURCE");
    return true;
}

int main(void) {
    if (!test_layout_and_p0_packing() ||
        !test_all_packed_bytes() ||
        !test_scale_encoding_and_boundaries() ||
        !test_metadata_validation() ||
        !test_zero_midpoint_and_scale_ties() ||
        !test_python_golden_vectors() ||
        !test_fixed_random_reference_optimized_match() ||
        !test_nonfinite_rejection_and_output_atomicity()) {
        fprintf(stderr, "ROCmFP2 Phase-1 reference tests FAILED after %llu checks\n", test_checks);
        return 1;
    }

    printf("PASS rocmfp2_phase1_reference: checks=%llu arithmetic=IEEE754-binary64 search=127-scales exhaustive\n",
           test_checks);
    return 0;
}
