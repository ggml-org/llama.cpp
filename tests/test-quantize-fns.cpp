// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 = 0.0070f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 = 0.0020f;
// MXFP Hadamard pipeline thresholds (mxfp_rmse, which computes sqrt(sum/n)).
// These represent actual RMSE through the full KV cache write/read path.
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP4 = 0.40f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP8 = 0.08f;
constexpr float MAX_MXFP_PIPELINE_ERROR_MXFP6 = 0.10f;

constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_MXFP = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// MXFP RMSE: sqrt(sum/n), used with MAX_MXFP_PIPELINE_ERROR_* thresholds
static float mxfp_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf((float)(sum / n));
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    ggml_cpu_init();

    int num_failed = 0;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4       ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 :
                type == GGML_TYPE_MXFP4_E2M1 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 :
                type == GGML_TYPE_MXFP6_E2M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 :
                type == GGML_TYPE_MXFP8_E4M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 : MAX_QUANTIZATION_TOTAL_ERROR;
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                                            type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                                          ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                                          : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0
                                          ? MAX_DOT_PRODUCT_ERROR_TERNARY
                                          : type == GGML_TYPE_NVFP4
                                          ? MAX_DOT_PRODUCT_ERROR_FP4
                                          : type == GGML_TYPE_MXFP4_E2M1 || type == GGML_TYPE_MXFP6_E2M3 || type == GGML_TYPE_MXFP8_E4M3
                                          ? MAX_DOT_PRODUCT_ERROR_MXFP
                                          : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    // MXFP SoA roundtrip via traits
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float_soa || !qfns_cpu->to_float_soa) {
            continue;
        }

        const size_t buf_size = ggml_row_size(type, test_size);
        std::vector<uint8_t> tmp_q(buf_size);
        std::vector<float> tmp_out(test_size);

        qfns_cpu->from_float_soa(test_data.data(), tmp_q.data(), test_size);
        qfns_cpu->to_float_soa(tmp_q.data(), tmp_out.data(), test_size);

        const float soa_error = array_rmse(test_data.data(), tmp_out.data(), test_size);
        const float max_soa_error =
            type == GGML_TYPE_MXFP4_E2M1 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP4 :
            type == GGML_TYPE_MXFP6_E2M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP6 :
            type == GGML_TYPE_MXFP8_E4M3 ? MAX_QUANTIZATION_TOTAL_ERROR_MXFP8 : MAX_QUANTIZATION_TOTAL_ERROR;
        failed = !(soa_error < max_soa_error);
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s SoA quantization error:          %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], soa_error);
        }
    }

    // MXFP traits: SoA required, MXFP6/MXFP8 are KV-cache-only (no AoS dequant)
    {
        const ggml_type all_mxfp_types[] = { GGML_TYPE_MXFP4_E2M1, GGML_TYPE_MXFP8_E4M3, GGML_TYPE_MXFP6_E2M3 };
        for (ggml_type type : all_mxfp_types) {
            const auto * cpu = ggml_get_type_traits_cpu(type);

            failed = !(cpu->from_float_soa && cpu->to_float_soa);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s SoA traits present:               %s\n", ggml_type_name(type), RESULT_STR[failed]);
            }
        }

        // KV-cache-only types: no AoS dequant
        const ggml_type kv_only_types[] = { GGML_TYPE_MXFP8_E4M3, GGML_TYPE_MXFP6_E2M3 };
        for (ggml_type type : kv_only_types) {
            const auto * cpu = ggml_get_type_traits_cpu(type);
            failed = (cpu->to_float != nullptr);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s AoS CPU to_float absent:          %s\n", ggml_type_name(type), RESULT_STR[failed]);
            }
        }
    }

    // Hadamard self-inverse: H(H(x)) == x
    {
        float original[32], transformed[32];
        for (int i = 0; i < 32; i++) {
            original[i] = 0.1f + 2.0f * cosf(i + 0.5f);
            transformed[i] = original[i];
        }
        ggml_hadamard_32_inplace(transformed);
        ggml_hadamard_32_inplace(transformed); // apply twice = identity

        float max_err = 0.0f;
        for (int i = 0; i < 32; i++) {
            float err = fabsf(transformed[i] - original[i]);
            if (err > max_err) max_err = err;
        }
        // floating-point rounding tolerance
        failed = !(max_err < 1e-5f);
        num_failed += failed;
        if (failed || verbose) {
            printf("hadamard H(H(x))==x roundtrip:         %s (max_err=%.2e)\n", RESULT_STR[failed], max_err);
        }
    }

    // SoA SIMD vs scalar dequant
    {
        struct soa_cross_check {
            ggml_type type;
            void (*ref_dequant)(const void *, float *, int64_t);
        };

        const soa_cross_check checks[] = {
            { GGML_TYPE_MXFP4_E2M1, dequantize_row_mxfp4_soa },
            { GGML_TYPE_MXFP8_E4M3, dequantize_row_mxfp8_soa },
            { GGML_TYPE_MXFP6_E2M3, dequantize_row_mxfp6_soa },
        };

        for (const auto & c : checks) {
            const auto * cpu = ggml_get_type_traits_cpu(c.type);
            if (!cpu->from_float_soa || !cpu->to_float_soa) continue;

            const size_t buf_size = ggml_row_size(c.type, test_size);
            std::vector<uint8_t> tmp_q(buf_size);
            std::vector<float> out_ref(test_size);
            std::vector<float> out_simd(test_size);

            // Quantize with SoA
            cpu->from_float_soa(test_data.data(), tmp_q.data(), test_size);

            // Dequant with scalar reference
            c.ref_dequant(tmp_q.data(), out_ref.data(), test_size);

            // Dequant with CPU/SIMD path
            cpu->to_float_soa(tmp_q.data(), out_simd.data(), test_size);

            // Compare bitwise
            int mismatches = 0;
            for (size_t j = 0; j < test_size; j++) {
                uint32_t a, b;
                memcpy(&a, &out_ref[j], 4);
                memcpy(&b, &out_simd[j], 4);
                if (a != b) mismatches++;
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s SoA SIMD vs scalar ref:           %s (%zu/%zu match)\n",
                       ggml_type_name(c.type), RESULT_STR[failed],
                       test_size - mismatches, test_size);
            }
        }
    }

    // element converters vs canonical LUT values
    {
        struct lut_test {
            const char * name;
            const float * lut;
            int           count;
            float       (*converter)(uint8_t);
        };

        const lut_test lut_tests[] = {
            { "fp8_e4m3", kvalues_mxfp8_e4m3, 256, fp8_e4m3_to_float },
            { "fp8_e5m2", kvalues_mxfp8_e5m2, 256, fp8_e5m2_to_float },
            { "fp6_e2m3", kvalues_mxfp6_e2m3,  64, fp6_e2m3_to_float },
            { "fp6_e3m2", kvalues_mxfp6_e3m2,  64, fp6_e3m2_to_float },
        };

        for (const auto & t : lut_tests) {
            int mismatches = 0;
            for (int i = 0; i < t.count; i++) {
                const float converter_val = t.converter((uint8_t)i);
                const float lut_val       = t.lut[i];

                // both NaN = match
                if (isnan(converter_val) && isnan(lut_val)) continue;
                if (converter_val != lut_val) {
                    if (mismatches == 0 || verbose) {
                        printf("  %s LUT mismatch at [%d]: converter=%.8g, lut=%.8g\n",
                               t.name, i, converter_val, lut_val);
                    }
                    mismatches++;
                }
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s converter vs LUT:                %s (%d/%d values match)\n",
                       t.name, RESULT_STR[failed], t.count - mismatches, t.count);
            }
        }

        // FP4 E2M1
        {
            int mismatches = 0;
            for (int i = 0; i < 16; i++) {
                const float converter_val = ggml_mxfp_fp4_e2m1_to_float((uint8_t)i);
                const float lut_val       = kvalues_mxfp4_float[i];
                if (converter_val != lut_val) {
                    if (mismatches == 0 || verbose) {
                        printf("  fp4_e2m1 LUT mismatch at [%d]: converter=%.8g, lut=%.8g\n",
                               i, converter_val, lut_val);
                    }
                    mismatches++;
                }
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("fp4_e2m1 converter vs LUT:                %s (%d/16 values match)\n",
                       RESULT_STR[failed], 16 - mismatches);
            }
        }
    }

    // element converter edge cases (expected values validated against LUTs)
    {
        struct conv_check {
            const char * name;
            float        input;
            uint8_t      expected_bits;
            bool         is_saturation;  // true = input overflows, expected_bits is max finite
            const float * lut;           // canonical LUT to validate expected_bits against (NULL for FP4)
            float       (*to_float)(uint8_t);
            uint8_t     (*to_quant)(float);
        };

        const conv_check checks[] = {
            // FP4 E2M1 -[S(1)|E(2)|M(1)], bias=0
            { "fp4 zero",      0.0f,    0x00, false, nullptr, nullptr, nullptr },
            { "fp4 sub 0.5",   0.5f,    0x01, false, nullptr, nullptr, nullptr },
            { "fp4 norm 1.0",  1.0f,    0x02, false, nullptr, nullptr, nullptr },
            { "fp4 max 6.0",   6.0f,    0x07, false, nullptr, nullptr, nullptr },
            { "fp4 neg -3.0", -3.0f,    0x0D, false, nullptr, nullptr, nullptr },
            { "fp4 sat 100",  100.0f,   0x07, true,  nullptr, nullptr, nullptr },

            // FP8 E4M3 -[S(1)|E(4)|M(3)], bias=7
            { "e4m3 zero",      0.0f,     0x00, false, kvalues_mxfp8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3_rn },
            { "e4m3 sub",       1.f/512,  0x01, false, kvalues_mxfp8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3_rn },
            { "e4m3 max 448",   448.0f,   0x7E, false, kvalues_mxfp8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3_rn },
            { "e4m3 sat 500",   500.0f,   0x7E, true,  kvalues_mxfp8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3_rn },
            { "e4m3 neg -1",   -1.0f,     0xB8, false, kvalues_mxfp8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3_rn },

            // FP6 E2M3 -[S(1)|E(2)|M(3)], no NaN/Inf
            { "e2m3 zero",      0.0f,     0x00, false, kvalues_mxfp6_e2m3, fp6_e2m3_to_float, float_to_fp6_e2m3_rn },
            { "e2m3 sub",       0.125f,   0x01, false, kvalues_mxfp6_e2m3, fp6_e2m3_to_float, float_to_fp6_e2m3_rn },
            { "e2m3 max 7.5",   7.5f,     0x1F, false, kvalues_mxfp6_e2m3, fp6_e2m3_to_float, float_to_fp6_e2m3_rn },
            { "e2m3 sat 100",   100.0f,   0x1F, true,  kvalues_mxfp6_e2m3, fp6_e2m3_to_float, float_to_fp6_e2m3_rn },

            // FP6 E3M2 -[S(1)|E(3)|M(2)], no NaN/Inf, exp=7 is NORMAL
            { "e3m2 zero",      0.0f,     0x00, false, kvalues_mxfp6_e3m2, fp6_e3m2_to_float, float_to_fp6_e3m2_rn },
            { "e3m2 sub",       0.0625f,  0x01, false, kvalues_mxfp6_e3m2, fp6_e3m2_to_float, float_to_fp6_e3m2_rn },
            { "e3m2 max 28.0",  28.0f,    0x1F, false, kvalues_mxfp6_e3m2, fp6_e3m2_to_float, float_to_fp6_e3m2_rn },
            { "e3m2 exp7 16",   16.0f,    0x1C, false, kvalues_mxfp6_e3m2, fp6_e3m2_to_float, float_to_fp6_e3m2_rn },

            // FP8 E5M2 -[S(1)|E(5)|M(2)], bias=15
            { "e5m2 zero",      0.0f,     0x00, false, kvalues_mxfp8_e5m2, fp8_e5m2_to_float, float_to_fp8_e5m2_rn },
            { "e5m2 max",       57344.f,  0x7B, false, kvalues_mxfp8_e5m2, fp8_e5m2_to_float, float_to_fp8_e5m2_rn },
        };

        int conv_bad = 0;

        // validate expected_bits against LUTs
        for (const auto & c : checks) {
            if (c.lut && !c.is_saturation) {
                float lut_val = c.lut[c.expected_bits];
                if (c.input != lut_val && !(c.input == 0.0f && lut_val == 0.0f)) {
                    printf("  TEST BUG %s: expected_bits=0x%02X → LUT=%.8g, but input=%.8g\n",
                           c.name, c.expected_bits, lut_val, c.input);
                    conv_bad++;
                }
            } else if (!c.lut && !c.is_saturation) {
                float lut_val = kvalues_mxfp4_float[c.expected_bits];
                if (c.input != lut_val && !(c.input == 0.0f && lut_val == 0.0f)) {
                    printf("  TEST BUG %s: expected_bits=0x%02X → LUT=%.8g, but input=%.8g\n",
                           c.name, c.expected_bits, lut_val, c.input);
                    conv_bad++;
                }
            }
        }

        // Now test the quantize direction
        for (const auto & c : checks) {
            uint8_t got;
            if (c.to_quant) {
                got = c.to_quant(c.input);
            } else {
                got = ggml_mxfp_float_to_fp4_e2m1(c.input);
            }
            if (got != c.expected_bits) {
                if (conv_bad == 0 || verbose) {
                    printf("  %s: quantize(%.6g) = 0x%02X, expected 0x%02X\n",
                           c.name, c.input, got, c.expected_bits);
                }
                conv_bad++;
            }
        }

        // FP8 E4M3: 0x7F must dequantize to NaN
        {
            float nan_val = fp8_e4m3_to_float(0x7F);
            if (!isnan(nan_val)) {
                if (conv_bad == 0 || verbose) {
                    printf("  e4m3 0x7F dequant: expected NaN, got %.6g\n", nan_val);
                }
                conv_bad++;
            }
        }

        // FP6 E3M2: exp=7 must dequant to valid float (NOT Inf/NaN)
        {
            float exp7_val = fp6_e3m2_to_float(0x1F);  // max: exp=7, mant=3 → 28.0
            if (isnan(exp7_val) || exp7_val != 28.0f) {
                if (conv_bad == 0 || verbose) {
                    printf("  e3m2 0x1F dequant: expected 28.0, got %.6g\n", exp7_val);
                }
                conv_bad++;
            }
        }

        failed = (conv_bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  element converter edge cases:        %s (%d/%d passed)\n",
                   RESULT_STR[failed],
                   (int)(sizeof(checks)/sizeof(checks[0])) + 2 - conv_bad,
                   (int)(sizeof(checks)/sizeof(checks[0])) + 2);
        }
    }

    // FP6 pack/unpack round-trip
    {
        int pack_bad = 0;

        // Test all 64 possible 6-bit values in each of the 4 positions
        for (int pos = 0; pos < 4; pos++) {
            for (int val = 0; val < 64; val++) {
                uint8_t in[4] = {0, 0, 0, 0};
                in[pos] = (uint8_t)val;

                uint8_t packed[3], out[4];
                pack_fp6x4(in, packed);
                unpack_fp6x4(packed, out);

                if (out[pos] != (uint8_t)val) {
                    if (pack_bad == 0 || verbose) {
                        printf("  fp6 pack roundtrip: pos=%d val=0x%02X → got 0x%02X\n",
                               pos, val, out[pos]);
                    }
                    pack_bad++;
                }
                // no crosstalk
                for (int k = 0; k < 4; k++) {
                    if (k != pos && out[k] != 0) {
                        if (pack_bad == 0 || verbose) {
                            printf("  fp6 pack crosstalk: pos=%d val=0x%02X leaked to pos=%d (0x%02X)\n",
                                   pos, val, k, out[k]);
                        }
                        pack_bad++;
                    }
                }
            }
        }

        // known-answer: [0x3F, 0x00, 0x3F, 0x00] -> {0x3F, 0xF0, 0x03}
        {
            uint8_t in[4] = {0x3F, 0x00, 0x3F, 0x00};
            uint8_t packed[3];
            pack_fp6x4(in, packed);
            uint8_t expected[3] = {0x3F, 0xF0, 0x03};
            if (packed[0] != expected[0] || packed[1] != expected[1] || packed[2] != expected[2]) {
                if (pack_bad == 0 || verbose) {
                    printf("  fp6 known-answer: packed [%02X,%02X,%02X] expected [%02X,%02X,%02X]\n",
                           packed[0], packed[1], packed[2], expected[0], expected[1], expected[2]);
                }
                pack_bad++;
            }
        }

        failed = (pack_bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  fp6 pack/unpack round-trip:           %s\n", RESULT_STR[failed]);
        }
    }

    // E8M0 known-answer decode + HALF vs FULL (MXFP4 uses HALF, MXFP6/8 use FULL)
    {
        int e8m0_bad = 0;

        // Known-answer E8M0 decodes
        struct { uint8_t e; float expected; } e8m0_known[] = {
            { 127, 1.0f },     // 2^(127-127) = 2^0 = 1.0
            { 128, 2.0f },     // 2^(128-127) = 2^1 = 2.0
            { 126, 0.5f },     // 2^(126-127) = 2^(-1) = 0.5
            { 254, 1.70141183e+38f }, // 2^127 (max representable)
            {   1, 1.17549435e-38f }, // 2^(-126) (min normal)
        };
        for (const auto & t : e8m0_known) {
            float got = ggml_mxfp_e8m0_to_fp32(t.e);
            if (got != t.expected) {
                if (e8m0_bad == 0 || verbose) {
                    printf("  E8M0 decode e=%d: got %.8g, expected %.8g\n", t.e, got, t.expected);
                }
                e8m0_bad++;
            }
        }

        // HALF must be exactly half of FULL for all valid exponents
        for (int e = 2; e < 255; e++) {
            float full = ggml_mxfp_e8m0_to_fp32((uint8_t)e);
            float half = ggml_mxfp_e8m0_to_fp32_half((uint8_t)e);
            if (half != full * 0.5f) {
                if (e8m0_bad == 0 || verbose) {
                    printf("  E8M0 HALF!=FULL/2 at e=%d: half=%.8g, full/2=%.8g\n", e, half, full * 0.5f);
                }
                e8m0_bad++;
                break;  // one failure is enough to flag the pattern
            }
        }

        failed = (e8m0_bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  E8M0 known-answer + HALF/FULL:       %s\n", RESULT_STR[failed]);
        }
    }

    // E8M0 rounding at sqrt(2) threshold
    {
        int round_bad = 0;

        // amax=1.0: floor_log2=0, mantissa=0 → no round → e_base = 0 - 0 + 127 = 127
        {
            int e = ggml_mxfp_e8m0_base_estimate(1.0f, 0);
            if (e != 127) {
                printf("  E8M0 round: amax=1.0 → e=%d, expected 127\n", e);
                round_bad++;
            }
        }
        // amax=2.0: floor_log2=1, mantissa=0 → no round → e_base = 1 + 127 = 128
        {
            int e = ggml_mxfp_e8m0_base_estimate(2.0f, 0);
            if (e != 128) {
                printf("  E8M0 round: amax=2.0 → e=%d, expected 128\n", e);
                round_bad++;
            }
        }
        // amax just below sqrt(2): mantissa < 0x3504F3 → floor only → e=127
        {
            // 1.41421 has IEEE mantissa just below 0x3504F3
            float below = 1.4142f;
            int e = ggml_mxfp_e8m0_base_estimate(below, 0);
            if (e != 127) {
                printf("  E8M0 round: amax=%.6f → e=%d, expected 127 (no round)\n", below, e);
                round_bad++;
            }
        }
        // amax at sqrt(2): mantissa >= 0x3504F3 → rounds up → e=128
        {
            float at_sqrt2 = 1.41422f;
            int e = ggml_mxfp_e8m0_base_estimate(at_sqrt2, 0);
            if (e != 128) {
                printf("  E8M0 round: amax=%.6f → e=%d, expected 128 (rounds up)\n", at_sqrt2, e);
                round_bad++;
            }
        }
        // Verify emax_offset shifts the result
        {
            int e_no_off = ggml_mxfp_e8m0_base_estimate(448.0f, 0);
            int e_e4m3   = ggml_mxfp_e8m0_base_estimate(448.0f, MXFP8_E4M3_EMAX_OFFSET);
            if (e_no_off - e_e4m3 != MXFP8_E4M3_EMAX_OFFSET) {
                printf("  E8M0 emax_offset: diff=%d, expected %d\n",
                       e_no_off - e_e4m3, MXFP8_E4M3_EMAX_OFFSET);
                round_bad++;
            }
        }

        failed = (round_bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  E8M0 rounding boundary:              %s\n", RESULT_STR[failed]);
        }
    }

    // Element converter exhaustive round-trip: quantize(dequantize(i)) == i for all valid bit patterns.
    // Catches asymmetries between the to_float and to_quant paths.
    {
        struct rt_test {
            const char * name;
            int           count;
            float       (*to_float)(uint8_t);
            uint8_t     (*to_quant)(float);
            uint8_t       nan_bits;   // bit pattern for NaN (0 = no NaN in format)
        };

        const rt_test rt_tests[] = {
            { "fp8_e4m3", 256, fp8_e4m3_to_float, float_to_fp8_e4m3_rn, 0x7F },
            { "fp8_e5m2", 256, fp8_e5m2_to_float, float_to_fp8_e5m2_rn, 0    },
            { "fp6_e2m3",  64, fp6_e2m3_to_float, float_to_fp6_e2m3_rn, 0    },
            { "fp6_e3m2",  64, fp6_e3m2_to_float, float_to_fp6_e3m2_rn, 0    },
        };

        for (const auto & t : rt_tests) {
            int rt_bad = 0;
            for (int i = 0; i < t.count; i++) {
                if ((uint8_t)i == t.nan_bits) continue;  // skip NaN -quantize(NaN) is implementation-defined

                float f = t.to_float((uint8_t)i);
                if (isnan(f) || isinf(f)) continue;  // E5M2 Inf/NaN

                uint8_t back = t.to_quant(f);
                // Negative zero may round-trip to positive zero -both are valid
                if (back != (uint8_t)i && !(f == 0.0f && t.to_float(back) == 0.0f)) {
                    if (rt_bad == 0 || verbose) {
                        printf("  %s roundtrip: 0x%02X → %.6g → 0x%02X\n",
                               t.name, i, f, back);
                    }
                    rt_bad++;
                }
            }
            failed = (rt_bad > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s converter round-trip:             %s (%d/%d survived)\n",
                       t.name, RESULT_STR[failed], t.count - rt_bad, t.count);
            }
        }

        // FP4 E2M1: uses static inline converters (not GGML_API wrappers), only 16 values
        {
            int rt_bad = 0;
            for (int i = 0; i < 16; i++) {
                float f = ggml_mxfp_fp4_e2m1_to_float((uint8_t)i);
                uint8_t back = ggml_mxfp_float_to_fp4_e2m1(f);
                if (back != (uint8_t)i && !(f == 0.0f && ggml_mxfp_fp4_e2m1_to_float(back) == 0.0f)) {
                    if (rt_bad == 0 || verbose) {
                        printf("  fp4_e2m1 roundtrip: 0x%02X → %.6g → 0x%02X\n", i, f, back);
                    }
                    rt_bad++;
                }
            }
            failed = (rt_bad > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("fp4_e2m1 converter round-trip:             %s (%d/16 survived)\n",
                       RESULT_STR[failed], 16 - rt_bad);
            }
        }
    }

    // E8M0 scale computation: verify base exponent is reasonable for various amax values
    {
        const float test_amax[] = { 0.001f, 0.1f, 1.0f, 6.0f, 100.0f, 448.0f, 10000.0f };
        int bad = 0;
        for (float amax : test_amax) {
            // ggml_mxfp_e8m0_base_estimate returns unclamped e_base
            int e_base = ggml_mxfp_e8m0_base_estimate(amax, 0);
            if (e_base < 1 || e_base > 254) {
                if (bad == 0 || verbose) {
                    printf("  E8M0 bad e_base=%d for amax=%.4f\n", e_base, amax);
                }
                bad++;
                continue;
            }
            float scale = ggml_mxfp_e8m0_to_fp32((uint8_t)e_base);
            // Scale should be within 2x of amax (rough sanity check)
            float ratio = amax / scale;
            if (ratio < 0.25f || ratio > 4.0f) {
                if (bad == 0 || verbose) {
                    printf("  E8M0 scale=%.6g for amax=%.4f, ratio=%.4f (expected ~1)\n",
                           scale, amax, ratio);
                }
                bad++;
            }
        }
        failed = (bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  E8M0 scale sanity check:             %s (%d/%d passed)\n",
                   RESULT_STR[failed], (int)(sizeof(test_amax)/sizeof(test_amax[0])) - bad,
                   (int)(sizeof(test_amax)/sizeof(test_amax[0])));
        }
    }

    // SoA layout: verify offset macros produce correct byte positions
    {
        const struct { ggml_type type; int qs_per_block; } soa_types[] = {
            { GGML_TYPE_MXFP4_E2M1, MXFP4_SOA_QS_PER_BLOCK },
            { GGML_TYPE_MXFP8_E4M3, MXFP8_SOA_QS_PER_BLOCK },
            { GGML_TYPE_MXFP6_E2M3, MXFP6_SOA_QS_PER_BLOCK },
        };

        for (const auto & st : soa_types) {
            for (int nblocks : { 1, 4, 8, 32 }) {
                size_t expected_e8m0_off = (size_t)nblocks * st.qs_per_block;
                size_t actual_e8m0_off = MXFP_SOA_E8M0_OFFSET(nblocks, st.qs_per_block);
                size_t total = actual_e8m0_off + nblocks; // e8m0 region = 1 byte per block
                size_t row_size = ggml_row_size(st.type, nblocks * 32);

                bool offset_ok = (actual_e8m0_off == expected_e8m0_off);
                bool size_ok = (total == row_size);

                if (!offset_ok || !size_ok) {
                    failed = true;
                    num_failed++;
                    if (verbose) {
                        printf("  %s SoA layout nblocks=%d: e8m0_off=%zu (expected %zu), total=%zu (row_size=%zu)\n",
                               ggml_type_name(st.type), nblocks, actual_e8m0_off, expected_e8m0_off, total, row_size);
                    }
                }
            }
        }
        if (verbose) {
            printf("  SoA layout offset check:             %s\n", RESULT_STR[0]); // only prints failures above
        }
    }

    // block size consistency
    {
        failed = !(QK_MXFP4 == 32 && QK_MXFP8 == 32 && QK_MXFP6 == 32);
        num_failed += failed;
        if (failed || verbose) {
            printf("  MXFP block size == 32:               %s (QK4=%d, QK8=%d, QK6=%d)\n",
                   RESULT_STR[failed], QK_MXFP4, QK_MXFP8, QK_MXFP6);
        }
    }

    // EMAX_OFFSET produces valid E8M0 for each format's max finite value
    {
        struct emax_check {
            const char  * name;
            int           emax_offset;
            float         max_finite;    // from LUT / converter
        };

        const emax_check emax_checks[] = {
            { "fp4_e2m1", MXFP4_E2M1_EMAX_OFFSET, 6.0f     },
            { "fp6_e2m3", MXFP6_E2M3_EMAX_OFFSET, 7.5f     },
            { "fp6_e3m2", MXFP6_E3M2_EMAX_OFFSET, 28.0f    },
            { "fp8_e4m3", MXFP8_E4M3_EMAX_OFFSET, 448.0f   },
            { "fp8_e5m2", MXFP8_E5M2_EMAX_OFFSET, 57344.0f },
        };

        int emax_bad = 0;
        for (const auto & e : emax_checks) {
            // When amax == max_finite, the base estimate must produce a valid E8M0 (1..254)
            int e_base = ggml_mxfp_e8m0_base_estimate(e.max_finite, e.emax_offset);
            if (e_base < 1 || e_base > 254) {
                if (emax_bad == 0 || verbose) {
                    printf("  %s emax_offset=%d: max_finite=%.1f gives e_base=%d (out of range)\n",
                           e.name, e.emax_offset, e.max_finite, e_base);
                }
                emax_bad++;
            }
        }
        failed = (emax_bad > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  EMAX_OFFSET vs format max:           %s\n", RESULT_STR[failed]);
        }
    }

    // MXFP4 AoS vs SoA: two independent code paths, same result
    {
        const int nelems = 64;  // 2 blocks
        float input[64];
        for (int i = 0; i < 64; i++) {
            input[i] = 0.5f + 2.0f * sinf(i * 0.7f + 0.3f);
        }

        // Quantize and dequant via AoS (block_mxfp4 structs)
        std::vector<block_mxfp4> aos_q(nelems / QK_MXFP4);
        std::vector<float> aos_out(nelems);
        quantize_row_mxfp4_ref(input, aos_q.data(), nelems);
        dequantize_row_mxfp4(aos_q.data(), aos_out.data(), nelems);

        // Quantize and dequant via SoA
        const size_t soa_buf_size = ggml_row_size(GGML_TYPE_MXFP4_E2M1, nelems);
        std::vector<uint8_t> soa_q(soa_buf_size);
        std::vector<float> soa_out(nelems);
        quantize_row_mxfp4_soa(input, soa_q.data(), nelems);
        dequantize_row_mxfp4_soa(soa_q.data(), soa_out.data(), nelems);

        // Compare: both paths should produce identical results
        int mismatches = 0;
        for (int i = 0; i < nelems; i++) {
            uint32_t a, b;
            memcpy(&a, &aos_out[i], 4);
            memcpy(&b, &soa_out[i], 4);
            if (a != b) {
                if (mismatches == 0 || verbose) {
                    printf("  mxfp4 AoS/SoA mismatch at [%d]: AoS=%.8g, SoA=%.8g\n",
                           i, aos_out[i], soa_out[i]);
                }
                mismatches++;
            }
        }
        failed = (mismatches > 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("mxfp4 AoS vs SoA cross-check:          %s (%d/%d match)\n",
                   RESULT_STR[failed], nelems - mismatches, nelems);
        }
    }

    // Hadamard + quantize + dequant + Hadamard roundtrip (KV cache write/read path)
    {
        struct hadamard_pipeline_check {
            const char * name;
            ggml_type    type;
            float        max_err;
        };

        const hadamard_pipeline_check pipeline_checks[] = {
            { "mxfp4",     GGML_TYPE_MXFP4_E2M1, MAX_MXFP_PIPELINE_ERROR_MXFP4 },
            { "mxfp8",     GGML_TYPE_MXFP8_E4M3, MAX_MXFP_PIPELINE_ERROR_MXFP8 },
            { "mxfp6",     GGML_TYPE_MXFP6_E2M3, MAX_MXFP_PIPELINE_ERROR_MXFP6 },
        };

        for (const auto & p : pipeline_checks) {
            const auto * cpu = ggml_get_type_traits_cpu(p.type);

            std::vector<float> original(test_size);
            std::vector<float> rotated(test_size);
            std::vector<float> recovered(test_size);
            generate_data(2.0, test_size, original.data());

            // Write path: Hadamard each block, then quantize
            memcpy(rotated.data(), original.data(), test_size * sizeof(float));
            for (size_t b = 0; b < test_size / 32; b++) {
                ggml_hadamard_32_inplace(&rotated[b * 32]);
            }

            const size_t buf_size = ggml_row_size(p.type, test_size);
            std::vector<uint8_t> qbuf(buf_size);
            cpu->from_float_soa(rotated.data(), qbuf.data(), test_size);

            // Read path: dequant, then Hadamard each block (self-inverse)
            cpu->to_float_soa(qbuf.data(), recovered.data(), test_size);
            for (size_t b = 0; b < test_size / 32; b++) {
                ggml_hadamard_32_inplace(&recovered[b * 32]);
            }

            float err = mxfp_rmse(original.data(), recovered.data(), test_size);
            failed = !(err < p.max_err);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s Hadamard pipeline roundtrip:       %s (err=%.6f, max=%.6f)\n",
                       p.name, RESULT_STR[failed], err, p.max_err);
            }
        }
    }

    // Hadamard known output: H([1,0,...,0]) = [1/sqrt(32), ...]
    {
        float unit[32] = {};
        unit[0] = 1.0f;
        ggml_hadamard_32_inplace(unit);

        const float expected = MXFP_HADAMARD_32_NORM;  // 1/sqrt(32)
        float max_err = 0.0f;
        for (int i = 0; i < 32; i++) {
            float err = fabsf(unit[i] - expected);
            if (err > max_err) max_err = err;
        }
        failed = !(max_err < 1e-7f);
        num_failed += failed;
        if (failed || verbose) {
            printf("hadamard unit vector:                  %s (max_err=%.2e, expected %.8f)\n",
                   RESULT_STR[failed], max_err, expected);
        }
    }

    // zero block produces E8M0=0
    {
        float zeros[32] = {};
        const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP8_E4M3, 32);
        std::vector<uint8_t> buf(buf_size, 0xFF);  // fill with 0xFF to detect non-writes

        quantize_row_mxfp8_soa(zeros, buf.data(), 32);

        // E8M0 scale is at offset MXFP8_SOA_QS_PER_BLOCK (32) for 1 block
        uint8_t e8m0 = buf[MXFP8_SOA_QS_PER_BLOCK];
        failed = (e8m0 != 0);
        num_failed += failed;
        if (failed || verbose) {
            printf("  zero block E8M0:                     %s (e8m0=%d, expected 0)\n",
                   RESULT_STR[failed], e8m0);
        }
    }

    // SoA format spec: quantize, manually walk raw bytes, compare against reference dequant
    {
        // 2 blocks, asymmetric data
        const int nblocks = 2;
        const int nelems = nblocks * 32;
        float input[64];
        for (int i = 0; i < 64; i++) {
            // Block 0: small values, Block 1: large values -different E8M0 scales
            input[i] = (i < 32) ? 0.1f * sinf(i + 0.5f) : 3.0f * cosf(i + 0.5f);
        }

        // MXFP4
        {
            const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP4_E2M1, nelems);
            std::vector<uint8_t> buf(buf_size);
            std::vector<float> ref_out(nelems);
            std::vector<float> manual_out(nelems);

            quantize_row_mxfp4_soa(input, buf.data(), nelems);
            dequantize_row_mxfp4_soa(buf.data(), ref_out.data(), nelems);

            // manual dequant from raw bytes
            const uint8_t * qs = buf.data();
            const uint8_t * e8m0 = buf.data() + MXFP_SOA_E8M0_OFFSET(nblocks, MXFP4_SOA_QS_PER_BLOCK);

            for (int b = 0; b < nblocks; b++) {
                const float d = ggml_mxfp_e8m0_to_fp32_half(e8m0[b]);
                const uint8_t * block_qs = qs + MXFP_SOA_QS_OFFSET(b, MXFP4_SOA_QS_PER_BLOCK);
                for (int j = 0; j < 16; j++) {
                    // low nibble = first half, high nibble = second half
                    int8_t v_lo = kvalues_mxfp4[block_qs[j] & 0x0F];
                    int8_t v_hi = kvalues_mxfp4[block_qs[j] >>   4];
                    manual_out[b*32 + j]      = v_lo * d;
                    manual_out[b*32 + j + 16] = v_hi * d;
                }
            }

            int mismatches = 0;
            for (int i = 0; i < nelems; i++) {
                uint32_t a, b;
                memcpy(&a, &ref_out[i], 4);
                memcpy(&b, &manual_out[i], 4);
                if (a != b) mismatches++;
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("mxfp4 SoA format spec:                 %s (%d/%d match)\n",
                       RESULT_STR[failed], nelems - mismatches, nelems);
            }
        }

        // MXFP8
        {
            const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP8_E4M3, nelems);
            std::vector<uint8_t> buf(buf_size);
            std::vector<float> ref_out(nelems);
            std::vector<float> manual_out(nelems);

            quantize_row_mxfp8_soa(input, buf.data(), nelems);
            dequantize_row_mxfp8_soa(buf.data(), ref_out.data(), nelems);

            const uint8_t * qs = buf.data();
            const uint8_t * e8m0 = buf.data() + MXFP_SOA_E8M0_OFFSET(nblocks, MXFP8_SOA_QS_PER_BLOCK);

            for (int b = 0; b < nblocks; b++) {
                const float d = ggml_mxfp_e8m0_to_fp32(e8m0[b]);
                const uint8_t * block_qs = qs + MXFP_SOA_QS_OFFSET(b, MXFP8_SOA_QS_PER_BLOCK);
                for (int j = 0; j < 32; j++) {
                    // one byte per element
                    manual_out[b*32 + j] = fp8_e4m3_to_float(block_qs[j]) * d;
                }
            }

            int mismatches = 0;
            for (int i = 0; i < nelems; i++) {
                uint32_t a, b;
                memcpy(&a, &ref_out[i], 4);
                memcpy(&b, &manual_out[i], 4);
                if (a != b) mismatches++;
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("mxfp8 SoA format spec:                 %s (%d/%d match)\n",
                       RESULT_STR[failed], nelems - mismatches, nelems);
            }
        }

        // MXFP6
        {
            const size_t buf_size = ggml_row_size(GGML_TYPE_MXFP6_E2M3, nelems);
            std::vector<uint8_t> buf(buf_size);
            std::vector<float> ref_out(nelems);
            std::vector<float> manual_out(nelems);

            quantize_row_mxfp6_soa(input, buf.data(), nelems);
            dequantize_row_mxfp6_soa(buf.data(), ref_out.data(), nelems);

            const uint8_t * qs = buf.data();
            const uint8_t * e8m0 = buf.data() + MXFP_SOA_E8M0_OFFSET(nblocks, MXFP6_SOA_QS_PER_BLOCK);

            for (int b = 0; b < nblocks; b++) {
                const float d = ggml_mxfp_e8m0_to_fp32(e8m0[b]);
                const uint8_t * block_qs = qs + MXFP_SOA_QS_OFFSET(b, MXFP6_SOA_QS_PER_BLOCK);
                for (int j = 0; j < 32; j += 4) {
                    // 4 elements packed into 3 bytes
                    uint8_t vals[4];
                    unpack_fp6x4(&block_qs[j * 3 / 4], vals);
                    for (int k = 0; k < 4; k++) {
                        manual_out[b*32 + j + k] = fp6_e2m3_to_float(vals[k]) * d;
                    }
                }
            }

            int mismatches = 0;
            for (int i = 0; i < nelems; i++) {
                uint32_t a, b;
                memcpy(&a, &ref_out[i], 4);
                memcpy(&b, &manual_out[i], 4);
                if (a != b) mismatches++;
            }
            failed = (mismatches > 0);
            num_failed += failed;
            if (failed || verbose) {
                printf("mxfp6 SoA format spec:                 %s (%d/%d match)\n",
                       RESULT_STR[failed], nelems - mismatches, nelems);
            }
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
