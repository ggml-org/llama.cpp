// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

// vec_dot SIMD + generic prototypes (normally in internal quants.h)
extern "C" void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*cosf(i + offset);
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
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(test_data1, test_data2, test_size);
        return fabsf(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    qfns_cpu->from_float(test_data3, tmp_q1.data() + bx, test_size);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = fabsf(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

// regression test for exact -128 handling in the x86 int8 dot-product helpers
// (adapted from the unit test in #29356, thanks @AnkitNakhawa)
static int test_vec_dot_q8_0_i8_min(bool verbose) {
    struct test_block_q8_0 {
        ggml_fp16_t d;
        int8_t qs[32];
    };

    // the pair lives in block blk at byte offset off; the other block stays
    // zero, so with per-block x scales dx0/dx1 a dropped byte range, a missing
    // block or a wrong block/scale association changes the result
    const struct {
        int8_t blk;   // 0 or 1 (n = 64 -> two blocks)
        int8_t off;   // byte offset of the pair within the block
        int8_t x0;
        int8_t x1;
        int8_t y0;
        int8_t y1;
        float  dx0;   // x scale of block 0
        float  dx1;   // x scale of block 1
        float  expected;
    } cases[] = {
        { 0,  0,    -1,    0, -128,    0, 1.0f, 1.0f,   128.0f },
        { 0,  0,     1,    0, -128,    0, 1.0f, 1.0f,  -128.0f },
        { 0,  0,  -128,    0, -128,    0, 1.0f, 1.0f, 16384.0f },
        { 0,  0,  -128, -128, -128, -128, 1.0f, 1.0f, 32768.0f }, // two (-128)*(-128) products sum to 2^15
        { 0,  0,  -128, -128,  127,  127, 1.0f, 1.0f, -32512.0f }, // x = -128 without y = -128
        { 0,  0,   127,  127, -128, -128, 1.0f, 1.0f, -32512.0f },
        { 0,  8,  -128, -128, -128, -128, 1.0f, 1.0f, 32768.0f }, // second 8-byte half of the first 16
        { 0, 16,  -128, -128, -128, -128, 1.0f, 1.0f, 32768.0f }, // second 16-byte half
        { 0, 30,  -128, -128, -128, -128, 1.0f, 1.0f, 32768.0f }, // last pair of the block
        { 1,  0,  -128, -128, -128, -128, 1.0f, 1.0f, 32768.0f }, // second block needs the -128 check, too
        { 1, 30,  -128, -128, -128, -128, 1.0f, 1.0f, 32768.0f },
        { 1,  0,  -128, -128, -128, -128, 0.5f, 1.0f, 32768.0f }, // dot of block 1 must use block 1's scale
    };

    const auto * q8_0 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    int num_failed = 0;

    assert(sizeof(test_block_q8_0) == ggml_row_size(GGML_TYPE_Q8_0, 32));

    for (const auto & test : cases) {
        test_block_q8_0 x[2] = {};
        test_block_q8_0 y[2] = {};
        x[0].d = ggml_fp32_to_fp16(test.dx0);
        x[1].d = ggml_fp32_to_fp16(test.dx1);
        y[0].d = y[1].d = ggml_fp32_to_fp16(1.0f);
        x[test.blk].qs[test.off + 0] = test.x0;
        x[test.blk].qs[test.off + 1] = test.x1;
        y[test.blk].qs[test.off + 0] = test.y0;
        y[test.blk].qs[test.off + 1] = test.y1;

        float result = 0.0f;
        q8_0->vec_dot(64, &result, 0, x, 0, y, 0, 1);

        const bool failed = result != test.expected;
        num_failed += failed;
        if (failed || verbose) {
            printf(" q8_0 vec_dot blk=%d off=%2d x={%4d,%4d} y={%4d,%4d}: %s (ref=%f got=%f)\n",
                   test.blk, test.off, test.x0, test.x1, test.y0, test.y1,
                   RESULT_STR[failed], test.expected, result);
        }
    }

    return num_failed;
}

// regression test: a -128 byte in the activation (y) side must be handled exactly
// for every in-scope type, catching operand-order regressions at the call sites.
// Each type is tested with its q8_0 / q8_K activation pairing (the pairings whose
// vec_dot goes through the swapped int8 helpers).

// dequant helpers for types whose trait has no to_float (iq1_s, iq1_m, q8_K)
extern "C" void dequantize_row_iq1_s(const void * x, float * y, int64_t k);
extern "C" void dequantize_row_iq1_m(const void * x, float * y, int64_t k);
extern "C" void dequantize_row_q8_K (const void * x, float * y, int64_t k);

static int test_vec_dot_neg128(bool verbose) {
    const size_t test_size = 4096;  // multiple of 32 (q8_0) and 256 (q8_K)

    // deterministic correlated input (xf = yf -> large coherent dot product)
    std::vector<float> x_f32(test_size), y_f32(test_size);
    generate_data(0.0, test_size, x_f32.data());
    memcpy(y_f32.data(), x_f32.data(), test_size * sizeof(float));

    // iq1_s requires an importance matrix; uniform weights are fine
    // (block_q8_K layout: float d, int8 qs[256], int16 bsums[16])
    struct test_block_q8_K {
        float d;
        int8_t qs[256];
        int16_t bsums[16];
    };
    std::vector<float> imatrix(256, 1.0f);

    struct {
        ggml_type wtype;
        const char * name;
        void (*sim)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
        void (*gen)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    } types[] = {
        { GGML_TYPE_Q4_0,   "q4_0",   ggml_vec_dot_q4_0_q8_0,   ggml_vec_dot_q4_0_q8_0_generic   },
        { GGML_TYPE_Q5_0,   "q5_0",   ggml_vec_dot_q5_0_q8_0,   ggml_vec_dot_q5_0_q8_0_generic   },
        { GGML_TYPE_MXFP4,  "mxfp4",  ggml_vec_dot_mxfp4_q8_0,  ggml_vec_dot_mxfp4_q8_0_generic  },
        { GGML_TYPE_NVFP4,  "nvfp4",  ggml_vec_dot_nvfp4_q8_0,  ggml_vec_dot_nvfp4_q8_0_generic  },
        { GGML_TYPE_IQ4_NL, "iq4_nl", ggml_vec_dot_iq4_nl_q8_0, ggml_vec_dot_iq4_nl_q8_0_generic },
        { GGML_TYPE_IQ1_S,  "iq1_s",  ggml_vec_dot_iq1_s_q8_K,  ggml_vec_dot_iq1_s_q8_K_generic  },
        { GGML_TYPE_IQ1_M,  "iq1_m",  ggml_vec_dot_iq1_m_q8_K,  ggml_vec_dot_iq1_m_q8_K_generic  },
        { GGML_TYPE_IQ4_XS, "iq4_xs", ggml_vec_dot_iq4_xs_q8_K, ggml_vec_dot_iq4_xs_q8_K_generic },
    };

    int num_failed = 0;

    for (const auto & t : types) {
        ggml_quantize_init(t.wtype);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(t.wtype);
        const ggml_type atype = qfns_cpu->vec_dot_type;
        const auto * atf_cpu = ggml_get_type_traits_cpu(atype);

        const size_t bx = ggml_row_size(t.wtype, test_size);
        const size_t by = ggml_row_size(atype, test_size);
        std::vector<uint8_t> xb(bx), yb(by);

        // quantize weights (iq1_s / iq1_m have no trait from_float)
        if (qfns_cpu->from_float) {
            qfns_cpu->from_float(x_f32.data(), xb.data(), test_size);
        } else {
            ggml_quantize_chunk(t.wtype, x_f32.data(), xb.data(), 0, 1, test_size, imatrix.data());
        }

        // quantize activations
        atf_cpu->from_float(y_f32.data(), yb.data(), test_size);

        // inject -128 into the activation bytes (and fix up q8_K bsums)
        if (atype == GGML_TYPE_Q8_K) {
            struct test_block_q8_K {
                float d;
                int8_t qs[256];
                int16_t bsums[16];
            };
            test_block_q8_K * y = (test_block_q8_K *) yb.data();
            const size_t nb = test_size / 256;
            for (size_t i = 0; i < nb; i++) {
                const int old = y[i].qs[5];
                y[i].qs[5] = -128;
                y[i].bsums[0] += (-128 - old);
            }
        } else {  // Q8_0
            struct test_block_q8_0 {
                ggml_fp16_t d;
                int8_t qs[32];
            };
            test_block_q8_0 * y = (test_block_q8_0 *) yb.data();
            const size_t nb = test_size / 32;
            for (size_t i = 0; i < nb; i++) {
                y[i].qs[5] = -128;
            }
        }

        float sim = 0.0f, gen = 0.0f;
        t.sim(test_size, &sim, 0, xb.data(), 0, yb.data(), 0, 1);
        t.gen(test_size, &gen, 0, xb.data(), 0, yb.data(), 0, 1);

        const float err = fabsf(sim - gen) / fmaxf(fabsf(gen), 1e-30f);
        const bool failed = !(err < 1e-4f);
        num_failed += failed;
        if (failed || verbose) {
            printf(" %-5s vec_dot -128 contract:         %s (sim=%f gen=%f rel=%f)\n",
                   t.name, RESULT_STR[failed], sim, gen, err);
        }
    }

    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);

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
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
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

            const float vec_dot_error = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), nullptr, nullptr, 1);
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

// In every group, all values with importance are equal, and the max (0) has none.
// The scale search in make_qkx3_quants then fits min == max and passes inf/nan to nearest_int (#29804).
static int test_quantize_imatrix_degenerate(bool verbose) {
    const int64_t n = 256;
    std::vector<float> x(n);
    std::vector<float> imatrix(n);
    std::vector<float> out(n);
    int num_failed = 0;

    printf("Testing degenerate imatrix:\n");
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        const int64_t group = type == GGML_TYPE_Q2_K ? 16 : 32;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = i / group;
            const int64_t p = i % group;
            const bool important = p % 3 == 1;
            x[i] = important ? -0.02f*(g + 1) : (p % 2 ? -1.0f : 0.0f);
            imatrix[i] = important ? 1.0f : 0.0f;
        }
        printf("  - %s\n", ggml_type_name(type));

        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, imatrix.data());
        ggml_get_type_traits(type)->to_float(q.data(), out.data(), n);

        const bool failed = !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s imatrix degenerate groups:      %s\n", ggml_type_name(type), RESULT_STR[failed]);
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

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

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q8_0_i8_min(verbose);
    num_failed += test_vec_dot_neg128(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_quantize_imatrix_degenerate(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
