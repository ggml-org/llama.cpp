#include "vec.h"

#include <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

#if defined(GGML_SIMD)
    float sumf = 0.0f;

    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svfloat32_t sum4 = svdup_n_f32(0.0f);
        svfloat32_t sum5 = svdup_n_f32(0.0f);
        svfloat32_t sum6 = svdup_n_f32(0.0f);
        svfloat32_t sum7 = svdup_n_f32(0.0f);
        svfloat32_t sum8 = svdup_n_f32(0.0f);
        svfloat32_t ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8;
        svfloat32_t ay1,ay2,ay3,ay4,ay5,ay6,ay7,ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            sum2 = GGML_F32_VEC_FMA(sum2, ax2, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            sum3 = GGML_F32_VEC_FMA(sum3, ax3, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            sum4 = GGML_F32_VEC_FMA(sum4, ax4, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            sum5 = GGML_F32_VEC_FMA(sum5, ax5, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            sum6 = GGML_F32_VEC_FMA(sum6, ax6, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            sum7 = GGML_F32_VEC_FMA(sum7, ax7, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            sum8 = GGML_F32_VEC_FMA(sum8, ax8, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        if (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmad_f32_m(pg, ax1, ay1, sum1);
        }
        // reduce sum1,sum2 to sum1
        GGML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif
#elif defined(__riscv) && defined(__riscv_v)
    float sumf = 0.0f;
    __asm__ volatile(
        "vsetvli         t0,       zero,     e32, m4,tu,mu      \n\t"
        "vxor.vv         v16,      v16,      v16                \n\t"
        "LOOP%=:                                                \n\t"
        "vsetvli         t0,       %[n],     e32, m4,tu,mu      \n\t"
        "slli            t1,       t0,       2                  \n\t"
        "vle32.v         v0,       (%[lhs])                     \n\t"
        "add             %[lhs],   %[lhs],   t1                 \n\t"
        "vle32.v         v8,       (%[rhs])                     \n\t"
        "add             %[rhs],   %[rhs],   t1                 \n\t"
        "vfmacc.vv       v16,       v0,      v8                 \n\t"
        "sub             %[n],     %[n],     t0                 \n\t"
        "bnez            %[n],     LOOP%=                       \n\t"
        "vsetvli         t0,       zero,     e32, m4,tu,mu      \n\t"
        "vxor.vv         v24,      v24,      v24                \n\t"
        "vfredusum.vs    v24,      v16,      v24                \n\t"
        "vfmv.f.s        %[res],   v24                          \n\t"
        : [ n ] "+r"(n), [ lhs ] "+r"(x), [ rhs ] "+r"(y), [ res ] "=f"(sumf)
        :
        : "cc", "t0", "t1");
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;

#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

            sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
    }

    // if you hit this, you are likely running outside the FP range
    assert(!isnan(sumf) && !isinf(sumf));
#elif defined(__riscv) && defined(__riscv_v)
    float result = 0.0f;
    __asm__ volatile(
        "vsetvli         t0,       zero,     e32,    m4,tu,mu       \n\t"
        "vxor.vv         v16,      v16,      v16                    \n\t"
        "LOOP%=:                                                    \n\t"
        "vsetvli         t0,       %[n],     e16,    m2,tu,mu       \n\t"
        "slli            t1,       t0,       1                      \n\t"
        "vle16.v         v0,       (%[lhs])                         \n\t"
        "add             %[lhs],   %[lhs],   t1                     \n\t"
        "vle16.v         v2,       (%[rhs])                         \n\t"
        "add             %[rhs],   %[rhs],   t1                     \n\t"
        "vfwcvt.f.f.v    v4,       v0                               \n\t"
        "vfwcvt.f.f.v    v8,       v2                               \n\t"
        "vsetvli         t0,       %[n],     e32,    m4,tu,mu       \n\t"
        "vfmacc.vv       v16,      v4,       v8                     \n\t"
        "sub             %[n],     %[n],     t0                     \n\t"
        "bnez            %[n],     LOOP%=                           \n\t"
        "vsetvli         t0,       zero,     e32,    m4,tu,mu       \n\t"
        "vxor.vv         v24,      v24,      v24                    \n\t"
        "vfredusum.vs    v24,      v16,      v24                    \n\t"
        "vfmv.f.s        %[res],   v24                              \n\t"
        : [ n ] "+r"(n), [ lhs ] "+r"(x), [ rhs ] "+r"(y), [ res ] "=f"(result)
        :
        : "cc", "t0", "t1");
    sumf += result;
#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
    }
#endif

    *s = sumf;
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#elif defined(__riscv) && defined(__riscv_v)
    int N = n;
    i += n;
    constexpr struct {
    float LowerRange;
    float UpperRange;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_10;
    float beta_8;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float one_half;
    } LogisticConstants = {
        -18.0f,
        18.0f,
        4.37031012579801e-11f,
        1.15627324459942e-07f,
        6.08574864600143e-05f,
        8.51377133304701e-03f,
        2.48287947061529e-01f,
        6.10247389755681e-13f,
        5.76102136993427e-09f,
        6.29106785017040e-06f,
        1.70198817374094e-03f,
        1.16817656904453e-01f,
        9.93151921023180e-01f,
        0.5f,
    };
    __asm__ volatile(
        "LOOP%=:                                            \n\t"
        "vsetvli  t0,       %[n],     e32,     m1,tu,mu     \n\t"
        "sub      %[n],     %[n],     t0                    \n\t"
        "slli     t0,       t0,       2                     \n\t"
        "vfmv.v.f v20,      %[b0]                           \n\t"
        "vfmv.v.f v21,      %[a1]                           \n\t"
        "vfmv.v.f v22,      %[b2]                           \n\t"
        "vfmv.v.f v23,      %[a3]                           \n\t"
        "vfmv.v.f v24,      %[b4]                           \n\t"
        "vfmv.v.f v25,      %[a5]                           \n\t"
        "vfmv.v.f v26,      %[b6]                           \n\t"
        "vfmv.v.f v27,      %[a7]                           \n\t"
        "vfmv.v.f v28,      %[b8]                           \n\t"
        "vle32.v  v0,       (%[x])                          \n\t"
        "add      %[x],     %[x],     t0                    \n\t"
        "vfmax.vf v1,       v0,       %[lr]                 \n\t"
        "vfmin.vf v1,       v1,       %[ur]                 \n\t"
        "vfmul.vv v4,       v1,       v1                    \n\t"
        "vmv.v.v   v8,       v4                             \n\t"
        "vfmadd.vf v8,      %[a9],    v27                   \n\t"
        "vfmadd.vv v8,      v4,       v25                   \n\t"
        "vfmadd.vv v8,      v4,       v23                   \n\t"
        "vfmadd.vv v8,      v4,       v21                   \n\t"
        "vfmul.vv v8,       v8,       v1                    \n\t"
        "vmv.v.v   v12,      v4                             \n\t"
        "vfmadd.vf v12,     %[b10],   v28                   \n\t"
        "vfmadd.vv v12,     v4,       v26                   \n\t"
        "vfmadd.vv v12,     v4,       v24                   \n\t"
        "vfmadd.vv v12,     v4,       v22                   \n\t"
        "vfmadd.vv v12,     v4,       v20                   \n\t"
        "vfdiv.vv v12,      v8,       v12                   \n\t"
        "vfadd.vf v12,      v12,      %[onehalf]            \n\t"
        "vfmul.vv v12,      v12,      v0                    \n\t"  // sigmo
        "vse32.v  v12,      (%[y])                          \n\t"
        "add      %[y],     %[y],     t0                    \n\t"
        "bnez     %[n],     LOOP%=                          \n\t"
        : [ n ] "+r"(N), [ x ] "+r"(x), [ y ] "+r"(y)
        : [ lr ] "f"(LogisticConstants.LowerRange),
            [ ur ] "f"(LogisticConstants.UpperRange),
            [ a1 ] "f"(LogisticConstants.alpha_1),
            [ a3 ] "f"(LogisticConstants.alpha_3),
            [ a5 ] "f"(LogisticConstants.alpha_5),
            [ a7 ] "f"(LogisticConstants.alpha_7),
            [ a9 ] "f"(LogisticConstants.alpha_9),
            [ b0 ] "f"(LogisticConstants.beta_0),
            [ b2 ] "f"(LogisticConstants.beta_2),
            [ b4 ] "f"(LogisticConstants.beta_4),
            [ b6 ] "f"(LogisticConstants.beta_6),
            [ b8 ] "f"(LogisticConstants.beta_8),
            [ b10 ] "f"(LogisticConstants.beta_10),
            [ onehalf ] "f"(LogisticConstants.one_half)
        : "cc", "t0");
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_silu(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_silu(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_silu(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_silu(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#elif defined(__riscv) && defined(__riscv_v) && defined(__riscv_zba)
    int N = n;
    i += n;
    float* src = const_cast<float*>(reinterpret_cast<const float*>(x));
    float* dst = reinterpret_cast<float*>(y);
    float Accumulator = 0.0f;
    const float Neg_Max = -max;

    const float LowerRange = -103.9720840454f;
    const float UpperRange = 88.7762626647950f;
    const float LowerRangeSumExp = -88.3762626647949f;
    const float UpperRangeSumExp = 88.3762626647949f;
    const float RoundingBias = 12582912.f;
    const float Log2Reciprocal = 1.44269504088896341f;
    const float Log2High = -6.93145752e-1f;
    const float Log2Low = -1.42860677e-6f;
    const float poly_0 = 0x1.694000p-10;
    const float poly_1 = 0x1.125edcp-7;
    const float poly_2 = 0x1.555b5ap-5;
    const float poly_3 = 0x1.555450p-3;
    const float poly_4 = 0x1.fffff6p-2;
    const float poly_56 = 0x1.000000p+0;
    // int32_t MinimumExponent = int32_t(0xC1000000);    //unused
    const int32_t MaximumExponent = int32_t(0x3F800000);

    __asm__ volatile(
        "mv                   t3, %[LEN]                                  \n\t"
        "mv                   s1, %[SRC]                                  \n\t"
        "mv                   s2, %[DST]                                  \n\t"

        /* 2.0 Compute exp() and accumulate and store to cache_buffer */
        "vsetvli              t0, zero, e32, m4,tu,mu                     \n\t"
        "vxor.vv              v16, v8, v8                                 \n\t"
        "vxor.vv              v0, v8, v8                                  \n\t"

        ".align 4                                                         \n\t"
        "_EXPACC_LEN_LPST:                                                \n\t"
        "vsetvli              t0, t3, e32, m4,tu,mu                       \n\t"

        "vle32.v              v0, (s1)                                    \n\t"
        "sh2add               s1, t0, s1                                  \n\t"

        /* 2.1 START exp()  */
        "vfadd.vf             v0, v0, %[NEG_MAX]                          \n\t"  // v4 = x - max

        // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed
        // safely with a simple shift into the exponent field. xmin =
        // round(-126.5 * log(2), single, RU) ~ -87.68311309814453125 const
        // float xmin = -0x1.5ebb82p6;
        "vfmax.vf             v0, v0, %[LowerRangeSumExp]                 \n\t"

        // 2.1.0. Reduction x = s * q ln(2)
        // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
        // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
        // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single,
        // RN);
        "vfmv.v.f             v4, %[RoundingBias]                         \n\t"
        "vfmacc.vf            v4, %[Log2Reciprocal], v0                   \n\t"  // biased in mlas
        "vfsub.vf             v8, v4, %[RoundingBias]                     \n\t"  // v12_a = float(x - n);

        // Use Cody-Waite range reduction method (note two constants to
        // represent log(2)) to improve accuracy.
        "vfmacc.vf            v0, %[Log2High], v8                         \n\t"
        "vfmacc.vf            v0, %[Log2Low], v8                          \n\t"
        "vfcvt.x.f.V          v8, v4                                      \n\t"

        // 2.1.1. Approximate e^s by degree-6 polynomial approximation
        "vfmv.v.f             v4, %[poly_0]                               \n\t"
        "vfmv.v.f             v12, %[poly_1]                              \n\t"
        "vfmadd.vv            v4, v0, v12                                 \n\t"
        "vfmv.v.f             v12, %[poly_2]                              \n\t"
        "vfmadd.vv            v4, v0, v12                                 \n\t"
        "vfmv.v.f             v12, %[poly_3]                              \n\t"
        "vfmadd.vv            v4, v0, v12                                 \n\t"
        "vfmv.v.f             v12, %[poly_4]                              \n\t"
        "vfmadd.vv            v4, v0, v12                                 \n\t"
        "vfmv.v.f             v12, %[poly_56]                             \n\t"
        "vfmadd.vv            v4, v0, v12                                 \n\t"
        "vfmv.v.f             v12, %[poly_56]                             \n\t"
        "vfmadd.vv            v4, v0, v12                                 \n\t"  // v8 = poly(input - max)

        // 2.1.2. Reconstruction: compute u = u*2^q
        // const int16_t p = (24 - 1);
        // const int16_t bias = (128 - 1);
        "vsll.vi              v8, v8, 23                                  \n\t"
        "vadd.vx              v8, v8, %[MaximumExponent]                  \n\t"
        //"vfcvt.f.x.v          v12, v8                                   \n\t"

        "vfmul.vv             v0, v4, v8                                  \n\t"
        /* 2.1 END exp()  */

        "vse32.v              v0, (s2)                                    \n\t"  // exp(输入-max)输出
        "sh2add               s2, t0, s2                                  \n\t"
        "vfadd.vv             v16, v16, v0                                \n\t"
        "sub                  t3, t3, t0                                  \n\t"
        "bgtz                 t3, _EXPACC_LEN_LPST                        \n\t"

        "_EXPACC_LEN_LPND:                                                \n\t"

        "vsetvli              t0, zero, e32, m4,tu,mu                     \n\t"
        "vxor.vv              v24, v8, v8                                 \n\t"
        "vfredosum.vs         v24, v16, v24                               \n\t"
        "vfmv.f.s             %[RTN], v24                                 \n\t"  // ft2 = sum(exp( ))

        : [ RTN ] "=f"(Accumulator), [ SRC ] "+r"(src), [ DST ] "+r"(dst)
        : [ LEN ] "r"(N), [ NEG_MAX ] "f"(Neg_Max), [ LowerRange ] "f"(LowerRange), [ UpperRange ] "f"(UpperRange),
          [ LowerRangeSumExp ] "f"(LowerRangeSumExp), [ UpperRangeSumExp ] "f"(UpperRangeSumExp),
          [ RoundingBias ] "f"(RoundingBias), [ Log2Reciprocal ] "f"(Log2Reciprocal), [ Log2High ] "f"(Log2High),
          [ Log2Low ] "f"(Log2Low), [ poly_0 ] "f"(poly_0), [ poly_1 ] "f"(poly_1), [ poly_2 ] "f"(poly_2),
          [ poly_3 ] "f"(poly_3), [ poly_4 ] "f"(poly_4), [ poly_56 ] "f"(poly_56),
          [ MaximumExponent ] "r"(MaximumExponent)
        : "cc", "s1", "s2", "t0", "t3");
    sum += (ggml_float)Accumulator;
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (ggml_float)val;
        y[i] = val;
    }
    return sum;
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}
