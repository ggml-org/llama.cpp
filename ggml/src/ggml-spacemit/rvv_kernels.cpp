#include "rvv_kernels.h"

#include "common.h"
#include "ggml.h"
#include "string.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#else
#    include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#if defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace spacemit_kernels::rvv {

namespace {

auto align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// head dims above vlmax(e16m2) = 128 are processed in 128-wide segments
static constexpr int64_t FA_VLEN1024_MAX_D = 256;
// split-KV decode handles up to this many query tokens (speculative-decoding verification)
static constexpr int64_t FA_SPLIT_KV_MAX_Q = 4;

static inline bool flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    return d > 0 && d <= FA_VLEN1024_MAX_D;
}

static inline bool flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    return flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

static inline float reduce_sum_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    return __riscv_vfmv_f_s_f32m1_f32(s_v);
}

static inline void rvv_fill_f16(_Float16 * dst, float value, int64_t n) {
    for (int64_t d0 = 0; d0 < n;) {
        const size_t vl = __riscv_vsetvl_e16m2(n - d0);
        __riscv_vse16_v_f16m2(dst + d0, __riscv_vfmv_v_f_f16m2((_Float16) value, vl), vl);
        d0 += vl;
    }
}

static inline void rvv_scale_f16(_Float16 * dst, float scale, int64_t n) {
    for (int64_t d0 = 0; d0 < n;) {
        const size_t vl = __riscv_vsetvl_e16m2(n - d0);
        vfloat16m2_t v  = __riscv_vle16_v_f16m2(dst + d0, vl);
        __riscv_vse16_v_f16m2(dst + d0, __riscv_vfmul_vf_f16m2(v, scale, vl), vl);
        d0 += vl;
    }
}

// acc = acc * ms + vs * v (the multiply by ms is skipped when ms == 1, as in the online-softmax update)
static inline void rvv_scale_mad_f16(_Float16 * acc, float ms, float vs, const _Float16 * v, int64_t n) {
    for (int64_t d0 = 0; d0 < n;) {
        const size_t vl = __riscv_vsetvl_e16m2(n - d0);
        vfloat16m2_t a  = __riscv_vle16_v_f16m2(acc + d0, vl);
        if (ms != 1.0f) {
            a = __riscv_vfmul_vf_f16m2(a, ms, vl);
        }
        a = __riscv_vfmacc_vf_f16m2(a, vs, __riscv_vle16_v_f16m2(v + d0, vl), vl);
        __riscv_vse16_v_f16m2(acc + d0, a, vl);
        d0 += vl;
    }
}

// f16 . f16 with f32 products, reduced per segment
static inline float rvv_dot_f16_f32(const _Float16 * a, const _Float16 * b, int64_t n) {
    float s = 0.0f;
    for (int64_t d0 = 0; d0 < n;) {
        const size_t       vl = __riscv_vsetvl_e16m2(n - d0);
        const vfloat32m4_t p =
            __riscv_vfwmul_vv_f32m4(__riscv_vle16_v_f16m2(a + d0, vl), __riscv_vle16_v_f16m2(b + d0, vl), vl);
        vfloat32m1_t r = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        s += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(p, r, vl));
        d0 += vl;
    }
    return s;
}

// dst = VKQ16 / S, after folding in the attention sink (if any)
static inline void flash_attn_ext_finalize_row_f16(float *          dst,
                                                   const _Float16 * vkq16,
                                                   int64_t          dv,
                                                   float            S,
                                                   float            M,
                                                   const float *    sink) {
    float ms = 1.0f;
    if (sink) {
        const float s  = *sink;
        float       vs = 1.0f;
        if (s > M) {
            ms = expf(M - s);
        } else {
            vs = expf(s - M);
        }
        S = S * ms + vs;
    }
    const float S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    for (int64_t d0 = 0; d0 < dv;) {
        const size_t vl = __riscv_vsetvl_e16m2(dv - d0);
        vfloat32m4_t v  = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(vkq16 + d0, vl), vl);
        if (ms != 1.0f) {
            v = __riscv_vfmul_vf_f32m4(v, ms, vl);
        }
        __riscv_vse32_v_f32m4(dst + d0, __riscv_vfmul_vf_f32m4(v, S_inv, vl), vl);
        d0 += vl;
    }
}

static inline float reduce_sum_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    return __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
static inline vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    const vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    if (!__riscv_vcpop_m_b16(c, vl)) {
        return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    const vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    const vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    const vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    return __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

static inline vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    const vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    const vfloat32m2_t exp_term    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    const vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_term, 1.0f, vl);
    const vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_term, 1.0f, vl);
    const vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    const vbool16_t    neg_mask    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    const vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    return __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_mask, vl);
}

static void rvv_softcap_tanh_inplace_f32(float * dst, int64_t dst_stride, int64_t tile_rows, int64_t n, float softcap) {
    for (int tq = 0; tq < tile_rows; ++tq, dst += dst_stride) {
        float * dst_row   = dst;
        int64_t remaining = n;
        while (remaining > 0) {
            const size_t vl = __riscv_vsetvl_e32m2(remaining);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_row, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_row, v, vl);
            dst_row += vl;
            remaining -= vl;
        }
    }
}

static inline float rvv_softmax_exp_inplace_f32(float * dst, int64_t n, float max_value) {
    float row_sum = 0.0f;
    while (n > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_value, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        row_sum += reduce_sum_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    return row_sum;
}

static inline float rvv_add_max_inplace_f32(float * dst, const float * src, int64_t n) {
    float max_val = -INFINITY;
    while (n > 0) {
        const size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, seed, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(seed);

        dst += vl;
        src += vl;
        n -= vl;
    }
    return max_val;
}

static inline float rvv_softcap_add_max_inplace_f32(float * dst, const float * src, int64_t n, float softcap) {
    if (softcap == 0.0f) {
        return rvv_add_max_inplace_f32(dst, src, n);
    }

    float max_val = -INFINITY;
    while (n > 0) {
        const size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, seed, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(seed);

        dst += vl;
        src += vl;
        n -= vl;
    }
    return max_val;
}

static inline void rvv_zero_f32(float * dst, int64_t n) {
    while (n > 0) {
        const size_t       vl = __riscv_vsetvl_e32m4(n);
        const vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

static inline void rvv_scale_f32(float * dst, float scale, int64_t n) {
    while (n > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

static inline void rvv_add_inplace_f32(float *       dst,
                                       int64_t       dst_stride,
                                       const float * src,
                                       int64_t       src_stride,
                                       int64_t       tile_rows,
                                       int64_t       n) {
    for (int tq = 0; tq < tile_rows; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       remaining = n;
        float *       dst_row   = dst;
        const float * src_row   = src;
        while (remaining > 0) {
            const size_t vl   = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_row, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_row, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_row, vdst, vl);
            dst_row += vl;
            src_row += vl;
            remaining -= vl;
        }
    }
}

static inline float rvv_max_f32(const float * src, int64_t n) {
    float max_val = -INFINITY;
    while (n > 0) {
        const size_t       vl   = __riscv_vsetvl_e32m4(n);
        const vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed                    = __riscv_vfredmax_vs_f32m4_f32m1(v, seed, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(seed);
        src += vl;
        n -= vl;
    }
    return max_val;
}

static void rvv_pack_f32_as_scaled_f16(void *       dst,
                                       int64_t      dst_row_stride,
                                       const void * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       float        scale) {
    for (int tq = 0; tq < tile_rows; ++tq) {
        const float * row_ptr     = (const float *) ((const char *) src + tq * src_row_stride);
        _Float16 *    dst_row_ptr = (_Float16 *) ((char *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        while (remaining > 0) {
            const size_t vl        = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            const vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_row_ptr, v16, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

static void rvv_pack_scaled_f16_as_f32(void *       dst,
                                       int64_t      dst_row_stride,
                                       const void * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       float        scale) {
    for (int tq = 0; tq < tile_rows; ++tq) {
        const _Float16 * row_ptr     = (const _Float16 *) ((const char *) src + tq * src_row_stride);
        float *          dst_row_ptr = (float *) ((char *) dst + tq * dst_row_stride);
        int64_t          remaining   = n;
        while (remaining > 0) {
            const size_t       vl  = __riscv_vsetvl_e16m2(remaining);
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(row_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

static void rvv_pack_scaled_f32_as_f32(void *       dst,
                                       int64_t      dst_row_stride,
                                       const void * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       float *      scale) {
    for (int tq = 0; tq < tile_rows; ++tq) {
        const float * row_ptr     = (const float *) ((const char *) src + tq * src_row_stride);
        float *       dst_row_ptr = (float *) ((char *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        while (remaining > 0) {
            const size_t vl  = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

static inline void rvv_transposed_s32_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ volatile(
        "vsetvli                t0, zero, e32, m1, tu, mu     \n\t"
        "mul                    t3, t0, %[os0]                \n\t"
        "srli                   t2, %[isz0], 3                \n\t"
        "blez                   t2, M1%=                      \n\t"

        "LOOP_M8%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "add                    s2, %[src], %[is0]            \n\t"
        "add                    s3, s2, %[is0]                \n\t"
        "add                    s4, s3, %[is0]                \n\t"
        "add                    s5, s4, %[is0]                \n\t"
        "add                    s6, s5, %[is0]                \n\t"
        "add                    s7, s6, %[is0]                \n\t"
        "add                    s8, s7, %[is0]                \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M8N%=:                                          \n\t"
        "vsetvli                t0, t1, e32, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle32.v                v0, (s1)                      \n\t"
        "sh2add                 s1, t0, s1                    \n\t"
        "vle32.v                v1, (s2)                      \n\t"
        "sh2add                 s2, t0, s2                    \n\t"
        "vle32.v                v2, (s3)                      \n\t"
        "sh2add                 s3, t0, s3                    \n\t"
        "vle32.v                v3, (s4)                      \n\t"
        "sh2add                 s4, t0, s4                    \n\t"
        "vle32.v                v4, (s5)                      \n\t"
        "sh2add                 s5, t0, s5                    \n\t"
        "vle32.v                v5, (s6)                      \n\t"
        "sh2add                 s6, t0, s6                    \n\t"
        "vle32.v                v6, (s7)                      \n\t"
        "sh2add                 s7, t0, s7                    \n\t"
        "vle32.v                v7, (s8)                      \n\t"
        "sh2add                 s8, t0, s8                    \n\t"
        "vssseg8e32.v           v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M8N%=                \n\t"
        "sh3add                 %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 32            \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M8%=                 \n\t"

        "M1%=:                                                \n\t"
        "andi                   t2, %[isz0], 7                \n\t"
        "blez                   t2, END%=                     \n\t"

        "LOOP_M1%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M1N%=:                                          \n\t"
        "vsetvli                t0, t1, e32, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle32.v                v0, (s1)                      \n\t"
        "sh2add                 s1, t0, s1                    \n\t"
        "vsse32.v               v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M1N%=                \n\t"
        "add                    %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 4             \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M1%=                 \n\t"
        "END%=:                                               \n\t"

        : [src] "+r"(in), [dst] "+r"(out), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

static inline void rvv_transposed_s16_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ volatile(
        "vsetvli                t0, zero, e16, m1, tu, mu     \n\t"
        "mul                    t3, t0, %[os0]                \n\t"
        "srli                   t2, %[isz0], 3                \n\t"
        "blez                   t2, M1%=                      \n\t"

        "LOOP_M8%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "add                    s2, %[src], %[is0]            \n\t"
        "add                    s3, s2, %[is0]                \n\t"
        "add                    s4, s3, %[is0]                \n\t"
        "add                    s5, s4, %[is0]                \n\t"
        "add                    s6, s5, %[is0]                \n\t"
        "add                    s7, s6, %[is0]                \n\t"
        "add                    s8, s7, %[is0]                \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M8N%=:                                          \n\t"
        "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle16.v                v0, (s1)                      \n\t"
        "sh1add                 s1, t0, s1                    \n\t"
        "vle16.v                v1, (s2)                      \n\t"
        "sh1add                 s2, t0, s2                    \n\t"
        "vle16.v                v2, (s3)                      \n\t"
        "sh1add                 s3, t0, s3                    \n\t"
        "vle16.v                v3, (s4)                      \n\t"
        "sh1add                 s4, t0, s4                    \n\t"
        "vle16.v                v4, (s5)                      \n\t"
        "sh1add                 s5, t0, s5                    \n\t"
        "vle16.v                v5, (s6)                      \n\t"
        "sh1add                 s6, t0, s6                    \n\t"
        "vle16.v                v6, (s7)                      \n\t"
        "sh1add                 s7, t0, s7                    \n\t"
        "vle16.v                v7, (s8)                      \n\t"
        "sh1add                 s8, t0, s8                    \n\t"
        "vssseg8e16.v           v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M8N%=                \n\t"
        "sh3add                 %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 16            \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M8%=                 \n\t"

        "M1%=:                                                \n\t"
        "andi                   t2, %[isz0], 7                \n\t"
        "blez                   t2, END%=                     \n\t"

        "LOOP_M1%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M1N%=:                                          \n\t"
        "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
        "sub                    t1, t1, t0                    \n\t"
        "vle16.v                v0, (s1)                      \n\t"
        "sh1add                 s1, t0, s1                    \n\t"
        "vsse16.v               v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M1N%=                \n\t"
        "add                    %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 2             \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M1%=                 \n\t"
        "END%=:                                               \n\t"

        : [src] "+r"(in), [dst] "+r"(out), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

static inline void rvv_qk_dot_tile_f16_x1(float *          dst,
                                          const _Float16 * q_row,
                                          const _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    const size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_row[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

static inline void rvv_qk_dot_tile_f16_x4(float *          dst0,
                                          float *          dst1,
                                          float *          dst2,
                                          float *          dst3,
                                          const _Float16 * q0,
                                          const _Float16 * q1,
                                          const _Float16 * q2,
                                          const _Float16 * q3,
                                          const _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    const size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc0                     = __riscv_vfwmacc_vf_f32m2(acc0, q0[d], k_vec, vl);
        acc1                     = __riscv_vfwmacc_vf_f32m2(acc1, q1[d], k_vec, vl);
        acc2                     = __riscv_vfwmacc_vf_f32m2(acc2, q2[d], k_vec, vl);
        acc3                     = __riscv_vfwmacc_vf_f32m2(acc3, q3[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst0, acc0, vl);
    __riscv_vse32_v_f32m2(dst1, acc1, vl);
    __riscv_vse32_v_f32m2(dst2, acc2, vl);
    __riscv_vse32_v_f32m2(dst3, acc3, vl);
}

static inline void rvv_pv_accumulate_f16_x1(float *          dst,
                                            const float *    prob,
                                            const _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_left = dv;
    int64_t d_off  = 0;

    while (d_left > 0) {
        const size_t vl  = __riscv_vsetvl_e16m2(d_left);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        for (int64_t tk = 0; tk < kv_tile; ++tk) {
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            const vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_left -= vl;
        d_off += vl;
    }
}

static inline void rvv_pv_accumulate_f16_x4(float *          dst0,
                                            float *          dst1,
                                            float *          dst2,
                                            float *          dst3,
                                            const float *    prob0,
                                            const float *    prob1,
                                            const float *    prob2,
                                            const float *    prob3,
                                            const _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_left = dv;
    int64_t d_off  = 0;

    while (d_left > 0) {
        const size_t vl   = __riscv_vsetvl_e16m2(d_left);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        for (int64_t tk = 0; tk < kv_tile; ++tk) {
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            const vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_left -= vl;
        d_off += vl;
    }
}

static inline void rvv_qk_dot_tile(float *       dst,
                                   const float * q_row,
                                   const float * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_stride,
                                   int64_t       kv_tile,
                                   float         scale) {
    const size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_stride, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_row[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

static inline void rvv_pv_accumulate(float *       dst,
                                     const float * prob,
                                     const float * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_left = dv;
    int64_t d_off  = 0;

    while (d_left > 0) {
        const size_t vl  = __riscv_vsetvl_e32m4(d_left);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        for (int64_t tk = 0; tk < kv_tile; ++tk) {
            const vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_left -= vl;
        d_off += vl;
    }
}

static void permute_transpose_impl(const ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   int                 ith,
                                   int                 nth) {
    GGML_ASSERT(n_src_stride == sizeof(int32_t) || n_src_stride == sizeof(int16_t));

    if (n_src_stride == sizeof(int32_t)) {
        for (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_to_nm((int8_t *) ((char *) dst->data + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((char *) src0->data + bi * batch_stride), m_src_stride, m, n);
        }
    } else if (n_src_stride == sizeof(int16_t)) {
        for (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_to_nm((int8_t *) ((char *) dst->data + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((char *) src0->data + bi * batch_stride), m_src_stride, m, n);
        }
    } else {
        GGML_ABORT("not implemented");
    }
}

template <size_t QLEN>
static void flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(float **            pq,
                                                                  const char *        k_data_row,
                                                                  const char *        v_data_row,
                                                                  const ggml_fp16_t * mp,
                                                                  float **            sinks,
                                                                  float **            dst,
                                                                  float               scale,
                                                                  float               logit_softcap,
                                                                  float               slope,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  void *              tcm_buffer,
                                                                  size_t              tcm_buffer_size,
                                                                  float **            partial         = nullptr,
                                                                  int64_t             n_mask_rows     = 1,
                                                                  size_t              mask_row_stride = 0) {
    // QLEN rows share K/V; groups of 4 rows are computed from the packed K/V tile in TCM.
    // Rows may come from n_mask_rows query tokens (QLEN / n_mask_rows consecutive rows per token), each with
    // its own mask row at mp + m * mask_row_stride bytes; a KV position is packed if any token can see it.
    // With partial != nullptr row i writes {M, S, unnormalized VKQ[DV]} to partial[i] (for a later split-KV
    // reduction) instead of the normalized output to dst[i].
    static_assert(QLEN == 2 || QLEN == 4 || QLEN == 8 || QLEN == 16, "unsupported QLEN");
    GGML_ASSERT(n_mask_rows >= 1 && QLEN % n_mask_rows == 0 && n_mask_rows <= 16);
    const int64_t rows_per_mask = QLEN / n_mask_rows;
    auto          mask_at       = [&](int64_t m, int64_t ic) -> float {
        return mp ? slope * ((const _Float16 *) ((const char *) mp + m * mask_row_stride))[ic] : 0.0f;
    };
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    float S[QLEN];  // sum
    float M[QLEN];  // maximum KQ value
    for (size_t i = 0; i < QLEN; ++i) {
        S[i] = 0.0f;
        M[i] = -INFINITY;
    }

    // rows longer than vlmax(e16m2) = 128 are processed in segments
    const int64_t seg = (int64_t) __riscv_vsetvlmax_e16m2();

    _Float16 *   kq16_buffer          = (_Float16 *) tcm_buffer;
    _Float16 *   qv_buffer            = kq16_buffer + QLEN * DV;
    const size_t qkv_temp_buffer_size = (QLEN * DV + QLEN * DK) * sizeof(_Float16);
    char *       kv_tile_buffer       = (char *) (qv_buffer + QLEN * DK);

    for (size_t i = 0; i < QLEN; ++i) {
        rvv_fill_f16(kq16_buffer + i * DV, 0.0f, DV);
        for (int64_t d0 = 0; d0 < DK; d0 += seg) {
            const size_t vl = __riscv_vsetvl_e16m2(DK - d0);
            __riscv_vse16_v_f16m2(qv_buffer + i * DK + d0,
                                  __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i] + d0, vl), vl), vl);
        }
    }

    const uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_buffer);
    const size_t    scratch_size = tcm_buffer_size > qkv_temp_buffer_size ? tcm_buffer_size - qkv_temp_buffer_size : 0;
    const uintptr_t kq_tile_addr = align_up(scratch_addr, alignof(float));
    const size_t    scratch_prefix = kq_tile_addr - scratch_addr;
    const size_t    packed_tile_size =
        QLEN * sizeof(float) + DK * sizeof(_Float16) + DV * sizeof(_Float16) + n_mask_rows * sizeof(float);
    const int64_t max_ic_tile_step = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    const int64_t max_fit_by_tcm =
        scratch_size > scratch_prefix ? (int64_t) ((scratch_size - scratch_prefix) / packed_tile_size) : 0;
    const int64_t ic_tile_step = std::min(max_ic_tile_step, max_fit_by_tcm) & ~((int64_t) 7);

    const uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_step * sizeof(float);
    const uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_step * sizeof(_Float16);
    const uintptr_t mv_tile_addr = v_tile_addr + ic_tile_step * DV * sizeof(_Float16);

    if (ic_tile_step >= 8) {
        float *    kq_tile_buffer = reinterpret_cast<float *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        float *    mv_tile_pack   = reinterpret_cast<float *>(mv_tile_addr);

        const int64_t k_tile_byte_stride = ic_tile_step * (int64_t) sizeof(_Float16);

        int64_t ic_step = 0;
        for (int64_t ic = 0; ic < nek1; ++ic) {
            float mvs[16];
            bool  visible = false;
            for (int64_t m = 0; m < n_mask_rows; ++m) {
                mvs[m] = mask_at(m, ic);
                visible |= mvs[m] != -INFINITY;
            }

            if (visible) {
                const _Float16 * k_data = (const _Float16 *) (k_data_row + ic * nbk1);
                const _Float16 * v_data = (const _Float16 *) (v_data_row + ic * nbv1);

                // K tile is stored transposed (d-major), V tile row-major
                for (int64_t d0 = 0; d0 < DK; d0 += seg) {
                    const size_t       vl       = __riscv_vsetvl_e16m2(DK - d0);
                    const vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2(k_data + d0, vl);
                    __riscv_vsse16_v_f16m2(k_tile_pack + d0 * ic_tile_step + ic_step, k_tile_byte_stride, k_data_v,
                                           vl);
                }
                for (int64_t d0 = 0; d0 < DV; d0 += seg) {
                    const size_t       vl       = __riscv_vsetvl_e16m2(DV - d0);
                    const vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2(v_data + d0, vl);
                    __riscv_vse16_v_f16m2(v_tile_pack + ic_step * DV + d0, v_data_v, vl);
                }
                for (int64_t m = 0; m < n_mask_rows; ++m) {
                    mv_tile_pack[m * ic_tile_step + ic_step] = mvs[m];
                }
                ic_step++;
            }

            if (ic_step > 0 && (ic_step == ic_tile_step || ic == (nek1 - 1))) {
                const size_t qk_vl = __riscv_vsetvl_e16m1(ic_step);
                for (size_t g = 0; g < QLEN; g += 4) {
                    if constexpr (QLEN >= 4) {
                        const _Float16 * q0 = qv_buffer + (g + 0) * DK;
                        const _Float16 * q1 = qv_buffer + (g + 1) * DK;
                        const _Float16 * q2 = qv_buffer + (g + 2) * DK;
                        const _Float16 * q3 = qv_buffer + (g + 3) * DK;

                        vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                        vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                        vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                        vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                        for (int64_t d = 0; d < DK; ++d) {
                            const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                            qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, q0[d], k_vec, qk_vl);
                            qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, q1[d], k_vec, qk_vl);
                            qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, q2[d], k_vec, qk_vl);
                            qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, q3[d], k_vec, qk_vl);
                        }

                        qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                        qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                        qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                        qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                        __riscv_vse32_v_f32m2(kq_tile_buffer + (g + 0) * ic_tile_step, qk_acc0, qk_vl);
                        __riscv_vse32_v_f32m2(kq_tile_buffer + (g + 1) * ic_tile_step, qk_acc1, qk_vl);
                        __riscv_vse32_v_f32m2(kq_tile_buffer + (g + 2) * ic_tile_step, qk_acc2, qk_vl);
                        __riscv_vse32_v_f32m2(kq_tile_buffer + (g + 3) * ic_tile_step, qk_acc3, qk_vl);
                    } else {
                        vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                        vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                        for (int64_t d = 0; d < DK; ++d) {
                            const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                            qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_buffer[0 * DK + d], k_vec, qk_vl);
                            qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_buffer[1 * DK + d], k_vec, qk_vl);
                        }

                        qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                        qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                        __riscv_vse32_v_f32m2(kq_tile_buffer + 0 * ic_tile_step, qk_acc0, qk_vl);
                        __riscv_vse32_v_f32m2(kq_tile_buffer + 1 * ic_tile_step, qk_acc1, qk_vl);
                    }
                }

                for (size_t i = 0; i < QLEN; ++i) {
                    float *     row_ptr = kq_tile_buffer + i * ic_tile_step;
                    const float tile_max = rvv_softcap_add_max_inplace_f32(
                        row_ptr, mv_tile_pack + (i / rows_per_mask) * ic_tile_step, ic_step, logit_softcap);

                    const float Mold = M[i];

                    if (tile_max > Mold) {
                        const float ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;
                        rvv_scale_f16(kq16_buffer + i * DV, ms, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(row_ptr, ic_step, M[i]);
                }

                for (int64_t d0 = 0; d0 < DV; d0 += seg) {
                    const size_t vl = __riscv_vsetvl_e16m2(DV - d0);
                    for (size_t g = 0; g < QLEN; g += 4) {
                        if constexpr (QLEN >= 4) {
                            _Float16 *    o0 = kq16_buffer + (g + 0) * DV + d0;
                            _Float16 *    o1 = kq16_buffer + (g + 1) * DV + d0;
                            _Float16 *    o2 = kq16_buffer + (g + 2) * DV + d0;
                            _Float16 *    o3 = kq16_buffer + (g + 3) * DV + d0;
                            const float * p0 = kq_tile_buffer + (g + 0) * ic_tile_step;
                            const float * p1 = kq_tile_buffer + (g + 1) * ic_tile_step;
                            const float * p2 = kq_tile_buffer + (g + 2) * ic_tile_step;
                            const float * p3 = kq_tile_buffer + (g + 3) * ic_tile_step;

                            vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(o0, vl);
                            vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(o1, vl);
                            vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(o2, vl);
                            vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(o3, vl);

                            for (int64_t tk = 0; tk < ic_step; ++tk) {
                                const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV + d0, vl);
                                pv_acc0 = __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) p0[tk], v16, vl);
                                pv_acc1 = __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) p1[tk], v16, vl);
                                pv_acc2 = __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) p2[tk], v16, vl);
                                pv_acc3 = __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) p3[tk], v16, vl);
                            }

                            __riscv_vse16_v_f16m2(o0, pv_acc0, vl);
                            __riscv_vse16_v_f16m2(o1, pv_acc1, vl);
                            __riscv_vse16_v_f16m2(o2, pv_acc2, vl);
                            __riscv_vse16_v_f16m2(o3, pv_acc3, vl);
                        } else {
                            vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV + d0, vl);
                            vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV + d0, vl);

                            for (int64_t tk = 0; tk < ic_step; ++tk) {
                                const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV + d0, vl);
                                pv_acc0 = __riscv_vfmacc_vf_f16m2(
                                    pv_acc0, (_Float16) kq_tile_buffer[0 * ic_tile_step + tk], v16, vl);
                                pv_acc1 = __riscv_vfmacc_vf_f16m2(
                                    pv_acc1, (_Float16) kq_tile_buffer[1 * ic_tile_step + tk], v16, vl);
                            }

                            __riscv_vse16_v_f16m2(kq16_buffer + 0 * DV + d0, pv_acc0, vl);
                            __riscv_vse16_v_f16m2(kq16_buffer + 1 * DV + d0, pv_acc1, vl);
                        }
                    }
                }

                ic_step = 0;
            }
        }
    } else {
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const _Float16 * k_data = (const _Float16 *) (k_data_row + ic * nbk1);
            const _Float16 * v_data = (const _Float16 *) (v_data_row + ic * nbv1);

            for (size_t i = 0; i < QLEN; ++i) {
                const float mv = mask_at(i / rows_per_mask, ic);
                if (mv == -INFINITY) {
                    continue;
                }
                float s = rvv_dot_f16_f32(k_data, qv_buffer + i * DK, DK) * scale;
                if (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                const float Mold = M[i];

                float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                if (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                rvv_scale_mad_f16(kq16_buffer + i * DV, ms, vs, v_data, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    for (size_t i = 0; i < QLEN; ++i) {
        if (partial) {
            float * pr = partial[i];
            pr[0]      = M[i];
            pr[1]      = S[i];
            for (int64_t d0 = 0; d0 < DV; d0 += seg) {
                const size_t vl = __riscv_vsetvl_e16m2(DV - d0);
                __riscv_vse32_v_f32m4(pr + 2 + d0,
                                      __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(kq16_buffer + i * DV + d0, vl), vl),
                                      vl);
            }
        } else {
            flash_attn_ext_finalize_row_f16(dst[i], kq16_buffer + i * DV, DV, S[i], M[i], sinks[i]);
        }
    }
}

static void flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(const float *       pq,
                                                                const char *        k_data_row,
                                                                const char *        v_data_row,
                                                                const ggml_fp16_t * mp,
                                                                const float *       sinks,
                                                                float *             dst,
                                                                float               scale,
                                                                float               logit_softcap,
                                                                float               slope,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    float S = 0.0f;       // sum
    float M = -INFINITY;  // maximum KQ value

    _Float16 VKQ16[FA_VLEN1024_MAX_D];
    _Float16 Q16[FA_VLEN1024_MAX_D];

    const int64_t seg = (int64_t) __riscv_vsetvlmax_e16m2();
    rvv_fill_f16(VKQ16, 0.0f, DV);
    for (int64_t d0 = 0; d0 < DK; d0 += seg) {
        const size_t vl = __riscv_vsetvl_e16m2(DK - d0);
        __riscv_vse16_v_f16m2(Q16 + d0, __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq + d0, vl), vl), vl);
    }

    for (int64_t ic = 0; ic < nek1; ++ic) {
        const float mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }

        const _Float16 * k_data = (const _Float16 *) (k_data_row + ic * nbk1);

        float s = rvv_dot_f16_f32(k_data, Q16, DK);

        s = s * scale;  // scale KQ value

        if (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        const float Mold = M;

        float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        if (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);
        } else {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        // V = V*expf(Mold - M) + vs*v
        rvv_scale_mad_f16(VKQ16, ms, vs, (const _Float16 *) (v_data_row + ic * nbv1), DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    flash_attn_ext_finalize_row_f16(dst, VKQ16, DV, S, M, sinks);
}

}  // namespace

void memcpy1d(void * dst, const void * src, int64_t size) {
    size_t byte_size_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    if (vlen == 256) {
        // 1024 bytes
        __asm__ volatile(
            //
            "srli           t0, %[size], 10             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v8, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v16, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v24, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v8, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v16, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v24, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 1023           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1");
    } else if (vlen == 1024) {
        // 2048 bytes
        __asm__ volatile(
            //
            "srli           t0, %[size], 11             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "addi           t2, %[s], 1024              \n\t"
            "addi           t3, %[d], 1024              \n\t"
            "li             t5, 2048                    \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t5              \n\t"
            "vle8.v         v8, (t2)                    \n\t"
            "add            t2, t2, t5                  \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t5              \n\t"
            "vse8.v         v8, (t3)                    \n\t"
            "add            t3, t3, t5                  \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 2047           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m2, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } else {
        __asm__ volatile(
            //
            "add            t1, %[size], zero           \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            : [s] "+r"(src), [d] "+r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

void memcpy2d(void * dst, int64_t dst_stride, const void * src, int64_t src_stride, int64_t tile_rows, int64_t size) {
    for (int64_t i = 0; i < tile_rows; ++i) {
        memcpy1d((char *) dst + i * dst_stride, (const char *) src + i * src_stride, size);
    }
}

void forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                        ggml_tensor *             dst,
                                                        int                       ir0,
                                                        int                       ir1) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    const int64_t rk2 = neq2 / nek2;
    const int64_t rk3 = neq3 / nek3;

    const int64_t rv2 = neq2 / nev2;
    const int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    float scale         = *((float *) dst->op_params + 0);
    float max_bias      = *((float *) dst->op_params + 1);
    float logit_softcap = *((float *) dst->op_params + 2);

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int KV_row_size = DK * sizeof(_Float16) + DV * sizeof(_Float16);

    void * tcm_buffer = ctx.shared.data;
    size_t tcm_buffer_size = ctx.shared.size;
    int ith     = ctx.ith;
    int ir_step = 1;
    for (int ir = ir0; ir < ir1; ir += ir_step) {
        // q indices
        const int iq3 = ir / (neq2 * neq1);
        const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        const int iq3_1 = (ir + 1) / (neq2 * neq1);
        const int iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        const int iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        const int iq3_2 = (ir + 2) / (neq2 * neq1);
        const int iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        const int iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        const int iq3_3 = (ir + 3) / (neq2 * neq1);
        const int iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        const int iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        const ggml_fp16_t * mp =
            mask ? (ggml_fp16_t *) ((char *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                                    (iq3 % mask->ne[3]) * mask->nb[3]) :
                   NULL;

        const bool mp_equal_2 = iq1_1 == iq1 && (iq2 % mask->ne[2]) == (iq2_1 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_1 % mask->ne[3]);

        const bool mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % mask->ne[2]) == (iq2_2 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_2 % mask->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % mask->ne[2]) == (iq2_3 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_3 % mask->ne[3]);

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        const int ik3_1 = iq3_1 / rk3;
        const int ik2_1 = iq2_1 / rk2;

        const int ik3_2 = iq3_2 / rk3;
        const int ik2_2 = iq2_2 / rk2;

        const int ik3_3 = iq3_3 / rk3;
        const int ik2_3 = iq2_3 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const int iv3_1 = iq3_1 / rv3;
        const int iv2_1 = iq2_1 / rv2;

        const int iv3_2 = iq3_2 / rv3;
        const int iv2_2 = iq2_2 / rv2;

        const int iv3_3 = iq3_3 / rv3;
        const int iv2_3 = iq2_3 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<float *, 4> pq_buffer;
        std::array<float *, 4> sinks_buffer;
        std::array<float *, 4> dst_buffer;

        if (tcm_buffer != nullptr && 4 * KV_row_size < tcm_buffer_size && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 4;

            pq_buffer[0] = (float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (float *) ((char *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_buffer[2] = (float *) ((char *) q->data + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_buffer[3] = (float *) ((char *) q->data + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_buffer[0] = sinks ? ((float *) ((char *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((float *) ((char *) sinks->data)) + iq2_1 : nullptr;
            sinks_buffer[2] = sinks ? ((float *) ((char *) sinks->data)) + iq2_2 : nullptr;
            sinks_buffer[3] = sinks ? ((float *) ((char *) sinks->data)) + iq2_3 : nullptr;

            dst_buffer[0] = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (float *) ((char *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_buffer[2] = (float *) ((char *) dst->data + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_buffer[3] = (float *) ((char *) dst->data + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_buffer.data(),                                      //
                (const char *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (const char *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } else if (tcm_buffer != nullptr && 2 * KV_row_size < tcm_buffer_size && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 2;

            pq_buffer[0] = (float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (float *) ((char *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_buffer[0] = sinks ? ((float *) ((char *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((float *) ((char *) sinks->data)) + iq2_1 : nullptr;

            dst_buffer[0] = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (float *) ((char *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_buffer.data(),                                      //
                (const char *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (const char *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } else {
            ir_step = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (const char *) k->data + (ik2 * nbk2 + ik3 * nbk3),                          //
                (const char *) v->data + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((float *) ((char *) sinks->data)) + h : nullptr,                    //
                (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

// Split-KV decode (flash-decoding) for 1..FA_SPLIT_KV_MAX_Q query tokens (single-token decode and
// speculative-decoding verification): one work unit is (KV head, KV chunk) and computes all rk2 query
// heads that share that KV head, token by token (each token has its own mask row), so every K/V row is
// read once per token instead of once per query row. Partials {M, S, VKQ[DV]} go to a shared buffer laid
// out [token][q head][chunk]; a second pass (after a grid barrier) merges the chunks of each row.
int flash_attn_ext_f16_split_kv_chunks_vlen1024_vf16(const ggml_tensor * dst, int nth) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const float max_bias = ((const float *) dst->op_params)[1];

    if (q->ne[1] < 1 || q->ne[1] > FA_SPLIT_KV_MAX_Q || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] ||
        q->ne[2] % k->ne[2] != 0 || max_bias != 0.0f || (mask && (mask->ne[2] != 1 || mask->ne[3] != 1))) {
        return 0;
    }
    const int64_t rk2 = q->ne[2] / k->ne[2];
    if (rk2 != 2 && rk2 != 4 && rk2 != 8) {
        return 0;
    }
    const int64_t n_chunks = std::max<int64_t>(1, nth / k->ne[2]);
    // short contexts: the per-row path is already cheap
    if (k->ne[1] < 128 * n_chunks) {
        return 0;
    }
    return (int) n_chunks;
}

void forward_flash_attn_ext_f16_split_kv_partial_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                               ggml_tensor *               dst,
                                                               int                         n_chunks,
                                                               float *                     partials,
                                                               void *                      tcm_buffer,
                                                               size_t                      tcm_buffer_size) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t DK    = k->ne[0];
    const int64_t DV    = v->ne[0];
    const int64_t n_kv  = k->ne[1];
    const int64_t n_kvh = k->ne[2];
    const int64_t nh    = q->ne[2];
    const int64_t nq    = q->ne[1];
    const int64_t rk2   = nh / n_kvh;

    float scale         = ((const float *) dst->op_params)[0];
    float logit_softcap = ((const float *) dst->op_params)[2];
    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const int64_t chunk  = (n_kv + n_chunks - 1) / n_chunks;

    std::array<float *, 16> pq;
    std::array<float *, 16> pr;
    std::array<float *, 16> unused = {};

    // all query tokens of a unit in one pass (one K/V pack) when rk2 * nq is a supported row count
    const int64_t rows   = rk2 * nq;
    const bool    fused  = rows == 2 || rows == 4 || rows == 8 || rows == 16;
    const int64_t n_pass = fused ? 1 : nq;
    const int64_t prows  = fused ? rows : rk2;

    for (int64_t u = ctx.ith; u < n_kvh * n_chunks; u += ctx.nth) {
        const int64_t kvh = u / n_chunks;
        const int64_t c   = u % n_chunks;
        const int64_t c0  = std::min(n_kv, c * chunk);
        const int64_t c1  = std::min(n_kv, c0 + chunk);
        const int64_t h0  = kvh * rk2;
        const char *  kd  = (const char *) k->data + kvh * k->nb[2] + c0 * k->nb[1];
        const char *  vd  = (const char *) v->data + kvh * v->nb[2] + c0 * v->nb[1];

        for (int64_t p = 0; p < n_pass; ++p) {
            for (int64_t r = 0; r < prows; ++r) {
                const int64_t t = fused ? r / rk2 : p;
                const int64_t i = fused ? r % rk2 : r;
                pq[r] = (float *) ((char *) q->data + t * q->nb[1] + (h0 + i) * q->nb[2]);
                pr[r] = partials + ((t * nh + h0 + i) * n_chunks + c) * (2 + DV);
            }
            const int64_t       t0 = fused ? 0 : p;
            const ggml_fp16_t * mp =
                mask ? (const ggml_fp16_t *) ((const char *) mask->data + t0 * mask->nb[1]) + c0 : nullptr;
            const int64_t n_mask_rows = fused ? nq : 1;
            const size_t  mstride     = mask ? mask->nb[1] : 0;

#define FA_SPLIT_MROW(QL)                                                                                          \
    flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<QL>(pq.data(), kd, vd, mp, unused.data(), unused.data(),     \
                                                              scale, logit_softcap, 1.0f, c1 - c0, k->nb[1],          \
                                                              v->nb[1], DV, DK, tcm_buffer, tcm_buffer_size, pr.data(), \
                                                              n_mask_rows, mstride)
            switch (prows) {
                case 2: FA_SPLIT_MROW(2); break;
                case 4: FA_SPLIT_MROW(4); break;
                case 8: FA_SPLIT_MROW(8); break;
                default: FA_SPLIT_MROW(16); break;
            }
#undef FA_SPLIT_MROW
        }
    }
}

void forward_flash_attn_ext_f16_split_kv_reduce_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                              ggml_tensor *               dst,
                                                              int                         n_chunks,
                                                              const float *               partials) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];

    const int64_t DV = v->ne[0];
    const int64_t nh = q->ne[2];
    const int64_t nq = q->ne[1];

    float acc[FA_VLEN1024_MAX_D];

    for (int64_t r = ctx.ith; r < nq * nh; r += ctx.nth) {
        const int64_t t  = r / nh;
        const int64_t h  = r % nh;
        const float * pr = partials + r * n_chunks * (2 + DV);

        float M = -INFINITY;
        for (int c = 0; c < n_chunks; ++c) {
            M = std::max(M, pr[c * (2 + DV)]);
        }

        rvv_zero_f32(acc, DV);
        float S = 0.0f;
        if (M != -INFINITY) {
            for (int c = 0; c < n_chunks; ++c) {
                const float * pc = pr + c * (2 + DV);
                if (pc[0] == -INFINITY) {
                    continue;
                }
                const float w = expf(pc[0] - M);
                S += w * pc[1];
                for (int64_t d0 = 0; d0 < DV;) {
                    const size_t vl = __riscv_vsetvl_e32m4(DV - d0);
                    vfloat32m4_t a  = __riscv_vle32_v_f32m4(acc + d0, vl);
                    a               = __riscv_vfmacc_vf_f32m4(a, w, __riscv_vle32_v_f32m4(pc + 2 + d0, vl), vl);
                    __riscv_vse32_v_f32m4(acc + d0, a, vl);
                    d0 += vl;
                }
            }
        }

        float ms = 1.0f;
        if (sinks) {
            const float s  = ((const float *) sinks->data)[h];
            float       vs = 1.0f;
            if (s > M) {
                ms = M == -INFINITY ? 0.0f : expf(M - s);
            } else {
                vs = expf(s - M);
            }
            S = S * ms + vs;
        }
        const float f = S == 0.0f ? 0.0f : ms / S;

        float * out = (float *) ((char *) dst->data + (t * dst->ne[1] + h) * dst->nb[1]);
        for (int64_t d0 = 0; d0 < DV;) {
            const size_t vl = __riscv_vsetvl_e32m4(DV - d0);
            __riscv_vse32_v_f32m4(out + d0, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(acc + d0, vl), f, vl), vl);
            d0 += vl;
        }
    }
}

void forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ggml::spacemit::context & ctx,
                                                    ggml_tensor *             dst,
                                                    int                       ir0,
                                                    int                       ir1) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    const ggml_type kv_type = k->type;

    // broadcast factors
    const int64_t rk2 = neq2 / nek2;
    const int64_t rk3 = neq3 / nek3;

    const int64_t rv2 = neq2 / nev2;
    const int64_t rv3 = neq3 / nev3;

    float * param_list    = (float *) dst->op_params;
    float   scale         = param_list[0];
    float   max_bias      = param_list[1];
    float   logit_softcap = param_list[2];

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    int ith = ctx.ith;

    static constexpr int Q_TILE_SZ  = ggml_fa_tile_config::Q;
    static constexpr int KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    float * base = (float *) ctx.workspace +
                   ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV +
                          KV_TILE_SZ * DK + ggml::spacemit::cache_line_size_f32);
    const size_t base_size =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            sizeof(float) +
        ggml::spacemit::cache_line_size_f32 * sizeof(float);

    if (base_size <= ctx.shared.size && ctx.shared.data != nullptr) {
        base = (float *) ctx.shared.data;
    }

    float   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    float * S = S_M_Buf;
    float * M = S_M_Buf + Q_TILE_SZ;

    int ir = ir0;
    while (ir < ir1) {
        // q indices for the start of this tile
        const int iq3 = ir / (neq2 * neq1);
        const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        const int tile_rows = MIN(Q_TILE_SZ, MIN((int) (ir1 - ir), (int) (neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        for (int i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        float *    Q_f32  = base;
        float *    KQ     = (float *) ((char *) base + Q_TILE_SZ * DK * sizeof(float));
        float *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        float *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        float *    V32    = VKQ32 + Q_TILE_SZ * DV;
        float *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        if (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * sizeof(_Float16), (uint8_t *) pq, nbq1, tile_rows, DK,
                                       scale);
        } else {
            memcpy2d(Q_f32, DK * sizeof(float), pq, nbq1, tile_rows, DK * sizeof(float));
        }

        for (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            const int kv_tile = (int) std::min((int64_t) KV_TILE_SZ, nek1 - ic);
            int       active_kv[Q_TILE_SZ];
            int       max_active_kv = kv_tile;

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            if (mask) {
                const ggml_fp16_t * mp_row =
                    (const ggml_fp16_t *) ((const char *) mask->data + iq1 * mask->nb[1] +
                                           (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * sizeof(float), mp_row + ic, mask->nb[1], tile_rows,
                                           kv_tile, slope);

                max_active_kv = 0;
                for (int tq = 0; tq < tile_rows; tq++) {
                    int tk = kv_tile;
                    while (tk > 0 && mask32[tq * KV_TILE_SZ + tk - 1] == -INFINITY) {
                        --tk;
                    }
                    active_kv[tq] = tk;
                    max_active_kv = std::max(max_active_kv, tk);
                }

                if (max_active_kv == 0) {
                    continue;
                }
            } else {
                std::fill_n(active_kv, tile_rows, kv_tile);
            }

            if (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_to_nm((int8_t *) K_f16, KV_TILE_SZ * sizeof(_Float16),
                                            (int8_t *) k->data + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1,
                                            max_active_kv,
                                            DK);

                int tq = 0;
                for (; tq + 3 < tile_rows; tq += 4) {
                    const int group_active = std::max(std::max(active_kv[tq + 0], active_kv[tq + 1]),
                                                      std::max(active_kv[tq + 2], active_kv[tq + 3]));
                    if (group_active > 0) {
                        rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                               KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                               Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                               Q_f16 + (tq + 3) * DK, K_f16, DK, group_active);
                    }
                }
                for (; tq < tile_rows; ++tq) {
                    if (active_kv[tq] > 0) {
                        rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, active_kv[tq]);
                    }
                }
            } else {
                for (int tk = 0; tk < max_active_kv; tk++) {
                    const char *  k_data = (const char *) k->data + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    float *       k_col  = K_f32 + tk;
                    const float * k_src  = (const float *) k_data;
                    for (int64_t dk = 0; dk < DK; ++dk) {
                        k_col[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                for (int tq = 0; tq < tile_rows; ++tq) {
                    if (active_kv[tq] > 0) {
                        rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, active_kv[tq],
                                       scale);
                    }
                }
            }

            bool skip[Q_TILE_SZ] = {};

            for (int tq = 0; tq < tile_rows; tq++) {
                float * kq_row = KQ + tq * KV_TILE_SZ;
                const int row_active = active_kv[tq];

                if (row_active == 0) {
                    skip[tq] = true;
                    continue;
                }

                if (logit_softcap != 0.0f) {
                    rvv_softcap_tanh_inplace_f32(kq_row, KV_TILE_SZ, 1, row_active, logit_softcap);
                }
                if (mask) {
                    rvv_add_inplace_f32(kq_row, KV_TILE_SZ, mask32 + tq * KV_TILE_SZ, KV_TILE_SZ, 1, row_active);
                }

                const float tile_max = rvv_max_f32(kq_row, row_active);

                if (tile_max == -INFINITY) {
                    skip[tq] = true;
                    continue;
                }

                const float Mold = M[tq];
                const float Mnew = fmaxf(Mold, tile_max);

                if (Mnew > Mold) {
                    const float ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_row, row_active, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            if (kv_type == GGML_TYPE_F16) {
                const char * v_data = (const char *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * sizeof(_Float16), v_data, nbv1, max_active_kv, DV * sizeof(_Float16));

                int tq = 0;
                for (; tq + 3 < tile_rows; tq += 4) {
                    if (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        for (int i = 0; i < 4; ++i) {
                            if (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         active_kv[tq + i], DV);
                            }
                        }
                        continue;
                    }

                    const int group_active = std::max(std::max(active_kv[tq + 0], active_kv[tq + 1]),
                                                      std::max(active_kv[tq + 2], active_kv[tq + 3]));
                    for (int i = 0; i < 4; ++i) {
                        std::fill(KQ + (tq + i) * KV_TILE_SZ + active_kv[tq + i],
                                  KQ + (tq + i) * KV_TILE_SZ + group_active, 0.0f);
                    }
                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, group_active, DV);
                }
                for (; tq < tile_rows; ++tq) {
                    if (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, active_kv[tq], DV);
                    }
                }
            } else {
                const char * v_data = (const char *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * sizeof(float), v_data, nbv1, max_active_kv, DV * sizeof(float));

                for (int tq = 0; tq < tile_rows; ++tq) {
                    if (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, active_kv[tq], DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        if (sinks) {
            const float s = ((float *) ((char *) sinks->data))[h];

            for (int tq = 0; tq < tile_rows; tq++) {
                float ms = 1.0f;
                float vs = 1.0f;

                if (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } else {
                    vs = expf(s - M[tq]);
                }

                float S_temp = S[tq] * ms + vs;
                S[tq]        = S_temp == 0.0f ? 0.0f : 1.0f / S_temp;
            }
        } else {
            for (int tq = 0; tq < tile_rows; tq++) {
                const float S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        float * dst_ptr = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * sizeof(float), tile_rows, DV, S);

        ir += tile_rows;
    }
}

void forward_rms_norm_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    int ith = ctx.ith;
    int nth = ctx.nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float epsilon = *((float *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * input  = (char *) src0->data;
    auto * output = (char *) dst->data;

    const auto hidden_size     = ne00;
    const auto task_count      = ne01 * ne02 * ne03;
    const auto task_per_thread = (task_count + nth - 1) / nth;

    const auto task_begin = ith * task_per_thread;
    const auto task_end   = std::min((ith + 1) * task_per_thread, task_count);

    for (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_input       = (float *) (input + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (float *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        while (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);
            sum_sq                = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);
            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_input += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);

        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;

        mean_square = sqrt(mean_square + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        while (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
            src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_data, gvl);
            p_temp_output += gvl;
            p_output += gvl;
            length -= gvl;
        }
    }
}

template <size_t MB_ROWS>
void quantize_a_nrow_i8_ref(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS);

        for (size_t row = 0; row < MB_ROWS; row++) {
            float max_abs_a = 0.0f;
            for (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * count_k + k + bk]));
            }

            float rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            int16_t a_sum = 0;
            for (size_t bk = 0; bk < blk_len; bk++) {
                const int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[row * count_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[row * blk_len + bk] = quantized;
                a_sum += quantized;
            }
            a_sum_ptr[row] = -a_sum;
        }
    }
}

template <size_t MB_ROWS>
void quantize_a_nrow_i8_hp_ref(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    const size_t     subblk_count = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    float   scale_temp[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_size(blk_len, true, true);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_subblk_stride     = q8_hp_blk_size(k_subblk_len, false, false) * MB_ROWS;

    for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);

        float scale_avg = 0.0f;
        for (size_t kk = 0; kk < subblk_count; kk++) {
            float max_abs_a = 0.0f;
            for (size_t row = 0; row < MB_ROWS; row++) {
                for (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * count_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_temp[kk];
        }

        scale_avg /= subblk_count;
        float scale_factor = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - sizeof(_Float16) * MB_ROWS);
        scale_avg_ptr[0] = scale_avg;

        for (size_t kk = 0; kk < subblk_count; kk++) {
            uint8_t *  a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16) * MB_ROWS);

            scale_a_ptr[0] = static_cast<_Float16>(scale_temp[kk] * scale_factor);

            const float rep_scale_a = 1.0f / scale_temp[kk];

            for (size_t row = 0; row < MB_ROWS; row++) {
                int16_t a_sum = 0;
                for (size_t bk = 0; bk < k_subblk_len; bk++) {
                    const int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * count_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[row * k_subblk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * subblk_count + kk] = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

template <size_t MB_ROWS>
void quantize_a_nrow_i8k_ref(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_sum_size          = 256 / 16;

    for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * MB_ROWS + sizeof(int16_t) * a_sum_size * MB_ROWS);

        for (size_t row = 0; row < MB_ROWS; row++) {
            float max_a     = 0.0f;
            float max_abs_a = 0.0f;
            for (size_t bk = 0; bk < blk_len; bk++) {
                float ax = std::abs(a_ptr[row * count_k + k + bk]);
                if (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[row * count_k + k + bk];
                }
            }

            if (!max_abs_a) {
                scale_a_ptr[row] = 0;
                for (size_t bki = 0; bki < a_sum_size; bki++) {
                    for (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[row * blk_len + bk] = 0;
                    }
                    a_sum_ptr[row * a_sum_size + bki] = 0;
                }
                continue;
            }

            float rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            for (size_t bki = 0; bki < a_sum_size; bki++) {
                int16_t a_sum = 0;
                for (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    const int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * count_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[row * blk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * a_sum_size + bki] = -a_sum;
            }
        }
    }
}

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_size(blk_len, true);
    size_t  vlenb        = __riscv_vlenb();

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) + sizeof(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            float scale_a     = max_abs_a / ((1 << 7) - 1);
            float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
            int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
            a_sum_ptr[0]       = -a_sum;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) + sizeof(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            float scale_a     = max_abs_a / ((1 << 7) - 1);
            float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
            int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
            a_sum_ptr[0]       = -a_sum;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } else {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_size(blk_len, true);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * 4 + sizeof(int16_t) * 4);

            for (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * count_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                float scale_a     = max_abs_a / ((1 << 7) - 1);
                float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[mi]      = -a_sum;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * 4 + sizeof(int16_t) * 4);

            for (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * count_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                float scale_a     = max_abs_a / ((1 << 7) - 1);
                float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[mi]      = -a_sum;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_count             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, true, true);
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, false, false);
    size_t           vlenb                    = __riscv_vlenb();
    float            scale_temp[subblk_count] = { 0.0f };

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - sizeof(_Float16));
            float      scale_avg     = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16));
                const float * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                float        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[kk]      = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - sizeof(_Float16));
            float      scale_avg     = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16));
                const float * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                float        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[kk]      = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_4row_i8_hp(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_count             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, true, true);
    int64_t          a_nrow_block_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, false, false) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    float            scale_temp[subblk_count] = { 0.0f };

    if (vlenb == 128) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - sizeof(_Float16) * 4);
            float scale_avg = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr0 = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1 = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2 = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3 = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl       = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0     = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1     = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2     = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3     = __riscv_vle32_v_f32m1(a_src_ptr3, vl);
                vfloat32m1_t v_a0_abs = __riscv_vfabs_v_f32m1(v_a0, vl);
                vfloat32m1_t v_a1_abs = __riscv_vfabs_v_f32m1(v_a1, vl);
                vfloat32m1_t v_a2_abs = __riscv_vfabs_v_f32m1(v_a2, vl);
                vfloat32m1_t v_a3_abs = __riscv_vfabs_v_f32m1(v_a3, vl);

                vfloat32m1_t v_max_abs = __riscv_vfmax_vv_f32m1(v_a0_abs, v_a1_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m1(v_max_abs, v_a2_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m1(v_max_abs, v_a3_abs, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16) * 4);
                const float * a_src_ptr0    = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1    = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2    = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3    = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                float rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m1_t v_a0_scale    = __riscv_vfmul_vf_f32m1(v_a0, rep_scale_a, vl);
                vfloat32m1_t v_a1_scale    = __riscv_vfmul_vf_f32m1(v_a1, rep_scale_a, vl);
                vfloat32m1_t v_a2_scale    = __riscv_vfmul_vf_f32m1(v_a2, rep_scale_a, vl);
                vfloat32m1_t v_a3_scale    = __riscv_vfmul_vf_f32m1(v_a3, rep_scale_a, vl);
                vint16mf2_t  v_a0_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a0_scale, vl);
                vint16mf2_t  v_a1_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a1_scale, vl);
                vint16mf2_t  v_a2_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a2_scale, vl);
                vint16mf2_t  v_a3_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a3_scale, vl);
                vint8mf4_t   v_a0_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a0_quant, vl);
                vint8mf4_t   v_a1_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a1_quant, vl);
                vint8mf4_t   v_a2_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a2_quant, vl);
                vint8mf4_t   v_a3_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a3_quant, vl);

                vint16m1_t tmp_sum0 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum1 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum3 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a0_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_sum_ptr[0 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_count);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - sizeof(_Float16) * 4);
            float scale_avg = 0.0f;

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                const float * a_src_ptr0 = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1 = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2 = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3 = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                vfloat32m4_t v_a0_abs = __riscv_vfabs_v_f32m4(v_a0, vl);
                vfloat32m4_t v_a1_abs = __riscv_vfabs_v_f32m4(v_a1, vl);
                vfloat32m4_t v_a2_abs = __riscv_vfabs_v_f32m4(v_a2, vl);
                vfloat32m4_t v_a3_abs = __riscv_vfabs_v_f32m4(v_a3, vl);

                vfloat32m4_t v_max_abs = __riscv_vfmax_vv_f32m4(v_a0_abs, v_a1_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m4(v_max_abs, v_a2_abs, vl);
                v_max_abs              = __riscv_vfmax_vv_f32m4(v_max_abs, v_a3_abs, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_max_abs, tmp, vl);
                float        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_count;
            const float scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            for (size_t kk = 0; kk < subblk_count; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + sizeof(_Float16) * 4);
                const float * a_src_ptr0    = a_ptr + 0 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr1    = a_ptr + 1 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr2    = a_ptr + 2 * count_k + k + kk * k_subblk_len;
                const float * a_src_ptr3    = a_ptr + 3 * count_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                float rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_temp[kk] * scale_factor);

                vfloat32m4_t v_a0_scale    = __riscv_vfmul_vf_f32m4(v_a0, rep_scale_a, vl);
                vfloat32m4_t v_a1_scale    = __riscv_vfmul_vf_f32m4(v_a1, rep_scale_a, vl);
                vfloat32m4_t v_a2_scale    = __riscv_vfmul_vf_f32m4(v_a2, rep_scale_a, vl);
                vfloat32m4_t v_a3_scale    = __riscv_vfmul_vf_f32m4(v_a3, rep_scale_a, vl);
                vint16m2_t   v_a0_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a0_scale, vl);
                vint16m2_t   v_a1_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a1_scale, vl);
                vint16m2_t   v_a2_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a2_scale, vl);
                vint16m2_t   v_a3_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a3_scale, vl);
                vint8m1_t    v_a0_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a0_quant, vl);
                vint8m1_t    v_a1_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a1_quant, vl);
                vint8m1_t    v_a2_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a2_quant, vl);
                vint8m1_t    v_a3_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a3_quant, vl);

                vint16m1_t tmp_sum0 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum1 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum2 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t tmp_sum3 = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a0_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_sum = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_sum_ptr[0 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_count + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_size(256);
    constexpr int64_t a_sum_size   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    if (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) + sizeof(int16_t) * a_sum_size);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            float        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            float scale_a     = max_abs_a / ((1 << 7) - 1);
            float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            for (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[bki]     = -a_sum;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } else if (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) + sizeof(int16_t) * a_sum_size);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            for (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            float        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            float scale_a     = max_abs_a / ((1 << 7) - 1);
            float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            for (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_sum, vl);
                int16_t    a_sum   = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                a_sum_ptr[bki]     = -a_sum;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } else {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void quantize_a_4row_i8k(size_t blk_len, const float * a_ptr, size_t count_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_size(256);
    constexpr int64_t a_nrow_block_stride = a_blk_stride * 4;
    constexpr int64_t a_sum_size          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    if (vlenb == 128) {
        // vlen = 1024 bits
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * 4 + sizeof(int16_t) * a_sum_size * 4);

            for (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                for (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * count_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                float        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                float scale_a     = max_abs_a / ((1 << 7) - 1);
                float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                for (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * count_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_sum               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_sum               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
                    int16_t    a_sum                 = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                    a_sum_ptr[mi * a_sum_size + bki] = -a_sum;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } else if (vlenb == 32) {
        // vlen = 256 bits
        for (size_t k = 0; k < count_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            float *   scale_a_ptr = reinterpret_cast<float *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + sizeof(float) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + sizeof(float) * 4 + sizeof(int16_t) * a_sum_size * 4);

            for (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                for (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * count_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                float        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                float scale_a     = max_abs_a / ((1 << 7) - 1);
                float rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                for (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * count_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_sum               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_sum               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_sum, vl);
                    int16_t    a_sum                 = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
                    a_sum_ptr[mi * a_sum_size + bki] = -a_sum;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } else {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, count_k, quant_a_ptr);
    }
}

void forward_cpy_with_permute(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    const int           ith  = ctx.ith;
    const int           nth  = ctx.nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = src0->ne[2] * src0->ne[3];
    int64_t m     = src0->ne[1];
    int64_t n     = src0->ne[0];

    int64_t batch_stride = src0->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = n_src_stride * m;

    permute_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

void forward_cont_with_permute(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    const int           ith  = ctx.ith;
    const int           nth  = ctx.nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = dst->ne[2] * dst->ne[3];
    int64_t n     = dst->ne[1];
    int64_t m     = dst->ne[0];

    int64_t batch_stride = dst->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = dst->nb[1];

    permute_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

void forward_norm_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    int ith = ctx.ith;
    int nth = ctx.nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float epsilon = *((float *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * input  = (char *) src0->data;
    auto * output = (char *) dst->data;

    const auto hidden_size     = ne00;
    const auto task_count      = ne01 * ne02 * ne03;
    const auto task_per_thread = (task_count + nth - 1) / nth;

    const auto task_begin = ith * task_per_thread;
    const auto task_end   = std::min((ith + 1) * task_per_thread, task_count);

    for (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_input       = (float *) (input + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (float *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        while (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            // load data
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_input, gvl);

            sum    = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
            sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_input += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        float        mean   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t mean_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum, 0), __riscv_vget_v_f32m4_f32m1(sum, 1), gvl);
        mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 2), gvl);
        mean_v = __riscv_vfadd_vv_f32m1(mean_v, __riscv_vget_v_f32m4_f32m1(sum, 3), gvl);
        mean_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_v, zero_v, gvl);
        mean   = __riscv_vfmv_f_s_f32m1_f32(mean_v);
        mean /= hidden_size;

        vfloat32m1_t mean_square_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(sum_sq, 0), __riscv_vget_v_f32m4_f32m1(sum_sq, 1), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 2), gvl);
        mean_square_v = __riscv_vfadd_vv_f32m1(mean_square_v, __riscv_vget_v_f32m4_f32m1(sum_sq, 3), gvl);
        mean_square_v = __riscv_vfredusum_vs_f32m1_f32m1(mean_square_v, zero_v, gvl);

        float mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;
        mean_square = sqrt(mean_square - mean * mean + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        while (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_temp_output, gvl);
            src_data              = __riscv_vfsub_vf_f32m4(src_data, mean, gvl);
            src_data              = __riscv_vfmul_vf_f32m4(src_data, mean_square, gvl);
            __riscv_vse32_v_f32m4(p_output, src_data, gvl);
            p_temp_output += gvl;
            p_output += gvl;
            length -= gvl;
        }
    }
}

template <ggml_op op_type, typename T> void forward_binary(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    auto src0_rows = ggml_nrows(src0);
    auto src1_rows = ggml_nrows(src1);

    int ith = ctx.ith;
    int nth = ctx.nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(T));
    GGML_ASSERT(nb00 == sizeof(T));

    const int64_t nr  = ggml_nrows(src0);
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    auto compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        if constexpr (op_type == GGML_OP_ADD) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if constexpr (op_type == GGML_OP_SUB) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if constexpr (op_type == GGML_OP_MUL) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if constexpr (op_type == GGML_OP_DIV) {
            if constexpr (std::is_same_v<T, float>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } else if constexpr (std::is_same_v<T, _Float16>) {
                for (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else {
            GGML_ABORT("fatal error");
        }
    };

    if (src0_rows == src1_rows && src0_rows == 1 && ne00 == ne10) {
        int64_t task_per_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_per_thread;
        int64_t task_end        = std::min((ith + 1) * task_per_thread, ne00);

        T * dst_ptr  = ((T *) dst->data) + task_begin;
        T * src0_ptr = ((T *) src0->data) + task_begin;
        T * src1_ptr = ((T *) src1->data) + task_begin;

        compute_func_vv(task_end - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } else if (ne10 > 1) {
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            for (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } else {
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((char *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((char *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((char *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            for (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                if constexpr (op_type == GGML_OP_ADD) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else if constexpr (op_type == GGML_OP_SUB) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else if constexpr (op_type == GGML_OP_MUL) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else if constexpr (op_type == GGML_OP_DIV) {
                    if constexpr (std::is_same_v<T, float>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } else if constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> void forward_sum_rows(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    const int ith = ctx.ith;
    const int nth = ctx.nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_per_thread = (n_task + nth - 1) / nth;
    int64_t ir_start        = ith * task_per_thread;
    int64_t ir_end          = std::min(ir_start + task_per_thread, n_task);

    for (int64_t ir = ir_start; ir < ir_end; ir++) {
        const int64_t i3 = ir / (ne02 * ne01);
        const int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        const int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_row = (T *) ((char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_row = (T *) ((char *) op->data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        float row_sum = 0;

        if constexpr (std::is_same_v<T, float>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       length  = ne00;
            const float * p_data  = src_row;

            while (length > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(length);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_data, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_data += vl;
                length -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t sum_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            sum_v               = __riscv_vfredusum_vs_f32m1_f32m1(sum_v, zero_v, gvl);
            row_sum             = __riscv_vfmv_f_s_f32m1_f32(sum_v);
        } else if constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          length  = ne00;
            const _Float16 * p_data  = src_row;

            while (length > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(length);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_data, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_data += vl;
                length -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t sum_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            sum_v               = __riscv_vfadd_vv_f32m1(sum_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            sum_v               = __riscv_vfredusum_vs_f32m1_f32m1(sum_v, zero_v, gvl);
            row_sum             = __riscv_vfmv_f_s_f32m1_f32(sum_v);
        } else {
            GGML_ABORT("fatal error");
        }

        dst_row[0] = row_sum;
    }
}

template <typename T> void forward_repeat_nrows(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    const int ith = ctx.ith;
    const int nth = ctx.nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_per_thread = (nrows + nth - 1) / nth;
    int64_t ir_start         = ith * nrows_per_thread;
    int64_t ir_end           = std::min(ir_start + nrows_per_thread, nrows);

    if (src0->ne[0] == 1) {
        for (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((char *) src0->data + ir * src0->nb[1]);
            T * dst_row = (T *) ((char *) dst->data + ir * dst->nb[1]);

            T src_scalar = src_row[0];

            int64_t length = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            while (length > 0) {
                if constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } else if constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } else {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                length -= vl;
            }
        }
    } else if (src0->ne[0] == dst->ne[0]) {
        for (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((char *) src0->data + ir * src0->nb[1]);
            T * dst_row = (T *) ((char *) dst->data + ir * dst->nb[1]);

            int64_t length = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            while (length > 0) {
                if constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_row + idx, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } else if constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_row + idx), vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } else {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                length -= vl;
            }
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

template <typename T> void forward_repeat_dim1(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    const int ith = ctx.ith;
    const int nth = ctx.nth;

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const int64_t total_batches      = ne2 * ne3;
    const int64_t batches_per_thread = (total_batches + nth - 1) / nth;
    const int64_t batch_start        = ith * batches_per_thread;
    const int64_t batch_end          = std::min(batch_start + batches_per_thread, total_batches);

    for (int64_t b = batch_start; b < batch_end; b++) {
        const int64_t i3 = b / ne2;
        const int64_t i2 = b % ne2;

        T * src_base  = (T *) ((char *) src0->data + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3]);

        for (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((char *) dst_batch + i1 * dst->nb[1]);
            int64_t length  = ne0;
            int64_t idx     = 0;

            while (length > 0) {
                if constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_base + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    length -= vl;
                } else if constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_base + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    length -= vl;
                } else {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> void forward_get_rows(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    assert(ne0 == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(T));
    assert(ggml_nrows(op) == nr);

    const int ith = ctx.ith;
    const int nth = ctx.nth;

    int rows_nth = nth;
    int cols_nth = 1;

    if (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    // rows per thread
    const int dr = (nr + rows_nth - 1) / rows_nth;
    const int dc = (nc + cols_nth - 1) / cols_nth;

    int rows_ith = ith % rows_nth;
    int cols_ith = ith % cols_nth;

    // row range for this thread
    const int ir0 = dr * rows_ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const int cr0 = dc * cols_ith;
    const int cr1 = MIN(cr0 + dc, nc);

    if (ir0 >= ir1 || cr0 >= cr1) {
        return;
    }

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((char *) dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * sizeof(T),
                 ((char *) src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * sizeof(T),
                 (cr1 - cr0) * sizeof(T));
    }
}

template <typename T> void forward_concat(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(T));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == sizeof(T) && nb1 == sizeof(T) * (ne00 + ne10));

    const int64_t nr = ggml_nrows(dst);
    const int64_t nc = ne0;

    const int ith = ctx.ith;
    const int nth = ctx.nth;

    int rows_nth = nth;
    int cols_nth = 1;

    if (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    const int dr = (nr + rows_nth - 1) / rows_nth;
    const int dc = (nc + cols_nth - 1) / cols_nth;

    int rows_ith = ith % rows_nth;
    int cols_ith = ith % cols_nth;

    // row range for this thread
    const int ir0 = dr * rows_ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const int cr0 = dc * cols_ith;
    const int cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    const T * x;

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i3 = i / (ne02 * ne01);
        const int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        const int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        for (int i0 = cr0; i0 < cr1; i0++) {
            if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (const T *) ((const char *) src0->data + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } else {
                x = (const T *) ((const char *) src1->data + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            T * y = (T *) ((char *) dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

// Fused single-pass row update: decays the state row, derives the delta-rule
// correction from the decayed row, applies it and writes the updated row with
// one load/store pair. Each state row is independent within a token, so the
// separate decay-store / update-reload passes of the reference implementation
// collapse into this. `src` and `dst` may only alias when they are the same
// row (the row is loaded before it is stored). Returns the attention output.
static inline float rvv_gdn_fused_row_f32(float *            dst,
                                          const float *      src,
                                          const vfloat32m8_t kv,
                                          const vfloat32m8_t qv,
                                          const vfloat32m8_t dv,
                                          const float        v_j,
                                          const float        beta,
                                          const float        kq,
                                          const float        scale,
                                          const size_t       vl) {
    vfloat32m8_t s      = __riscv_vle32_v_f32m8(src, vl);
    s                   = __riscv_vfmul_vv_f32m8(s, dv, vl);
    vfloat32m1_t sum_k  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k               = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, kv, vl), sum_k, vl);
    sum_q               = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, qv, vl), sum_q, vl);
    const float state_k = __riscv_vfmv_f_s_f32m1_f32(sum_k);
    const float state_q = __riscv_vfmv_f_s_f32m1_f32(sum_q);
    const float delta_j = (v_j - state_k) * beta;
    s                   = __riscv_vfmacc_vf_f32m8(s, delta_j, kv, vl);
    __riscv_vse32_v_f32m8(dst, s, vl);
    return (state_q + delta_j * kq) * scale;
}

// A head that fits in m4 leaves registers for the state and reduction
// temporaries alongside k, q and decay, avoiding whole-vector row-loop spills.
static inline float rvv_gdn_fused_row_f32(float *            dst,
                                          const float *      src,
                                          const vfloat32m4_t kv,
                                          const vfloat32m4_t qv,
                                          const vfloat32m4_t dv,
                                          const float        v_j,
                                          const float        beta,
                                          const float        kq,
                                          const float        scale,
                                          const size_t       vl) {
    vfloat32m4_t s      = __riscv_vle32_v_f32m4(src, vl);
    s                   = __riscv_vfmul_vv_f32m4(s, dv, vl);
    vfloat32m1_t sum_k  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k               = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(s, kv, vl), sum_k, vl);
    sum_q               = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(s, qv, vl), sum_q, vl);
    const float state_k = __riscv_vfmv_f_s_f32m1_f32(sum_k);
    const float state_q = __riscv_vfmv_f_s_f32m1_f32(sum_q);
    const float delta_j = (v_j - state_k) * beta;
    s                   = __riscv_vfmacc_vf_f32m4(s, delta_j, kv, vl);
    __riscv_vse32_v_f32m4(dst, s, vl);
    return (state_q + delta_j * kq) * scale;
}

static inline float rvv_gdn_dot_f32(const float * x, const float * y, int64_t n) {
    const size_t       vl   = __riscv_vsetvl_e32m8(n);
    const vfloat32m8_t xv   = __riscv_vle32_v_f32m8(x, vl);
    const vfloat32m8_t yv   = __riscv_vle32_v_f32m8(y, vl);
    const vfloat32m8_t prod = __riscv_vfmul_vv_f32m8(xv, yv, vl);
    vfloat32m1_t       sum  = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum                     = __riscv_vfredusum_vs_f32m8_f32m1(prod, sum, vl);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

struct rvv_gdn_decay_dots_f32 {
    float state_k;
    float state_q;
};

// Two-pass row update of the reference implementation: decay the state row,
// store it back and reduce the two dot products in the same sweep, then apply
// the delta-rule correction in a second sweep over the rows. The early store
// keeps the loads of subsequent rows pipelined on the in-order cores, which
// makes this form the faster one once the chunk has enough tokens that the
// state stays bandwidth-bound (measured crossover vs the fused form: ~8
// tokens/chunk; prefill chunks are 64, kda chunks 16).
static inline rvv_gdn_decay_dots_f32 rvv_gdn_decay_dots_f32_impl(float *       state,
                                                                 const float * decay,
                                                                 float         decay_scalar,
                                                                 const float * k,
                                                                 const float * q,
                                                                 int64_t       n,
                                                                 bool          kda) {
    const size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t s  = __riscv_vle32_v_f32m8(state, vl);
    if (kda) {
        const vfloat32m8_t d = __riscv_vle32_v_f32m8(decay, vl);
        s                    = __riscv_vfmul_vv_f32m8(s, d, vl);
    } else {
        s = __riscv_vfmul_vf_f32m8(s, decay_scalar, vl);
    }
    __riscv_vse32_v_f32m8(state, s, vl);

    const vfloat32m8_t kv    = __riscv_vle32_v_f32m8(k, vl);
    const vfloat32m8_t qv    = __riscv_vle32_v_f32m8(q, vl);
    vfloat32m1_t       sum_k = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t       sum_q = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k                    = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, kv, vl), sum_k, vl);
    sum_q                    = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, qv, vl), sum_q, vl);
    return { __riscv_vfmv_f_s_f32m1_f32(sum_k), __riscv_vfmv_f_s_f32m1_f32(sum_q) };
}

static inline void rvv_gdn_update_f32(float * state, const float * k, float delta, int64_t n) {
    const size_t       vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t       s  = __riscv_vle32_v_f32m8(state, vl);
    const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k, vl);
    s                     = __riscv_vfmacc_vf_f32m8(s, delta, kv, vl);
    __riscv_vse32_v_f32m8(state, s, vl);
}

// Ported from ggml-cpu/spacemit (linpengt/qwen35-opt); the IME variant (off by default there) is not.
// ---- chunkwise gated delta net (prefill) ----
// Per head, with S the k x v state (the op stores its transpose), chunk-local cumulative log decay
// gam_r = sum_{s<=r} g_s, Gam_r = exp(gam_r) and E_rs = exp(gam_r - gam_s) <= 1 for s <= r:
//   U   = (I + L)^-1 * beta (V - Gam K S0),  L_rs = beta_r E_rs (k_r . k_s), s < r   (forward substitution)
//   O   = scale (Gam Q S0 + (E o QK^T)_lower-incl U)
//   S1  = Gam_C S0 + (E_C. o K)^T U
// which equals the token recurrence S_t = e^{g_t} S_{t-1} + k_t beta_t (v_t - e^{g_t} S_{t-1}^T k_t)^T,
// o_t = scale S_t^T q_t, up to float rounding. All products are 4-row blocked RVV matmuls.
// Default chunk 16: everything runs at the A100's fp32 vector rate, so smaller chunks (less C^2 work) are
// faster (Qwen3.6 pp512 GDN ms/token: chunk 8 1.23, 16 1.29, 32 1.53, 64 1.98, 128 2.74; recurrence 2.56).
// KLD vs the recurrence is 0.004-0.005 for all chunk sizes (rounding-level for this MoE model).
constexpr int64_t GDN_CHUNK_SIZE    = 16;
// 128 bytes is a suitable alignment size for the TCM state buffer.
constexpr size_t  GDN_TCM_ALIGNMENT = 128;

// C[m x n] = (C0 ? c0s * C0 : 0) + A[m x k] B[k x n]. A rows strided by lda; B, C, C0 rows contiguous
// with leading dims ldb, ldc, ldc0; n <= vlmax(e32m4). C may alias C0 (row-wise in place).
static void gdn_gemm(int64_t       m,
                     int64_t       n,
                     int64_t       k,
                     const float * A,
                     int64_t       lda,
                     const float * B,
                     int64_t       ldb,
                     float *       C,
                     int64_t       ldc,
                     const float * C0,
                     int64_t       ldc0,
                     float         c0s) {
    const size_t vl = __riscv_vsetvl_e32m4(n);
    int64_t      r  = 0;
    for (; r + 4 <= m; r += 4) {
        vfloat32m4_t c0, c1, c2, c3;
        if (C0) {
            c0 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 0) * ldc0, vl), c0s, vl);
            c1 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 1) * ldc0, vl), c0s, vl);
            c2 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 2) * ldc0, vl), c0s, vl);
            c3 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 3) * ldc0, vl), c0s, vl);
        } else {
            c0 = c1 = c2 = c3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }
        const float * a0 = A + (r + 0) * lda;
        const float * a1 = A + (r + 1) * lda;
        const float * a2 = A + (r + 2) * lda;
        const float * a3 = A + (r + 3) * lda;
        for (int64_t p = 0; p < k; ++p) {
            const vfloat32m4_t b = __riscv_vle32_v_f32m4(B + p * ldb, vl);
            c0                   = __riscv_vfmacc_vf_f32m4(c0, a0[p], b, vl);
            c1                   = __riscv_vfmacc_vf_f32m4(c1, a1[p], b, vl);
            c2                   = __riscv_vfmacc_vf_f32m4(c2, a2[p], b, vl);
            c3                   = __riscv_vfmacc_vf_f32m4(c3, a3[p], b, vl);
        }
        __riscv_vse32_v_f32m4(C + (r + 0) * ldc, c0, vl);
        __riscv_vse32_v_f32m4(C + (r + 1) * ldc, c1, vl);
        __riscv_vse32_v_f32m4(C + (r + 2) * ldc, c2, vl);
        __riscv_vse32_v_f32m4(C + (r + 3) * ldc, c3, vl);
    }
    for (; r < m; ++r) {
        vfloat32m4_t  c = C0 ? __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + r * ldc0, vl), c0s, vl) :
                               __riscv_vfmv_v_f_f32m4(0.0f, vl);
        const float * a = A + r * lda;
        for (int64_t p = 0; p < k; ++p) {
            c = __riscv_vfmacc_vf_f32m4(c, a[p], __riscv_vle32_v_f32m4(B + p * ldb, vl), vl);
        }
        __riscv_vse32_v_f32m4(C + r * ldc, c, vl);
    }
}

// dst[j * ld_dst + i] = src[i * ld_src + j] for an n x n block (n <= vlmax(e32m4))
static void gdn_transpose(int64_t       n_rows,
                          int64_t       n_cols,
                          const float * src,
                          int64_t       ld_src,
                          float *       dst,
                          int64_t       ld_dst) {
    const size_t vl = __riscv_vsetvl_e32m4(n_cols);
    for (int64_t i = 0; i < n_rows; ++i) {
        __riscv_vsse32_v_f32m4(dst + i, ld_dst * sizeof(float), __riscv_vle32_v_f32m4(src + i * ld_src, vl), vl);
    }
}

struct gdn_chunk_args {
    const ggml_tensor *src_q, *src_k, *src_v, *src_g, *src_beta;
    int64_t            S_v, H, n_tokens;
    float              scale;
};

// One (head, seq): s_in / s_out are the transposed states (S_v x S_v), attn_data the first token's output.
static void gdn_chunked_head(const gdn_chunk_args & a,
                             int64_t                iq1,
                             int64_t                ik1,
                             int64_t                iv1,
                             int64_t                iq3,
                             int64_t                ik3,
                             int64_t                iv3,
                             const float *          s_in,
                             float *                s_out,
                             float *                attn_data) {
    const int64_t D  = a.S_v;
    const int64_t CS = GDN_CHUNK_SIZE;

    static thread_local std::vector<float> buf;
    // State, seven token tiles, three chunk matrices, and four per-token vectors.
    // All accesses use their logical lengths; no extra tail padding is needed.
    const size_t                           need = D * D + 7 * CS * D + 3 * CS * CS + 4 * CS;
    if (buf.size() < need) {
        buf.resize(need);
    }
    float * S   = buf.data();
    float * Qc  = S + D * D;
    float * Kc  = Qc + CS * D;
    float * U   = Kc + CS * D;   // v rows, then U in place
    float * KT  = U + CS * D;    // D x CS
    float * KTs = KT + CS * D;   // D x CS, columns scaled by exp(gam_C - gam_s)
    float * P   = KTs + CS * D;  // K S0, then reused
    float * QS  = P + CS * D;    // Q S0 (rows scaled by Gam)
    float * KK  = QS + CS * D;   // CS x CS
    float * QK  = KK + CS * CS;
    float * E   = QK + CS * CS;
    float * gam = E + CS * CS;
    float * Gam = gam + CS;
    float * bet = Gam + CS;
    float * wst = bet + CS;

    const size_t vl = __riscv_vsetvl_e32m4(D);

    gdn_transpose(D, D, s_in, D, S, D);  // S[i][j] = s_in[j][i]

    GGML_TENSOR_LOCALS(size_t, nbq, a.src_q, nb);
    GGML_TENSOR_LOCALS(size_t, nbk, a.src_k, nb);
    GGML_TENSOR_LOCALS(size_t, nbv, a.src_v, nb);
    GGML_TENSOR_LOCALS(size_t, nbg, a.src_g, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, a.src_beta, nb);

    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += CS) {
        const int64_t Cn = std::min(CS, a.n_tokens - t0);

        float acc_g = 0.0f;
        for (int64_t r = 0; r < Cn; ++r) {
            const int64_t t = t0 + r;
            memcpy(Qc + r * D, (const char *) a.src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1, D * sizeof(float));
            memcpy(Kc + r * D, (const char *) a.src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1, D * sizeof(float));
            memcpy(U + r * D, (const char *) a.src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1, D * sizeof(float));
            acc_g += *(const float *) ((const char *) a.src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);
            gam[r] = acc_g;
            Gam[r] = expf(acc_g);
            bet[r] = *(const float *) ((const char *) a.src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
        }
        for (int64_t r = 0; r < Cn; ++r) {
            for (int64_t c = 0; c <= r; ++c) {
                E[r * CS + c] = expf(gam[r] - gam[c]);
            }
            wst[r] = expf(gam[Cn - 1] - gam[r]);
        }

        gdn_transpose(Cn, D, Kc, D, KT, CS);                           // KT[i][r] = K[r][i]
        gdn_gemm(Cn, Cn, D, Kc, D, KT, CS, KK, CS, nullptr, 0, 0.0f);  // K K^T
        gdn_gemm(Cn, Cn, D, Qc, D, KT, CS, QK, CS, nullptr, 0, 0.0f);  // Q K^T
        gdn_gemm(Cn, D, D, Kc, D, S, D, P, D, nullptr, 0, 0.0f);       // K S0
        gdn_gemm(Cn, D, D, Qc, D, S, D, QS, D, nullptr, 0, 0.0f);      // Q S0

        // U_r = beta_r (v_r - Gam_r P_r) - sum_{s<r} beta_r E_rs KK_rs U_s
        for (int64_t r = 0; r < Cn; ++r) {
            vfloat32m4_t u = __riscv_vle32_v_f32m4(U + r * D, vl);
            u              = __riscv_vfnmsac_vf_f32m4(u, Gam[r], __riscv_vle32_v_f32m4(P + r * D, vl), vl);
            u              = __riscv_vfmul_vf_f32m4(u, bet[r], vl);
            for (int64_t c = 0; c < r; ++c) {
                const float l = bet[r] * E[r * CS + c] * KK[r * CS + c];
                u             = __riscv_vfnmsac_vf_f32m4(u, l, __riscv_vle32_v_f32m4(U + c * D, vl), vl);
            }
            __riscv_vse32_v_f32m4(U + r * D, u, vl);
        }

        // O = scale (Gam Q S0 + A U), A_rs = E_rs QK_rs for s <= r (zero above the diagonal)
        for (int64_t r = 0; r < Cn; ++r) {
            float * qkr = QK + r * CS;
            for (int64_t c = 0; c <= r; ++c) {
                qkr[c] *= E[r * CS + c];
            }
            for (int64_t c = r + 1; c < Cn; ++c) {
                qkr[c] = 0.0f;
            }
            float * qs = QS + r * D;
            __riscv_vse32_v_f32m4(qs, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(qs, vl), Gam[r], vl), vl);
        }
        gdn_gemm(Cn, D, Cn, QK, CS, U, D, P, D, QS, D, 1.0f);
        for (int64_t r = 0; r < Cn; ++r) {
            __riscv_vse32_v_f32m4(attn_data + (t0 + r) * D * a.H,
                                  __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(P + r * D, vl), a.scale, vl), vl);
        }

        // S = Gam_C S + KTs U
        for (int64_t i = 0; i < D; ++i) {
            for (int64_t c = 0; c < Cn; ++c) {
                KTs[i * CS + c] = KT[i * CS + c] * wst[c];
            }
        }
        gdn_gemm(D, D, Cn, KTs, CS, U, D, S, D, S, D, Gam[Cn - 1]);
    }

    gdn_transpose(D, D, S, D, s_out, D);
}

// Transposed state makes each vector lane an independent output row. Four partial
// sums shorten the dot-product dependency chains without reducing across lanes.
// The four-way unroll requires d % 4 == 0, checked by the dispatcher.
static __attribute__((noinline)) void gdn_recurrent_transposed(float *       state,
                                                               const float * k,
                                                               const float * q,
                                                               float         decay_scalar,
                                                               const float * v,
                                                               float         beta,
                                                               float         kq,
                                                               float         scale,
                                                               float *       out,
                                                               int64_t       d) {
    for (int64_t j = 0; j < d;) {
        const size_t vl = __riscv_vsetvl_e32m2(d - j);
        auto         k0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        auto         k1 = k0, k2 = k0, k3 = k0, q0 = k0, q1 = k0, q2 = k0, q3 = k0;
        for (int64_t i = 0; i < d; i += 4) {
            auto s0 = __riscv_vle32_v_f32m2(state + (i + 0) * d + j, vl);
            auto s1 = __riscv_vle32_v_f32m2(state + (i + 1) * d + j, vl);
            auto s2 = __riscv_vle32_v_f32m2(state + (i + 2) * d + j, vl);
            auto s3 = __riscv_vle32_v_f32m2(state + (i + 3) * d + j, vl);
            s0      = __riscv_vfmul_vf_f32m2(s0, decay_scalar, vl);
            s1      = __riscv_vfmul_vf_f32m2(s1, decay_scalar, vl);
            s2      = __riscv_vfmul_vf_f32m2(s2, decay_scalar, vl);
            s3      = __riscv_vfmul_vf_f32m2(s3, decay_scalar, vl);
            k0      = __riscv_vfadd_vv_f32m2(k0, __riscv_vfmul_vf_f32m2(s0, k[i + 0], vl), vl);
            k1      = __riscv_vfadd_vv_f32m2(k1, __riscv_vfmul_vf_f32m2(s1, k[i + 1], vl), vl);
            k2      = __riscv_vfadd_vv_f32m2(k2, __riscv_vfmul_vf_f32m2(s2, k[i + 2], vl), vl);
            k3      = __riscv_vfadd_vv_f32m2(k3, __riscv_vfmul_vf_f32m2(s3, k[i + 3], vl), vl);
            q0      = __riscv_vfadd_vv_f32m2(q0, __riscv_vfmul_vf_f32m2(s0, q[i + 0], vl), vl);
            q1      = __riscv_vfadd_vv_f32m2(q1, __riscv_vfmul_vf_f32m2(s1, q[i + 1], vl), vl);
            q2      = __riscv_vfadd_vv_f32m2(q2, __riscv_vfmul_vf_f32m2(s2, q[i + 2], vl), vl);
            q3      = __riscv_vfadd_vv_f32m2(q3, __riscv_vfmul_vf_f32m2(s3, q[i + 3], vl), vl);
        }
        const auto sk =
            __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(k0, k1, vl), __riscv_vfadd_vv_f32m2(k2, k3, vl), vl);
        const auto sq =
            __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(q0, q1, vl), __riscv_vfadd_vv_f32m2(q2, q3, vl), vl);
        auto delta = __riscv_vfsub_vv_f32m2(__riscv_vle32_v_f32m2(v + j, vl), sk, vl);
        delta      = __riscv_vfmul_vf_f32m2(delta, beta, vl);
        __riscv_vse32_v_f32m2(out + j, __riscv_vfmul_vf_f32m2(__riscv_vfmacc_vf_f32m2(sq, kq, delta, vl), scale, vl),
                              vl);
        for (int64_t i = 0; i < d; i += 4) {
            auto s0 = __riscv_vle32_v_f32m2(state + (i + 0) * d + j, vl);
            auto s1 = __riscv_vle32_v_f32m2(state + (i + 1) * d + j, vl);
            auto s2 = __riscv_vle32_v_f32m2(state + (i + 2) * d + j, vl);
            auto s3 = __riscv_vle32_v_f32m2(state + (i + 3) * d + j, vl);
            s0      = __riscv_vfmul_vf_f32m2(s0, decay_scalar, vl);
            s1      = __riscv_vfmul_vf_f32m2(s1, decay_scalar, vl);
            s2      = __riscv_vfmul_vf_f32m2(s2, decay_scalar, vl);
            s3      = __riscv_vfmul_vf_f32m2(s3, decay_scalar, vl);
            s0      = __riscv_vfmacc_vf_f32m2(s0, k[i + 0], delta, vl);
            s1      = __riscv_vfmacc_vf_f32m2(s1, k[i + 1], delta, vl);
            s2      = __riscv_vfmacc_vf_f32m2(s2, k[i + 2], delta, vl);
            s3      = __riscv_vfmacc_vf_f32m2(s3, k[i + 3], delta, vl);
            __riscv_vse32_v_f32m2(state + (i + 0) * d + j, s0, vl);
            __riscv_vse32_v_f32m2(state + (i + 1) * d + j, s1, vl);
            __riscv_vse32_v_f32m2(state + (i + 2) * d + j, s2, vl);
            __riscv_vse32_v_f32m2(state + (i + 3) * d + j, s3, vl);
        }
        j += vl;
    }
}

// Decode with a full m4 row: load the next two independent rows before updating this pair.
// Two-way unrolling overlaps row loads with computation; the dispatcher requires even d.
static __attribute__((noinline)) void gdn_recurrent_rows_lookahead(float *       dst,
                                                                   const float * src,
                                                                   const float * k,
                                                                   const float * q,
                                                                   const float * v,
                                                                   float         decay,
                                                                   float         beta,
                                                                   float         kq,
                                                                   float         scale,
                                                                   int64_t       d,
                                                                   float *       out) {
    const size_t vl = __riscv_vsetvl_e32m4(d);
    const auto   kv = __riscv_vle32_v_f32m4(k, vl);
    const auto   qv = __riscv_vle32_v_f32m4(q, vl);
    auto         s0 = __riscv_vle32_v_f32m4(src, vl);
    auto         s1 = __riscv_vle32_v_f32m4(src + d, vl);
    for (int64_t j = 0; j < d; j += 2) {
        auto next0 = s0;
        auto next1 = s1;
        if (j + 2 < d) {
            next0 = __riscv_vle32_v_f32m4(src + (j + 2) * d, vl);
            next1 = __riscv_vle32_v_f32m4(src + (j + 3) * d, vl);
            // Keep both loads before the arithmetic below in generated code.
            asm volatile("" : "+vr"(next0), "+vr"(next1));
        }
        s0                 = __riscv_vfmul_vf_f32m4(s0, decay, vl);
        s1                 = __riscv_vfmul_vf_f32m4(s1, decay, vl);
        const auto  zero   = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        const auto  pk0    = __riscv_vfmul_vv_f32m4(s0, kv, vl);
        const auto  pk1    = __riscv_vfmul_vv_f32m4(s1, kv, vl);
        const auto  sk0    = __riscv_vfredusum_vs_f32m4_f32m1(pk0, zero, vl);
        const auto  sk1    = __riscv_vfredusum_vs_f32m4_f32m1(pk1, zero, vl);
        const auto  pq0    = __riscv_vfmul_vv_f32m4(s0, qv, vl);
        const auto  pq1    = __riscv_vfmul_vv_f32m4(s1, qv, vl);
        const auto  sq0    = __riscv_vfredusum_vs_f32m4_f32m1(pq0, zero, vl);
        const auto  sq1    = __riscv_vfredusum_vs_f32m4_f32m1(pq1, zero, vl);
        const float delta0 = (v[j] - __riscv_vfmv_f_s_f32m1_f32(sk0)) * beta;
        const float delta1 = (v[j + 1] - __riscv_vfmv_f_s_f32m1_f32(sk1)) * beta;
        s0                 = __riscv_vfmacc_vf_f32m4(s0, delta0, kv, vl);
        s1                 = __riscv_vfmacc_vf_f32m4(s1, delta1, kv, vl);
        __riscv_vse32_v_f32m4(dst + j * d, s0, vl);
        __riscv_vse32_v_f32m4(dst + (j + 1) * d, s1, vl);
        const float y0 = (__riscv_vfmv_f_s_f32m1_f32(sq0) + delta0 * kq) * scale;
        const float y1 = (__riscv_vfmv_f_s_f32m1_f32(sq1) + delta1 * kq) * scale;
        out[j]         = y0;
        out[j + 1]     = y1;
        if (j + 2 < d) {
            s0 = next0;
            s1 = next1;
        }
    }
}

void forward_gated_delta_net(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src_q     = op->src[0];
    const ggml_tensor * src_k     = op->src[1];
    const ggml_tensor * src_v     = op->src[2];
    const ggml_tensor * src_g     = op->src[3];
    const ggml_tensor * src_beta  = op->src[4];
    const ggml_tensor * src_state = op->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);
    GGML_ASSERT(K >= 1 && S_v <= (int64_t) __riscv_vsetvlmax_e32m8());

    const int64_t nr  = H * n_seqs;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = std::min(nr, ir0 + dr);

    const int64_t state_seq_stride    = src_state->nb[3] / sizeof(float);
    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base       = (float *) op->data;
    float *       state_out_base      = attn_out_base + attn_score_elems;
    const float * state_in_base       = (const float *) src_state->data;

    const float scale = 1.0f / sqrtf((float) S_v);
    // Scalar GDN uses g[0]; KDA supplies one gate per state column.
    // When S_v == 1 the two forms coincide; retain the existing KDA dispatch.
    const bool  kda   = src_g->ne[0] == S_v;

    // Amortize the TCM transpose over prefill tokens; short tails use the row kernels.
    const uintptr_t shared_begin = reinterpret_cast<uintptr_t>(ctx.shared.data);
    const size_t    tcm_offset   = align_up(shared_begin, GDN_TCM_ALIGNMENT) - shared_begin;
    const size_t    tcm_bytes    = (S_v * S_v + 3 * S_v) * sizeof(float);
    const bool      transposed   = K == 1 && !kda && n_tokens >= GDN_CHUNK_SIZE && S_v % 4 == 0 &&
                            S_v >= (int64_t) __riscv_vsetvlmax_e32m2() && S_v <= (int64_t) __riscv_vsetvlmax_e32m4() &&
                            ctx.shared.data && tcm_offset <= ctx.shared.size &&
                            tcm_bytes <= ctx.shared.size - tcm_offset;
    float * tcm_state = transposed ? reinterpret_cast<float *>(shared_begin + tcm_offset) : nullptr;
    float * staged    = transposed ? tcm_state + S_v * S_v : nullptr;

    // prefill: chunkwise form (the token recurrence below handles the remaining cases)
    if (!transposed && K == 1 && !kda && n_tokens >= GDN_CHUNK_SIZE && S_v <= (int64_t) __riscv_vsetvlmax_e32m4() &&
        src_q->type == GGML_TYPE_F32 && src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32 &&
        src_g->type == GGML_TYPE_F32 && src_beta->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src_q) &&
        ggml_is_contiguous_rows(src_k) && ggml_is_contiguous_rows(src_v) && ggml_is_contiguous(src_g) &&
        ggml_is_contiguous(src_beta) && src_beta->ne[0] == 1) {
        const int64_t        rq3c = n_seqs / src_q->ne[3];
        const int64_t        rk3c = n_seqs / src_k->ne[3];
        const gdn_chunk_args args = { src_q, src_k, src_v, src_g, src_beta, S_v, H, n_tokens, scale };
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t iv1 = ir % H;
            const int64_t iv3 = ir / H;
            gdn_chunked_head(args, iv1 % src_q->ne[1], iv1 % src_k->ne[1], iv1, iv3 / rq3c, iv3 / rk3c, iv3,
                             state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v,
                             state_out_base + (iv3 * H + iv1) * S_v * S_v,
                             attn_out_base + (iv3 * n_tokens * H + iv1) * S_v);
        }
        return;
    }
    // Decode (n_tokens == 1, any n_seqs) and prefill tail chunks up to 4 tokens
    // run the fused single-pass row update: the state is read/written once per
    // token instead of once per phase, which halves the kernel time at
    // n_tokens == 1. Larger chunks keep the reference two-pass form — its early
    // state store keeps row loads pipelined, and with the chunk state L2
    // resident it already runs at the DRAM floor, so the fused form cannot win
    // there (measured crossover ~8 tokens; chunks are 64, or 16 with kda).
    const bool         fused = n_tokens <= 4;
    // KDA needs the per-column decay vector for every state row; the fused
    // row update consumes it as a vector, so only this small buffer remains
    // of the reference implementation's scratch space.
    std::vector<float> kda_decay(kda && fused ? (size_t) S_v : 0);
    // K > 1 (chunked prefill snapshots) evolves a private working copy.
    std::vector<float> state_work((fused && K > 1) ? (size_t) (S_v * S_v) : 0);
    // Two-pass scratch: per-row delta and attention sums, plus its own K > 1
    // working copy of the state.
    std::vector<float> scratch(!fused && !transposed ? (size_t) (3 * S_v + (K > 1 ? S_v * S_v : 0)) : 0);
    float *            delta         = !fused && !transposed ? scratch.data() : nullptr;
    float *            decay         = !fused && !transposed ? delta + S_v : nullptr;
    float *            state_q       = !fused && !transposed ? decay + S_v : nullptr;
    float *            state_work_tp = (!fused && !transposed && K > 1) ? state_q + S_v : nullptr;

    const int64_t rq3 = src_v->ne[3] / src_q->ne[3];
    const int64_t rk3 = src_v->ne[3] / src_k->ne[3];

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;
        const int64_t iv3 = ir / H;
        const int64_t iq1 = iv1 % src_q->ne[1];
        const int64_t ik1 = iv1 % src_k->ne[1];
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float *       s_out = transposed ? tcm_state :
                              K > 1      ? (fused ? state_work.data() : state_work_tp) :
                                           state_out_base + (iv3 * H + iv1) * S_v * S_v;
        const float * s_in  = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        if (transposed) {
            gdn_transpose(S_v, S_v, s_in, S_v, s_out, S_v);
        } else if (!fused) {
            memcpy(s_out, s_in, (size_t) (S_v * S_v) * sizeof(float));
        }
        // In the fused form the first token is fused with the state copy: it
        // reads the input state row and writes the updated row straight to the
        // output, so decode (n_tokens == 1) never materialises an intermediate
        // copy.
        const float * s_cur = s_in;

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * q    = (const float *) ((const char *) src_q->data + iq3 * src_q->nb[3] + t * src_q->nb[2] +
                                               iq1 * src_q->nb[1]);
            const float * k    = (const float *) ((const char *) src_k->data + ik3 * src_k->nb[3] + t * src_k->nb[2] +
                                               ik1 * src_k->nb[1]);
            const float * v    = (const float *) ((const char *) src_v->data + iv3 * src_v->nb[3] + t * src_v->nb[2] +
                                               iv1 * src_v->nb[1]);
            const float   beta = *(const float *) ((const char *) src_beta->data + iv3 * src_beta->nb[3] +
                                                 t * src_beta->nb[2] + iv1 * src_beta->nb[1]);
            const float * g    = (const float *) ((const char *) src_g->data + iv3 * src_g->nb[3] + t * src_g->nb[2] +
                                               iv1 * src_g->nb[1]);

            if (transposed) {
                memcpy(staged, k, S_v * sizeof(float));
                memcpy(staged + S_v, q, S_v * sizeof(float));
                memcpy(staged + 2 * S_v, v, S_v * sizeof(float));
                k = staged;
                q = staged + S_v;
                v = staged + 2 * S_v;
            }
            const float kq = rvv_gdn_dot_f32(k, q, S_v);

            if (transposed) {
                gdn_recurrent_transposed(s_out, k, q, expf(g[0]), v, beta, kq, scale, attn_data, S_v);
            } else if (fused) {
                float decay_scalar = 0.0f;
                if (kda) {
                    for (int64_t i = 0; i < S_v; ++i) {
                        kda_decay[i] = expf(g[i]);
                    }
                } else {
                    decay_scalar = expf(g[0]);
                }

                if (n_tokens == 1 && K == 1 && !kda && S_v % 2 == 0 && S_v == (int64_t) __riscv_vsetvlmax_e32m4()) {
                    gdn_recurrent_rows_lookahead(s_out, s_cur, k, q, v, decay_scalar, beta, kq, scale, S_v, attn_data);
                } else if (S_v <= (int64_t) __riscv_vsetvlmax_e32m4()) {
                    const size_t       vl = __riscv_vsetvl_e32m4(S_v);
                    const vfloat32m4_t kv = __riscv_vle32_v_f32m4(k, vl);
                    const vfloat32m4_t qv = __riscv_vle32_v_f32m4(q, vl);
                    const vfloat32m4_t dv =
                        kda ? __riscv_vle32_v_f32m4(kda_decay.data(), vl) : __riscv_vfmv_v_f_f32m4(decay_scalar, vl);
                    for (int64_t j = 0; j < S_v; ++j) {
                        attn_data[j] = rvv_gdn_fused_row_f32(s_out + j * S_v, s_cur + j * S_v, kv, qv, dv, v[j], beta,
                                                             kq, scale, vl);
                    }
                } else {
                    const size_t       vl = __riscv_vsetvl_e32m8(S_v);
                    const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k, vl);
                    const vfloat32m8_t qv = __riscv_vle32_v_f32m8(q, vl);
                    const vfloat32m8_t dv =
                        kda ? __riscv_vle32_v_f32m8(kda_decay.data(), vl) : __riscv_vfmv_v_f_f32m8(decay_scalar, vl);
                    for (int64_t j = 0; j < S_v; ++j) {
                        attn_data[j] = rvv_gdn_fused_row_f32(s_out + j * S_v, s_cur + j * S_v, kv, qv, dv, v[j], beta,
                                                             kq, scale, vl);
                    }
                }
                s_cur = s_out;
            } else {
                float decay_scalar = 0.0f;
                if (kda) {
                    for (int64_t i = 0; i < S_v; ++i) {
                        decay[i] = expf(g[i]);
                    }
                } else {
                    decay_scalar = expf(g[0]);
                }

                for (int64_t j = 0; j < S_v; ++j) {
                    const auto dots = rvv_gdn_decay_dots_f32_impl(s_out + j * S_v, decay, decay_scalar, k, q, S_v, kda);
                    delta[j]        = (v[j] - dots.state_k) * beta;
                    state_q[j]      = dots.state_q;
                }
                for (int64_t j = 0; j < S_v; ++j) {
                    rvv_gdn_update_f32(s_out + j * S_v, k, delta[j], S_v);
                    attn_data[j] = (state_q[j] + delta[j] * kq) * scale;
                }
            }
            attn_data += S_v * H;

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * state_snapshot =
                        state_out_base + target_slot * state_size_per_snap + (iv3 * H + iv1) * S_v * S_v;
                    memcpy(state_snapshot, s_out, (size_t) (S_v * S_v) * sizeof(float));
                }
            }
        }
        if (transposed) {
            gdn_transpose(S_v, S_v, s_out, S_v, state_out_base + (iv3 * H + iv1) * S_v * S_v, S_v);
        }
    }
}

void forward_scale_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, op->op_params, sizeof(float));

    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const int64_t ne0 = src0->ne[0], ne1 = src0->ne[1],
                  ne2 = src0->ne[2], ne3 = src0->ne[3];
    const int64_t nrows = ne1 * ne2 * ne3;
    const int64_t dr    = (nrows + nth - 1) / nth;
    const int64_t ir0   = dr * ith;
    const int64_t ir1   = MIN(ir0 + dr, nrows);

    for (int64_t ir = ir0; ir < ir1; ir++) {
        const int64_t i3 = ir / (ne1 * ne2);
        const int64_t i2 = (ir - i3 * ne1 * ne2) / ne1;
        const int64_t i1 = ir % ne1;

        const float * src_row = (const float *)((const char *)src0->data
            + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);
        float * dst_row = (float *)((char *)dst->data
            + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);

        int64_t remaining = ne0;
        while (remaining > 0) {
            const size_t vl = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v  = __riscv_vle32_v_f32m4(src_row, vl);
            v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
            __riscv_vse32_v_f32m4(dst_row, v, vl);
            src_row   += vl;
            dst_row   += vl;
            remaining -= vl;
        }
    }
}

void forward_cpy_strided_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    const int64_t ne    = ggml_nelements(src0);
    const int64_t ith   = ctx.ith;
    const int64_t nth   = ctx.nth;

    // Include the source row gaps when checking for overlap. Keep aliased
    // views on the coordinate-aware path below.
    const uintptr_t src_begin = reinterpret_cast<uintptr_t>(src0->data);
    const uintptr_t dst_begin = reinterpret_cast<uintptr_t>(dst->data);
    const size_t src_bytes = ggml_nbytes(src0);
    const size_t dst_bytes = ggml_nbytes(dst);
    const bool disjoint = src_begin < dst_begin ? dst_begin - src_begin >= src_bytes
                                               : src_begin - dst_begin >= dst_bytes;

    // The convolution state view contains three floats out of each four-float
    // row. Load exactly those three fields, including on the final row where
    // a four-field load would read beyond the backing allocation.
    if (disjoint && src0->ne[0] == 3 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src0->nb[0] == sizeof(float) && src0->nb[1] == 4 * sizeof(float)) {
        const int64_t rows_per_thread = (src0->ne[1] + nth - 1) / nth;
        const int64_t row0 = rows_per_thread * ith;
        const int64_t row1 = MIN(row0 + rows_per_thread, src0->ne[1]);
        if (row0 >= row1) {
            return;
        }

        for (int64_t row = row0; row < row1;) {
            const size_t vl = __riscv_vsetvl_e32m1(row1 - row);
            const float * src_ptr = (const float *) ((const char *) src0->data + row * src0->nb[1]);
            float * dst_ptr = (float *) dst->data + 3 * row;
            const vfloat32m1x3_t rows = __riscv_vlsseg3e32_v_f32m1x3(src_ptr, 4 * sizeof(float), vl);
            __riscv_vsseg3e32_v_f32m1x3(dst_ptr, rows, vl);
            row += vl;
        }
        return;
    }

    const int64_t dr    = (ne + nth - 1) / nth;
    const int64_t i0    = dr * ith;
    const int64_t i1    = MIN(i0 + dr, ne);
    if (i0 >= i1) {
        return;
    }

    // A contiguous source can be copied with the RVV memcpy primitive, but
    // CPY is also used for state/view updates where source and destination
    // may alias.  Only take the flat-copy path when the complete byte ranges
    // are disjoint; otherwise retain the coordinate-aware implementation
    // below (which is safe for exact/self-overlapping views).
    if (disjoint && ggml_is_contiguous(src0)) {
        memcpy1d((char *) dst->data + i0 * sizeof(float),
                 (const char *) src0->data + i0 * sizeof(float),
                 (i1 - i0) * sizeof(float));
        return;
    }

    float *       dst_ptr = (float *) dst->data + i0;

    for (int64_t i = i0; i < i1; i++) {
        int64_t rem = i;
        const int64_t i3_  = rem / (src0->ne[0] * src0->ne[1] * src0->ne[2]); rem -= i3_ * src0->ne[0] * src0->ne[1] * src0->ne[2];
        const int64_t i2_  = rem / (src0->ne[0] * src0->ne[1]);                rem -= i2_ * src0->ne[0] * src0->ne[1];
        const int64_t i1_  = rem / src0->ne[0];
        const int64_t i0_  = rem % src0->ne[0];
        const float * src_elem = (const float *)((const char *)src0->data
            + i0_ * src0->nb[0] + i1_ * src0->nb[1]
            + i2_ * src0->nb[2] + i3_ * src0->nb[3]);
        dst_ptr[i - i0] = *src_elem;
    }
}

void forward_cpy_f32_row_strided(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32);
    GGML_ASSERT(op->nb[0] == sizeof(float) && src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    const int64_t nc  = src0->ne[0];
    const int64_t nr1 = src0->ne[1];
    const int64_t nr2 = src0->ne[2];
    const int64_t nr3 = src0->ne[3];
    const int64_t nrows = nr1 * nr2 * nr3; // rows of nc floats

    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    // rows per thread, flattened over the outer dims
    const int64_t dr = (nrows + nth - 1) / nth;
    const int64_t r0 = dr * ith;
    const int64_t r1 = MIN(r0 + dr, nrows);
    if (r0 >= r1) {
        return;
    }

    // each row is contiguous (nb[0] == sizeof(float)); the row strides (nb[1..3]) are arbitrary on
    // either side, e.g. the GDN rollback-snapshot copies into row-strided cache_s views
    const size_t row_bytes = nc * sizeof(float);
    for (int64_t r = r0; r < r1; ++r) {
        int64_t rem = r;
        const int64_t i3 = rem / (nr1 * nr2); rem -= i3 * nr1 * nr2;
        const int64_t i2 = rem / nr1;
        const int64_t i1 = rem - i2 * nr1;

        const char * sp = (const char *) src0->data + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3];
        char *       dp = (char *)       dst->data + i1 * dst->nb[1]  + i2 * dst->nb[2]  + i3 * dst->nb[3];
        memcpy1d(dp, sp, row_bytes);
    }
}

void forward_cpy_f32_to_f16(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    const int64_t ne  = ggml_nelements(src0);
    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const int64_t dr = (ne + nth - 1) / nth;
    const int64_t i0 = dr * ith;
    const int64_t i1 = MIN(i0 + dr, ne);

    const float *    sp = (const float *)    src0->data + i0;
    _Float16 *       dp = (_Float16 *)       dst->data  + i0;

    int64_t remaining = i1 - i0;
    while (remaining > 0) {
        const size_t     vl  = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t     v32 = __riscv_vle32_v_f32m4(sp, vl);
        vfloat16m2_t     v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
        __riscv_vse16_v_f16m2(dp, v16, vl);
        sp        += vl;
        dp        += vl;
        remaining -= vl;
    }
}

void forward_cpy_f16_to_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    const int64_t ne  = ggml_nelements(src0);
    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const int64_t dr = (ne + nth - 1) / nth;
    const int64_t i0 = dr * ith;
    const int64_t i1 = MIN(i0 + dr, ne);

    const _Float16 * sp = (const _Float16 *) src0->data + i0;
    float *          dp = (float *)          dst->data  + i0;

    int64_t remaining = i1 - i0;
    while (remaining > 0) {
        const size_t     vl  = __riscv_vsetvl_e16m2(remaining);
        vfloat16m2_t     v16 = __riscv_vle16_v_f16m2(sp, vl);
        vfloat32m4_t     v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
        __riscv_vse32_v_f32m4(dp, v32, vl);
        sp        += vl;
        dp        += vl;
        remaining -= vl;
    }
}

void forward_unary_tanh_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t ne  = ggml_nelements(src0);
    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const int64_t dr = (ne + nth - 1) / nth;
    const int64_t i0 = dr * ith;
    const int64_t i1 = MIN(i0 + dr, ne);

    const float * src_ptr = (const float *) src0->data + i0;
    float *       dst_ptr = (float *) dst->data + i0;

    int64_t remaining = i1 - i0;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(src_ptr, vl);
        v               = rvv_tanh_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst_ptr, v, vl);
        src_ptr   += vl;
        dst_ptr   += vl;
        remaining -= vl;
    }
}

void forward_unary_gelu_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t ne  = ggml_nelements(src0);
    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const int64_t dr = (ne + nth - 1) / nth;
    const int64_t i0 = dr * ith;
    const int64_t i1 = MIN(i0 + dr, ne);

    const float * src_ptr = (const float *) src0->data;
    float *       dst_ptr = (float *) dst->data;

    static constexpr float GELU_ALPHA = 0.7978845608f;
    static constexpr float GELU_BETA  = 0.044715f;

    int64_t i = i0;
    while (i < i1) {
        const size_t vl = __riscv_vsetvl_e32m2(i1 - i);

        vfloat32m2_t x     = __riscv_vle32_v_f32m2(src_ptr + i, vl);
        vfloat32m2_t x2    = __riscv_vfmul_vv_f32m2(x, x, vl);
        vfloat32m2_t inner = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), GELU_BETA, x2, vl);
        inner              = __riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(x, inner, vl), GELU_ALPHA, vl);
        vfloat32m2_t th    = rvv_tanh_approx_f32m2(inner, vl);
        vfloat32m2_t out   = __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(x, __riscv_vfadd_vf_f32m2(th, 1.0f, vl), vl), 0.5f, vl);
        __riscv_vse32_v_f32m2(dst_ptr + i, out, vl);
        i += vl;
    }
}

void forward_unary_silu_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int64_t ne  = ggml_nelements(src0);
    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const int64_t dr = (ne + nth - 1) / nth;
    const int64_t i0 = dr * ith;
    const int64_t i1 = MIN(i0 + dr, ne);

    const float * src_ptr = (const float *) src0->data;
    float *       dst_ptr = (float *) dst->data;

    int64_t i = i0;
    while (i < i1) {
        const size_t     vl      = __riscv_vsetvl_e32m2(i1 - i);
        vfloat32m2_t     x       = __riscv_vle32_v_f32m2(src_ptr + i, vl);
        vfloat32m2_t     neg_x   = __riscv_vfneg_v_f32m2(x, vl);
        vfloat32m2_t     exp_neg = rvv_expf_approx_f32m2(neg_x, vl);
        vfloat32m2_t     denom   = __riscv_vfadd_vf_f32m2(exp_neg, 1.0f, vl);
        vfloat32m2_t     out     = __riscv_vfdiv_vv_f32m2(x, denom, vl);
        __riscv_vse32_v_f32m2(dst_ptr + i, out, vl);
        i += vl;
    }
}

// Helper macro: simple elementwise F32 unary kernel with RVV
#define DEFINE_UNARY_F32_RVV(name, rvv_expr) \
void name(ggml::spacemit::context & ctx, ggml_tensor * op) { \
    const ggml_tensor * src0 = op->src[0]; \
    ggml_tensor *       dst  = op; \
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32); \
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_are_same_shape(src0, dst)); \
    const int64_t ne  = ggml_nelements(src0); \
    const int64_t ith = ctx.ith, nth = ctx.nth; \
    const int64_t dr  = (ne + nth - 1) / nth; \
    const int64_t i0  = dr * ith, i1 = MIN(i0 + dr, ne); \
    const float * sp  = (const float *) src0->data; \
    float *       dp  = (float *)       dst->data; \
    int64_t i = i0; \
    while (i < i1) { \
        const size_t vl = __riscv_vsetvl_e32m2(i1 - i); \
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(sp + i, vl); \
        vfloat32m2_t r  = (rvv_expr); \
        __riscv_vse32_v_f32m2(dp + i, r, vl); \
        i += vl; \
    } \
}

DEFINE_UNARY_F32_RVV(forward_unary_exp_f32,
    rvv_expf_approx_f32m2(v, vl))
DEFINE_UNARY_F32_RVV(forward_unary_sigmoid_f32,
    __riscv_vfdiv_vv_f32m2(
        __riscv_vfmv_v_f_f32m2(1.0f, vl),
        __riscv_vfadd_vf_f32m2(rvv_expf_approx_f32m2(__riscv_vfneg_v_f32m2(v, vl), vl), 1.0f, vl),
        vl))
DEFINE_UNARY_F32_RVV(forward_unary_neg_f32,
    __riscv_vfneg_v_f32m2(v, vl))
DEFINE_UNARY_F32_RVV(forward_unary_softplus_f32,
    /* softplus(x) = log(1 + exp(x)) */
    __riscv_vfadd_vv_f32m2(
        __riscv_vfmax_vf_f32m2(v, 0.0f, vl),
        rvv_expf_approx_f32m2(
            __riscv_vfneg_v_f32m2(__riscv_vfabs_v_f32m2(v, vl), vl), vl),
        vl))

// Plain softmax (no mask, no sinks, no ALiBi) — RVV fast path
static void forward_soft_max_plain_f32(ggml::spacemit::context & ctx, ggml_tensor * op, float scale) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t ncols      = ne00;
    const int64_t task_count = ne01 * ne02 * ne03;
    const int64_t ith        = ctx.ith;
    const int64_t nth        = ctx.nth;

    const int64_t dr    = (task_count + nth - 1) / nth;
    const int64_t row0  = dr * ith;
    const int64_t row1  = MIN(row0 + dr, task_count);

    for (int64_t row = row0; row < row1; row++) {
        const int64_t i03 = row / (ne02 * ne01);
        const int64_t i02 = (row - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = row - i03 * ne02 * ne01 - i02 * ne01;

        const float * src_row = (const float *) ((const char *) src0->data
            + i01 * nb01 + i02 * nb02 + i03 * nb03);
        float * dst_row = (float *) ((const char *) dst->data
            + i01 * nb1  + i02 * nb2  + i03 * nb3);

        // find max
        float max_val = -INFINITY;
        {
            int64_t n = ncols;
            const float * p = src_row;
            while (n > 0) {
                const size_t     vl  = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t     v   = __riscv_vle32_v_f32m4(p, vl);
                vfloat32m1_t     red = __riscv_vfredmax_vs_f32m4_f32m1(
                    v, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vl);
                float cur_max = __riscv_vfmv_f_s_f32m1_f32(red);
                if (cur_max > max_val) max_val = cur_max;
                p += vl;
                n -= vl;
            }
        }
        max_val *= scale;

        // copy src*scale into dst, then compute exp(dst[i] - max_val) in-place and sum
        {
            int64_t       n  = ncols;
            const float * sp = src_row;
            float *       dp = dst_row;
            while (n > 0) {
                const size_t vl = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t v  = __riscv_vle32_v_f32m4(sp, vl);
                v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
                __riscv_vse32_v_f32m4(dp, v, vl);
                sp += vl;
                dp += vl;
                n  -= vl;
            }
        }
        const float sum = rvv_softmax_exp_inplace_f32(dst_row, ncols, max_val);

        // normalize
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        {
            int64_t n  = ncols;
            float * dp = dst_row;
            while (n > 0) {
                const size_t vl = __riscv_vsetvl_e32m4(n);
                vfloat32m4_t v  = __riscv_vle32_v_f32m4(dp, vl);
                v               = __riscv_vfmul_vf_f32m4(v, inv_sum, vl);
                __riscv_vse32_v_f32m4(dp, v, vl);
                dp += vl;
                n  -= vl;
            }
        }
    }
}

void forward_soft_max_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];  // mask (f32 or f16), optional
    const ggml_tensor * src2 = op->src[2];  // sinks (f32 [ne2]), optional
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (const float *) op->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

    if (!src1 && !src2 && max_bias == 0.0f) {
        forward_soft_max_plain_f32(ctx, op, scale);
        return;
    }

    // mask / sinks / ALiBi path — mirrors ggml_compute_forward_soft_max_f32
    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    const bool use_f16 = src1 && src1->type == GGML_TYPE_F16;
    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const float * sk = src2 ? (const float *) src2->data : nullptr;

    for (int64_t i3 = 0; i3 < ne03; ++i3) {
        for (int64_t i2 = 0; i2 < ne02; ++i2) {
            for (int64_t i1 = ith; i1 < ne01; i1 += nth) {
                const float * sp = (const float *) ((const char *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                float       * dp = (float *)       ((char *) dst->data  + i1 * nb1  + i2 * nb2  + i3 * nb3);

                const char * mp = nullptr;
                if (src1) {
                    const int64_t i12 = i2 % ne12;
                    const int64_t i13 = i3 % ne13;
                    mp = (const char *) src1->data + i1 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3];
                }

                const uint32_t h     = i2;
                const float    slope = (max_bias > 0.0f)
                    ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1))
                    : 1.0f;

                // pass 1: max of src*scale (+ slope*mask), corrected by the sink
                float max_val = -INFINITY;
                for (int64_t i0 = 0; i0 < ne00; ++i0) {
                    float val = sp[i0] * scale;
                    if (mp) {
                        const float mv = use_f16
                            ? ggml_compute_fp16_to_fp32(*(const ggml_fp16_t *) (mp + i0 * src1->nb[0]))
                            : *(const float *) (mp + i0 * src1->nb[0]);
                        val += slope * mv;
                    }
                    if (val > max_val) max_val = val;
                }
                if (sk && sk[i2] > max_val) max_val = sk[i2];

                // pass 2: exp and normalise (double accumulation like ggml-cpu)
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < ne00; ++i0) {
                    float val = sp[i0] * scale;
                    if (mp) {
                        const float mv = use_f16
                            ? ggml_compute_fp16_to_fp32(*(const ggml_fp16_t *) (mp + i0 * src1->nb[0]))
                            : *(const float *) (mp + i0 * src1->nb[0]);
                        val += slope * mv;
                    }
                    dp[i0] = expf(val - max_val);
                    sum   += dp[i0];
                }
                if (sk) sum += expf(sk[i2] - max_val);

                const float inv_sum = (sum > 0.0) ? (float) (1.0 / sum) : 0.0f;
                for (int64_t i0 = 0; i0 < ne00; ++i0) dp[i0] *= inv_sum;
            }
        }
    }
}

// F16 weight (src0[K,M]) x F32 activation (src1[K,N]) -> F32 output (dst[M,N])
// Parallel over output rows m across ctx.nth cores.
void forward_mul_mat_f16_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(_Float16) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int64_t M    = src0->ne[1];
    const int64_t K    = src0->ne[0];
    const int64_t N    = src1->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t bn2  = src1->ne[2];
    const int64_t bn3  = src1->ne[3];
    const int64_t ith  = ctx.ith;
    const int64_t nth  = ctx.nth;

    // one work unit per (batch3, batch2, m): a-rows across all b batches
    const int64_t nr = M * bn2 * bn3;
    const int64_t dr = (nr + nth - 1) / nth;
    const int64_t r0 = ith * dr;
    const int64_t r1 = MIN(r0 + dr, nr);

    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i3 = r / (M * bn2);
        const int64_t i2 = (r - i3 * M * bn2) / M;
        const int64_t m  = r - i3 * M * bn2 - i2 * M;
        // src0 batch dims broadcast by block-stretch like ggml-cpu: i02 = i12/r2
        const int64_t r2 = bn2 / ne02;
        const int64_t r3 = bn3 / ne03;
        const int64_t a2 = i2 / r2;
        const int64_t a3 = i3 / r3;

        const _Float16 * w_row = (const _Float16 *)
            ((const char *) src0->data + a2 * src0->nb[2] + a3 * src0->nb[3] + m * src0->nb[1]);

        for (int64_t n = 0; n < N; ++n) {
            const float * x_row = (const float *)
                ((const char *) src1->data + i2 * src1->nb[2] + i3 * src1->nb[3] + n * src1->nb[1]);

            vfloat32m4_t     acc    = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
            int64_t          k_left = K;
            const _Float16 * wp     = w_row;
            const float *    xp     = x_row;

            while (k_left > 0) {
                const size_t vl  = __riscv_vsetvl_e16m2(k_left);
                vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp, vl);
                vfloat32m4_t w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
                vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, w32, x32, vl);
                wp     += vl;
                xp     += vl;
                k_left -= vl;
            }

            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t sum  = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero, __riscv_vsetvlmax_e32m4());
            // dst is [M, N, bn2, bn3]: row n of batch (i2, i3) holds M results
            float * d_row = (float *) ((char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3] + n * dst->nb[1]);
            d_row[m]      = __riscv_vfmv_f_s_f32m1_f32(sum);
        }
    }
}

// General stride-aware contiguous copy for F32/F16.
// src0 may be non-contiguous; dst is always contiguous.
// Parallel over elements across ctx.nth cores.
template <typename T>
void forward_cont_general(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(T));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    const int64_t ith   = ctx.ith;
    const int64_t nth   = ctx.nth;
    const int64_t nr    = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t dr    = (nr + nth - 1) / nth;
    const int64_t r0    = dr * ith;
    const int64_t r1    = MIN(r0 + dr, nr);
    const size_t row_size = (size_t) src0->ne[0] * sizeof(T);

    // Match ggml-cpu's fast duplicate path: when the source row itself is
    // contiguous, copy complete rows instead of doing per-element index math.
    if (src0->nb[0] == sizeof(T)) {
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t i3 = r / (src0->ne[2] * src0->ne[1]);
            const int64_t rem = r - i3 * src0->ne[2] * src0->ne[1];
            const int64_t i2 = rem / src0->ne[1];
            const int64_t i1 = rem - i2 * src0->ne[1];
            const char * src_row = (const char *) src0->data + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3];
            char * dst_row = (char *) dst->data + r * row_size;
            memcpy(dst_row, src_row, row_size);
        }
        return;
    }

    // Fully general strided fallback (rare for model graphs).
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i3 = r / (src0->ne[2] * src0->ne[1]);
        const int64_t rem = r - i3 * src0->ne[2] * src0->ne[1];
        const int64_t i2 = rem / src0->ne[1];
        const int64_t i1 = rem - i2 * src0->ne[1];
        for (int64_t i0 = 0; i0 < src0->ne[0]; ++i0) {
            const T * src_elem = (const T *)((const char *)src0->data + i0 * src0->nb[0] + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);
            ((T *) dst->data)[r * src0->ne[0] + i0] = *src_elem;
        }
    }
}

template void forward_cont_general<float>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_cont_general<_Float16>(ggml::spacemit::context & ctx, ggml_tensor * op);

void forward_glu_geglu_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int64_t nc    = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr    = ggml_nrows(src0);
    const int64_t total = nr * nc;
    const int64_t ith   = ctx.ith;
    const int64_t nth   = ctx.nth;

    const int64_t dr = (total + nth - 1) / nth;
    const int64_t e0 = dr * ith;
    const int64_t e1 = MIN(e0 + dr, total);

    const int32_t swapped = ggml_get_op_params_i32(op, 1);

    static constexpr float GELU_ALPHA = 0.79788456080286535587989211986876f;
    static constexpr float GELU_BETA  = 0.044715f;

    int64_t e = e0;
    while (e < e1) {
        const int64_t r   = e / nc;
        const int64_t c   = e % nc;
        const int64_t run = MIN(nc - c, e1 - e);

        const float * x_row = (const float *) ((const char *) src0->data + r * src0->nb[1]);
        const float * g_row;
        if (src1) {
            g_row = (const float *) ((const char *) src1->data + r * src1->nb[1]);
        } else {
            x_row = x_row + (swapped ? nc : 0);
            g_row = (const float *) ((const char *) src0->data + r * src0->nb[1]) + (swapped ? 0 : nc);
        }
        float * y_row = (float *) ((char *) op->data + r * op->nb[1]);

        const float * xp = x_row + c;
        const float * gp = g_row + c;
        float *       yp = y_row + c;

        int64_t remaining = run;
        while (remaining > 0) {
            const size_t vl = __riscv_vsetvl_e32m2(remaining);

            vfloat32m2_t xv = __riscv_vle32_v_f32m2(xp, vl);
            vfloat32m2_t gv = __riscv_vle32_v_f32m2(gp, vl);

            vfloat32m2_t x2    = __riscv_vfmul_vv_f32m2(xv, xv, vl);
            vfloat32m2_t inner = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), GELU_BETA, x2, vl);
            inner              = __riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(xv, inner, vl), GELU_ALPHA, vl);

            vfloat32m2_t th   = rvv_tanh_approx_f32m2(inner, vl);
            vfloat32m2_t gelu = __riscv_vfmul_vf_f32m2(
                    __riscv_vfmul_vv_f32m2(xv, __riscv_vfadd_vf_f32m2(th, 1.0f, vl), vl), 0.5f, vl);

            __riscv_vse32_v_f32m2(yp, __riscv_vfmul_vv_f32m2(gelu, gv, vl), vl);

            xp        += vl;
            gp        += vl;
            yp        += vl;
            remaining -= vl;
        }
        e += run;
    }
}

template <typename T>
static void forward_rope_impl(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0         = op->src[0];
    const ggml_tensor * src1         = op->src[1];
    const ggml_tensor * src2         = op->src[2];
    const float *       freq_factors = src2 ? (const float *) src2->data : nullptr;

    const int n_dims     = ggml_get_op_params_i32(op, 1);
    const int mode       = ggml_get_op_params_i32(op, 2);
    const int n_ctx_orig = ggml_get_op_params_i32(op, 4);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op->op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  op->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  op->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, op->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   op->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   op->op_params + 10, sizeof(float));

    int sections[4] = {0, 0, 0, 0};
    memcpy(sections, op->op_params + 11, sizeof(int) * 4);

    const bool mrope_used = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool is_imrope  = (mode == GGML_ROPE_TYPE_IMROPE);

    // YaRN correction dims
    float corr_dims[2] = { 0.0f, 0.0f };
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    const int64_t ne2 = op->ne[2];  // seq-len dimension
    const int64_t nr  = ggml_nrows(op);
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);
    const int32_t * pos = (const int32_t *) src1->data;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w     = sections[0] + sections[1];
    const int sec_e     = sec_w + sections[2];

    // The small F32 NEOX shape used by Qwen3-0.6B is latency-sensitive and
    // was faster on the original scalar path.  Keep that path shape-based
    // (never model-name based), while larger rotary dimensions use the RVV
    // implementation below.
    const bool legacy_small_f32 = std::is_same_v<T, float> &&
        mode == GGML_ROPE_TYPE_NEOX && op->ne[0] <= 128 &&
        src0->nb[0] == sizeof(float);
    float legacy_cache[512];
    float * cache = legacy_small_f32 ? legacy_cache : nullptr;
    if (!legacy_small_f32) {
        GGML_ASSERT(ctx.workspace_size >= (size_t) ctx.nth * (size_t) (op->ne[0] + ggml::spacemit::cache_line_size_f32) * sizeof(float));
        cache = (float *) ctx.workspace +
                ctx.ith * (op->ne[0] + ggml::spacemit::cache_line_size_f32);
    }
    int64_t last_i2 = -1;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (op->ne[2] * op->ne[1]);
        const int64_t i2 = (ir / op->ne[1]) % op->ne[2];
        const int64_t i1 = ir % op->ne[1];

        if (i2 != last_i2) {
            if (!mrope_used) {
                // Standard / YaRN rope
                float theta = (float) pos[i2];
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float ff           = freq_factors ? freq_factors[i0 / 2] : 1.0f;
                    const float theta_extrap = theta / ff;
                    const float theta_interp = freq_scale * theta_extrap;
                    float t;
                    if (ext_factor != 0.0f) {
                        const float y        = (i0 / 2 - corr_dims[0]) / fmaxf(0.001f, corr_dims[1] - corr_dims[0]);
                        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                        t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    } else {
                        t = theta_interp;
                    }
                    float s, c;
                    s = sinf(t);
                    c = cosf(t);
                    cache[i0 + 0] = c * mscale;
                    cache[i0 + 1] = s * mscale;
                    theta *= theta_scale;
                }
            } else {
                // M-RoPE / IMROPE: multiple position sequences in src1
                const float p_t = (float) pos[i2];
                const float p_h = (float) pos[i2 + ne2];
                const float p_w = (float) pos[i2 + ne2 * 2];
                const float p_e = (float) pos[i2 + ne2 * 3];

                float theta_t = p_t;
                float theta_h = p_h;
                float theta_w = p_w;
                float theta_e = p_e;

                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
                    int sector = (sect_dims > 0) ? (i0 / 2) % sect_dims : 0;

                    float theta;
                    if (is_imrope) {
                        if      (sector % 3 == 0 && sector < 3 * sections[0]) theta = theta_t;
                        else if (sector % 3 == 1 && sector < 3 * sections[1]) theta = theta_h;
                        else if (sector % 3 == 2 && sector < 3 * sections[2]) theta = theta_w;
                        else                                                    theta = theta_e;
                    } else {
                        if      (sector < sections[0])  theta = theta_t;
                        else if (sector < sec_w)         theta = theta_h;
                        else if (sector < sec_e)         theta = theta_w;
                        else                             theta = theta_e;
                    }

                    const float theta_extrap = theta / ff;
                    const float theta_interp = freq_scale * theta_extrap;
                    float t;
                    if (ext_factor != 0.0f) {
                        const float y        = (i0 / 2 - corr_dims[0]) / fmaxf(0.001f, corr_dims[1] - corr_dims[0]);
                        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                        t = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    } else {
                        t = theta_interp;
                    }
                    float s, c;
                    s = sinf(t);
                    c = cosf(t);
                    cache[i0 + 0] = c * mscale;
                    cache[i0 + 1] = s * mscale;

                    theta_t *= theta_scale;
                    theta_h *= theta_scale;
                    theta_w *= theta_scale;
                    theta_e *= theta_scale;
                }
            }
            last_i2 = i2;
        }

        const T * src = (const T *) ((const char *) src0->data + i3 * src0->nb[3] + i2 * src0->nb[2] + i1 * src0->nb[1]);
        T * dst_row   = (T *) ((char *) op->data + i3 * op->nb[3] + i2 * op->nb[2] + i1 * op->nb[1]);

        if (legacy_small_f32) {
            // Preserve the original low-dimensional F32 NEOX implementation.
            const int offset = n_dims / 2;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const int ic = i0 / 2;
                const float x0 = src[ic];
                const float x1 = src[ic + offset];
                dst_row[ic]          = x0 * cache[i0] - x1 * cache[i0 + 1];
                dst_row[ic + offset] = x0 * cache[i0 + 1] + x1 * cache[i0];
            }
        } else {
        // NEOX / MROPE / IMROPE use half-offset rotation.  Qwen3.5 uses
        // contiguous F32 rows here; process all pairs with RVV loads instead
        // of the old scalar element loop.  Keep NORMAL and non-F32 layouts on
        // the reference path because their pair mapping differs.
        if constexpr (std::is_same_v<T, float>) {
            if (mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE) {
                const int64_t pairs = n_dims / 2;
                int64_t j = 0;
                while (j < pairs) {
                    const size_t vl = __riscv_vsetvl_e32m4((size_t) (pairs - j));
                    const vfloat32m4_t x0 = __riscv_vle32_v_f32m4(src + j, vl);
                    const vfloat32m4_t x1 = __riscv_vle32_v_f32m4(src + j + pairs, vl);
                    const vfloat32m4_t c = __riscv_vlse32_v_f32m4(cache + 2*j, (ptrdiff_t) (2*sizeof(float)), vl);
                    const vfloat32m4_t s = __riscv_vlse32_v_f32m4(cache + 2*j + 1, (ptrdiff_t) (2*sizeof(float)), vl);
                    const vfloat32m4_t y0 = __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(x0, c, vl),
                                                                    __riscv_vfmul_vv_f32m4(x1, s, vl), vl);
                    const vfloat32m4_t y1 = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(x0, s, vl),
                                                                    __riscv_vfmul_vv_f32m4(x1, c, vl), vl);
                    __riscv_vse32_v_f32m4(dst_row + j, y0, vl);
                    __riscv_vse32_v_f32m4(dst_row + j + pairs, y1, vl);
                    j += vl;
                }
            } else {
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float x0 = src[i0];
                    const float x1 = src[i0 + 1];
                    dst_row[i0] = x0 * cache[i0] - x1 * cache[i0 + 1];
                    dst_row[i0 + 1] = x0 * cache[i0 + 1] + x1 * cache[i0];
                }
            }
        } else {
            const int offset = n_dims / 2;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const int ic = (mode == GGML_ROPE_TYPE_NORMAL) ? i0 : i0 / 2;
                const int no = (mode == GGML_ROPE_TYPE_NORMAL) ? 1 : offset;
                const float x0 = (float) src[ic];
                const float x1 = (float) src[ic + no];
                dst_row[ic]          = (T) (x0 * cache[i0] - x1 * cache[i0 + 1]);
                dst_row[ic + no] = (T) (x0 * cache[i0 + 1] + x1 * cache[i0]);
            }
        }
        }
        for (int64_t i0 = n_dims; i0 < op->ne[0]; ++i0) {
            dst_row[i0] = src[i0];
        }
    }
}

void forward_rope(ggml::spacemit::context & ctx, ggml_tensor * op) {
    if (op->src[0]->type == GGML_TYPE_F32) {
        forward_rope_impl<float>(ctx, op);
    } else {
        forward_rope_impl<_Float16>(ctx, op);
    }
}

void forward_set_rows(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    const int64_t nc  = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t dc  = (nc + ctx.nth - 1) / ctx.nth;
    const int64_t c0  = dc * ctx.ith;
    const int64_t c1  = MIN(c0 + dc, nc);

    // Every core owns a disjoint column range but visits source rows in the
    // same order. If indices repeat, this preserves SET_ROWS' last-row-wins
    // semantics without cross-core write races.
    for (int64_t i03 = 0; i03 < src0->ne[3]; ++i03) {
        for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) {
            const int64_t i11 = i02 % src1->ne[1];
            const int64_t i12 = i03 % src1->ne[2];
            for (int64_t i = 0; i < nr; ++i) {
                const int64_t idx = src1->type == GGML_TYPE_I64
                        ? *(const int64_t *) ((const char *) src1->data + i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2])
                        : *(const int32_t *) ((const char *) src1->data + i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]);
                GGML_ASSERT(idx >= 0 && idx < dst->ne[1]);

                const char * src = (const char *) src0->data + i * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3];
                char * out = (char *) dst->data + idx * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3];

                int64_t c = c0;
                while (c < c1) {
                    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                        const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
                        const vfloat32m4_t v32 = __riscv_vle32_v_f32m4((const float *) src + c, vl);
                        __riscv_vse16_v_f16m2((_Float16 *) out + c, __riscv_vfncvt_f_f_w_f16m2(v32, vl), vl);
                        c += vl;
                    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
                        const size_t vl = __riscv_vsetvl_e16m2(c1 - c);
                        const vfloat16m2_t v16 = __riscv_vle16_v_f16m2((const _Float16 *) src + c, vl);
                        __riscv_vse32_v_f32m4((float *) out + c, __riscv_vfwcvt_f_f_v_f32m4(v16, vl), vl);
                        c += vl;
                    } else if (src0->type == GGML_TYPE_F32) {
                        const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
                        __riscv_vse32_v_f32m4((float *) out + c, __riscv_vle32_v_f32m4((const float *) src + c, vl), vl);
                        c += vl;
                    } else {
                        const size_t vl = __riscv_vsetvl_e16m2(c1 - c);
                        __riscv_vse16_v_f16m2((_Float16 *) out + c,
                                              __riscv_vle16_v_f16m2((const _Float16 *) src + c, vl), vl);
                        c += vl;
                    }
                }
            }
        }
    }
}

void forward_glu_swiglu_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int64_t nc      = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr      = ggml_nrows(src0);
    const int64_t total   = nr * nc;
    const int64_t dr      = (total + ctx.nth - 1) / ctx.nth;
    const int64_t e0      = dr * ctx.ith;
    const int64_t e1      = MIN(e0 + dr, total);
    const int32_t swapped = ggml_get_op_params_i32(op, 1);

    int64_t e = e0;
    while (e < e1) {
        const int64_t r   = e / nc;
        const int64_t c   = e % nc;
        const int64_t run = MIN(nc - c, e1 - e);

        const float * x_row = (const float *) ((const char *) src0->data + r * src0->nb[1]);
        const float * g_row;
        if (src1) {
            g_row = (const float *) ((const char *) src1->data + r * src1->nb[1]);
        } else {
            x_row += swapped ? nc : 0;
            g_row = (const float *) ((const char *) src0->data + r * src0->nb[1]) + (swapped ? 0 : nc);
        }
        float * y_row = (float *) ((char *) op->data + r * op->nb[1]);

        const float * xp = x_row + c;
        const float * gp = g_row + c;
        float *       yp = y_row + c;
        int64_t remaining = run;
        while (remaining > 0) {
            const size_t vl = __riscv_vsetvl_e32m2(remaining);
            const vfloat32m2_t x = __riscv_vle32_v_f32m2(xp, vl);
            const vfloat32m2_t g = __riscv_vle32_v_f32m2(gp, vl);
            const vfloat32m2_t exp_neg_x = rvv_expf_approx_f32m2(__riscv_vfneg_v_f32m2(x, vl), vl);
            const vfloat32m2_t silu = __riscv_vfdiv_vv_f32m2(
                    x, __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl), vl);
            __riscv_vse32_v_f32m2(yp, __riscv_vfmul_vv_f32m2(silu, g, vl), vl);
            xp += vl;
            gp += vl;
            yp += vl;
            remaining -= vl;
        }
        e += run;
    }
}

template void forward_binary<GGML_OP_ADD, float>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_SUB, float>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_MUL, float>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_DIV, float>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_ADD, _Float16>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_SUB, _Float16>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_MUL, _Float16>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_binary<GGML_OP_DIV, _Float16>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_sum_rows<float>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_sum_rows<_Float16>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_repeat_nrows<int32_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_repeat_nrows<int16_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_repeat_dim1<int32_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_repeat_dim1<int16_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_get_rows<int32_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_get_rows<int16_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
// Ported from ggml-cpu/spacemit (linpengt/qwen35-opt).
// y = rms_norm(x) * w, fused (the RMS_NORM + MUL pair that ggml_cpu_try_fuse_ops would otherwise run as a
// scalar loop). One pass for the sum of squares, one for y = x * s * w. Rows of at most vlmax(e32m4) elements are
// done four at a time so the four reductions overlap. Each output row is written after its input row is read,
// the same aliasing contract as the generic fused kernel.
void forward_rms_norm_mul_f32(ggml::spacemit::context & ctx, const ggml_tensor * rms, ggml_tensor * mul) {
    const ggml_tensor * src0 = rms->src[0];
    const ggml_tensor * w    = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    ggml_tensor *       dst  = mul;

    const int ith = (int64_t) ctx.ith;
    const int nth = (int64_t) ctx.nth;

    const int64_t n   = src0->ne[0];
    const int64_t ne1 = src0->ne[1], ne2 = src0->ne[2], ne3 = src0->ne[3];
    float         eps;
    memcpy(&eps, rms->op_params, sizeof(float));

    const int64_t rows = ne1 * ne2 * ne3;
    const int64_t per  = (rows + nth - 1) / nth;
    const int64_t r0   = std::min(rows, per * ith);
    const int64_t r1   = std::min(rows, r0 + per);

    auto x_row = [&](int64_t r, int64_t & i1, int64_t & i2, int64_t & i3) {
        i3 = r / (ne2 * ne1);
        i2 = (r - i3 * ne2 * ne1) / ne1;
        i1 = r - i3 * ne2 * ne1 - i2 * ne1;
        return (const float *) ((const char *) src0->data + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);
    };
    auto y_row = [&](int64_t i1, int64_t i2, int64_t i3) {
        return (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
    };
    auto w_row = [&](int64_t i1, int64_t i2, int64_t i3) {
        return (const float *) ((const char *) w->data + (i1 % w->ne[1]) * w->nb[1] + (i2 % w->ne[2]) * w->nb[2] +
                                (i3 % w->ne[3]) * w->nb[3]);
    };

    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    int64_t            r    = r0;
    if (n <= (int64_t) __riscv_vsetvlmax_e32m4() && w->ne[1] == 1 && w->ne[2] == 1 && w->ne[3] == 1) {
        const size_t       vl = __riscv_vsetvl_e32m4(n);
        const vfloat32m4_t wv = __riscv_vle32_v_f32m4((const float *) w->data, vl);
        for (; r + 4 <= r1; r += 4) {
            int64_t       a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3;
            const float * xa = x_row(r + 0, a1, a2, a3);
            const float * xb = x_row(r + 1, b1, b2, b3);
            const float * xc = x_row(r + 2, c1, c2, c3);
            const float * xd = x_row(r + 3, d1, d2, d3);
            vfloat32m4_t  va = __riscv_vle32_v_f32m4(xa, vl), vb = __riscv_vle32_v_f32m4(xb, vl);
            vfloat32m4_t  vc = __riscv_vle32_v_f32m4(xc, vl), vd = __riscv_vle32_v_f32m4(xd, vl);
            vfloat32m1_t  sa = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(va, va, vl), zero, vl);
            vfloat32m1_t  sb = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vb, vb, vl), zero, vl);
            vfloat32m1_t  sc = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vc, vc, vl), zero, vl);
            vfloat32m1_t  sd = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vd, vd, vl), zero, vl);
            const float   fa = 1.0f / sqrtf(__riscv_vfmv_f_s_f32m1_f32(sa) / n + eps);
            const float   fb = 1.0f / sqrtf(__riscv_vfmv_f_s_f32m1_f32(sb) / n + eps);
            const float   fc = 1.0f / sqrtf(__riscv_vfmv_f_s_f32m1_f32(sc) / n + eps);
            const float   fd = 1.0f / sqrtf(__riscv_vfmv_f_s_f32m1_f32(sd) / n + eps);
            __riscv_vse32_v_f32m4(y_row(a1, a2, a3), __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(va, fa, vl), wv, vl), vl);
            __riscv_vse32_v_f32m4(y_row(b1, b2, b3), __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vb, fb, vl), wv, vl), vl);
            __riscv_vse32_v_f32m4(y_row(c1, c2, c3), __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vc, fc, vl), wv, vl), vl);
            __riscv_vse32_v_f32m4(y_row(d1, d2, d3), __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vd, fd, vl), wv, vl), vl);
        }
    }
    for (; r < r1; r++) {
        int64_t       i1, i2, i3;
        const float * x  = x_row(r, i1, i2, i3);
        float *       y  = y_row(i1, i2, i3);
        const float * wr = w_row(i1, i2, i3);
        size_t        vl = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t  acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (int64_t i = 0; i < n; i += vl) {
            vl                = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t v    = __riscv_vle32_v_f32m8(x + i, vl);
            acc               = __riscv_vfmacc_vv_f32m8_tu(acc, v, v, vl);
        }
        const float s =
            1.0f / sqrtf(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(acc, zero, __riscv_vsetvlmax_e32m8())) / n + eps);
        for (int64_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            __riscv_vse32_v_f32m8(
                y + i, __riscv_vfmul_vv_f32m8(__riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(x + i, vl), s, vl),
                                              __riscv_vle32_v_f32m8(wr + i, vl), vl), vl);
        }
    }
}

// ssm_conv fused with its concat(conv_states, transpose(qkv)) input: reads the conv state (d_conv - 1
// columns per channel) and the untransposed qkv rows directly, so every token is one contiguous load across
// channels instead of a strided gather. Same arithmetic order as forward_ssm_conv_f32 (bitwise identical).
bool forward_ssm_conv_concat_f32(ggml::spacemit::context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cat = dst->src[0];  // concat {d_conv - 1 + n_t, d_inner, n_s}
    const ggml_tensor * st  = cat->src[0];  // conv states {d_conv - 1, d_inner, n_s}
    const ggml_tensor * qk  = cat->src[1];  // transposed qkv view {n_t, d_inner, n_s}, channels contiguous
    const ggml_tensor * src1 = dst->src[1]; // conv1d.weight {d_conv, d_inner}

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = cat->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    if (cat->op != GGML_OP_CONCAT || nc != 4 || st->ne[0] != 3 || qk->ne[0] != n_t || src1->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 || qk->type != GGML_TYPE_F32 ||
        st->nb[0] != sizeof(float) || qk->nb[1] != sizeof(float) || src1->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float) || dst->ne[0] != nr || st->ne[1] != nr || qk->ne[1] != nr ||
        st->ne[2] != n_s || qk->ne[2] != n_s) {
        return false;
    }

    const ptrdiff_t ss = st->nb[1];  // channel stride of the conv state
    const ptrdiff_t ts = qk->nb[0];  // token stride of qkv
    const ptrdiff_t ws = src1->nb[1];

    const int64_t dr  = (nr + (int64_t) ctx.nth - 1) / (int64_t) ctx.nth;
    const int64_t ir0 = dr * (int64_t) ctx.ith;
    const int64_t ir1 = std::min(ir0 + dr, nr);

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t c = ir0; c < ir1;) {
            const size_t vl = __riscv_vsetvl_e32m4(ir1 - c);

            const char * wp = (const char *) src1->data + c * ws;
            const vfloat32m4_t w0 = __riscv_vlse32_v_f32m4((const float *) wp + 0, ws, vl);
            const vfloat32m4_t w1 = __riscv_vlse32_v_f32m4((const float *) wp + 1, ws, vl);
            const vfloat32m4_t w2 = __riscv_vlse32_v_f32m4((const float *) wp + 2, ws, vl);
            const vfloat32m4_t w3 = __riscv_vlse32_v_f32m4((const float *) wp + 3, ws, vl);

            const float * sp = (const float *) ((const char *) st->data + c * ss + i3 * st->nb[2]);
            vfloat32m4_t  x0 = __riscv_vlse32_v_f32m4(sp + 0, ss, vl);
            vfloat32m4_t  x1 = __riscv_vlse32_v_f32m4(sp + 1, ss, vl);
            vfloat32m4_t  x2 = __riscv_vlse32_v_f32m4(sp + 2, ss, vl);

            const char * qp = (const char *) qk->data + c * sizeof(float) + i3 * qk->nb[2];
            char *       yp = (char *) dst->data + c * sizeof(float) + i3 * dst->nb[2];

            for (int64_t t = 0; t < n_t; ++t) {
                const vfloat32m4_t x3 = __riscv_vle32_v_f32m4((const float *) (qp + t * ts), vl);

                vfloat32m4_t acc = __riscv_vfmul_vv_f32m4(x0, w0, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x1, w1, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x2, w2, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x3, w3, vl);

                __riscv_vse32_v_f32m4((float *) (yp + t * dst->nb[1]), acc, vl);

                x0 = x1;
                x1 = x2;
                x2 = x3;
            }
            c += vl;
        }
    }
    return true;
}

// f32 concat along dim 0 where src1 is a transposed view (nb10 != 4, nb11 == 4), e.g. the delta-net
// conv_input = concat(conv_states, transpose(qkv_mixed)). Threads split dim 1 (channels); each token's run
// of channels is loaded contiguously and written to the dst channel rows with a strided store.
// Only columns >= col0 are written (col0 > 0 when SSM_CONV reads the inputs directly and only the
// trailing columns are read back through views).
bool forward_concat_dim0_transposed_f32(ggml::spacemit::context & ctx, ggml_tensor * op, int64_t col0) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * dst  = op;

    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_get_op_params_i32(op, 0) != 0 || op->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 || nb0 != sizeof(float) || nb00 != sizeof(float) ||
        nb11 != sizeof(float) || nb10 == sizeof(float) || ne01 != ne1 || ne11 != ne1 || ne02 != ne2 ||
        ne12 != ne2 || ne03 != ne3 || ne13 != ne3) {
        return false;
    }

    const int64_t dc = (ne1 + (int64_t) ctx.nth - 1) / (int64_t) ctx.nth;
    const int64_t c0 = dc * (int64_t) ctx.ith;
    const int64_t c1 = std::min(c0 + dc, ne1);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const char * s0 = (const char *) src0->data + i2 * nb02 + i3 * nb03;
            const char * s1 = (const char *) src1->data + i2 * nb12 + i3 * nb13;
            char *       d  = (char *) op->data + i2 * nb2 + i3 * nb3;

            for (int64_t c = c0; c < c1;) {
                const size_t vl = __riscv_vsetvl_e32m4(c1 - c);
                for (int64_t i0 = col0; i0 < ne00; i0++) {
                    vfloat32m4_t v = __riscv_vlse32_v_f32m4((const float *) (s0 + i0 * nb00 + c * nb01), nb01, vl);
                    __riscv_vsse32_v_f32m4((float *) (d + i0 * nb0 + c * nb1), nb1, v, vl);
                }
                for (int64_t t = std::max<int64_t>(0, col0 - ne00); t < ne10; t++) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4((const float *) (s1 + t * nb10 + c * nb11), vl);
                    __riscv_vsse32_v_f32m4((float *) (d + (ne00 + t) * nb0 + c * nb1), nb1, v, vl);
                }
                c += vl;
            }
        }
    }
    return true;
}

// ssm_conv for d_conv == 4: dst[c, t] = sum_k x[c, t + k] * w[c, k]. Vectorized over channels with the
// 4-wide sliding window kept in registers (one strided load per token).
bool forward_ssm_conv_f32(ggml::spacemit::context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // conv_x {d_conv - 1 + n_t, d_inner, n_s}
    const ggml_tensor * src1 = dst->src[1];  // conv1d.weight {d_conv, d_inner}

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    if (nc != 4 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        dst->ne[0] != nr) {
        return false;
    }

    const ptrdiff_t xs = src0->nb[1];  // channel stride of x
    const ptrdiff_t ws = src1->nb[1];  // channel stride of w

    const int64_t dr  = (nr + (int64_t) ctx.nth - 1) / (int64_t) ctx.nth;
    const int64_t ir0 = dr * (int64_t) ctx.ith;
    const int64_t ir1 = std::min(ir0 + dr, nr);

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t c = ir0; c < ir1;) {
            const size_t vl = __riscv_vsetvl_e32m4(ir1 - c);

            const char * wp = (const char *) src1->data + c * ws;
            const vfloat32m4_t w0 = __riscv_vlse32_v_f32m4((const float *) wp + 0, ws, vl);
            const vfloat32m4_t w1 = __riscv_vlse32_v_f32m4((const float *) wp + 1, ws, vl);
            const vfloat32m4_t w2 = __riscv_vlse32_v_f32m4((const float *) wp + 2, ws, vl);
            const vfloat32m4_t w3 = __riscv_vlse32_v_f32m4((const float *) wp + 3, ws, vl);

            const float * xp = (const float *) ((const char *) src0->data + c * xs + i3 * src0->nb[2]);
            vfloat32m4_t  x0 = __riscv_vlse32_v_f32m4(xp + 0, xs, vl);
            vfloat32m4_t  x1 = __riscv_vlse32_v_f32m4(xp + 1, xs, vl);
            vfloat32m4_t  x2 = __riscv_vlse32_v_f32m4(xp + 2, xs, vl);

            for (int64_t t = 0; t < n_t; ++t) {
                const vfloat32m4_t x3 = __riscv_vlse32_v_f32m4(xp + t + 3, xs, vl);

                vfloat32m4_t acc = __riscv_vfmul_vv_f32m4(x0, w0, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x1, w1, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x2, w2, vl);
                acc              = __riscv_vfmacc_vv_f32m4(acc, x3, w3, vl);

                float * y = (float *) ((char *) dst->data + c * sizeof(float) + t * dst->nb[1] + i3 * dst->nb[2]);
                __riscv_vse32_v_f32m4(y, acc, vl);

                x0 = x1;
                x1 = x2;
                x2 = x3;
            }
            c += vl;
        }
    }
    return true;
}

template void forward_concat<int32_t>(ggml::spacemit::context & ctx, ggml_tensor * op);
template void forward_concat<int16_t>(ggml::spacemit::context & ctx, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
