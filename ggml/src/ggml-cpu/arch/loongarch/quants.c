#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include "../../ggml-cpu-quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(__loongarch_sx)

static __m128i lsx_packs_w(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_w(a, 15);
    tmp1 = __lsx_vsat_w(b, 15);
    return __lsx_vpickev_h(tmp1, tmp);
}

static __m128i lsx_packs_h(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_h(a, 7);
    tmp1 = __lsx_vsat_h(b, 7);
    return __lsx_vpickev_b(tmp1, tmp);
}

static __m128i lsx_packus_h(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_hu(a, 7);
    tmp1 = __lsx_vsat_hu(b, 7);
    return __lsx_vpickev_b(tmp1, tmp);
}

static __m128i lsx_maddubs_h(__m128i a, __m128i b) {
    __m128i tmp1, tmp2;
    tmp1 = __lsx_vmulwev_h_b(a, b);
    tmp2 = __lsx_vmulwod_h_b(a, b);
    return __lsx_vsadd_h(tmp1, tmp2);
}

static __m128i lsx_madd_h(__m128i a, __m128i b) {
    __m128i tmp1, tmp2;
    tmp1 = __lsx_vmulwev_w_h(a, b);
    tmp2 = __lsx_vmulwod_w_h(a, b);
    return __lsx_vadd_w(tmp1, tmp2);
}

static __m128i lsx_set_w(int32_t a, int32_t b, int32_t c, int32_t d) {
    v4i32 __ret = {d, c, b, a};
    return (__m128i)__ret;
}

static __m128i lsx_shuffle_b(__m128i a, __m128i b) {
    __m128i mask_f, zero, tmp0, tmp2, mask;
    int f = 0x8f;
    mask_f = __lsx_vreplgr2vr_b(f);
    zero = __lsx_vldi(0);
    tmp0 = __lsx_vand_v(b, mask_f); // get mask with low 4 bit and sign bits
    tmp0 = __lsx_vori_b(tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
    mask = __lsx_vsle_b(zero, tmp0); // if mask >= 0, set mask
    tmp2 = __lsx_vand_v(tmp0, mask); // maskout the in2 < ones
    return __lsx_vshuf_b(a, zero, tmp2);
}

static __m128i lsx_hadd_h(__m128i a, __m128i b) {
    __m128i tmp1 = __lsx_vpickev_h(b, a);
    __m128i tmp2 = __lsx_vpickod_h(b, a);
    return __lsx_vadd_h(tmp1, tmp2);
}

static __m128i lsx_hadd_w(__m128i a, __m128i b) {
    __m128i tmp1 = __lsx_vpickev_w(b, a);
    __m128i tmp2 = __lsx_vpickod_w(b, a);
    return __lsx_vadd_w(tmp1, tmp2);
}

static __m128 lsx_hadd_s(__m128 a, __m128 b) {
    __m128 tmp1 = (__m128)__lsx_vpickev_w((__m128i)b, (__m128i)a);
    __m128 tmp2 = (__m128)__lsx_vpickod_w((__m128i)b, (__m128i)a);

    return __lsx_vfadd_s(tmp1, tmp2);
}

static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =lsx_hadd_s(a, b);
    __m128 res_1 =lsx_hadd_s(c, d);
    __m128 res =lsx_hadd_s(res_0, res_1);
    res =lsx_hadd_s(res, res);
    res =lsx_hadd_s(res, res);

    return ((v4f32)res)[0];
}
#endif

#if defined(__loongarch_asx)

#ifdef __clang__
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
// Convert __m128i to __m256i
static inline __m256i ____m256i(__m128i in) {
    __m256i out = __lasx_xvldi(0);
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX"\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "+f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert two __m128i to __m256i
static inline __m256i lasx_set_q(__m128i inhi, __m128i inlo) {
    __m256i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".ifnc %[out], %[hi]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
        "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out), [hi] "+f" (inhi)
        : [lo] "f" (inlo)
    );
    return out;
}
// Convert __m256i low part to __m128i
static inline __m128i lasx_extracti128_lo(__m256i in) {
    __m128i out;
    __asm__ volatile (
        ".ifnc %[out], %[in]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    vori.b $vr\\i, $vr\\j, 0        \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert __m256i high part to __m128i
static inline __m128i lasx_extracti128_hi(__m256i in) {
    __m128i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}

static __m256i lasx_set_w(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0) {
    v8i32 __ret = {e0, e1, e2, e3, e4, e5, e6, e7};
    return (__m256i)__ret;
}

static __m256i lasx_set_d(int64_t a, int64_t b, int64_t c, int64_t d) {
    v4i64 __ret = {d, c, b, a};
    return (__m256i)__ret;
}

static __m256i lasx_insertf128( __m128i x, __m128i y) {
    return lasx_set_q(x, y);
}

static __m256i lasx_shuffle_b(__m256i a, __m256i b) {
    __m256i mask_f, zero, tmp0, tmp2, mask;
    int f = 0x8f;
    mask_f = __lasx_xvreplgr2vr_b(f);
    zero = __lasx_xvldi(0);
    tmp0 = __lasx_xvand_v(b, mask_f); // get mask with low 4 bit and sign bits
    tmp0 = __lasx_xvori_b(tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
    mask = __lasx_xvsle_b(zero, tmp0); // if mask >= 0, set mask
    tmp2 = __lasx_xvand_v(tmp0, mask); // maskout the in2 < ones
    return __lasx_xvshuf_b(a, zero, tmp2);
}

static __m256i lasx_extu8_16(__m128i a) {
    return __lasx_vext2xv_hu_bu(____m256i(a));
}

static __m256i lasx_ext8_16(__m128i a) {
    return __lasx_vext2xv_h_b(____m256i(a));
}

static __m256i lasx_ext16_32(__m128i a) {
    return __lasx_vext2xv_w_h(____m256i(a));
}

static __m128i lasx_extracti128( __m256i a, int pos) {
    __m128i ret;
    if( pos == 0)
    {
       ret = lasx_extracti128_lo(a);
    } else {
       ret = lasx_extracti128_hi(a);
    }
    return ret;
}

static __m128 lasx_extractf128( __m256 a, int pos) {
    __m128 ret;
    if( pos == 0)
    {
       ret = (__m128)lasx_extracti128_lo((__m256i)a);
    } else {
       ret = (__m128)lasx_extracti128_hi((__m256i)a);
    }
    return ret;
}

static __m256i lasx_maddubs_h(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_h_b(a, b);
    tmp2 = __lasx_xvmulwod_h_b(a, b);
    return __lasx_xvsadd_h(tmp1, tmp2);
}

static __m256i lasx_madd_h(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_w_h(a, b);
    tmp2 = __lasx_xvmulwod_w_h(a, b);
    return __lasx_xvadd_w(tmp1, tmp2);
}

static __m256i lasx_packs_w(__m256i a, __m256i b) {
    __m256i tmp, tmp1;
    tmp = __lasx_xvsat_w(a, 15);
    tmp1 = __lasx_xvsat_w(b, 15);
    return __lasx_xvpickev_h(tmp1, tmp);
}

static __m256i lasx_packs_h(__m256i a, __m256i b) {
    __m256i tmp, tmp1;
    tmp = __lasx_xvsat_h(a, 7);
    tmp1 = __lasx_xvsat_h(b, 7);
    return __lasx_xvpickev_b(tmp1, tmp);
}

static inline __m256i lasx_madd_h_b(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_h_b(a, b);
    tmp2 = __lasx_xvmulwod_h_b(a, b);
    return __lasx_xvadd_h(tmp1, tmp2);
}

static inline __m256i lasx_xvrepl128vei_h(__m256i a, const unsigned int b) {
    switch (b) {
        case 0: return __lasx_xvrepl128vei_h(a, 0);
        case 1: return __lasx_xvrepl128vei_h(a, 1);
        case 2: return __lasx_xvrepl128vei_h(a, 2);
        case 3: return __lasx_xvrepl128vei_h(a, 3);
        case 4: return __lasx_xvrepl128vei_h(a, 4);
        case 5: return __lasx_xvrepl128vei_h(a, 5);
        case 6: return __lasx_xvrepl128vei_h(a, 6);
        case 7: return __lasx_xvrepl128vei_h(a, 7);
        default: __builtin_unreachable();
    }
}

static inline __m256i lasx_xvandi_b_bit(__m256i a, const unsigned int b) {
    switch (b) {
        case 0: return __lasx_xvandi_b(a, 1 << 0);
        case 1: return __lasx_xvandi_b(a, 1 << 1);
        case 2: return __lasx_xvandi_b(a, 1 << 2);
        case 3: return __lasx_xvandi_b(a, 1 << 3);
        case 4: return __lasx_xvandi_b(a, 1 << 4);
        case 5: return __lasx_xvandi_b(a, 1 << 5);
        case 6: return __lasx_xvandi_b(a, 1 << 6);
        case 7: return __lasx_xvandi_b(a, 1 << 7);
        default: __builtin_unreachable();
    }
}

// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = __lsx_vsigncov_b(x, x);
    // Sign the values of the y vectors
    const __m128i sy = __lsx_vsigncov_b(x, y);
    // Perform multiplication and create 16-bit values
    const __m128i dot = lsx_maddubs_h(ax, sy);
    const __m128i ones = __lsx_vreplgr2vr_h(1);
    return lsx_madd_h(ones, dot);
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = lasx_extractf128(x, 1);
    res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
    res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
    res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
    return ((v4f32)res)[0];
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    __m128i ev = __lsx_vpickev_w(a, a);
    __m128i od = __lsx_vpickod_w(a, a);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {

    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = lasx_set_d(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);

    __m256i bytes = lasx_shuffle_b(__lasx_xvreplgr2vr_w(x32), shuf_mask);
    const __m256i bit_mask = __lasx_xvreplgr2vr_d(0x7fbfdfeff7fbfdfe);
    bytes = __lasx_xvor_v(bytes, bit_mask);
    return __lasx_xvseq_b(bytes, __lasx_xvreplgr2vr_d(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i lo = __lsx_vld((const __m128i *)rsi, 0);
    __m128i hi = __lsx_vsrli_h(lo, 4);
    return __lasx_xvandi_b(lasx_insertf128(hi, lo), 0xf);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    __m256i v = __lasx_xvpackod_h(x, x);
    __m256i summed_pairs = __lasx_xvaddwev_w_h(x, v);
    return __lasx_xvffint_s_w(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    // Perform multiplication and create 16-bit values
    const __m256i dot = lasx_maddubs_h(ax, sy);
    return sum_i16_pairs_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i dot = lasx_madd_h_b(x, y);
    return sum_i16_pairs_float(dot);
}

static inline __m128i packNibbles( __m256i bytes ) {
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i lowByte = __lasx_xvreplgr2vr_h(0xFF);
     __m256i high = __lasx_xvandn_v(lowByte, bytes);
    __m256i low = __lasx_xvand_v(lowByte, bytes);
    high = __lasx_xvsrli_h(high, 4);
    bytes = __lasx_xvor_v(low, high);
    // Compress uint16_t lanes into bytes
    __m128i *r0 = (__m128i *)&bytes;
    __m256i tmp_h128 = __lasx_xvpermi_q(bytes, bytes, 0x11);
    __m128i *r1 = (__m128i *)&tmp_h128;

    __m128i zero = __lsx_vldi(0);
    __m128i tmp, tmp2, tmp3;

    tmp = __lsx_vmax_h(zero, *r0);
    tmp2 = __lsx_vsat_hu(tmp, 7);

    tmp = __lsx_vmax_h(zero, *r1);
    tmp3 = __lsx_vsat_hu(tmp, 7);
    return  __lsx_vpickev_b(tmp3, tmp2);
}
#endif  //__loongarch_asx

void quantize_row_q8_0_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        __m256 v0 = (__m256)__lasx_xvld( x , 0);
        __m256 v1 = (__m256)__lasx_xvld( x , 32);
        __m256 v2 = (__m256)__lasx_xvld( x , 64);
        __m256 v3 = (__m256)__lasx_xvld( x , 96);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs , 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w( max4, 1 ), 0 ));
        const float max_scalar = ((v4f32)max4)[0];

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = (__m256)__lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128( i0, 0 );
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0);
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);

    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        __m256 v0 = (__m256)__lasx_xvld( x , 0 );
        __m256 v1 = (__m256)__lasx_xvld( x , 32 );
        __m256 v2 = (__m256)__lasx_xvld( x , 64 );
        __m256 v3 = (__m256)__lasx_xvld( x , 96 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs, 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vextrins_w((__m128i)tmp, (__m128i)max4, 0x10 ));
        const float max_scalar = ((v4f32)max4)[0];

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = __lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128(i0, 0);
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0 );
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = __lsx_vadd_w(__lsx_vadd_w(ni0, ni1), __lsx_vadd_w(ni2, ni3));
        const __m128i s1 = __lsx_vadd_w(__lsx_vadd_w(ni4, ni5), __lsx_vadd_w(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(__lsx_vadd_w(s0, s1)));

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};


//===================================== Dot products =================================

//
// Helper functions
//

#if defined(__loongarch_asx)
// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return __lasx_xvld((const __m256i*)k_shuffle + i, 0);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return __lasx_xvld((const __m256i*)k_shuffle + i, 0);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return __lsx_vld((const __m128i*)k_shuffle + i, 0);
}
#endif

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = __lasx_xvreplfr2vr_s( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = __lasx_xvreplgr2vr_b( 8 );
        qx = __lasx_xvsub_b( qx, off );

        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);

#elif defined(__loongarch_sx)
    // set constants
    const __m128i low_mask = __lsx_vreplgr2vr_b(0xF);
    const __m128i off = __lsx_vreplgr2vr_b(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = (__m128)__lsx_vldi(0);
    __m128 acc_1 = (__m128)__lsx_vldi(0);
    __m128 acc_2 = (__m128)__lsx_vldi(0);
    __m128 acc_3 = (__m128)__lsx_vldi(0);

    for (; ib + 1 < nb; ib += 2) {

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = (__m128)__lsx_vreplgr2vr_w( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = __lsx_vld((const __m128i *)x[ib].qs, 0);

        __m128i bx_0 = __lsx_vand_v(low_mask, tmp_0_1);
        __m128i by_0 = __lsx_vld((const __m128i *)y[ib].qs, 0);
        bx_0 = __lsx_vsub_b(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_0_1, 4));
        __m128i by_1 = __lsx_vld((const __m128i *)(y[ib].qs + 16), 0);
        bx_1 = __lsx_vsub_b(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        //_mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        //_mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = (__m128)__lsx_vreplgr2vr_w( GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = __lsx_vld((const __m128i *)x[ib + 1].qs, 0);

        __m128i bx_2 = __lsx_vand_v(low_mask, tmp_2_3);
        __m128i by_2 = __lsx_vld((const __m128i *)y[ib + 1].qs, 0);
        bx_2 = __lsx_vsub_b(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_2_3, 4));
        __m128i by_3 = __lsx_vld((const __m128i *)(y[ib + 1].qs + 16), 0);
        bx_3 = __lsx_vsub_b(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = __lsx_vffint_s_w(i32_0);
        __m128 p1 = __lsx_vffint_s_w(i32_1);
        __m128 p2 = __lsx_vffint_s_w(i32_2);
        __m128 p3 = __lsx_vffint_s_w(i32_3);

        // Apply the scale
        __m128 p0_d = __lsx_vfmul_s( d_0_1, p0 );
        __m128 p1_d = __lsx_vfmul_s( d_0_1, p1 );
        __m128 p2_d = __lsx_vfmul_s( d_2_3, p2 );
        __m128 p3_d = __lsx_vfmul_s( d_2_3, p3 );

        // Acummulate
        acc_0 = __lsx_vfadd_s(p0_d, acc_0);
        acc_1 = __lsx_vfadd_s(p1_d, acc_1);
        acc_2 = __lsx_vfadd_s(p2_d, acc_2);
        acc_3 = __lsx_vfadd_s(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);

#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

void ggml_vec_dot_q4_1_q8_1_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    float summs = 0;

    // Main loop
    for (; ib < nb; ++ib) {
        const float d0 = GGML_FP16_TO_FP32(x[ib].d);
        const float d1 = GGML_FP16_TO_FP32(y[ib].d);

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        const __m256 d0v = __lasx_xvreplfr2vr_s( d0 );
        const __m256 d1v = __lasx_xvreplfr2vr_s( d1 );

        // Compute combined scales
        const __m256 d0d1 = __lasx_xvfmul_s( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        const __m256i qy = __lasx_xvld( (const __m256i *)y[ib].qs, 0);

        const __m256 xy = mul_sum_us8_pairs_float(qx, qy);

        // Accumulate d0*d1*x*y
        acc = __lasx_xvfmadd_s( d0d1, xy, acc );
    }

    sumf = hsum_float_8(acc) + summs;

#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d)); //FIXME

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = __lasx_xvandn_v(bxhi, __lasx_xvreplgr2vr_b((char)0xF0));
        qx = __lasx_xvor_v(qx, bxhi);

        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s(d, q, acc);
    }

    sumf = hsum_float_8(acc);
    
#endif
    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    float summs = 0.0f;

    // Main loop
    for (; ib < nb; ++ib) {
        const __m256 dx = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d));

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = __lasx_xvand_v(bxhi, __lasx_xvreplgr2vr_b(0x10));
        qx = __lasx_xvor_v(qx, bxhi);

        const __m256 dy = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(y[ib].d));
        const __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_us8_pairs_float(qx, qy);

        acc = __lasx_xvfmadd_s(q, __lasx_xvfmul_s(dx, dy), acc);
    }

    sumf = hsum_float_8(acc) + summs;
    
#endif
    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = __lasx_xvld((const __m256i *)x[ib].qs, 0);
        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);

#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales128 = __lsx_vld((const __m128i*)x[i].scales, 0);
        const __m128i scales128 = __lsx_vandi_b(mins_and_scales128, 0xf);
        const __m256i mins = lasx_ext8_16(__lsx_vsrli_b(mins_and_scales128, 4));
        const __m256i prod = lasx_madd_h(mins, __lasx_xvld((const __m256i*)y[i].bsums, 0));

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(dmin), __lasx_xvffint_s_w(prod), acc);

        const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        const __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = __lasx_xvld((const __m256i*)q2, 0); q2 += 32;

            const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            const __m256i q2_0 = __lasx_xvandi_b(q2bits, 3);
            const __m256i q2_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 2), 3);
            const __m256i q2_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q2bits, 4), 3);
            const __m256i q2_3 = __lasx_xvsrli_b(q2bits, 6);

            __m256i p0 = lasx_madd_h_b(q2_0, q8_0);
            __m256i p1 = lasx_madd_h_b(q2_1, q8_1);
            __m256i p2 = lasx_madd_h_b(q2_2, q8_2);
            __m256i p3 = lasx_madd_h_b(q2_3, q8_3);

            p0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p0);
            p1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p1);
            p2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p2);
            p3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p3);

            p0 = __lasx_xvadd_w(p0, p1);
            p2 = __lasx_xvadd_w(p2, p3);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p0, p2));
        }

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);

    }

    *s = hsum_float_8(acc);

#else

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

void ggml_vec_dot_q3_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __loongarch_asx

    const __m128i m32 = __lsx_vreplgr2vr_b(32);

    __m256 acc = (__m256)__lasx_xvldi(0);

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = lsx_set_w(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = __lsx_vsub_b(scales128, m32);

        const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        const __m256i scales_shuffled = lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

        // high bit
        const __m256i hbits = __lasx_xvld((const __m256i*)x[i].hmask, 0);

        // integer accumulator
        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            const __m256i q3bits = __lasx_xvld((const __m256i*)q3, 0); q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = __lasx_xvandi_b(q3bits, 3);
            const __m256i q3l_1 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 2), 3);
            const __m256i q3l_2 = __lasx_xvandi_b(__lasx_xvsrli_b(q3bits, 4), 3);
            const __m256i q3l_3 = __lasx_xvsrli_b(q3bits, 6);
            const __m256i q3h_0 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 0), 0), 2);
            const __m256i q3h_1 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 1), 0), 2);
            const __m256i q3h_2 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 2), 0), 2);
            const __m256i q3h_3 = __lasx_xvslli_b(__lasx_xvseqi_b(lasx_xvandi_b_bit(hbits, 4 * j + 3), 0), 2);
            const __m256i q3_0 = __lasx_xvor_v(q3h_0, q3l_0);
            const __m256i q3_1 = __lasx_xvor_v(q3h_1, q3l_1);
            const __m256i q3_2 = __lasx_xvor_v(q3h_2, q3l_2);
            const __m256i q3_3 = __lasx_xvor_v(q3h_3, q3l_3);

            // load Q8 quants
            const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            __m256i p16_0 = lasx_madd_h_b(q8_0, q3_0);
            __m256i p16_1 = lasx_madd_h_b(q8_1, q3_1);
            __m256i p16_2 = lasx_madd_h_b(q8_2, q3_2);
            __m256i p16_3 = lasx_madd_h_b(q8_3, q3_3);

            // multiply with scales
            p16_0 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p16_0);
            p16_1 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p16_1);
            p16_2 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p16_2);
            p16_3 = lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p16_3);

            // accumulate
            p16_0 = __lasx_xvadd_w(p16_0, p16_1);
            p16_2 = __lasx_xvadd_w(p16_2, p16_3);
            sumi  = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_2));
        }
        // multiply with block scale and accumulate
        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else
    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}

void ggml_vec_dot_q4_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);
    __m128 acc_m = (__m128)__lsx_vldi(0);

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales128 = lsx_set_w(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i mins128 = __lsx_vexth_h_b(mins_and_scales128);
        const __m128i scales128 = __lsx_vsllwil_h_b(mins_and_scales128, 0);

        const __m256i q8sums = __lasx_xvld((const __m256i*)y[i].bsums, 0);
        const __m128i q8s = lsx_hadd_h(lasx_extracti128(q8sums, 0), lasx_extracti128(q8sums, 1));
        const __m128i prod = lsx_madd_h(mins128, q8s);
        acc_m = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(dmin), __lsx_vffint_s_w(prod), acc_m);

        const __m256i scales = lasx_insertf128(scales128, scales128);

        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = lasx_xvrepl128vei_h(scales, 2 * j + 0);
            const __m256i scale_h = lasx_xvrepl128vei_h(scales, 2 * j + 1);

            const __m256i q4bits = __lasx_xvld((const __m256i*)q4, 0); q4 += 32;
            const __m256i q4l = __lasx_xvandi_b(q4bits, 0xf);
            const __m256i q4h = __lasx_xvsrli_b(q4bits, 4);

            const __m256i q8l = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            __m256i p16l = lasx_madd_h_b(q4l, q8l);
            p16l = lasx_madd_h(scale_l, p16l);

            const __m256i q8h = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            __m256i p16h = lasx_madd_h_b(q4h, q8h);
            p16h = lasx_madd_h(scale_h, p16h);
            const __m256i sumj = __lasx_xvadd_w(p16l, p16h);

            sumi = __lasx_xvadd_w(sumi, sumj);
        }

        __m256 vd = __lasx_xvreplfr2vr_s(d);
        acc = __lasx_xvfmadd_s(vd, __lasx_xvffint_s_w(sumi), acc);

    }

    acc_m = __lsx_vfadd_s(acc_m, (__m128)__lsx_vpermi_w((__m128i)acc_m, (__m128i)acc_m, 0xee));
    __m128i tmp1 = __lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w((__m128i)acc_m, 1), 0);
    acc_m = __lsx_vfadd_s(acc_m, (__m128)tmp1);


    *s = hsum_float_8(acc) + ((v4f32)acc_m)[0];

#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}