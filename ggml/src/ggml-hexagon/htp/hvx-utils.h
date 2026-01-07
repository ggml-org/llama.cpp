#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include "ops-utils.h"

#include <stdbool.h>
#include <stdint.h>

#define SIZEOF_FP32 (4)
#define SIZEOF_FP16 (2)
#define VLEN        (128)
#define VLEN_FP32   (VLEN / SIZEOF_FP32)
#define VLEN_FP16   (VLEN / SIZEOF_FP16)

typedef union {
    HVX_Vector v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    float      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_VectorAlias;

/* Q6_Vsf_equals_Vw is only available on v73+.*/
#if __HVX_ARCH__ < 73
static inline HVX_Vector int32_to_qfloat(HVX_Vector const in)
{
    HVX_Vector const vzero = Q6_V_vzero();
    HVX_VectorPred is_zero = Q6_Q_vcmp_eq_VwVw(in, vzero);
    HVX_Vector lshift = Q6_Vw_vnormamt_Vw(in);
    HVX_Vector normalized = Q6_Vw_vasl_VwVw(in, lshift);
    HVX_Vector vexp = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7f + 30), lshift);
    HVX_Vector mant = Q6_V_vand_VV(Q6_V_vsplat_R(0xFFFFFF00), normalized);
    HVX_Vector ret = Q6_V_vmux_QVV(is_zero, vzero, Q6_Vw_vadd_VwVw(mant, vexp));
    return ret;
}

static inline HVX_Vector Q6_Vsf_equals_Vw(HVX_Vector const in)
{
    return Q6_Vsf_equals_Vqf32(int32_to_qfloat(in));
}
#endif

static inline HVX_Vector hvx_vec_splat_fp32(float v) {
    union {
        float    f;
        uint32_t i;
    } fp32 = { .f = v };

    return Q6_V_vsplat_R(fp32.i);
}

static inline HVX_Vector hvx_vec_splat_fp16(float v) {
    union {
        __fp16   f;
        uint16_t i;
    } fp16 = { .f = v };

    return Q6_Vh_vsplat_R(fp16.i);
}

static inline void hvx_vec_store_u(void * addr, uint32_t n, HVX_Vector v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (size_t) addr);

    uint32_t left_off  = (size_t) addr & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_Vector *) addr + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) addr, v);
}

static inline void hvx_vec_store_a(void * ptr, size_t n, HVX_Vector v) {
    assert((unsigned long) ptr % 128 == 0);

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) ptr);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(n);
    ql_not                = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) ptr, v);
}

static inline HVX_Vector hvx_vec_repl4(HVX_Vector v) {
    // vdelta control to replicate first 4 bytes across all elements
    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };

    HVX_Vector ctrl = *(HVX_Vector *) repl;
    return Q6_V_vdelta_VV(v, ctrl);
}

// copy n fp16 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_fp16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), v);
    }
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    HVX_Vector * restrict vsrc  = (HVX_Vector *) src;

    assert((unsigned long) src % 128 == 0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), v);
    }
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_fp16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_UVector * restrict vsrc = (HVX_UVector *) src;

    assert((unsigned long) dst % 128 == 0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), v);
    }
}

// copy n fp32 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_fp32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), v);
    }
}

// copy n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_fp32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    HVX_Vector * restrict vsrc  = (HVX_Vector *) src;

    assert((unsigned long) src % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), v);
    }
}

// copy n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_fp32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_UVector * restrict vsrc = (HVX_UVector *) src;

    assert((unsigned long) dst % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), v);
    }
}

// copy n fp32 elements : source is unaligned, destination unaligned
static inline void hvx_copy_fp32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    HVX_UVector * restrict vsrc = (HVX_UVector *) src;

    assert((unsigned long) dst % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        HVX_Vector v = vsrc[i];
        vdst[i]      = v;
    }

    if (nloe) {
        HVX_Vector v = vsrc[i];
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), v);
    }
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_fp16_fp32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst; // fp16
    HVX_UVector * restrict vsrc = (HVX_UVector *) src; // fp32

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        vdst[i] = Q6_Vh_vdeal_Vh(s_hf);
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf));
    }
}

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is unaligned
static inline void hvx_copy_fp16_fp32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst; // fp16
    HVX_Vector  * restrict vsrc = (HVX_Vector *)  src; // fp32

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        vdst[i] = Q6_Vh_vdeal_Vh(s_hf);
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf));
    }
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is aligned
static inline void hvx_copy_fp16_fp32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    HVX_Vector  * restrict vdst = (HVX_Vector *)  dst; // fp16
    HVX_UVector * restrict vsrc = (HVX_UVector *) src; // fp32

    const HVX_Vector zero = Q6_V_vsplat_R(0);

    uint32_t nvec = n / 64;
    uint32_t nloe = n % 64;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        vdst[i] = Q6_Vh_vdeal_Vh(s_hf);
    }

    if (nloe) {
        // Load y (fp32) and convert into fp16
        HVX_Vector s0_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+0], zero); // 32 elements
        HVX_Vector s1_qf = Q6_Vqf32_vsub_VsfVsf(vsrc[i*2+1], zero); // 32 elements
        HVX_Vector s_hf  = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(s1_qf, s0_qf));
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(__fp16), Q6_Vh_vdeal_Vh(s_hf));
    }
}

// bcast 1 fp32 element from source to n fp32 elements in destination : destination is aligned
static inline void hvx_bcast_fp32_a(uint8_t * restrict dst, float elem, uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;

    HVX_Vector velem = hvx_vec_splat_fp32(elem);

    assert((unsigned long) dst % 128 == 0);

    uint32_t nvec = n / 32;
    uint32_t nloe = n % 32;

    uint32_t i = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = velem;
    }

    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * sizeof(float), velem);
    }
}


/* Return whether 'n' elements from vector are in the one chunk of 'chunk_size'. */
static __attribute__((always_inline)) int32_t is_in_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;
}

static void hvx_vec_dump_fp16_n(char * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    const uint32_t n0 = n / 16;
    const uint32_t n1 = n % 16;
    int            i  = 0;
    for (; i < n0; i++) {
        htp_dump_fp16_line(pref, u.fp16 + (16 * i), 16);
    }
    if (n1) {
        htp_dump_fp16_line(pref, u.fp16 + (16 * i), n1);
    }
}

static void hvx_vec_dump_fp16(char * pref, HVX_Vector v) {
    hvx_vec_dump_fp16_n(pref, v, 64);
}

static void hvx_vec_dump_fp32_n(char * pref, HVX_Vector v, uint32_t n) {
    union {
        HVX_Vector v;
        float      d[32];
    } u = { .v = v };

    const uint32_t n0 = n / 16;
    const uint32_t n1 = n % 16;
    int            i  = 0;
    for (; i < n0; i++) {
        htp_dump_fp32_line(pref, u.d + (16 * i), 16);
    }
    if (n1) {
        htp_dump_fp32_line(pref, u.d + (16 * i), n1);
    }
}

static void hvx_vec_dump_fp32_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        float      d[32];
    } u = { .v = v };

    FARF(HIGH, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

static void hvx_vec_dump_fp32(char * pref, HVX_Vector v) {
    hvx_vec_dump_fp32_n(pref, v, 32);
}

static void hvx_vec_dump_int32(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    for (int i = 0; i < 32 / 16; i++) {
        htp_dump_int32_line(pref, u.d + (16 * i), 16);
    }
}

static void hvx_vec_dump_int32_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(HIGH, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

static void hvx_vec_dump_int8_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(HIGH, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

static void hvx_vec_dump_int8(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    for (int i = 0; i < 128 / 16; i++) {
        htp_dump_int8_line(pref, u.d + (16 * i), 16);
    }
}

static void hvx_vec_dump_uint8(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    for (int i = 0; i < 128 / 16; i++) {
        htp_dump_uint8_line(pref, u.d + (16 * i), 16);
    }
}

static bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    typedef union {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    for (int i = 0; i < n; i++) {
        if (u0.d[i] != u1.d[i]) {
            return false;
        }
    }

    return true;
}

static inline float hvx_vec_get_fp32(HVX_Vector v) {
    float __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    return x;
}

static inline HVX_Vector hvx_vec_int32_reduce_sum_n(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // int32

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);     // rotate right
        sum   = Q6_Vw_vadd_VwVw(sum_t, sum);  // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_int32_reduce_sum(HVX_Vector in) {
    return hvx_vec_int32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_qf32_reduce_sum_n(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum), width);  // rotate right
        sum   = Q6_Vqf32_vadd_Vqf32Vsf(sum, sum_t);             // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_qf32_reduce_sum(HVX_Vector in) {
    return hvx_vec_qf32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_fp32_reduce_sum_n(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);                               // rotate right
        sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, sum_t));  // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_fp32_reduce_sum(HVX_Vector in) {
    return hvx_vec_fp32_reduce_sum_n(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_max_fp16(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_fp16(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max_fp32(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_fp32(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_abs_fp16(HVX_Vector v) {
    // abs by clearing the fp16 sign bit
    HVX_Vector mask = Q6_Vh_vsplat_R(0x7fff);
    return Q6_V_vand_VV(v, mask);
}

static inline HVX_Vector hvx_vec_neg_fp16(HVX_Vector v) {
    // neg by setting the fp16 sign bit
    HVX_Vector mask = Q6_Vh_vsplat_R(0x8000);
    return Q6_V_vxor_VV(v, mask);
}

static inline HVX_Vector hvx_vec_abs_fp32(HVX_Vector v) {
    // abs by clearing the fp32 sign bit
    HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
    return Q6_V_vand_VV(v, mask);
}

static inline HVX_Vector hvx_vec_neg_fp32(HVX_Vector v) {
#if __HVX_ARCH__ > 75
    return Q6_Vsf_vfneg_Vsf(v);
#else
    // neg by setting the fp32 sign bit
    HVX_Vector mask = Q6_V_vsplat_R(0x80000000);
    return Q6_V_vxor_VV(v, mask);
#endif  // __HVX_ARCH__ > 75
}

// ====================================================
// FUNCTION: 1/(x+1)     y(0) = 1,  y(0.5) = 0.6667, y(1) = 0.5
// Order:3; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 14
// Peak Error: 1.1295e-04  Rms Error: 2.8410e-05   Mean Error: 1.1370e-05
//      32769  -32706   31252  -10589
//      32590  -30635   22793   -4493
//      32066  -27505   16481   -2348
//      31205  -24054   11849   -1306

static inline HVX_Vector hvx_vec_recip_xp1_O3_unsigned(HVX_Vector vx) {
    // input is 0..0xffff representing 0.0  .. 1.0
    HVX_Vector p;
    p = Q6_Vh_vlut4_VuhPh(vx, 0xFAE6F6D4EE73D6A3ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x2E49406159097A14ull);
    p = Q6_Vh_vmps_VhVhVuhPuh_sat(p, vx, 0x5DF66B7177AB7FC2ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x79E57D427F4E8001ull);
    return p;  // signed result, 14 fractional bits
}

// Find reciprocal of fp16.
// (1) first, convert to fp32, multiplying by 1.0; this is done to
//    handle denormals. Ignoring sign and zero, result should be at
//    least 5.9604645e-08 (32-bit code 0x33800000) and at most 131008 (0x47ffe000)
//    (exponent in range [103,143])
// (2) extract the mantissa into 16-bit unsigned; find reciprocal using a fitted poly
// (3) put this, along with '253-exp' (exp from (1)) together to make an qf32
// (4) convert that to fp16
// (5) put sign back in. Also, if the original value (w/o sign) was <0x81, replace
//     the result with the max value.
static inline HVX_Vector hvx_vec_inverse_fp16(HVX_Vector vals) {
    HVX_Vector     em_mask  = Q6_Vh_vsplat_R(0x7FFF);
    HVX_Vector     avals    = Q6_V_vand_VV(vals, em_mask);
    HVX_VectorPred is_neg   = Q6_Q_vcmp_gt_VhVh(avals, vals);
    // is too small to 1/x ? for 'standard' fp16, this would be 0x101
    HVX_VectorPred is_small = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0x101), avals);

    HVX_VectorPair to_qf32  = Q6_Wqf32_vmpy_VhfVhf(avals, Q6_Vh_vsplat_R(0x3C00));  // *1.0
    HVX_Vector     to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(to_qf32));
    HVX_Vector     to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(to_qf32));

    // bits 22..13 contain the mantissa now (w/o hidden bit); move to bit 14..5 of a 16-bit vector
    HVX_Vector mant_u16 = Q6_Vh_vshuffo_VhVh(Q6_Vw_vasl_VwR(to_f32_1, 9), Q6_Vw_vasl_VwR(to_f32_0, 9));
    // likewise extract the upper 16 from each, containing the exponents in range 103..142
    HVX_Vector exp_u16  = Q6_Vh_vshuffo_VhVh(to_f32_1, to_f32_0);
    //Get exponent in IEEE 32-bit representation
    exp_u16             = Q6_Vuh_vlsr_VuhR(exp_u16, 7);

    // so, mant_u16 contains an unbiased mantissa in upper 10 bits of each u16 lane
    // We can consider it to be x-1.0, with 16 fractional bits, where 'x' is in range [1.0,2.0)
    // Use poly to transform to 1/x, with 14 fractional bits
    //
    HVX_Vector rm = hvx_vec_recip_xp1_O3_unsigned(mant_u16);

    HVX_Vector vcl0 = Q6_Vuh_vcl0_Vuh(rm);  //count leading zeros

    // Get mantissa for 16-bit represenation
    HVX_Vector mant_recip = Q6_V_vand_VV(Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhVh(rm, vcl0), 5), Q6_Vh_vsplat_R(0x03FF));

    //Compute Reciprocal Exponent
    HVX_Vector exp_recip =
        Q6_Vh_vsub_VhVh(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(254), exp_u16), Q6_Vh_vsub_VhVh(vcl0, Q6_Vh_vsplat_R(1)));
    //Convert it for 16-bit representation
    exp_recip = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsub_VhVh(exp_recip, Q6_Vh_vsplat_R(127)), Q6_Vh_vsplat_R(15));
    exp_recip = Q6_Vh_vasl_VhR(exp_recip, 10);

    //Merge exponent and mantissa for reciprocal
    HVX_Vector recip = Q6_V_vor_VV(exp_recip, mant_recip);
    // map 'small' inputs to standard largest value 0x7bff
    recip            = Q6_V_vmux_QVV(is_small, Q6_Vh_vsplat_R(0x7bff), recip);
    // add sign back
    recip            = Q6_V_vandor_VQR(recip, is_neg, 0x80008000);
    return recip;
}

#define IEEE_VSF_EXPLEN   (8)
#define IEEE_VSF_EXPBIAS  (127)
#define IEEE_VSF_EXPMASK  (0xFF)
#define IEEE_VSF_MANTLEN  (23)
#define IEEE_VSF_MANTMASK (0x7FFFFF)
#define IEEE_VSF_MIMPMASK (0x800000)

static inline HVX_Vector hvx_vec_truncate_fp32(HVX_Vector in_vec) {
    HVX_Vector mask_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v = Q6_V_vzero();

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

    HVX_Vector expval_v = in_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_Vector rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_Vector mant_v = in_vec & mask_mant_v;                  // obtain mantissa
    HVX_Vector vout   = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_Vector neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    return (vout);
}

static inline HVX_Vector hvx_vec_floor_fp32(HVX_Vector in_vec) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

    HVX_Vector expval_v = in_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, in_vec, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, in_vec);

    // if expval < 0 (q_negexp)         // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval                // fraction bits to mask off
    //    vout = ~(mask)                // apply mask to remove fraction
    //    if (qneg)                     // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                             // already an integer
    //    ;                             // no change

    // compute floor
    mask_mant_v >>= expval_v;
    HVX_Vector neg_addin_v    = mask_impl_v >> expval_v;
    HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(in_vec, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, in_vec);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(in_vec, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = in_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, in_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    return vout;
}

static inline HVX_Vector hvx_vec_i16_from_hf_rnd_sat(HVX_Vector vin) {
    // This looks complicated.
    // Ideally should just be Q6_Vh_equals_Vhf(vin)
    // but that instruction does not do proper rounding.

    // convert to qf32, multiplying by 1.0 in the process.
    HVX_VectorPair v32 = Q6_Wqf32_vmpy_VhfVhf(vin, Q6_Vh_vsplat_R(0x3C00));

    // 'in-range' values are +/32752.
    // add 192K to it, convert to sf
    HVX_Vector v192K = Q6_V_vsplat_R(0x48400000);
    HVX_Vector vsf_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K));
    HVX_Vector vsf_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(v32), v192K));

    // for in-range cases, result is {163858... 229360} so the exponent is always 144.
    // if we extract bits 21..0 as a signed quantity, and round 6 bits off, that will be the answer.
    // Start by <<10 to get the final 'sign' bit in bit 15...
    vsf_0 = Q6_Vw_vasl_VwR(vsf_0, 10);
    vsf_1 = Q6_Vw_vasl_VwR(vsf_1, 10);

    // now round down to 16
    return Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
}

static inline HVX_Vector hvx_vec_inverse_fp32(HVX_Vector v_sf) {
    HVX_Vector inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    HVX_Vector two_sf       = hvx_vec_splat_fp32(2.0);

    // First approximation
    HVX_Vector i_sf = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);

    HVX_Vector r_qf;

    // Refine
    r_qf = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));

    return Q6_Vsf_equals_Vqf32(r_qf);
}

#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

static inline HVX_Vector hvx_vec_fast_sigmoid_fp32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_fp32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_fp32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

#define EXP_COEFF_5 (0x39506967)  // 0.000198757 = 1/(7!)
#define EXP_COEFF_4 (0x3AB743CE)  // 0.0013982   = 1/(6!)
#define EXP_COEFF_3 (0x3C088908)  // 0.00833345  = 1/(5!)
#define EXP_COEFF_2 (0x3D2AA9C1)  // 0.416658    = 1/(4!)
#define EXP_COEFF_1 (0x3E2AAAAA)  // 0.16666667  = 1/(3!)
#define EXP_COEFF_0 (0x3F000000)  // 0.5         = 1/(2!)
#define EXP_LOGN2   (0x3F317218)  // ln(2)   = 0.6931471805
#define EXP_LOG2E   (0x3FB8AA3B)  // log2(e) = 1/ln(2) = 1.4426950408
#define EXP_ONE     (0x3f800000)  // 1.0
#define EXP_RANGE_R (0x41a00000)  // 20.0
#define EXP_RANGE_L (0xc1a00000)  // -20.0

static inline HVX_Vector hvx_vec_exp_fp32(HVX_Vector in_vec) {
    HVX_Vector z_qf32_v;
    HVX_Vector x_v;
    HVX_Vector x_qf32_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector epsilon_v;
    HVX_Vector log2e = Q6_V_vsplat_R(EXP_LOG2E);
    HVX_Vector logn2 = Q6_V_vsplat_R(EXP_LOGN2);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // exp(x) is approximated as follows:
    //   f = floor(x/ln(2)) = floor(x*log2(e))
    //   epsilon = x - f*ln(2)
    //   exp(x) = exp(epsilon+f*ln(2))
    //          = exp(epsilon)*exp(f*ln(2))
    //          = exp(epsilon)*2^f
    //
    //   Since epsilon is close to zero, it can be approximated with its Taylor series:
    //            exp(x) ~= 1+x+x^2/2!+x^3/3!+...+x^n/n!+...
    //   Preserving the first eight elements, we get:
    //            exp(x) ~= 1+x+e0*x^2+e1*x^3+e2*x^4+e3*x^5+e4*x^6+e5*x^7
    //                   =  1+x+(E0+(E1+(E2+(E3+(E4+E5*x)*x)*x)*x)*x)*x^2

    HVX_Vector temp_v = in_vec;

    // Clamp inputs to (-20.0, 20.0)
    HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, Q6_V_vsplat_R(EXP_RANGE_R));
    HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(EXP_RANGE_L), in_vec);

    in_vec = Q6_V_vmux_QVV(pred_cap_right, Q6_V_vsplat_R(EXP_RANGE_R), temp_v);
    in_vec = Q6_V_vmux_QVV(pred_cap_left, Q6_V_vsplat_R(EXP_RANGE_L), temp_v);

    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, in_vec);
    epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = hvx_vec_floor_fp32(epsilon_v);
    k_v = hvx_vec_truncate_fp32(f_v);

    x_qf32_v = Q6_Vqf32_vadd_VsfVsf(in_vec, zero_v);

    //  x = x - f_v * logn2;
    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
    x_qf32_v  = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
    // normalize before every QFloat's vmpy
    x_qf32_v  = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);

    // z = x * x;
    z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
    z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

    x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

    // y = E4 + E5 * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_5);
    y_v     = Q6_Vqf32_vmpy_VsfVsf(E_const, x_v);
    E_const = Q6_V_vsplat_R(EXP_COEFF_4);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E3 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_3);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E2 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_2);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E1 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_1);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E0 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_0);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = x + y * z;
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = y + 1.0;
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(EXP_ONE));

    // insert exponents
    //        y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent

    y_v = Q6_Vsf_equals_Vqf32(y_v);

    // add k_v to the exponent of y_v
    HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

    y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, IEEE_VSF_MANTLEN + 1);
    y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    // exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, IEEE_VSF_MANTLEN);

    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

    return y_v;
}

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

static inline HVX_Vector hvx_vec_rsqrt_fp32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>2)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, temp;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    return Q6_Vsf_equals_Vqf32(temp);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_fp32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_fp32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_Vector hvx_vec_tanh_fp32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_fp32(2.0f);
    HVX_Vector one = hvx_vec_splat_fp32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1
    HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_fp32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    return Q6_Vsf_equals_Vqf32(res);
}

static inline void hvx_fast_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems) {
    int step_of_1 = num_elems >> 5;
    int remaining = num_elems - step_of_1 * VLEN_FP32;

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1

    const HVX_Vector one     = hvx_vec_splat_fp32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    const HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        v_dst[i] = hvx_vec_fast_sigmoid_fp32_guard(v_src[i], one, max_exp, min_exp);
    }

    if (remaining > 0) {
        const float * srcf = ((const float *) src) + step_of_1* VLEN_FP32;
        float *       dstf = (float *) dst + step_of_1*VLEN_FP32;

        HVX_Vector in  = *(HVX_UVector *) srcf;
        HVX_Vector out = hvx_vec_fast_sigmoid_fp32_guard(in, one, max_exp, min_exp);
        hvx_vec_store_u((void *) dstf, remaining * SIZEOF_FP32, out);
    }
}

static inline void hvx_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems){
    int step_of_1 = num_elems >> 5;  // divby 32, because 32 float = 128 bytes per HVX vector
    int leftover = num_elems - (step_of_1 * VLEN_FP32);

    int32_t leftover_size = leftover * sizeof(float);

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1

    const HVX_Vector one     = hvx_vec_splat_fp32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    const HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    const float *input = (float *)src;
    float *output = (float *)dst;

    HVX_Vector *  input_v_ptr  = (HVX_Vector *) input;
    HVX_UVector * output_v_ptr = (HVX_UVector *) output;

    HVX_Vector slinep;
    HVX_Vector slinec;
    HVX_Vector sline;

    slinep = *input_v_ptr++;
    #pragma unroll(4)
    for (int i = step_of_1 - 1; i > 0; i--) {
        slinec                              = *input_v_ptr++;
        sline                               = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        *((HVX_UVector *) (output_v_ptr++)) = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        /* Prepare slinep for next iteration */
        slinep                              = slinec;
    }

    if (step_of_1 > 0) {
        slinec = htp_is_aligned(input_v_ptr, 128) && leftover == 0 ? slinep : *input_v_ptr++;
        sline  = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        *((HVX_UVector *) (output_v_ptr++)) = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        ;

        slinep = slinec;
    }
    if (leftover > 0) {
        slinec = (is_in_one_chunk(input_v_ptr, leftover_size, 128) ? slinep : *input_v_ptr++);

        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

        HVX_Vector sout = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        hvx_vec_store_u(output_v_ptr, leftover_size, sout);
    }
}

static inline void hvx_scale_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    int nvec = n / VLEN_FP32;
    int nloe = n % VLEN_FP32;

    HVX_Vector vs = hvx_vec_splat_fp32(scale);

    HVX_Vector * vsrc = (HVX_Vector *) src;
    HVX_Vector * vdst = (HVX_Vector *) dst;

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; ++i) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);
        vdst[i]      = Q6_Vsf_equals_Vqf32(v);
    }

    if (nloe) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);
        hvx_vec_store_u((void *) &vdst[i], nloe * 4, Q6_Vsf_equals_Vqf32(v));
    }
}

static inline void hvx_scale_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    int nvec = n / VLEN_FP32;
    int nloe = n % VLEN_FP32;

    HVX_Vector vs = hvx_vec_splat_fp32(scale);

    HVX_UVector * vsrc = (HVX_UVector *) src;
    HVX_UVector * vdst = (HVX_UVector *) dst;

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; ++i) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);
        vdst[i]      = Q6_Vsf_equals_Vqf32(v);
    }

    if (nloe) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);
        hvx_vec_store_u((void *) &vdst[i], nloe * 4, Q6_Vsf_equals_Vqf32(v));
    }
}

static inline void hvx_scale_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    if (htp_is_aligned((void *) src, VLEN) && htp_is_aligned((void *) dst, VLEN)) {
        hvx_scale_f32_aa(dst, src, n, scale);
    } else {
        hvx_scale_f32_uu(dst, src, n, scale);
    }
}

static inline void hvx_scale_offset_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    int nvec = n / VLEN_FP32;
    int nloe = n % VLEN_FP32;

    HVX_Vector vs = hvx_vec_splat_fp32(scale);
    HVX_Vector vo = hvx_vec_splat_fp32(offset);

    HVX_Vector * vsrc = (HVX_Vector *) src;
    HVX_Vector * vdst = (HVX_Vector *) dst;

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; ++i) {
        HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo);
        vdst[i] = Q6_Vsf_equals_Vqf32(v);
    }

    if (nloe) {
        HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo);
        hvx_vec_store_u((void *) &vdst[i], nloe * 4, Q6_Vsf_equals_Vqf32(v));
    }
}

static inline void hvx_scale_offset_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    int nvec = n / VLEN_FP32;
    int nloe = n % VLEN_FP32;

    HVX_Vector vs = hvx_vec_splat_fp32(scale);
    HVX_Vector vo = hvx_vec_splat_fp32(offset);

    HVX_UVector * vsrc = (HVX_UVector *) src;
    HVX_UVector * vdst = (HVX_UVector *) dst;

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; ++i) {
        HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo);
        vdst[i] = Q6_Vsf_equals_Vqf32(v);
    }

    if (nloe) {
        HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo);
        hvx_vec_store_u((void *) &vdst[i], nloe * 4, Q6_Vsf_equals_Vqf32(v));
    }
}

static inline void hvx_scale_offset_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (htp_is_aligned((void *) src, VLEN) && htp_is_aligned((void *) dst, VLEN)) {
        hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
    } else {
        hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
    }
}

float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_mul_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_mul_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_mul_mul_f32_opt(const uint8_t * restrict src0,
                          const uint8_t * restrict src1,
                          const uint8_t * restrict src2,
                          uint8_t * restrict dst,
                          const int num_elems);
void  hvx_mul_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_add_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_add_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_add_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_sub_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_sub_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_sub_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_inverse_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems);
void  hvx_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems);
void  hvx_exp_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems, bool negate);
float hvx_self_max_f32(const uint8_t * restrict src, const int num_elems);
float hvx_self_sum_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_min_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_clamp_scalar_f32(const uint8_t * restrict src,
                           const float limit_left,
                           const float limit_right,
                           uint8_t * restrict dst,
                           const int num_elems);

#endif /* HVX_UTILS_H */
