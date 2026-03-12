#ifndef HVX_DUMP_H
#define HVX_DUMP_H

#include <HAP_farf.h>

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils.h"
#include "hvx-types.h"

static void hvx_vec_dump_f16_n(char * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    const uint32_t n0 = n / 16;
    const uint32_t n1 = n % 16;
    int            i  = 0;
    for (; i < n0; i++) {
        hex_dump_f16_line(pref, u.fp16 + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f16_line(pref, u.fp16 + (16 * i), n1);
    }
}

static void hvx_vec_dump_f16(char * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

static void hvx_vec_dump_f32_n(char * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    const uint32_t n0 = n / 16;
    const uint32_t n1 = n % 16;
    int            i  = 0;
    for (; i < n0; i++) {
        hex_dump_f32_line(pref, u.fp32 + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f32_line(pref, u.fp32 + (16 * i), n1);
    }
}

static void hvx_vec_dump_f32_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        float      d[32];
    } u = { .v = v };

    FARF(HIGH, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

static void hvx_vec_dump_f32(char * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

static void hvx_vec_dump_int32(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    for (int i = 0; i < 32 / 16; i++) {
        hex_dump_int32_line(pref, u.d + (16 * i), 16);
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
        hex_dump_int8_line(pref, u.d + (16 * i), 16);
    }
}

static void hvx_vec_dump_uint8(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    for (int i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_line(pref, u.d + (16 * i), 16);
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

#endif /* HVX_DUMP_H */
