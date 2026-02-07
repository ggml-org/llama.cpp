#ifndef HVX_TYPES_H
#define HVX_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#include <hexagon_types.h>

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

typedef struct {
    HVX_Vector v[2];
} HVX_Vector_x2;

typedef struct {
    HVX_Vector v[4];
} HVX_Vector_x4;

typedef struct {
    HVX_Vector v[8];
} HVX_Vector_x8;

#endif /* HVX_TYPES_H */
