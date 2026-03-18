// HVX internal helpers used by HMX infrastructure (conversion, matmul, etc.).
// Ported from htp-ops-lib/include/dsp/hvx_internal.h. (https://github.com/haozixu/htp-ops-lib)
//
// Provides HVX_DV union, vmem macro, VLEN constants, and utility functions
// that are prerequisites for hmx-hvx-convert.h and hmx-matmul-ops.c.

#ifndef HMX_HVX_INTERNAL_H
#define HMX_HVX_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>

#define hmx_vmem(A)   *((HVX_Vector *)(A))
#define hmx_vmemu(A)  *((HVX_UVector *)(A))

#define HMX_HVX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

#ifndef HMX_LOG2VLEN
#define HMX_LOG2VLEN 7
#endif
#define HMX_VLEN       (1 << HMX_LOG2VLEN)       // 128 bytes
#define HMX_VLEN_SHORT ((1 << HMX_LOG2VLEN) >> 1) // 64 half-words
#define HMX_VLEN_WORD  ((1 << HMX_LOG2VLEN) >> 2) // 32 words

typedef union {
    HVX_VectorPair VV;
    struct {
        HVX_Vector lo;
        HVX_Vector hi;
    } V;
} hmx_HVX_DV;

static HMX_HVX_INLINE_ALWAYS int32_t hmx_is_aligned(const void *addr, uint32_t align) {
    return ((size_t)addr & (align - 1)) == 0;
}

static HMX_HVX_INLINE_ALWAYS uint16_t hmx_fp16_to_bits(__fp16 *x) {
    union { __fp16 f; uint16_t i; } fp16 = { .f = *x };
    return fp16.i;
}

#endif // HMX_HVX_INTERNAL_H
