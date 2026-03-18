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

#ifndef HMX_LOG2VLEN
#define HMX_LOG2VLEN 7
#endif
#define HMX_VLEN       (1 << HMX_LOG2VLEN)       // 128 bytes
#define HMX_VLEN_SHORT ((1 << HMX_LOG2VLEN) >> 1) // 64 half-words
#define HMX_VLEN_WORD  ((1 << HMX_LOG2VLEN) >> 2) // 32 words

#endif // HMX_HVX_INTERNAL_H
