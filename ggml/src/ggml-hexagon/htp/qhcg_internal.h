/**=============================================================================
@file
    hvx_internal.h

@brief
    Header file for HVX routines.

Copyright (c) 2020 Qualcomm Technologies Incorporated.
All Rights Reserved. Qualcomm Proprietary and Confidential.
=============================================================================**/

#ifndef _HVX_INTERNAL_H
#define _HVX_INTERNAL_H

#include <stddef.h> // size_t
#include <hexagon_types.h>

#define HVX_INLINE_ALWAYS inline __attribute__((unused,always_inline))

#ifndef LOG2VLEN
#define LOG2VLEN    7
#endif
#define VLEN        (1<<LOG2VLEN)    // HVX vector - number of int8_t elements
#define VLEN_SHORT  (1<<LOG2VLEN)>>1 // HVX vector - number of int16_t elements
#define VLEN_WORD   (1<<LOG2VLEN)>>2 // HVX vector - number of int32_t elements

typedef union
{
    HVX_VectorPair VV;
    struct
    {
        HVX_Vector lo;
        HVX_Vector hi;
    } V;
} HVX_DV;

static HVX_INLINE_ALWAYS void l2fetch(const void *p, uint32_t stride,
                                      uint32_t width, uint32_t height,
                                      uint32_t dir)
{
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

/* Return whether address is aligned. */

static HVX_INLINE_ALWAYS int32_t is_aligned(void *addr, uint32_t align)
{
    return ((size_t) addr & (align - 1)) == 0;
}

/* Return whether 'n' elements from vector are in the one chunk of 'chunk_size'. */

static HVX_INLINE_ALWAYS int32_t is_in_one_chunk(void *addr, uint32_t n,
                                                 uint32_t chunk_size)
{
    uint32_t left_off = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;
}

/*
 * This function stores the first n bytes from vector vin to address 'addr'.
 * n must be in range 1..128 and addr may have any alignment. Does one or
 * two masked stores.
 */

static HVX_INLINE_ALWAYS void vstu_variable(void *addr, uint32_t n,
                                            HVX_Vector vin)
{
    /* Rotate as needed. */
    vin = Q6_V_vlalign_VVR(vin, vin, (size_t) addr);

    uint32_t left_off = (size_t) addr & 127;
    uint32_t right_off = left_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((size_t) addr);
    HVX_VectorPred qr = Q6_Q_vsetq2_R(right_off);

    if (right_off > 128)
    {
        Q6_vmem_QRIV(qr, (HVX_Vector*) addr + 1, vin);
        /* all 1's */
        qr = Q6_Q_vcmp_eq_VbVb(vin, vin);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector*) addr, vin);
}

#endif /* _HVX_INTERNAL_H */
