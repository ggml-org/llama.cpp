
#ifndef __KERNELS_DIV_H__
#define __KERNELS_DIV_H__

#include "ocl_defs.h"

OCL_KERNEL void kernel_div(OCL_GLOBAL char * src0,
                           ulong             offset0,
                           OCL_GLOBAL char * src1,
                           ulong             offset1,
                           OCL_GLOBAL char * dst,
                           ulong             offsetd,
                           ulong             nb00,
                           ulong             nb01,
                           ulong             nb02,
                           ulong             nb03,
                           int               ne10,
                           int               ne11,
                           int               ne12,
                           int               ne13,
                           ulong             nb10,
                           ulong             nb11,
                           ulong             nb12,
                           ulong             nb13,
                           int               ne0,
                           ulong             nb0,
                           ulong             nb1,
                           ulong             nb2,
                           ulong             nb3);

#endif  // __KERNELS_DIV_H__
