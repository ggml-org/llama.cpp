
#ifndef __OCL_DEFS_H__
#define __OCL_DEFS_H__

#ifdef __OPENCL_C_VERSION__
// Device (OpenCL) Definitions
#    define OCL_KERNEL          kernel
#    define OCL_GLOBAL          global
#else
// Host (C++) Definitions
#    define OCL_KERNEL
#    define OCL_GLOBAL
#    define __kernel
#    define __global
#    define ulong cl_ulong
#endif

#endif  // __OCL_DEFS_H__
