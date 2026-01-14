
#ifndef __OCL_DEFS_H__
#define __OCL_DEFS_H__

#ifdef __OPENCL_C_VERSION__
// Device (OpenCL) Definitions
#else
// Host (C++) Definitions
#    define kernel
#    define global
#    define __kernel
#    define __global
#    define ulong cl_ulong
#endif

#endif  // __OCL_DEFS_H__
