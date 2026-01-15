
#ifndef __OCL_DEFS_H__
#define __OCL_DEFS_H__

#ifdef __OPENCL_C_VERSION__
// Device (OpenCL) Definitions
#    define OCL_KERNEL          kernel
#    define OCL_GLOBAL          global
#    define ocl_global_char_ptr global char *
#else
// Host (C++) Definitions
#    define OCL_KERNEL
#    define OCL_GLOBAL
#    define ocl_global_char_ptr cl_mem
#    define __kernel
#    define __global
#    define ulong cl_ulong
#endif

#endif  // __OCL_DEFS_H__
