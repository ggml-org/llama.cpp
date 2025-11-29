/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_KERNEL_MACRO_DEFS_H
#define BLIS_KERNEL_MACRO_DEFS_H


// -- Define default threading parameters --------------------------------------

// -- Conventional (large code path) values --

// These BLIS_THREAD_RATIO_? macros distort the amount of work in the m and n
// dimensions for the purposes of factorizing the total number of threads into
// ways of parallelism in the ic and jc loops. See bli_rntm.c to see how these
// macros are used.
#ifndef BLIS_THREAD_RATIO_M
#define BLIS_THREAD_RATIO_M     1
#endif

#ifndef BLIS_THREAD_RATIO_N
#define BLIS_THREAD_RATIO_N     1
#endif

// These BLIS_THREAD_MAX_?R macros place a ceiling on the maximum amount of
// parallelism allowed when performing automatic factorization. See bli_rntm.c
// to see how these macros are used.
#ifndef BLIS_THREAD_MAX_IR
#define BLIS_THREAD_MAX_IR      1
#endif

#ifndef BLIS_THREAD_MAX_JR
#define BLIS_THREAD_MAX_JR      4
#endif

#if 0
// -- Skinny/small possibly-unpacked (sup code path) values --

#ifndef BLIS_THREAD_SUP_RATIO_M
#define BLIS_THREAD_SUP_RATIO_M   1
#endif

#ifndef BLIS_THREAD_SUP_RATIO_N
#define BLIS_THREAD_SUP_RATIO_N   2
#endif

#ifndef BLIS_THREAD_SUP_MAX_IR
#define BLIS_THREAD_SUP_MAX_IR    1
#endif

#ifndef BLIS_THREAD_SUP_MAX_JR
#define BLIS_THREAD_SUP_MAX_JR    8
#endif
#endif


// -- Memory allocation --------------------------------------------------------

// hbwmalloc.h provides hbw_malloc() and hbw_free() on systems with
// libmemkind. But disable use of libmemkind if BLIS_DISABLE_MEMKIND
// was explicitly defined.
#ifdef BLIS_DISABLE_MEMKIND
  #undef BLIS_ENABLE_MEMKIND
#endif
#ifdef BLIS_ENABLE_MEMKIND
  #include <hbwmalloc.h>
#endif

// Memory allocation functions. These macros define the three types of
// malloc()-style functions, and their free() counterparts: one for each
// type of memory to be allocated.
// NOTE: ANY ALTERNATIVE TO malloc()/free() USED FOR ANY OF THE FOLLOWING
// THREE PAIRS OF MACROS MUST USE THE SAME FUNCTION PROTOTYPE AS malloc()
// and free():
//
//   void* malloc( size_t size );
//   void  free( void* p );
//

// This allocation function is called to allocate memory for blocks within
// BLIS's internal memory pools.
#ifndef BLIS_MALLOC_POOL
  // If use of libmemkind was enabled at configure-time, the default
  // memory allocation function for memory pools should be hbw_malloc()
  // instead of malloc().
  #ifdef  BLIS_ENABLE_MEMKIND
  #define BLIS_MALLOC_POOL               hbw_malloc
  #else
  #define BLIS_MALLOC_POOL               malloc
  #endif
#endif

#ifndef BLIS_FREE_POOL
  // If use of libmemkind was enabled at configure-time, the default
  // memory deallocation function for memory pools should be hbw_free()
  // instead of free().
  #ifdef  BLIS_ENABLE_MEMKIND
  #define BLIS_FREE_POOL                 hbw_free
  #else
  #define BLIS_FREE_POOL                 free
  #endif
#endif

// This allocation function is called to allocate memory for internally-
// used objects and structures, such as control tree nodes.
#ifndef BLIS_MALLOC_INTL
#define BLIS_MALLOC_INTL                 malloc
#endif

#ifndef BLIS_FREE_INTL
#define BLIS_FREE_INTL                   free
#endif

// This allocation function is called to allocate memory for objects
// created by user-level API functions, such as bli_obj_create().
#ifndef BLIS_MALLOC_USER
#define BLIS_MALLOC_USER                 malloc
#endif

#ifndef BLIS_FREE_USER
#define BLIS_FREE_USER                   free
#endif


// -- Other system-related definitions -----------------------------------------

// Size of a virtual memory page. This is used to align blocks within the
// memory pools.
#ifndef BLIS_PAGE_SIZE
#define BLIS_PAGE_SIZE                   4096
#endif

// The maximum number of named SIMD vector registers available for use.
// When configuring with umbrella configuration families, this should be
// set to the maximum number of registers across all sub-configurations in
// the family.
#ifndef BLIS_SIMD_MAX_NUM_REGISTERS
#define BLIS_SIMD_MAX_NUM_REGISTERS      32
#endif

// The maximum size (in bytes) of each SIMD vector.
// When configuring with umbrella configuration families, this should be
// set to the maximum SIMD size across all sub-configurations in the family.
#ifndef BLIS_SIMD_MAX_SIZE
#define BLIS_SIMD_MAX_SIZE               64
#endif

// Alignment size (in bytes) needed by the instruction set for aligned
// SIMD/vector instructions.
#ifndef BLIS_SIMD_ALIGN_SIZE
#define BLIS_SIMD_ALIGN_SIZE             BLIS_SIMD_MAX_SIZE
#endif

// The maximum size in bytes of local stack buffers within macro-kernel
// functions. These buffers are usually used to store a temporary copy
// of a single microtile. The reason we multiply by 4 is to handle induced
// methods, where we use real domain register blocksizes in units of
// complex elements, as well as mixed-precision, where data may be
// converted to a wider type. Specifically, the macro-kernels will need this larger
// micro-tile footprint, even though the virtual micro-kernels will only
// ever be writing to half (real or imaginary part) at a time.
#ifndef BLIS_STACK_BUF_MAX_SIZE
#define BLIS_STACK_BUF_MAX_SIZE          ( BLIS_SIMD_MAX_NUM_REGISTERS * \
                                           BLIS_SIMD_MAX_SIZE * 4 )
#endif

// Alignment size used to align local stack buffers within macro-kernel
// functions.
#ifndef BLIS_STACK_BUF_ALIGN_SIZE
#define BLIS_STACK_BUF_ALIGN_SIZE        BLIS_SIMD_ALIGN_SIZE
#endif

// Alignment size used when allocating memory via BLIS_MALLOC_USER.
// To disable heap alignment, set this to 1.
#ifndef BLIS_HEAP_ADDR_ALIGN_SIZE
#define BLIS_HEAP_ADDR_ALIGN_SIZE        BLIS_SIMD_ALIGN_SIZE
#endif

// Alignment size used when sizing leading dimensions of memory allocated
// via BLIS_MALLOC_USER.
#ifndef BLIS_HEAP_STRIDE_ALIGN_SIZE
#define BLIS_HEAP_STRIDE_ALIGN_SIZE      BLIS_SIMD_ALIGN_SIZE
#endif

// Alignment sizes used when allocating blocks to the internal memory
// pool, via BLIS_MALLOC_POOL.
#ifndef BLIS_POOL_ADDR_ALIGN_SIZE_A
#define BLIS_POOL_ADDR_ALIGN_SIZE_A      BLIS_PAGE_SIZE
#endif

#ifndef BLIS_POOL_ADDR_ALIGN_SIZE_B
#define BLIS_POOL_ADDR_ALIGN_SIZE_B      BLIS_PAGE_SIZE
#endif

#ifndef BLIS_POOL_ADDR_ALIGN_SIZE_C
#define BLIS_POOL_ADDR_ALIGN_SIZE_C      BLIS_PAGE_SIZE
#endif

#ifndef BLIS_POOL_ADDR_ALIGN_SIZE_GEN
#define BLIS_POOL_ADDR_ALIGN_SIZE_GEN    BLIS_PAGE_SIZE
#endif

// Offsets from alignment specified by BLIS_POOL_ADDR_ALIGN_SIZE_*.
#ifndef BLIS_POOL_ADDR_OFFSET_SIZE_A
#define BLIS_POOL_ADDR_OFFSET_SIZE_A     0
#endif

#ifndef BLIS_POOL_ADDR_OFFSET_SIZE_B
#define BLIS_POOL_ADDR_OFFSET_SIZE_B     0
#endif

#ifndef BLIS_POOL_ADDR_OFFSET_SIZE_C
#define BLIS_POOL_ADDR_OFFSET_SIZE_C     0
#endif

#ifndef BLIS_POOL_ADDR_OFFSET_SIZE_GEN
#define BLIS_POOL_ADDR_OFFSET_SIZE_GEN   0
#endif


// -- MR and NR blocksizes (only for reference kernels) ------------------------

// The build system defines BLIS_IN_REF_KERNEL, but only when compiling
// reference kernels. By using compile-time constants for MR and NR, the
// compiler can perform certain optimizations, such as unrolling and
// vectorization, that would not be otherwise be possible.
#ifdef BLIS_IN_REF_KERNEL

#ifndef BLIS_MR_s
#define BLIS_MR_s 4
#endif

#ifndef BLIS_MR_d
#define BLIS_MR_d 4
#endif

#ifndef BLIS_MR_c
#define BLIS_MR_c 4
#endif

#ifndef BLIS_MR_z
#define BLIS_MR_z 4
#endif

#ifndef BLIS_NR_s
#define BLIS_NR_s 16
#endif

#ifndef BLIS_NR_d
#define BLIS_NR_d 8
#endif

#ifndef BLIS_NR_c
#define BLIS_NR_c 8
#endif

#ifndef BLIS_NR_z
#define BLIS_NR_z 4
#endif

#ifndef BLIS_BBM_s
#define BLIS_BBM_s 1
#endif

#ifndef BLIS_BBM_d
#define BLIS_BBM_d 1
#endif

#ifndef BLIS_BBM_c
#define BLIS_BBM_c 1
#endif

#ifndef BLIS_BBM_z
#define BLIS_BBM_z 1
#endif

#ifndef BLIS_BBN_s
#define BLIS_BBN_s 1
#endif

#ifndef BLIS_BBN_d
#define BLIS_BBN_d 1
#endif

#ifndef BLIS_BBN_c
#define BLIS_BBN_c 1
#endif

#ifndef BLIS_BBN_z
#define BLIS_BBN_z 1
#endif

#ifndef BLIS_PACKMR_s
#define BLIS_PACKMR_s (BLIS_MR_s*BLIS_BBM_s)
#endif

#ifndef BLIS_PACKMR_d
#define BLIS_PACKMR_d (BLIS_MR_d*BLIS_BBM_d)
#endif

#ifndef BLIS_PACKMR_c
#define BLIS_PACKMR_c (BLIS_MR_c*BLIS_BBM_c)
#endif

#ifndef BLIS_PACKMR_z
#define BLIS_PACKMR_z (BLIS_MR_z*BLIS_BBM_z)
#endif

#ifndef BLIS_PACKNR_s
#define BLIS_PACKNR_s (BLIS_NR_s*BLIS_BBN_s)
#endif

#ifndef BLIS_PACKNR_d
#define BLIS_PACKNR_d (BLIS_NR_d*BLIS_BBN_d)
#endif

#ifndef BLIS_PACKNR_c
#define BLIS_PACKNR_c (BLIS_NR_c*BLIS_BBN_c)
#endif

#ifndef BLIS_PACKNR_z
#define BLIS_PACKNR_z (BLIS_NR_z*BLIS_BBN_z)
#endif

#endif

#endif

