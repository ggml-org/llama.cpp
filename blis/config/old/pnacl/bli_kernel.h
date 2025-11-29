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

#ifndef BLIS_KERNEL_H
#define BLIS_KERNEL_H

/*
 * SIMD-enabled (SP only) PNaCl shipped in Chrome 36 and it is not backward-compatible.
 * Therefore, if compilation targets an older Chrome release, we use scalar kernels.
 * The target Chrome version is indicated by PPAPI_MACRO defined in the header below.
 */
#include <ppapi/c/pp_macros.h>

// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

#define BLIS_SIMD_ALIGN_SIZE             16

// -- Cache blocksizes --

//
// Constraints:
//
// (1) MC must be a multiple of:
//     (a) MR (for zero-padding purposes)
//     (b) NR (for zero-padding purposes when MR and NR are "swapped")
// (2) NC must be a multiple of
//     (a) NR (for zero-padding purposes)
//     (b) MR (for zero-padding purposes when MR and NR are "swapped")
//

#if PPAPI_RELEASE >= 36
#define BLIS_DEFAULT_MC_S              256
#define BLIS_DEFAULT_KC_S              256
#define BLIS_DEFAULT_NC_S              8192
#else
#define BLIS_DEFAULT_MC_S              252
#define BLIS_DEFAULT_KC_S              264
#define BLIS_DEFAULT_NC_S              8196
#endif

#define BLIS_DEFAULT_MC_D              1080
#define BLIS_DEFAULT_KC_D              120
#define BLIS_DEFAULT_NC_D              8400

#if PPAPI_RELEASE >= 36
#define BLIS_DEFAULT_MC_C              128
#define BLIS_DEFAULT_KC_C              256
#define BLIS_DEFAULT_NC_C              4096
#else
#define BLIS_DEFAULT_MC_C              120
#define BLIS_DEFAULT_KC_C              264
#define BLIS_DEFAULT_NC_C              4092
#endif

#define BLIS_DEFAULT_MC_Z              60
#define BLIS_DEFAULT_KC_Z              264
#define BLIS_DEFAULT_NC_Z              2040

// -- Register blocksizes --

#if PPAPI_RELEASE >= 36
#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              4
#else
#define BLIS_DEFAULT_MR_S              4
#define BLIS_DEFAULT_NR_S              3 
#endif

#define BLIS_DEFAULT_MR_D              4
#define BLIS_DEFAULT_NR_D              3

#if PPAPI_RELEASE >= 36
#define BLIS_DEFAULT_MR_C              4
#define BLIS_DEFAULT_NR_C              4
#else
#define BLIS_DEFAULT_MR_C              2
#define BLIS_DEFAULT_NR_C              3
#endif

#define BLIS_DEFAULT_MR_Z              2
#define BLIS_DEFAULT_NR_Z              3

// NOTE: If the micro-kernel, which is typically unrolled to a factor
// of f, handles leftover edge cases (ie: when k % f > 0) then these
// register blocksizes in the k dimension can be defined to 1.

//#define BLIS_DEFAULT_KR_S              1
//#define BLIS_DEFAULT_KR_D              1
//#define BLIS_DEFAULT_KR_C              1
//#define BLIS_DEFAULT_KR_Z              1

// -- Maximum cache blocksizes (for optimizing edge cases) --

// NOTE: These cache blocksize "extensions" have the same constraints as
// the corresponding default blocksizes above. When these values are
// larger than the default blocksizes, blocksizes used at edge cases are
// enlarged if such an extension would encompass the remaining portion of
// the matrix dimension.

//#define BLIS_MAXIMUM_MC_S              (BLIS_DEFAULT_MC_S + BLIS_DEFAULT_MC_S/4)
//#define BLIS_MAXIMUM_KC_S              (BLIS_DEFAULT_KC_S + BLIS_DEFAULT_KC_S/4)
//#define BLIS_MAXIMUM_NC_S              (BLIS_DEFAULT_NC_S + BLIS_DEFAULT_NC_S/4)

//#define BLIS_MAXIMUM_MC_D              (BLIS_DEFAULT_MC_D + BLIS_DEFAULT_MC_D/4)
//#define BLIS_MAXIMUM_KC_D              (BLIS_DEFAULT_KC_D + BLIS_DEFAULT_KC_D/4)
//#define BLIS_MAXIMUM_NC_D              (BLIS_DEFAULT_NC_D + BLIS_DEFAULT_NC_D/4)

//#define BLIS_MAXIMUM_MC_C              (BLIS_DEFAULT_MC_C + BLIS_DEFAULT_MC_C/4)
//#define BLIS_MAXIMUM_KC_C              (BLIS_DEFAULT_KC_C + BLIS_DEFAULT_KC_C/4)
//#define BLIS_MAXIMUM_NC_C              (BLIS_DEFAULT_NC_C + BLIS_DEFAULT_NC_C/4)

//#define BLIS_MAXIMUM_MC_Z              (BLIS_DEFAULT_MC_Z + BLIS_DEFAULT_MC_Z/4)
//#define BLIS_MAXIMUM_KC_Z              (BLIS_DEFAULT_KC_Z + BLIS_DEFAULT_KC_Z/4)
//#define BLIS_MAXIMUM_NC_Z              (BLIS_DEFAULT_NC_Z + BLIS_DEFAULT_NC_Z/4)

// -- Packing register blocksize (for packed micro-panels) --

// NOTE: These register blocksize "extensions" determine whether the
// leading dimensions used within the packed micro-panels are equal to
// or greater than their corresponding register blocksizes above.

//#define BLIS_PACKDIM_MR_S              (BLIS_DEFAULT_MR_S + ...)
//#define BLIS_PACKDIM_NR_S              (BLIS_DEFAULT_NR_S + ...)

//#define BLIS_PACKDIM_MR_D              (BLIS_DEFAULT_MR_D + ...)
//#define BLIS_PACKDIM_NR_D              (BLIS_DEFAULT_NR_D + ...)

//#define BLIS_PACKDIM_MR_C              (BLIS_DEFAULT_MR_C + ...)
//#define BLIS_PACKDIM_NR_C              (BLIS_DEFAULT_NR_C + ...)

//#define BLIS_PACKDIM_MR_Z              (BLIS_DEFAULT_MR_Z + ...)
//#define BLIS_PACKDIM_NR_Z              (BLIS_DEFAULT_NR_Z + ...)



// -- LEVEL-2 KERNEL CONSTANTS -------------------------------------------------




// -- LEVEL-1F KERNEL CONSTANTS ------------------------------------------------




// -- LEVEL-3 KERNEL DEFINITIONS -----------------------------------------------

// -- gemm --

#if PPAPI_RELEASE >= 36
#define BLIS_SGEMM_UKERNEL         bli_sgemm_opt
#define BLIS_CGEMM_UKERNEL         bli_cgemm_opt
#endif

// -- trsm-related --




// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

// -- unpackm --




// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

// -- axpy2v --

// -- dotaxpyv --

// -- axpyf --

// -- dotxf --

// -- dotxaxpyf --




// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------

// -- addv --

// -- axpyv --
#if PPAPI_RELEASE >= 36
#define BLIS_SAXPYV_KERNEL         bli_saxpyv_opt
#define BLIS_CAXPYV_KERNEL         bli_caxpyv_opt
#endif

// -- copyv --

// -- dotv --
#define BLIS_SDOTV_KERNEL          bli_sdotv_opt
#define BLIS_DDOTV_KERNEL          bli_ddotv_opt
#define BLIS_CDOTV_KERNEL          bli_cdotv_opt
#define BLIS_ZDOTV_KERNEL          bli_zdotv_opt

// -- dotxv --

// -- invertv --

// -- scal2v --

// -- scalv --

// -- setv --

// -- subv --

// -- swapv --



#endif

