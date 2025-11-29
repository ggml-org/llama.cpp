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

#define BLIS_DEFAULT_MC_S              256
#define BLIS_DEFAULT_KC_S              256
#define BLIS_DEFAULT_NC_S              8192

#define BLIS_DEFAULT_MC_D              32
#define BLIS_DEFAULT_KC_D              128
#define BLIS_DEFAULT_NC_D              1024

#define BLIS_DEFAULT_MC_C              128
#define BLIS_DEFAULT_KC_C              256
#define BLIS_DEFAULT_NC_C              4096

#define BLIS_DEFAULT_MC_Z              64
#define BLIS_DEFAULT_KC_Z              256
#define BLIS_DEFAULT_NC_Z              2048

// -- Register blocksizes --

#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              4

#define BLIS_DEFAULT_MR_D              4
#define BLIS_DEFAULT_NR_D              4

#define BLIS_DEFAULT_MR_C              8
#define BLIS_DEFAULT_NR_C              4

#define BLIS_DEFAULT_MR_Z              8
#define BLIS_DEFAULT_NR_Z              4

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

#define BLIS_DGEMM_UKERNEL         bli_dgemm_opt_4x4

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

// -- copyv --

// -- dotv --

// -- dotxv --

// -- invertv --

// -- scal2v --

// -- scalv --

// -- setv --

// -- subv --

// -- swapv --



#endif

