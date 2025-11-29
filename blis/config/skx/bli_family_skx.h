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

//#ifndef BLIS_FAMILY_H
//#define BLIS_FAMILY_H

// -- THREADING PARAMETERS -----------------------------------------------------

#define BLIS_THREAD_RATIO_M     3
#define BLIS_THREAD_RATIO_N     2

#define BLIS_THREAD_MAX_IR      1
#define BLIS_THREAD_MAX_JR      4

// -- MEMORY ALLOCATION --------------------------------------------------------

#define BLIS_SIMD_ALIGN_SIZE             64

#define BLIS_SIMD_MAX_SIZE               64
#define BLIS_SIMD_MAX_NUM_REGISTERS      32

//#include <stdlib.h>

//#define BLIS_MALLOC_POOL malloc
//#define BLIS_FREE_POOL free


#if 0
// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

// -- Cache and register blocksizes --

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

#define BLIS_DGEMM_UKERNEL             bli_dgemm_opt_16x12_l2
#define BLIS_DEFAULT_MC_D              144
#define BLIS_DEFAULT_KC_D              336
#define BLIS_DEFAULT_NC_D              5760
#define BLIS_DEFAULT_MR_D              16
#define BLIS_DEFAULT_NR_D              12
#define BLIS_PACKDIM_MR_D              16
#define BLIS_PACKDIM_NR_D              12

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

#define BLIS_MAXIMUM_MC_S              (BLIS_DEFAULT_MC_S + BLIS_DEFAULT_MC_S/4)
#define BLIS_MAXIMUM_KC_S              (BLIS_DEFAULT_KC_S + BLIS_DEFAULT_KC_S/4)
#define BLIS_MAXIMUM_NC_S              (BLIS_DEFAULT_NC_S +                   0)

#define BLIS_MAXIMUM_MC_D              (BLIS_DEFAULT_MC_D + BLIS_DEFAULT_MC_D/4)
#define BLIS_MAXIMUM_KC_D              (BLIS_DEFAULT_KC_D + BLIS_DEFAULT_KC_D/4)
#define BLIS_MAXIMUM_NC_D              (BLIS_DEFAULT_NC_D +                   0)

//#define BLIS_MAXIMUM_MC_C              (BLIS_DEFAULT_MC_C + BLIS_DEFAULT_MC_C/4)
//#define BLIS_MAXIMUM_KC_C              (BLIS_DEFAULT_KC_C + BLIS_DEFAULT_KC_C/4)
//#define BLIS_MAXIMUM_NC_C              (BLIS_DEFAULT_NC_C + BLIS_DEFAULT_NC_C/4)

//#define BLIS_MAXIMUM_MC_Z              (BLIS_DEFAULT_MC_Z + BLIS_DEFAULT_MC_Z/4)
//#define BLIS_MAXIMUM_KC_Z              (BLIS_DEFAULT_KC_Z + BLIS_DEFAULT_KC_Z/4)
//#define BLIS_MAXIMUM_NC_Z              (BLIS_DEFAULT_NC_Z + BLIS_DEFAULT_NC_Z/4)


#endif


//#endif

