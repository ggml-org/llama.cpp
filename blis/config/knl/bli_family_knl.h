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

#define BLIS_THREAD_RATIO_M     4
#define BLIS_THREAD_RATIO_N     1

#define BLIS_THREAD_MAX_IR      1
#define BLIS_THREAD_MAX_JR      1


// -- MEMORY ALLOCATION --------------------------------------------------------

//#define BLIS_TREE_BARRIER
//#define BLIS_TREE_BARRIER_ARITY 4

#define BLIS_SIMD_ALIGN_SIZE             64

#define BLIS_SIMD_MAX_SIZE               64
#define BLIS_SIMD_MAX_NUM_REGISTERS      32

/*
#ifdef BLIS_NO_HBWMALLOC

#include <stdlib.h>

#define BLIS_MALLOC_POOL malloc
#define BLIS_FREE_POOL free

#else

#include <hbwmalloc.h>

#define BLIS_MALLOC_POOL hbw_malloc
#define BLIS_FREE_POOL hbw_free

#endif
*/

//#define BLIS_MALLOC_INTL hbw_malloc
//#define BLIS_FREE_INTL hbw_free


#if 0
// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_SGEMM_UKERNEL             bli_sgemm_opt_30x16_knc
#define BLIS_DEFAULT_MC_S              240
#define BLIS_DEFAULT_KC_S              240
#define BLIS_DEFAULT_NC_S              14400
#define BLIS_DEFAULT_MR_S              30
#define BLIS_DEFAULT_NR_S              16
#define BLIS_PACKDIM_MR_S              32
#define BLIS_PACKDIM_NR_S              16

#if 0

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_DGEMM_UKERNEL             bli_dgemm_opt_30x8_knc
#define BLIS_DEFAULT_MC_D              120
#define BLIS_DEFAULT_KC_D              240
#define BLIS_DEFAULT_NC_D              14400
#define BLIS_DEFAULT_MR_D              30
#define BLIS_DEFAULT_NR_D              8
#define BLIS_PACKDIM_MR_D              32
#define BLIS_PACKDIM_NR_D              8

#elif 0

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_DGEMM_UKERNEL             bli_dgemm_opt_30x8
#define BLIS_DEFAULT_MC_D              120
#define BLIS_DEFAULT_KC_D              240
#define BLIS_DEFAULT_NC_D              14400
#define BLIS_DEFAULT_MR_D              30
#define BLIS_DEFAULT_NR_D              8
#define BLIS_PACKDIM_MR_D              32
#define BLIS_PACKDIM_NR_D              8

#define BLIS_DPACKM_8XK_KERNEL         bli_dpackm_8xk_opt
#define BLIS_DPACKM_30XK_KERNEL        bli_dpackm_30xk_opt

#else

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_DGEMM_UKERNEL             bli_dgemm_opt_24x8
#define BLIS_DEFAULT_MR_D              24
#define BLIS_DEFAULT_NR_D              8
#define BLIS_PACKDIM_MR_D              24
#define BLIS_PACKDIM_NR_D              8
#define BLIS_DEFAULT_MC_D              120
#define BLIS_DEFAULT_KC_D              336
#define BLIS_DEFAULT_NC_D              14400

#define BLIS_DPACKM_8XK_KERNEL         bli_dpackm_8xk_opt
#define BLIS_DPACKM_24XK_KERNEL        bli_dpackm_24xk_opt

#endif

#define BLIS_MAXIMUM_MC_S              (BLIS_DEFAULT_MC_S + BLIS_DEFAULT_MC_S/4)
#define BLIS_MAXIMUM_KC_S              (BLIS_DEFAULT_KC_S + BLIS_DEFAULT_KC_S/4)
#define BLIS_MAXIMUM_NC_S              (BLIS_DEFAULT_NC_S +                   0) 

#define BLIS_MAXIMUM_MC_D              (BLIS_DEFAULT_MC_D + BLIS_DEFAULT_MC_D/4)
#define BLIS_MAXIMUM_KC_D              (BLIS_DEFAULT_KC_D + BLIS_DEFAULT_KC_D/4)
#define BLIS_MAXIMUM_NC_D              (BLIS_DEFAULT_NC_D +                   0)

#endif


//#endif

