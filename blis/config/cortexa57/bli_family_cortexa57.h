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


// -- MEMORY ALLOCATION --------------------------------------------------------

#define BLIS_SIMD_ALIGN_SIZE           16


#if 0
// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

#define BLIS_SGEMM_UKERNEL             bli_sgemm_opt_8x12
#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              12
#define BLIS_DEFAULT_MC_S              120 //1536 //336 //416 // 1280 //160 // 160 // 160 //2048 //336 
#define BLIS_DEFAULT_KC_S              640 //1536 //336 //704 //1280 //672 //528 // 856 //2048 //528 
#define BLIS_DEFAULT_NC_S              3072

#define BLIS_DGEMM_UKERNEL             bli_dgemm_opt_6x8
#define BLIS_DEFAULT_MR_D              6
#define BLIS_DEFAULT_NR_D              8
#define BLIS_DEFAULT_MC_D              120 //1536 //160 //80 //176 
#define BLIS_DEFAULT_KC_D              240 //1536 //304 //336 //368 
#define BLIS_DEFAULT_NC_D              3072

#define BLIS_DEFAULT_MR_C              8
#define BLIS_DEFAULT_NR_C              4
#define BLIS_DEFAULT_MC_C              64
#define BLIS_DEFAULT_KC_C              128
#define BLIS_DEFAULT_NC_C              4096

#define BLIS_DEFAULT_MR_Z              8
#define BLIS_DEFAULT_NR_Z              4
#define BLIS_DEFAULT_MC_Z              64
#define BLIS_DEFAULT_KC_Z              128
#define BLIS_DEFAULT_NC_Z              4096
#endif


//#endif

