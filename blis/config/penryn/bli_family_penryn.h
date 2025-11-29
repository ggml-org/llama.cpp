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

#define BLIS_SGEMM_UKERNEL             bli_sgemm_asm_8x4
#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              4
#define BLIS_DEFAULT_MC_S              768
#define BLIS_DEFAULT_KC_S              384
#define BLIS_DEFAULT_NC_S              4096

#define BLIS_DGEMM_UKERNEL             bli_dgemm_asm_4x4
#define BLIS_DEFAULT_MR_D              4
#define BLIS_DEFAULT_NR_D              4
#define BLIS_DEFAULT_MC_D              384
#define BLIS_DEFAULT_KC_D              384
#define BLIS_DEFAULT_NC_D              4096

#define BLIS_DGEMMTRSM_L_UKERNEL       bli_dgemmtrsm_l_asm_4x4
#define BLIS_DGEMMTRSM_U_UKERNEL       bli_dgemmtrsm_u_asm_4x4


// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

#define BLIS_DAXPY2V_KERNEL     bli_daxpy2v_int_var1
#define BLIS_DDOTAXPYV_KERNEL   bli_ddotaxpyv_int_var1
#define BLIS_DAXPYF_KERNEL      bli_daxpyf_int_var1
#define BLIS_DDOTXF_KERNEL      bli_ddotxf_int_var1
#define BLIS_DDOTXAXPYF_KERNEL  bli_ddotxaxpyf_int_var1


// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------

#define BLIS_DAXPYV_KERNEL      bli_daxpyv_opt_var1
#define BLIS_DDOTV_KERNEL       bli_ddotv_opt_var1
#endif



//#endif

