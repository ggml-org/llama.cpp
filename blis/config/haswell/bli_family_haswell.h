/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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


#if 0
// -- LEVEL-3 MICRO-KERNEL CONSTANTS AND DEFINITIONS ---------------------------

// -- sgemm micro-kernel --

#if 0
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_4x24
#define BLIS_DEFAULT_MC_S          256
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          4
#define BLIS_DEFAULT_NR_S          24

#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 1
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_6x16
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          6
#define BLIS_DEFAULT_NR_S          16

#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 0
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_16x6
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          16
#define BLIS_DEFAULT_NR_S          6
#endif

// -- dgemm micro-kernel --

#if 0
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_4x12
#define BLIS_DEFAULT_MC_D          152
#define BLIS_DEFAULT_KC_D          160
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          4
#define BLIS_DEFAULT_NR_D          12

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 1
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_6x8
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          6
#define BLIS_DEFAULT_NR_D          8

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 0
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_8x6
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          8
#define BLIS_DEFAULT_NR_D          6
#endif

// -- cgemm micro-kernel --

#if 1
#define BLIS_CGEMM_UKERNEL         bli_cgemm_asm_3x8
#define BLIS_DEFAULT_MC_C          144
#define BLIS_DEFAULT_KC_C          256
#define BLIS_DEFAULT_NC_C          4080
#define BLIS_DEFAULT_MR_C          3
#define BLIS_DEFAULT_NR_C          8

#define BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 0
#define BLIS_CGEMM_UKERNEL         bli_cgemm_asm_8x3
#define BLIS_DEFAULT_MC_C          144
#define BLIS_DEFAULT_KC_C          256
#define BLIS_DEFAULT_NC_C          4080
#define BLIS_DEFAULT_MR_C          8
#define BLIS_DEFAULT_NR_C          3
#endif

//  -- zgemm micro-kernel --

#if 1
#define BLIS_ZGEMM_UKERNEL         bli_zgemm_asm_3x4
#define BLIS_DEFAULT_MC_Z          72
#define BLIS_DEFAULT_KC_Z          256
#define BLIS_DEFAULT_NC_Z          4080
#define BLIS_DEFAULT_MR_Z          3
#define BLIS_DEFAULT_NR_Z          4

#define BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 0
#define BLIS_ZGEMM_UKERNEL         bli_zgemm_asm_4x3
#define BLIS_DEFAULT_MC_Z          72
#define BLIS_DEFAULT_KC_Z          256
#define BLIS_DEFAULT_NC_Z          4080
#define BLIS_DEFAULT_MR_Z          4
#define BLIS_DEFAULT_NR_Z          3
#endif

#endif


//#endif

