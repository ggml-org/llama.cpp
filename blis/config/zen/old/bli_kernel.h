/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

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


// -- LEVEL-3 MICRO-KERNEL CONSTANTS AND DEFINITIONS ---------------------------

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

// threading related
// By default it is effective to paralleize the 
// outerloops. Setting these macros to 1 will force
// JR and NR inner loops to be not paralleized.
#define BLIS_DEFAULT_MR_THREAD_MAX 1 
#define BLIS_DEFAULT_NR_THREAD_MAX 1 

// sgemm micro-kernel

#if 0

#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_24x4
#define BLIS_DEFAULT_MC_S          264
#define BLIS_DEFAULT_KC_S          128
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          24
#define BLIS_DEFAULT_NR_S          4
#endif

#if 0
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_16x6
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          16
#define BLIS_DEFAULT_NR_S          6
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

// dgemm micro-kernel

#if 0

#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_12x4
#define BLIS_DEFAULT_MC_D          96
#define BLIS_DEFAULT_KC_D          192
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          12
#define BLIS_DEFAULT_NR_D          4
#endif

#if 0
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_8x6
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          8
#define BLIS_DEFAULT_NR_D          6
#endif

#if 1
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_6x8
#define BLIS_DEFAULT_MC_D         510  // 72 /* Improves performance for large Matrices */
#define BLIS_DEFAULT_KC_D         1024 // 256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          6
#define BLIS_DEFAULT_NR_D          8

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

// cgemm micro-kernel

#if 1
#define BLIS_CGEMM_UKERNEL         bli_cgemm_asm_3x8
#define BLIS_DEFAULT_MC_C          144
#define BLIS_DEFAULT_KC_C          256
#define BLIS_DEFAULT_NC_C          4080
#define BLIS_DEFAULT_MR_C          3
#define BLIS_DEFAULT_NR_C          8

#define BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

// zgemm micro-kernel

#if 1
#define BLIS_ZGEMM_UKERNEL         bli_zgemm_asm_3x4
#define BLIS_DEFAULT_MC_Z          72
#define BLIS_DEFAULT_KC_Z          256
#define BLIS_DEFAULT_NC_Z          4080
#define BLIS_DEFAULT_MR_Z          3
#define BLIS_DEFAULT_NR_Z          4

#define BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif


// zgemm micro-kernel

#if 1
#define BLIS_ZGEMM_UKERNEL         bli_zgemm_asm_3x4
#define BLIS_DEFAULT_MC_Z          72
#define BLIS_DEFAULT_KC_Z          256
#define BLIS_DEFAULT_NC_Z          4080
#define BLIS_DEFAULT_MR_Z          3
#define BLIS_DEFAULT_NR_Z          4

#define BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

// -- trsm-related --

#define BLIS_STRSM_L_UKERNEL   bli_strsm_l_int_6x16
#define BLIS_DTRSM_L_UKERNEL   bli_dtrsm_l_int_6x8

// --gemmtrsm-related --
#define BLIS_SGEMMTRSM_L_UKERNEL bli_sgemmtrsm_l_6x16
#define BLIS_DGEMMTRSM_L_UKERNEL bli_dgemmtrsm_l_6x8

#define BLIS_SMALL_MATRIX_ENABLE
//This will select the threshold below which small matrix code will be called.
#define BLIS_SMALL_MATRIX_THRES 700
#define BLIS_SMALL_M_RECT_MATRIX_THRES 160
#define BLIS_SMALL_K_RECT_MATRIX_THRES 128

gint_t bli_gemm_small_matrix
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     );

// -- LEVEL-2 KERNEL CONSTANTS -------------------------------------------------




// -- LEVEL-1F KERNEL CONSTANTS ------------------------------------------------




// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

// -- unpackm --

#define BLIS_DEFAULT_1F_S    8
#define BLIS_DEFAULT_1F_D    4


// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

// -- axpy2v --

// -- dotaxpyv --

// -- axpyf --
#define BLIS_SAXPYF_KERNEL           bli_saxpyf_int_var1
#define BLIS_DAXPYF_KERNEL           bli_daxpyf_int_var1

// -- dotxf --
#define BLIS_SDOTXF_KERNEL           bli_sdotxf_int_var1
#define BLIS_DDOTXF_KERNEL           bli_ddotxf_int_var1

// -- dotxaxpyf --


// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

// -- unpackm --




// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------
// -- amax --
#define BLIS_SAMAXV_KERNEL         bli_samaxv_opt_var1
#define BLIS_DAMAXV_KERNEL         bli_damaxv_opt_var1
// -- addv --

// -- axpyv --
#define BLIS_DAXPYV_KERNEL         bli_daxpyv_opt_var10
#define BLIS_SAXPYV_KERNEL         bli_saxpyv_opt_var10


// -- copyv --

// -- dotv --
#define BLIS_DDOTV_KERNEL          bli_ddotv_opt_var1
#define BLIS_SDOTV_KERNEL          bli_sdotv_opt_var1

// -- dotxv --
#define BLIS_SDOTXV_KERNEL         bli_sdotxv_unb_var1
#define BLIS_DDOTXV_KERNEL         bli_ddotxv_unb_var1

// -- invertv --

// -- scal2v --

// -- scalv --
#define BLIS_SSCALV_KERNEL   bli_sscalv_opt_var2
#define BLIS_DSCALV_KERNEL   bli_dscalv_opt_var2

// -- setv --

// -- subv --

// -- swapv --



#endif

