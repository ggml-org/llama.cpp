/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

// This avoids a known issue with GCC15+ ("error: bp cannot be used in asm here", #845).
// Make sure the compiler isn't clang since it also confusingly defines __GNUC__
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 15
#pragma GCC optimize("-fno-tree-vectorize")
#endif

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

static const int64_t mask_3[4] = {-1, -1, -1, 0};
static const int64_t mask_1[4] = {-1, 0, 0, 0};

static void bli_dgemmsup_rv_haswell_asm_6x7m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     );

static void bli_dgemmsup_rv_haswell_asm_6x5m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     );

static void bli_dgemmsup_rv_haswell_asm_6x3m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     );

static void bli_dgemmsup_rv_haswell_asm_6x1m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     );

#define C_TRANSPOSE_6x7_TILE(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Broadcasting Beta into ymm15 vector register*/\
	vbroadcastsd(mem(rbx), ymm15)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3, R4 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm15, ymm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm(R3))\
	vfmadd231pd(mem(rcx, rax, 1), ymm15, ymm(R4))\
	/*Storing it back to C matrix.*/ \
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	/*Moving to operate on last 2 rows of 6 rows.*/ \
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2, 3 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm15, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm15, xmm2)\
	vfmadd231pd(mem(rdx, rax, 1), xmm15, xmm3)\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpckhpd(ymm(R8), ymm(R7), ymm1)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vunpckhpd(ymm(R10), ymm(R9), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)\
\
	vfmadd231pd(mem(rcx        ), ymm15, ymm5)\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm7)\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm9)\
\
	vmovupd(ymm5, mem(rcx        ))\
	vmovupd(ymm7, mem(rcx, rsi, 1))\
	vmovupd(ymm9, mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
	vunpckhpd(ymm(R12), ymm(R11), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm4)\
\
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm15, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm15, xmm2)\
\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x7_TILE_BZ(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Storing transposed 4x4 tile back to C matrix*/\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Storing transposed 2x4 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpckhpd(ymm(R8), ymm(R7), ymm1)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vunpckhpd(ymm(R10), ymm(R9), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)\
\
	/*Storing transposed 4x3 tile back to C matrix*/\
	vmovupd(ymm5, mem(rcx        ))\
	vmovupd(ymm7, mem(rcx, rsi, 1))\
	vmovupd(ymm9, mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
	vunpckhpd(ymm(R12), ymm(R11), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm4)\
\
	/*Storing transposed 2x3 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x5_TILE(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Broadcasting Beta into ymm15 vector register*/\
	vbroadcastsd(mem(rbx), ymm15)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3, R4 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm15, ymm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm(R3))\
	vfmadd231pd(mem(rcx, rax, 1), ymm15, ymm(R4))\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2, 3 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm15, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm15, xmm2)\
	vfmadd231pd(mem(rdx, rax, 1), xmm15, xmm3)\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x1 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
\
	vfmadd231pd(mem(rcx        ), ymm15, ymm5)\
	vmovupd(ymm5, mem(rcx        ))\
\
	/*Transposing 2x1 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
\
	vmovupd(xmm0, mem(rdx        ))

#define C_TRANSPOSE_6x5_TILE_BZ(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Storing transposed 4x4 tile back to C matrix*/\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Storing transposed 4x2 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x1 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
\
	/*Storing transposed 4x1 tile back to C matrix*/\
	vmovupd(ymm5, mem(rcx        ))\
\
	/*Transposing 2x1 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
\
	/*Storing transposed 2x1 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))

#define C_TRANSPOSE_6x3_TILE(R1, R2, R3, R4, R5, R6) \
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
\
	vbroadcastsd(mem(rbx), ymm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm3, ymm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm(R3))\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm3, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x3_TILE_BZ(R1, R2, R3, R4, R5, R6) \
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm4)\
\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x1_TILE(R1, R2, R3, R4, R5, R6) \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
\
	vbroadcastsd(mem(rbx), ymm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm3, ymm(R1))\
	vmovupd(ymm(R1), mem(rcx        ))\
\
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm3, xmm0)\
	vmovupd(xmm0, mem(rdx        ))\

#define C_TRANSPOSE_6x1_TILE_BZ(R1, R2, R3, R4, R5, R6) \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
\
	vmovupd(ymm(R1), mem(rcx        ))\
\
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
\
	vmovupd(xmm0, mem(rdx        ))\
/*
   rrr:
	 --------        ------        --------
	 --------        ------        --------
	 --------   +=   ------ ...    --------
	 --------        ------        --------
	 --------        ------            :
	 --------        ------            :

   rcr:
	 --------        | | | |       --------
	 --------        | | | |       --------
	 --------   +=   | | | | ...   --------
	 --------        | | | |       --------
	 --------        | | | |           :
	 --------        | | | |           :

   Assumptions:
   - B is row-stored;
   - A is row- or column-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels explicitly support column-oriented IO, implemented
   via an in-register transpose. And thus they also support the crr and
   ccr cases, though only crr is ever utilized (because ccr is handled by
   transposing the operation and executing rcr, which does not incur the
   cost of the in-register transpose).

   crr:
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |  +=   ------ ...    --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------            :
	 | | | | | | | |       ------            :
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


void bli_dgemmsup_rv_haswell_asm_6x8m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t n_left = n0 % 8;

	// First check whether this is a edge case in the n dimension. If so,
	// dispatch other 6x?m kernels, as needed.
	if ( n_left )
	{
		double* restrict cij = c;
		double* restrict bj  = b;
		double* restrict ai  = a;

		switch(n_left)
		{
			case 7:
			{
				bli_dgemmsup_rv_haswell_asm_6x7m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 6:
			{
				bli_dgemmsup_rv_haswell_asm_6x6m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 5:
			{
				bli_dgemmsup_rv_haswell_asm_6x5m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 4:
			{
				bli_dgemmsup_rv_haswell_asm_6x4m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 3:
			{
				bli_dgemmsup_rv_haswell_asm_6x3m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 2:
			{
				bli_dgemmsup_rv_haswell_asm_6x2m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 1:
			{
				bli_dgemmsup_rv_haswell_asm_6x1m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			default:
			{
				break;
			}
		}
		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter_8 = k0 / 8;     //unroll by 8
	uint64_t k_left   = k0 % 8;
	uint64_t k_iter_4 = k_left / 4; //unroll by 4
	k_left            = k_left % 4;

	//printf("%d and %d and %d and %d\n", k_iter_8, k_iter_4, k_left, k_left);

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)                      // reset rax to current upanel of a.


	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8

	lea(mem(r10, r10, 4), rcx)         // rcx = 3*rs_b;


    /**
	 * Prefetching Strategy
	 * B matrix prefetch brings the next rows of B into cache before it is accessed.
	 * It follows streaming access, which is cache-friendly.
	 * Very short prefetch distance typically the next rows to be accessed in loop.
	 * 
	 * A matrix prefetch brings future elements of A(64 bytes ahead, 1 unrolled iterations apart.)
	 * Since prefetched data are to be reused within unrolled iterations itself, data will be accessed,
	 * before it gets evicted from cache.
	 */
	mov(var(k_iter_8), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // Main unrolled loop (8 iterations per loop)


	// ---------------------------------- iteration 0
	// Prefetch 4th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))      
	vmovupd(mem(rbx,  0*32), ymm0)      // Load B[0:3]
	vmovupd(mem(rbx,  1*32), ymm1)      // Load B[4:7]
	add(r10, rbx)                      // b += rs_b;

	// Prefetch A[8] (64 bytes ahead of current A)
	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1
	// Prefetch 5th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	// Prefetch A[9] (64 bytes ahead of current A)
	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	// Prefetch 6th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	// Prefetch A[10] (64 bytes ahead of current A)
	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3
	// Prefetch 7th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	// Prefetch A[11] (64 bytes ahead of current A)
	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 4
	// Prefetch 8th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	// Prefetch A[12] (64 bytes ahead of current A)
	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 5
	// Prefetch 9th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	// Prefetch A[13] (64 bytes ahead of current A)
	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 6
	// Prefetch 10th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 7
	// Prefetch 11th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.


	label(.DLOOPKLOOP2)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), r9)                // load cs_c to rsi (temporarily)
	lea(mem(, r9, 8), r9)            // cs_c *= sizeof(double)
	lea(mem(r12, r9, 2), rdx)         //
	lea(mem(rdx, r9, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, r9, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, r9, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, r9, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 7*cs_c

	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r9, 8), r9)              // rs_a *= sizeof(double)

	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT_8)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP


	// ---------------------------------- iteration 0
	// Prefetch 4th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1
	// Prefetch 5th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	// Prefetch 6th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3
	// Prefetch 7th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 4
	// Prefetch 8th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 5
	// Prefetch 9th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 6
	// Prefetch 10th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;

	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 7
	// Prefetch 11th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;

	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)                   // iterate again if i != 0.


	label(.DCONSIDKLEFT_8)

	mov(var(k_iter_4), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	label(.DLOOPKITER_4)

	// ---------------------------------- iteration 0
	// Prefetch 4th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1
	// Prefetch 5th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	// Prefetch 6th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3
	// Prefetch 7th row of B from current position
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)

	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER_4)                 // iterate again if i != 0.

	label(.DCONSIDKLEFT)
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm1, ymm1, ymm1)           // set ymm0 to zero.
	vucomisd(xmm1, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vmulpd(ymm0, ymm5, ymm5)
	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmulpd(ymm0, ymm6, ymm6)
	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vmulpd(ymm0, ymm7, ymm7)
	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))

	vmulpd(ymm0, ymm8, ymm8)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rbx, 0*32))

	vmulpd(ymm0, ymm9, ymm9)
	vfmadd231pd(mem(rbx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rbx, 1*32))
	add(rdi, rbx)

	vmulpd(ymm0, ymm10, ymm10)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rbx, 0*32))

	vmulpd(ymm0, ymm11, ymm11)
	vfmadd231pd(mem(rbx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rbx, 1*32))

	vmulpd(ymm0, ymm12, ymm12)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rdx, 0*32))

	vmulpd(ymm0, ymm13, ymm13)
	vfmadd231pd(mem(rdx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rdx, 1*32))
	add(rdi, rdx)

	vmulpd(ymm0, ymm14, ymm14)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rdx, 0*32))

	vmulpd(ymm0, ymm15, ymm15)
	vfmadd231pd(mem(rdx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X8I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()


    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter_8] "m" (k_iter_8),
      [k_iter_4] "m" (k_iter_4),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t   nr_cur = 8;
		const dim_t   i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		gemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x8,
		  bli_dgemmsup_rv_haswell_asm_2x8,
		  bli_dgemmsup_rv_haswell_asm_3x8,
		  bli_dgemmsup_rv_haswell_asm_4x8,
		  bli_dgemmsup_rv_haswell_asm_5x8
		};

		gemmsup_ker_ft  ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
}

void bli_dgemmsup_rv_haswell_asm_6x6m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm1,  ymm1,  ymm1)         // zero ymm1 since we only use the lower
	vxorpd(ymm4,  ymm4,  ymm4)         // half (xmm1), and nans/infs may slow us
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r10, r10, 4), rcx)         // rcx = 3*cs_a;


	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), r9)                // load cs_c to rsi (temporarily)
	lea(mem(, r9, 8), r9)            // cs_c *= sizeof(double)
	lea(mem(r12, r9, 2), rdx)         //
	lea(mem(rdx, r9, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, r9, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, r9, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, r9, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 7*cs_c

	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r9, 8), r9)              // rs_a *= sizeof(double)

	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)                   // iterate again if i != 0.


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm1, ymm1, ymm1)           // set ymm0 to zero.
	vucomisd(xmm1, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vmulpd(ymm0, ymm4, ymm4)
	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vmulpd(ymm0, ymm5, ymm5)
	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmulpd(ymm0, ymm6, ymm6)
	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vmulpd(ymm0, ymm7, ymm7)
	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, 1*32))

	vmulpd(ymm0, ymm8, ymm8)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rbx, 0*32))

	vmulpd(ymm0, ymm9, ymm9)
	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm9)
	vmovupd(xmm9, mem(rbx, 1*32))
	add(rdi, rbx)

	vmulpd(ymm0, ymm10, ymm10)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rbx, 0*32))

	vmulpd(ymm0, ymm11, ymm11)
	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm11)
	vmovupd(xmm11, mem(rbx, 1*32))

	vmulpd(ymm0, ymm12, ymm12)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rdx, 0*32))

	vmulpd(ymm0, ymm13, ymm13)
	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm13)
	vmovupd(xmm13, mem(rdx, 1*32))
	add(rdi, rdx)

	vmulpd(ymm0, ymm14, ymm14)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rdx, 0*32))

	vmulpd(ymm0, ymm15, ymm15)
	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm15)
	vmovupd(xmm15, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(xmm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(xmm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X8I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t   nr_cur = 6;
		const dim_t   i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			gemmsup_ker_ft  ker_fp1 = NULL;
			gemmsup_ker_ft  ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x6;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x6;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x6;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		gemmsup_ker_ft  ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x6,
		  bli_dgemmsup_rv_haswell_asm_2x6,
		  bli_dgemmsup_rv_haswell_asm_3x6,
		  bli_dgemmsup_rv_haswell_asm_4x6,
		  bli_dgemmsup_rv_haswell_asm_5x6
		};

		gemmsup_ker_ft  ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
}

void bli_dgemmsup_rv_haswell_asm_6x4m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)

	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8

	lea(mem(r10, r10, 4), rcx)         // rcx = 3*cs_a;




	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)

	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)


	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)


	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), r9)                // load cs_c to rsi (temporarily)
	lea(mem(, r9, 8), r9)            // cs_c *= sizeof(double)
	lea(mem(r12, r9, 2), rdx)         //
	lea(mem(rdx, r9, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, r9, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, r9, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, r9, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 7*cs_c

	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r9, 8), r9)              // rs_a *= sizeof(double)

	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)


	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)


	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)       // iterate again if i != 0.

	vaddpd(ymm5, ymm4, ymm4)
	vaddpd(ymm7, ymm6, ymm6)
	vaddpd(ymm9, ymm8, ymm8)
	vaddpd(ymm11, ymm10, ymm10)
	vaddpd(ymm13, ymm12, ymm12)
	vaddpd(ymm15, ymm14, ymm14)



	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm1, ymm1, ymm1)           // set ymm0 to zero.
	vucomisd(xmm1, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vmulpd(ymm0, ymm4, ymm4)
	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))
	add(rdi, rcx)

	vmulpd(ymm0, ymm6, ymm6)
	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vmulpd(ymm0, ymm8, ymm8)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rbx, 0*32))
	add(rdi, rbx)

	vmulpd(ymm0, ymm10, ymm10)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rbx, 0*32))

	vmulpd(ymm0, ymm12, ymm12)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rdx, 0*32))
	add(rdi, rdx)

	vmulpd(ymm0, ymm14, ymm14)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rdx, 0*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	vmovupd(ymm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X4I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14",  "ymm15", "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t   nr_cur = 4;
		const dim_t   i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			gemmsup_ker_ft  ker_fp1 = NULL;
			gemmsup_ker_ft  ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x4;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x4;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x4;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		gemmsup_ker_ft  ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x4,
		  bli_dgemmsup_rv_haswell_asm_2x4,
		  bli_dgemmsup_rv_haswell_asm_3x4,
		  bli_dgemmsup_rv_haswell_asm_4x4,
		  bli_dgemmsup_rv_haswell_asm_5x4
		};

		gemmsup_ker_ft  ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
}

void bli_dgemmsup_rv_haswell_asm_6x2m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X2I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(xmm4,  xmm4,  xmm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)

	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8

	lea(mem(r10, r10, 1), rcx)         // rcx = 3*cs_a;


	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)


	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)


	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), r9)                // load cs_c to rsi (temporarily)
	lea(mem(, r9, 8), r9)            // cs_c *= sizeof(double)
	lea(mem(r12, r9, 2), rdx)         //
	lea(mem(rdx, r9, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, r9, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, r9, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, r9, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 7*cs_c

	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r9, 8), r9)              // rs_a *= sizeof(double)

	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP

	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)


	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)


	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)                   // iterate again if i != 0.

	vaddpd(ymm5, ymm4, ymm4)
	vaddpd(ymm7, ymm6, ymm6)
	vaddpd(ymm9, ymm8, ymm8)
	vaddpd(ymm11, ymm10, ymm10)
	vaddpd(ymm13, ymm12, ymm12)
	vaddpd(ymm15, ymm14, ymm14)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm1, ymm1, ymm1)           // set ymm0 to zero.
	vucomisd(xmm1, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vmulpd(ymm0, ymm4, ymm4)
	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)

	vmulpd(ymm0, ymm6, ymm6)
	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32))

	vmulpd(ymm0, ymm8, ymm8)
	vfmadd231pd(mem(rbx, 0*32), xmm3, xmm8)
	vmovupd(xmm8, mem(rbx, 0*32))
	add(rdi, rbx)

	vmulpd(ymm0, ymm10, ymm10)
	vfmadd231pd(mem(rbx, 0*32), xmm3, xmm10)
	vmovupd(xmm10, mem(rbx, 0*32))

	vmulpd(ymm0, ymm12, ymm12)
	vfmadd231pd(mem(rdx, 0*32), xmm3, xmm12)
	vmovupd(xmm12, mem(rdx, 0*32))
	add(rdi, rdx)

	vmulpd(ymm0, ymm14, ymm14)
	vfmadd231pd(mem(rdx, 0*32), xmm3, xmm14)
	vmovupd(xmm14, mem(rdx, 0*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	                                   // begin I/O on columns 0-3
	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	vmovupd(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm12, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm14, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	                                   // begin I/O on columns 0-3
	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X2I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t   nr_cur = 2;
		const dim_t   i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			gemmsup_ker_ft  ker_fp1 = NULL;
			gemmsup_ker_ft  ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x2;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x2;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x2;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		gemmsup_ker_ft  ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x2,
		  bli_dgemmsup_rv_haswell_asm_2x2,
		  bli_dgemmsup_rv_haswell_asm_3x2,
		  bli_dgemmsup_rv_haswell_asm_4x2,
		  bli_dgemmsup_rv_haswell_asm_5x2
		};

		gemmsup_ker_ft  ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
}

static void bli_dgemmsup_rv_haswell_asm_6x7m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     )
{
// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  -1 |  -1 | 0   |  ----> Mask vector( mask_3 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//
// Since we have 7 elements to load, kernel will use one normal load
// that loads 4 elements into vector register and for remainder 3 elements,
// kernel is using mask_3 which is set to -1, -1, -1, 0 so that the
// 3 elements will be loaded and 4th element will be set to 0 in destination vector.
//
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_3;

	if ( m_iter == 0 ) goto consider_edge_cases_7;

	// -------------------------------------------------------------------------
	begin_asm()

	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X7I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm3)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r10, r10, 4), rcx)
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif

	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)

	label(.DLOOPKITER)                 // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rdi)                // load cs_c to rsi (temporarily)
	lea(mem(, rdi, 8), rdi)            // cs_c *= sizeof(double)
	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*cs_c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)       // iterate again if i != 0.

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	                                   // now avoid loading C if beta == 0
	vxorpd(ymm2, ymm2, ymm2)           // set ymm0 to zero.
	vucomisd(xmm2, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vmulpd(ymm0, ymm3, ymm3)
	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	//Loads 4 element
	vmovupd(ymm3, mem(rcx, 0*32))

	vmulpd(ymm0, ymm4, ymm4)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm1, ymm4)
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1
	vmulpd(ymm0, ymm5, ymm5)
	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm5)
	vmovupd(ymm5, mem(rcx, 0*32))

	vmulpd(ymm0, ymm6, ymm6)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm3)
	vfmadd231pd(ymm3, ymm1, ymm6)
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	//-----------------------2
	vmulpd(ymm0, ymm7, ymm7)
	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm7)
	vmovupd(ymm7, mem(rbx, 0*32))

	vmulpd(ymm0, ymm8, ymm8)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm4)
	vfmadd231pd(ymm4, ymm1, ymm8)
	vmaskmovpd(ymm8, ymm15, mem(rbx, 1*32))

	add(rdi, rbx)
	//-----------------------3
	vmulpd(ymm0, ymm9, ymm9)
	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm9)
	vmovupd(ymm9, mem(rbx, 0*32))

	vmulpd(ymm0, ymm10, ymm10)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm5)
	vfmadd231pd(ymm5, ymm1, ymm10)
	vmaskmovpd(ymm10, ymm15, mem(rbx, 1*32))

	//-----------------------4
	vmulpd(ymm0, ymm11, ymm11)
	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm11)
	vmovupd(ymm11, mem(rdx, 0*32))

	vmulpd(ymm0, ymm12, ymm12)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm6)
	vfmadd231pd(ymm6, ymm1, ymm12)
	vmaskmovpd(ymm12, ymm15, mem(rdx, 1*32))

	add(rdi, rdx)
	//-----------------------5
	vmulpd(ymm0, ymm13, ymm13)
	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm13)
	vmovupd(ymm13, mem(rdx, 0*32))

	vmulpd(ymm0, ymm14, ymm14)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm7)
	vfmadd231pd(ymm7, ymm1, ymm14)
	vmaskmovpd(ymm14, ymm15, mem(rdx, 1*32))

	//-----------------------6

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)
	vmulpd(ymm0, ymm3, ymm3)
	vmulpd(ymm0, ymm4, ymm4)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	C_TRANSPOSE_6x7_TILE(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)
	vmulpd(ymm0, ymm3, ymm3)
	vmulpd(ymm0, ymm4, ymm4)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------2

	vmovupd(ymm7, mem(rcx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------3

	vmovupd(ymm9, mem(rcx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------4

	vmovupd(ymm11, mem(rcx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------5

	vmovupd(ymm13, mem(rcx, 0*32))
	vmaskmovpd(ymm14, ymm15, mem(rcx, 1*32))

	//-----------------------6



	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	vmulpd(ymm0, ymm3, ymm3)
	vmulpd(ymm0, ymm4, ymm4)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	C_TRANSPOSE_6x7_TILE_BZ(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.RESETPARAM)
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask
	jmp(.DDONE)

	label(.DDONE)
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X7I)                    // iterate again if ii != 0.


	label(.DRETURN)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[m_iter]   "m" (m_iter),
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[ps_a8]    "m" (ps_a8),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[mask_vec] "m" (mask_vec),
	[rs_c]     "m" (rs_c),
	[n0]       "m" (n0),
	[cs_c]     "m" (cs_c)
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm7", "ymm8", "ymm9",
	  "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
	  "ymm15",
	  "memory"
	)

	consider_edge_cases_7:
	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		gemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x7,
			bli_dgemmsup_rv_haswell_asm_2x7,
			bli_dgemmsup_rv_haswell_asm_3x7,
			bli_dgemmsup_rv_haswell_asm_4x7,
			bli_dgemmsup_rv_haswell_asm_5x7
		};

		gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}

}

static void bli_dgemmsup_rv_haswell_asm_6x5m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     )
{
// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  0  |  0  | 0   |  ----> Mask vector( mask_1 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  0  |  0  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//
// Since we have 5 elements to load, kernel will use one normal load
// that loads 4 elements into vector register and for remainder 1 element,
// kernel is using mask_1 which is set to -1, 0, 0, 0 static that the
// 1 element will be loaded and other 3 elements will be set to 0 in destination vector.
//
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_1;

	if ( m_iter == 0 ) goto consider_edge_cases_5;

	// -------------------------------------------------------------------------
	begin_asm()

	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X5I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm3)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r10, r10, 2), rcx)


	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)

	label(.DLOOPKITER)                 // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rdi)                // load cs_c to rsi (temporarily)
	lea(mem(, rdi, 8), rdi)            // cs_c *= sizeof(double)
	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*cs_c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP
	// ---------------------------------- iteration 0
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 1
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 2
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 3
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 4
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 5
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 6
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 7
	prefetch(0, 0x00(rbx, rcx, 1))
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)       // iterate again if i != 0.

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	                                   // now avoid loading C if beta == 0
	vxorpd(ymm2, ymm2, ymm2)           // set ymm0 to zero.
	vucomisd(xmm2, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	//Loads 4 element
	vmovupd(ymm3, mem(rcx, 0*32))

	vmulpd(ymm0, ymm4, ymm4)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm1, ymm4)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1
	vmulpd(ymm0, ymm5, ymm5)
	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm5)
	vmovupd(ymm5, mem(rcx, 0*32))

	vmulpd(ymm0, ymm6, ymm6)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm3)
	vfmadd231pd(ymm3, ymm1, ymm6)
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	//-----------------------2
	vmulpd(ymm0, ymm7, ymm7)
	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm7)
	vmovupd(ymm7, mem(rbx, 0*32))

	vmulpd(ymm0, ymm8, ymm8)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm4)
	vfmadd231pd(ymm4, ymm1, ymm8)
	vmaskmovpd(ymm8, ymm15, mem(rbx, 1*32))

	add(rdi, rbx)
	//-----------------------3
	vmulpd(ymm0, ymm9, ymm9)
	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm9)
	vmovupd(ymm9, mem(rbx, 0*32))

	vmulpd(ymm0, ymm10, ymm10)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm5)
	vfmadd231pd(ymm5, ymm1, ymm10)
	vmaskmovpd(ymm10, ymm15, mem(rbx, 1*32))

	//-----------------------4
	vmulpd(ymm0, ymm11, ymm11)
	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm11)
	vmovupd(ymm11, mem(rdx, 0*32))

	vmulpd(ymm0, ymm12, ymm12)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm6)
	vfmadd231pd(ymm6, ymm1, ymm12)
	vmaskmovpd(ymm12, ymm15, mem(rdx, 1*32))

	add(rdi, rdx)
	//-----------------------5
	vmulpd(ymm0, ymm13, ymm13)
	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm13)
	vmovupd(ymm13, mem(rdx, 0*32))

	vmulpd(ymm0, ymm14, ymm14)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm7)
	vfmadd231pd(ymm7, ymm1, ymm14)
	vmaskmovpd(ymm14, ymm15, mem(rdx, 1*32))

	//-----------------------6

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)

	C_TRANSPOSE_6x5_TILE(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------2

	vmovupd(ymm7, mem(rcx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------3

	vmovupd(ymm9, mem(rcx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------4

	vmovupd(ymm11, mem(rcx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------5

	vmovupd(ymm13, mem(rcx, 0*32))
	vmaskmovpd(ymm14, ymm15, mem(rcx, 1*32))

	//-----------------------6



	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)

	C_TRANSPOSE_6x5_TILE_BZ(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.RESETPARAM)
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask
	jmp(.DDONE)

	label(.DDONE)
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X5I)                    // iterate again if ii != 0.


	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter]   "m" (m_iter),
      [k_iter]   "m" (k_iter),
      [k_left]   "m" (k_left),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [ps_a8]    "m" (ps_a8),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [mask_vec] "m" (mask_vec),
      [rs_c]     "m" (rs_c),
      [n0]       "m" (n0),
      [cs_c]     "m" (cs_c)/*,
      [a_next]   "m" (a_next),
      [b_next]   "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases_5:
	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		gemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x5,
			bli_dgemmsup_rv_haswell_asm_2x5,
			bli_dgemmsup_rv_haswell_asm_3x5,
			bli_dgemmsup_rv_haswell_asm_4x5,
			bli_dgemmsup_rv_haswell_asm_5x5
		};

		gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}

}

static void bli_dgemmsup_rv_haswell_asm_6x3m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     )
{

// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  -1 |  -1 | 0   |  ----> Mask vector( mask_3 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//

// kernel is using mask_3 which is set to -1, -1, -1, 0 so that the
// 3 elements will be loaded and 4th element will be set to 0 in destination vector.
//
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_3;

	if ( m_iter == 0 ) goto consider_edge_cases_nleft_3;

	begin_asm()
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b

	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X3I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm14)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8

	lea(mem(r10, r10, 1), rcx)         // rcx = 3*cs_a;

	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)

	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 1

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 2

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 4

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 5

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 6

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 7

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), r9)                // load cs_c to rsi (temporarily)
	lea(mem(, r9, 8), r9)            // cs_c *= sizeof(double)
	lea(mem(r12, r9, 2), rdx)         //
	lea(mem(rdx, r9, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, r9, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, r9, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, r9, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 7*cs_c

	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r9, 8), r9)              // rs_a *= sizeof(double)

	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP

	// ---------------------------------- iteration 0

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 1

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 2

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 4

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 5

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 6

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 7

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)                   // iterate again if i != 0.

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.

	label(.DLOOPKLEFT)                 // EDGE LOOP

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  4*rs_c;

	//Loads 3 elements as per mask_3 mask vector
	vmulpd(ymm0, ymm4, ymm4)
	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm1)
	vfmadd231pd(ymm1, ymm3, ymm4)
	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmulpd(ymm0, ymm6, ymm6)
	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm3, ymm6)
	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))

	vmulpd(ymm0, ymm8, ymm8)
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm1)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vmaskmovpd(ymm8, ymm15, mem(rbx, 0*32))
	add(rdi, rbx)

	vmulpd(ymm0, ymm10, ymm10)
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm3, ymm10)
	vmaskmovpd(ymm10, ymm15, mem(rbx, 0*32))

	vmulpd(ymm0, ymm12, ymm12)
	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm1)
	vfmadd231pd(ymm1, ymm3, ymm12)
	vmaskmovpd(ymm12, ymm15, mem(rdx, 0*32))
	add(rdi, rdx)

	vmulpd(ymm0, ymm14, ymm14)
	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm3, ymm14)
	vmaskmovpd(ymm14, ymm15, mem(rdx, 0*32))

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	C_TRANSPOSE_6x3_TILE(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm8, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm10, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm12, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm14, ymm15, mem(rcx, 0*32))


	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORBZ)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	C_TRANSPOSE_6x3_TILE_BZ(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DDONE)
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X3I)                    // iterate again if ii != 0.

	label(.DRETURN)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[m_iter]   "m" (m_iter),
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[ps_a8]    "m" (ps_a8),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[rs_c]     "m" (rs_c),
	[n0]       "m" (n0),
	[mask_vec] "m" (mask_vec),
	[cs_c]     "m" (cs_c)/*,
	[a_next]   "m" (a_next),
	[b_next]   "m" (b_next)*/
	: // register clobber list
	"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	"xmm0", "xmm1", "xmm2", "xmm3",
	"xmm4", "xmm5", "xmm6", "xmm7",
	"xmm8", "xmm9", "xmm10", "xmm11",
	"xmm12", "xmm13", "xmm14", "xmm15",
	"ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	"ymm6", "ymm8", "ymm10", "ymm12", "ymm14",
	"ymm15", "memory"
	)

	consider_edge_cases_nleft_3:
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		gemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x3,
			bli_dgemmsup_rv_haswell_asm_2x3,
			bli_dgemmsup_rv_haswell_asm_3x3,
			bli_dgemmsup_rv_haswell_asm_4x3,
			bli_dgemmsup_rv_haswell_asm_5x3
		};

		gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
			conja, conjb, m_left, nr_cur, k0,
			alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}

}


static void bli_dgemmsup_rv_haswell_asm_6x1m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx

     )
{

// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  0   |  0  | 0   |  ----> Mask vector( mask_1 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  0  |  0  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//

// kernel is using mask_1 which is set to -1, 0, 0, 0 so that the
// 1 element will be loaded and 4th element will be set to 0 in destination vector.
//
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 8;
	uint64_t k_left = k0 % 8;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_1;

	if ( m_iter == 0 ) goto consider_edge_cases_nleft_1;

	begin_asm()
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b

	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X1I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm1)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8

	mov(var(k_iter), rsi)              // i = k_iter;
	sub(imm(0x10), rsi)
	jle(.DLOOPKLOOP2)

	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 1
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)


	// ---------------------------------- iteration 2
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)

	// ---------------------------------- iteration 4
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 5
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)


	// ---------------------------------- iteration 6
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 7
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DLOOPKLOOP2)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), r9)                // load cs_c to rsi (temporarily)
	lea(mem(, r9, 8), r9)            // cs_c *= sizeof(double)
	lea(mem(r12, r9, 2), rdx)         //
	lea(mem(rdx, r9, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, r9, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, r9, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, r9, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, r9, 2, 7*8)) // prefetch c + 7*cs_c

	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r9, 8), r9)              // rs_a *= sizeof(double)

	label(.DPOSTPFETCH)                // done prefetching c

	add(imm(0x10), rsi)
	jle(.DCONSIDKLEFT)

	label(.DLOOPKITERPOSTPREFETCH)                 // MAIN LOOP

	// ---------------------------------- iteration 0
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	prefetch(0, 0x40(rax))
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 1
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	prefetch(0, 0x40(rax, r8, 1))
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)


	// ---------------------------------- iteration 2
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	prefetch(0, 0x40(rax, r8, 2))
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	prefetch(0, 0x40(rax, r13, 1))
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)

	// ---------------------------------- iteration 4
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	prefetch(0, 0x40(rax, r8, 4))
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 5
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	prefetch(0, 0x40(rax, r15, 1))
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)


	// ---------------------------------- iteration 6
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 7
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITERPOSTPREFETCH)                   // iterate again if i != 0.

	vaddpd(ymm5, ymm4, ymm4)
	vaddpd(ymm7, ymm6, ymm6)
	vaddpd(ymm9, ymm8, ymm8)
	vaddpd(ymm11, ymm10, ymm10)
	vaddpd(ymm13, ymm12, ymm12)
	vaddpd(ymm1, ymm14, ymm14)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                           // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	//Loads 1 element as per mask_1 mask vector
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm1)
	vfmadd231pd(ymm1, ymm3, ymm4)
	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmulpd(ymm0, ymm6, ymm6)
	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm3, ymm6)
	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))

	vmulpd(ymm0, ymm8, ymm8)
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm4)
	vfmadd231pd(ymm4, ymm3, ymm8)
	vmaskmovpd(ymm8, ymm15, mem(rbx, 0*32))
	add(rdi, rbx)

	vmulpd(ymm0, ymm10, ymm10)
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm1)
	vfmadd231pd(ymm1, ymm3, ymm10)
	vmaskmovpd(ymm10, ymm15, mem(rbx, 0*32))

	vmulpd(ymm0, ymm12, ymm12)
	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm3, ymm12)
	vmaskmovpd(ymm12, ymm15, mem(rdx, 0*32))
	add(rdi, rdx)

	vmulpd(ymm0, ymm14, ymm14)
	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm4)
	vfmadd231pd(ymm4, ymm3, ymm14)
	vmaskmovpd(ymm14, ymm15, mem(rdx, 0*32))

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	C_TRANSPOSE_6x1_TILE(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm8, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm10, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm12, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm14, ymm15, mem(rcx, 0*32))


	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORBZ)
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	C_TRANSPOSE_6x1_TILE_BZ(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DDONE)
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X1I)                    // iterate again if ii != 0.

	label(.DRETURN)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[m_iter]   "m" (m_iter),
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[ps_a8]    "m" (ps_a8),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[rs_c]     "m" (rs_c),
	[n0]       "m" (n0),
	[mask_vec] "m" (mask_vec),
	[cs_c]     "m" (cs_c)/*,
	[a_next]   "m" (a_next),
	[b_next]   "m" (b_next)*/
	: // register clobber list
	"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	"xmm0", "xmm1", "xmm2", "xmm3",
	"xmm4", "xmm5", "xmm6", "xmm7",
	"xmm8", "xmm9", "xmm10", "xmm11",
	"xmm12", "xmm13", "xmm14", "xmm15",
	"ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	"ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	"ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)

	consider_edge_cases_nleft_1:
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		gemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x1,
			bli_dgemmsup_rv_haswell_asm_2x1,
			bli_dgemmsup_rv_haswell_asm_3x1,
			bli_dgemmsup_rv_haswell_asm_4x1,
			bli_dgemmsup_rv_haswell_asm_5x1
		};

		gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
			conja, conjb, m_left, nr_cur, k0,
			alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}

}

