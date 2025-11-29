/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

//3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14
#define C_TRANSPOSE_5x7_TILE(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10) \
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
	/*Moving to operate on last 1 row of 5 rows.*/ \
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 1x4 tile*/ \
	vmovlpd(mem(rdx        ), xmm0, xmm0)\
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)\
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)\
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)\
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)\
\
	vfmadd213pd(ymm(R5), ymm15, ymm0)\
	vextractf128(imm(1), ymm0, xmm1)\
	vmovlpd(xmm0, mem(rdx        ))\
	vmovhpd(xmm0, mem(rdx, rsi, 1))\
	vmovlpd(xmm1, mem(rdx, rsi, 2))\
	vmovhpd(xmm1, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R7), ymm(R6), ymm0)\
	vunpckhpd(ymm(R7), ymm(R6), ymm1)\
	vunpcklpd(ymm(R9), ymm(R8), ymm2)\
	vunpckhpd(ymm(R9), ymm(R8), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R6))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R7))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R8))\
\
	vfmadd231pd(mem(rcx        ), ymm15, ymm(R6))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm(R7))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm(R8))\
	vmovupd(ymm(R6), mem(rcx        ))\
	vmovupd(ymm(R7), mem(rcx, rsi, 1))\
	vmovupd(ymm(R8), mem(rcx, rsi, 2))\
\
	vmovlpd(mem(rdx        ), xmm0, xmm0)\
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)\
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)\
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)\
\
	/*Transposing 1x3 tile*/ \
	vfmadd213pd(ymm(R10), ymm15, ymm0)\
	vextractf128(imm(1), ymm0, xmm1)\
	vmovlpd(xmm0, mem(rdx        ))\
	vmovhpd(xmm0, mem(rdx, rsi, 1))\
	vmovlpd(xmm1, mem(rdx, rsi, 2))

#define C_TRANSPOSE_5x7_TILE_BZ(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10) \
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
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 1x4 tile*/ \
	vextractf128(imm(1), ymm(R5), xmm1)\
	vmovlpd(xmm(R5), mem(rdx        ))\
	vmovhpd(xmm(R5), mem(rdx, rsi, 1))\
	vmovlpd(xmm1, mem(rdx, rsi, 2))\
	vmovhpd(xmm1, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R7), ymm(R6), ymm0)\
	vunpckhpd(ymm(R7), ymm(R6), ymm1)\
	vunpcklpd(ymm(R9), ymm(R8), ymm2)\
	vunpckhpd(ymm(R9), ymm(R8), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R6))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R7))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R8))\
\
	vmovupd(ymm(R6), mem(rcx        ))\
	vmovupd(ymm(R7), mem(rcx, rsi, 1))\
	vmovupd(ymm(R8), mem(rcx, rsi, 2))\
\
	/*Transposing 1x3 tile*/ \
	vextractf128(imm(1), ymm(R10), xmm1)\
	vmovlpd(xmm(R10), mem(rdx        ))\
	vmovhpd(xmm(R10), mem(rdx, rsi, 1))\
	vmovlpd(xmm1, mem(rdx, rsi, 2))

#define C_TRANSPOSE_4x7_TILE(R1, R2, R3, R4, R5, R6, R7, R8) \
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
	vbroadcastsd(mem(rbx), ymm15)\
\
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
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vunpcklpd(ymm(R8), ymm(R7), ymm2)\
	vunpckhpd(ymm(R8), ymm(R7), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R5))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R6))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R7))\
\
	vfmadd231pd(mem(rcx        ), ymm15, ymm(R5))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm(R6))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm(R7))\
	vmovupd(ymm(R5), mem(rcx        ))\
	vmovupd(ymm(R6), mem(rcx, rsi, 1))\
	vmovupd(ymm(R7), mem(rcx, rsi, 2))

#define C_TRANSPOSE_4x7_TILE_BZ(R1, R2, R3, R4, R5, R6, R7, R8) \
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
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vunpcklpd(ymm(R8), ymm(R7), ymm2)\
	vunpckhpd(ymm(R8), ymm(R7), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R5))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R6))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R7))\
\
	vmovupd(ymm(R5), mem(rcx        ))\
	vmovupd(ymm(R6), mem(rcx, rsi, 1))\
	vmovupd(ymm(R7), mem(rcx, rsi, 2))

//3, 5, 7, 4, 6, 8
#define C_TRANSPOSE_3x7_TILE(R1, R2, R3, R4, R5, R6) \
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm10, ymm(R3), ymm2)\
	vunpckhpd(ymm10, ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm10)\
\
	/*Transposing 1x4 tile*/ \
	vextractf128(imm(0x1), ymm(R1), xmm12)\
	vextractf128(imm(0x1), ymm(R2), xmm13)\
	vextractf128(imm(0x1), ymm(R3), xmm14)\
	vextractf128(imm(0x1), ymm10, xmm15)\
\
	vbroadcastsd(mem(rbx), ymm11)\
\
	vfmadd231pd(mem(rcx        ), xmm11, xmm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), xmm11, xmm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), xmm11, xmm(R3))\
	vfmadd231pd(mem(rcx, rax, 1), xmm11, xmm10)\
	vmovupd(xmm(R1), mem(rcx        ))\
	vmovupd(xmm(R2), mem(rcx, rsi, 1))\
	vmovupd(xmm(R3), mem(rcx, rsi, 2))\
	vmovupd(xmm10, mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	vfmadd231sd(mem(rdx        ), xmm11, xmm12)\
	vfmadd231sd(mem(rdx, rsi, 1), xmm11, xmm13)\
	vfmadd231sd(mem(rdx, rsi, 2), xmm11, xmm14)\
	vfmadd231sd(mem(rdx, rax, 1), xmm11, xmm15)\
	vmovsd(xmm12, mem(rdx        ))\
	vmovsd(xmm13, mem(rdx, rsi, 1))\
	vmovsd(xmm14, mem(rdx, rsi, 2))\
	vmovsd(xmm15, mem(rdx, rax, 1))\
	\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R5), ymm(R4), ymm0)\
	vunpckhpd(ymm(R5), ymm(R4), ymm1)\
	vunpcklpd(ymm11, ymm(R6), ymm2)\
	vunpckhpd(ymm11, ymm(R6), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R4))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R5))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R6))\
\
	/*Transposing 1x3 tile*/ \
	vextractf128(imm(0x1), ymm(R4), xmm12)\
	vextractf128(imm(0x1), ymm(R5), xmm13)\
	vextractf128(imm(0x1), ymm(R6), xmm14)\
\
	vfmadd231pd(mem(rcx        ), xmm11, xmm(R4))\
	vfmadd231pd(mem(rcx, rsi, 1), xmm11, xmm(R5))\
	vfmadd231pd(mem(rcx, rsi, 2), xmm11, xmm(R6))\
	vmovupd(xmm(R4), mem(rcx        ))\
	vmovupd(xmm(R5), mem(rcx, rsi, 1))\
	vmovupd(xmm(R6), mem(rcx, rsi, 2))\
\
	vfmadd231sd(mem(rdx        ), xmm11, xmm12)\
	vfmadd231sd(mem(rdx, rsi, 1), xmm11, xmm13)\
	vfmadd231sd(mem(rdx, rsi, 2), xmm11, xmm14)\
	vmovsd(xmm12, mem(rdx        ))\
	vmovsd(xmm13, mem(rdx, rsi, 1))\
	vmovsd(xmm14, mem(rdx, rsi, 2))

#define C_TRANSPOSE_3x7_TILE_BZ(R1, R2, R3, R4, R5, R6) \
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm10, ymm(R3), ymm2)\
	vunpckhpd(ymm10, ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm10)\
\
	/*Transposing 1x4 tile*/ \
	vextractf128(imm(0x1), ymm(R1), xmm12)\
	vextractf128(imm(0x1), ymm(R2), xmm13)\
	vextractf128(imm(0x1), ymm(R3), xmm14)\
	vextractf128(imm(0x1), ymm10, xmm15)\
\
	vmovupd(xmm(R1), mem(rcx        ))\
	vmovupd(xmm(R2), mem(rcx, rsi, 1))\
	vmovupd(xmm(R3), mem(rcx, rsi, 2))\
	vmovupd(xmm10, mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
	vmovsd(xmm12, mem(rdx        ))\
	vmovsd(xmm13, mem(rdx, rsi, 1))\
	vmovsd(xmm14, mem(rdx, rsi, 2))\
	vmovsd(xmm15, mem(rdx, rax, 1))\
	\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R5), ymm(R4), ymm0)\
	vunpckhpd(ymm(R5), ymm(R4), ymm1)\
	vunpcklpd(ymm11, ymm(R6), ymm2)\
	vunpckhpd(ymm11, ymm(R6), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R4))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R5))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R6))\
\
	/*Transposing 1x3 tile*/ \
	vextractf128(imm(0x1), ymm(R4), xmm12)\
	vextractf128(imm(0x1), ymm(R5), xmm13)\
	vextractf128(imm(0x1), ymm(R6), xmm14)\
\
	vmovupd(xmm(R4), mem(rcx        ))\
	vmovupd(xmm(R5), mem(rcx, rsi, 1))\
	vmovupd(xmm(R6), mem(rcx, rsi, 2))\
\
	vmovsd(xmm12, mem(rdx        ))\
	vmovsd(xmm13, mem(rdx, rsi, 1))\
	vmovsd(xmm14, mem(rdx, rsi, 2))

//3, 5, 4, 6
#define C_TRANSPOSE_2x7_TILE(R1, R2, R3, R4) \
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm7)\
\
	vbroadcastsd(mem(rbx), ymm3)\
	vfmadd231pd(mem(rcx        ), xmm3, xmm0)\
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)\
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)\
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm7)\
	vmovupd(xmm0, mem(rcx        ))\
	vmovupd(xmm1, mem(rcx, rsi, 1))\
	vmovupd(xmm2, mem(rcx, rsi, 2))\
	vmovupd(xmm7, mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R4), ymm(R3), ymm0)\
	vunpckhpd(ymm(R4), ymm(R3), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
\
	vfmadd231pd(mem(rcx        ), xmm3, xmm0)\
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)\
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)\
	vmovupd(xmm0, mem(rcx        ))\
	vmovupd(xmm1, mem(rcx, rsi, 1))\
	vmovupd(xmm2, mem(rcx, rsi, 2))


#define C_TRANSPOSE_2x7_TILE_BZ(R1, R2, R3, R4) \
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm7)\
\
	vmovupd(xmm0, mem(rcx        ))\
	vmovupd(xmm1, mem(rcx, rsi, 1))\
	vmovupd(xmm2, mem(rcx, rsi, 2))\
	vmovupd(xmm7, mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R4), ymm(R3), ymm0)\
	vunpckhpd(ymm(R4), ymm(R3), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
\
	vmovupd(xmm0, mem(rcx        ))\
	vmovupd(xmm1, mem(rcx, rsi, 1))\
	vmovupd(xmm2, mem(rcx, rsi, 2))


#define C_TRANSPOSE_1x7_TILE(R1, R2) \
	vmovlpd(mem(rcx        ), xmm0, xmm0)\
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)\
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)\
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)\
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)\
\
	vbroadcastsd(mem(rbx), ymm15)\
	vfmadd213pd(ymm(R1), ymm15, ymm0)\
\
	vextractf128(imm(1), ymm0, xmm1)\
	vmovlpd(xmm0, mem(rcx        ))\
	vmovhpd(xmm0, mem(rcx, rsi, 1))\
	vmovlpd(xmm1, mem(rcx, rsi, 2))\
	vmovhpd(xmm1, mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	vmovlpd(mem(rcx        ), xmm0, xmm0)\
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)\
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)\
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)\
\
	vfmadd213pd(ymm(R2), ymm15, ymm0)\
\
	vextractf128(imm(1), ymm0, xmm1)\
	vmovlpd(xmm0, mem(rcx        ))\
	vmovhpd(xmm0, mem(rcx, rsi, 1))\
	vmovlpd(xmm1, mem(rcx, rsi, 2))


#define C_TRANSPOSE_1x7_TILE_BZ(R1, R2) \
	vextractf128(imm(1), ymm(R1), xmm1)\
	vmovlpd(xmm(R1), mem(rcx        ))\
	vmovhpd(xmm(R1), mem(rcx, rsi, 1))\
	vmovlpd(xmm1, mem(rcx, rsi, 2))\
	vmovhpd(xmm1, mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
	vextractf128(imm(1), ymm(R2), xmm1)\
	vmovlpd(xmm(R2), mem(rcx        ))\
	vmovhpd(xmm(R2), mem(rcx, rsi, 1))\
	vmovlpd(xmm1, mem(rcx, rsi, 2))

static const int64_t mask_3[4] = {-1, -1, -1, 0};

void bli_dgemmsup_rv_haswell_asm_5x7
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a, inc_t rs_a0, inc_t cs_a0,
       const void*      b, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

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
	int64_t const *mask_vec = mask_3;
	// -------------------------------------------------------------------------

	begin_asm()

	vzeroall()                         // zero all xmm/ymm registers.
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         6*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 6*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 6*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         6*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 6*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rsi, rsi, 2), rdx)         // rdx = 3*cs_c;
	prefetch(0, mem(rcx,         4*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         4*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 6*cs_c
	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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
	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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
	add(r9, rax)                       // a += cs_a;
	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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
	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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
	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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
	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rax)         // load address of c +  1*rs_c;
	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;
	lea(mem(rbx, rdi, 1), r8)         // load address of c +  3*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)

	vfmadd231pd(mem(rax, 0*32), ymm1, ymm5)
	vmaskmovpd(mem(rax, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm6)

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm7)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm8)

	vfmadd231pd(mem(r8, 0*32), ymm1, ymm9)
	vmaskmovpd(mem(r8, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm10)

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm11)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm12)


	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	vmovupd(ymm5, mem(rax, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rax, 1*32))

	vmovupd(ymm7, mem(rbx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rbx, 1*32))

	vmovupd(ymm9, mem(r8, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(r8, 1*32))

	vmovupd(ymm11, mem(rdx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	C_TRANSPOSE_5x7_TILE(3, 5, 7, 9, 11, 4, 6, 8, 10, 12)
	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)

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

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_5x7_TILE_BZ(3, 5, 7, 9, 11, 4, 6, 8, 10, 12)
	jmp(.DDONE)                        // jump to end.

	label(.DDONE)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[n0]       "m" (n0),
	[rs_c]     "m" (rs_c),
	[mask_vec] "m" (mask_vec),
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
	  "ymm10", "ymm11", "ymm12", "ymm15",
	  "memory"
	)
}


void bli_dgemmsup_rv_haswell_asm_4x7
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a, inc_t rs_a0, inc_t cs_a0,
       const void*      b, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

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
	int64_t const *mask_vec = mask_3;
	// -------------------------------------------------------------------------

	begin_asm()

	vzeroall()                         // zero all xmm/ymm registers.
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         6*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 6*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 6*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         6*8)) // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rsi, rsi, 2), rdx)         // rdx = 3*cs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 6*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;
	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rax)         // load address of c +  1*rs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;
	lea(mem(rdx, rdi, 1), rbx)         // load address of c +  3*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)

	vfmadd231pd(mem(rax, 0*32), ymm1, ymm5)
	vmaskmovpd(mem(rax, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm6)

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm7)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm8)

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm9)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm10)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	vmovupd(ymm5, mem(rax, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rax, 1*32))

	vmovupd(ymm7, mem(rdx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rdx, 1*32))

	vmovupd(ymm9, mem(rbx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rbx, 1*32))
	//-----------------------4

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	C_TRANSPOSE_4x7_TILE(3, 5, 7, 9, 4, 6, 8, 10)
	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)

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
	//-----------------------4
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_4x7_TILE_BZ(3, 5, 7, 9, 4, 6, 8, 10)
	jmp(.DDONE)                        // jump to end.

	label(.DDONE)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[n0]       "m" (n0),
	[rs_c]     "m" (rs_c),
	[mask_vec] "m" (mask_vec),
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
	  "ymm10", "ymm11", "ymm12", "ymm15",
	  "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_3x7
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a, inc_t rs_a0, inc_t cs_a0,
       const void*      b, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

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
	int64_t const *mask_vec = mask_3;
	// -------------------------------------------------------------------------

	begin_asm()

	vzeroall()                         // zero all xmm/ymm registers.
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         6*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 6*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 6*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rsi, rsi, 2), rdx)         // rdx = 3*cs_c;
	prefetch(0, mem(rcx,         2*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         2*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 6*cs_c
	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	add(r9, rax)                       // a += cs_a;
	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm9, ymm3, ymm3)
	vaddpd(ymm10, ymm4, ymm4)
	vaddpd(ymm11, ymm5, ymm5)
	vaddpd(ymm12, ymm6, ymm6)
	vaddpd(ymm13, ymm7, ymm7)
	vaddpd(ymm14, ymm8, ymm8)


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
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

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)



	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rbx)         // load address of c +  1*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm5)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm1, ymm6)

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm7)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm8)


	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	vmovupd(ymm5, mem(rbx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rbx, 1*32))

	vmovupd(ymm7, mem(rdx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rdx, 1*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	C_TRANSPOSE_3x7_TILE(3, 5, 7, 4, 6, 8)
	jmp(.DDONE)                        // jump to end.


	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)

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

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_3x7_TILE_BZ(3, 5, 7, 4, 6, 8)
	jmp(.DDONE)                        // jump to end.

	label(.DDONE)
	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[n0]       "m" (n0),
	[rs_c]     "m" (rs_c),
	[mask_vec] "m" (mask_vec),
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
	  "ymm10", "ymm11", "ymm12", "ymm13",
	  "ymm14", "ymm15", "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_2x7
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a, inc_t rs_a0, inc_t cs_a0,
       const void*      b, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

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
	int64_t const *mask_vec = mask_3;
	// -------------------------------------------------------------------------

	begin_asm()

	vzeroall()                         // zero all xmm/ymm registers.
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         6*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 6*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rsi, rsi, 2), rdx)         // rdx = 3*cs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 6*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm14)
	vfmadd231pd(ymm0, ymm14, ymm5)
	vfmadd231pd(ymm1, ymm14, ymm6)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm10)
	vfmadd231pd(ymm1, ymm2, ymm11)

	vbroadcastsd(mem(rax, r8,  1), ymm14)
	vfmadd231pd(ymm0, ymm14, ymm12)
	vfmadd231pd(ymm1, ymm14, ymm13)

	add(r9, rax)                       // a += cs_a;
	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm14)
	vfmadd231pd(ymm0, ymm14, ymm5)
	vfmadd231pd(ymm1, ymm14, ymm6)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm10)
	vfmadd231pd(ymm1, ymm2, ymm11)

	vbroadcastsd(mem(rax, r8,  1), ymm14)
	vfmadd231pd(ymm0, ymm14, ymm12)
	vfmadd231pd(ymm1, ymm14, ymm13)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm10, ymm3, ymm3)
	vaddpd(ymm11, ymm4, ymm4)
	vaddpd(ymm12, ymm5, ymm5)
	vaddpd(ymm13, ymm6, ymm6)


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm14)
	vfmadd231pd(ymm0, ymm14, ymm5)
	vfmadd231pd(ymm1, ymm14, ymm6)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rdx)         // load address of c +  1*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm5)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm1, ymm6)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	vmovupd(ymm5, mem(rdx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	C_TRANSPOSE_2x7_TILE(3, 5, 4, 6)
	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_2x7_TILE_BZ(3, 5, 4, 6)
	jmp(.DDONE)                        // jump to end.

	label(.DDONE)
	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[n0]       "m" (n0),
	[rs_c]     "m" (rs_c),
	[mask_vec] "m" (mask_vec),
	[cs_c]     "m" (cs_c)
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm8", "ymm10","ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_1x7
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a, inc_t rs_a0, inc_t cs_a0,
       const void*      b, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

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
	int64_t const *mask_vec = mask_3;
	// -------------------------------------------------------------------------

	begin_asm()

	vzeroall()                         // zero all xmm/ymm registers.
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         6*8)) // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rsi, rsi, 2), rdx)         // rdx = 3*cs_c;
	prefetch(0, mem(rcx,         0*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         0*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 6*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)


	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm8)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm9)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm10)
	vfmadd231pd(ymm8, ymm10, ymm6)
	vfmadd231pd(ymm9, ymm10, ymm7)

	add(r9, rax)                       // a += cs_a;
	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm8)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm9)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm10)
	vfmadd231pd(ymm8, ymm10, ymm6)
	vfmadd231pd(ymm9, ymm10, ymm7)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.


	vaddpd(ymm6, ymm3, ymm3)
	vaddpd(ymm7, ymm4, ymm4)


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	C_TRANSPOSE_1x7_TILE(3, 4)
	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_1x7_TILE_BZ(3, 4)
	jmp(.DDONE)                        // jump to end.

	label(.DDONE)
	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[n0]       "m" (n0),
	[rs_c]     "m" (rs_c),
	[mask_vec] "m" (mask_vec),
	[cs_c]     "m" (cs_c)
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm12", "ymm15", "memory"
	)
}
