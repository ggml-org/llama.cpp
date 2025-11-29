/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Forschunszentrum Juelich
   Copyright (C) 2020, The University of Tokyo

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

// Single-precision composite instructions.
#include "armsve_asm_macros_scomplex.h"

// 2vx10 microkernels.
#include "armsve_asm_2vx10cmplx.h"


void bli_cgemm_armsve_asm_2vx10_unindexed
     (
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha, \
       const void*      a, \
       const void*      b, \
       const void*      beta, \
             void*      c, inc_t rs_c0, inc_t cs_c0, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     )
{
  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k / 4;
  uint64_t k_left = k % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  uint64_t info   = 0;

  GEMM_UKR_SETUP_CT_ANY( c, m, 10, false );

  __asm__ volatile (
" whilelo         p0.s, xzr, %12                  \n\t"
// " ldr             x0, %[a]                        \n\t"
// " ldr             x1, %[b]                        \n\t"
" mov             x2, xzr                         \n\t"
" incw            x2, ALL, MUL #1                 \n\t" // Column-skip of A.
" mov             x3, #10                         \n\t" // Row-skip of B.
"                                                 \n\t"
// " ldr             x2, %[c]                        \n\t"
// " ldr             x3, %[rs_c]                     \n\t" // Row-skip of C.
// " ldr             x4, %[cs_c]                     \n\t" // Column-skip of C.
#ifdef _A64FX
" mov             x16, 0x1                        \n\t" // Tag A address.
" lsl             x16, x16, #56                   \n\t"
" orr             %0, %0, x16                     \n\t"
" mov             x16, 0x2                        \n\t" // Tag B address.
" lsl             x16, x16, #56                   \n\t"
" orr             %1, %1, x16                     \n\t"
" mov             x16, 0x3                        \n\t" // Tag C address.
" lsl             x16, x16, #56                   \n\t"
" orr             %2, %2, x16                     \n\t"
#endif
"                                                 \n\t"
" mov             x16, #8                         \n\t" // Multiply some address skips by sizeof(scomplex).
" madd            x2, x16, x2, xzr                \n\t" // cs_a
" madd            x3, x16, x3, xzr                \n\t" // rs_b
" madd            %4, x16, %4, xzr                \n\t" // cs_c
"                                                 \n\t"
// " ldr             x5, %[k_mker]                   \n\t" // Number of loops.
// " ldr             x6, %[k_left]                   \n\t"
"                                                 \n\t"
LABEL(LOAD_ABC)
" cmp             %5, #0                          \n\t" // Don't preload if no microkernel there.
BEQ(END_CCOL_PRFM)
"                                                 \n\t"
" ld1rw           z20.s, p0/z, [%1, 4*0]          \n\t" // Load B's real 8/10, no imaginary.
" ld1rw           z21.s, p0/z, [%1, 4*2]          \n\t"
" ld1rw           z22.s, p0/z, [%1, 4*4]          \n\t"
" ld1rw           z23.s, p0/z, [%1, 4*6]          \n\t"
" ld1rw           z24.s, p0/z, [%1, 4*8]          \n\t"
" ld1rw           z25.s, p0/z, [%1, 4*10]         \n\t"
" ld1rw           z26.s, p0/z, [%1, 4*12]         \n\t"
" ld1rw           z27.s, p0/z, [%1, 4*14]         \n\t"
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
"                                                 \n\t"
LABEL(CCOL_PRFM)
" cmp             %3, #1                          \n\t"
BNE(END_CCOL_PRFM) // Do not prefetch for generic C storage.
" mov             x16, %2                         \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, %4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
LABEL(END_CCOL_PRFM)
"                                                 \n\t"
CLEAR_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19)
"                                                 \n\t"
" cmp             %5, #0                          \n\t" // If no 4-microkernel can be applied.
BEQ(K_LEFT_LOOP)
"                                                 \n\t"
LABEL(K_MKER_LOOP)
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z30,z31,p0,%0,x2)
GEMM_2VX10CMPLX_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
GEMM_2VX10CMPLX_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z30,z31,p0,%0,x2)
GEMM_2VX10CMPLX_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
" subs            %5, %5, #1                      \n\t" // Decrease counter before final replica.
BEQ(FIN_MKER_LOOP) // Branch early to avoid reading excess mem.
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
GEMM_2VX10CMPLX_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
BRANCH(K_MKER_LOOP)
"                                                 \n\t"
LABEL(FIN_MKER_LOOP)
GEMM_2VX10CMPLX_MKER_LOOP_PLAIN_C_2_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
LABEL(K_LEFT_LOOP)
" cmp             %6, #0                          \n\t" // End of execution.
BEQ(WRITE_MEM_PREP)
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
" ld1rw           z20.s, p0/z, [%1, 4*0]          \n\t" // Load B's real 8/10, no imaginary.
" ld1rw           z21.s, p0/z, [%1, 4*2]          \n\t"
" ld1rw           z22.s, p0/z, [%1, 4*4]          \n\t"
" ld1rw           z23.s, p0/z, [%1, 4*6]          \n\t"
" ld1rw           z24.s, p0/z, [%1, 4*8]          \n\t"
" ld1rw           z25.s, p0/z, [%1, 4*10]         \n\t"
" ld1rw           z26.s, p0/z, [%1, 4*12]         \n\t"
" ld1rw           z27.s, p0/z, [%1, 4*14]         \n\t"
GEMM_2VX10CMPLX_MKER_LOOP_PLAIN_C_1_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
" sub             %6, %6, #1                      \n\t"
BRANCH(K_LEFT_LOOP)
"                                                 \n\t"
LABEL(WRITE_MEM_PREP)
"                                                 \n\t"
// " ldr             x7, %[alpha]                    \n\t" // Load alpha & beta (address).
// " ldr             x8, %[beta]                     \n\t"
" ld1rw           z28.s, p0/z, [%7]               \n\t" // Real(alpha).
" ld1rw           z29.s, p0/z, [%7, 4]            \n\t" // Imag(alpha).
" ld1rw           z30.s, p0/z, [%8]               \n\t" // Real(beta).
" ld1rw           z31.s, p0/z, [%8, 4]            \n\t" // Imag(beta).
"                                                 \n\t"
LABEL(PREFETCH_ABNEXT)
// " ldr             x9,  %[a_next]                  \n\t"
// " ldr             x10, %[b_next]                  \n\t"
#ifdef _A64FX
" mov             x16, 0x1                        \n\t" // Tag A address.
" lsl             x16, x16, #56                   \n\t"
" orr             %9, %9, x16                     \n\t"
" mov             x16, 0x2                        \n\t" // Tag B address.
" lsl             x16, x16, #56                   \n\t"
" orr             %10, %10, x16                   \n\t"
#endif
" prfm            PLDL1STRM, [%9]                 \n\t"
" prfm            PLDL1STRM, [%9, 256*1]          \n\t"
" prfm            PLDL1STRM, [%10]                \n\t"
" prfm            PLDL1STRM, [%10, 256*1]         \n\t"
"                                                 \n\t"
LABEL(WRITE_MEM)
" fmov            s27, #1.0                       \n\t"
" fcmp            s29, #0.0                       \n\t" // Whether Imag(alpha) == 0.
" fccmp           s28, s27, 0, eq                 \n\t" // Whether Real(alpha) == 1.
BEQ(UNIT_ALPHA)
"                                                 \n\t"
GEMM_FMULCMPLX_COL2(z20,z21,z22,z23,p0,z0 ,z1 ,z2 ,z3 ,z28,z29)
GEMM_FMULCMPLX_COL2(z24,z25,z26,z27,p0,z4 ,z5 ,z6 ,z7 ,z28,z29)
GEMM_FMULCMPLX_COL2(z0 ,z1 ,z2 ,z3 ,p0,z8, z9, z10,z11,z28,z29)
GEMM_FMULCMPLX_COL2(z4 ,z5 ,z6 ,z7 ,p0,z12,z13,z14,z15,z28,z29)
GEMM_FMULCMPLX_COL2(z8 ,z9 ,z10,z11,p0,z16,z17,z18,z19,z28,z29)
BRANCH(WRITE_MEM_EXEC)
"                                                 \n\t"
LABEL(UNIT_ALPHA)
MOV_COL2(z20,z21,z22,z23,z0 ,z1 ,z2 ,z3 )
MOV_COL2(z24,z25,z26,z27,z4 ,z5 ,z6 ,z7 )
MOV_COL2(z0 ,z1 ,z2 ,z3 ,z8, z9, z10,z11)
MOV_COL2(z4 ,z5 ,z6 ,z7 ,z12,z13,z14,z15)
MOV_COL2(z8 ,z9 ,z10,z11,z16,z17,z18,z19)
"                                                 \n\t"
LABEL(WRITE_MEM_EXEC)
" mov             x9, %2                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is %2 itself.
" cmp             %3, #1                          \n\t"
BNE(WRITE_MEM_G)
"                                                 \n\t"
LABEL(WRITE_MEM_C)
" fmov            s29, wzr                        \n\t"
" fcmp            s31, #0.0                       \n\t" // Whether Imag(beta) == 0.
" fccmp           s30, s29, 0, eq                 \n\t" // Whether Real(beta) == 0.
BEQ(ZERO_BETA_C_0_1_2_3)
GEMM_CCMPLX_LOAD_COL2_C(z12,z13,z14,z15,p0,x9,%4)
GEMM_CCMPLX_LOAD_COL2_C(z16,z17,z18,z19,p0,x9,%4)
GEMM_FMLACMPLX_COL2(z20,z21,z22,z23,p0,z12,z13,z14,z15,z30,z31)
GEMM_FMLACMPLX_COL2(z24,z25,z26,z27,p0,z16,z17,z18,z19,z30,z31)
LABEL(ZERO_BETA_C_0_1_2_3)
GEMM_CCMPLX_STORE_COL2_C(z20,z21,z22,z23,p0,%2,%4)
GEMM_CCMPLX_STORE_COL2_C(z24,z25,z26,z27,p0,%2,%4)
"                                                 \n\t"
BEQ(ZERO_BETA_C_4_5_6_7_8_9)
GEMM_CCMPLX_LOAD_COL2_C(z12,z13,z14,z15,p0,x9,%4)
GEMM_CCMPLX_LOAD_COL2_C(z16,z17,z18,z19,p0,x9,%4)
GEMM_CCMPLX_LOAD_COL2_C(z20,z21,z22,z23,p0,x9,%4)
GEMM_FMLACMPLX_COL2(z0 ,z1 ,z2 ,z3 ,p0,z12,z13,z14,z15,z30,z31)
GEMM_FMLACMPLX_COL2(z4 ,z5 ,z6 ,z7 ,p0,z16,z17,z18,z19,z30,z31)
GEMM_FMLACMPLX_COL2(z8 ,z9 ,z10,z11,p0,z20,z21,z22,z23,z30,z31)
LABEL(ZERO_BETA_C_4_5_6_7_8_9)
GEMM_CCMPLX_STORE_COL2_C(z0 ,z1 ,z2 ,z3 ,p0,%2,%4)
GEMM_CCMPLX_STORE_COL2_C(z4 ,z5 ,z6 ,z7 ,p0,%2,%4)
GEMM_CCMPLX_STORE_COL2_C(z8 ,z9 ,z10,z11,p0,%2,%4)
BRANCH(END_WRITE_MEM)
// General-storage case -- Mainly for Column-storage or other aligned cases.
LABEL(WRITE_MEM_G)
" add             %3, %3, %3                      \n\t" // Skips passed to index is multiplied by 2,
" index           z28.s, wzr, %w3                 \n\t" //  s.t. 2*sizeof(float) = 2*4 = 8.
" fmov            s29, wzr                        \n\t"
" fcmp            s31, #0.0                       \n\t" // Whether Imag(beta) == 0.
" fccmp           s30, s29, 0, eq                 \n\t" // Whether Real(beta) == 0.
BEQ(ZERO_BETA_G_0_1_2_3)
GEMM_CCMPLX_LOAD_COL2_G(z12,z13,z14,z15,p0,z28,x9,%4,x16)
GEMM_CCMPLX_LOAD_COL2_G(z16,z17,z18,z19,p0,z28,x9,%4,x16)
GEMM_FMLACMPLX_COL2(z20,z21,z22,z23,p0,z12,z13,z14,z15,z30,z31)
GEMM_FMLACMPLX_COL2(z24,z25,z26,z27,p0,z16,z17,z18,z19,z30,z31)
LABEL(ZERO_BETA_G_0_1_2_3)
GEMM_CCMPLX_STORE_COL2_G(z20,z21,z22,z23,p0,z28,%2,%4,x16)
GEMM_CCMPLX_STORE_COL2_G(z24,z25,z26,z27,p0,z28,%2,%4,x16)
"                                                 \n\t"
BEQ(ZERO_BETA_G_4_5_6_7_8_9)
GEMM_CCMPLX_LOAD_COL2_G(z12,z13,z14,z15,p0,z28,x9,%4,x16)
GEMM_CCMPLX_LOAD_COL2_G(z16,z17,z18,z19,p0,z28,x9,%4,x16)
GEMM_CCMPLX_LOAD_COL2_G(z20,z21,z22,z23,p0,z28,x9,%4,x16)
GEMM_FMLACMPLX_COL2(z0 ,z1 ,z2 ,z3 ,p0,z12,z13,z14,z15,z30,z31)
GEMM_FMLACMPLX_COL2(z4 ,z5 ,z6 ,z7 ,p0,z16,z17,z18,z19,z30,z31)
GEMM_FMLACMPLX_COL2(z8 ,z9 ,z10,z11,p0,z20,z21,z22,z23,z30,z31)
LABEL(ZERO_BETA_G_4_5_6_7_8_9)
GEMM_CCMPLX_STORE_COL2_G(z0 ,z1 ,z2 ,z3 ,p0,z28,%2,%4,x16)
GEMM_CCMPLX_STORE_COL2_G(z4 ,z5 ,z6 ,z7 ,p0,z28,%2,%4,x16)
GEMM_CCMPLX_STORE_COL2_G(z8 ,z9 ,z10,z11,p0,z28,%2,%4,x16)
"                                                 \n\t"
LABEL(END_WRITE_MEM)
BRANCH(END_EXEC)
"                                                 \n\t"
LABEL(END_EXEC)
" mov             %11, #0                         \n\t" // Return normal.
: "+r" (a),      // %0
  "+r" (b),      // %1
  "+r" (c),      // %2
  "+r" (rs_c),   // %3
  "+r" (cs_c),   // %4
  "+r" (k_mker), // %5
  "+r" (k_left), // %6
  "+r" (alpha),  // %7
  "+r" (beta),   // %8
  "+r" (a_next), // %9
  "+r" (b_next), // %10
  "=r" (info)    // %11
: "r"  (m)       // %12
: "x2","x3","x9","x16",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
  );

  GEMM_UKR_FLUSH_CT( c );
}

