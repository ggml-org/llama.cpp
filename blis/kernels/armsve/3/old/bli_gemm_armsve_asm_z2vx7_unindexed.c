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

// Double-precision composite instructions.
#include "armsve_asm_macros_dcomplex.h"

// 2vx7 microkernels.
#include "armsve_asm_2vx7cmplx.h"


void bli_zgemm_armsve_asm_2vx7_unindexed
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a,
       dcomplex*  restrict b,
       dcomplex*  restrict beta,
       dcomplex*  restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  void* a_next = bli_auxinfo_next_a( data );
  void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k / 4;
  uint64_t k_left = k % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  uint64_t info   = 0;

  uint64_t mr = bli_vl_bytes_armsve() * 2 / 16;
  GEMM_UKR_SETUP_CT( z, mr, 7, false );

  __asm__ volatile (
// " ldr             x0, %[a]                        \n\t"
// " ldr             x1, %[b]                        \n\t"
" mov             x2, xzr                         \n\t"
" incd            x2, ALL, MUL #1                 \n\t" // Column-skip of A.
" mov             x3, #7                          \n\t" // Row-skip of B.
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
" mov             x16, #16                        \n\t" // Multiply some address skips by sizeof(dcomplex).
" madd            x2, x16, x2, xzr                \n\t" // cs_a
" madd            x3, x16, x3, xzr                \n\t" // rs_b
" madd            %4, x16, %4, xzr                \n\t" // cs_c
" ptrue           p0.d                            \n\t"
"                                                 \n\t"
// " ldr             x5, %[k_mker]                   \n\t" // Number of loops.
// " ldr             x6, %[k_left]                   \n\t"
"                                                 \n\t"
" LOAD_ABC:                                       \n\t"
" cmp             %5, #0                          \n\t" // Don't preload if no microkernel there.
" b.eq            END_CCOL_PRFM                   \n\t"
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
"                                                 \n\t"
" ld1rd           z14.d, p0/z, [%1, 8*0]          \n\t" // Load B's real & imaginary.
" ld1rd           z15.d, p0/z, [%1, 8*2]          \n\t"
" ld1rd           z16.d, p0/z, [%1, 8*4]          \n\t"
" ld1rd           z17.d, p0/z, [%1, 8*6]          \n\t"
" ld1rd           z18.d, p0/z, [%1, 8*8]          \n\t"
" ld1rd           z19.d, p0/z, [%1, 8*10]         \n\t"
" ld1rd           z20.d, p0/z, [%1, 8*12]         \n\t"
" ld1rd           z21.d, p0/z, [%1, 8*1]          \n\t"
" ld1rd           z22.d, p0/z, [%1, 8*3]          \n\t"
" ld1rd           z23.d, p0/z, [%1, 8*5]          \n\t"
" ld1rd           z24.d, p0/z, [%1, 8*7]          \n\t"
" ld1rd           z25.d, p0/z, [%1, 8*9]          \n\t"
" ld1rd           z26.d, p0/z, [%1, 8*11]         \n\t"
" ld1rd           z27.d, p0/z, [%1, 8*13]         \n\t"
" add             %1, %1, x3                      \n\t"
"                                                 \n\t"
" CCOL_PRFM:                                      \n\t"
" cmp             %3, #1                          \n\t"
" b.ne            END_CCOL_PRFM                   \n\t" // Do not prefetch for generic C storage.
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
" END_CCOL_PRFM:                                  \n\t"
"                                                 \n\t"
CLEAR_COL14(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13)
"                                                 \n\t"
" cmp             %5, #0                          \n\t" // If no 4-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t"
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z30,z31,p0,%0,x2)
GEMM_2VX7CMPLX_MKER_LOOP_PLAIN_C(z0,z2,z4,z6,z8,z10,z12,z1,z3,z5,z7,z9,z11,z13,p0,z28,z29,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
GEMM_2VX7CMPLX_MKER_LOOP_PLAIN_C(z0,z2,z4,z6,z8,z10,z12,z1,z3,z5,z7,z9,z11,z13,p0,z30,z31,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z30,z31,p0,%0,x2)
GEMM_2VX7CMPLX_MKER_LOOP_PLAIN_C(z0,z2,z4,z6,z8,z10,z12,z1,z3,z5,z7,z9,z11,z13,p0,z28,z29,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
" subs            %5, %5, #1                      \n\t" // Decrease counter before final replica.
" b.eq            FIN_MKER_LOOP                   \n\t" // Branch early to avoid reading excess mem.
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
GEMM_2VX7CMPLX_MKER_LOOP_PLAIN_C(z0,z2,z4,z6,z8,z10,z12,z1,z3,z5,z7,z9,z11,z13,p0,z30,z31,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
" b               K_MKER_LOOP                     \n\t"
"                                                 \n\t"
" FIN_MKER_LOOP:                                  \n\t"
GEMM_2VX7CMPLX_MKER_LOOP_PLAIN_C_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z1,z3,z5,z7,z9,z11,z13,p0,z30,z31,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t"
" cmp             %6, #0                          \n\t" // End of execution.
" b.eq            WRITE_MEM_PREP                  \n\t"
"                                                 \n\t"
GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(z28,z29,p0,%0,x2)
" ld1rd           z14.d, p0/z, [%1, 8*0]          \n\t"
" ld1rd           z15.d, p0/z, [%1, 8*2]          \n\t"
" ld1rd           z16.d, p0/z, [%1, 8*4]          \n\t"
" ld1rd           z17.d, p0/z, [%1, 8*6]          \n\t"
" ld1rd           z18.d, p0/z, [%1, 8*8]          \n\t"
" ld1rd           z19.d, p0/z, [%1, 8*10]         \n\t"
" ld1rd           z20.d, p0/z, [%1, 8*12]         \n\t"
" ld1rd           z21.d, p0/z, [%1, 8*1]          \n\t"
" ld1rd           z22.d, p0/z, [%1, 8*3]          \n\t"
" ld1rd           z23.d, p0/z, [%1, 8*5]          \n\t"
" ld1rd           z24.d, p0/z, [%1, 8*7]          \n\t"
" ld1rd           z25.d, p0/z, [%1, 8*9]          \n\t"
" ld1rd           z26.d, p0/z, [%1, 8*11]         \n\t"
" ld1rd           z27.d, p0/z, [%1, 8*13]         \n\t"
" add             %1, %1, x3                      \n\t"
GEMM_2VX7CMPLX_MKER_LOOP_PLAIN_C_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z1,z3,z5,z7,z9,z11,z13,p0,z28,z29,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,%1,x3)
" sub             %6, %6, #1                      \n\t"
" b               K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" WRITE_MEM_PREP:                                 \n\t"
"                                                 \n\t"
// " ldr             x7, %[alpha]                    \n\t" // Load alpha & beta (address).
// " ldr             x8, %[beta]                     \n\t"
" ld1rd           z28.d, p0/z, [%7]               \n\t" // Real(alpha).
" ld1rd           z29.d, p0/z, [%7, 8]            \n\t" // Imag(alpha).
" ld1rd           z30.d, p0/z, [%8]               \n\t" // Real(beta).
" ld1rd           z31.d, p0/z, [%8, 8]            \n\t" // Imag(beta).
"                                                 \n\t"
" PREFETCH_ABNEXT:                                \n\t"
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
" WRITE_MEM:                                      \n\t"
"                                                 \n\t"
GEMM_FMULCMPLX_COL7(z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,p0,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z28,z29)
"                                                 \n\t"
" UNIT_ALPHA:                                     \n\t"
" mov             x9, %2                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is %2 itself.
" cmp             %3, #1                          \n\t"
" b.ne            WRITE_MEM_G                     \n\t"
"                                                 \n\t"
" WRITE_MEM_C:                                    \n\t"
GEMM_CCMPLX_LOAD_COL7_C(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,p0,x9,%4)
GEMM_FMLACMPLX_COL7(z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,p0,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z30,z31)
GEMM_CCMPLX_STORE_COL7_C(z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,p0,%2,%4)
" b               END_WRITE_MEM                   \n\t"
"                                                 \n\t"
" WRITE_MEM_G:                                    \n\t"
" add             %3, %3, %3                      \n\t" // Skips passed to index is multiplied by 2,
" index           z28.d, xzr, %3                  \n\t" //  s.t. 2*sizeof(double) = 2*8 = 16.
GEMM_CCMPLX_LOAD_COL7_G(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,p0,z28,x9,%4,x16)
GEMM_FMLACMPLX_COL7(z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,p0,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z30,z31)
GEMM_CCMPLX_STORE_COL7_G(z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,p0,z28,%2,%4,x16)
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t"
" b               END_EXEC                        \n\t"
"                                                 \n\t"
" END_EXEC:                                       \n\t"
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
:
: "x2","x3","x9","x16",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
  );

  GEMM_UKR_FLUSH_CT( z );
}


