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
#include "armsve_asm_macros_double.h"

// 2vx10 microkernels.
#include "armsve_asm_2vx10.h"


void bli_dgemm_armsve_asm_2vx10_unindexed
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

  GEMM_UKR_SETUP_CT_ANY( d, m, 10, false );

  __asm__ volatile (
" mov             x0, xzr                         \n\t"
" ldr             x1, %[m]                        \n\t"
" whilelo         p0.d, x0, x1 \n\t" " incd x0    \n\t"
" whilelo         p1.d, x0, x1                    \n\t"
"                                                 \n\t"
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, xzr                         \n\t"
" incd            x2, ALL, MUL #2                 \n\t" // Column-skip of A.
" mov             x3, #10                         \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
#ifdef _A64FX
" mov             x8, 0x3                         \n\t" // Tag C address.
" lsl             x8, x8, #56                     \n\t"
" orr             x5, x5, x8                      \n\t"
" mov             x8, 0x2                         \n\t" // Tag B address.
" lsl             x8, x8, #56                     \n\t"
" orr             x1, x1, x8                      \n\t"
" mov             x8, 0x1                         \n\t" // Tag A address.
" lsl             x8, x8, #56                     \n\t"
" orr             x0, x0, x8                      \n\t"
#endif
"                                                 \n\t"
" mov             x8, #8                          \n\t" // Multiply some address skips by sizeof(double).
" madd            x2, x8, x2, xzr                 \n\t" // cs_a
" madd            x3, x8, x3, xzr                 \n\t" // rs_b
" madd            x7, x8, x7, xzr                 \n\t" // cs_c
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
LABEL(LOAD_ABC)
" cmp             x4, #0                          \n\t" // Don't preload if no microkernel there.
BEQ(END_CCOL_PRFM)

" ld1rd           z20.d, p0/z, [x1]               \n\t" // Load 8/10 of first B row.
" ld1rd           z21.d, p0/z, [x1, 8]            \n\t"
" ld1rd           z22.d, p0/z, [x1, 16]           \n\t"
" ld1rd           z23.d, p0/z, [x1, 24]           \n\t"
" ld1rd           z24.d, p0/z, [x1, 32]           \n\t"
" ld1rd           z25.d, p0/z, [x1, 40]           \n\t"
" ld1rd           z26.d, p0/z, [x1, 48]           \n\t"
" ld1rd           z27.d, p0/z, [x1, 56]           \n\t"
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p1,x0)
"                                                 \n\t"
LABEL(CCOL_PRFM)
" cmp             x6, #1                          \n\t"
BNE(END_CCOL_PRFM) // Do not prefetch for generic C storage.
" mov             x16, x5                         \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
LABEL(END_CCOL_PRFM)
"                                                 \n\t"
CLEAR_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19)
"                                                 \n\t"
" cmp             x4, #0                          \n\t" // If no 4-microkernel can be applied
BEQ(K_LEFT_LOOP)
"                                                 \n\t"
LABEL(K_MKER_LOOP)
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p1,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p1,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p1,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_3(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
"                                                 \n\t"
" subs            x4, x4, #1                      \n\t" // Decrease counter before final replica.
BEQ(FIN_MKER_LOOP) // Branch early to avoid reading excess mem.
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p1,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_4(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
BRANCH(K_MKER_LOOP)
"                                                 \n\t"
LABEL(FIN_MKER_LOOP)
GEMM_2VX10_MKER_LOOP_PLAIN_C_4_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
" add             x0, x0, x2                      \n\t" // Forward A to fill the blank.
"                                                 \n\t"
LABEL(K_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of execution.
BEQ(WRITE_MEM_PREP)
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p1,x0)
" ld1rd           z20.d, p0/z, [x1]               \n\t" // Load 8/10 of first B row.
" ld1rd           z21.d, p0/z, [x1, 8]            \n\t"
" ld1rd           z22.d, p0/z, [x1, 16]           \n\t"
" ld1rd           z23.d, p0/z, [x1, 24]           \n\t"
" ld1rd           z24.d, p0/z, [x1, 32]           \n\t"
" ld1rd           z25.d, p0/z, [x1, 40]           \n\t"
" ld1rd           z26.d, p0/z, [x1, 48]           \n\t"
" ld1rd           z27.d, p0/z, [x1, 56]           \n\t"
" ld1rd           z28.d, p0/z, [x1, 64]           \n\t"
" ld1rd           z29.d, p0/z, [x1, 72]           \n\t"
GEMM_FMLA2(z0,z1,p0,z30,z31,z20)
GEMM_FMLA2(z2,z3,p0,z30,z31,z21)
GEMM_FMLA2(z4,z5,p0,z30,z31,z22)
GEMM_FMLA2(z6,z7,p0,z30,z31,z23)
GEMM_FMLA2(z8,z9,p0,z30,z31,z24)
GEMM_FMLA2(z10,z11,p0,z30,z31,z25)
GEMM_FMLA2(z12,z13,p0,z30,z31,z26)
GEMM_FMLA2(z14,z15,p0,z30,z31,z27)
GEMM_FMLA2(z16,z17,p0,z30,z31,z28)
GEMM_FMLA2(z18,z19,p0,z30,z31,z29)
" add             x0, x0, x2                      \n\t" // Forward A.
" add             x1, x1, x3                      \n\t" // Forward B.
" sub             x8, x8, #1                      \n\t"
BRANCH(K_LEFT_LOOP)
"                                                 \n\t"
LABEL(WRITE_MEM_PREP)
"                                                 \n\t"
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ldr             x4, [x4]                        \n\t" // Load alpha & beta (value).
" ldr             x8, [x8]                        \n\t"
" dup             z30.d, x4                       \n\t" // Broadcast alpha & beta into vectors.
" dup             z31.d, x8                       \n\t"
" fmov            d28, #1.0                       \n\t" // Prepare FP 1.0.
" fmov            x16, d28                        \n\t"
"                                                 \n\t"
LABEL(PREFETCH_ABNEXT)
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
#ifdef _A64FX
" mov             x8, 0x2                         \n\t" // Tag B address.
" lsl             x8, x8, #56                     \n\t"
" orr             x1, x1, x8                      \n\t"
" mov             x8, 0x1                         \n\t" // Tag A address.
" lsl             x8, x8, #56                     \n\t"
" orr             x0, x0, x8                      \n\t"
#endif
" prfm            PLDL1STRM, [x0]                 \n\t"
" prfm            PLDL1STRM, [x0, 256*1]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*2]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*3]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*4]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*5]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*6]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*7]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*8]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*9]          \n\t"
// " prfm            PLDL2KEEP, [x0, 256*10]         \n\t"
// " prfm            PLDL2KEEP, [x0, 256*11]         \n\t"
// " prfm            PLDL2KEEP, [x0, 256*12]         \n\t"
// " prfm            PLDL2KEEP, [x0, 256*13]         \n\t"
// " prfm            PLDL2KEEP, [x0, 256*14]         \n\t"
// " prfm            PLDL2KEEP, [x0, 256*15]         \n\t"
" prfm            PLDL1STRM, [x1]                 \n\t"
" prfm            PLDL1STRM, [x1, 256*1]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*2]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*3]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*4]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*5]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*6]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*7]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*8]          \n\t"
// " prfm            PLDL2KEEP, [x1, 256*9]          \n\t"
"                                                 \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x6, #1                          \n\t" // Preload first half of C for contiguous case.
BNE(WRITE_MEM)
GEMM_C_LOAD_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p1,x9,x7)
"                                                 \n\t"
LABEL(WRITE_MEM)
"                                                 \n\t"
" cmp             x16, x4                         \n\t"
BEQ(UNIT_ALPHA)
"                                                 \n\t"
SCALE_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z30)
"                                                 \n\t"
LABEL(UNIT_ALPHA)
" cmp             x6, #1                          \n\t"
BNE(WRITE_MEM_G)
"                                                 \n\t"
LABEL(WRITE_MEM_C)
"                                                 \n\t" // Available scratch: Z[20-30].
"                                                 \n\t" // Here used scratch: Z[20-29].
" fcmp            d31, #0.0                       \n\t" // Skip loading if *beta == 0 to override NaN.
BEQ(BETA_ZERO_C)
// First half of C is already loaded in this case.
// GEMM_C_FMAD_LOAD_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p1,z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z31,x9,x7)
GEMM_C_FMLA_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z31)
GEMM_C_LOAD_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p1,x9,x7)
GEMM_C_FMLA_UKER(z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,p0,z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z31)
LABEL(BETA_ZERO_C)
GEMM_C_STORE_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,p1,x5,x7)
GEMM_C_STORE_UKER_C(z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,p0,p1,x5,x7)
BRANCH(END_EXEC)
// Generic-storage case -- Mainly for transposed storage.
LABEL(WRITE_MEM_G)
" mov             x8, xzr                         \n\t"
" incb            x8                              \n\t"
" madd            x8, x8, x6, xzr                 \n\t" // C-column's logical 1-vector skip.
" index           z30.d, xzr, x6                  \n\t" // Skips passed to index is not multiplied by 8.
"                                                 \n\t"
" fcmp            d31, #0.0                       \n\t" // Skip loading if *beta == 0 to override NaN.
BEQ(BETA_ZERO_G)
GEMM_C_LOAD_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p0,p1,x9,x7,x8,x16)
GEMM_C_FMLA_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z31)
GEMM_C_LOAD_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p0,p1,x9,x7,x8,x16)
GEMM_C_FMLA_UKER(z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,p0,z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z31)
LABEL(BETA_ZERO_G)
GEMM_C_STORE_UKER_G(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z30,p0,p1,x5,x7,x8,x16)
GEMM_C_STORE_UKER_G(z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z30,p0,p1,x5,x7,x8,x16)
LABEL(END_EXEC)
" mov             x0, #0                          \n\t" // Return normal.
:
: [m]      "m" (m),
  [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_c]   "m" (rs_c),
  [cs_c]   "m" (cs_c),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next)
: "x0","x1","x2","x3","x4","x5","x6","x7","x8",
  "x9","x16",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
   );

  GEMM_UKR_FLUSH_CT( d );
}

