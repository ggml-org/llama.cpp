/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, The University of Tokyo
   Copyright (C) 2019, Forschunszentrum Juelich

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

// Half-precision composite instructions.
#include "armsve_asm_macros_half.h"

// 2vx10 microkernels.
#include "armsve_asm_2vx10.h"

// Gather-load / scatter-store instruction for half-precision
//  needs being defined separately.
#undef GEMM_CCOL_GATHER_LOAD_FWD
#undef GEMM_CCOL_SCATTER_STORE_FWD

#define GEMM_CCOL_GATHER_LOAD_FWD(ZFH,ZLH,ZIDX2,PT,CRS2,CADDR,CCS,CVSKIP,CTEMP) \
" add   x28, "#CADDR", "#CRS2"        \n\t" \
" ld1h  z31.s, "#PT"/z, ["#CADDR", "#ZIDX2".s, uxtw #1] \n\t" \
" ld1h  "#ZFH".s, "#PT"/z, [x28, "#ZIDX2".s, uxtw #1]   \n\t" \
" revh  "#ZFH".s, "#PT"/m, "#ZFH".s                     \n\t" \
" fadd  "#ZFH".h, "#ZFH".h, z31.h     \n\t" \
" add   "#CTEMP", "#CADDR", "#CVSKIP" \n\t" \
" add   x28, "#CTEMP", "#CRS2"        \n\t" \
" ld1h  z31.s, "#PT"/z, ["#CTEMP", "#ZIDX2".s, uxtw #1] \n\t" \
" ld1h  "#ZLH".s, "#PT"/z, [x28, "#ZIDX2".s, uxtw #1]   \n\t" \
" revh  "#ZLH".s, "#PT"/m, "#ZLH".s                     \n\t" \
" fadd  "#ZLH".h, "#ZLH".h, z31.h     \n\t" \
" add  "#CADDR", "#CADDR", "#CCS"     \n\t"

#define GEMM_CCOL_SCATTER_STORE_FWD(ZFH,ZLH,ZIDX2,PT,CRS2,CADDR,CCS,CVSKIP,CTEMP) \
" add   x28, "#CADDR", "#CRS2"        \n\t" \
" st1h  "#ZFH".s, "#PT", ["#CADDR", "#ZIDX2".s, uxtw #1] \n\t" \
" revh  "#ZFH".s, "#PT"/m, "#ZFH".s   \n\t" \
" st1h  "#ZFH".s, "#PT", [x28, "#ZIDX2".s, uxtw #1]      \n\t" \
" add   "#CTEMP", "#CADDR", "#CVSKIP" \n\t" \
" add   x28, "#CTEMP", "#CRS2"        \n\t" \
" st1h  "#ZLH".s, "#PT", ["#CTEMP", "#ZIDX2".s, uxtw #1] \n\t" \
" revh  "#ZLH".s, "#PT"/m, "#ZLH".s   \n\t" \
" st1h  "#ZLH".s, "#PT", [x28, "#ZIDX2".s, uxtw #1]      \n\t" \
" add   "#CADDR", "#CADDR", "#CCS"    \n\t"


void bli_shgemm_armsve_asm_2vx10_unindexed
     (
       dim_t               k0,
       void*      restrict alpha,
       void*      restrict a,
       void*      restrict b,
       void*      restrict beta,
       void*      restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  void* a_next = bli_auxinfo_next_a( data );
  void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  __asm__ volatile (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, xzr                         \n\t"
" inch            x2, ALL, MUL #2                 \n\t" // Column-skip of A.
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
" mov             x8, #2                          \n\t" // Multiply some address skips by sizeof(float16_t).
" madd            x2, x8, x2, xzr                 \n\t" // cs_a
" madd            x3, x8, x3, xzr                 \n\t" // rs_b
" madd            x7, x8, x7, xzr                 \n\t" // cs_c
" ptrue           p0.b                            \n\t"
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
" LOAD_ABC:                                       \n\t"
" cmp             x4, #0                          \n\t" // Don't preload if no microkernel there.
" b.eq            END_CCOL_PRFM                   \n\t"

" ld1rh           z20.h, p0/z, [x1]               \n\t" // Load 8/10 of first B row.
" ld1rh           z21.h, p0/z, [x1, 2]            \n\t"
" ld1rh           z22.h, p0/z, [x1, 4]            \n\t"
" ld1rh           z23.h, p0/z, [x1, 6]            \n\t"
" ld1rh           z24.h, p0/z, [x1, 8]            \n\t"
" ld1rh           z25.h, p0/z, [x1, 10]           \n\t"
" ld1rh           z26.h, p0/z, [x1, 12]           \n\t"
" ld1rh           z27.h, p0/z, [x1, 14]           \n\t"
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
"                                                 \n\t"
" CCOL_PRFM:                                      \n\t"
" cmp             x6, #1                          \n\t"
" b.ne            END_CCOL_PRFM                   \n\t" // Do not prefetch for generic C storage.
" mov             x16, x5                         \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x16, x7                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" END_CCOL_PRFM:                                  \n\t"
"                                                 \n\t"
CLEAR_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19)
"                                                 \n\t"
" cmp             x4, #0                          \n\t" // If no 4-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t"
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_3(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
"                                                 \n\t"
" subs            x4, x4, #1                      \n\t" // Decrease counter before final replica.
" b.eq            FIN_MKER_LOOP                   \n\t" // Branch early to avoid reading excess mem.
"                                                 \n\t"
" add             x0, x0, x2                      \n\t" // Forward A's address to the next column.
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p0,p0,x0)
GEMM_2VX10_MKER_LOOP_PLAIN_C_4(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
" b               K_MKER_LOOP                     \n\t"
"                                                 \n\t"
" FIN_MKER_LOOP:                                  \n\t"
GEMM_2VX10_MKER_LOOP_PLAIN_C_4_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x1,x3)
" add             x0, x0, x2                      \n\t" // Forward A to fill the blank.
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t"
" cmp             x8, #0                          \n\t" // End of execution.
" b.eq            WRITE_MEM_PREP                  \n\t"
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p0,p0,x0)
" ld1rh           z20.h, p0/z, [x1]               \n\t" // Load 8/10 of first B row.
" ld1rh           z21.h, p0/z, [x1, 2]            \n\t"
" ld1rh           z22.h, p0/z, [x1, 4]            \n\t"
" ld1rh           z23.h, p0/z, [x1, 6]            \n\t"
" ld1rh           z24.h, p0/z, [x1, 8]            \n\t"
" ld1rh           z25.h, p0/z, [x1, 10]           \n\t"
" ld1rh           z26.h, p0/z, [x1, 12]           \n\t"
" ld1rh           z27.h, p0/z, [x1, 14]           \n\t"
" ld1rh           z28.h, p0/z, [x1, 16]           \n\t"
" ld1rh           z29.h, p0/z, [x1, 18]           \n\t"
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
" b               K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" WRITE_MEM_PREP:                                 \n\t"
"                                                 \n\t"
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1rh           z30.h, p0/z, [x4]               \n\t" // Load alpha & beta into vectors.
" ld1rh           z31.h, p0/z, [x8]               \n\t"
" fmov            w4, h28                         \n\t" // Copy alpha & beta to GP registers.
" fmov            w8, h29                         \n\t"
"                                                 \n\t"
" PREFETCH_ABNEXT:                                \n\t"
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
" prfm            PLDL2KEEP, [x0]                 \n\t"
" prfm            PLDL2KEEP, [x0, 256*1]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*2]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*3]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*4]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*5]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*6]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*7]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*8]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*9]          \n\t"
" prfm            PLDL2KEEP, [x0, 256*10]         \n\t"
" prfm            PLDL2KEEP, [x0, 256*11]         \n\t"
" prfm            PLDL2KEEP, [x0, 256*12]         \n\t"
" prfm            PLDL2KEEP, [x0, 256*13]         \n\t"
" prfm            PLDL2KEEP, [x0, 256*14]         \n\t"
" prfm            PLDL2KEEP, [x0, 256*15]         \n\t"
" prfm            PLDL2KEEP, [x1]                 \n\t"
" prfm            PLDL2KEEP, [x1, 256*1]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*2]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*3]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*4]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*5]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*6]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*7]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*8]          \n\t"
" prfm            PLDL2KEEP, [x1, 256*9]          \n\t"
"                                                 \n\t"
" WRITE_MEM:                                      \n\t"
"                                                 \n\t"
" fmov            h28, #1.0                       \n\t"
" fmov            w16, h28                        \n\t"
" cmp             w16, w4                         \n\t"
" b.eq            UNIT_ALPHA                      \n\t"
"                                                 \n\t"
SCALE_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z30)
"                                                 \n\t"
" UNIT_ALPHA:                                     \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x6, #1                          \n\t"
" b.ne            WRITE_MEM_G                     \n\t"
"                                                 \n\t"
" WRITE_MEM_C:                                    \n\t" // Available scratch: Z[20-30].
"                                                 \n\t" // Here used scratch: Z[20-29].
GEMM_C_LOAD_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p0,x9,x7)
GEMM_C_FMAD_UKER(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p0,z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z31)
GEMM_C_LOAD_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,p0,x9,x7)
"                                                 \n\t"
GEMM_C_STORE_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p0,x5,x7)
GEMM_C_FMAD_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,p0,z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z31)
GEMM_C_STORE_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,p0,x5,x7)
" b               END_WRITE_MEM                   \n\t"
"                                                 \n\t"
" WRITE_MEM_G:                                    \n\t" // Available scratch: Z[20-30].
"                                                 \n\t" // Here used scratch: Z[20-30] - Z30 as index.
" mov             x10, xzr                        \n\t"
" incb            x10                             \n\t"
" madd            x10, x10, x6, xzr               \n\t" // C-column's logical 1-vector skip.
" mov             x28, #2                         \n\t"
" madd            x6, x28, x6, xzr                \n\t" // Double index skip for half-precision case.
" index           z30.s, wzr, w6                  \n\t" // Skips passed to index is not multiplied by 8.
GEMM_C_LOAD_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p0,x6,x9,x7,x10,x16)
" dup             z31.h, w8                       \n\t" // Restore beta destroyed by loading.
GEMM_C_FMAD_UKER(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p0,p0,z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z31)
GEMM_C_LOAD_UKER_G(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z30,p0,x6,x9,x7,x10,x16)
"                                                 \n\t"
" dup             z31.h, w8                       \n\t" // Restore beta destroyed by loading.
GEMM_C_STORE_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p0,x6,x5,x7,x10,x16)
GEMM_C_FMAD_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p0,p0,z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z31)
GEMM_C_STORE_UKER_G(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z30,p0,x6,x5,x7,x10,x16)
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t"
" b               END_EXEC                        \n\t"
"                                                 \n\t"
" END_ERROR:                                      \n\t"
" mov             x0, #1                          \n\t" // Return error.
" END_EXEC:                                       \n\t"
" mov             x0, #0                          \n\t" // Return normal.
:
: [a]      "m" (a),
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
  "x9","x16","x10","x28",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
   );
}

