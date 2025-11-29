/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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
#include <assert.h>

// Double-precision composite instructions.
#include "../armsve_asm_macros_double.h"

// 2vx10 microkernels.
#include "../armsve_asm_2vx10.h"

// Prototype reference kernel.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_armsve_ref2 )

void __attribute__ ((optimize(0))) bli_dgemmsup_rv_armsve_2vx10_unindexed
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  static int called = 0;
  if ( !called )
  {
    fprintf(stderr, "rv called.\n");
    called = 1;
  }
  // r*r requires B to be stored in rows.
  assert(cs_b0 == 1);

  dim_t n0_mker = n0 / 10;
  dim_t n0_left = n0 % 10;

  if ( n0_left )
  {
    // A[:, ::]
    // B[::, n0_mker*10:n0]
    // C[: , n0_mker*10:n0]
    double *ai = a;
    double *bi = b + n0_mker * 10 * cs_b0;
    double *ci = c + n0_mker * 10 * cs_c0;
    bli_dgemmsup_r_armsve_ref2
    (
      conja, conjb,
      m0, n0_left, k0,
      alpha,
      ai, rs_a0, cs_a0,
      bi, rs_b0, cs_b0,
      beta,
      ci, rs_c0, cs_c0,
      data,
      cntx
    );
  }
  // Return if it's a pure edge case.
  if ( !n0_mker )
    return;

  // Determine VL.
  uint64_t vlen2;
  __asm__ (
    " mov  x0, xzr          \n\t"
    " incd x0, ALL, MUL #2  \n\t"
    " mov  %[vlen2], x0     \n\t"
  : [vlen2] "=r" (vlen2)
  :
  : "x0"
   );

  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  // uint64_t cs_b   = 1;

  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;
  uint64_t m_mker = m0 / vlen2;
  uint64_t m_left = m0 % vlen2;
  if ( m_left )
  {
    // Edge case on A side can be handled with one more (predicated) loop.
    m_mker++;
  } else
    m_left = vlen2;
  uint64_t ps_a = bli_auxinfo_ps_a( data );
  // uint64_t ps_b = bli_auxinfo_ps_b( data );

  for ( dim_t in0_mker = 0; in0_mker < n0_mker; ++in0_mker )
  {
    double *ai = a;
    double *bi = b + in0_mker * 10 * cs_b0;
    double *ci = c + in0_mker * 10 * cs_c0;

    void* a_next = bli_auxinfo_next_a( data );
    void* b_next = bli_auxinfo_next_b( data );

    __asm__ volatile (
" ldr             x0, %[ai]                       \n\t"
" ldr             x1, %[rs_a]                     \n\t" // Row-skip of A (element skip of A[:, l]).
" ldr             x2, %[cs_a]                     \n\t" // Column-skip of A.
" ldr             x3, %[ps_a]                     \n\t" // Panel-skip (vlen2*k) of A.
" ldr             x4, %[rs_b]                     \n\t" // Row-Skip of B.
"                                                 \n\t" // Element skip of B[l, :] is guaranteed to be 1.
" ldr             x5, %[ci]                       \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
#ifdef _A64FX
" mov             x16, 0x1                        \n\t" // Tag C address.
" lsl             x16, x16, #56                   \n\t"
" orr             x5, x5, x16                     \n\t"
" mov             x16, 0x2                        \n\t" // Tag A address.
" lsl             x16, x16, #56                   \n\t"
" orr             x0, x0, x16                     \n\t"
#endif
"                                                 \n\t"
" mov             x8, #8                          \n\t" // Multiply some address skips by sizeof(double).
" madd            x2, x8, x2, xzr                 \n\t" // cs_a
" madd            x3, x8, x3, xzr                 \n\t" // ps_a
" madd            x4, x8, x4, xzr                 \n\t" // rs_b
" madd            x7, x8, x7, xzr                 \n\t" // cs_c
" mov             x8, xzr                         \n\t"
" incb            x8                              \n\t"
" madd            x14, x8, x1, xzr                \n\t" // A-column's logical 1-vector skip.
" mov             x8, #4                          \n\t"
" madd            x15, x8, x2, xzr                \n\t" // Logical K=4 microkernel skip for A.
// " mov             x8, #4                          \n\t"
// " madd            x17, x8, x4, xzr                \n\t" // Logical K=4 microkernel skip for B.
"                                                 \n\t"
" ldr             x8, %[m_mker]                   \n\t" // Number of M-loops.
" ptrue           p0.d                            \n\t"
" ptrue           p1.d                            \n\t"
" ptrue           p2.d                            \n\t"
"                                                 \n\t"
" MILLIKER_MLOOP:                                 \n\t"
"                                                 \n\t"
" cmp             x8, #1                          \n\t"
" b.ne            UKER_BEGIN                      \n\t"
"                                                 \n\t"
" ldr             x10, %[m_left]                  \n\t" // Final (incomplete) millikernel loop.
" mov             x11, xzr                        \n\t"
" incd            x11                             \n\t"
" whilelo         p1.d, xzr, x10                  \n\t" // Overwrite p1/p2.
" whilelo         p2.d, x11, x10                  \n\t"
"                                                 \n\t"
" UKER_BEGIN:                                     \n\t"
" mov             x10, x0                         \n\t" // A's address.
" ldr             x11, %[bi]                      \n\t" // B's address.
" ldr             x12, %[k_mker]                  \n\t"
" ldr             x13, %[k_left]                  \n\t"
#ifdef _A64FX
" mov             x16, 0x3                        \n\t" // Tag B address.
" lsl             x16, x16, #56                   \n\t"
" orr             x11, x11, x16                   \n\t"
#endif
"                                                 \n\t"
" mov             x16, x11                        \n\t" // Prefetch first kernel of B.
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
" add             x16, x16, x4                    \n\t"
" prfm            PLDL1KEEP, [x16]                \n\t"
"                                                 \n\t"
" ld1rd           z20.d, p0/z, [x11]              \n\t" // (Partial) first B row.
" ld1rd           z21.d, p0/z, [x11, #8]          \n\t"
" ld1rd           z22.d, p0/z, [x11, #16]         \n\t"
" ld1rd           z23.d, p0/z, [x11, #24]         \n\t"
" ld1rd           z24.d, p0/z, [x11, #32]         \n\t"
" ld1rd           z25.d, p0/z, [x11, #40]         \n\t"
" ld1rd           z26.d, p0/z, [x11, #48]         \n\t"
" ld1rd           z27.d, p0/z, [x11, #56]         \n\t"
"                                                 \n\t"
" index           z29.d, xzr, x1                  \n\t" // First A column.
"                                                 \n\t" // Skips passed to index is not multiplied by 8.
GEMM_ACOL_GATHER_LOAD(z28,z29,z29,p1,p2,x10,x14,x16)
"                                                 \n\t"
CLEAR_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19)
"                                                 \n\t"
" cmp             x12, #0                         \n\t" // If no 4-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t" // Unroll the 4-loop.
"                                                 \n\t"
" index           z31.d, xzr, x1                  \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_G(z30,z31,z31,p1,p2,x10,x15,x3,x2,x14,x16,noprfm,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_C_1(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x11,x4)
"                                                 \n\t"
" index           z29.d, xzr, x1                  \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_G(z28,z29,z29,p1,p2,x10,x15,x3,x2,x14,x16,noprfm,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_C_2(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x11,x4)
"                                                 \n\t"
" index           z31.d, xzr, x1                  \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_G(z30,z31,z31,p1,p2,x10,x15,x3,x2,x14,x16,noprfm,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_C_3(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x11,x4)
"                                                 \n\t"
" subs            x12, x12, #1                    \n\t" // Decrease counter before final replica.
" b.eq            FIN_MKER_LOOP                   \n\t" // Branch early to avoid reading excess mem.
"                                                 \n\t"
" index           z29.d, xzr, x1                  \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_G(z28,z29,z29,p1,p2,x10,x15,x3,x2,x14,x16,noprfm,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_C_4(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x11,x4)
" b               K_MKER_LOOP                     \n\t"
"                                                 \n\t"
" FIN_MKER_LOOP:                                  \n\t"
GEMM_2VX10_MKER_LOOP_PLAIN_C_4_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x11,x4)
" add             x10, x10, x2                    \n\t" // Forward A to fill the blank.
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t"
" cmp             x13, #0                         \n\t"
" b.eq            WRITE_MEM_PREP                  \n\t"
"                                                 \n\t"
" index           z31.d, xzr, x1                  \n\t"
GEMM_ACOL_GATHER_LOAD(z30,z31,z31,p1,p2,x10,x14,x16)
" ld1rd           z20.d, p0/z, [x11]              \n\t"
" ld1rd           z21.d, p0/z, [x11, #8]          \n\t"
" ld1rd           z22.d, p0/z, [x11, #16]         \n\t"
" ld1rd           z23.d, p0/z, [x11, #24]         \n\t"
" ld1rd           z24.d, p0/z, [x11, #32]         \n\t"
" ld1rd           z25.d, p0/z, [x11, #40]         \n\t"
" ld1rd           z26.d, p0/z, [x11, #48]         \n\t"
" ld1rd           z27.d, p0/z, [x11, #56]         \n\t"
" ld1rd           z28.d, p0/z, [x11, #64]         \n\t"
" ld1rd           z29.d, p0/z, [x11, #72]         \n\t"
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
" add             x10, x10, x2                    \n\t" // Forward A.
" add             x11, x11, x4                    \n\t" // Forward B.
" sub             x13, x13, #1                    \n\t"
" b               K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" WRITE_MEM_PREP:                                 \n\t"
"                                                 \n\t"
" ldr             x11, %[bi]                      \n\t"
" ldr             x12, %[alpha]                   \n\t" // Load alpha & beta.
" ldr             x13, %[beta]                    \n\t"
" ld1rd           z30.d, p0/z, [x12]              \n\t"
" ld1rd           z31.d, p0/z, [x13]              \n\t"
" ldr             x12, [x12]                      \n\t"
"                                                 \n\t"
" cmp             x8, #1                          \n\t"
" b.eq            PREFETCH_ABNEXT                 \n\t"
" prfm            PLDL2STRM, [x11]                \n\t"
" b               WRITE_MEM                       \n\t"
"                                                 \n\t"
" PREFETCH_ABNEXT:                                \n\t"
" ldr             x1, %[a_next]                   \n\t" // Final Millikernel loop, x1 and x2 not needed.
" ldr             x2, %[b_next]                   \n\t"
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
" prfm            PLDL2KEEP, [x1, 256*10]         \n\t"
" prfm            PLDL2KEEP, [x1, 256*11]         \n\t"
" prfm            PLDL2KEEP, [x1, 256*12]         \n\t"
" prfm            PLDL2KEEP, [x1, 256*13]         \n\t"
" prfm            PLDL2KEEP, [x1, 256*14]         \n\t"
" prfm            PLDL2KEEP, [x1, 256*15]         \n\t"
" prfm            PLDL2KEEP, [x2]                 \n\t"
" prfm            PLDL2KEEP, [x2, 256*1]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*2]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*3]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*4]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*5]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*6]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*7]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*8]          \n\t"
" prfm            PLDL2KEEP, [x2, 256*9]          \n\t"
"                                                 \n\t"
" WRITE_MEM:                                      \n\t"
"                                                 \n\t"
" fmov            d28, #1.0                       \n\t"
" fmov            x16, d28                        \n\t"
" cmp             x16, x12                        \n\t"
" b.eq            UNIT_ALPHA                      \n\t"
"                                                 \n\t"
SCALE_COL20(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z30)
"                                                 \n\t"
" UNIT_ALPHA:                                     \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
" mov             x10, x5                         \n\t" // C address for storing.
" cmp             x6, #1                          \n\t"
" b.ne            WRITE_MEM_G                     \n\t"
"                                                 \n\t"
" WRITE_MEM_C:                                    \n\t" // Available scratch: Z[20-30].
"                                                 \n\t" // Here used scratch: Z[20-29].
" mov             x13, xzr                        \n\t" // C-column's physical 1-vector skip.
" incb            x13                             \n\t"
GEMM_C_LOAD_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,x9,x7)
GEMM_C_FMAD_UKER(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z31)
GEMM_C_LOAD_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,x9,x7)
"                                                 \n\t"
GEMM_C_STORE_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,x10,x7)
GEMM_C_FMAD_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z31)
GEMM_C_STORE_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,x10,x7)
" b               END_WRITE_MEM                   \n\t"
"                                                 \n\t"
" WRITE_MEM_G:                                    \n\t" // Available scratch: Z[20-30].
"                                                 \n\t" // Here used scratch: Z[20-30] - Z30 as index.
" mov             x12, xzr                        \n\t"
" incb            x12                             \n\t"
" madd            x13, x12, x6, xzr               \n\t" // C-column's logical 1-vector skip.
" index           z30.d, xzr, x6                  \n\t" // Skips passed to index is not multiplied by 8.
GEMM_C_LOAD_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p1,p2,x9,x7,x13,x16)
GEMM_C_FMAD_UKER(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z31)
GEMM_C_LOAD_UKER_G(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z30,p1,p2,x9,x7,x13,x16)
"                                                 \n\t"
GEMM_C_STORE_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p1,p2,x10,x7,x13,x16)
GEMM_C_FMAD_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z31)
GEMM_C_STORE_UKER_G(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z30,p1,p2,x10,x7,x13,x16)
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t"
" subs            x8, x8, #1                      \n\t"
" b.eq            END_EXEC                        \n\t"
"                                                 \n\t"
" add             x0, x0, x3                      \n\t" // Forward A's base address to the next logic panel.
" add             x5, x5, x13                     \n\t" // Forward C's base address to the next logic panel.
" add             x5, x5, x13                     \n\t"
" b               MILLIKER_MLOOP                  \n\t"
"                                                 \n\t"
" END_ERROR:                                      \n\t"
" mov             x0, #1                          \n\t" // Return error.
" END_EXEC:                                       \n\t"
" mov             x0, #0                          \n\t" // Return normal.
:
: [ai]     "m" (ai),
  [rs_a]   "m" (rs_a),
  [cs_a]   "m" (cs_a),
  [ps_a]   "m" (ps_a),
  [rs_b]   "m" (rs_b),
  [ci]     "m" (ci),
  [rs_c]   "m" (rs_c),
  [cs_c]   "m" (cs_c),
  [m_mker] "m" (m_mker),
  [m_left] "m" (m_left),
  [bi]     "m" (bi),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next)
: "x0","x1","x2","x3","x4","x5","x6","x7","x8",
  "x9","x10","x11","x12","x13","x14","x15","x16",//"x17",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
     );
  }
}

