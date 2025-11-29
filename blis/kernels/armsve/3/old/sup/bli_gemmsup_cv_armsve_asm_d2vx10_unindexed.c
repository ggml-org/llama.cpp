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
GEMMSUP_KER_PROT( double,   d, gemmsup_c_armsve_ref2 )

void __attribute__ ((noinline,optimize(0))) bli_dgemmsup_cv_armsve_2vx10_unindexed
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
  // c*c requires A to be stored in columns.
  assert( rs_a0 == 1 );

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
    bli_dgemmsup_c_armsve_ref2
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
  // uint64_t rs_a   = 1;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;

  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;
  uint64_t n_mker = n0_mker;

  dim_t m0_mker = m0 / vlen2;
  dim_t m0_left = m0 % vlen2;
  if ( m0_left )
  {
    // Edge case on A side can be handled with one more (predicated) loop.
    m0_mker++;
  } else
    m0_left = vlen2;
  // uint64_t ps_a = bli_auxinfo_ps_a( data );
  uint64_t ps_b = bli_auxinfo_ps_b( data );

  for ( dim_t im0_mker = 0; im0_mker < m0_mker; ++im0_mker )
  {
    uint64_t m_curr = vlen2;
    if ( im0_mker == m0_mker - 1 )
    {
      // Last m-loop. Maybe unnecessary.
      m_curr = m0_left;
    }
    double *ai = a + im0_mker * vlen2 * rs_a0;
    double *bi = b;
    double *ci = c + im0_mker * vlen2 * rs_c0;

    void* a_next = bli_auxinfo_next_a( data );
    void* b_next = bli_auxinfo_next_b( data );

    __asm__ volatile (
" ldr             x0, %[bi]                       \n\t"
" ldr             x1, %[rs_b]                     \n\t" // Row-skip of B.
" ldr             x2, %[cs_b]                     \n\t" // Column-skip of B (element skip of B[l, :]).
" ldr             x3, %[ps_b]                     \n\t" // Panel-skip (10*k) of B.
" ldr             x4, %[cs_a]                     \n\t" // Column-Skip of A.
"                                                 \n\t" // Element skip of A[:, l] is guaranteed to be 1.
" ldr             x5, %[ci]                       \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
#ifdef _A64FX
" mov             x16, 0x1                        \n\t" // Tag C address.
" lsl             x16, x16, #56                   \n\t"
" orr             x5, x5, x16                     \n\t"
" mov             x16, 0x2                        \n\t" // Tag B address.
" lsl             x16, x16, #56                   \n\t"
" orr             x0, x0, x16                     \n\t"
#endif
"                                                 \n\t"
" mov             x8, #8                          \n\t" // Multiply some address skips by sizeof(double).
" madd            x1, x8, x1, xzr                 \n\t" // rs_b
" madd            x2, x8, x2, xzr                 \n\t" // cs_b
" madd            x3, x8, x3, xzr                 \n\t" // ps_b
" madd            x4, x8, x4, xzr                 \n\t" // cs_a
" madd            x7, x8, x7, xzr                 \n\t" // cs_c
" mov             x8, #4                          \n\t"
" madd            x15, x8, x4, xzr                \n\t" // Logical K=4 microkernel skip for A.
"                                                 \n\t"
#ifdef _A64FX
" mov             x16, 0x20                       \n\t" // Higher 6bit for Control#2:
" lsl             x16, x16, #58                   \n\t" // Valid|Strong|Strong|NoAlloc|Load|Strong
" orr             x16, x16, x4                    \n\t" // Stride.
" msr             S3_3_C11_C6_2, x16              \n\t" // Write system register.
#endif
"                                                 \n\t"
" ldr             x8, %[m_curr]                   \n\t" // Size of first dimension.
" mov             x9, xzr                         \n\t"
" incd            x9                              \n\t"
" ptrue           p0.d                            \n\t"
" whilelo         p1.d, xzr, x8                   \n\t"
" whilelo         p2.d, x9, x8                    \n\t"
"                                                 \n\t"
" ldr             x8, %[n_mker]                   \n\t" // Number of N-loops.
"                                                 \n\t"
" ldr             x20, %[ai]                      \n\t" // Parameters to be reloaded
" ldr             x21, %[k_mker]                  \n\t" //  within each millikernel loop.
" ldr             x22, %[k_left]                  \n\t"
" ldr             x23, %[alpha]                   \n\t"
" ldr             x24, %[beta]                    \n\t"
" ldr             x25, %[a_next]                  \n\t"
" ldr             x26, %[b_next]                  \n\t"
" ldr             x23, [x23]                      \n\t" // Directly load alpha and beta.
" ldr             x24, [x24]                      \n\t"
"                                                 \n\t"
" MILLIKER_MLOOP:                                 \n\t"
"                                                 \n\t"
" mov             x11, x0                         \n\t" // B's address.
// " ldr             x10, %[ai]                      \n\t" // A's address.
" mov             x10, x20                        \n\t"
// " ldr             x12, %[k_mker]                  \n\t"
" mov             x12, x21                        \n\t"
// " ldr             x13, %[k_left]                  \n\t"
" mov             x13, x22                        \n\t"
#ifdef _A64FX
" mov             x16, 0x3                        \n\t" // Tag A address.
" lsl             x16, x16, #56                   \n\t"
" orr             x10, x10, x16                   \n\t"
" mov             x16, 0xa                        \n\t" // Control#2 for A address.
" lsl             x16, x16, #60                   \n\t"
" orr             x10, x10, x16                   \n\t"
#endif
"                                                 \n\t"
" cmp             x12, #0                         \n\t" // Don't preload if no microkernel there.
" b.eq            END_CCOL_PRFM                   \n\t"
"                                                 \n\t"
" mov             x14, x11                        \n\t"
" ld1rd           z20.d, p0/z, [x14]              \n\t" // Load 8/10 of first B row.
" add             x14, x14, x2                    \n\t"
" ld1rd           z21.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z22.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z23.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z24.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z25.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z26.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z27.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" prfm            PLDL1KEEP, [x14]                \n\t" // And prefetch the 2/10 left.
" add             x14, x14, x2                    \n\t"
" prfm            PLDL1KEEP, [x14]                \n\t"
" sub             x14, x14, x2                    \n\t" // Restore x14 to load edge.
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z28,z29,p1,p2,x10)
" add             x16, x10, x4                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t" // Prefetch 3/4 of A.
" add             x16, x10, x4                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
" add             x16, x10, x4                    \n\t"
" prfm            PLDL1STRM, [x16]                \n\t"
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
" cmp             x12, #0                         \n\t" // If no 4-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t"
"                                                 \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_C(z30,z31,p1,p2,x10,x15,x4,x16,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_G_1(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x11,x14,x1,x2)
"                                                 \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_C(z28,z29,p1,p2,x10,x15,x4,x16,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_G_2(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x11,x14,x1,x2)
"                                                 \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_C(z30,z31,p1,p2,x10,x15,x4,x16,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_G_3(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z28,z29,z20,z21,z22,z23,z24,z25,z26,z27,x11,x14,x1,x2)
"                                                 \n\t"
" subs            x12, x12, #1                    \n\t" // Decrease counter before final replica.
" b.eq            FIN_MKER_LOOP                   \n\t" // Branch early to avoid reading excess mem.
"                                                 \n\t"
GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_C(z28,z29,p1,p2,x10,x15,x4,x16,noprfm)
GEMM_2VX10_MKER_LOOP_PLAIN_G_4(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x11,x14,x1,x2)
" b               K_MKER_LOOP                     \n\t"
"                                                 \n\t"
" FIN_MKER_LOOP:                                  \n\t"
GEMM_2VX10_MKER_LOOP_PLAIN_G_4_RESIDUAL(z0,z2,z4,z6,z8,z10,z12,z14,z16,z18,z1,z3,z5,z7,z9,z11,z13,z15,z17,z19,p0,z30,z31,z20,z21,z22,z23,z24,z25,z26,z27,x11,x14,x1,x2)
" add             x10, x10, x4                    \n\t" // Forward A to fill the blank.
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t"
" cmp             x13, #0                         \n\t" // End of execution.
" b.eq            WRITE_MEM_PREP                  \n\t"
"                                                 \n\t"
GEMM_ACOL_CONTIGUOUS_LOAD(z30,z31,p1,p2,x10)
" mov             x14, x11                        \n\t"
" ld1rd           z20.d, p0/z, [x14]              \n\t" // Load 10/10 B.
" add             x14, x14, x2                    \n\t"
" ld1rd           z21.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z22.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z23.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z24.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z25.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z26.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z27.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z28.d, p0/z, [x14]              \n\t"
" add             x14, x14, x2                    \n\t"
" ld1rd           z29.d, p0/z, [x14]              \n\t"
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
" add             x10, x10, x4                    \n\t" // Forward A.
" add             x11, x11, x1                    \n\t" // Forward B.
" sub             x13, x13, #1                    \n\t"
" b               K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" WRITE_MEM_PREP:                                 \n\t"
"                                                 \n\t"
// " ldr             x10, %[ai]                      \n\t"
" mov             x10, x20                        \n\t"
" add             x11, x0, x3                     \n\t"
" dup             z30.d, x23                      \n\t" // Broadcast alpha & beta into vectors.
" dup             z31.d, x24                      \n\t"
"                                                 \n\t"
" cmp             x8, #1                          \n\t"
" b.eq            PREFETCH_ABNEXT                 \n\t"
" prfm            PLDL1STRM, [x10]                \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" add             x11, x11, x2                    \n\t"
" prfm            PLDL1KEEP, [x11]                \n\t"
" b               WRITE_MEM                       \n\t"
"                                                 \n\t"
" PREFETCH_ABNEXT:                                \n\t"
// " ldr             x1, %[a_next]                   \n\t" // Final Millikernel loop, x1 and x2 not needed.
" mov             x1, x25                         \n\t"
// " ldr             x2, %[b_next]                   \n\t"
" mov             x2, x26                         \n\t"
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
" cmp             x16, x23                        \n\t"
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
" mov             x13, xzr                        \n\t" // C-column's physical 1-vector skip.
" incb            x13                             \n\t"
GEMM_C_LOAD_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,x9,x7)
GEMM_C_FMAD_UKER(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z31)
GEMM_C_LOAD_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,x9,x7)
"                                                 \n\t"
GEMM_C_STORE_UKER_C(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,p1,p2,x5,x7)
GEMM_C_FMAD_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z31)
GEMM_C_STORE_UKER_C(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,x5,x7)
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
GEMM_C_STORE_UKER_G(z20,z22,z24,z26,z28,z21,z23,z25,z27,z29,z30,p1,p2,x5,x7,x13,x16)
GEMM_C_FMAD_UKER(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,p1,p2,z10,z12,z14,z16,z18,z11,z13,z15,z17,z19,z31)
GEMM_C_STORE_UKER_G(z0,z2,z4,z6,z8,z1,z3,z5,z7,z9,z30,p1,p2,x5,x7,x13,x16)
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t"
" subs            x8, x8, #1                      \n\t"
" b.eq            END_EXEC                        \n\t"
"                                                 \n\t" // Address of C already forwarded to next column.
" add             x0, x0, x3                      \n\t" // Forward B's base address to the next logic panel.
" b               MILLIKER_MLOOP                  \n\t"
"                                                 \n\t"
" END_ERROR:                                      \n\t"
" mov             x0, #1                          \n\t" // Return error.
" END_EXEC:                                       \n\t"
" mov             x0, #0                          \n\t" // Return normal.
:
: [bi]     "m" (bi),
  [rs_b]   "m" (rs_b),
  [cs_b]   "m" (cs_b),
  [ps_b]   "m" (ps_b),
  [cs_a]   "m" (cs_a),
  [ci]     "m" (ci),
  [rs_c]   "m" (rs_c),
  [cs_c]   "m" (cs_c),
  [m_curr] "m" (m_curr),
  [n_mker] "m" (n_mker),
  [ai]     "m" (ai),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next)
: "x0","x1","x2","x3","x4","x5","x6","x7","x8",
  "x9","x10","x11","x12","x13","x14","x15","x16","x17",
  "x20","x21","x22","x23","x24","x25","x26",
  "z0","z1","z2","z3","z4","z5","z6","z7",
  "z8","z9","z10","z11","z12","z13","z14","z15",
  "z16","z17","z18","z19",
  "z20","z21","z22","z23",
  "z24","z25","z26","z27",
  "z28","z29","z30","z31"
     );
  }
}

void bli_dgemmsup_rv_armsve_10x2v_unindexed
     (
       conj_t              conjat,
       conj_t              conjbt,
       dim_t               m0t,
       dim_t               n0t,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict at, inc_t rs_at0, inc_t cs_at0,
       double*    restrict bt, inc_t rs_bt0, inc_t cs_bt0,
       double*    restrict beta,
       double*    restrict ct, inc_t rs_ct0, inc_t cs_ct0,
       auxinfo_t* restrict datat,
       cntx_t*    restrict cntx
     )
{
  auxinfo_t data;
  bli_auxinfo_set_next_a( bli_auxinfo_next_b( datat ), &data );
  bli_auxinfo_set_next_b( bli_auxinfo_next_a( datat ), &data );
  bli_auxinfo_set_ps_a( bli_auxinfo_ps_b( datat ), &data );
  bli_auxinfo_set_ps_b( bli_auxinfo_ps_a( datat ), &data );
  bli_dgemmsup_cv_armsve_2vx10_unindexed
  (
    conjbt, conjat,
    n0t, m0t, k0,
    alpha,
    bt, cs_bt0, rs_bt0,
    at, cs_at0, rs_at0,
    beta,
    ct, cs_ct0, rs_ct0,
    &data,
    cntx
  );
}

