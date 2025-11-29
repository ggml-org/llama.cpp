/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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
#include "armsve512_asm_transpose_d8x2.h"
#include "armsve512_asm_transpose_d8x8.h"
#include "../3/armsve_asm_macros.h"

// assumption:
//   SVE vector length = 512 bits.

void bli_dpackm_armsve512_asm_16x10
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim_,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n_,
             dim_t   n_max_,
       const void*   kappa,
       const void*   a, inc_t inca_, inc_t lda_,
             void*   p,              inc_t ldp_,
       const void*   params,
       const cntx_t* cntx
     )
{
    const int64_t cdim  = cdim_;
    const int64_t mr    = 16;
    const int64_t nr    = 10;
    const int64_t n     = n_;
    const int64_t n_max = n_max_;
    const int64_t inca  = inca_;
    const int64_t lda   = lda_;
    const int64_t ldp   = ldp_;
    const bool    gs    = inca != 1 && lda != 1;
    const bool    unitk = bli_deq1( *(( double* )kappa) );

// This never would have worked in the first place since GEMM packing used
// BLIS_PACKED_ROW_PANELS and BLIS_PACKED_COL_PANELS, but with the removal
// of the row/column packing bit it can't work via the schema anyways.
#if 0
#ifdef _A64FX
    {
        // Infer whether A or B is being packed.
        if ( schema == BLIS_PACKED_ROWS )
            p = ( (uint64_t)0x1 << 56 ) | (uint64_t)p;
        if ( schema == BLIS_PACKED_COLUMNS )
            p = ( (uint64_t)0x2 << 56 ) | (uint64_t)p;
    }
#endif
#endif

    if ( cdim == mr && cdim_bcast == 1 && !gs && unitk )
    {
        uint64_t n_mker = n / 8;
        uint64_t n_left = n % 8;
        __asm__ volatile (
            "mov  x0, %[a] \n\t"
            "mov  x1, %[p] \n\t"
            "mov  x2, %[ldp] \n\t"
            "mov  x3, %[lda] \n\t"
            "mov  x4, %[inca] \n\t"
            "cmp  x4, #1 \n\t"
            // Skips by sizeof(double).
            "mov  x8, #8 \n\t"
            "madd x2, x2, x8, xzr \n\t"
            "madd x3, x3, x8, xzr \n\t"
            "madd x4, x4, x8, xzr \n\t"

            // "mov  x8, 0x8 \n\t" // Control#0 for A address.
            // "mov  x8, 0x24 \n\t" // Higher 6bit for Control#0:
            // "lsl  x8, x8, #58 \n\t" // Valid|Strong|Strong|Alloc|Load|Strong
            // "orr  x8, x8, x3 \n\t" // Stride.
            // "msr  S3_3_C11_C6_0, x8 \n\t" // Write system register.

            // Loop constants.
            "mov  x8, %[n_mker] \n\t"
            "mov  x9, %[n_left] \n\t"
            "ptrue p0.d \n\t"
            BNE(MAROWSTOR)
            // A stored in columns.
            LABEL(MACOLSTOR)
            // Prefetch distance.
            "mov  x17, #8 \n\t"
            "madd x17, x17, x3, xzr \n\t"
#ifdef _A64FX
            "mov  x16, 0x6 \n\t" // Disable hardware prefetch for A.
            "lsl  x16, x16, #60 \n\t"
            "orr  x0, x0, x16 \n\t"
#endif
            // "add  x5, x0, x3 \n\t"
            // "add  x6, x5, x3 \n\t"
            // "add  x7, x6, x3 \n\t"
            // "prfm PLDL1STRM, [x0] \n\t"
            // "prfm PLDL1STRM, [x5] \n\t"
            // "prfm PLDL1STRM, [x6] \n\t"
            // "prfm PLDL1STRM, [x7] \n\t"
            // "add  x18, x7, x3 \n\t"
            // "add  x5, x18, x3 \n\t"
            // "add  x6, x5, x3 \n\t"
            // "add  x7, x6, x3 \n\t"
            // "prfm PLDL1STRM, [x18] \n\t"
            // "prfm PLDL1STRM, [x5] \n\t"
            // "prfm PLDL1STRM, [x6] \n\t"
            // "prfm PLDL1STRM, [x7] \n\t"
            LABEL(MACOLSTORMKER)
            "cmp  x8, xzr \n\t"
            BEQ(MACOLSTORMKEREND)
            "add  x5, x0, x3 \n\t"
            "add  x6, x5, x3 \n\t"
            "add  x7, x6, x3 \n\t"
            "add  x10, x1, x2 \n\t"
            "add  x11, x10, x2 \n\t"
            "add  x12, x11, x2 \n\t"
            "add  x13, x12, x2 \n\t"
            "add  x14, x13, x2 \n\t"
            "add  x15, x14, x2 \n\t"
            "add  x16, x15, x2 \n\t"
            "ld1d z0.d, p0/z, [x0] \n\t"
            "ld1d z1.d, p0/z, [x0, #1, mul vl] \n\t"
            "ld1d z2.d, p0/z, [x5] \n\t"
            "ld1d z3.d, p0/z, [x5, #1, mul vl] \n\t"
            "ld1d z4.d, p0/z, [x6] \n\t"
            "ld1d z5.d, p0/z, [x6, #1, mul vl] \n\t"
            "ld1d z6.d, p0/z, [x7] \n\t"
            "ld1d z7.d, p0/z, [x7, #1, mul vl] \n\t"
            "add  x18, x17, x0 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x5 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x6 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x7 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x0, x7, x3 \n\t"
            "add  x5, x0, x3 \n\t"
            "add  x6, x5, x3 \n\t"
            "add  x7, x6, x3 \n\t"
            "ld1d z8.d, p0/z, [x0] \n\t"
            "ld1d z9.d, p0/z, [x0, #1, mul vl] \n\t"
            "ld1d z10.d, p0/z, [x5] \n\t"
            "ld1d z11.d, p0/z, [x5, #1, mul vl] \n\t"
            "ld1d z12.d, p0/z, [x6] \n\t"
            "ld1d z13.d, p0/z, [x6, #1, mul vl] \n\t"
            "ld1d z14.d, p0/z, [x7] \n\t"
            "ld1d z15.d, p0/z, [x7, #1, mul vl] \n\t"
            "add  x18, x17, x0 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x5 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x6 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x7 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "st1d z0.d, p0, [x1] \n\t"
            "st1d z1.d, p0, [x1, #1, mul vl] \n\t"
            "st1d z2.d, p0, [x10] \n\t"
            "st1d z3.d, p0, [x10, #1, mul vl] \n\t"
            "st1d z4.d, p0, [x11] \n\t"
            "st1d z5.d, p0, [x11, #1, mul vl] \n\t"
            "st1d z6.d, p0, [x12] \n\t"
            "st1d z7.d, p0, [x12, #1, mul vl] \n\t"
            "st1d z8.d, p0, [x13] \n\t"
            "st1d z9.d, p0, [x13, #1, mul vl] \n\t"
            "st1d z10.d, p0, [x14] \n\t"
            "st1d z11.d, p0, [x14, #1, mul vl] \n\t"
            "st1d z12.d, p0, [x15] \n\t"
            "st1d z13.d, p0, [x15, #1, mul vl] \n\t"
            "st1d z14.d, p0, [x16] \n\t"
            "st1d z15.d, p0, [x16, #1, mul vl] \n\t"
            "add  x0, x7, x3 \n\t"
            "add  x1, x16, x2 \n\t"
            "sub  x8, x8, #1 \n\t"
            BRANCH(MACOLSTORMKER)
            LABEL(MACOLSTORMKEREND)
            LABEL(MACOLSTORLEFT)
            "cmp  x9, xzr \n\t"
            BEQ(MUNITKDONE)
            "ld1d z0.d, p0/z, [x0] \n\t"
            "ld1d z1.d, p0/z, [x0, #1, mul vl] \n\t"
            "st1d z0.d, p0, [x1] \n\t"
            "st1d z1.d, p0, [x1, #1, mul vl] \n\t"
            "add  x0, x0, x3 \n\t"
            "add  x1, x1, x2 \n\t"
            "sub  x9, x9, #1 \n\t"
            BRANCH(MACOLSTORLEFT)
            // A stored in rows.
            LABEL(MAROWSTOR)
            // Prepare predicates for in-reg transpose.
            SVE512_IN_REG_TRANSPOSE_d8x8_PREPARE(x16,p0,p1,p2,p3,p8,p4,p6)
            LABEL(MAROWSTORMKER) // X[10-16] for A here not P. Be careful.
            "cmp  x8, xzr \n\t"
            BEQ(MAROWSTORMKEREND)
            "add  x10, x0, x4 \n\t"
            "add  x11, x10, x4 \n\t"
            "add  x12, x11, x4 \n\t"
            "add  x13, x12, x4 \n\t"
            "add  x14, x13, x4 \n\t"
            "add  x15, x14, x4 \n\t"
            "add  x16, x15, x4 \n\t"
            "ld1d z0.d, p0/z, [x0] \n\t"
            "ld1d z1.d, p0/z, [x10] \n\t"
            "ld1d z2.d, p0/z, [x11] \n\t"
            "ld1d z3.d, p0/z, [x12] \n\t"
            "ld1d z4.d, p0/z, [x13] \n\t"
            "ld1d z5.d, p0/z, [x14] \n\t"
            "ld1d z6.d, p0/z, [x15] \n\t"
            "ld1d z7.d, p0/z, [x16] \n\t"
            "add  x5, x16, x4 \n\t"
            "add  x10, x5, x4 \n\t"
            "add  x11, x10, x4 \n\t"
            "add  x12, x11, x4 \n\t"
            "add  x13, x12, x4 \n\t"
            "add  x14, x13, x4 \n\t"
            "add  x15, x14, x4 \n\t"
            "add  x16, x15, x4 \n\t"
            "ld1d z16.d, p0/z, [x5] \n\t"
            "ld1d z17.d, p0/z, [x10] \n\t"
            "ld1d z18.d, p0/z, [x11] \n\t"
            "ld1d z19.d, p0/z, [x12] \n\t"
            "ld1d z20.d, p0/z, [x13] \n\t"
            "ld1d z21.d, p0/z, [x14] \n\t"
            "ld1d z22.d, p0/z, [x15] \n\t"
            "ld1d z23.d, p0/z, [x16] \n\t"
            // Transpose first 8 rows.
            SVE512_IN_REG_TRANSPOSE_d8x8(z8,z9,z10,z11,z12,z13,z14,z15,z0,z1,z2,z3,z4,z5,z6,z7,p0,p1,p2,p3,p8,p4,p6)
            // Transpose last 8 rows.
            SVE512_IN_REG_TRANSPOSE_d8x8(z24,z25,z26,z27,z28,z29,z30,z31,z16,z17,z18,z19,z20,z21,z22,z23,p0,p1,p2,p3,p8,p4,p6)
            "add  x10, x1, x2 \n\t"
            "add  x11, x10, x2 \n\t"
            "add  x12, x11, x2 \n\t"
            "add  x13, x12, x2 \n\t"
            "add  x14, x13, x2 \n\t"
            "add  x15, x14, x2 \n\t"
            "add  x16, x15, x2 \n\t"
            "st1d z8.d, p0, [x1] \n\t"
            "st1d z24.d, p0, [x1, #1, mul vl] \n\t"
            "st1d z9.d, p0, [x10] \n\t"
            "st1d z25.d, p0, [x10, #1, mul vl] \n\t"
            "st1d z10.d, p0, [x11] \n\t"
            "st1d z26.d, p0, [x11, #1, mul vl] \n\t"
            "st1d z11.d, p0, [x12] \n\t"
            "st1d z27.d, p0, [x12, #1, mul vl] \n\t"
            "st1d z12.d, p0, [x13] \n\t"
            "st1d z28.d, p0, [x13, #1, mul vl] \n\t"
            "st1d z13.d, p0, [x14] \n\t"
            "st1d z29.d, p0, [x14, #1, mul vl] \n\t"
            "st1d z14.d, p0, [x15] \n\t"
            "st1d z30.d, p0, [x15, #1, mul vl] \n\t"
            "st1d z15.d, p0, [x16] \n\t"
            "st1d z31.d, p0, [x16, #1, mul vl] \n\t"
            "add  x0, x0, #64 \n\t"
            "add  x1, x16, x2 \n\t"
            "sub  x8, x8, #1 \n\t"
            BRANCH(MAROWSTORMKER)
            LABEL(MAROWSTORMKEREND)
            "mov  x4, %[inca] \n\t" // Restore unshifted inca.
            "index z30.d, xzr, x4 \n\t" // Generate index.
            "lsl  x4, x4, #3 \n\t" // Shift again.
            "lsl  x5, x4, #3 \n\t" // Virtual column vl.
            LABEL(MAROWSTORLEFT)
            "cmp  x9, xzr \n\t"
            BEQ(MUNITKDONE)
            "add  x6, x0, x5 \n\t"
            "ld1d z0.d, p0/z, [x0, z30.d, lsl #3] \n\t"
            "ld1d z1.d, p0/z, [x6, z30.d, lsl #3] \n\t"
            "st1d z0.d, p0, [x1] \n\t"
            "st1d z1.d, p0, [x1, #1, mul vl] \n\t"
            "add  x1, x1, x2 \n\t"
            "add  x0, x0, #8 \n\t"
            "sub  x9, x9, #1 \n\t"
            BRANCH(MAROWSTORLEFT)
            LABEL(MUNITKDONE)
            "mov  x0, #0 \n\t"
            :
            : [a]      "r" (a),
              [p]      "r" (p),
              [lda]    "r" (lda),
              [ldp]    "r" (ldp),
              [inca]   "r" (inca),
              [n_mker] "r" (n_mker),
              [n_left] "r" (n_left)
            : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
              "x8", "x9", "x10","x11","x12","x13","x14","x15",
              "x16","x17","x18",
              "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10","z11","z12","z13","z14","z15",
              // "z16","z17","z18","z19","z20","z21","z22","z23",
              // "z24","z25","z26","z27","z28","z29","z30","z31",
              "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"
            );
    }
    else if ( cdim == nr && cdim_bcast == 1 && !gs && unitk )
    {
        uint64_t n_mker = n / 8;
        uint64_t n_left = n % 8;
        __asm__ volatile (
            "mov  x0, %[a] \n\t"
            "mov  x1, %[p] \n\t"
            "mov  x2, %[ldp] \n\t"
            "mov  x3, %[lda] \n\t"
            "mov  x4, %[inca] \n\t"
            "cmp  x4, #1 \n\t"
            // Skips by sizeof(double).
            "mov  x8, #8 \n\t"
            "madd x2, x2, x8, xzr \n\t"
            "madd x3, x3, x8, xzr \n\t"
            "madd x4, x4, x8, xzr \n\t"
            // Loop constants.
            "mov  x8, %[n_mker] \n\t"
            "mov  x9, %[n_left] \n\t"
            "ptrue p0.d \n\t"
            BNE(NAROWSTOR)
            // A stored in columns.
            LABEL(NACOLSTOR)
            // Prefetch distance.
            "mov  x17, #8 \n\t"
            "madd x17, x17, x3, xzr \n\t"
#ifdef _A64FX
            // Disable hardware prefetch for A.
            "mov  x16, 0x6 \n\t"
            "lsl  x16, x16, #60 \n\t"
            "orr  x0, x0, x16 \n\t"
#endif
            LABEL(NACOLSTORMKER)
            "cmp  x8, xzr \n\t"
            BEQ(NACOLSTORMKEREND)
            "add  x5, x0, x3 \n\t"
            "add  x6, x5, x3 \n\t"
            "add  x7, x6, x3 \n\t"
            "ld1d z0.d, p0/z, [x0] \n\t"
            "ldr  q1, [x0, #64] \n\t"
            "ld1d z2.d, p0/z, [x5] \n\t"
            "ldr  q3, [x5, #64] \n\t"
            "ld1d z4.d, p0/z, [x6] \n\t"
            "ldr  q5, [x6, #64] \n\t"
            "ld1d z6.d, p0/z, [x7] \n\t"
            "ldr  q7, [x7, #64] \n\t"
            "add  x18, x17, x0 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x5 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x6 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x7 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x0, x7, x3 \n\t"
            "add  x5, x0, x3 \n\t"
            "add  x6, x5, x3 \n\t"
            "add  x7, x6, x3 \n\t"
            "ld1d z8.d, p0/z, [x0] \n\t"
            "ldr  q9, [x0, #64] \n\t"
            "ld1d z10.d, p0/z, [x5] \n\t"
            "ldr  q11, [x5, #64] \n\t"
            "ld1d z12.d, p0/z, [x6] \n\t"
            "ldr  q13, [x6, #64] \n\t"
            "ld1d z14.d, p0/z, [x7] \n\t"
            "ldr  q15, [x7, #64] \n\t"
            "add  x18, x17, x0 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x5 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x6 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            "add  x18, x17, x7 \n\t"
            "prfm PLDL1STRM, [x18] \n\t"
            // Plain storage
            "add  x10, x1, x2 \n\t"
            "add  x11, x10, x2 \n\t"
            "add  x12, x11, x2 \n\t"
            "add  x13, x12, x2 \n\t"
            "add  x14, x13, x2 \n\t"
            "add  x15, x14, x2 \n\t"
            "add  x16, x15, x2 \n\t"
            "st1d z0.d, p0, [x1] \n\t"
            "str  q1, [x1, #64] \n\t"
            "st1d z2.d, p0, [x10] \n\t"
            "str  q3, [x10, #64] \n\t"
            "st1d z4.d, p0, [x11] \n\t"
            "str  q5, [x11, #64] \n\t"
            "st1d z6.d, p0, [x12] \n\t"
            "str  q7, [x12, #64] \n\t"
            "st1d z8.d, p0, [x13] \n\t"
            "str  q9, [x13, #64] \n\t"
            "st1d z10.d, p0, [x14] \n\t"
            "str  q11, [x14, #64] \n\t"
            "st1d z12.d, p0, [x15] \n\t"
            "str  q13, [x15, #64] \n\t"
            "st1d z14.d, p0, [x16] \n\t"
            "str  q15, [x16, #64] \n\t"
            "add  x1, x16, x2 \n\t"
            // Realign and store.
            // "ext  z1.b, z1.b, z1.b, #16 \n\t"
            // "ext  z1.b, z1.b, z2.b, #48 \n\t"
            // "ext  z2.b, z2.b, z3.b, #16 \n\t"
            // "ext  z2.b, z2.b, z4.b, #32 \n\t"
            // "ext  z4.b, z4.b, z5.b, #16 \n\t"
            // "ext  z4.b, z4.b, z6.b, #16 \n\t"
            // "ext  z6.b, z6.b, z7.b, #16 \n\t"
            // "ext  z9.b, z9.b, z9.b, #16 \n\t"
            // "ext  z9.b, z9.b, z10.b, #48 \n\t"
            // "ext  z10.b, z10.b, z11.b, #16 \n\t"
            // "ext  z10.b, z10.b, z12.b, #32 \n\t"
            // "ext  z12.b, z12.b, z13.b, #16 \n\t"
            // "ext  z12.b, z12.b, z14.b, #16 \n\t"
            // "ext  z14.b, z14.b, z15.b, #16 \n\t"
            // "st1d z0.d, p0, [x1] \n\t"
            // "st1d z1.d, p0, [x1, #1, mul vl] \n\t"
            // "st1d z2.d, p0, [x1, #2, mul vl] \n\t"
            // "st1d z4.d, p0, [x1, #3, mul vl] \n\t"
            // "st1d z6.d, p0, [x1, #4, mul vl] \n\t"
            // "add  x1, x1, #320 \n\t"
            // "st1d z8.d, p0, [x1] \n\t"
            // "st1d z9.d, p0, [x1, #1, mul vl] \n\t"
            // "st1d z10.d, p0, [x1, #2, mul vl] \n\t"
            // "st1d z12.d, p0, [x1, #3, mul vl] \n\t"
            // "st1d z14.d, p0, [x1, #4, mul vl] \n\t"
            // "add  x1, x1, #320 \n\t"
            "add  x0, x7, x3 \n\t"
            "sub  x8, x8, #1 \n\t"
            BRANCH(NACOLSTORMKER)
            LABEL(NACOLSTORMKEREND)
            LABEL(NACOLSTORLEFT)
            "cmp  x9, xzr \n\t"
            BEQ(NUNITKDONE)
            "ld1d z0.d, p0/z, [x0] \n\t"
            "ldr  q1, [x0, #64] \n\t"
            "st1d z0.d, p0, [x1] \n\t"
            "str  q1, [x1, #64] \n\t"
            "add  x0, x0, x3 \n\t"
            "add  x1, x1, x2 \n\t"
            "sub  x9, x9, #1 \n\t"
            BRANCH(NACOLSTORLEFT)
            // A stored in rows.
            LABEL(NAROWSTOR)
            // Prepare predicates for in-reg transpose.
            SVE512_IN_REG_TRANSPOSE_d8x8_PREPARE(x16,p0,p1,p2,p3,p8,p4,p6)
            LABEL(NAROWSTORMKER) // X[10-16] for A here not P. Be careful.
            "cmp  x8, xzr \n\t"
            BEQ(NAROWSTORMKEREND)
            "add  x10, x0, x4 \n\t"
            "add  x11, x10, x4 \n\t"
            "add  x12, x11, x4 \n\t"
            "add  x13, x12, x4 \n\t"
            "add  x14, x13, x4 \n\t"
            "add  x15, x14, x4 \n\t"
            "add  x16, x15, x4 \n\t"
            "add  x17, x16, x4 \n\t"
            "add  x18, x17, x4 \n\t"
            "ld1d z0.d, p0/z, [x0] \n\t"
            "ld1d z1.d, p0/z, [x10] \n\t"
            "ld1d z2.d, p0/z, [x11] \n\t"
            "ld1d z3.d, p0/z, [x12] \n\t"
            "ld1d z4.d, p0/z, [x13] \n\t"
            "ld1d z5.d, p0/z, [x14] \n\t"
            "ld1d z6.d, p0/z, [x15] \n\t"
            "ld1d z7.d, p0/z, [x16] \n\t"
            "ld1d z22.d, p0/z, [x17] \n\t"
            "ld1d z23.d, p0/z, [x18] \n\t"
            // Transpose first 8 rows.
            SVE512_IN_REG_TRANSPOSE_d8x8(z8,z9,z10,z11,z12,z13,z14,z15,z0,z1,z2,z3,z4,z5,z6,z7,p0,p1,p2,p3,p8,p4,p6)
            // Transpose last 2 rows.
            SVE512_IN_REG_TRANSPOSE_d8x2(z16,z17,z18,z19,z20,z21,z22,z23,p0,p1,p2,p3)
            // Plain storage.
            "add  x10, x1, x2 \n\t"
            "add  x11, x10, x2 \n\t"
            "add  x12, x11, x2 \n\t"
            "add  x13, x12, x2 \n\t"
            "add  x14, x13, x2 \n\t"
            "add  x15, x14, x2 \n\t"
            "add  x16, x15, x2 \n\t"
            "st1d z8.d, p0, [x1] \n\t"
            "str  q16, [x1, #64] \n\t"
            "st1d z9.d, p0, [x10] \n\t"
            "str  q17, [x10, #64] \n\t"
            "st1d z10.d, p0, [x11] \n\t"
            "str  q18, [x11, #64] \n\t"
            "st1d z11.d, p0, [x12] \n\t"
            "str  q19, [x12, #64] \n\t"
            "st1d z12.d, p0, [x13] \n\t"
            "str  q20, [x13, #64] \n\t"
            "st1d z13.d, p0, [x14] \n\t"
            "str  q21, [x14, #64] \n\t"
            "st1d z14.d, p0, [x15] \n\t"
            "str  q22, [x15, #64] \n\t"
            "st1d z15.d, p0, [x16] \n\t"
            "str  q23, [x16, #64] \n\t"
            "add  x1, x16, x2 \n\t"
            "add  x0, x0, #64 \n\t"
            "sub  x8, x8, #1 \n\t"
            BRANCH(NAROWSTORMKER)
            LABEL(NAROWSTORMKEREND)
            "mov  x4, %[inca] \n\t" // Restore unshifted inca.
            "index z30.d, xzr, x4 \n\t" // Generate index.
            "lsl  x4, x4, #3 \n\t" // Shift again.
            "lsl  x5, x4, #3 \n\t" // Virtual column vl.
            LABEL(NAROWSTORLEFT)
            "cmp  x9, xzr \n\t"
            BEQ(NUNITKDONE)
            "add  x6, x0, x5 \n\t"
            "add  x7, x6, x4 \n\t"
            "ld1d z0.d, p0/z, [x0, z30.d, lsl #3] \n\t"
            "ldr  d1, [x6] \n\t"
            "ldr  d2, [x7] \n\t"
            "trn1 v1.2d, v1.2d, v2.2d \n\t"
            "st1d z0.d, p0, [x1] \n\t"
            "str  q1, [x1, #64] \n\t"
            "add  x1, x1, x2 \n\t"
            "add  x0, x0, #8 \n\t"
            "sub  x9, x9, #1 \n\t"
            BRANCH(NAROWSTORLEFT)
            LABEL(NUNITKDONE)
            "mov  x0, #0 \n\t"
            :
            : [a]      "r" (a),
              [p]      "r" (p),
              [lda]    "r" (lda),
              [ldp]    "r" (ldp),
              [inca]   "r" (inca),
              [n_mker] "r" (n_mker),
              [n_left] "r" (n_left)
            : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
              "x8", "x9", "x10","x11","x12","x13","x14","x15",
              "x16","x17","x18",
              "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10","z11","z12","z13","z14","z15",
              "z16","z17","z18","z19","z20","z21","z22","z23",
              // "z24","z25","z26","z27","z28","z29",
              "z30","z31",
              "p0", "p1", "p2", "p3", "p4", // "p5",
              "p6", "p7", "p8"
            );
    }
	else
	{
		bli_dscal2bbs_mxn
		(
		  conja,
		  cdim_,
		  n_,
		  kappa,
		  a,       inca, lda,
		  p, cdim_bcast, ldp
		);
	}

	bli_dset0s_edge
	(
	  cdim_*cdim_bcast, cdim_max*cdim_bcast,
	  n_, n_max_,
	  p, ldp
	);
}
