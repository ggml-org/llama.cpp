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
#include "assert.h"


// Label locality & misc.
#include "../armv8a_asm_utils.h"

#define DGEMM_3X1X2_NKER_SUBLOOP(C0,C1,C2,A0,A1,A2,B) \
" fmla v"#C0".2d, v"#A0".2d, v"#B".2d      \n\t" \
" fmla v"#C1".2d, v"#A1".2d, v"#B".2d      \n\t" \
" fmla v"#C2".2d, v"#A2".2d, v"#B".2d      \n\t"

#define DGEMM_3X8X2_K_MKER_LOOP_PLAIN(C00,C01,C02,C03,C04,C05,C06,C07,C10,C11,C12,C13,C14,C15,C16,C17,C20,C21,C22,C23,C24,C25,C26,C27,A0,A1,A2,B0,B1,B2,B3,BADDR,BELEMADDR,BELEMST,LOADNEXT) \
  /* Always load before forwarding to the next line. */ \
  DGEMM_3X1X2_NKER_SUBLOOP(C00,C10,C20,A0,A1,A2,B0) \
  DGEMM_LOAD1V_K_load(B0,BELEMADDR,BELEMST) \
  DGEMM_3X1X2_NKER_SUBLOOP(C01,C11,C21,A0,A1,A2,B1) \
  DGEMM_LOAD1V_K_load(B1,BELEMADDR,BELEMST) \
  DGEMM_3X1X2_NKER_SUBLOOP(C02,C12,C22,A0,A1,A2,B2) \
  DGEMM_LOAD1V_K_load(B2,BELEMADDR,BELEMST) \
  DGEMM_3X1X2_NKER_SUBLOOP(C03,C13,C23,A0,A1,A2,B3) \
  DGEMM_LOAD1V_K_load(B3,BELEMADDR,BELEMST) \
  \
" add  "#BADDR", "#BADDR", #16             \n\t" \
" mov  "#BELEMADDR", "#BADDR"              \n\t" \
  DGEMM_3X1X2_NKER_SUBLOOP(C04,C14,C24,A0,A1,A2,B0) \
  DGEMM_LOAD1V_K_ ##LOADNEXT (B0,BELEMADDR,BELEMST) \
  DGEMM_3X1X2_NKER_SUBLOOP(C05,C15,C25,A0,A1,A2,B1) \
  DGEMM_LOAD1V_K_ ##LOADNEXT (B1,BELEMADDR,BELEMST) \
  DGEMM_3X1X2_NKER_SUBLOOP(C06,C16,C26,A0,A1,A2,B2) \
  DGEMM_LOAD1V_K_ ##LOADNEXT (B2,BELEMADDR,BELEMST) \
  DGEMM_3X1X2_NKER_SUBLOOP(C07,C17,C27,A0,A1,A2,B3) \
  DGEMM_LOAD1V_K_ ##LOADNEXT (B3,BELEMADDR,BELEMST)

#define DGEMM_LOAD1V_K_noload(V,ELEMADDR,ELEMST)
#define DGEMM_LOAD1V_K_load(V,ELEMADDR,ELEMST) \
" ldr  q"#V", [ "#ELEMADDR" ]              \n\t" \
" add  "#ELEMADDR", "#ELEMADDR", "#ELEMST" \n\t"

// For row-storage of C.
#define DLOADC_4V_R_FWD(C0,C1,C2,C3,CADDR,CSHIFT,RSC) \
  DLOAD4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_4V_R_FWD(C0,C1,C2,C3,CADDR,CSHIFT,RSC) \
  DSTORE4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

// For column-storage of C.
#define DLOADC_1V_1ELM_C_FWD(C0,CSCALAR,CIDX,CADDR,CSHIFT,CSC) \
  DLOAD1V(C0,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", #"#CSHIFT"+16    \n\t" \
" ld1  {v"#CSCALAR".d}["#CIDX"], ["#CADDR"] \n\t" \
" sub  "#CADDR", "#CADDR", #"#CSHIFT"+16    \n\t" \
" add  "#CADDR", "#CADDR", "#CSC"           \n\t"
#define DSTOREC_1V_1ELM_C_FWD(C0,CSCALAR,CIDX,CADDR,CSHIFT,CSC) \
  DSTORE1V(C0,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", #"#CSHIFT"+16    \n\t" \
" st1  {v"#CSCALAR".d}["#CIDX"], ["#CADDR"] \n\t" \
" sub  "#CADDR", "#CADDR", #"#CSHIFT"+16    \n\t" \
" add  "#CADDR", "#CADDR", "#CSC"           \n\t"

#define DSCALE12V(V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,A,IDX) \
  DSCALE4V(V0,V1,V2,V3,A,IDX) \
  DSCALE4V(V4,V5,V6,V7,A,IDX) \
  DSCALE4V(V8,V9,V10,V11,A,IDX)
#define DSCALEA12V(D0,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,A,IDX) \
  DSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  DSCALEA4V(D4,D5,D6,D7,S4,S5,S6,S7,A,IDX) \
  DSCALEA4V(D8,D9,D10,D11,S8,S9,S10,S11,A,IDX)

#define DPRFMC_FWD(CADDR,DLONGC) \
" prfm PLDL1KEEP, ["#CADDR"]         \n\t" \
" add  "#CADDR", "#CADDR", "#DLONGC" \n\t"


BLIS_INLINE
void bli_dgemmsup_rd_armv8a_inline_3x4m
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
  assert( n0 == 4 );

  for ( ; m0 >= 3; m0 -= 3 )
  {
    bli_dgemmsup_rd_armv8a_asm_3x4
    (
      conja, conjb, 3, 4, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, data, cntx
    );
    a = ( double* )a + 3 * rs_a0;
    c = ( double* )c + 3 * rs_c0;
  }

  if ( m0 > 0 )
  {
    bli_dgemmsup_rd_armv8a_int_3x4
    (
      conja, conjb, m0, 4, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, data, cntx
    );
  }
}

BLIS_INLINE
void bli_dgemmsup_rd_armv8a_inline_3xcm
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
  for ( ; m0 > 0; m0 -= 3 )
  {
    dim_t m_loc = ( m0 < 3 ) ? m0 : 3;

    bli_dgemmsup_rd_armv8a_int_3x4
    (
      conja, conjb, m_loc, n0, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, data, cntx
    );

    a = ( double* )a + 3 * rs_a0;
    c = ( double* )c + 3 * rs_c0;
  }
}


void bli_dgemmsup_rd_armv8a_asm_6x8m
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
  if ( n0 != 8 )
  {
    assert( n0 <= 13 );

    // Manual separation.
    gemmsup_ker_ft ker_fp1 = NULL;
    gemmsup_ker_ft ker_fp2 = NULL;
    gemmsup_ker_ft ker_fp3 = NULL;
    dim_t          nr1, nr2, nr3;

    switch ( n0 )
    {
      case 13:
        ker_fp1 = bli_dgemmsup_rd_armv8a_asm_6x8m; nr1 = 8; // This function.
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 3;
        ker_fp3 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr3 = 2; break;
      case 12:
        ker_fp1 = bli_dgemmsup_rd_armv8a_asm_6x8m; nr1 = 8; // This function.
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3x4m; nr2 = 4; break;
      case 11:
        ker_fp1 = bli_dgemmsup_rd_armv8a_asm_6x8m; nr1 = 8; // This function.
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 3; break;
      case 10:
        ker_fp1 = bli_dgemmsup_rd_armv8a_asm_6x8m; nr1 = 8; // This function.
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 2; break;
      case 9:
        ker_fp1 = bli_dgemmsup_rd_armv8a_asm_6x8m; nr1 = 8; // This function.
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 1; break;
      case 7:
        ker_fp1 = bli_dgemmsup_rd_armv8a_inline_3x4m; nr1 = 4;
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 3; break;
      case 6:
        ker_fp1 = bli_dgemmsup_rd_armv8a_inline_3x4m; nr1 = 4;
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 2; break;
      case 5:
        ker_fp1 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr1 = 3;
        ker_fp2 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr2 = 2; break;
      case 4:
        ker_fp1 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr1 = 4; break;
      default:
        ker_fp1 = bli_dgemmsup_rd_armv8a_inline_3xcm; nr1 = n0; break;
    }

    ker_fp1
    (
      conja, conjb, m0, nr1, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, data, cntx
    );
    b = ( double* )b + nr1 * cs_b0;
    c = ( double* )c + nr1 * cs_c0;
    if ( ker_fp2 )
    {
      ker_fp2
      (
        conja, conjb, m0, nr2, k0,
        alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
        beta, c, rs_c0, cs_c0, data, cntx
      );
      b = ( double* )b + nr2 * cs_b0;
      c = ( double* )c + nr2 * cs_c0;
    }
    if ( ker_fp3 )
      ker_fp3
      (
        conja, conjb, m0, nr3, k0,
        alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
        beta, c, rs_c0, cs_c0, data, cntx
      );

    return;
  }

  // LLVM has very bad routing ability for inline asm.
  // Limit number of registers in case of Clang compilation.
#ifndef __clang__
  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );
#endif

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;

  int64_t  m_iter = m0 / 3;
  int64_t  m_left = m0 % 3;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  assert( cs_a0 == 1 );
  assert( rs_b0 == 1 );

  if ( m_iter == 0 ) goto consider_edge_cases;

  __asm__ volatile
  (
" ldr             x10, %[a]                       \n\t"
" ldr             x13, %[c]                       \n\t"
" ldr             x12, %[m_iter]                  \n\t"
" ldr             x2, %[rs_a]                     \n\t" // Row-skip of A.
" ldr             x3, %[cs_b]                     \n\t" // Column-skip of B.
"                                                 \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(double).
" lsl             x2, x2, #3                      \n\t" // rs_a
" lsl             x3, x3, #3                      \n\t" // cs_b
" lsl             x6, x6, #3                      \n\t" // rs_c
" lsl             x7, x7, #3                      \n\t" // cs_c
"                                                 \n\t"
" mov             x1, x5                          \n\t"
" cmp             x7, #8                          \n\t" // Prefetch column-strided C.
BEQ(C_PREFETCH_COLS)
DPRFMC_FWD(x1,x6)
DPRFMC_FWD(x1,x6)
DPRFMC_FWD(x1,x6)
DPRFMC_FWD(x1,x6)
DPRFMC_FWD(x1,x6)
DPRFMC_FWD(x1,x6)
BRANCH(C_PREFETCH_END)
LABEL(C_PREFETCH_COLS)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
DPRFMC_FWD(x1,x7)
LABEL(C_PREFETCH_END)
//
// Millikernel.
LABEL(MILLIKER_MLOOP)
"                                                 \n\t"
" mov             x0, x10                         \n\t" // Parameters to be reloaded
" mov             x5, x13                         \n\t" //  within each millikernel loop.
" ldr             x1, %[b]                        \n\t"
" ldr             x4, %[k_mker]                   \n\t"
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:26] <- A
//  V[28:31] <- B
//  V[ 27  ] <- Not used.
// Under this scheme, the following is defined:
#define DGEMM_3X8X2_K_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3,BADDR,BELEMADDR,BELEMST,LOADNEXT) \
  DGEMM_3X8X2_K_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,B0,B1,B2,B3,BADDR,BELEMADDR,BELEMST,LOADNEXT)
// Load from memory.
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" mov             x11, x1                         \n\t" // Load B.
" ldr             q28, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
" ldr             q29, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
" ldr             q30, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
" ldr             q31, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
"                                                 \n\t"
" mov             x14, x0                         \n\t" // Load A.
" ldr             q24, [x14]                      \n\t"
" add             x14, x14, x2                    \n\t"
" ldr             q25, [x14]                      \n\t"
" add             x14, x14, x2                    \n\t"
" ldr             q26, [x14]                      \n\t"
// " add             x14, x14, x2                    \n\t"
" add             x0, x0, #16                     \n\t"
LABEL(CLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
// No-microkernel early return, once again.
BEQ(K_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_3X8X2_K_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,A2,B0,B1,B2,B3) \
  DGEMM_3X8X2_K_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3,x1,x11,x3,load) \
 "mov             x14, x0                         \n\t" \
 "ldr             q24, [x14]                      \n\t" \
 "add             x14, x14, x2                    \n\t" \
 "ldr             q25, [x14]                      \n\t" \
 "add             x14, x14, x2                    \n\t" \
 "ldr             q26, [x14]                      \n\t" \
 /*"add             x14, x14, x2                    \n\t"*/ \
 "add             x0, x0, #16                     \n\t"
// Start microkernel loop.
LABEL(K_MKER_LOOP)
DGEMM_3X8X2_K_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,28,29,30,31)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(FIN_MKER_LOOP)
DGEMM_3X8X2_K_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,28,29,30,31)
BRANCH(K_MKER_LOOP)
//
// Final microkernel loop.
LABEL(FIN_MKER_LOOP)
DGEMM_3X8X2_K_MKER_LOOP_PLAIN_LOC(24,25,26,28,29,30,31,x1,x11,x3,noload)
//
// If major kernel is executed,
//  an additional depth-summation is required.
" faddp           v0.2d, v0.2d, v1.2d             \n\t" // Line 0.
" faddp           v1.2d, v2.2d, v3.2d             \n\t"
" faddp           v2.2d, v4.2d, v5.2d             \n\t"
" faddp           v3.2d, v6.2d, v7.2d             \n\t"
" faddp           v4.2d, v8.2d, v9.2d             \n\t" // Line 1.
" faddp           v5.2d, v10.2d, v11.2d           \n\t"
" faddp           v6.2d, v12.2d, v13.2d           \n\t"
" faddp           v7.2d, v14.2d, v15.2d           \n\t"
" faddp           v8.2d, v16.2d, v17.2d           \n\t" // Line 2.
" faddp           v9.2d, v18.2d, v19.2d           \n\t"
" faddp           v10.2d, v20.2d, v21.2d          \n\t"
" faddp           v11.2d, v22.2d, v23.2d          \n\t"
"                                                 \n\t"
// Loops left behind microkernels.
LABEL(K_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(WRITE_MEM_PREP)
" mov             x11, x1                         \n\t" // Load B row.
" ld1             {v28.d}[0], [x11], x3           \n\t"
" ld1             {v28.d}[1], [x11], x3           \n\t"
" ld1             {v29.d}[0], [x11], x3           \n\t"
" ld1             {v29.d}[1], [x11], x3           \n\t"
" ld1             {v30.d}[0], [x11], x3           \n\t"
" ld1             {v30.d}[1], [x11], x3           \n\t"
" ld1             {v31.d}[0], [x11], x3           \n\t"
" ld1             {v31.d}[1], [x11], x3           \n\t"
" add             x1, x1, #8                      \n\t"
" mov             x14, x0                         \n\t" // Load A column.
" ld1             {v24.d}[0], [x14], x2           \n\t"
" ld1             {v24.d}[1], [x14], x2           \n\t"
" ld1             {v25.d}[0], [x14], x2           \n\t"
" add             x0, x0, #8                      \n\t"
" fmla            v0.2d, v28.2d, v24.d[0]         \n\t"
" fmla            v1.2d, v29.2d, v24.d[0]         \n\t"
" fmla            v2.2d, v30.2d, v24.d[0]         \n\t"
" fmla            v3.2d, v31.2d, v24.d[0]         \n\t"
" fmla            v4.2d, v28.2d, v24.d[1]         \n\t"
" fmla            v5.2d, v29.2d, v24.d[1]         \n\t"
" fmla            v6.2d, v30.2d, v24.d[1]         \n\t"
" fmla            v7.2d, v31.2d, v24.d[1]         \n\t"
" fmla            v8.2d, v28.2d, v25.d[0]         \n\t"
" fmla            v9.2d, v29.2d, v25.d[0]         \n\t"
" fmla            v10.2d, v30.2d, v25.d[0]         \n\t"
" fmla            v11.2d, v31.2d, v25.d[0]         \n\t"
" sub             x8, x8, #1                      \n\t"
BRANCH(K_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(WRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v30.2d}, [x4]                  \n\t" // Load alpha & beta (value).
" ld1r            {v31.2d}, [x8]                  \n\t"
"                                                 \n\t"
" fmov            d28, #1.0                       \n\t" // Don't scale for unit alpha.
" fcmp            d30, d28                        \n\t"
BEQ(UNIT_ALPHA)
DSCALE12V(0,1,2,3,4,5,6,7,8,9,10,11,30,0)
LABEL(UNIT_ALPHA)
"                                                 \n\t"
" mov             x1, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x7, #8                          \n\t" // Check for column-storage.
BNE(WRITE_MEM_C)
//
// C storage in rows.
LABEL(WRITE_MEM_R)
" fcmp            d31, #0.0                       \n\t" // Don't load for zero beta.
BEQ(ZERO_BETA_R)
DLOADC_4V_R_FWD(12,13,14,15,x1,0,x6)
DLOADC_4V_R_FWD(16,17,18,19,x1,0,x6)
DLOADC_4V_R_FWD(20,21,22,23,x1,0,x6)
DSCALEA12V(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,31,0)
LABEL(ZERO_BETA_R)
#ifndef __clang__
" cmp   x12, #1                       \n\t"
BRANCH(PRFM_END_R)
" prfm  PLDL1KEEP, [%[a_next], #16*0] \n\t"
" prfm  PLDL1KEEP, [%[a_next], #16*1] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*0] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*1] \n\t"
LABEL(PRFM_END_R)
#endif
DSTOREC_4V_R_FWD(0,1,2,3,x5,0,x6)
DSTOREC_4V_R_FWD(4,5,6,7,x5,0,x6)
DSTOREC_4V_R_FWD(8,9,10,11,x5,0,x6)
BRANCH(END_WRITE_MEM)
//
// C storage in columns.
LABEL(WRITE_MEM_C)
" trn1            v12.2d, v0.2d, v4.2d            \n\t"
" trn2            v13.2d, v0.2d, v4.2d            \n\t"
" trn1            v14.2d, v1.2d, v5.2d            \n\t"
" trn2            v15.2d, v1.2d, v5.2d            \n\t"
" trn1            v16.2d, v2.2d, v6.2d            \n\t"
" trn2            v17.2d, v2.2d, v6.2d            \n\t"
" trn1            v18.2d, v3.2d, v7.2d            \n\t"
" trn2            v19.2d, v3.2d, v7.2d            \n\t"
" fcmp            d31, #0.0                       \n\t" // Don't load for zero beta.
BEQ(ZERO_BETA_C)
DLOADC_1V_1ELM_C_FWD(0,20,0,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(1,20,1,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(2,21,0,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(3,21,1,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(4,22,0,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(5,22,1,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(6,23,0,x1,0,x7)
DLOADC_1V_1ELM_C_FWD(7,23,1,x1,0,x7)
DSCALEA12V(12,13,14,15,16,17,18,19,8,9,10,11,0,1,2,3,4,5,6,7,20,21,22,23,31,0)
LABEL(ZERO_BETA_C)
#ifndef __clang__
" cmp   x12, #1                       \n\t"
BRANCH(PRFM_END_C)
" prfm  PLDL1KEEP, [%[a_next], #16*0] \n\t"
" prfm  PLDL1KEEP, [%[a_next], #16*1] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*0] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*1] \n\t"
LABEL(PRFM_END_C)
#endif
DSTOREC_1V_1ELM_C_FWD(12,8,0,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(13,8,1,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(14,9,0,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(15,9,1,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(16,10,0,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(17,10,1,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(18,11,0,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(19,11,1,x5,0,x7)
//
// End of this microkernel.
LABEL(END_WRITE_MEM)
"                                                 \n\t"
" subs            x12, x12, #1                    \n\t"
BEQ(END_EXEC)
"                                                 \n\t"
" mov             x8, #3                          \n\t"
" madd            x13, x6, x8, x13                \n\t" // Forward C's base address to the next logic panel.
" madd            x10, x2, x8, x10                \n\t" // Forward A's base address to the next logic panel.
BRANCH(MILLIKER_MLOOP)
//
// End of execution.
LABEL(END_EXEC)
:
: [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_a]   "m" (rs_a),
  [cs_b]   "m" (cs_b),
  [rs_c]   "m" (rs_c),
  [cs_c]   "m" (cs_c),
  // In Clang, even "m"-passed parameter takes 1 register.
  // Have to disable prefetching to pass compilation.
#ifndef __clang__
  [a_next] "r" (a_next),
  [b_next] "r" (b_next),
#endif
  [m_iter] "m" (m_iter),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta)
: "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
  "x8", "x9", "x10","x11","x12","x13","x14",
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
  "v8", "v9", "v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19","v20","v21","v22","v23",
  "v24","v25","v26","v27","v28","v29","v30","v31"
  );

consider_edge_cases:
  // TODO: Implement optimized kernel for this.
  //
  // Forward address.
  a = ( double* )a + m_iter * 3 * rs_a;
  c = ( double* )c + m_iter * 3 * rs_c;
  for ( ; m_left > 0; m_left -= 2 )
  {
    dim_t m_loc = ( m_left < 2 ) ? m_left : 2;

    bli_dgemmsup_rd_armv8a_int_2x8
    (
      conja, conjb, m_loc, 8, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, data, cntx
    );
    a = ( double* )a + 2 * rs_a0;
    c = ( double* )c + 2 * rs_c0;
  }
}

