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

// Nanokernel operations.
#include "../armv8a_asm_d2x2.h"

/* Order of row-major DGEMM_6x8's execution in 2x2 blocks:
 *
 * +---+ +---+ +---+ +---+
 * | 0 | | 1 | | 6 | | 7 |
 * +---+ +---+ +---+ +---+
 * +---+ +---+ +---+ +---+
 * | 2 | | 3 | | 8 | | 9 |
 * +---+ +---+ +---+ +---+
 * +---+ +---+ +---+ +---+
 * | 4 | | 5 | | 10| | 11|
 * +---+ +---+ +---+ +---+
 *
 */
#define DGEMM_6X8_MKER_LOOP_PLAIN(C00,C01,C02,C03,C10,C11,C12,C13,C20,C21,C22,C23,C30,C31,C32,C33,C40,C41,C42,C43,C50,C51,C52,C53,A0,A1,A2,B0,B1,B2,B3,AELEMADDR,AELEMST,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_2X2_NANOKERNEL(C00,C10,B0,A0) \
  DGEMM_2X2_NANOKERNEL(C01,C11,B1,A0) \
  DGEMM_2X2_NANOKERNEL(C20,C30,B0,A1) \
  DGEMM_2X2_NANOKERNEL(C21,C31,B1,A1) \
  DGEMM_2X2_NANOKERNEL(C40,C50,B0,A2) \
  DGEMM_2X2_NANOKERNEL(C41,C51,B1,A2) \
  DGEMM_LOAD2V_ ##LOADNEXT (B0,B1,BADDR,BSHIFT) \
  DGEMM_2X2_NANOKERNEL(C02,C12,B2,A0) \
  DGEMM_2X2_NANOKERNEL(C03,C13,B3,A0) \
  DGEMM_LOAD1V_G_ ##LOADNEXT (A0,AELEMADDR,AELEMST) \
  DGEMM_2X2_NANOKERNEL(C22,C32,B2,A1) \
  DGEMM_2X2_NANOKERNEL(C23,C33,B3,A1) \
  DGEMM_LOAD1V_G_ ##LOADNEXT (A1,AELEMADDR,AELEMST) \
  DGEMM_2X2_NANOKERNEL(C42,C52,B2,A2) \
  DGEMM_2X2_NANOKERNEL(C43,C53,B3,A2)

// Interleaving load or not.
#define DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define DGEMM_LOAD1V_load(V1,ADDR,IMM) \
" ldr  q"#V1", ["#ADDR", #"#IMM"] \n\t"

#define DGEMM_LOAD2V_noload(V1,V2,ADDR,IMM)
#define DGEMM_LOAD2V_load(V1,V2,ADDR,IMM) \
  DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  DGEMM_LOAD1V_load(V2,ADDR,IMM+16)

#define DGEMM_LOAD1V_G_noload(V1,ADDR,ST)
#define DGEMM_LOAD1V_G_load(V1,ADDR,ST) \
" ld1  {v"#V1".d}[0], ["#ADDR"], "#ST" \n\t" \
" ld1  {v"#V1".d}[1], ["#ADDR"], "#ST" \n\t"

// Prefetch C in the long direction.
#define DPRFMC_FWD(CADDR,DLONGC) \
" prfm PLDL1KEEP, ["#CADDR"]      \n\t" \
" add  "#CADDR", "#CADDR", "#DLONGC" \n\t"

// For row-storage of C.
#define DLOADC_4V_R_FWD(C0,C1,C2,C3,CADDR,CSHIFT,RSC) \
  DLOAD4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_4V_R_FWD(C0,C1,C2,C3,CADDR,CSHIFT,RSC) \
  DSTORE4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

// For column-storage of C.
#define DLOADC_3V_C_FWD(C0,C1,C2,CADDR,CSHIFT,CSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
  DLOAD1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t"
#define DSTOREC_3V_C_FWD(C0,C1,C2,CADDR,CSHIFT,CSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
  DSTORE1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t"

#define DSCALE6V(V0,V1,V2,V3,V4,V5,A,IDX) \
  DSCALE4V(V0,V1,V2,V3,A,IDX) \
  DSCALE2V(V4,V5,A,IDX)
#define DSCALEA6V(D0,D1,D2,D3,D4,D5,S0,S1,S2,S3,S4,S5,A,IDX) \
  DSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  DSCALEA2V(D4,D5,S4,S5,A,IDX)


/*
 * 6x8 dgemmsup kernel with extending 1st dimension.
 *
 * Recommanded usage case: (L1 cache latency) * (Num. FPU) < 17 cycles.
 *
 * Calls 4x8 for edge cases.
 */
void bli_dgemmsup_rv_armv8a_asm_6x8m
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
    dim_t          nr1, nr2;

    if ( n0 == 13 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x7m; nr1 = 7;
      ker_fp2 = bli_dgemmsup_rv_armv8a_asm_6x6m; nr2 = 6;
    }
    if ( n0 == 12 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x6m; nr1 = 6;
      ker_fp2 = bli_dgemmsup_rv_armv8a_asm_6x6m; nr2 = 6;
    }
    if ( n0 == 11 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x6m; nr1 = 6;
      ker_fp2 = bli_dgemmsup_rv_armv8a_asm_6x5m; nr2 = 5;
    }
    if ( n0 == 10 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x5m; nr1 = 5;
      ker_fp2 = bli_dgemmsup_rv_armv8a_asm_6x5m; nr2 = 5;
    }
    if ( n0 == 9 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x5m; nr1 = 5;
      ker_fp2 = bli_dgemmsup_rv_armv8a_int_6x4mn; nr2 = 4;
    }
    if ( n0 == 7 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x7m; nr1 = 7;
    }
    if ( n0 == 6 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x6m; nr1 = 6;
    }
    if ( n0 == 5 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_asm_6x5m; nr1 = 5;
    }
    if ( n0 <= 4 )
    {
      ker_fp1 = bli_dgemmsup_rv_armv8a_int_6x4mn; nr1 = n0;
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
      ker_fp2
      (
        conja, conjb, m0, nr2, k0,
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
  uint64_t ps_a   = bli_auxinfo_ps_a( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;

  int64_t  m_iter = m0 / 6;
  int64_t  m_left = m0 % 6;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  // uint64_t cs_b   = cs_b0;
  assert( cs_b0 == 1 );

  if ( m_iter == 0 ) goto consider_edge_cases;

  __asm__ volatile
  (
" ldr             x10, %[a]                       \n\t"
" ldr             x13, %[c]                       \n\t"
" ldr             x12, %[m_iter]                  \n\t"
" ldr             x11, %[ps_a]                    \n\t" // Panel-skip of A.
" ldr             x9, %[rs_a]                     \n\t" // Row-skip of A.
" ldr             x2, %[cs_a]                     \n\t" // Column-skip of A.
" ldr             x3, %[rs_b]                     \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(double).
" lsl             x11, x11, #3                    \n\t" // ps_a
" lsl             x9, x9, #3                      \n\t" // rs_a
" lsl             x2, x2, #3                      \n\t" // cs_a
" lsl             x3, x3, #3                      \n\t" // rs_b
" lsl             x6, x6, #3                      \n\t" // rs_c
" lsl             x7, x7, #3                      \n\t" // cs_c
"                                                 \n\t"
" mov             x1, x5                          \n\t"
" cmp             x7, #8                          \n\t" // Prefetch column-strided C.
BEQ(C_PREFETCH_COLS)
// This prefetch will not cover further mker perts. Skip.
//
// DPRFMC_FWD(x1,x6)
// DPRFMC_FWD(x1,x6)
// DPRFMC_FWD(x1,x6)
// DPRFMC_FWD(x1,x6)
// DPRFMC_FWD(x1,x6)
// DPRFMC_FWD(x1,x6)
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
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define DGEMM_6X8_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3,AELEMADDR,AELEMST,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_6X8_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,B0,B1,B2,B3,AELEMADDR,AELEMST,BADDR,BSHIFT,LOADNEXT)
// Load from memory.
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" mov             x14, x0                         \n\t" // Load A.
" ld1             {v24.d}[0], [x14], x9           \n\t"
" ld1             {v24.d}[1], [x14], x9           \n\t"
" ld1             {v25.d}[0], [x14], x9           \n\t"
" ld1             {v25.d}[1], [x14], x9           \n\t"
" ld1             {v26.d}[0], [x14], x9           \n\t"
" ld1             {v26.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v27.d}[0], [x14], x9           \n\t"
" ld1             {v27.d}[1], [x14], x9           \n\t"
"                                                 \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B.
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" ldr             q31, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
LABEL(CLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
// No-microkernel early return, once again.
BEQ(K_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_6X8_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,A2,B0,B1,B2,B3) \
  DGEMM_6X8_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3,x14,x9,x1,0,load) \
 "add             x0, x0, x2                      \n\t" \
 "mov             x14, x0                         \n\t" \
 "ld1             {v"#A2".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A2".d}[1], [x14], x9        \n\t" \
 "ldr             q"#B2", [x1, #16*2]             \n\t" \
 "ldr             q"#B3", [x1, #16*3]             \n\t" \
 "add             x1, x1, x3                      \n\t"
// Start microkernel loop.
LABEL(K_MKER_LOOP)
DGEMM_6X8_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,28,29,30,31)
DGEMM_6X8_MKER_LOOP_PLAIN_LOC_FWD(27,24,25,28,29,30,31)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(FIN_MKER_LOOP)
DGEMM_6X8_MKER_LOOP_PLAIN_LOC_FWD(26,27,24,28,29,30,31)
DGEMM_6X8_MKER_LOOP_PLAIN_LOC_FWD(25,26,27,28,29,30,31)
BRANCH(K_MKER_LOOP)
//
// Final microkernel loop.
LABEL(FIN_MKER_LOOP)
DGEMM_6X8_MKER_LOOP_PLAIN_LOC(26,27,24,28,29,30,31,x14,x9,x1,0,load)
" add             x0, x0, x2                      \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" ldr             q31, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
DGEMM_6X8_MKER_LOOP_PLAIN_LOC(25,26,27,28,29,30,31,xzr,-1,xzr,-1,noload)
//
// Loops left behind microkernels.
LABEL(K_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(WRITE_MEM_PREP)
" mov             x14, x0                         \n\t"
" ld1             {v24.d}[0], [x14], x9           \n\t" // Load A col.
" ld1             {v24.d}[1], [x14], x9           \n\t"
" ld1             {v25.d}[0], [x14], x9           \n\t"
" ld1             {v25.d}[1], [x14], x9           \n\t"
" ld1             {v26.d}[0], [x14], x9           \n\t"
" ld1             {v26.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B row.
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" ldr             q31, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
DGEMM_6X8_MKER_LOOP_PLAIN_LOC(24,25,26,28,29,30,31,xzr,-1,xzr,-1,noload)
BRANCH(K_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(WRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
"                                                 \n\t"
" mov             x1, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x7, #8                          \n\t" // Check for column-storage.
BNE(WRITE_MEM_C)
//
// C storage in rows.
LABEL(WRITE_MEM_R)
" ld1r            {v24.2d}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v25.2d}, [x8]                  \n\t"
" fmov            d26, #1.0                       \n\t"
" fcmp            d24, d26                        \n\t"
BEQ(UNIT_ALPHA_R)
DSCALE8V(0,1,2,3,4,5,6,7,24,0)
DSCALE8V(8,9,10,11,12,13,14,15,24,0)
DSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(UNIT_ALPHA_R)
" fcmp            d25, #0.0                       \n\t"
BEQ(ZERO_BETA_R_1)
DLOADC_4V_R_FWD(26,27,28,29,x1,0,x6)
DSCALEA4V(0,1,2,3,26,27,28,29,25,0)
DLOADC_4V_R_FWD(26,27,28,29,x1,0,x6)
DSCALEA4V(4,5,6,7,26,27,28,29,25,0)
LABEL(ZERO_BETA_R_1)
DSTOREC_4V_R_FWD(0,1,2,3,x5,0,x6)
BEQ(ZERO_BETA_R_2)
DLOADC_4V_R_FWD(26,27,28,29,x1,0,x6)
DLOADC_4V_R_FWD(0,1,2,3,x1,0,x6)
DSCALEA8V(8,9,10,11,12,13,14,15,26,27,28,29,0,1,2,3,25,0)
DLOADC_4V_R_FWD(26,27,28,29,x1,0,x6)
DLOADC_4V_R_FWD(0,1,2,3,x1,0,x6)
DSCALEA8V(16,17,18,19,20,21,22,23,26,27,28,29,0,1,2,3,25,0)
LABEL(ZERO_BETA_R_2)
#ifndef __clang__
" cmp   x12, #1                       \n\t"
BRANCH(PRFM_END_R)
" prfm  PLDL1KEEP, [%[a_next], #16*0] \n\t"
" prfm  PLDL1KEEP, [%[a_next], #16*1] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*0] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*1] \n\t"
LABEL(PRFM_END_R)
#endif
DSTOREC_4V_R_FWD(4,5,6,7,x5,0,x6)
DSTOREC_4V_R_FWD(8,9,10,11,x5,0,x6)
DSTOREC_4V_R_FWD(12,13,14,15,x5,0,x6)
DSTOREC_4V_R_FWD(16,17,18,19,x5,0,x6)
DSTOREC_4V_R_FWD(20,21,22,23,x5,0,x6)
BRANCH(END_WRITE_MEM)
//
// C storage in columns.
LABEL(WRITE_MEM_C)
// In-register transpose,
//  do transposition in row-order.
" trn1            v24.2d, v0.2d, v4.2d            \n\t" // Row 0-1.
" trn2            v25.2d, v0.2d, v4.2d            \n\t"
" trn1            v26.2d, v1.2d, v5.2d            \n\t"
" trn2            v27.2d, v1.2d, v5.2d            \n\t"
" trn1            v28.2d, v2.2d, v6.2d            \n\t"
" trn2            v29.2d, v2.2d, v6.2d            \n\t"
" trn1            v30.2d, v3.2d, v7.2d            \n\t"
" trn2            v31.2d, v3.2d, v7.2d            \n\t"
"                                                 \n\t"
" trn1            v0.2d, v8.2d, v12.2d            \n\t" // Row 2-3.
" trn2            v1.2d, v8.2d, v12.2d            \n\t"
" trn1            v2.2d, v9.2d, v13.2d            \n\t"
" trn2            v3.2d, v9.2d, v13.2d            \n\t"
" trn1            v4.2d, v10.2d, v14.2d           \n\t"
" trn2            v5.2d, v10.2d, v14.2d           \n\t"
" trn1            v6.2d, v11.2d, v15.2d           \n\t"
" trn2            v7.2d, v11.2d, v15.2d           \n\t"
"                                                 \n\t"
" trn1            v8.2d, v16.2d, v20.2d           \n\t" // Row 4-5.
" trn2            v9.2d, v16.2d, v20.2d           \n\t"
" trn1            v10.2d, v17.2d, v21.2d          \n\t" // AMARI
" trn2            v11.2d, v17.2d, v21.2d          \n\t" // AMARI
" trn1            v12.2d, v18.2d, v22.2d          \n\t" // AMARI
" trn2            v13.2d, v18.2d, v22.2d          \n\t" // AMARI
" trn1            v14.2d, v19.2d, v23.2d          \n\t" // AMARI
" trn2            v15.2d, v19.2d, v23.2d          \n\t" // AMARI
"                                                 \n\t"
" ld1r            {v16.2d}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v17.2d}, [x8]                  \n\t"
" fmov            d18, #1.0                       \n\t"
" fcmp            d16, d18                        \n\t"
BEQ(UNIT_ALPHA_C)
DSCALE8V(24,25,26,27,28,29,30,31,16,0)
DSCALE8V(0,1,2,3,4,5,6,7,16,0)
DSCALE8V(8,9,10,11,12,13,14,15,16,0)
LABEL(UNIT_ALPHA_C)
" fcmp            d17, #0.0                       \n\t"
BEQ(ZERO_BETA_C_1)
DLOADC_3V_C_FWD(18,19,20,x1,0,x7)
DLOADC_3V_C_FWD(21,22,23,x1,0,x7)
DSCALEA6V(24,0,8,25,1,9,18,19,20,21,22,23,17,0)
LABEL(ZERO_BETA_C_1)
DSTOREC_3V_C_FWD(24,0,8,x5,0,x7)
DSTOREC_3V_C_FWD(25,1,9,x5,0,x7)
BEQ(ZERO_BETA_C_2)
DLOADC_3V_C_FWD(18,19,20,x1,0,x7)
DLOADC_3V_C_FWD(21,22,23,x1,0,x7)
DLOADC_3V_C_FWD(24,0,8,x1,0,x7)
DLOADC_3V_C_FWD(25,1,9,x1,0,x7)
DSCALEA6V(26,2,10,27,3,11,18,19,20,21,22,23,17,0)
DSCALEA6V(28,4,12,29,5,13,24,0,8,25,1,9,17,0)
LABEL(ZERO_BETA_C_2)
#ifndef __clang__
" cmp   x12, #1                       \n\t"
BRANCH(PRFM_END_C)
" prfm  PLDL1KEEP, [%[a_next], #16*0] \n\t"
" prfm  PLDL1KEEP, [%[a_next], #16*1] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*0] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*1] \n\t"
LABEL(PRFM_END_C)
" fcmp            d17, #0.0           \n\t" // Not the end. Reset branching reg.
#endif
DSTOREC_3V_C_FWD(26,2,10,x5,0,x7)
DSTOREC_3V_C_FWD(27,3,11,x5,0,x7)
BEQ(ZERO_BETA_C_3)
DLOADC_3V_C_FWD(18,19,20,x1,0,x7)
DLOADC_3V_C_FWD(21,22,23,x1,0,x7)
DSCALEA6V(30,6,14,31,7,15,18,19,20,21,22,23,17,0)
LABEL(ZERO_BETA_C_3)
DSTOREC_3V_C_FWD(28,4,12,x5,0,x7)
DSTOREC_3V_C_FWD(29,5,13,x5,0,x7)
DSTOREC_3V_C_FWD(30,6,14,x5,0,x7)
DSTOREC_3V_C_FWD(31,7,15,x5,0,x7)
//
// End of this microkernel.
LABEL(END_WRITE_MEM)
"                                                 \n\t"
" subs            x12, x12, #1                    \n\t"
BEQ(END_EXEC)
"                                                 \n\t"
" mov             x8, #6                          \n\t"
" madd            x13, x6, x8, x13                \n\t" // Forward C's base address to the next logic panel.
" add             x10, x10, x11                   \n\t" // Forward A's base address to the next logic panel.
BRANCH(MILLIKER_MLOOP)
//
// End of execution.
LABEL(END_EXEC)
:
: [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_a]   "m" (rs_a),
  [cs_a]   "m" (cs_a),
  [ps_a]   "m" (ps_a),
  [rs_b]   "m" (rs_b),
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
  // Forward address.
  a = ( double* )a + m_iter * ps_a;
  c = ( double* )c + m_iter * 6 * rs_c;
  auxinfo_t data_d6x4mn = *data;
  bli_auxinfo_set_ps_b( 4 * cs_b0, &data_d6x4mn );
  bli_dgemmsup_rv_armv8a_int_6x4mn
  (
    conja, conjb, m_left, 8, k0,
    alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
    beta, c, rs_c0, cs_c0, &data_d6x4mn, cntx
  );

}

