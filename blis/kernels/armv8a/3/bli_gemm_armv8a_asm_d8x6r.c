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

// Label locality & misc.
#include "armv8a_asm_utils.h"

// Nanokernel operations.
#include "armv8a_asm_d2x2.h"

/* Order of row-major SGEMM_12x8's execution in 4x5 blocks:
 *
 * +---+ +---+
 * | 0 | | 1 |
 * +---+ +---+
 * +---+ +---+
 * | 2 | | 3 |
 * +---+ +---+
 * +---+ +---+
 * | 4 | | 5 |
 * +---+ +---+
 */
#define SGEMM_12X8_MKER_LOOP_PLAIN(C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,C60,C61,C70,C71,C80,C81,C90,C91,CA0,CA1,CB0,CB1,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  SGEMM_4X4_NANOKERNEL(C00,C10,C20,C30,B0,A0) \
  SGEMM_4X4_NANOKERNEL(C01,C11,C21,C31,B1,A0) \
  DGEMM_LOAD1V_ ##LOADNEXT (A0,AADDR,ASHIFT) /* Contiguous load is the same across S/D. */ \
  SGEMM_4X4_NANOKERNEL(C40,C50,C60,C70,B0,A1) \
  SGEMM_4X4_NANOKERNEL(C41,C51,C61,C71,B1,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (A1,AADDR,ASHIFT+16) \
  SGEMM_4X4_NANOKERNEL(C80,C90,CA0,CB0,B0,A2) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  SGEMM_4X4_NANOKERNEL(C81,C91,CA1,CB1,B1,A2)

// For contiguous storage of C, SLOAD is the same as DLOAD.
#define SLOADC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define SSTOREC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

/* Order of row-major DGEMM_8x6's execution in 2x2 blocks:
 *
 * +---+ +---+ +---+
 * | 0 | | 2 | | 4 |
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 1 | | 3 | | 5 |
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 6 | | 8 | | 10|
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 7 | | 9 | | 11|
 * +---+ +---+ +---+
 *
 */
#define DGEMM_8X6_MKER_LOOP_PLAIN(C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_2X2_NANOKERNEL(C00,C10,B0,A0) \
  DGEMM_2X2_NANOKERNEL(C20,C30,B0,A1) \
  DGEMM_2X2_NANOKERNEL(C01,C11,B1,A0) \
  DGEMM_2X2_NANOKERNEL(C21,C31,B1,A1) \
  DGEMM_2X2_NANOKERNEL(C02,C12,B2,A0) \
  DGEMM_2X2_NANOKERNEL(C22,C32,B2,A1) \
  DGEMM_LOAD2V_ ##LOADNEXT (A0,A1,AADDR,ASHIFT) \
  DGEMM_2X2_NANOKERNEL(C40,C50,B0,A2) \
  DGEMM_2X2_NANOKERNEL(C60,C70,B0,A3) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  DGEMM_2X2_NANOKERNEL(C41,C51,B1,A2) \
  DGEMM_2X2_NANOKERNEL(C61,C71,B1,A3) \
  DGEMM_LOAD1V_ ##LOADNEXT (B1,BADDR,BSHIFT+16) \
  DGEMM_2X2_NANOKERNEL(C42,C52,B2,A2) \
  DGEMM_2X2_NANOKERNEL(C62,C72,B2,A3)

// Interleaving load or not.
#define DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  DLOAD1V(V1,ADDR,IMM)

#define DGEMM_LOAD2V_noload(V1,V2,ADDR,IMM)
#define DGEMM_LOAD2V_load(V1,V2,ADDR,IMM) \
  DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  DGEMM_LOAD1V_load(V2,ADDR,IMM+16)

// For contiguous storage of C.
#define DLOADC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
  DLOAD1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
  DSTORE1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

// Prefetch C.
#define PRFMC_FWD(CADDR,RSC,LASTB) \
" prfm PLDL1KEEP, ["#CADDR"]           \n\t" \
" prfm PLDL1KEEP, ["#CADDR", "#LASTB"] \n\t" \
" add  "#CADDR", "#CADDR", "#RSC"      \n\t"

void bli_sgemm_armv8a_asm_12x8r
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
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

  GEMM_UKR_SETUP_CT( s, 12, 8, true );

  __asm__ volatile
  (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #12                         \n\t" // Column-skip of A.
" mov             x3, #8                          \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C. (column-skip == 1)
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(float).
" lsl             x2, x2, #2                      \n\t" // cs_a
" lsl             x3, x3, #2                      \n\t" // rs_b
" lsl             x6, x6, #2                      \n\t" // rs_c
"                                                 \n\t"
" cmp             %w[ct], wzr                     \n\t"
" mov             x9, x5                          \n\t"
BNE(SEND_PRFMC_FH)
PRFMC_FWD(x9,x6,32) // Prefetch C 01/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 02/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 03/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 04/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 05/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 06/12.
LABEL(SEND_PRFMC_FH)
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define SGEMM_12X8_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  SGEMM_12X8_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT)
// Load from memory.
LABEL(SLOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(SCLEAR_CCOLS)
"                                                 \n\t"
" ldr             q24, [x0, #16*0]                \n\t" // Load A.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q27, [x0, #16*0]                \n\t"
"                                                 \n\t"
" cmp             %w[ct], wzr                     \n\t"
BNE(SEND_PRFMC_LH)
PRFMC_FWD(x9,x6,32) // Prefetch C 07/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 08/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 09/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 10/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 11/12.
PRFMC_FWD(x9,x6,32) // Prefetch C 12/12.
LABEL(SEND_PRFMC_LH)
" cmp             x4, #0                          \n\t" // Reset branching flag.
"                                                 \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B.
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q30, [x1, #16*0]                \n\t"
" ldr             q31, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
LABEL(SCLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
// No-microkernel early return, once again.
BEQ(SK_LEFT_LOOP)
//
// Microkernel is defined here as:
#define SGEMM_12X8_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,A2,B0,B1) \
  SGEMM_12X8_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,x0,16,x1,0,load) \
 "add             x0, x0, x2                      \n\t" \
 "ldr             q"#A2", [x0, #16*0]             \n\t" \
 "ldr             q"#B1", [x1, #16*1]             \n\t" \
 "add             x1, x1, x3                      \n\t"
// Start microkernel loop.
LABEL(SK_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,28,29)
SGEMM_12X8_MKER_LOOP_PLAIN_LOC_FWD(27,24,25,30,31)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(SFIN_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_PLAIN_LOC_FWD(26,27,24,28,29)
SGEMM_12X8_MKER_LOOP_PLAIN_LOC_FWD(25,26,27,30,31)
BRANCH(SK_MKER_LOOP)
//
// Final microkernel loop.
LABEL(SFIN_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_PLAIN_LOC(26,27,24,28,29,xzr,-1,xzr,-1,noload)
" ldr             q26, [x0, #16*1]                \n\t"
" ldr             q27, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
SGEMM_12X8_MKER_LOOP_PLAIN_LOC(25,26,27,30,31,xzr,-1,xzr,-1,noload)
//
// Loops left behind microkernels.
LABEL(SK_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(SWRITE_MEM_PREP)
" ldr             q24, [x0, #16*0]                \n\t" // Load A col.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B row.
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
SGEMM_12X8_MKER_LOOP_PLAIN_LOC(24,25,26,28,29,xzr,-1,xzr,-1,noload)
BRANCH(SK_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(SWRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v24.4s}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v25.4s}, [x8]                  \n\t"
"                                                 \n\t"
LABEL(SPREFETCH_ABNEXT)
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
" prfm            PLDL1STRM, [x0, 64*0]           \n\t" // Do not know cache line size,
" prfm            PLDL1STRM, [x0, 64*1]           \n\t" //  issue some number of prfm instructions
" prfm            PLDL1STRM, [x0, 64*2]           \n\t" //  to try to activate hardware prefetcher.
" prfm            PLDL1STRM, [x1, 64*0]           \n\t"
" prfm            PLDL1STRM, [x1, 64*1]           \n\t"
" prfm            PLDL1STRM, [x1, 64*3]           \n\t"
"                                                 \n\t"
" fmov            d26, #1.0                       \n\t"
" fcvt            s26, d26                        \n\t"
" fcmp            s24, s26                        \n\t"
BEQ(SUNIT_ALPHA)
SSCALE8V(0,1,2,3,4,5,6,7,24,0)
SSCALE8V(8,9,10,11,12,13,14,15,24,0)
SSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(SUNIT_ALPHA)
"                                                 \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
//
// Contiguous C-storage.
LABEL(SWRITE_MEM_R)
" fcmp            s25, #0.0                       \n\t" // Sets conditional flag whether *beta == 0.
"                                                 \n\t" // This conditional flag will be used
"                                                 \n\t" //  multiple times for skipping load.
// Row 0 & 1 & 2:
BEQ(SZERO_BETA_R_0_1_2)
SLOADC_2V_R_FWD(26,27,x9,0,x6)
SLOADC_2V_R_FWD(28,29,x9,0,x6)
SLOADC_2V_R_FWD(30,31,x9,0,x6)
SSCALEA2V(0,1,26,27,25,0)
SSCALEA2V(2,3,28,29,25,0)
SSCALEA2V(4,5,30,31,25,0)
LABEL(SZERO_BETA_R_0_1_2)
SSTOREC_2V_R_FWD(0,1,x5,0,x6)
SSTOREC_2V_R_FWD(2,3,x5,0,x6)
SSTOREC_2V_R_FWD(4,5,x5,0,x6)
// Row 3 & 4 & 5 & 6 & 7 & 8:
BEQ(SZERO_BETA_R_3_4_5_6_7_8)
SLOADC_2V_R_FWD(26,27,x9,0,x6)
SLOADC_2V_R_FWD(28,29,x9,0,x6)
SLOADC_2V_R_FWD(30,31,x9,0,x6)
SLOADC_2V_R_FWD(0,1,x9,0,x6)
SLOADC_2V_R_FWD(2,3,x9,0,x6)
SLOADC_2V_R_FWD(4,5,x9,0,x6)
SSCALEA4V(6,7,8,9,26,27,28,29,25,0)
SSCALEA4V(10,11,12,13,30,31,0,1,25,0)
SSCALEA4V(14,15,16,17,2,3,4,5,25,0)
LABEL(SZERO_BETA_R_3_4_5_6_7_8)
SSTOREC_2V_R_FWD(6,7,x5,0,x6)
SSTOREC_2V_R_FWD(8,9,x5,0,x6)
SSTOREC_2V_R_FWD(10,11,x5,0,x6)
SSTOREC_2V_R_FWD(12,13,x5,0,x6)
SSTOREC_2V_R_FWD(14,15,x5,0,x6)
SSTOREC_2V_R_FWD(16,17,x5,0,x6)
// Row 9 & 10 & 11
BEQ(SZERO_BETA_R_9_10_11)
SLOADC_2V_R_FWD(26,27,x9,0,x6)
SLOADC_2V_R_FWD(28,29,x9,0,x6)
SLOADC_2V_R_FWD(30,31,x9,0,x6)
SSCALEA2V(18,19,26,27,25,0)
SSCALEA2V(20,21,28,29,25,0)
SSCALEA2V(22,23,30,31,25,0)
LABEL(SZERO_BETA_R_9_10_11)
SSTOREC_2V_R_FWD(18,19,x5,0,x6)
SSTOREC_2V_R_FWD(20,21,x5,0,x6)
SSTOREC_2V_R_FWD(22,23,x5,0,x6)
// Done.
LABEL(SEND_WRITE_MEM)
:
: [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_c]   "m" (rs_c),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next),
  [ct]     "r" (_use_ct) // Defined by macro.
: "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",
  "v0","v1","v2","v3","v4","v5","v6","v7",
  "v8","v9","v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19",
  "v20","v21","v22","v23",
  "v24","v25","v26","v27",
  "v28","v29","v30","v31"
  );

  GEMM_UKR_FLUSH_CT( s );
}

/*
 * Differences from the col-major 6x8 in HW modeling:
 * * Stream HW prefetcher is assumed s.t. PRFM instructions for packed A&B are omitted.
 */
void bli_dgemm_armv8a_asm_8x6r
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
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

  GEMM_UKR_SETUP_CT( d, 8, 6, true );

  __asm__ volatile
  (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #8                          \n\t" // Column-skip of A.
" mov             x3, #6                          \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C. (column-skip == 1)
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(double).
" lsl             x2, x2, #3                      \n\t" // cs_a
" lsl             x3, x3, #3                      \n\t" // rs_b
" lsl             x6, x6, #3                      \n\t" // rs_c
"                                                 \n\t"
" cmp             %w[ct], wzr                     \n\t"
" mov             x9, x5                          \n\t"
BNE(DEND_PRFMC)
PRFMC_FWD(x9,x6,40) // Prefetch C 1/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 2/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 3/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 4/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 5/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 6/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 7/8.
PRFMC_FWD(x9,x6,40) // Prefetch C 8/8.
LABEL(DEND_PRFMC)
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define DGEMM_8X6_MKER_LOOP_PLAIN_LOC(A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_8X6_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT)
// Load from memory.
LABEL(DLOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(DCLEAR_CCOLS)
"                                                 \n\t"
" ldr             q24, [x0, #16*0]                \n\t" // Load A.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" ldr             q27, [x0, #16*3]                \n\t"
" add             x0, x0, x2                      \n\t"
"                                                 \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B.
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q31, [x1, #16*0]                \n\t"
LABEL(DCLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
// No-microkernel early return, once again.
BEQ(DK_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_8X6_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_PLAIN_LOC(A0,A1,A2,A3,B0,B1,B2,x0,0,x1,16,load) \
 "add             x1, x1, x3                      \n\t" \
 "ldr             q"#B2", [x1, #16*0]             \n\t" \
 "ldr             q"#A2", [x0, #16*2]             \n\t" \
 "ldr             q"#A3", [x0, #16*3]             \n\t" \
 "add             x0, x0, x2                      \n\t"
// Start microkernel loop.
LABEL(DK_MKER_LOOP)
DGEMM_8X6_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,27,28,29,30)
DGEMM_8X6_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,27,31,28,29)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(DFIN_MKER_LOOP)
DGEMM_8X6_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,27,30,31,28)
DGEMM_8X6_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,27,29,30,31)
BRANCH(DK_MKER_LOOP)
//
// Final microkernel loop.
LABEL(DFIN_MKER_LOOP)
DGEMM_8X6_MKER_LOOP_PLAIN_LOC(24,25,26,27,30,31,28,x0,0,x1,16,load)
" add             x1, x1, x3                      \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" ldr             q27, [x0, #16*3]                \n\t"
" add             x0, x0, x2                      \n\t"
DGEMM_8X6_MKER_LOOP_PLAIN_LOC(24,25,26,27,29,30,31,xzr,-1,xzr,-1,noload)
//
// Loops left behind microkernels.
LABEL(DK_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(DWRITE_MEM_PREP)
" ldr             q24, [x0, #16*0]                \n\t" // Load A col.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" ldr             q27, [x0, #16*3]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B row.
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
DGEMM_8X6_MKER_LOOP_PLAIN_LOC(24,25,26,27,28,29,30,xzr,-1,xzr,-1,noload)
BRANCH(DK_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(DWRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v24.2d}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v25.2d}, [x8]                  \n\t"
"                                                 \n\t"
LABEL(DPREFETCH_ABNEXT)
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
" prfm            PLDL1STRM, [x0, 64*0]           \n\t" // Do not know cache line size,
" prfm            PLDL1STRM, [x0, 64*1]           \n\t" //  issue some number of prfm instructions
" prfm            PLDL1STRM, [x0, 64*2]           \n\t" //  to try to activate hardware prefetcher.
" prfm            PLDL1STRM, [x1, 64*0]           \n\t"
" prfm            PLDL1STRM, [x1, 64*1]           \n\t"
" prfm            PLDL1STRM, [x1, 64*3]           \n\t"
"                                                 \n\t"
" fmov            d26, #1.0                       \n\t"
" fcmp            d24, d26                        \n\t"
BEQ(DUNIT_ALPHA)
DSCALE8V(0,1,2,3,4,5,6,7,24,0)
DSCALE8V(8,9,10,11,12,13,14,15,24,0)
DSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(DUNIT_ALPHA)
"                                                 \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
//
// Contiguous C-storage.
LABEL(DWRITE_MEM_R)
" fcmp            d25, #0.0                       \n\t" // Sets conditional flag whether *beta == 0.
"                                                 \n\t" // This conditional flag will be used
"                                                 \n\t" //  multiple times for skipping load.
// Row 0 & 1:
BEQ(DZERO_BETA_R_0_1)
DLOADC_3V_R_FWD(26,27,28,x9,0,x6)
DLOADC_3V_R_FWD(29,30,31,x9,0,x6)
DSCALEA2V(0,1,26,27,25,0)
DSCALEA2V(2,3,28,29,25,0)
DSCALEA2V(4,5,30,31,25,0)
LABEL(DZERO_BETA_R_0_1)
DSTOREC_3V_R_FWD(0,1,2,x5,0,x6)
DSTOREC_3V_R_FWD(3,4,5,x5,0,x6)
// Row 2 & 3 & 4 & 5:
BEQ(DZERO_BETA_R_2_3_4_5)
DLOADC_3V_R_FWD(26,27,28,x9,0,x6)
DLOADC_3V_R_FWD(29,30,31,x9,0,x6)
DLOADC_3V_R_FWD(0,1,2,x9,0,x6)
DLOADC_3V_R_FWD(3,4,5,x9,0,x6)
DSCALEA4V(6,7,8,9,26,27,28,29,25,0)
DSCALEA4V(10,11,12,13,30,31,0,1,25,0)
DSCALEA4V(14,15,16,17,2,3,4,5,25,0)
LABEL(DZERO_BETA_R_2_3_4_5)
DSTOREC_3V_R_FWD(6,7,8,x5,0,x6)
DSTOREC_3V_R_FWD(9,10,11,x5,0,x6)
DSTOREC_3V_R_FWD(12,13,14,x5,0,x6)
DSTOREC_3V_R_FWD(15,16,17,x5,0,x6)
// Row 6 & 7
BEQ(DZERO_BETA_R_6_7)
DLOADC_3V_R_FWD(26,27,28,x9,0,x6)
DLOADC_3V_R_FWD(29,30,31,x9,0,x6)
DSCALEA2V(18,19,26,27,25,0)
DSCALEA2V(20,21,28,29,25,0)
DSCALEA2V(22,23,30,31,25,0)
LABEL(DZERO_BETA_R_6_7)
DSTOREC_3V_R_FWD(18,19,20,x5,0,x6)
DSTOREC_3V_R_FWD(21,22,23,x5,0,x6)
// Done.
LABEL(DEND_WRITE_MEM)
:
: [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_c]   "m" (rs_c),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next),
  [ct]     "r" (_use_ct) // Defined by macro.
: "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",
  "v0","v1","v2","v3","v4","v5","v6","v7",
  "v8","v9","v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19",
  "v20","v21","v22","v23",
  "v24","v25","v26","v27",
  "v28","v29","v30","v31"
  );

  GEMM_UKR_FLUSH_CT( d );
}

