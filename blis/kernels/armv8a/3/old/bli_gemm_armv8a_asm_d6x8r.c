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
#define DGEMM_6X8_MKER_LOOP_PLAIN(C00,C01,C02,C03,C10,C11,C12,C13,C20,C21,C22,C23,C30,C31,C32,C33,C40,C41,C42,C43,C50,C51,C52,C53,A0,A1,A2,B0,B1,B2,B3,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_2X2_NANOKERNEL(C00,C10,B0,A0) \
  DGEMM_2X2_NANOKERNEL(C01,C11,B1,A0) \
  DGEMM_2X2_NANOKERNEL(C20,C30,B0,A1) \
  DGEMM_2X2_NANOKERNEL(C21,C31,B1,A1) \
  DGEMM_2X2_NANOKERNEL(C40,C50,B0,A2) \
  DGEMM_2X2_NANOKERNEL(C41,C51,B1,A2) \
  DGEMM_LOAD2V_ ##LOADNEXT (B0,B1,BADDR,BSHIFT) \
  DGEMM_2X2_NANOKERNEL(C02,C12,B2,A0) \
  DGEMM_2X2_NANOKERNEL(C03,C13,B3,A0) \
  DGEMM_LOAD1V_ ##LOADNEXT (A0,AADDR,ASHIFT) \
  DGEMM_2X2_NANOKERNEL(C22,C32,B2,A1) \
  DGEMM_2X2_NANOKERNEL(C23,C33,B3,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (A1,AADDR,ASHIFT+16) \
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

// For contiguous storage of C.
#define DLOADC_4V_R_FWD(C0,C1,C2,C3,CADDR,CSHIFT,RSC) \
  DLOAD4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_4V_R_FWD(C0,C1,C2,C3,CADDR,CSHIFT,RSC) \
  DSTORE4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DPRFMC_FWD(CADDR,RSC) \
" prfm PLDL1KEEP, ["#CADDR"]      \n\t" \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

// For scattered storage of C.
#define DLOADC_GATHER_4V_R_FWD(C0,C1,C2,C3,CADDR,CELEM,CSC,RSC) \
" mov  "#CELEM", "#CADDR"         \n\t" \
  DLOAD1V_GATHER_ELMFWD(C0,CELEM,CSC) \
  DLOAD1V_GATHER_ELMFWD(C1,CELEM,CSC) \
  DLOAD1V_GATHER_ELMFWD(C2,CELEM,CSC) \
  DLOAD1V_GATHER_ELMFWD(C3,CELEM,CSC) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

#define DSTOREC_SCATTER_4V_R_FWD(C0,C1,C2,C3,CADDR,CELEM,CSC,RSC) \
" mov  "#CELEM", "#CADDR"         \n\t" \
  DSTORE1V_SCATTER_ELMFWD(C0,CELEM,CSC) \
  DSTORE1V_SCATTER_ELMFWD(C1,CELEM,CSC) \
  DSTORE1V_SCATTER_ELMFWD(C2,CELEM,CSC) \
  DSTORE1V_SCATTER_ELMFWD(C3,CELEM,CSC) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"


void bli_dgemm_armv8a_asm_6x8r
     (
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
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

  __asm__ volatile
  (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #6                          \n\t" // Column-skip of A.
" mov             x3, #8                          \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(double).
" lsl             x2, x2, #3                      \n\t" // cs_a
" lsl             x3, x3, #3                      \n\t" // rs_b
" lsl             x6, x6, #3                      \n\t" // rs_c
" lsl             x7, x7, #3                      \n\t" // cs_c
"                                                 \n\t"
" mov             x9, x5                          \n\t"
" cmp             x7, #8                          \n\t" // Do not prefetch C for generic strided.
BNE(C_PREFETCH_END)
DPRFMC_FWD(x9,x6)
DPRFMC_FWD(x9,x6)
DPRFMC_FWD(x9,x6)
DPRFMC_FWD(x9,x6)
DPRFMC_FWD(x9,x6)
DPRFMC_FWD(x9,x6)
LABEL(C_PREFETCH_END)
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define DGEMM_6X8_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_6X8_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,B0,B1,B2,B3,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT)
// Load from memory.
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" ldr             q24, [x0, #16*0]                \n\t" // Load A.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q27, [x0, #16*0]                \n\t"
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
  DGEMM_6X8_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3,x0,1*16,x1,0,load) \
 "add             x0, x0, x2                      \n\t" \
 "ldr             q"#A2", [x0, #16*0]             \n\t" \
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
DGEMM_6X8_MKER_LOOP_PLAIN_LOC(26,27,24,28,29,30,31,x0,1*16,x1,0,load)
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
" ldr             q24, [x0, #16*0]                \n\t" // Load A col.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
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
" ld1r            {v24.2d}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v25.2d}, [x8]                  \n\t"
"                                                 \n\t"
LABEL(PREFETCH_ABNEXT)
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
BEQ(UNIT_ALPHA)
DSCALE8V(0,1,2,3,4,5,6,7,24,0)
DSCALE8V(8,9,10,11,12,13,14,15,24,0)
DSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(UNIT_ALPHA)
"                                                 \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x7, #8                          \n\t" // Check for generic storage.
BNE(WRITE_MEM_G)
//
// Contiguous C-storage.
LABEL(WRITE_MEM_R)
" fcmp            d25, #0.0                       \n\t" // Sets conditional flag whether *beta == 0.
"                                                 \n\t" // This conditional flag will be used
"                                                 \n\t" //  multiple times for skipping load.
// Row 0:
BEQ(ZERO_BETA_R_0)
DLOADC_4V_R_FWD(26,27,28,29,x9,0,x6)
DSCALEA4V(0,1,2,3,26,27,28,29,25,0)
LABEL(ZERO_BETA_R_0)
DSTOREC_4V_R_FWD(0,1,2,3,x5,0,x6)
// Row 1 & 2:
BEQ(ZERO_BETA_R_1_2)
DLOADC_4V_R_FWD(26,27,28,29,x9,0,x6)
DLOADC_4V_R_FWD(0,1,2,3,x9,0,x6)
DSCALEA8V(4,5,6,7,8,9,10,11,26,27,28,29,0,1,2,3,25,0)
LABEL(ZERO_BETA_R_1_2)
DSTOREC_4V_R_FWD(4,5,6,7,x5,0,x6)
DSTOREC_4V_R_FWD(8,9,10,11,x5,0,x6)
// Row 3 & 4 & 5:
BEQ(ZERO_BETA_R_3_4_5)
DLOADC_4V_R_FWD(0,1,2,3,x9,0,x6)
DLOADC_4V_R_FWD(4,5,6,7,x9,0,x6)
DLOADC_4V_R_FWD(8,9,10,11,x9,0,x6)
DSCALEA8V(12,13,14,15,16,17,18,19,0,1,2,3,4,5,6,7,25,0)
DSCALEA4V(20,21,22,23,8,9,10,11,25,0)
LABEL(ZERO_BETA_R_3_4_5)
DSTOREC_4V_R_FWD(12,13,14,15,x5,0,x6)
DSTOREC_4V_R_FWD(16,17,18,19,x5,0,x6)
DSTOREC_4V_R_FWD(20,21,22,23,x5,0,x6)
BRANCH(END_WRITE_MEM)
//
// Generic-strided C-storage.
LABEL(WRITE_MEM_G)
" fcmp            d25, #0.0                       \n\t" // Sets conditional flag whether *beta == 0.
"                                                 \n\t"
// Row 0:
BEQ(ZERO_BETA_G_0)
DLOADC_GATHER_4V_R_FWD(26,27,28,29,x9,x0,x7,x6)
DSCALEA4V(0,1,2,3,26,27,28,29,25,0)
LABEL(ZERO_BETA_G_0)
DSTOREC_SCATTER_4V_R_FWD(0,1,2,3,x5,x1,x7,x6)
// Row 1 & 2:
BEQ(ZERO_BETA_G_1_2)
DLOADC_GATHER_4V_R_FWD(26,27,28,29,x9,x0,x7,x6)
DLOADC_GATHER_4V_R_FWD(0,1,2,3,x9,x0,x7,x6)
DSCALEA8V(4,5,6,7,8,9,10,11,26,27,28,29,0,1,2,3,25,0)
LABEL(ZERO_BETA_G_1_2)
DSTOREC_SCATTER_4V_R_FWD(4,5,6,7,x5,x1,x7,x6)
DSTOREC_SCATTER_4V_R_FWD(8,9,10,11,x5,x1,x7,x6)
// Row 3 & 4 & 5:
BEQ(ZERO_BETA_G_3_4_5)
DLOADC_GATHER_4V_R_FWD(0,1,2,3,x9,x0,x7,x6)
DLOADC_GATHER_4V_R_FWD(4,5,6,7,x9,x0,x7,x6)
DLOADC_GATHER_4V_R_FWD(8,9,10,11,x9,x0,x7,x6)
DSCALEA8V(12,13,14,15,16,17,18,19,0,1,2,3,4,5,6,7,25,0)
DSCALEA4V(20,21,22,23,8,9,10,11,25,0)
LABEL(ZERO_BETA_G_3_4_5)
DSTOREC_SCATTER_4V_R_FWD(12,13,14,15,x5,x1,x7,x6)
DSTOREC_SCATTER_4V_R_FWD(16,17,18,19,x5,x1,x7,x6)
DSTOREC_SCATTER_4V_R_FWD(20,21,22,23,x5,x1,x7,x6)
LABEL(END_WRITE_MEM)
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
: "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9",
  "v0","v1","v2","v3","v4","v5","v6","v7",
  "v8","v9","v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19",
  "v20","v21","v22","v23",
  "v24","v25","v26","v27",
  "v28","v29","v30","v31"
  );
}

