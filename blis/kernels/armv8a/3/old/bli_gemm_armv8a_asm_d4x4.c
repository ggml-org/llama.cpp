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
#include "armv8a_asm_utils.h"

// Nanokernel operations.
#include "armv8a_asm_d2x2.h"

#define DGEMM_4X4_MKER_LOOP_PLAIN(C00,C10,C01,C11,C02,C12,C03,C13,A0,A1,B0,B1) \
  DGEMM_2X2_NANOKERNEL(C00,C01,A0,B0) \
  DGEMM_2X2_NANOKERNEL(C10,C11,A1,B0) \
  DGEMM_2X2_NANOKERNEL(C02,C03,A0,B1) \
  DGEMM_2X2_NANOKERNEL(C12,C13,A1,B1)

// For contiguous storage of C.
#define DLOADC_2V_C_FWD(C0,C1,CADDR,CSHIFT,LDC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#LDC" \n\t"
#define DSTOREC_2V_C_FWD(C0,C1,CADDR,CSHIFT,LDC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#LDC" \n\t"

void bli_dgemm_armv8a_asm_4x4
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
  uint64_t k_mker = k0 / 6;
  uint64_t k_left = k0 % 6;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  __asm__ volatile
  (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #4                          \n\t" // Column-skip of A.
" mov             x3, #4                          \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
"                                                 \n\t"
" mov             x8, #8                          \n\t" // Multiply some address skips by sizeof(double).
" madd            x2, x8, x2, xzr                 \n\t" // cs_a
" madd            x3, x8, x3, xzr                 \n\t" // rs_b
" madd            x7, x8, x7, xzr                 \n\t" // cs_c
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:7 ] <- C
//  V[ 8:19] <- B
//  V[20:31] <- A
// Under this scheme, the following is defined:
#define DGEMM_4X4_MKER_LOOP_PLAIN_LOC(A0,A1,B0,B1) \
  DGEMM_4X4_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,A0,A1,B0,B1)
// TODO: Prefetch C.
// Load from memory.
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" ldr             q20, [x0, #16*0]                \n\t"
" ldr             q21, [x0, #16*1]                \n\t"
" ldr             q22, [x0, #16*2]                \n\t"
" ldr             q23, [x0, #16*3]                \n\t"
" ldr             q24, [x0, #16*4]                \n\t"
" ldr             q25, [x0, #16*5]                \n\t"
" add             x0, x0, x2                      \n\t"
" add             x0, x0, x2                      \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q26, [x0, #16*0]                \n\t"
" ldr             q27, [x0, #16*1]                \n\t"
" ldr             q28, [x0, #16*2]                \n\t"
" ldr             q29, [x0, #16*3]                \n\t"
" ldr             q30, [x0, #16*4]                \n\t"
" ldr             q31, [x0, #16*5]                \n\t"
" add             x0, x0, x2                      \n\t"
" add             x0, x0, x2                      \n\t"
" add             x0, x0, x2                      \n\t"
"                                                 \n\t"
" ldr             q8, [x1, #16*0]                 \n\t"
" ldr             q9, [x1, #16*1]                 \n\t"
" ldr             q10, [x1, #16*2]                \n\t"
" ldr             q11, [x1, #16*3]                \n\t"
" ldr             q12, [x1, #16*4]                \n\t"
" ldr             q13, [x1, #16*5]                \n\t"
" add             x1, x1, x3                      \n\t"
" add             x1, x1, x3                      \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q14, [x1, #16*0]                \n\t"
" ldr             q15, [x1, #16*1]                \n\t"
" ldr             q16, [x1, #16*2]                \n\t"
" ldr             q17, [x1, #16*3]                \n\t"
" ldr             q18, [x1, #16*4]                \n\t"
" ldr             q19, [x1, #16*5]                \n\t"
" add             x1, x1, x3                      \n\t"
" add             x1, x1, x3                      \n\t"
" add             x1, x1, x3                      \n\t"
"                                                 \n\t"
LABEL(CLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
// No-microkernel early return, once again.
BEQ(K_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,B0,B1) \
  DGEMM_4X4_MKER_LOOP_PLAIN_LOC(A0,A1,B0,B1) \
 "ldr             q"#A0", [x0, #16*0]             \n\t" \
 "ldr             q"#A1", [x0, #16*1]             \n\t" \
 "add             x0, x0, x2                      \n\t" \
 "ldr             q"#B0", [x1, #16*0]             \n\t" \
 "ldr             q"#B1", [x1, #16*1]             \n\t" \
 "add             x1, x1, x3                      \n\t"
// Start microkernel loop.
LABEL(K_MKER_LOOP)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(FIN_MKER_LOOP)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(20,21,8,9)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(22,23,10,11)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(24,25,12,13)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(26,27,14,15)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(28,29,16,17)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC_FWD(30,31,18,19)
BRANCH(K_MKER_LOOP)
//
// Final microkernel loop.
LABEL(FIN_MKER_LOOP)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(20,21,8,9)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(22,23,10,11)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(24,25,12,13)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(26,27,14,15)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(28,29,16,17)
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(30,31,18,19)
//
// Loops left behind microkernels.
LABEL(K_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(WRITE_MEM_PREP)
" ldr             q20, [x0, #16*0]                \n\t"
" ldr             q21, [x0, #16*1]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q8, [x1, #16*0]                 \n\t"
" ldr             q9, [x1, #16*1]                 \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
DGEMM_4X4_MKER_LOOP_PLAIN_LOC(20,21,8,9)
BRANCH(K_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(WRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ldr             d8, [x4]                        \n\t" // Load alpha & beta (value).
" ldr             d9, [x8]                        \n\t"
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
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x6, #1                          \n\t" // Check for generic storage.
BNE(WRITE_MEM_G)
//
// Contiguous C-storage.
LABEL(WRITE_MEM_C)
DLOADC_2V_C_FWD(10,11,x9,0,x7)
DLOADC_2V_C_FWD(12,13,x9,0,x7)
DLOADC_2V_C_FWD(14,15,x9,0,x7)
DLOADC_2V_C_FWD(16,17,x9,0,x7)
DSCALE8V(10,11,12,13,14,15,16,17,9,0)
DSCALEA8V(10,11,12,13,14,15,16,17,0,1,2,3,4,5,6,7,8,0)
DSTOREC_2V_C_FWD(10,11,x5,0,x7)
DSTOREC_2V_C_FWD(12,13,x5,0,x7)
DSTOREC_2V_C_FWD(14,15,x5,0,x7)
DSTOREC_2V_C_FWD(16,17,x5,0,x7)
BRANCH(END_WRITE_MEM)
//
// Generic-strided C-storage.
LABEL(WRITE_MEM_G)
// TODO: Implement.
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
: "x0","x1","x2","x3","x4","x5","x6","x7","x8",
  "x9","x16",
  "v0","v1","v2","v3","v4","v5","v6","v7",
  "v8","v9","v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19",
  "v20","v21","v22","v23",
  "v24","v25","v26","v27",
  "v28","v29","v30","v31"
  );

}
