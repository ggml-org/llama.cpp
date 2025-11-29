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

// Supplimentary fixed-size gemmsup.

#include "blis.h"
#include "assert.h"

// Label locality & misc.
#include "../../armv8a_asm_utils.h"

#define DGEMM_3X1X2_NKER_SUBLOOP(C0,C1,C2,A0,A1,A2,B) \
" fmla v"#C0".2d, v"#A0".2d, v"#B".2d      \n\t" \
" fmla v"#C1".2d, v"#A1".2d, v"#B".2d      \n\t" \
" fmla v"#C2".2d, v"#A2".2d, v"#B".2d      \n\t"

#define DGEMM_3X4X2_K_MKER_LOOP_PLAIN(C00,C01,C02,C03,C10,C11,C12,C13,C20,C21,C22,C23,A0,A1,A2,B0,B1,B2,B3) \
  DGEMM_3X1X2_NKER_SUBLOOP(C00,C10,C20,A0,A1,A2,B0) \
  DGEMM_3X1X2_NKER_SUBLOOP(C01,C11,C21,A0,A1,A2,B1) \
  DGEMM_3X1X2_NKER_SUBLOOP(C02,C12,C22,A0,A1,A2,B2) \
  DGEMM_3X1X2_NKER_SUBLOOP(C03,C13,C23,A0,A1,A2,B3)

// For row-storage of C.
#define DLOADC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
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

#define DSCALE6V(V0,V1,V2,V3,V4,V5,A,IDX) \
  DSCALE4V(V0,V1,V2,V3,A,IDX) \
  DSCALE2V(V4,V5,A,IDX)
#define DSCALEA6V(D0,D1,D2,D3,D4,D5,S0,S1,S2,S3,S4,S5,A,IDX) \
  DSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  DSCALEA2V(D4,D5,S4,S5,A,IDX)

void bli_dgemmsup_rd_armv8a_asm_3x4
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
  assert( m0 == 3 );
  assert( n0 == 4 );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  assert( cs_a0 == 1 );
  assert( rs_b0 == 1 );

  __asm__ volatile
  (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" ldr             x2, %[rs_a]                     \n\t" // Row-skip of A.
" ldr             x3, %[cs_b]                     \n\t" // Column-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(double).
" lsl             x2, x2, #3                      \n\t" // rs_a
" lsl             x3, x3, #3                      \n\t" // cs_b
" lsl             x6, x6, #3                      \n\t" // rs_c
" lsl             x7, x7, #3                      \n\t" // cs_c
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t"
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:11] <- C
//  V[12:14] <- A
//  V[16:19] <- B
// Under this scheme, the following is defined:
#define DGEMM_3X4X2_K_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3) \
  DGEMM_3X4X2_K_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,A0,A1,A2,B0,B1,B2,B3)
// Load from memory.
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" mov             x11, x1                         \n\t" // Load B.
" ldr             q16, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
" ldr             q17, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
" ldr             q18, [x11]                      \n\t"
" add             x11, x11, x3                    \n\t"
" ldr             q19, [x11]                      \n\t"
" add             x1, x1, #16                     \n\t"
"                                                 \n\t"
" mov             x14, x0                         \n\t" // Load A.
" ldr             q12, [x14]                      \n\t"
" add             x14, x14, x2                    \n\t"
" ldr             q13, [x14]                      \n\t"
" add             x14, x14, x2                    \n\t"
" ldr             q14, [x14]                      \n\t"
" add             x0, x0, #16                     \n\t"
LABEL(CLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR4V(8,9,10,11)
// No-microkernel early return, once again.
BEQ(K_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_3X4X2_K_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,A2,B0,B1,B2,B3) \
  DGEMM_3X4X2_K_MKER_LOOP_PLAIN_LOC(A0,A1,A2,B0,B1,B2,B3) \
 "mov             x11, x1                         \n\t" \
 "ldr             q"#B0", [x11]                   \n\t" \
 "add             x11, x11, x3                    \n\t" \
 "ldr             q"#B1", [x11]                   \n\t" \
 "add             x11, x11, x3                    \n\t" \
 "ldr             q"#B2", [x11]                   \n\t" \
 "add             x11, x11, x3                    \n\t" \
 "ldr             q"#B3", [x11]                   \n\t" \
 "add             x1, x1, #16                     \n\t" \
 "mov             x14, x0                         \n\t" \
 "ldr             q"#A0", [x14]                   \n\t" \
 "add             x14, x14, x2                    \n\t" \
 "ldr             q"#A1", [x14]                   \n\t" \
 "add             x14, x14, x2                    \n\t" \
 "ldr             q"#A2", [x14]                   \n\t" \
 "add             x0, x0, #16                     \n\t"
// Start microkernel loop.
LABEL(K_MKER_LOOP)
DGEMM_3X4X2_K_MKER_LOOP_PLAIN_LOC_FWD(12,13,14,16,17,18,19)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(FIN_MKER_LOOP)
DGEMM_3X4X2_K_MKER_LOOP_PLAIN_LOC_FWD(12,13,14,16,17,18,19)
BRANCH(K_MKER_LOOP)
//
// Final microkernel loop.
LABEL(FIN_MKER_LOOP)
DGEMM_3X4X2_K_MKER_LOOP_PLAIN_LOC(12,13,14,16,17,18,19)
//
// If major kernel is executed,
//  an additional depth-summation is required.
" faddp           v0.2d, v0.2d, v1.2d             \n\t" // Line 0.
" faddp           v1.2d, v2.2d, v3.2d             \n\t"
" faddp           v2.2d, v4.2d, v5.2d             \n\t" // Line 1.
" faddp           v3.2d, v6.2d, v7.2d             \n\t"
" faddp           v4.2d, v8.2d, v9.2d             \n\t" // Line 2.
" faddp           v5.2d, v10.2d, v11.2d           \n\t"
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
" add             x1, x1, #8                      \n\t"
" mov             x14, x0                         \n\t" // Load A column.
" ld1             {v24.d}[0], [x14], x2           \n\t"
" ld1             {v24.d}[1], [x14], x2           \n\t"
" ld1             {v25.d}[0], [x14], x2           \n\t"
" add             x0, x0, #8                      \n\t"
" fmla            v0.2d, v28.2d, v24.d[0]         \n\t"
" fmla            v1.2d, v29.2d, v24.d[0]         \n\t"
" fmla            v2.2d, v28.2d, v24.d[1]         \n\t"
" fmla            v3.2d, v29.2d, v24.d[1]         \n\t"
" fmla            v4.2d, v28.2d, v25.d[0]         \n\t"
" fmla            v5.2d, v29.2d, v25.d[0]         \n\t"
" sub             x8, x8, #1                      \n\t"
BRANCH(K_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(WRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v30.2d}, [x4]                  \n\t" // Load alpha & beta (value).
" ld1r            {v31.2d}, [x8]                  \n\t"
DSCALE6V(0,1,2,3,4,5,30,0)
"                                                 \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x7, #8                          \n\t" // Check for column-storage.
BNE(WRITE_MEM_C)
//
// C storage in rows.
LABEL(WRITE_MEM_R)
" fcmp            d31, #0.0                       \n\t"
BEQ(ZERO_BETA_R)
DLOADC_2V_R_FWD(12,13,x9,0,x6)
DLOADC_2V_R_FWD(14,15,x9,0,x6)
DLOADC_2V_R_FWD(16,17,x9,0,x6)
DSCALEA6V(0,1,2,3,4,5,12,13,14,15,16,17,31,0)
LABEL(ZERO_BETA_R)
DSTOREC_2V_R_FWD(0,1,x5,0,x6)
DSTOREC_2V_R_FWD(2,3,x5,0,x6)
DSTOREC_2V_R_FWD(4,5,x5,0,x6)
BRANCH(END_WRITE_MEM)
//
// C storage in columns.
LABEL(WRITE_MEM_C)
" trn1            v6.2d, v0.2d, v2.2d            \n\t"
" trn2            v7.2d, v0.2d, v2.2d            \n\t"
" trn1            v8.2d, v1.2d, v3.2d            \n\t"
" trn2            v9.2d, v1.2d, v3.2d            \n\t"
" fcmp            d31, #0.0                       \n\t"
BEQ(ZERO_BETA_C)
DLOADC_1V_1ELM_C_FWD(12,20,0,x9,0,x7)
DLOADC_1V_1ELM_C_FWD(13,20,1,x9,0,x7)
DLOADC_1V_1ELM_C_FWD(14,21,0,x9,0,x7)
DLOADC_1V_1ELM_C_FWD(15,21,1,x9,0,x7)
DSCALEA6V(6,7,8,9,4,5,12,13,14,15,20,21,31,0)
LABEL(ZERO_BETA_C)
DSTOREC_1V_1ELM_C_FWD(6,4,0,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(7,4,1,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(8,5,0,x5,0,x7)
DSTOREC_1V_1ELM_C_FWD(9,5,1,x5,0,x7)
//
// End of this microkernel.
LABEL(END_WRITE_MEM)
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

}

