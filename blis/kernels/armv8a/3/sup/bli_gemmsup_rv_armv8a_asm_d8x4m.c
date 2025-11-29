/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

/*
 * +---+ +---+
 * | 0 | | 4 |
 * +---+ +---+
 * +---+ +---+
 * | 1 | | 5 |
 * +---+ +---+
 * +---+ +---+
 * | 2 | | 6 |
 * +---+ +---+
 * +---+ +---+
 * | 3 | | 7 |
 * +---+ +---+
 *
 */
#define DGEMM_8X4_MKER_LOOP_PLAIN(C00,C10,C20,C30,C01,C11,C21,C31,C02,C12,C22,C32,C03,C13,C23,C33,A0,A1,A2,A3,B0,B1,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_2X2_NANOKERNEL(C00,C01,A0,B0) \
  DGEMM_2X2_NANOKERNEL(C10,C11,A1,B0) \
  DGEMM_2X2_NANOKERNEL(C20,C21,A2,B0) \
  DGEMM_2X2_NANOKERNEL(C30,C31,A3,B0) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  DGEMM_2X2_NANOKERNEL(C02,C03,A0,B1) \
  DGEMM_2X2_NANOKERNEL(C12,C13,A1,B1) \
  DGEMM_2X2_NANOKERNEL(C22,C23,A2,B1) \
  DGEMM_2X2_NANOKERNEL(C32,C33,A3,B1)

// Interleaving load or not.
#define DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define DGEMM_LOAD1V_load(V1,ADDR,IMM) \
" ldr  q"#V1", ["#ADDR", #"#IMM"] \n\t"

#define DLOADC_4V_C_FWD(C0,C1,C2,C3,CADDR,CSHIFT,LDC) \
  DLOAD4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#LDC" \n\t"
#define DSTOREC_4V_C_FWD(C0,C1,C2,C3,CADDR,CSHIFT,LDC) \
  DSTORE4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#LDC" \n\t"

#define DLOADC_4V_R_FWD(C00,C01,C10,C11,CADDR,CSHIFT,RSC) \
  DLOAD2V(C00,C01,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t" \
  DLOAD2V(C10,C11,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_4V_R_FWD(C00,C01,C10,C11,CADDR,CSHIFT,RSC) \
  DSTORE2V(C00,C01,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t" \
  DSTORE2V(C10,C11,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

/*
 * 8x4 kernel for dgemmsup.
 *
 * R-dimension too short.
 * Not recommanded for use.
 */
void bli_dgemmsup_rv_armv8a_asm_8x4m
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
  // Fixme: This uker has no dispatching for unalighed sizes.
  // Currently it only serves as a dispatch target for other kernels
  //  and cannot be registered in configurations.
  assert( n0 == 4 );

  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );

  uint64_t ps_a   = bli_auxinfo_ps_a( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 6;
  uint64_t k_left = k0 % 6;

  int64_t  m_iter = m0 / 8;
  int64_t  m_left = m0 % 8;

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
" ldr             x2, %[cs_a]                     \n\t" // Column-skip of A.
" ldr             x9, %[rs_a]                     \n\t" // Row-skip of A.
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
LABEL(MILLIKER_MLOOP)
"                                                 \n\t"
" mov             x0, x10                         \n\t" // Parameters to be reloaded
" mov             x5, x13                         \n\t" //  within each millikernel loop.
" ldr             x1, %[b]                        \n\t"
" ldr             x4, %[k_mker]                   \n\t"
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:15] <- C
//  V[16:19] <- B; Allowed latency: 24 cycles / # of FPUs.
//  V[20:31] <- A; Allowed latency: 32 cycles / # of FPUs.
// Under this scheme, the following is defined:
#define DGEMM_8X4_MKER_LOOP_PLAIN_LOC(A0,A1,A2,A3,B0,B1,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_8X4_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,A0,A1,A2,A3,B0,B1,BADDR,BSHIFT,LOADNEXT)
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v20.d}[0], [x14], x9           \n\t"
" ld1             {v20.d}[1], [x14], x9           \n\t"
" ld1             {v21.d}[0], [x14], x9           \n\t"
" ld1             {v21.d}[1], [x14], x9           \n\t"
" ld1             {v22.d}[0], [x14], x9           \n\t"
" ld1             {v22.d}[1], [x14], x9           \n\t"
" ld1             {v23.d}[0], [x14], x9           \n\t"
" ld1             {v23.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v24.d}[0], [x14], x9           \n\t"
" ld1             {v24.d}[1], [x14], x9           \n\t"
" ld1             {v25.d}[0], [x14], x9           \n\t"
" ld1             {v25.d}[1], [x14], x9           \n\t"
" ld1             {v26.d}[0], [x14], x9           \n\t"
" ld1             {v26.d}[1], [x14], x9           \n\t"
" ld1             {v27.d}[0], [x14], x9           \n\t"
" ld1             {v27.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v28.d}[0], [x14], x9           \n\t"
" ld1             {v28.d}[1], [x14], x9           \n\t"
" ld1             {v29.d}[0], [x14], x9           \n\t"
" ld1             {v29.d}[1], [x14], x9           \n\t"
" ld1             {v30.d}[0], [x14], x9           \n\t"
" ld1             {v30.d}[1], [x14], x9           \n\t"
" ld1             {v31.d}[0], [x14], x9           \n\t"
" ld1             {v31.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
"                                                 \n\t"
" ldr             q16, [x1, #16*0]                \n\t"
" ldr             q17, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q18, [x1, #16*0]                \n\t"
" ldr             q19, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
LABEL(CLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
// No-microkernel early return, once again.
BEQ(K_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,A2,A3,B0,B1) \
  DGEMM_8X4_MKER_LOOP_PLAIN_LOC(A0,A1,A2,A3,B0,B1,x1,0,load) \
 "mov             x14, x0                         \n\t" \
 "ld1             {v"#A0".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A0".d}[1], [x14], x9        \n\t" \
 "ld1             {v"#A1".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A1".d}[1], [x14], x9        \n\t" \
 "ld1             {v"#A2".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A2".d}[1], [x14], x9        \n\t" \
 "ld1             {v"#A3".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A3".d}[1], [x14], x9        \n\t" \
 "ldr             q"#B1", [x1, #16*1]             \n\t" \
 "add             x1, x1, x3                      \n\t" \
 "add             x0, x0, x2                      \n\t"
// Start microkernel loop.
LABEL(K_MKER_LOOP)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(20,21,22,23,16,17)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,27,18,19)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(28,29,30,31,16,17)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(FIN_MKER_LOOP)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(20,21,22,23,18,19)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(24,25,26,27,16,17)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC_FWD(28,29,30,31,18,19)
BRANCH(K_MKER_LOOP)
//
// Final microkernel loop.
LABEL(FIN_MKER_LOOP)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC(20,21,22,23,18,19,x1,0,load)
" ldr             q19, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
DGEMM_8X4_MKER_LOOP_PLAIN_LOC(24,25,26,27,16,17,xzr,-1,noload)
DGEMM_8X4_MKER_LOOP_PLAIN_LOC(28,29,30,31,18,19,xzr,-1,noload)
//
// Loops left behind microkernels.
LABEL(K_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(WRITE_MEM_PREP)
" mov             x14, x0                         \n\t"
" ld1             {v20.d}[0], [x14], x9           \n\t" // Load A col.
" ld1             {v20.d}[1], [x14], x9           \n\t"
" ld1             {v21.d}[0], [x14], x9           \n\t"
" ld1             {v21.d}[1], [x14], x9           \n\t"
" ld1             {v22.d}[0], [x14], x9           \n\t"
" ld1             {v22.d}[1], [x14], x9           \n\t"
" ld1             {v23.d}[0], [x14], x9           \n\t"
" ld1             {v23.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q16, [x1, #16*0]                \n\t" // Load B col.
" ldr             q17, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
DGEMM_8X4_MKER_LOOP_PLAIN_LOC(20,21,22,23,16,17,xzr,-1,noload)
BRANCH(K_LEFT_LOOP)
//
// Scale and write to memory.
LABEL(WRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v16.2d}, [x4]                  \n\t" // Load alpha & beta (value).
" ld1r            {v17.2d}, [x8]                  \n\t"
" fmov            d18, #1.0                       \n\t"
" fcmp            d16, d18                        \n\t"
BEQ(UNIT_ALPHA)
DSCALE8V(0,1,2,3,4,5,6,7,16,0)
DSCALE8V(8,9,10,11,12,13,14,15,16,0)
LABEL(UNIT_ALPHA)
"                                                 \n\t"
" mov             x1, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
" cmp             x6, #8                          \n\t" // Check for row-storage.
BNE(WRITE_MEM_R)
//
// C storage in columns.
LABEL(WRITE_MEM_C)
" fcmp            d17, #0.0                       \n\t"
BEQ(ZERO_BETA_C)
DLOADC_4V_C_FWD(20,21,22,23,x1,0,x7)
DLOADC_4V_C_FWD(24,25,26,27,x1,0,x7)
DSCALEA8V(0,1,2,3,4,5,6,7,20,21,22,23,24,25,26,27,17,0)
//
DLOADC_4V_C_FWD(20,21,22,23,x1,0,x7)
DLOADC_4V_C_FWD(24,25,26,27,x1,0,x7)
DSCALEA8V(8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,17,0)
LABEL(ZERO_BETA_C)
//
DSTOREC_4V_C_FWD(0,1,2,3,x5,0,x7)
DSTOREC_4V_C_FWD(4,5,6,7,x5,0,x7)
DSTOREC_4V_C_FWD(8,9,10,11,x5,0,x7)
DSTOREC_4V_C_FWD(12,13,14,15,x5,0,x7)
BRANCH(END_WRITE_MEM)
//
// C storage in rows.
LABEL(WRITE_MEM_R)
// In-register transpose.
" trn1            v16.2d, v0.2d, v4.2d            \n\t" // Row 0.
" trn1            v17.2d, v8.2d, v12.2d           \n\t"
" trn2            v18.2d, v0.2d, v4.2d            \n\t" // Row 1.
" trn2            v19.2d, v8.2d, v12.2d           \n\t"
" trn1            v20.2d, v1.2d, v5.2d            \n\t" // Row 2.
" trn1            v21.2d, v9.2d, v13.2d           \n\t"
" trn2            v22.2d, v1.2d, v5.2d            \n\t" // Row 3.
" trn2            v23.2d, v9.2d, v13.2d           \n\t"
" trn1            v24.2d, v2.2d, v6.2d            \n\t" // Row 4.
" trn1            v25.2d, v10.2d, v14.2d          \n\t"
" trn2            v26.2d, v2.2d, v6.2d            \n\t" // Row 5.
" trn2            v27.2d, v10.2d, v14.2d          \n\t"
" trn1            v28.2d, v3.2d, v7.2d            \n\t" // Row 6.
" trn1            v29.2d, v11.2d, v15.2d          \n\t"
" trn2            v30.2d, v3.2d, v7.2d            \n\t" // Row 7.
" trn2            v31.2d, v11.2d, v15.2d          \n\t"
// " ld1r            {v14.2d}, [x4]                  \n\t" // Reload alpha & beta (value).
" ld1r            {v15.2d}, [x8]                  \n\t"
" fcmp            d15, #0.0                       \n\t"
BEQ(ZERO_BETA_R)
DLOADC_4V_R_FWD(0,1,2,3,x1,0,x6)
DLOADC_4V_R_FWD(4,5,6,7,x1,0,x6)
DSCALEA8V(16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7,15,0)
//
DLOADC_4V_R_FWD(0,1,2,3,x1,0,x6)
DLOADC_4V_R_FWD(4,5,6,7,x1,0,x6)
DSCALEA8V(24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,15,0)
LABEL(ZERO_BETA_R)
//
DSTOREC_4V_R_FWD(16,17,18,19,x5,0,x6)
DSTOREC_4V_R_FWD(20,21,22,23,x5,0,x6)
DSTOREC_4V_R_FWD(24,25,26,27,x5,0,x6)
DSTOREC_4V_R_FWD(28,29,30,31,x5,0,x6)
//
// End of this microkernel.
LABEL(END_WRITE_MEM)
"                                                 \n\t"
" subs            x12, x12, #1                    \n\t"
BEQ(END_EXEC)
"                                                 \n\t"
" mov             x8, #8                          \n\t"
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
  a = ( double* )a + m_iter * ps_a;
  c = ( double* )c + m_iter * 8 * rs_c;
  // Edge case is within 1 millikernel loop of THIS kernel.
  // Regarding the 6x?m kernel, the panel stride should be always local.
  auxinfo_t data_6xkm = *data;
  bli_auxinfo_set_ps_a( 6 * rs_a, &data_6xkm );
  if ( m_left )
  {
    bli_dgemmsup_rv_armv8a_int_6x4mn
    (
      conja, conjb, m_left, 4, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, &data_6xkm, cntx
    );
  }

  // Issue prefetch instructions only after
  //  execution is done.
  __asm__
  (
" mov   x0, %[a_next] \n\t"
" mov   x1, %[b_next] \n\t"
" prfm  PLDL1STRM, [x0, #16*0] \n\t"
" prfm  PLDL1STRM, [x0, #16*1] \n\t"
" prfm  PLDL1STRM, [x0, #16*2] \n\t"
" prfm  PLDL1KEEP, [x1, #16*0] \n\t"
" prfm  PLDL1KEEP, [x1, #16*1] \n\t"
" prfm  PLDL1KEEP, [x1, #16*2] \n\t"
:
: [a_next] "r" (a_next),
  [b_next] "r" (b_next)
: "x0", "x1"
  );
}

