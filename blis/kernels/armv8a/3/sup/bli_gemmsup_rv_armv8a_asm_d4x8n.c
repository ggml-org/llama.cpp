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
 * +---+ +---+ +---+ +---+
 * | 0 | | 2 | | 4 | | 6 |
 * +---+ +---+ +---+ +---+
 * +---+ +---+ +---+ +---+
 * | 1 | | 3 | | 5 | | 7 |
 * +---+ +---+ +---+ +---+
 */
#define DGEMM_4X8_MKER_LOOP_PLAIN(C00,C01,C02,C03,C10,C11,C12,C13,C20,C21,C22,C23,C30,C31,C32,C33,A0,A1,B0,B1,B2,B3,BADDR,BSHIFT0,BSHIFT1,BSHIFT2,LOADNEXT) \
  DGEMM_2X2_NANOKERNEL(C00,C10,B0,A0) \
  DGEMM_2X2_NANOKERNEL(C20,C30,B0,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT0) \
  DGEMM_2X2_NANOKERNEL(C01,C11,B1,A0) \
  DGEMM_2X2_NANOKERNEL(C21,C31,B1,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (B1,BADDR,BSHIFT1) \
  DGEMM_2X2_NANOKERNEL(C02,C12,B2,A0) \
  DGEMM_2X2_NANOKERNEL(C22,C32,B2,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (B2,BADDR,BSHIFT2) \
  DGEMM_2X2_NANOKERNEL(C03,C13,B3,A0) \
  DGEMM_2X2_NANOKERNEL(C23,C33,B3,A1)


// Interleaving load or not.
#define DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define DGEMM_LOAD1V_load(V1,ADDR,IMM) \
" ldr  q"#V1", ["#ADDR", #"#IMM"] \n\t"

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
#define DLOADC_4V_C_FWD(C00,C10,C01,C11,CADDR,CSHIFT,CSC) \
  DLOAD2V(C00,C10,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t" \
  DLOAD2V(C01,C11,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t"
#define DSTOREC_4V_C_FWD(C00,C10,C01,C11,CADDR,CSHIFT,CSC) \
  DSTORE2V(C00,C10,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t" \
  DSTORE2V(C01,C11,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t"


/*
 * 4x8 dgemmsup kernel with extending 2nd dimension.
 *
 * Recommanded usage case:
 * o 16 < (L1 cache latency) * (Num. FPU) < 25.
 * o L1 cache has a bandwidth not too low (true in most cases).
 * o (FMLA latency) * (Num. FPU) < 32 cycles (true in almost all cases).
 */
void bli_dgemmsup_rv_armv8a_asm_4x8n
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
  assert( m0 == 4 );

  // LLVM has very bad routing ability for inline asm.
  // Limit number of registers in case of Clang compilation.
#ifndef __clang__
  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );
#endif
  uint64_t ps_b   = bli_auxinfo_ps_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;

  int64_t  n_iter = n0 / 8;
  int64_t  n_left = n0 % 8;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  // uint64_t cs_b   = cs_b0;
  assert( cs_b0 == 1 );

  if ( n_iter == 0 ) goto consider_edge_cases;

  __asm__ volatile
  (
" ldr             x10, %[b]                       \n\t"
" ldr             x13, %[c]                       \n\t"
" ldr             x12, %[n_iter]                  \n\t"
" ldr             x11, %[ps_b]                    \n\t" // Panel-skip of B.
" ldr             x3, %[rs_b]                     \n\t" // Row-skip of B.
" ldr             x9, %[rs_a]                     \n\t" // Row-skip of A.
" ldr             x2, %[cs_a]                     \n\t" // Column-skip of A.
"                                                 \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C.
" ldr             x7, %[cs_c]                     \n\t" // Column-skip of C.
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(double).
" lsl             x11, x11, #3                    \n\t" // ps_b
" lsl             x9, x9, #3                      \n\t" // rs_a
" lsl             x2, x2, #3                      \n\t" // cs_a
" lsl             x3, x3, #3                      \n\t" // rs_b
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
BRANCH(C_PREFETCH_END)
LABEL(C_PREFETCH_COLS)
// This prefetch will not cover further mker perts. Skip.
//
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
// DPRFMC_FWD(x1,x7)
LABEL(C_PREFETCH_END)
//
// Millikernel.
LABEL(MILLIKER_MLOOP)
"                                                 \n\t"
" mov             x1, x10                         \n\t" // Parameters to be reloaded
" mov             x5, x13                         \n\t" //  within each millikernel loop.
" ldr             x0, %[a]                        \n\t"
" ldr             x4, %[k_mker]                   \n\t"
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:15] <- C
//  V[16:23] <- A; Allowed latency: 48 cycles / # of FPUs.
//  V[24:31] <- B; Allowed latency: 28 cycles / # of FPUs.
// Under this scheme, the following is defined:
#define DGEMM_4X8_MKER_LOOP_PLAIN_LOC(A0,A1,B0,B1,B2,B3,BADDR,BSHIFT0,BSHIFT1,BSHIFT2,LOADNEXT) \
  DGEMM_4X8_MKER_LOOP_PLAIN(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,A0,A1,B0,B1,B2,B3,BADDR,BSHIFT0,BSHIFT1,BSHIFT2,LOADNEXT)
LABEL(LOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(CLEAR_CCOLS)
"                                                 \n\t"
" ldr             q24, [x1, #16*0]                \n\t" // Load B first.
" ldr             q25, [x1, #16*1]                \n\t"
" ldr             q26, [x1, #16*2]                \n\t"
" ldr             q27, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t"
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" ldr             q31, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
"                                                 \n\t"
" mov             x14, x0                         \n\t" // Load A.
" ld1             {v16.d}[0], [x14], x9           \n\t" // We want A to be kept in L1.
" ld1             {v16.d}[1], [x14], x9           \n\t"
" ld1             {v17.d}[0], [x14], x9           \n\t"
" ld1             {v17.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v18.d}[0], [x14], x9           \n\t"
" ld1             {v18.d}[1], [x14], x9           \n\t"
" ld1             {v19.d}[0], [x14], x9           \n\t"
" ld1             {v19.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v20.d}[0], [x14], x9           \n\t"
" ld1             {v20.d}[1], [x14], x9           \n\t"
" ld1             {v21.d}[0], [x14], x9           \n\t"
" ld1             {v21.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" mov             x14, x0                         \n\t"
" ld1             {v22.d}[0], [x14], x9           \n\t"
" ld1             {v22.d}[1], [x14], x9           \n\t"
" ld1             {v23.d}[0], [x14], x9           \n\t"
" ld1             {v23.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
LABEL(CLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
// No-microkernel early return, once again.
BEQ(K_LEFT_LOOP)
//
// Microkernel is defined here as:
#define DGEMM_4X8_MKER_LOOP_PLAIN_LOC_FWD(A0,A1,B0,B1,B2,B3) \
  DGEMM_4X8_MKER_LOOP_PLAIN_LOC(A0,A1,B0,B1,B2,B3,x1,0,16*1,16*2,load) \
 "ldr             q"#B3", [x1, #16*3]             \n\t" \
 "mov             x14, x0                         \n\t" \
 "ld1             {v"#A0".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A0".d}[1], [x14], x9        \n\t" \
 "ld1             {v"#A1".d}[0], [x14], x9        \n\t" \
 "ld1             {v"#A1".d}[1], [x14], x9        \n\t" \
 "add             x0, x0, x2                      \n\t" \
 "add             x1, x1, x3                      \n\t"
// Start microkernel loop.
LABEL(K_MKER_LOOP)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(FIN_MKER_LOOP)
DGEMM_4X8_MKER_LOOP_PLAIN_LOC_FWD(16,17,24,25,26,27)
DGEMM_4X8_MKER_LOOP_PLAIN_LOC_FWD(18,19,28,29,30,31)
DGEMM_4X8_MKER_LOOP_PLAIN_LOC_FWD(20,21,24,25,26,27)
DGEMM_4X8_MKER_LOOP_PLAIN_LOC_FWD(22,23,28,29,30,31)
BRANCH(K_MKER_LOOP)
//
// Final microkernel loop.
LABEL(FIN_MKER_LOOP)
DGEMM_4X8_MKER_LOOP_PLAIN_LOC(16,17,24,25,26,27,x1,0,16*1,16*2,load)
" ldr             q27, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
DGEMM_4X8_MKER_LOOP_PLAIN_LOC(18,19,28,29,30,31,x1,0,16*1,16*2,load)
" ldr             q31, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
DGEMM_4X8_MKER_LOOP_PLAIN_LOC(20,21,24,25,26,27,xzr,-1,-1,-1,noload)
DGEMM_4X8_MKER_LOOP_PLAIN_LOC(22,23,28,29,30,31,xzr,-1,-1,-1,noload)
//
// Loops left behind microkernels.
LABEL(K_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(WRITE_MEM_PREP)
" ldr             q24, [x1, #16*0]                \n\t" // Load B row.
" ldr             q25, [x1, #16*1]                \n\t"
" ldr             q26, [x1, #16*2]                \n\t"
" ldr             q27, [x1, #16*3]                \n\t"
" add             x1, x1, x3                      \n\t"
" mov             x14, x0                         \n\t" // Load A col.
" ld1             {v16.d}[0], [x14], x9           \n\t"
" ld1             {v16.d}[1], [x14], x9           \n\t"
" ld1             {v17.d}[0], [x14], x9           \n\t"
" ld1             {v17.d}[1], [x14], x9           \n\t"
" add             x0, x0, x2                      \n\t"
" sub             x8, x8, #1                      \n\t"
DGEMM_4X8_MKER_LOOP_PLAIN_LOC(16,17,24,25,26,27,xzr,-1,-1,-1,noload)
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
" ld1r            {v16.2d}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v17.2d}, [x8]                  \n\t"
" fcmp            d17, #0.0                       \n\t"
DSCALE8V(0,1,2,3,4,5,6,7,16,0)
DSCALE8V(8,9,10,11,12,13,14,15,16,0)
BEQ(ZERO_BETA_R)
DLOADC_4V_R_FWD(20,21,22,23,x1,0,x6)
DLOADC_4V_R_FWD(24,25,26,27,x1,0,x6)
DSCALEA8V(0,1,2,3,4,5,6,7,20,21,22,23,24,25,26,27,17,0)
//
DLOADC_4V_R_FWD(20,21,22,23,x1,0,x6)
DLOADC_4V_R_FWD(24,25,26,27,x1,0,x6)
DSCALEA8V(8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,17,0)
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
//
DSTOREC_4V_R_FWD(0,1,2,3,x5,0,x6)
DSTOREC_4V_R_FWD(4,5,6,7,x5,0,x6)
DSTOREC_4V_R_FWD(8,9,10,11,x5,0,x6)
DSTOREC_4V_R_FWD(12,13,14,15,x5,0,x6)
BRANCH(END_WRITE_MEM)
//
// C storage in columns.
LABEL(WRITE_MEM_C)
// In-register transpose.
" trn1            v16.2d, v0.2d, v4.2d            \n\t" // Column 0.
" trn1            v17.2d, v8.2d, v12.2d           \n\t"
" trn2            v18.2d, v0.2d, v4.2d            \n\t" // Column 1.
" trn2            v19.2d, v8.2d, v12.2d           \n\t"
" trn1            v20.2d, v1.2d, v5.2d            \n\t" // Column 2.
" trn1            v21.2d, v9.2d, v13.2d           \n\t"
" trn2            v22.2d, v1.2d, v5.2d            \n\t" // Column 3.
" trn2            v23.2d, v9.2d, v13.2d           \n\t"
" trn1            v24.2d, v2.2d, v6.2d            \n\t" // Column 4.
" trn1            v25.2d, v10.2d, v14.2d          \n\t"
" trn2            v26.2d, v2.2d, v6.2d            \n\t" // Column 5.
" trn2            v27.2d, v10.2d, v14.2d          \n\t"
" trn1            v28.2d, v3.2d, v7.2d            \n\t" // Column 6.
" trn1            v29.2d, v11.2d, v15.2d          \n\t"
" trn2            v30.2d, v3.2d, v7.2d            \n\t" // Column 7.
" trn2            v31.2d, v11.2d, v15.2d          \n\t"
" ld1r            {v14.2d}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v15.2d}, [x8]                  \n\t"
DSCALE8V(16,17,18,19,20,21,22,23,14,0)
DSCALE8V(24,25,26,27,28,29,30,31,14,0)
DLOADC_4V_C_FWD(0,1,2,3,x1,0,x7)
DLOADC_4V_C_FWD(4,5,6,7,x1,0,x7)
DSCALEA8V(16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7,15,0)
//
DLOADC_4V_C_FWD(0,1,2,3,x1,0,x7)
DLOADC_4V_C_FWD(4,5,6,7,x1,0,x7)
DSCALEA8V(24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,15,0)
#ifndef __clang__
" cmp   x12, #1                       \n\t"
BRANCH(PRFM_END_C)
" prfm  PLDL1KEEP, [%[a_next], #16*0] \n\t"
" prfm  PLDL1KEEP, [%[a_next], #16*1] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*0] \n\t"
" prfm  PLDL1STRM, [%[b_next], #16*1] \n\t"
LABEL(PRFM_END_C)
#endif
//
DSTOREC_4V_C_FWD(16,17,18,19,x5,0,x7)
DSTOREC_4V_C_FWD(20,21,22,23,x5,0,x7)
DSTOREC_4V_C_FWD(24,25,26,27,x5,0,x7)
DSTOREC_4V_C_FWD(28,29,30,31,x5,0,x7)
//
// End of this microkernel.
LABEL(END_WRITE_MEM)
"                                                 \n\t"
" subs            x12, x12, #1                    \n\t"
BEQ(END_EXEC)
"                                                 \n\t"
" mov             x8, #8                          \n\t"
" madd            x13, x7, x8, x13                \n\t" // Forward C's base address to the next logic panel.
" add             x10, x10, x11                   \n\t" // Forward B's base address to the next logic panel.
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
  [ps_b]   "m" (ps_b),
  [rs_b]   "m" (rs_b),
  [rs_c]   "m" (rs_c),
  [cs_c]   "m" (cs_c),
  // In Clang, even "m"-passed parameter takes 1 register.
  // Have to disable prefetching to pass compilation.
#ifndef __clang__
  [a_next] "r" (a_next),
  [b_next] "r" (b_next),
#endif
  [n_iter] "m" (n_iter),
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
  b = ( double* )b + n_iter * ps_b;
  c = ( double* )c + n_iter * 8 * cs_c;
  if ( n_left )
  {
    auxinfo_t data_d6x4mn = *data;
    bli_auxinfo_set_ps_b( 4 * cs_b0, &data_d6x4mn );

    bli_dgemmsup_rv_armv8a_int_6x4mn
    (
      conja, conjb, 4, n_left, k0,
      alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
      beta, c, rs_c0, cs_c0, &data_d6x4mn, cntx
    );
  }

}

