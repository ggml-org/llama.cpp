/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

static const int32_t mask[8][8] = { {0, 0, 0, 0, 0, 0, 0, 0},
                                    {-1, 0, 0, 0, 0, 0, 0, 0},
                                    {-1, -1, 0, 0, 0, 0, 0, 0},
                                    {-1, -1, -1, 0, 0, 0, 0, 0},
                                    {-1, -1, -1, -1, 0, 0, 0, 0},
                                    {-1, -1, -1, -1, -1, 0, 0, 0},
                                    {-1, -1, -1, -1, -1, -1, 0, 0},
                                    {-1, -1, -1, -1, -1, -1, -1, 0},
                                  };

void bli_sgemmsup_rv_zen_asm_5x8_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------
    begin_asm()

    vzeroall()                         // zero all xmm/ymm registers.
    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)            //load mask elements

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)


    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12,        4*4)) // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1,4*4)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2,4*4)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx,        4*4)) // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1,4*4)) // prefetch c + 4*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c
    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm7)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm12, ymm12)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm7)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm4)
    vmaskmovps(ymm4, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm6)
    vmaskmovps(ymm6, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm8)
    vmaskmovps(ymm8, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm10)
    vmaskmovps(ymm10, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm12)
    vmaskmovps(ymm12, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(ymm4, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm6, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm8, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm10, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm12, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r13", "r14",
      "xmm0", "xmm7",
      "ymm0", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8", "ymm10", "ymm12",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_4x8_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------

    begin_asm()

    vzeroall()                         // zero all xmm/ymm registers.
    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)           //load mask elements

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)


    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12,        4*4)) // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1,4*4)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2,4*4)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx,        4*4)) // prefetch c + 3*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm7)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm7)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm4)
    vmaskmovps(ymm4, ymm3, mem(rcx, 0*32))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm6)
    vmaskmovps(ymm6, ymm3, mem(rcx, 0*32))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm8)
    vmaskmovps(ymm8, ymm3, mem(rcx, 0*32))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm10)
    vmaskmovps(ymm10, ymm3, mem(rcx, 0*32))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(ymm4, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm6, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm8, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm10, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SDONE)

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r13", "r14",
      "xmm0", "xmm7",
      "ymm0", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8", "ymm10",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_3x8_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------

    begin_asm()

    vzeroall()                         // zero all xmm/ymm registers.
    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)            //load mask elements

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)

    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(r12,        4*4)) // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1,4*4)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2,4*4)) // prefetch c + 2*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.


    label(.SLOOPKITER)                 // MAIN LOOP


    // ---------------------------------- iteration 0
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm7)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm7)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm4)
    vmaskmovps(ymm4, ymm3, mem(rcx, 0*32))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm6)
    vmaskmovps(ymm6, ymm3, mem(rcx, 0*32))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm8)
    vmaskmovps(ymm8, ymm3, mem(rcx, 0*32))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(ymm4, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm6, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm8, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SDONE)

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14",
      "xmm0", "xmm7",
      "ymm0", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm7", "ymm8",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_2x8_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------

    begin_asm()

    vzeroall()                         // zero all xmm/ymm registers.
    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)           //load mask elements

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)


    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c


    prefetch(0, mem(r12,        4*4)) // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1,4*4)) // prefetch c + 1*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm7)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm7)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm4)
    vmaskmovps(ymm4, ymm3, mem(rcx, 0*32))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm6)
    vmaskmovps(ymm6, ymm3, mem(rcx, 0*32))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(ymm4, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm6, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SDONE)

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14",
      "xmm0", "xmm7",
      "ymm0", "ymm2", "ymm3", "ymm4", "ymm6", "ymm7",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_1x8_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------

    begin_asm()

    vzeroall()                         // zero all xmm/ymm registers.
    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)       //load

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)


    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(r12,        4*4)) // prefetch c + 0*rs

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0

    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1

    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)

    add(r9, rax)                       // a += cs_a;


    // ---------------------------------- iteration 2
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)

    add(r9, rax)                       // a += cs_a;


    // ---------------------------------- iteration 3
    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmaskmovps(mem(rbx, 0*32), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm7)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm7)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vmaskmovps(mem(rcx, 0*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm7, ymm4)
    vmaskmovps(ymm4, ymm3, mem(rcx, 0*32))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support*/

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(ymm4, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter]     "m"    (k_iter),
      [k_left]     "m"    (k_left),
      [a]          "m"    (a),
      [rs_a]       "m"    (rs_a),
      [cs_a]       "m"    (cs_a),
      [ps_a4]      "m"    (ps_a4),
      [b]          "m"    (b),
      [rs_b]       "m"    (rs_b),
      [cs_b]       "m"    (cs_b),
      [alpha]      "m"    (alpha),
      [beta]       "m"    (beta),
      [c]          "m"    (c),
      [rs_c]       "m"    (rs_c),
      [cs_c]       "m"    (cs_c),
      [mask_vec]   "m"    (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14",
      "xmm0", "xmm7",
      "ymm0", "ymm2", "ymm3", "ymm4", "ymm7",
      "memory"
    )
}

