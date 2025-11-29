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

void bli_sgemmsup_rv_zen_asm_5x4_mask
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
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    const int32_t *mask_vec = mask[n0];

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), xmm7)            //load mask elements

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

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)
    vxorps(xmm8,  xmm8,  xmm8)
    vxorps(xmm10, xmm10, xmm10)
    vxorps(xmm12, xmm12, xmm12)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 0))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 0)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 0)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 0))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 0)) // prefetch c + 4*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)


    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    vmulps(xmm0, xmm12, xmm12)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmaskmovps(xmm8, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm10)
    vmaskmovps(xmm10, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm12)
    vmaskmovps(xmm12, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm8, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm10, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm12, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]   "m"   (m_iter),
      [k_iter]   "m"   (k_iter),
      [k_left]   "m"   (k_left),
      [a]        "m"   (a),
      [rs_a]     "m"   (rs_a),
      [cs_a]     "m"   (cs_a),
      [ps_a4]    "m"   (ps_a4),
      [b]        "m"   (b),
      [rs_b]     "m"   (rs_b),
      [cs_b]     "m"   (cs_b),
      [alpha]    "m"   (alpha),
      [beta]     "m"   (beta),
      [c]        "m"   (c),
      [rs_c]     "m"   (rs_c),
      [cs_c]     "m"   (cs_c),
      [mask_vec] "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm6", "xmm7",
      "xmm8", "xmm10", "xmm12",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_4x4_mask
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
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    const int32_t *mask_vec = mask[n0];

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), xmm7)            //load mask elements

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

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)
    vxorps(xmm8,  xmm8,  xmm8)
    vxorps(xmm10, xmm10, xmm10)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 0))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 0)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 0)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 0))         // prefetch c + 3*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmaskmovps(xmm8, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm10)
    vmaskmovps(xmm10, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm8, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm10, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]   "m"  (m_iter),
      [k_iter]   "m"  (k_iter),
      [k_left]   "m"  (k_left),
      [a]        "m"  (a),
      [rs_a]     "m"  (rs_a),
      [cs_a]     "m"  (cs_a),
      [ps_a4]    "m"  (ps_a4),
      [b]        "m"  (b),
      [rs_b]     "m"  (rs_b),
      [cs_b]     "m"  (cs_b),
      [alpha]    "m"  (alpha),
      [beta]     "m"  (beta),
      [c]        "m"  (c),
      [rs_c]     "m"  (rs_c),
      [cs_c]     "m"  (cs_c),
      [mask_vec] "m"  (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm6", "xmm7",
      "xmm8", "xmm10",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_3x4_mask
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
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    const int32_t *mask_vec = mask[n0];

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), xmm7)            //load mask elements

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

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)
    vxorps(xmm8,  xmm8,  xmm8)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(r12, 0))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 0)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 0)) // prefetch c + 2*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm8)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm8)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm8)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm8)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm8)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmaskmovps(xmm8, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm8, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]   "m"   (m_iter),
      [k_iter]   "m"   (k_iter),
      [k_left]   "m"   (k_left),
      [a]        "m"   (a),
      [rs_a]     "m"   (rs_a),
      [cs_a]     "m"   (cs_a),
      [ps_a4]    "m"   (ps_a4),
      [b]        "m"   (b),
      [rs_b]     "m"   (rs_b),
      [cs_b]     "m"   (cs_b),
      [alpha]    "m"   (alpha),
      [beta]     "m"   (beta),
      [c]        "m"   (c),
      [rs_c]     "m"   (rs_c),
      [cs_c]     "m"   (cs_c),
      [mask_vec] "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r14",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm6", "xmm7",
      "xmm8", "xmm10",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_2x4_mask
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
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    const int32_t *mask_vec = mask[n0];

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), xmm7)            //load mask elements

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

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(r12, 0))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 0)) // prefetch c + 1*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmaskmovps(xmm6, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm6, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]   "m"   (m_iter),
      [k_iter]   "m"   (k_iter),
      [k_left]   "m"   (k_left),
      [a]        "m"   (a),
      [rs_a]     "m"   (rs_a),
      [cs_a]     "m"   (cs_a),
      [ps_a4]    "m"   (ps_a4),
      [b]        "m"   (b),
      [rs_b]     "m"   (rs_b),
      [cs_b]     "m"   (cs_b),
      [alpha]    "m"   (alpha),
      [beta]     "m"   (beta),
      [c]        "m"   (c),
      [rs_c]     "m"   (rs_c),
      [cs_c]     "m"   (cs_c),
      [mask_vec] "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r14",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm6", "xmm7",
      "memory"
    )
}

void bli_sgemmsup_rv_zen_asm_1x4_mask
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
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    const int32_t *mask_vec = mask[n0];

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), xmm7)            //load mask elements

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

    vxorps(xmm4,  xmm4,  xmm4)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(r12, 0))         // prefetch c + 0*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm4)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm4)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm4)

    add(r9, rax)                       // a += cs_a;

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm4)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vfmadd231ps(xmm0, xmm2, xmm4)

    add(r9, rax)                       // a += cs_a;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmaskmovps(xmm4, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(xmm4, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]   "m"  (m_iter),
      [k_iter]   "m"  (k_iter),
      [k_left]   "m"  (k_left),
      [a]        "m"  (a),
      [rs_a]     "m"  (rs_a),
      [cs_a]     "m"  (cs_a),
      [ps_a4]    "m"  (ps_a4),
      [b]        "m"  (b),
      [rs_b]     "m"  (rs_b),
      [cs_b]     "m"  (cs_b),
      [alpha]    "m"  (alpha),
      [beta]     "m"  (beta),
      [c]        "m"  (c),
      [rs_c]     "m"  (rs_c),
      [cs_c]     "m"  (cs_c),
      [mask_vec] "m"  (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r14",
      "xmm0", "xmm2", "xmm3",
      "xmm4", "xmm7",
      "memory"
    )
}
