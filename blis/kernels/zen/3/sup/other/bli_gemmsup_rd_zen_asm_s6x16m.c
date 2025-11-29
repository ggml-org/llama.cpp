/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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

/*
   rrc:
     --------        ------        | | | | | | | |
     --------        ------        | | | | | | | |
     --------   +=   ------ ...    | | | | | | | |
     --------        ------        | | | | | | | |
     --------        ------              :
     --------        ------              :

   Assumptions:
   - C is row-stored and B is column-stored;
   - A is row-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential microkernel is well-suited for
   a dot-product-based accumulation that performs vector loads from
   both A and B.

   NOTE: These kernels implicitly support column-oriented IO, implemented
   via an a high-level transposition of the entire operation. A and B will
   effectively remain row- and column-stored, respectively, but C will then
   effectively appear column-stored. Thus, this kernel may be used for both
   rrc and crc cases.
*/

// Prototype reference microkernels.

void bli_sgemmsup_rd_zen_asm_6x16m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

    uint64_t n_left = n0 % 16;

    // First check whether this is a edge case in the n dimension. If so,
    // dispatch other 6x?m kernels, as needed.
    if ( n_left )
    {
        float* restrict cij = c;
        float* restrict bj  = b;
        float* restrict ai  = a;

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            bli_sgemmsup_rd_zen_asm_6x8m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }
        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;

            bli_sgemmsup_rd_zen_asm_6x4m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }
        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rd_zen_asm_6x2m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }
        if ( 1 == n_left )
        {
            bli_sgemv_ex
            (
              BLIS_NO_TRANSPOSE, conjb, m0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
            );
        }
        return;
    }

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------

    begin_asm()

    mov(var(rs_a), r8)                 // load rs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)

    mov(var(b), rdx)                   // load address of b.
    mov(var(cs_b), r11)                // load cs_b
    lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

    lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
    lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

    // r12 = rcx = c
    // r14 = rax = a
    // rdx = rbx = b
    // r9  = m dim index ii
    // r15 = n dim index jj

    mov(imm(0), r15)                   // jj = 0;

    label(.SLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



    mov(var(a), r14)                   // load address of a
    mov(var(c), r12)                   // load address of c
    mov(var(b), rdx)

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(imm(1*4), rsi)                // rsi *= cs_c = 1*8
    lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(r11, rsi)                     // rsi *= cs_b;
    lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;



    mov(var(m_iter), r9)               // ii = m_iter;

    label(.SLOOP3X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


                                       // zen2 can execute 4 vxorpd ipc with
                                       // a latency of 1 cycle

    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm5,  ymm5,  ymm5)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm7,  ymm7,  ymm7)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm9,  ymm9,  ymm9)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm11, ymm11, ymm11)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm13, ymm13, ymm13)
    vxorps(ymm14, ymm14, ymm14)
    vxorps(ymm15, ymm15, ymm15)

    lea(mem(r12), rcx)                 // rcx = c_iijj;
    lea(mem(r14), rax)                 // rax = a_ii;
    lea(mem(rdx), rbx)                 // rbx = b_jj;

    lea(mem(r8,  r8,  4), rdi)         // rdi = 5*rs_a

    mov(var(k_iter32), rsi)            // i = k_iter32;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKITER8)                 // if i == 0, jump to code that
                                       // contains the k_iter8 loop.

    label(.SLOOPKITER32)               // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    // ---------------------------------- iteration 1
    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)


    // ---------------------------------- iteration 3
    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER32)                 // iterate again if i != 0.

    label(.SCONSIDKITER8)

    mov(var(k_iter8), rsi)             // i = k_iter8;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT1)                 // if i == 0, jump to code that
                                       // considers k_left1 loop.
                                       // else, we prepare to enter k_iter8 loop.

    label(.SLOOPKITER8)                // EDGE LOOP (ymm)

    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER8)                  // iterate again if i != 0.

    label(.SCONSIDKLEFT1)

    mov(var(k_left1), rsi)             // i = k_left1;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left1 loop.

    label(.SLOOPKLEFT1)                // EDGE LOOP (scalar)
                                       // NOTE: We must use ymm registers here bc
                                       // using the xmm registers would zero out the
                                       // high bits of the destination registers,
                                       // which would destory intermediate results.

    vmovss(mem(rax       ), xmm0)
    vmovss(mem(rax, r8, 1), xmm1)
    vmovss(mem(rax, r8, 2), xmm2)
    add(imm(1*4), rax)                 // a += 1*cs_b = 1*4;

    vmovss(mem(rbx        ), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovss(mem(rbx, r11, 1), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovss(mem(rbx, r11, 2), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovss(mem(rbx, r13, 1), xmm3)
    add(imm(1*4), rbx)                 // b += 1*rs_b = 1*4;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT1)                  // iterate again if i != 0.

    label(.SPOSTACCUM)
                                       // ymm4  ymm7  ymm10 ymm13  
                                       // ymm5  ymm8  ymm11 ymm14
                                       // ymm6  ymm9  ymm12 ymm15 
    vhaddps( ymm7, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)
    vhaddps( ymm13, ymm10, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

    vhaddps(xmm2,xmm0,xmm4)

    vhaddps( ymm8, ymm5, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps( ymm14, ymm11, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )

    vhaddps(xmm2,xmm0,xmm5)

    vhaddps( ymm9, ymm6, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps( ymm15, ymm12, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )

    vhaddps(xmm2,xmm0,xmm6)
                                       // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
                                       // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
                                       // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm5, xmm5)
    vmulps(xmm0, xmm6, xmm6)
                                           // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    label(.SROWSTORED)

    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm5)
    vmovups(xmm5, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    label(.SROWSTORBZ)

    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm5, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm6, mem(rcx))

    label(.SDONE)

    lea(mem(r12, rdi, 2), r12)         //
    lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

    lea(mem(r14, r8,  2), r14)         //
    lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

    dec(r9)                            // ii -= 1;
    jne(.SLOOP3X4I)                    // iterate again if ii != 0.

    add(imm(4), r15)                   // jj += 4;
    cmp(imm(16), r15)                   // compare jj to 4
    jl(.SLOOP3X4J)                    // if jj <= 4, jump to beginning
                                       // of jj loop; otherwise, loop ends.
    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter32] "m" (k_iter32),
      [k_iter8] "m" (k_iter8),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    )
    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 16;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen_asm_2x16
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            //cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen_asm_1x16
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

void bli_sgemmsup_rd_zen_asm_6x8m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------

    begin_asm()

    mov(var(rs_a), r8)                 // load rs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)

    mov(var(b), rdx)                   // load address of b.
    mov(var(cs_b), r11)                // load cs_b
    lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

    lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
    lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

    // r12 = rcx = c
    // r14 = rax = a
    // rdx = rbx = b
    // r9  = m dim index ii
    // r15 = n dim index jj

    mov(imm(0), r15)                   // jj = 0;

    label(.SLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



    mov(var(a), r14)                   // load address of a
    mov(var(c), r12)                   // load address of c
    mov(var(b), rdx)

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(imm(1*4), rsi)                // rsi *= cs_c = 1*8
    lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(r11, rsi)                     // rsi *= cs_b;
    lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;



    mov(var(m_iter), r9)               // ii = m_iter;

    label(.SLOOP3X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


                                       // zen2 can execute 4 vxorpd ipc with
                                       // a latency of 1 cycle

    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm5,  ymm5,  ymm5)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm7,  ymm7,  ymm7)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm9,  ymm9,  ymm9)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm11, ymm11, ymm11)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm13, ymm13, ymm13)
    vxorps(ymm14, ymm14, ymm14)
    vxorps(ymm15, ymm15, ymm15)

    lea(mem(r12), rcx)                 // rcx = c_iijj;
    lea(mem(r14), rax)                 // rax = a_ii;
    lea(mem(rdx), rbx)                 // rbx = b_jj;

    lea(mem(r8,  r8,  4), rdi)         // rdi = 5*rs_a

    mov(var(k_iter32), rsi)            // i = k_iter32;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKITER8)                 // if i == 0, jump to code that
                                       // contains the k_iter8 loop.

    label(.SLOOPKITER32)               // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)


    // ---------------------------------- iteration 1

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)


    // ---------------------------------- iteration 2
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)


    // ---------------------------------- iteration 3

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER32)                 // iterate again if i != 0.

    label(.SCONSIDKITER8)

    mov(var(k_iter8), rsi)             // i = k_iter8;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT1)                 // if i == 0, jump to code that
                                       // considers k_left1 loop.
                                       // else, we prepare to enter k_iter8 loop.


    label(.SLOOPKITER8)                // EDGE LOOP (ymm)

    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER8)                  // iterate again if i != 0.

    label(.SCONSIDKLEFT1)

    mov(var(k_left1), rsi)             // i = k_left1;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left1 loop.

    label(.SLOOPKLEFT1)                // EDGE LOOP (scalar)
                                       // NOTE: We must use ymm registers here bc
                                       // using the xmm registers would zero out the
                                       // high bits of the destination registers,
                                       // which would destory intermediate results.

    vmovss(mem(rax       ), xmm0)
    vmovss(mem(rax, r8, 1), xmm1)
    vmovss(mem(rax, r8, 2), xmm2)
    add(imm(1*4), rax)                 // a += 1*cs_b = 1*4;

    vmovss(mem(rbx        ), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovss(mem(rbx, r11, 1), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovss(mem(rbx, r11, 2), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovss(mem(rbx, r13, 1), xmm3)
    add(imm(1*4), rbx)                 // b += 1*rs_b = 1*4;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT1)                  // iterate again if i != 0.

    label(.SPOSTACCUM)
                                       // ymm4  ymm7  ymm10 ymm13  
                                       // ymm5  ymm8  ymm11 ymm14
                                       // ymm6  ymm9  ymm12 ymm15 
    vhaddps( ymm7, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)
    vhaddps( ymm13, ymm10, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

    vhaddps(xmm2,xmm0,xmm4)

    vhaddps( ymm8, ymm5, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps( ymm14, ymm11, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )

    vhaddps(xmm2,xmm0,xmm5)

    vhaddps( ymm9, ymm6, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps( ymm15, ymm12, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )

    vhaddps(xmm2,xmm0,xmm6)
                                       // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
                                       // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
                                       // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm5, xmm5)
    vmulps(xmm0, xmm6, xmm6)
                                           // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    label(.SROWSTORED)

    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm5)
    vmovups(xmm5, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    label(.SROWSTORBZ)

    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm5, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm6, mem(rcx))

    label(.SDONE)

    lea(mem(r12, rdi, 2), r12)         //
    lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

    lea(mem(r14, r8,  2), r14)         //
    lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

    dec(r9)                            // ii -= 1;
    jne(.SLOOP3X4I)                    // iterate again if ii != 0.

    add(imm(4), r15)                   // jj += 4;
    cmp(imm(8), r15)                   // compare jj to 4
    jl(.SLOOP3X4J)                    // if jj <= 4, jump to beginning
                                       // of jj loop; otherwise, loop ends.
    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter32] "m" (k_iter32),
      [k_iter8] "m" (k_iter8),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    )
    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 8;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen_asm_2x8
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            //cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen_asm_1x8
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}



void bli_sgemmsup_rd_zen_asm_6x4m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------

    begin_asm()

    mov(var(rs_a), r8)                 // load rs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)

    mov(var(b), rdx)                   // load address of b.
    mov(var(cs_b), r11)                // load cs_b
    lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

    lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
    lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

    // r12 = rcx = c
    // r14 = rax = a
    // rdx = rbx = b
    // r9  = m dim index ii
    // r15 = n dim index jj

    mov(imm(0), r15)                   // jj = 0;

    label(.SLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



    mov(var(a), r14)                   // load address of a
    mov(var(c), r12)                   // load address of c
    mov(var(b), rdx)

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(imm(1*4), rsi)                // rsi *= cs_c = 1*8
    lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(r11, rsi)                     // rsi *= cs_b;
    lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;



    mov(var(m_iter), r9)               // ii = m_iter;

    label(.SLOOP3X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


                                       // zen2 can execute 4 vxorpd ipc with
                                       // a latency of 1 cycle

    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm5,  ymm5,  ymm5)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm7,  ymm7,  ymm7)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm9,  ymm9,  ymm9)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm11, ymm11, ymm11)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm13, ymm13, ymm13)
    vxorps(ymm14, ymm14, ymm14)
    vxorps(ymm15, ymm15, ymm15)

    lea(mem(r12), rcx)                 // rcx = c_iijj;
    lea(mem(r14), rax)                 // rax = a_ii;
    lea(mem(rdx), rbx)                 // rbx = b_jj;

    lea(mem(r8,  r8,  4), rdi)         // rdi = 5*rs_a

    mov(var(k_iter32), rsi)            // i = k_iter32;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKITER8)                 // if i == 0, jump to code that
                                       // contains the k_iter8 loop.

    label(.SLOOPKITER32)               // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    // ---------------------------------- iteration 1
    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)


    // ---------------------------------- iteration 2
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    // ---------------------------------- iteration 3
    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)


    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER32)                 // iterate again if i != 0.


    label(.SCONSIDKITER8)

    mov(var(k_iter8), rsi)             // i = k_iter8;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT1)                 // if i == 0, jump to code that
                                       // considers k_left1 loop.
                                       // else, we prepare to enter k_iter8 loop.


    label(.SLOOPKITER8)                // EDGE LOOP (ymm)

    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovups(mem(rbx, r11, 2), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovups(mem(rbx, r13, 1), ymm3)
    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER8)                  // iterate again if i != 0.

    label(.SCONSIDKLEFT1)

    mov(var(k_left1), rsi)             // i = k_left1;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left1 loop.

    label(.SLOOPKLEFT1)                // EDGE LOOP (scalar)
                                       // NOTE: We must use ymm registers here bc
                                       // using the xmm registers would zero out the
                                       // high bits of the destination registers,
                                       // which would destory intermediate results.

    vmovss(mem(rax       ), xmm0)
    vmovss(mem(rax, r8, 1), xmm1)
    vmovss(mem(rax, r8, 2), xmm2)
    add(imm(1*4), rax)                 // a += 1*cs_b = 1*4;

    vmovss(mem(rbx        ), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovss(mem(rbx, r11, 1), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    vmovss(mem(rbx, r11, 2), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    vfmadd231ps(ymm2, ymm3, ymm12)

    vmovss(mem(rbx, r13, 1), xmm3)
    add(imm(1*4), rbx)                 // b += 1*rs_b = 1*4;
    vfmadd231ps(ymm0, ymm3, ymm13)
    vfmadd231ps(ymm1, ymm3, ymm14)
    vfmadd231ps(ymm2, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT1)                  // iterate again if i != 0.

    label(.SPOSTACCUM)
                                       // ymm4  ymm7  ymm10 ymm13  
                                       // ymm5  ymm8  ymm11 ymm14
                                       // ymm6  ymm9  ymm12 ymm15 
    vhaddps( ymm7, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)
    vhaddps( ymm13, ymm10, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

    vhaddps(xmm2,xmm0,xmm4)

    vhaddps( ymm8, ymm5, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps( ymm14, ymm11, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )

    vhaddps(xmm2,xmm0,xmm5)

    vhaddps( ymm9, ymm6, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps( ymm15, ymm12, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddps( xmm2, xmm1, xmm2 )

    vhaddps(xmm2,xmm0,xmm6)
                                       // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
                                       // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
                                       // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm5, xmm5)
    vmulps(xmm0, xmm6, xmm6)
                                           // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    label(.SROWSTORED)

    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm5)
    vmovups(xmm5, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    label(.SROWSTORBZ)

    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm5, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm6, mem(rcx))

    label(.SDONE)

    lea(mem(r12, rdi, 2), r12)         //
    lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

    lea(mem(r14, r8,  2), r14)         //
    lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

    dec(r9)                            // ii -= 1;
    jne(.SLOOP3X4I)                    // iterate again if ii != 0.

    add(imm(4), r15)                   // jj += 4;
    cmp(imm(4), r15)                   // compare jj to 4
    jl(.SLOOP3X4J)                    // if jj <= 4, jump to beginning
                                       // of jj loop; otherwise, loop ends.
    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter32] "m" (k_iter32),
      [k_iter8] "m" (k_iter8),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    )
    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 4;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen_asm_2x4
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            //cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen_asm_1x4
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

void bli_sgemmsup_rd_zen_asm_6x2m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter32 = k0 / 32;
    uint64_t k_left32 = k0 % 32;
    uint64_t k_iter8  = k_left32 / 8;
    uint64_t k_left1  = k_left32 % 8;

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    if ( m_iter == 0 ) goto consider_edge_cases;
 
    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(rs_a), r8)                 // load rs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)

    mov(var(b), rdx)                   // load address of b.
    mov(var(cs_b), r11)                // load cs_b
    lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

    lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
    lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a

    // r12 = rcx = c
    // r14 = rax = a
    // rdx = rbx = b
    // r9  = m dim index ii
    // r15 = n dim index jj

    mov(imm(0), r15)                   // jj = 0;

    label(.SLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



    mov(var(a), r14)                   // load address of a
    mov(var(c), r12)                   // load address of c
    mov(var(b), rdx)

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(imm(1*4), rsi)                // rsi *= cs_c = 1*8
    lea(mem(r12, rsi, 1), r12)         // r12 = c + 4*jj*cs_c;

    lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
    imul(r11, rsi)                     // rsi *= cs_b;
    lea(mem(rdx, rsi, 1), rdx)         // rbx = b + 4*jj*cs_b;



    mov(var(m_iter), r9)               // ii = m_iter;

    label(.SLOOP3X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


                                       // zen2 can execute 4 vxorpd ipc with
                                       // a latency of 1 cycle

    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm5,  ymm5,  ymm5)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm7,  ymm7,  ymm7)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm9,  ymm9,  ymm9)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm11, ymm11, ymm11)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm13, ymm13, ymm13)
    vxorps(ymm14, ymm14, ymm14)
    vxorps(ymm15, ymm15, ymm15)

    lea(mem(r12), rcx)                 // rcx = c_iijj;
    lea(mem(r14), rax)                 // rax = a_ii;
    lea(mem(rdx), rbx)                 // rbx = b_jj;

    lea(mem(r8,  r8,  4), rdi)         // rdi = 5*rs_a

    mov(var(k_iter32), rsi)            // i = k_iter32;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKITER8)                 // if i == 0, jump to code that
                                       // contains the k_iter8 loop.

    label(.SLOOPKITER32)               // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;

    // ---------------------------------- iteration 1
    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;

    // ---------------------------------- iteration 2
    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;
    // ---------------------------------- iteration 3
    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER32)                 // iterate again if i != 0.

    label(.SCONSIDKITER8)

    mov(var(k_iter8), rsi)             // i = k_iter8;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT1)                 // if i == 0, jump to code that
                                       // considers k_left1 loop.
                                       // else, we prepare to enter k_iter8 loop.

    label(.SLOOPKITER8)                // EDGE LOOP (ymm)

    prefetch(0, mem(rax, r10, 1, 0*8)) // prefetch rax + 3*cs_a
    prefetch(0, mem(rax, r8,  4, 0*8)) // prefetch rax + 4*cs_a
    prefetch(0, mem(rax, rdi, 1, 0*8)) // prefetch rax + 5*cs_a

    vmovups(mem(rax       ), ymm0)
    vmovups(mem(rax, r8, 1), ymm1)
    vmovups(mem(rax, r8, 2), ymm2)
    add(imm(8*4), rax)                 // a += 4*cs_b = 4*8;

    vmovups(mem(rbx        ), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovups(mem(rbx, r11, 1), ymm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    add(imm(8*4), rbx)                 // b += 4*rs_b = 4*8;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER8)                  // iterate again if i != 0.

    label(.SCONSIDKLEFT1)

    mov(var(k_left1), rsi)             // i = k_left1;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left1 loop.

    label(.SLOOPKLEFT1)                // EDGE LOOP (scalar)
                                       // NOTE: We must use ymm registers here bc
                                       // using the xmm registers would zero out the
                                       // high bits of the destination registers,
                                       // which would destory intermediate results.

    vmovss(mem(rax       ), xmm0)
    vmovss(mem(rax, r8, 1), xmm1)
    vmovss(mem(rax, r8, 2), xmm2)
    add(imm(1*4), rax)                 // a += 1*cs_b = 1*4;

    vmovss(mem(rbx        ), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm4)
    vfmadd231ps(ymm1, ymm3, ymm5)
    vfmadd231ps(ymm2, ymm3, ymm6)

    vmovss(mem(rbx, r11, 1), xmm3)
    vfmadd231ps(ymm0, ymm3, ymm7)
    vfmadd231ps(ymm1, ymm3, ymm8)
    vfmadd231ps(ymm2, ymm3, ymm9)

    add(imm(1*4), rbx)                 // b += 1*rs_b = 1*4;

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT1)                  // iterate again if i != 0.

    label(.SPOSTACCUM)
                                       // ymm4  ymm7  
                                       // ymm5  ymm8 
                                       // ymm6  ymm9 
    vhaddps( ymm7, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)
    vhaddps(xmm0,xmm0,xmm4)

    vhaddps( ymm8, ymm5, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps(xmm0,xmm0,xmm5)

    vhaddps( ymm9, ymm6, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddps( xmm0, xmm1, xmm0 )
    vhaddps(xmm0,xmm0,xmm6)
                                       // ymm4 = sum(ymm4) sum(ymm7)
                                       // ymm5 = sum(ymm5) sum(ymm8)
                                       // ymm6 = sum(ymm6) sum(ymm9)
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm5, xmm5)
    vmulps(xmm0, xmm6, xmm6)
                                           // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    label(.SROWSTORED)

    vmovsd(mem(rcx), xmm0)////a0a1
    vfmadd231ps(xmm0, xmm3, xmm4)//c*beta+(a0a1)
    vmovsd(xmm4, mem(rcx))//a0a1
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm5)
    vmovsd(xmm5, mem(rcx))
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm6)
    vmovsd(xmm6, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)
    label(.SROWSTORBZ)

    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)
    vmovsd(xmm5, mem(rcx))
    add(rdi, rcx)
    vmovsd(xmm6, mem(rcx))

    label(.SDONE)

    lea(mem(r12, rdi, 2), r12)         //
    lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

    lea(mem(r14, r8,  2), r14)         //
    lea(mem(r14, r8,  1), r14)         // a_ii = r14 += 3*rs_a

    dec(r9)                            // ii -= 1;
    jne(.SLOOP3X4I)                    // iterate again if ii != 0.

    add(imm(4), r15)                   // jj += 4;
    cmp(imm(4), r15)                   // compare jj to 4
    jl(.SLOOP3X4J)                    // if jj <= 4, jump to beginning
                                       // of jj loop; otherwise, loop ends.
    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter32] "m" (k_iter32),
      [k_iter8] "m" (k_iter8),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    )
    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 2;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = c + i_edge*rs_c;
        float* restrict bj  = b;
        float* restrict ai  = a + i_edge*rs_a;

        if ( 2 == m_left )
        {
            const dim_t mr_cur = 2;

            bli_sgemmsup_rd_zen_asm_2x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            //cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
        }
        if ( 1 == m_left )
        {
            const dim_t mr_cur = 1;

            bli_sgemmsup_rd_zen_asm_1x2
            (
              conja, conjb, mr_cur, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
        }
    }
}

