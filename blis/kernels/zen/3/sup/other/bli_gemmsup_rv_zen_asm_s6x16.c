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
   rrr:
    --------        ------        --------      
    --------        ------        --------      
    --------   +=   ------ ...    --------      
    --------        ------        --------      
    --------        ------            :         
    --------        ------            :         

   rcr:
    --------        | | | |       --------      
    --------        | | | |       --------      
    --------   +=   | | | | ...   --------      
    --------        | | | |       --------      
    --------        | | | |           :         
    --------        | | | |           :         

   Assumptions:
   - B is row-stored;
   - A is row- or column-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels explicitly support column-oriented IO, implemented
   via an in-register transpose. And thus they also support the crr and
   ccr cases, though only crr is ever utilized (because ccr is handled by
   transposing the operation and executing rcr, which does not incur the
   cost of the in-register transpose).

   crr:
    | | | | | | | |       ------        --------      
    | | | | | | | |       ------        --------      
    | | | | | | | |  +=   ------ ...    --------      
    | | | | | | | |       ------        --------      
    | | | | | | | |       ------            :         
    | | | | | | | |       ------            :         
*/
// Prototype reference microkernels.
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref )

void bli_sgemmsup_rv_zen_asm_5x16
     (
       conj_t     conja,
       conj_t     conjb,
       dim_t      m0,
       dim_t      n0,
       dim_t      k0,
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c
    lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 7*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)

    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
        
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm7, ymm7)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm9, ymm9)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm11, ymm11)
    vmulps(ymm0, ymm12, ymm12)
    vmulps(ymm0, ymm13, ymm13)
    
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm5)
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm7)
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)
        
    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm9)
    vmovups(ymm9, mem(rcx, rsi, 8))
    add(rdi, rcx)
        
    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm11)
    vmovups(ymm11, mem(rcx, rsi, 8))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm12)
    vmovups(ymm12, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm13)
    vmovups(ymm13, mem(rcx, rsi, 8))
    //add(rdi, rcx)    
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    /********************************************/
    vextractf128(imm(0x0), ymm12, xmm0)//e0-e3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm12, xmm0)//e4-e7
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c
    
    /*********************************************/
    vunpcklps(ymm7, ymm5, ymm0)
    vunpcklps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
        
    vunpckhps(ymm7, ymm5, ymm0)
    vunpckhps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    //lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c
    vextractf128(imm(0x0), ymm13, xmm0)//e0-e3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0,xmm0, xmm1)
    vshufps(imm(0x02), xmm0,xmm0, xmm2)
    vshufps(imm(0x03), xmm0,xmm0, xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm13, xmm0)//e4-e7
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0,xmm0, xmm1)
    vshufps(imm(0x02), xmm0,xmm0, xmm2)
    vshufps(imm(0x03), xmm0,xmm0, xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    jmp(.SDONE)                        // jump to end.
        
    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx))
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)
    
    vmovups(ymm8, mem(rcx))
    vmovups(ymm9, mem(rcx, rsi, 8))
    add(rdi, rcx)
    
    vmovups(ymm10, mem(rcx))
    vmovups(ymm11, mem(rcx, rsi, 8))
    add(rdi, rcx)
    
    vmovups(ymm12, mem(rcx))
    vmovups(ymm13, mem(rcx, rsi, 8))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    /********************************************/
    vextractf128(imm(0x0), ymm12, xmm0)//e0-e3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm12, xmm0)//e4-e7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c
    
    /*********************************************/
    vunpcklps(ymm7, ymm5, ymm0)
    vunpcklps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
        
    vunpckhps(ymm7, ymm5, ymm0)
    vunpckhps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    vextractf128(imm(0x0), ymm13, xmm0)//e0-e3
    vshufps(imm(0x01), xmm0,xmm0, xmm1)
    vshufps(imm(0x02), xmm0,xmm0, xmm2)
    vshufps(imm(0x03), xmm0,xmm0, xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm13, xmm0)//e4-e7
    vshufps(imm(0x01), xmm0,xmm0, xmm1)
    vshufps(imm(0x02), xmm0,xmm0, xmm2)
    vshufps(imm(0x03), xmm0,xmm0, xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_4x16
     (
       conj_t             conja,
       conj_t             conjb,
       dim_t              m0,
       dim_t              n0,
       dim_t              k0,
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c
    lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 7*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
        
    label(.SLOOPKITER)                 // MAIN LOOP
        
    // ---------------------------------- iteration 0
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)

    // ---------------------------------- iteration 2
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm7, ymm7)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm9, ymm9)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm11, ymm11)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                      // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm5)
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm7)
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm9)
    vmovups(ymm9, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm11)
    vmovups(ymm11, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    
    jmp(.SDONE)                        // jump to end.
    
    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c

    vunpcklps(ymm7, ymm5, ymm0)
    vunpcklps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
        
    vunpckhps(ymm7, ymm5, ymm0)
    vunpckhps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    

    jmp(.SDONE)                        // jump to end.


    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
        
    vmovups(ymm4, mem(rcx))
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx))
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm8, mem(rcx))
    vmovups(ymm9, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm10, mem(rcx))
    vmovups(ymm11, mem(rcx, rsi, 8))
    //add(rdi, rcx)
        
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    vunpcklps(ymm7, ymm5, ymm0)
    vunpcklps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
        
    vunpckhps(ymm7, ymm5, ymm0)
    vunpckhps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_3x16
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)



    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b

    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    //lea(mem(rcx, rdi, 2), rdx)         //
    //lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 2*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c
    lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 7*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
        
    // ---------------------------------- iteration 0
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)

    // ---------------------------------- iteration 3
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
        
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm7, ymm7)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm9, ymm9)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    //lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
    lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                          // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm5)
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm7)
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm9)
    vmovups(ymm9, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    /********************************************/
    vextractf128(imm(0x0), ymm8, xmm0)//c0-c3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm11)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm11, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm8, xmm0)//e4-e7
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e4
    vfmadd231ps(xmm6, xmm3, xmm1)//e5
    vfmadd231ps(xmm8, xmm3, xmm2)//e6
    vfmadd231ps(xmm10, xmm3, xmm14)//e7
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c
    
    /*********************************************/
    vunpcklps(ymm7, ymm5, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm7, ymm5, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    
    /********************************************/
    vextractf128(imm(0x0), ymm9, xmm0)//c0-c3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm9, xmm0)//e4-e7
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    jmp(.SDONE)                        // jump to end.


    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
        
    vmovups(ymm4, mem(rcx))
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx))
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm8, mem(rcx))
    vmovups(ymm9, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vperm2f128(imm(0x01),ymm0,ymm0,ymm0)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm2)
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a2b2
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    /********************************************/
    vextractf128(imm(0x0), ymm8, xmm0)//c0-c3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm8, xmm0)//c4-c7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c
    
    /*********************************************/
    vunpcklps(ymm7, ymm5, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm7, ymm5, ymm2)    //a2b2a3b3 a6b6a7b7    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vperm2f128(imm(0x01),ymm0,ymm0,ymm0)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm2)
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a2b2
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    
    /********************************************/
    vextractf128(imm(0x0), ymm9, xmm0)//c0-c3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm9, xmm0)//c4-c7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))        

    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_2x16
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 5*cs_c
    lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 7*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    // ---------------------------------- iteration 2    
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
        
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm7, ymm7)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    //lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
    //lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm5)
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm7)
    vmovups(ymm7, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)
    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c

    vunpcklps(ymm7, ymm5, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm7, ymm5, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    

    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx))
    vmovups(ymm7, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vperm2f128(imm(0x01),ymm0,ymm0,ymm0)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm2)
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a2b2
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c

    vunpcklps(ymm7, ymm5, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm7, ymm5, ymm2)    //a2b2a3b3 a6b6a7b7    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vperm2f128(imm(0x01),ymm0,ymm0,ymm0)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm2)
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a2b2
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    
    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_1x16
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)



    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 0*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 5*cs_c
    lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 7*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP    
    
    // ---------------------------------- iteration 0    
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
        
    // ---------------------------------- iteration 1
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    

    // ---------------------------------- iteration 2
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)    

    // ---------------------------------- iteration 3
    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.    
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
        
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.    
    
    label(.SPOSTACCUM)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
        
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm5)
    vmovups(ymm5, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vextractf128(imm(0x0), ymm4, xmm0)//c0-c3
    vmovss(mem(rcx),xmm7)
    vmovss(mem(rcx, rsi, 1),xmm6)
    vmovss(mem(rcx, rsi, 2),xmm11)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm7, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm11, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    vextractf128(imm(0x1), ymm4, xmm0)//e4-e7
    vmovss(mem(rcx),xmm4)
    vmovss(mem(rcx, rsi, 1),xmm6)
    vmovss(mem(rcx, rsi, 2),xmm8)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e4
    vfmadd231ps(xmm6, xmm3, xmm1)//e5
    vfmadd231ps(xmm8, xmm3, xmm2)//e6
    vfmadd231ps(xmm10, xmm3, xmm14)//e7
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    vextractf128(imm(0x0), ymm5, xmm0)//c0-c3
    vmovss(mem(rcx),xmm4)
    vmovss(mem(rcx, rsi, 1),xmm6)
    vmovss(mem(rcx, rsi, 2),xmm11)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm11, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c

    vextractf128(imm(0x1), ymm5, xmm0)//e4-e7
    vmovss(mem(rcx),xmm4)
    vmovss(mem(rcx, rsi, 1),xmm6)
    vmovss(mem(rcx, rsi, 2),xmm8)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e4
    vfmadd231ps(xmm6, xmm3, xmm1)//e5
    vfmadd231ps(xmm8, xmm3, xmm2)//e6
    vfmadd231ps(xmm10, xmm3, xmm14)//e7
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    vmovups(ymm5, mem(rcx, rsi, 8))
    //add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vextractf128(imm(0x0), ymm4, xmm0)//c0-c3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c

    vextractf128(imm(0x1), ymm4, xmm0)//e4-e7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    vextractf128(imm(0x0), ymm5, xmm0)//c0-c3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c        
    vextractf128(imm(0x1), ymm5, xmm0)//e4-e7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))    
    
    label(.SDONE)    
    

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_6x8
     (
       conj_t           conja,
       conj_t           conjb,
       dim_t            m0,
       dim_t            n0,
       dim_t            k0,
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;
        
    // -------------------------------------------------------------------------
    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
        
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*16), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*16), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)
        
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*16), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*16), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
        
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*16), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
        
    label(.SPOSTACCUM)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm12, ymm12)
    vmulps(ymm0, ymm14, ymm14)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                      // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm12)
    vmovups(ymm12, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm14)
    vmovups(ymm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /****6x8 tile is transposed and saved in col major as 8x6*****/    
    vunpcklps(ymm6, ymm4, ymm0)
    vunpcklps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    vunpcklps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vpermilpd(imm(1),xmm0,xmm5)//e1f1
    vpermilpd(imm(1),xmm2,xmm6)//e5f5
    vfmadd231ps(mem(rdx), xmm3, xmm0)
    vfmadd231ps(mem(rdx, rsi, 4), xmm3, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma44..gamma54 )
    lea(mem(rdx, rsi, 1), rdx)
    vfmadd231ps(mem(rdx), xmm3, xmm5)
    vfmadd231ps(mem(rdx, rsi, 4), xmm3, xmm6)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma41..gamma51 )    
    vmovlpd(xmm6, mem(rdx, rsi, 4)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 1), rdx)
    
    vunpckhps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vpermilpd(imm(1),xmm0,xmm5)
    vpermilpd(imm(1),xmm2,xmm6)    
    vfmadd231ps(mem(rdx), xmm3, xmm0)
    vfmadd231ps(mem(rdx, rsi, 4), xmm3, xmm2)    
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma46..gamma56 )    
    lea(mem(rdx, rsi, 1), rdx)
    vfmadd231ps(mem(rdx), xmm3, xmm5)
    vfmadd231ps(mem(rdx, rsi, 4), xmm3, xmm6)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma43..gamma53 )
    vmovlpd(xmm6, mem(rdx, rsi, 4)) // store ( gamma47..gamma57 )
        
    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
        
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
        
    vmovups(ymm10, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm12, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm14, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)
    vunpcklps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    /******************top right tile 8x2***************************/
    vunpcklps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma44..gamma54 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma41..gamma51 )    
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 1), rdx)
    
    vunpckhps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma46..gamma56 )    
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma43..gamma53 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma47..gamma57 )
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_5x8
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()

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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    

                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
        
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
        
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    vbroadcastss(mem(rax, r8,  4), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm12, ymm12)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                          // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
        
    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm12)
    vmovups(ymm12, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    /********************************************/
    vextractf128(imm(0x0), ymm12, xmm0)//e0-e3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm12, xmm0)//e4-e7
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm8)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm8, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))
    
    jmp(.SDONE)                        // jump to end.
        
    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
        
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
        
    vmovups(ymm10, mem(rcx))
    add(rdi, rcx)
        
    vmovups(ymm12, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    
    /********************************************/
    vextractf128(imm(0x0), ymm12, xmm0)//e0-e3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm12, xmm0)//e4-e7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_4x8
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;    
                                      // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    
    label(.SROWSTORED)
        
        
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
    
    vmovups(ymm10, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)    //c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2) 
    vblendps(imm(0xcc), ymm2, ymm0, ymm0) 
    vblendps(imm(0x33), ymm2, ymm1, ymm1) 
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)
    
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )        
    
    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_3x8
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 2*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx, 0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx, 0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx, 0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx, 0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx, 0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm8)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)

    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    
    label(.SROWSTORED)
    
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    
    /********************************************/
    vextractf128(imm(0x0), ymm8, xmm0)//c0-c3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm11)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm11, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rcx += cs_c

    vextractf128(imm(0x1), ymm8, xmm0)//c0-c3
    vmovss(mem(rdx),xmm4)
    vmovss(mem(rdx, rsi, 1),xmm6)
    vmovss(mem(rdx, rsi, 2),xmm11)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm11, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))    
    jmp(.SDONE)                        // jump to end.


    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case


    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)

    vmovups(ymm8, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    
    
    /********************************************/
    vextractf128(imm(0x0), ymm8, xmm0)//c0-c3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))

    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm8, xmm0)//c4-c7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rdx))
    vmovss(xmm1, mem(rdx, rsi, 1))
    vmovss(xmm2, mem(rdx, rsi, 2))
    vmovss(xmm14, mem(rdx, rax, 1))    
    
    label(.SDONE)
    
    

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_2x8
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)    
    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)

    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                      // now avoid loading C if beta == 0
    
    vxorps(xmm0,xmm0,xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )    

    jmp(.SDONE)                        // jump to end.
        
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    vmovups(ymm6, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)    //a0b0a1b1 a2b2a3b3
    vunpckhps(ymm6, ymm4, ymm2)    //a2b2a3b3 a6b6a7b7
    vperm2f128(imm(0x01),ymm0,ymm0,ymm11)
    vperm2f128(imm(0x01),ymm2,ymm2,ymm12)
    
    vshufpd(imm(0x01), xmm0, xmm0, xmm1)//a1b1
    vshufpd(imm(0x01), xmm2, xmm2, xmm10)//a3b3    
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm2)
    vfmadd231ps(xmm6, xmm3, xmm10)    
    vmovsd(xmm2, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )
    lea(mem(rcx, rsi, 4), rcx) // rcx += cs_c
    
    vshufpd(imm(0x01), xmm11, xmm11, xmm1)//a1b1
    vshufpd(imm(0x01), xmm12, xmm12, xmm10)//a3b3        
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rsi, 1),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm11)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm11, mem(rcx)) // store ( gamma00..gamma10 )
    vmovsd(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma11 )    
    
    vmovsd(mem(rcx, rsi, 2),xmm4)
    vmovsd(mem(rcx, rax, 1),xmm6)
    vfmadd231ps(xmm4, xmm3, xmm12)
    vfmadd231ps(xmm6, xmm3, xmm10)
    vmovsd(xmm12, mem(rcx, rsi, 2)) // store ( gamma02..gamma12 )
    vmovsd(xmm10, mem(rcx, rax, 1)) // store ( gamma03..gamma13 )

    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_1x8
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;
    
    // -------------------------------------------------------------------------

    begin_asm()
    
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
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
        
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm4)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                      // now avoid loading C if beta == 0
    
    vxorps(xmm0,xmm0,xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /********************************************/
    vextractf128(imm(0x0), ymm4, xmm0)//c0-c3
    vmovss(mem(rcx),xmm8)
    vmovss(mem(rcx, rsi, 1),xmm6)
    vmovss(mem(rcx, rsi, 2),xmm11)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm8, xmm3, xmm0)//e0
    vfmadd231ps(xmm6, xmm3, xmm1)//e1
    vfmadd231ps(xmm11, xmm3, xmm2)//e2
    vfmadd231ps(xmm10, xmm3, xmm14)//e3
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))

    lea(mem(rcx, rsi, 4), rcx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm4, xmm0)//e4-e7
    vmovss(mem(rcx),xmm4)
    vmovss(mem(rcx, rsi, 1),xmm6)
    vmovss(mem(rcx, rsi, 2),xmm8)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vfmadd231ps(xmm4, xmm3, xmm0)//e4
    vfmadd231ps(xmm6, xmm3, xmm1)//e5
    vfmadd231ps(xmm8, xmm3, xmm2)//e6
    vfmadd231ps(xmm10, xmm3, xmm14)//e7
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))    
    
    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
    
    vmovups(ymm4, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vextractf128(imm(0x0), ymm4, xmm0)//c0-c3
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))

    lea(mem(rcx, rsi, 4), rcx) // rdx += 4*cs_c

    vextractf128(imm(0x1), ymm4, xmm0)//c4-c7
    vshufps(imm(0x01), xmm0, xmm0,xmm1)
    vshufps(imm(0x02), xmm0, xmm0,xmm2)
    vshufps(imm(0x03), xmm0, xmm0,xmm14)
    vmovss(xmm0, mem(rcx))
    vmovss(xmm1, mem(rcx, rsi, 1))
    vmovss(xmm2, mem(rcx, rsi, 2))
    vmovss(xmm14, mem(rcx, rax, 1))    

    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_6x4
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()

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


    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
        
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
        
    label(.SLOOPKITER)                 // MAIN LOOP
        
    // ---------------------------------- iteration 0    
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    vmulps(xmm0, xmm12, xmm12)
    vmulps(xmm0, xmm14, xmm14)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                      // now avoid loading C if beta == 0
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)    
    
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm8)
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
        
    vfmadd231ps(mem(rcx), xmm3, xmm10)
    vmovups(xmm10, mem(rcx))
    add(rdi, rcx)
        
    vfmadd231ps(mem(rcx), xmm3, xmm12)
    vmovups(xmm12, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm14)
    vmovups(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1    
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklpd(xmm1, xmm0, xmm2)//a0b0c0d0
    vunpckhpd(xmm1, xmm0, xmm5)//a1b1c1d1
    
    vfmadd231ps(mem(rcx), xmm3, xmm2)
    vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm5)
    vmovups(xmm2, mem(rcx))
    vmovups(xmm5, mem(rcx, rsi, 1)) 
    lea(mem(rcx, rsi, 2), rcx) // rcx += 2*cs_c
    
    vunpckhps(xmm6, xmm4, xmm0)//a2b2a3b3    
    vunpckhps(xmm10, xmm8, xmm1)//c2d2c3d3
    vunpcklpd(xmm1, xmm0, xmm7)//a2b2c2d2
    vunpckhpd(xmm1, xmm0, xmm9)//a3b3c3d3
    
    vfmadd231ps(mem(rcx), xmm3, xmm7)
    vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm9)
    vmovups(xmm7, mem(rcx))
    vmovups(xmm9, mem(rcx, rsi, 1)) 
    
    vunpcklps(xmm14, xmm12, xmm0)//e0f0e1f1
    vunpckhps(xmm14, xmm12, xmm1)//e2f2e3f3    
    vmovsd(mem(rdx),xmm2)
    vmovsd(mem(rdx, rsi, 1),xmm4)    
    vmovsd(mem(rdx, rsi, 2),xmm6)
    vmovsd(mem(rdx, rax, 1),xmm8)
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//e1f1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//e3f3
    vfmadd231ps(xmm2, xmm3, xmm0)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vfmadd231ps(xmm8, xmm3, xmm7)
    vmovsd(xmm0, mem(rdx)) //e0f0
    vmovsd(xmm5, mem(rdx, rsi, 1)) //e1f1
    vmovsd(xmm1, mem(rdx, rsi, 2)) //e2f2
    vmovsd(xmm7, mem(rdx, rax, 1)) //e3f3

    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
        
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
        
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm10, mem(rcx))
    add(rdi, rcx)
        
    vmovups(xmm12, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1    
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklpd(xmm1, xmm0, xmm2)//a0b0c0d0
    vunpckhpd(xmm1, xmm0, xmm5)//a1b1c1d1
    
    vmovups(xmm2, mem(rcx))
    vmovups(xmm5, mem(rcx, rsi, 1)) 
    lea(mem(rcx, rsi, 2), rcx) // rcx += 2*cs_c
    
    vunpckhps(xmm6, xmm4, xmm0)//a2b2a3b3    
    vunpckhps(xmm10, xmm8, xmm1)//c2d2c3d3
    vunpcklpd(xmm1, xmm0, xmm7)//a2b2c2d2
    vunpckhpd(xmm1, xmm0, xmm9)//a3b3c3d3
    
    vmovups(xmm7, mem(rcx))
    vmovups(xmm9, mem(rcx, rsi, 1)) 
    
    vunpcklps(xmm14, xmm12, xmm0)//e0f0e1f1
    vunpckhps(xmm14, xmm12, xmm1)//e2f2e3f3    
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//e1f1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//e3f3
    vmovsd(xmm0, mem(rdx)) //e0f0
    vmovsd(xmm5, mem(rdx, rsi, 1)) //e1f1
    vmovsd(xmm1, mem(rdx, rsi, 2)) //e2f2
    vmovsd(xmm7, mem(rdx, rax, 1)) //e3f3    

    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_5x4
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()

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


    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(r9, rsi)                       // rsi = rs_b;
    sal(imm(5), rsi)                   // rsi = 16*rs_b;
    lea(mem(rax, rsi, 1), rdx)         // rdx = b + 16*rs_b;
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
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
    vmovups(mem(rbx,  0*32), xmm0)
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
    vmovups(mem(rbx,  0*32), xmm0)
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
    vmovups(mem(rbx,  0*32), xmm0)
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
    
    vmovups(mem(rbx,  0*32), xmm0)
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

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
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
    
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    
    label(.SROWSTORED)
        
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm8)
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm10)
    vmovups(xmm10, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm12)
    vmovups(xmm12, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1    
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklpd(xmm1, xmm0, xmm2)//a0b0c0d0
    vunpckhpd(xmm1, xmm0, xmm5)//a1b1c1d1
    
    vfmadd231ps(mem(rcx), xmm3, xmm2)
    vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm5)
    vmovups(xmm2, mem(rcx))
    vmovups(xmm5, mem(rcx, rsi, 1)) 
    lea(mem(rcx, rsi, 2), rcx) // rcx += 2*cs_c
    
    vunpckhps(xmm6, xmm4, xmm0)//a2b2a3b3    
    vunpckhps(xmm10, xmm8, xmm1)//c2d2c3d3
    vunpcklpd(xmm1, xmm0, xmm7)//a2b2c2d2
    vunpckhpd(xmm1, xmm0, xmm9)//a3b3c3d3
    
    vfmadd231ps(mem(rcx), xmm3, xmm7)
    vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm9)
    vmovups(xmm7, mem(rcx))
    vmovups(xmm9, mem(rcx, rsi, 1)) 
    
    vmovss(mem(rdx),xmm2)
    vmovss(mem(rdx, rsi, 1),xmm4)    
    vmovss(mem(rdx, rsi, 2),xmm6)
    vmovss(mem(rdx, rax, 1),xmm8)
    vshufps(imm(0x01), xmm12, xmm12,xmm1)
    vshufps(imm(0x02), xmm12, xmm12,xmm5)
    vshufps(imm(0x03), xmm12, xmm12,xmm7)    
    vfmadd231ps(xmm2, xmm3, xmm12)
    vfmadd231ps(xmm4, xmm3, xmm1)
    vfmadd231ps(xmm6, xmm3, xmm5)
    vfmadd231ps(xmm8, xmm3, xmm7)
    vmovss(xmm12, mem(rdx)) //e0
    vmovss(xmm1, mem(rdx, rsi, 1)) //e1
    vmovss(xmm5, mem(rdx, rsi, 2)) //e2
    vmovss(xmm7, mem(rdx, rax, 1)) //e3

    jmp(.SDONE)                        // jump to end.
        
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
        
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
        
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm10, mem(rcx))
    add(rdi, rcx)
        
    vmovups(xmm12, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1    
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklpd(xmm1, xmm0, xmm2)//a0b0c0d0
    vunpckhpd(xmm1, xmm0, xmm5)//a1b1c1d1
    
    vmovups(xmm2, mem(rcx))
    vmovups(xmm5, mem(rcx, rsi, 1))
    lea(mem(rcx, rsi, 2), rcx) // rcx += 2*cs_c
    
    vunpckhps(xmm6, xmm4, xmm0)//a2b2a3b3    
    vunpckhps(xmm10, xmm8, xmm1)//c2d2c3d3
    vunpcklpd(xmm1, xmm0, xmm7)//a2b2c2d2
    vunpckhpd(xmm1, xmm0, xmm9)//a3b3c3d3
    vmovups(xmm7, mem(rcx))
    vmovups(xmm9, mem(rcx, rsi, 1))

    vshufps(imm(0x01), xmm12, xmm12,xmm1)
    vshufps(imm(0x02), xmm12, xmm12,xmm5)
    vshufps(imm(0x03), xmm12, xmm12,xmm7)
    vmovss(xmm12, mem(rdx)) //e0
    vmovss(xmm1, mem(rdx, rsi, 1)) //e1
    vmovss(xmm5, mem(rdx, rsi, 2)) //e2
    vmovss(xmm7, mem(rdx, rax, 1)) //e3

    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_4x4
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()

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


    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)    
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    // ---------------------------------- iteration 2    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)
    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
    
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
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                      // now avoid loading C if beta == 0
    
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    

    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
    
    
    vfmadd231ps(mem(rcx), xmm3, xmm8)
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    
    
    vfmadd231ps(mem(rcx), xmm3, xmm10)
    vmovups(xmm10, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1    
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklpd(xmm1, xmm0, xmm2)//a0b0c0d0
    vunpckhpd(xmm1, xmm0, xmm5)//a1b1c1d1
    
    vfmadd231ps(mem(rcx), xmm3, xmm2)
    vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm5)
    vmovups(xmm2, mem(rcx))
    vmovups(xmm5, mem(rcx, rsi, 1)) 
    lea(mem(rcx, rsi, 2), rcx) // rcx += 2*cs_c
    
    vunpckhps(xmm6, xmm4, xmm0)//a2b2a3b3    
    vunpckhps(xmm10, xmm8, xmm1)//c2d2c3d3
    vunpcklpd(xmm1, xmm0, xmm7)//a2b2c2d2
    vunpckhpd(xmm1, xmm0, xmm9)//a3b3c3d3
    
    vfmadd231ps(mem(rcx), xmm3, xmm7)
    vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm9)
    vmovups(xmm7, mem(rcx))
    vmovups(xmm9, mem(rcx, rsi, 1)) 
    
    jmp(.SDONE)                        // jump to end.
        
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
        
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
        
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    
    vmovups(xmm10, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)
    
    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1    
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklpd(xmm1, xmm0, xmm2)//a0b0c0d0
    vunpckhpd(xmm1, xmm0, xmm5)//a1b1c1d1
    vmovups(xmm2, mem(rcx))
    vmovups(xmm5, mem(rcx, rsi, 1)) 
    lea(mem(rcx, rsi, 2), rcx) // rcx += 2*cs_c
    
    vunpckhps(xmm6, xmm4, xmm0)//a2b2a3b3    
    vunpckhps(xmm10, xmm8, xmm1)//c2d2c3d3
    vunpcklpd(xmm1, xmm0, xmm7)//a2b2c2d2
    vunpckhpd(xmm1, xmm0, xmm9)//a3b3c3d3    
    vmovups(xmm7, mem(rcx))
    vmovups(xmm9, mem(rcx, rsi, 1)) 
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_3x4
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()

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


    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 2*8))         // prefetch c + 3*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP    
    
    // ---------------------------------- iteration 0    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)    

    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
        
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
        
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm8)
    vmovups(xmm8, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.
    
    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//e0f0e1f1
    vunpckhps(xmm6, xmm4, xmm1)//e2f2e3f3    
    vmovsd(mem(rcx),xmm2)
    vmovsd(mem(rcx, rsi, 1),xmm4)    
    vmovsd(mem(rcx, rsi, 2),xmm6)
    vmovsd(mem(rcx, rax, 1),xmm10)
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//e1f1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//e3f3
    vfmadd231ps(xmm2, xmm3, xmm0)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vfmadd231ps(xmm10, xmm3, xmm7)
    vmovsd(xmm0, mem(rcx)) //e0f0
    vmovsd(xmm5, mem(rcx, rsi, 1)) //e1f1
    vmovsd(xmm1, mem(rcx, rsi, 2)) //e2f2
    vmovsd(xmm7, mem(rcx, rax, 1)) //e3f3
    
    vmovss(mem(rdx),xmm2)
    vmovss(mem(rdx, rsi, 1),xmm4)    
    vmovss(mem(rdx, rsi, 2),xmm6)
    vmovss(mem(rdx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm8, xmm8,xmm1)
    vshufps(imm(0x02), xmm8, xmm8,xmm5)
    vshufps(imm(0x03), xmm8, xmm8,xmm7)    
    vfmadd231ps(xmm2, xmm3, xmm8)
    vfmadd231ps(xmm4, xmm3, xmm1)
    vfmadd231ps(xmm6, xmm3, xmm5)
    vfmadd231ps(xmm10, xmm3, xmm7)
    vmovss(xmm8, mem(rdx)) //e0
    vmovss(xmm1, mem(rdx, rsi, 1)) //e1
    vmovss(xmm5, mem(rdx, rsi, 2)) //e2
    vmovss(xmm7, mem(rdx, rax, 1)) //e3
    
    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
    
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)

    vmovups(xmm8, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)
    
    vunpcklps(xmm6, xmm4, xmm0)//e0f0e1f1
    vunpckhps(xmm6, xmm4, xmm1)//e2f2e3f3    
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//e1f1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//e3f3
    vmovsd(xmm0, mem(rcx)) //e0f0
    vmovsd(xmm5, mem(rcx, rsi, 1)) //e1f1
    vmovsd(xmm1, mem(rcx, rsi, 2)) //e2f2
    vmovsd(xmm7, mem(rcx, rax, 1)) //e3f3
    
    vshufps(imm(0x01), xmm8, xmm8,xmm1)
    vshufps(imm(0x02), xmm8, xmm8,xmm5)
    vshufps(imm(0x03), xmm8, xmm8,xmm7)    
    vmovss(xmm8, mem(rdx)) //e0
    vmovss(xmm1, mem(rdx, rsi, 1)) //e1
    vmovss(xmm5, mem(rdx, rsi, 2)) //e2
    vmovss(xmm7, mem(rdx, rax, 1)) //e3

    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_2x4
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()

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


    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
            
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//e0f0e1f1
    vunpckhps(xmm6, xmm4, xmm1)//e2f2e3f3    
    vmovsd(mem(rcx),xmm2)
    vmovsd(mem(rcx, rsi, 1),xmm4)    
    vmovsd(mem(rcx, rsi, 2),xmm6)
    vmovsd(mem(rcx, rax, 1),xmm10)
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//e1f1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//e3f3
    vfmadd231ps(xmm2, xmm3, xmm0)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vfmadd231ps(xmm10, xmm3, xmm7)
    vmovsd(xmm0, mem(rcx)) //e0f0
    vmovsd(xmm5, mem(rcx, rsi, 1)) //e1f1
    vmovsd(xmm1, mem(rcx, rsi, 2)) //e2f2
    vmovsd(xmm7, mem(rcx, rax, 1)) //e3f3

    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
        
    label(.SROWSTORBZ)
        
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)    
    vmovups(xmm6, mem(rcx))

    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORBZ)
    
    vunpcklps(xmm6, xmm4, xmm0)//e0f0e1f1
    vunpckhps(xmm6, xmm4, xmm1)//e2f2e3f3 
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//e1f1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//e3f3
    vmovsd(xmm0, mem(rcx)) //e0f0
    vmovsd(xmm5, mem(rcx, rsi, 1)) //e1f1
    vmovsd(xmm1, mem(rcx, rsi, 2)) //e2f2
    vmovsd(xmm7, mem(rcx, rax, 1)) //e3f3
        
    label(.SDONE)
    

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_1x4
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()

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


    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 0*8))         // prefetch c + 3*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
        
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
        
    label(.SPOSTACCUM)

    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha    
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
                                      // now avoid loading C if beta == 0
    
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)    
    
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
        
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vmovss(mem(rcx),xmm2)
    vmovss(mem(rcx, rsi, 1),xmm6)    
    vmovss(mem(rcx, rsi, 2),xmm8)
    vmovss(mem(rcx, rax, 1),xmm10)
    vshufps(imm(0x01), xmm4, xmm4,xmm1)
    vshufps(imm(0x02), xmm4, xmm4,xmm5)
    vshufps(imm(0x03), xmm4, xmm4,xmm7)    
    vfmadd231ps(xmm2, xmm3, xmm4)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vfmadd231ps(xmm8, xmm3, xmm5)
    vfmadd231ps(xmm10, xmm3, xmm7)
    vmovss(xmm4, mem(rcx)) //e0
    vmovss(xmm1, mem(rcx, rsi, 1)) //e1
    vmovss(xmm5, mem(rcx, rsi, 2)) //e2
    vmovss(xmm7, mem(rcx, rax, 1)) //e3

    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
    
    vmovups(xmm4, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)
    
    vshufps(imm(0x01), xmm4, xmm4,xmm1)
    vshufps(imm(0x02), xmm4, xmm4,xmm5)
    vshufps(imm(0x03), xmm4, xmm4,xmm7)    
    vmovss(xmm4, mem(rcx)) //e0
    vmovss(xmm1, mem(rcx, rsi, 1)) //e1
    vmovss(xmm5, mem(rcx, rsi, 2)) //e2
    vmovss(xmm7, mem(rcx, rax, 1)) //e3
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_6x2
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 1*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 1*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    

    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)
    

    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)    
    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
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
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)

    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    vmulps(xmm0, xmm12, xmm12)
    vmulps(xmm0, xmm14, xmm14)
        
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
                                      // now avoid loading C if beta == 0
    
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
    
    vmovsd(mem(rcx), xmm0)////a0a1
    vfmadd231ps(xmm0, xmm3, xmm4)//c*beta+(a0a1)
    vmovsd(xmm4, mem(rcx))//a0a1
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm6)
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmovsd(xmm8, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm10)
    vmovsd(xmm10, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm12)
    vmovsd(xmm12, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm14)
    vmovsd(xmm14, mem(rcx))
        
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklps(xmm14, xmm12, xmm2)//e0f0 
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rdi, 2),xmm6)    
    vmovsd(mem(rcx, rdi, 4),xmm8)
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//c1d1
    vshufpd(imm(0x01), xmm2, xmm2, xmm9)//e1f1
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vfmadd231ps(xmm8, xmm3, xmm2)
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm1, mem(rcx, rdi, 2)) //c0d0
    vmovsd(xmm2, mem(rcx, rdi, 4)) //e0f0
    lea(mem(rcx, rsi, 1), rcx)
     
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rdi, 2),xmm6)    
    vmovsd(mem(rcx, rdi, 4),xmm8)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm6, xmm3, xmm7)
    vfmadd231ps(xmm8, xmm3, xmm9)
    vmovsd(xmm5, mem(rcx)) //a1b1
    vmovsd(xmm7, mem(rcx, rdi, 2)) //c1d1
    vmovsd(xmm9, mem(rcx, rdi, 4)) //e1f1
    
    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    

    label(.SROWSTORBZ)
    
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(xmm8, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm10, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm12, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vunpcklps(xmm14, xmm12, xmm2)//e0f0
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//c1d1      
    vshufpd(imm(0x01), xmm2, xmm2, xmm9)//e1f1     
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm1, mem(rcx, rdi, 2)) //c0d0
    vmovsd(xmm2, mem(rcx, rdi, 4)) //e0f0
    lea(mem(rcx, rsi, 1), rcx)
    vmovsd(xmm5, mem(rcx)) //e0f0
    vmovsd(xmm7, mem(rcx, rdi, 2)) //e1f1
    vmovsd(xmm9, mem(rcx, rdi, 4)) //e0f0
    
    label(.SDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_5x2
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 1*8)) // prefetch c + 4*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
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
    vmovups(mem(rbx,  0*32), xmm0)
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
    vmovups(mem(rbx,  0*32), xmm0)
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
    vmovups(mem(rbx,  0*32), xmm0)
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
    
    vmovups(mem(rbx,  0*32), xmm0)
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

                                    // now avoid loading C if beta == 0
    
    vxorps(xmm0, xmm0, xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)

    vmovsd(mem(rcx), xmm0)////a0a1
    vfmadd231ps(xmm0, xmm3, xmm4)//c*beta+(a0a1)
    vmovsd(xmm4, mem(rcx))//a0a1
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm6)
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmovsd(xmm8, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm10)
    vmovsd(xmm10, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm12)
    vmovsd(xmm12, mem(rcx))
    add(rdi, rcx)
    
    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rdi, 2),xmm6)    
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//c1d1
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm1, mem(rcx, rdi, 2)) //c0d0

    vmovss(mem(rcx, rdi, 4),xmm4)
    vshufps(imm(0x01), xmm12, xmm12, xmm9)//e1
    vfmadd231ps(xmm4, xmm3, xmm12)
    vmovss(xmm12,mem(rcx,rdi,4))//e0
    
    lea(mem(rcx, rsi, 1), rcx)     
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rdi, 2),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm6, xmm3, xmm7)
    vmovsd(xmm5, mem(rcx)) //a1b1
    vmovsd(xmm7, mem(rcx, rdi, 2)) //c1d1
    
    vmovss( mem(rcx, rdi, 4),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm9)
    vmovss(xmm9,mem(rcx,rdi,4))//e1

    jmp(.SDONE)                        // jump to end.
    
        
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    
    label(.SROWSTORBZ)
    
    vmovsd(xmm4, mem(rcx))//a0a1
    add(rdi, rcx)
    
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm8, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(xmm10, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(xmm12, mem(rcx))
    add(rdi, rcx)

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//c1d1
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm1, mem(rcx, rdi, 2)) //c0d0
    vshufps(imm(0x01), xmm12, xmm12, xmm9)//e1
    vmovss(xmm12,mem(rcx,rdi,4))//e0
    
    lea(mem(rcx, rsi, 1), rcx)     
    vmovsd(xmm5, mem(rcx)) //a1b1
    vmovsd(xmm7, mem(rcx, rdi, 2)) //c1d1    
    vmovss(xmm9,mem(rcx,rdi,4))//e1
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_4x2
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()
    
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
    
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(rcx, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rcx, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)
    
    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)
    
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)
    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
        
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    
        
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
                                      // now avoid loading C if beta == 0
    
    vxorps(xmm0,xmm0,xmm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    

    
    label(.SROWSTORED)
    
    vmovsd(mem(rcx), xmm0)////a0a1
    vfmadd231ps(xmm0, xmm3, xmm4)//c*beta+(a0a1)
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm6)
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmovsd(xmm8, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm10)
    vmovsd(xmm10, mem(rcx))
    
    
    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rdi, 2),xmm6)    
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1    
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//c1d1
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm6, xmm3, xmm1)
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm1, mem(rcx, rdi, 2)) //c0d0

    lea(mem(rcx, rsi, 1), rcx)     
    vmovsd(mem(rcx),xmm4)
    vmovsd(mem(rcx, rdi, 2),xmm6)    
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm6, xmm3, xmm7)
    vmovsd(xmm5, mem(rcx)) //a1b1
    vmovsd(xmm7, mem(rcx, rdi, 2)) //c1d1

    jmp(.SDONE)                        // jump to end.
    
    
    label(.SBETAZERO)


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    

    label(.SROWSTORBZ)
    
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)
        
    vmovsd(xmm8, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm10, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)
    
    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1
    vshufpd(imm(0x01), xmm1, xmm1, xmm7)//c1d1
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm1, mem(rcx, rdi, 2)) //c0d0
    lea(mem(rcx, rsi, 1), rcx)
    vmovsd(xmm5, mem(rcx)) //a1b1
    vmovsd(xmm7, mem(rcx, rdi, 2)) //c1d1        
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_3x2
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()
    
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
    
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
    
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    label(.SLOOPKITER)                 // MAIN LOOP
        
    // ---------------------------------- iteration 0
    vmovups(mem(rbx, 0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)    
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx, 0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)    

    // ---------------------------------- iteration 2
    vmovups(mem(rbx, 0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)    

    // ---------------------------------- iteration 3
    vmovups(mem(rbx, 0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
        
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
    
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
        
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx, 0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    vbroadcastss(mem(rax, r8,  2), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm8)
    
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
    
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)    
    lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    
                                      // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    
    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    label(.SROWSTORED)
        
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)

    vfmadd231ps(mem(rcx), xmm3, xmm8)
    vmovsd(xmm8, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.
    
    label(.SCOLSTORED)
    
    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vmovsd(mem(rcx),xmm4)
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1
    vfmadd231ps(xmm4, xmm3, xmm0)
    vmovsd(xmm0, mem(rcx)) //a0b0    
    vmovss(mem(rcx,rdi,2),xmm4)
    vshufps(imm(0x01), xmm8, xmm8, xmm9)//c1
    vfmadd231ps(xmm4, xmm3, xmm8)
    vmovss(xmm8,mem(rcx,rdi,2))//c0

    lea(mem(rcx, rsi, 1), rcx)     
    vmovsd(mem(rcx),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vmovsd(xmm5, mem(rcx)) //a1b1
    
    vmovss(mem(rcx,rdi,2),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm9)
    vmovss(xmm9,mem(rcx,rdi,2))//c1    
    
    jmp(.SDONE)                        // jump to end.
    
    
    label(.SBETAZERO)
    

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)
        
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)

    vmovsd(xmm6, mem(rcx))
    add(rdi, rcx)

    vmovsd(xmm8, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1
    vmovsd(xmm0, mem(rcx)) //a0b0    
    vshufps(imm(0x01), xmm8, xmm8, xmm9)//c1
    vmovss(xmm8,mem(rcx,rdi,2))//c0
    lea(mem(rcx, rsi, 1), rcx)     
    vmovsd(xmm5, mem(rcx)) //a1b1
    vmovss(xmm9,mem(rcx,rdi,2))//c1    
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_2x2
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------

    begin_asm()
    
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
    
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)
    
    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b
    
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
        
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;
    
    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.    
    
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
                                      // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case
    
    label(.SROWSTORED)
        

    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)
        
    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovsd(xmm6, mem(rcx))
    
    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORED)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vmovsd(mem(rcx),xmm4)
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1
    vfmadd231ps(xmm4, xmm3, xmm0)
    vmovsd(xmm0, mem(rcx)) //a0b0
    lea(mem(rcx, rsi, 1), rcx)
    vmovsd(mem(rcx),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vmovsd(xmm5, mem(rcx)) //a1b1

    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
    
    label(.SROWSTORBZ)
    
    vmovsd(xmm4, mem(rcx))
    add(rdi, rcx)
    
    vmovsd(xmm6, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)
    
    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vshufpd(imm(0x01), xmm0, xmm0, xmm5)//a1b1
    vmovsd(xmm0, mem(rcx)) //a0b0
    vmovsd(xmm5, mem(rcx, rsi, 1)) //a1b1

    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

void bli_sgemmsup_rv_zen_asm_1x2
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
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // -------------------------------------------------------------------------
    begin_asm()
    
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
    
    mov(var(a), rax)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    mov(var(b), rbx)                   // load address of b.
    mov(var(rs_b), r10)                // load rs_b

    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
                                      // NOTE: We cannot pre-load elements of a or b
                                      // because it could eventually, in the last
                                      // unrolled iter or the cleanup loop, result
                                      // in reading beyond the bounds allocated mem
                                      // (the likely result: a segmentation fault).

    mov(var(c), rcx)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c

    label(.SPOSTPFETCH)                // done prefetching c
    
    
    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                      // contains the k_left loop.
    
    
    label(.SLOOPKITER)                 // MAIN LOOP
    
    // ---------------------------------- iteration 0
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    // ---------------------------------- iteration 1
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    
    // ---------------------------------- iteration 2
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)

    // ---------------------------------- iteration 3
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
    
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.
        
    
    label(.SCONSIDKLEFT)
    
    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                      // else, we prepare to enter k_left loop.
    
    label(.SLOOPKLEFT)                 // EDGE LOOP
    
    vmovups(mem(rbx,  0*32), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm4)
        
    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.
        
    label(.SPOSTACCUM)
    
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate
    
    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    
    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)
    
                                        // now avoid loading C if beta == 0
    
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case
    

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    label(.SROWSTORED)
    
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovsd(xmm4, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    vshufps(imm(0x01), xmm4, xmm4, xmm9)//c1
    vmovss(mem(rcx),xmm6)    
    vfmadd231ps(xmm6, xmm3, xmm4)
    vmovss(xmm4,mem(rcx))//c0
    vmovss(mem(rcx,rsi,1),xmm6)
    vfmadd231ps(xmm6, xmm3, xmm9)
    vmovss(xmm9,mem(rcx,rsi,1))//c1    
        
    jmp(.SDONE)                        // jump to end.
    
    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case
        
    label(.SROWSTORBZ)
    vmovsd(xmm4, mem(rcx))

    jmp(.SDONE)                        // jump to end.
    
    label(.SCOLSTORBZ)
    
    vshufps(imm(0x01), xmm4, xmm4, xmm9)//c1
    vmovss(xmm4,mem(rcx))//c0
    vmovss(xmm9,mem(rcx,rsi,1))//c1    
    
    label(.SDONE)
    
    end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
}

// -----------------------------------------------------------------------------

// NOTE: Normally, for any "?x1" kernel, we would call the reference kernel.
// However, at least one other subconfiguration (zen) uses this kernel set, so
// we need to be able to call a set of "?x1" kernels that we know will actually
// exist regardless of which subconfiguration these kernels were used by. Thus,
// the compromise employed here is to inline the reference kernel so it gets
// compiled as part of the zen kernel set, and hence can unconditionally be
// called by other kernels within that kernel set.
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mdim ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t              conja, \
       conj_t              conjb, \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict b, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict beta, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx \
     ) \
{ \
    for ( dim_t i = 0; i < mdim; ++i ) \
    { \
        ctype* restrict ci = &c[ i*rs_c ]; \
        ctype* restrict ai = &a[ i*rs_a ]; \
\
        /* for ( dim_t j = 0; j < 1; ++j ) */ \
        { \
            ctype* restrict cij = ci /*[ j*cs_c ]*/ ; \
            ctype* restrict bj  = b  /*[ j*cs_b ]*/ ; \
            ctype           ab; \
\
            PASTEMAC(ch,set0s)( ab ); \
\
            /* Perform a dot product to update the (i,j) element of c. */ \
            for ( dim_t l = 0; l < k; ++l ) \
            { \
                ctype* restrict aij = &ai[ l*cs_a ]; \
                ctype* restrict bij = &bj[ l*rs_b ]; \
\
                PASTEMAC(ch,dots)( *aij, *bij, ab ); \
            } \
\
            /* If beta is one, add ab into c. If beta is zero, overwrite c
              with the result in ab. Otherwise, scale by beta and accumulate
              ab to c. */ \
            if ( PASTEMAC(ch,eq1)( *beta ) ) \
            { \
                PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
            } \
            else if ( PASTEMAC(d,eq0)( *beta ) ) \
            { \
                PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
            } \
            else \
            { \
                PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
            } \
        } \
    } \
}

GENTFUNC( float, s, gemmsup_r_zen_ref_6x1, 6 )
GENTFUNC( float, s, gemmsup_r_zen_ref_5x1, 5 )
GENTFUNC( float, s, gemmsup_r_zen_ref_4x1, 4 )
GENTFUNC( float, s, gemmsup_r_zen_ref_3x1, 3 )
GENTFUNC( float, s, gemmsup_r_zen_ref_2x1, 2 )
GENTFUNC( float, s, gemmsup_r_zen_ref_1x1, 1 )

