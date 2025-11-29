/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


void bli_dgemmsup_rv_haswell_asm_6x6
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

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(xmm0, xmm11, xmm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(xmm0, xmm13, xmm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(xmm0, xmm15, xmm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm9)
	vmovupd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm11)
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm13)
	vmovupd(xmm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm15)
	vmovupd(xmm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(xmm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(xmm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)



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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
	  "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12",
	  "ymm13", "ymm14", "ymm15",
	  "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_5x6
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

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx,         4*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         4*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(xmm0, xmm11, xmm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(xmm0, xmm13, xmm13)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rax)         // load address of c +  1*rs_c;
	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;
	lea(mem(rbx, rdi, 1), r8)         // load address of c +  3*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)

	vfmadd231pd(mem(rax, 0*32), ymm3, ymm6)
	vfmadd231pd(mem(rax, 1*32), xmm3, xmm7)

	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm9)

	vfmadd231pd(mem(r8, 0*32), ymm3, ymm10)
	vfmadd231pd(mem(r8, 1*32), xmm3, xmm11)

	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm13)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))

	vmovupd(ymm6, mem(rax, 0*32))
	vmovupd(xmm7, mem(rax, 1*32))

	vmovupd(ymm8, mem(rbx, 0*32))
	vmovupd(xmm9, mem(rbx, 1*32))

	vmovupd(ymm10, mem(r8, 0*32))
	vmovupd(xmm11, mem(r8, 1*32))

	vmovupd(ymm12, mem(rdx, 0*32))
	vmovupd(xmm13, mem(rdx, 1*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovlpd(mem(rdx        ), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm12, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovlpd(mem(rdx        ), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)

	vfmadd213pd(xmm13, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(xmm13, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovupd(ymm12, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovupd(ymm13, ymm0)

	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)



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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
	  "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
	  "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_4x6
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

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(xmm0, xmm11, xmm11)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rax)         // load address of c +  1*rs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;
	lea(mem(rdx, rdi, 1), rbx)         // load address of c +  3*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)

	vfmadd231pd(mem(rax, 0*32), ymm3, ymm6)
	vfmadd231pd(mem(rax, 1*32), xmm3, xmm7)

	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm8)
	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm9)

	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm11)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))

	vmovupd(ymm6, mem(rax, 0*32))
	vmovupd(xmm7, mem(rax, 1*32))

	vmovupd(ymm8, mem(rdx, 0*32))
	vmovupd(xmm9, mem(rdx, 1*32))

	vmovupd(ymm10, mem(rbx, 0*32))
	vmovupd(xmm11, mem(rbx, 1*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(xmm11, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)




	label(.DDONE)



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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
	  "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_3x6
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

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx,         2*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         2*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm10)
	vfmadd231pd(ymm1, ymm2, ymm11)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm14)
	vfmadd231pd(ymm1, ymm2, ymm15)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm10)
	vfmadd231pd(ymm1, ymm2, ymm11)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm14)
	vfmadd231pd(ymm1, ymm2, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm10, ymm4, ymm4)
	vaddpd(ymm11, ymm5, ymm5)
	vaddpd(ymm12, ymm6, ymm6)
	vaddpd(ymm13, ymm7, ymm7)
	vaddpd(ymm14, ymm8, ymm8)
	vaddpd(ymm15, ymm9, ymm9)


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rbx)         // load address of c +  1*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)
	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm6)
	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm7)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm8)
	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm9)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	vmovupd(ymm6, mem(rbx, 0*32))
	vmovupd(xmm7, mem(rbx, 1*32))
	vmovupd(ymm8, mem(rdx, 0*32))
	vmovupd(xmm9, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx        ), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), xmm3, xmm5)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm7)
	//vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm9)
	//vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm11)
	vmovupd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx        ), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vmovupd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)

	vmovupd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)



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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
	  "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_2x6
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

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx,         1*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         1*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm12)
	vbroadcastsd(mem(rax, r8,  1), ymm13)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm12, ymm8)
	vfmadd231pd(ymm1, ymm12, ymm9)
	vfmadd231pd(ymm0, ymm13, ymm10)
	vfmadd231pd(ymm1, ymm13, ymm11)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm12)
	vbroadcastsd(mem(rax, r8,  1), ymm13)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm12, ymm8)
	vfmadd231pd(ymm1, ymm12, ymm9)
	vfmadd231pd(ymm0, ymm13, ymm10)
	vfmadd231pd(ymm1, ymm13, ymm11)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm8, ymm4, ymm4)
	vaddpd(ymm9, ymm5, ymm5)
	vaddpd(ymm10, ymm6, ymm6)
	vaddpd(ymm11, ymm7, ymm7)


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 1), rdx)         // load address of c +  1*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)
	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm6)
	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm7)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	vmovupd(ymm6, mem(rdx, 0*32))
	vmovupd(xmm7, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), xmm3, xmm5)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm7)
	vmovupd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vmovupd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)




	label(.DDONE)



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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
	  "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "memory"
	)
}

void bli_dgemmsup_rv_haswell_asm_1x6
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

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx,         0*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         0*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 4*8))
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm6, ymm4, ymm4)
	vaddpd(ymm7, ymm5, ymm5)


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vmovlpd(mem(rcx        ), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm4, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-5
	vmovlpd(mem(rcx        ), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)

	vfmadd213pd(xmm5, xmm3, xmm0)

	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vmovupd(ymm4, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-5
	vmovupd(xmm5, xmm0)

	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)




	label(.DDONE)



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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "memory"
	)
}

