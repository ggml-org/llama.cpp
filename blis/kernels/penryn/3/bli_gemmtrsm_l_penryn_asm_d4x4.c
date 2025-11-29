/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#if 0
void bli_sgemmtrsm_l_penryn_asm_8x4
     (
     )
{
}
#endif

void bli_dgemmtrsm_l_penryn_asm_4x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k0,
       const void*      alpha,
       const void*      a10,
       const void*      a11,
       const void*      b01,
             void*      b11,
             void*      c11, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const void* b_next  = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMMTRSM_UKR_SETUP_CT( d, 4, 4, false );

	begin_asm()

		mov(var(a10), rax) // load address of a10.
		mov(var(b01), rbx) // load address of b01.
		//mov(var(b_next), r9) // load address of b_next.

		sub(imm(0-8*16), rax) // increment pointers to allow byte
		sub(imm(0-8*16), rbx) // offsets in the unrolled iterations.

		movaps(mem(rax, -8*16), xmm0) // initialize loop by pre-loading elements
		movaps(mem(rax, -7*16), xmm1) // of a and b.
		movaps(mem(rbx, -8*16), xmm2)

		//mov(var(c11), rcx) // load address of c11
		//mov(var(rs_c), rdi) // load cs_c
		//lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(double)
		//lea(mem(rcx, rdi, 2), rdx) // load address of c + 2*cs_c;

		//prefetch(2, mem(r9, 0*8)) // prefetch b_next

		xorpd(xmm3, xmm3)
		xorpd(xmm4, xmm4)
		xorpd(xmm5, xmm5)
		xorpd(xmm6, xmm6)

		//prefetch(2, mem(rcx, 3*8)) // prefetch c + 0*cs_c
		xorpd(xmm8, xmm8)
		movaps(xmm8, xmm9)
		//prefetch(2, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*cs_c
		movaps(xmm8, xmm10)
		movaps(xmm8, xmm11)
		//prefetch(2, mem(rdx, 3*8)) // prefetch c + 2*cs_c
		movaps(xmm8, xmm12)
		movaps(xmm8, xmm13)
		//prefetch(2, mem(rdx, rdi, 1, 3*8)) // prefetch c + 3*cs_c
		movaps(xmm8, xmm14)
		movaps(xmm8, xmm15)



		mov(var(k_iter), rsi) // i = k_iter;
		test(rsi, rsi) // check i via logical AND.
		je(.CONSIDERKLEFT) // if i == 0, jump to code that
		 // contains the k_left loop.


		label(.LOOPKITER) // MAIN LOOP

		//prefetch(0, mem(rax, 1264))
		prefetch(0, mem(rax, (4*35+1)*8))

		addpd(xmm3, xmm11) // iteration 0
		movaps(mem(rbx, -7*16), xmm3)
		addpd(xmm4, xmm15)
		movaps(xmm2, xmm4)
		pshufd(imm(0x4e), xmm2, xmm7)
		mulpd(xmm0, xmm2)
		mulpd(xmm1, xmm4)

		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)
		movaps(xmm7, xmm6)
		mulpd(xmm0, xmm7)
		mulpd(xmm1, xmm6)

		addpd(xmm2, xmm9)
		movaps(mem(rbx, -6*16), xmm2)
		addpd(xmm4, xmm13)
		movaps(xmm3, xmm4)
		pshufd(imm(0x4e), xmm3, xmm5)
		mulpd(xmm0, xmm3)
		mulpd(xmm1, xmm4)

		addpd(xmm7, xmm8)
		addpd(xmm6, xmm12)
		movaps(xmm5, xmm6)
		mulpd(xmm0, xmm5)
		movaps(mem(rax, -6*16), xmm0)
		mulpd(xmm1, xmm6)
		movaps(mem(rax, -5*16), xmm1)


		addpd(xmm3, xmm11) // iteration 1
		movaps(mem(rbx, -5*16), xmm3)
		addpd(xmm4, xmm15)
		movaps(xmm2, xmm4)
		pshufd(imm(0x4e), xmm2, xmm7)
		mulpd(xmm0, xmm2)
		mulpd(xmm1, xmm4)

		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)
		movaps(xmm7, xmm6)
		mulpd(xmm0, xmm7)
		mulpd(xmm1, xmm6)

		addpd(xmm2, xmm9)
		movaps(mem(rbx, -4*16), xmm2)
		addpd(xmm4, xmm13)
		movaps(xmm3, xmm4)
		pshufd(imm(0x4e), xmm3, xmm5)
		mulpd(xmm0, xmm3)
		mulpd(xmm1, xmm4)

		addpd(xmm7, xmm8)
		addpd(xmm6, xmm12)
		movaps(xmm5, xmm6)
		mulpd(xmm0, xmm5)
		movaps(mem(rax, -4*16), xmm0)
		mulpd(xmm1, xmm6)
		movaps(mem(rax, -3*16), xmm1)

		//prefetch(0, mem(rax, 1328))
		prefetch(0, mem(rax, (4*37+1)*8))

		addpd(xmm3, xmm11) // iteration 2
		movaps(mem(rbx, -3*16), xmm3)
		addpd(xmm4, xmm15)
		movaps(xmm2, xmm4)
		pshufd(imm(0x4e), xmm2, xmm7)
		mulpd(xmm0, xmm2)
		mulpd(xmm1, xmm4)

		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)
		movaps(xmm7, xmm6)
		mulpd(xmm0, xmm7)
		mulpd(xmm1, xmm6)

		addpd(xmm2, xmm9)
		movaps(mem(rbx, -2*16), xmm2)
		addpd(xmm4, xmm13)
		movaps(xmm3, xmm4)
		pshufd(imm(0x4e), xmm3, xmm5)
		mulpd(xmm0, xmm3)
		mulpd(xmm1, xmm4)

		addpd(xmm7, xmm8)
		addpd(xmm6, xmm12)
		movaps(xmm5, xmm6)
		mulpd(xmm0, xmm5)
		movaps(mem(rax, -2*16), xmm0)
		mulpd(xmm1, xmm6)
		movaps(mem(rax, -1*16), xmm1)


		addpd(xmm3, xmm11) // iteration 3
		movaps(mem(rbx, -1*16), xmm3)
		addpd(xmm4, xmm15)
		movaps(xmm2, xmm4)
		pshufd(imm(0x4e), xmm2, xmm7)
		mulpd(xmm0, xmm2)
		mulpd(xmm1, xmm4)

		sub(imm(0-4*4*8), rax) // a += 4*4 (unroll x mr)

		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)
		movaps(xmm7, xmm6)
		mulpd(xmm0, xmm7)
		mulpd(xmm1, xmm6)

		//sub(imm(-4*4*8), r9) // b_next += 4*4 (unroll x nr)

		addpd(xmm2, xmm9)
		movaps(mem(rbx, 0*16), xmm2)
		addpd(xmm4, xmm13)
		movaps(xmm3, xmm4)
		pshufd(imm(0x4e), xmm3, xmm5)
		mulpd(xmm0, xmm3)
		mulpd(xmm1, xmm4)

		sub(imm(0-4*4*8), rbx) // b += 4*4 (unroll x nr)

		addpd(xmm7, xmm8)
		addpd(xmm6, xmm12)
		movaps(xmm5, xmm6)
		mulpd(xmm0, xmm5)
		movaps(mem(rax, -8*16), xmm0)
		mulpd(xmm1, xmm6)
		movaps(mem(rax, -7*16), xmm1)

		//prefetch(2, mem(r9, 0*8)) // prefetch b_next[0]
		//prefetch(2, mem(r9, 8*8)) // prefetch b_next[8]


		dec(rsi) // i -= 1;
		jne(.LOOPKITER) // iterate again if i != 0.



		label(.CONSIDERKLEFT)

		mov(var(k_left), rsi) // i = k_left;
		test(rsi, rsi) // check i via logical AND.
		je(.POSTACCUM) // if i == 0, we're done; jump to end.
		 // else, we prepare to enter k_left loop.


		label(.LOOPKLEFT) // EDGE LOOP

		addpd(xmm3, xmm11) // iteration 0
		movaps(mem(rbx, -7*16), xmm3)
		addpd(xmm4, xmm15)
		movaps(xmm2, xmm4)
		pshufd(imm(0x4e), xmm2, xmm7)
		mulpd(xmm0, xmm2)
		mulpd(xmm1, xmm4)

		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)
		movaps(xmm7, xmm6)
		mulpd(xmm0, xmm7)
		mulpd(xmm1, xmm6)

		addpd(xmm2, xmm9)
		movaps(mem(rbx, -6*16), xmm2)
		addpd(xmm4, xmm13)
		movaps(xmm3, xmm4)
		pshufd(imm(0x4e), xmm3, xmm5)
		mulpd(xmm0, xmm3)
		mulpd(xmm1, xmm4)

		addpd(xmm7, xmm8)
		addpd(xmm6, xmm12)
		movaps(xmm5, xmm6)
		mulpd(xmm0, xmm5)
		movaps(mem(rax, -6*16), xmm0)
		mulpd(xmm1, xmm6)
		movaps(mem(rax, -5*16), xmm1)


		sub(imm(0-4*1*8), rax) // a += 4 (1 x mr)
		sub(imm(0-4*1*8), rbx) // b += 4 (1 x nr)


		dec(rsi) // i -= 1;
		jne(.LOOPKLEFT) // iterate again if i != 0.



		label(.POSTACCUM)

		addpd(xmm3, xmm11)
		addpd(xmm4, xmm15)
		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)



		mov(var(b11), rbx) // load address of b11.

		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab01  ( ab00  ( ab03  ( ab02
		 //   ab10 )  ab11 )  ab12 )  ab13 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab21  ( ab20  ( ab23  ( ab22
		 //   ab30 )  ab31 )  ab32 )  ab33 )
		movaps(xmm9, xmm0)
		movaps(xmm8, xmm1)
		unpcklpd(xmm8, xmm0)
		unpckhpd(xmm9, xmm1)

		movaps(xmm11, xmm4)
		movaps(xmm10, xmm5)
		unpcklpd(xmm10, xmm4)
		unpckhpd(xmm11, xmm5)

		movaps(xmm13, xmm2)
		movaps(xmm12, xmm3)
		unpcklpd(xmm12, xmm2)
		unpckhpd(xmm13, xmm3)

		movaps(xmm15, xmm6)
		movaps(xmm14, xmm7)
		unpcklpd(xmm14, xmm6)
		unpckhpd(xmm15, xmm7)

		 // xmm0: ( ab00 ab01 ) xmm4: ( ab02 ab03 )
		 // xmm1: ( ab10 ab11 ) xmm5: ( ab12 ab13 )
		 // xmm2: ( ab20 ab21 ) xmm6: ( ab22 ab23 )
		 // xmm3: ( ab30 ab31 ) xmm7: ( ab32 ab33 )

		mov(var(alpha), rax) // load address of alpha
		movddup(mem(rax), xmm15) // load alpha and duplicate

		movaps(mem(rbx, 0*16), xmm8)
		movaps(mem(rbx, 1*16), xmm12)
		mulpd(xmm15, xmm8) // xmm8  = alpha * ( beta00 beta01 )
		mulpd(xmm15, xmm12) // xmm12 = alpha * ( beta02 beta03 )
		movaps(mem(rbx, 2*16), xmm9)
		movaps(mem(rbx, 3*16), xmm13)
		mulpd(xmm15, xmm9) // xmm9  = alpha * ( beta10 beta11 )
		mulpd(xmm15, xmm13) // xmm13 = alpha * ( beta12 beta13 )
		movaps(mem(rbx, 4*16), xmm10)
		movaps(mem(rbx, 5*16), xmm14)
		mulpd(xmm15, xmm10) // xmm10 = alpha * ( beta20 beta21 )
		mulpd(xmm15, xmm14) // xmm14 = alpha * ( beta22 beta23 )
		movaps(mem(rbx, 6*16), xmm11)
		mulpd(xmm15, xmm11) // xmm11 = alpha * ( beta30 beta31 )
		mulpd(mem(rbx, 7*16), xmm15) // xmm15 = alpha * ( beta32 beta33 )

		 // (Now scaled by alpha:)
		 // xmm8:  ( beta00 beta01 ) xmm12: ( beta02 beta03 )
		 // xmm9:  ( beta10 beta11 ) xmm13: ( beta12 beta13 )
		 // xmm10: ( beta20 beta21 ) xmm14: ( beta22 beta23 )
		 // xmm11: ( beta30 beta31 ) xmm15: ( beta32 beta33 )

		subpd(xmm0, xmm8) // xmm8  -= xmm0
		subpd(xmm1, xmm9) // xmm9  -= xmm1
		subpd(xmm2, xmm10) // xmm10 -= xmm2
		subpd(xmm3, xmm11) // xmm11 -= xmm3
		subpd(xmm4, xmm12) // xmm12 -= xmm4
		subpd(xmm5, xmm13) // xmm13 -= xmm5
		subpd(xmm6, xmm14) // xmm14 -= xmm6
		subpd(xmm7, xmm15) // xmm15 -= xmm7



		label(.TRSM)


		mov(var(a11), rax) // load address of a11
		mov(var(c11), rcx) // load address of c11

		mov(var(rs_c), rsi) // load rs_c
		mov(var(cs_c), rdi) // load cs_c
		sal(imm(3), rsi) // rs_c *= sizeof( double )
		sal(imm(3), rdi) // cs_c *= sizeof( double )

		lea(mem(rcx, rdi, 2), rdx) // c11_2 = c11 + 2*cs_c



		 // iteration 0

		movddup(mem(0+0*4)*8(rax), xmm0) // load xmm0 = (1/alpha00)

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		mulpd(xmm0, xmm8)  // xmm8  *= (1/alpha00);
		mulpd(xmm0, xmm12) // xmm12 *= (1/alpha00);
#else
		divpd(xmm0, xmm8)  // xmm8  /= alpha00;
		divpd(xmm0, xmm12) // xmm12 /= alpha00;
#endif

		movaps(xmm8, mem(rbx, 0*16)) // store ( beta00 beta01 ) = xmm8
		movaps(xmm12, mem(rbx, 1*16)) // store ( beta02 beta03 ) = xmm12
		movlpd(xmm8, mem(rcx)) // store ( gamma00 ) = xmm8[0]
		movhpd(xmm8, mem(rcx, rdi, 1)) // store ( gamma01 ) = xmm8[1]
		movlpd(xmm12, mem(rdx)) // store ( gamma02 ) = xmm12[0]
		movhpd(xmm12, mem(rdx, rdi, 1)) // store ( gamma03 ) = xmm12[1]
		add(rsi, rcx) // c11   += rs_c
		add(rsi, rdx) // c11_2 += rs_c



		 // iteration 1

		movddup(mem(1+0*4)*8(rax), xmm0) // load xmm0 = alpha10
		movddup(mem(1+1*4)*8(rax), xmm1) // load xmm1 = (1/alpha11)

		movaps(xmm0, xmm4) // xmm4 = xmm0
		mulpd(xmm8, xmm0) // xmm0 = alpha10 * ( beta00 beta01 )
		mulpd(xmm12, xmm4) // xmm4 = alpha10 * ( beta02 beta03 )
		subpd(xmm0, xmm9) // xmm9  -= xmm0
		subpd(xmm4, xmm13) // xmm13 -= xmm4
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		mulpd(xmm1, xmm9)  // xmm9  *= (1/alpha11);
		mulpd(xmm1, xmm13) // xmm13 *= (1/alpha11);
#else
		divpd(xmm1, xmm9)  // xmm9  /= alpha11;
		divpd(xmm1, xmm13) // xmm13 /= alpha11;
#endif

		movaps(xmm9, mem(rbx, 2*16)) // store ( beta10 beta11 ) = xmm9
		movaps(xmm13, mem(rbx, 3*16)) // store ( beta12 beta13 ) = xmm13
		movlpd(xmm9, mem(rcx)) // store ( gamma10 ) = xmm9[0]
		movhpd(xmm9, mem(rcx, rdi, 1)) // store ( gamma11 ) = xmm9[1]
		movlpd(xmm13, mem(rdx)) // store ( gamma12 ) = xmm13[0]
		movhpd(xmm13, mem(rdx, rdi, 1)) // store ( gamma13 ) = xmm13[1]
		add(rsi, rcx) // c11   += rs_c
		add(rsi, rdx) // c11_2 += rs_c



		 // iteration 2

		movddup(mem(2+0*4)*8(rax), xmm0) // load xmm0 = alpha20
		movddup(mem(2+1*4)*8(rax), xmm1) // load xmm1 = alpha21
		movddup(mem(2+2*4)*8(rax), xmm2) // load xmm2 = (1/alpha22)

		movaps(xmm0, xmm4) // xmm4 = xmm0
		movaps(xmm1, xmm5) // xmm5 = xmm1
		mulpd(xmm8, xmm0) // xmm0 = alpha20 * ( beta00 beta01 )
		mulpd(xmm12, xmm4) // xmm4 = alpha20 * ( beta02 beta03 )
		mulpd(xmm9, xmm1) // xmm1 = alpha21 * ( beta10 beta11 )
		mulpd(xmm13, xmm5) // xmm5 = alpha21 * ( beta12 beta13 )
		addpd(xmm1, xmm0) // xmm0 += xmm1;
		addpd(xmm5, xmm4) // xmm4 += xmm5;
		subpd(xmm0, xmm10) // xmm10 -= xmm0
		subpd(xmm4, xmm14) // xmm14 -= xmm4
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		mulpd(xmm2, xmm10) // xmm10 *= (1/alpha22);
		mulpd(xmm2, xmm14) // xmm14 *= (1/alpha22);
#else
		divpd(xmm2, xmm10) // xmm10 /= alpha22;
		divpd(xmm2, xmm14) // xmm14 /= alpha22;
#endif

		movaps(xmm10, mem(rbx, 4*16)) // store ( beta20 beta21 ) = xmm10
		movaps(xmm14, mem(rbx, 5*16)) // store ( beta22 beta23 ) = xmm14
		movlpd(xmm10, mem(rcx)) // store ( gamma20 ) = xmm10[0]
		movhpd(xmm10, mem(rcx, rdi, 1)) // store ( gamma21 ) = xmm10[1]
		movlpd(xmm14, mem(rdx)) // store ( gamma22 ) = xmm14[0]
		movhpd(xmm14, mem(rdx, rdi, 1)) // store ( gamma23 ) = xmm14[1]
		add(rsi, rcx) // c11   += rs_c
		add(rsi, rdx) // c11_2 += rs_c



		 // iteration 3

		movddup(mem(3+0*4)*8(rax), xmm0) // load xmm0 = alpha30
		movddup(mem(3+1*4)*8(rax), xmm1) // load xmm1 = alpha31
		movddup(mem(3+2*4)*8(rax), xmm2) // load xmm2 = alpha32
		movddup(mem(3+3*4)*8(rax), xmm3) // load xmm3 = (1/alpha33)

		movaps(xmm0, xmm4) // xmm4 = xmm0
		movaps(xmm1, xmm5) // xmm5 = xmm1
		movaps(xmm2, xmm6) // xmm6 = xmm2
		mulpd(xmm8, xmm0) // xmm0 = alpha30 * ( beta00 beta01 )
		mulpd(xmm12, xmm4) // xmm4 = alpha30 * ( beta02 beta03 )
		mulpd(xmm9, xmm1) // xmm1 = alpha31 * ( beta10 beta11 )
		mulpd(xmm13, xmm5) // xmm5 = alpha31 * ( beta12 beta13 )
		mulpd(xmm10, xmm2) // xmm2 = alpha32 * ( beta20 beta21 )
		mulpd(xmm14, xmm6) // xmm6 = alpha32 * ( beta22 beta23 )
		addpd(xmm1, xmm0) // xmm0 += xmm1;
		addpd(xmm5, xmm4) // xmm4 += xmm5;
		addpd(xmm2, xmm0) // xmm0 += xmm2;
		addpd(xmm6, xmm4) // xmm4 += xmm6;
		subpd(xmm0, xmm11) // xmm11 -= xmm0
		subpd(xmm4, xmm15) // xmm15 -= xmm4
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		mulpd(xmm3, xmm11) // xmm11 *= (1/alpha33);
		mulpd(xmm3, xmm15) // xmm15 *= (1/alpha33);
#else
		divpd(xmm3, xmm11) // xmm11 /= alpha33;
		divpd(xmm3, xmm15) // xmm15 /= alpha33;
#endif

		movaps(xmm11, mem(rbx, 6*16)) // store ( beta30 beta31 ) = xmm11
		movaps(xmm15, mem(rbx, 7*16)) // store ( beta32 beta33 ) = xmm15
		movlpd(xmm11, mem(rcx)) // store ( gamma30 ) = xmm11[0]
		movhpd(xmm11, mem(rcx, rdi, 1)) // store ( gamma31 ) = xmm11[1]
		movlpd(xmm15, mem(rdx)) // store ( gamma32 ) = xmm15[0]
		movhpd(xmm15, mem(rdx, rdi, 1)) // store ( gamma33 ) = xmm15[1]




    end_asm(
		: // output operands (none)
		: // input operands
          [k_iter] "m" (k_iter), // 0
          [k_left] "m" (k_left), // 1
          [a10]    "m" (a10),    // 2
          [a11]    "m" (a11),    // 3
          [b01]    "m" (b01),    // 4
          [b11]    "m" (b11),    // 5
          [c11]    "m" (c11),    // 6
          [rs_c]   "m" (rs_c),   // 7
          [cs_c]   "m" (cs_c),   // 8
          [alpha]  "m" (alpha),  // 9
          [b_next] "m" (b_next)  // 10
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", //"r8", "r9", "r10",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	)

	GEMMTRSM_UKR_FLUSH_CT( d );
}


