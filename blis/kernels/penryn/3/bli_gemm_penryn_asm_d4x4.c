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

void bli_sgemm_penryn_asm_8x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	//const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT_ALIGNED( s, 8, 4, false, 16 );

	begin_asm()


		mov(var(a), rax) // load address of a.
		mov(var(b), rbx) // load address of b.
		mov(var(b_next), r9) // load address of b_next.

		sub(imm(0-8*16), rax) // increment pointers to allow byte
		sub(imm(0-8*16), rbx) // offsets in the unrolled iterations.

		movaps(mem(rax, -8*16), xmm0) // initialize loop by pre-loading elements
		movaps(mem(rax, -7*16), xmm1) // of a and b.
		movaps(mem(rbx, -8*16), xmm2)

		mov(var(c), rcx) // load address of c
		mov(var(cs_c), rdi) // load cs_c
		lea(mem(, rdi, 4), rdi) // cs_c *= sizeof(float)
		mov(rdi, r12) // make a copy of cs_c (in bytes)
		lea(mem(rcx, rdi, 2), r10) // load address of c + 2*cs_c;

		prefetch(2, mem(r9, 0*4)) // prefetch b_next

		xorps(xmm3, xmm3)
		xorps(xmm4, xmm4)
		xorps(xmm5, xmm5)
		xorps(xmm6, xmm6)

		prefetch(2, mem(rcx, 6*4)) // prefetch c + 0*cs_c
		xorps(xmm8, xmm8)
		xorps(xmm9, xmm9)
		prefetch(2, mem(rcx, rdi, 1, 6*4)) // prefetch c + 1*cs_c
		xorps(xmm10, xmm10)
		xorps(xmm11, xmm11)
		prefetch(2, mem(r10, 6*4)) // prefetch c + 2*cs_c
		xorps(xmm12, xmm12)
		xorps(xmm13, xmm13)
		prefetch(2, mem(r10, rdi, 1, 6*4)) // prefetch c + 3*cs_c
		xorps(xmm14, xmm14)
		xorps(xmm15, xmm15)



		mov(var(k_iter), rsi) // i = k_iter;
		test(rsi, rsi) // check i via logical AND.
		je(.SCONSIDKLEFT) // if i == 0, jump to code that
		 // contains the k_left loop.


		label(.SLOOPKITER) // MAIN LOOP

		prefetch(0, mem(rax, (4*35+1)*8))

		addps(xmm6, xmm10) // iteration 0
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)

		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)

		addps(xmm2, xmm8)
		movaps(mem(rbx, -7*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)

		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -6*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -5*16), xmm1)


		addps(xmm6, xmm10) // iteration 1
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)

		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)

		addps(xmm2, xmm8)
		movaps(mem(rbx, -6*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)

		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -4*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -3*16), xmm1)


		addps(xmm6, xmm10) // iteration 2
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)

		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)

		addps(xmm2, xmm8)
		movaps(mem(rbx, -5*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)

		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -2*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -1*16), xmm1)


		addps(xmm6, xmm10) // iteration 3
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)

		sub(imm(0-4*8*4), rax) // a += 4*8 (unroll x mr)

		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)

		sub(imm(0-4*4*4), r9) // b_next += 4*4 (unroll x nr)

		addps(xmm2, xmm8)
		movaps(mem(rbx, -4*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)

		sub(imm(0-4*4*4), rbx) // b += 4*4 (unroll x nr)

		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -8*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -7*16), xmm1)

		prefetch(2, mem(r9, 0*4)) // prefetch b_next[0]
		prefetch(2, mem(r9, 16*4)) // prefetch b_next[16]


		dec(rsi) // i -= 1;
		jne(.SLOOPKITER) // iterate again if i != 0.



		label(.SCONSIDKLEFT)

		mov(var(k_left), rsi) // i = k_left;
		test(rsi, rsi) // check i via logical AND.
		je(.SPOSTACCUM) // if i == 0, we're done; jump to end.
		 // else, we prepare to enter k_left loop.


		label(.SLOOPKLEFT) // EDGE LOOP

		addps(xmm6, xmm10) // iteration 0
		addps(xmm3, xmm14)
		movaps(xmm2, xmm3)
		pshufd(imm(0x39), xmm2, xmm7)
		mulps(xmm0, xmm2)
		mulps(xmm1, xmm3)

		addps(xmm4, xmm11)
		addps(xmm5, xmm15)
		movaps(xmm7, xmm5)
		pshufd(imm(0x39), xmm7, xmm6)
		mulps(xmm0, xmm7)
		mulps(xmm1, xmm5)

		addps(xmm2, xmm8)
		movaps(mem(rbx, -7*16), xmm2)
		addps(xmm3, xmm12)
		movaps(xmm6, xmm3)
		pshufd(imm(0x39), xmm6, xmm4)
		mulps(xmm0, xmm6)
		mulps(xmm1, xmm3)

		addps(xmm7, xmm9)
		addps(xmm5, xmm13)
		movaps(xmm4, xmm5)
		mulps(xmm0, xmm4)
		movaps(mem(rax, -6*16), xmm0)
		mulps(xmm1, xmm5)
		movaps(mem(rax, -5*16), xmm1)

		sub(imm(0-1*8*4), rax) // a += 8 (1 x mr)
		sub(imm(0-1*4*4), rbx) // b += 4 (1 x nr)


		dec(rsi) // i -= 1;
		jne(.SLOOPKLEFT) // iterate again if i != 0.



		label(.SPOSTACCUM)

		addps(xmm6, xmm10)
		addps(xmm3, xmm14)
		addps(xmm4, xmm11)
		addps(xmm5, xmm15)


		mov(var(alpha), rax) // load address of alpha
		mov(var(beta), rbx) // load address of beta
		movss(mem(rax), xmm6) // load alpha to bottom 4 bytes of xmm6
		movss(mem(rbx), xmm7) // load beta to bottom 4 bytes of xmm7
		pshufd(imm(0x00), xmm6, xmm6) // populate xmm6 with four alphas
		pshufd(imm(0x00), xmm7, xmm7) // populate xmm7 with four betas


		mov(var(rs_c), rsi) // load rs_c
		mov(rsi, r8) // make a copy of rs_c

		lea(mem(, rsi, 4), rsi) // rsi = rs_c * sizeof(float)
		lea(mem(rsi, rsi, 2), r11) // r11 = 3*(rs_c * sizeof(float))

		lea(mem(rcx, rsi, 4), rdx) // load address of c + 4*rs_c;

		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab00  ( ab01  ( ab02  ( ab03
		 //   ab11    ab12    ab13    ab10
		 //   ab22    ab23    ab20    ab21
		 //   ab33 )  ab30 )  ab31 )  ab32 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab40  ( ab41  ( ab42  ( ab43
		 //   ab51    ab52    ab53    ab50
		 //   ab62    ab63    ab60    ab61
		 //   ab73 )  ab70 )  ab71 )  ab72 )
		movaps(xmm9, xmm4)
		shufps(imm(0xd8), xmm8, xmm9)
		shufps(imm(0xd8), xmm11, xmm8)
		shufps(imm(0xd8), xmm10, xmm11)
		shufps(imm(0xd8), xmm4, xmm10)

		movaps(xmm8, xmm4)
		shufps(imm(0xd8), xmm10, xmm8)
		shufps(imm(0xd8), xmm4, xmm10)
		movaps(xmm9, xmm5)
		shufps(imm(0xd8), xmm11, xmm9)
		shufps(imm(0xd8), xmm5, xmm11)

		movaps(xmm13, xmm4)
		shufps(imm(0xd8), xmm12, xmm13)
		shufps(imm(0xd8), xmm15, xmm12)
		shufps(imm(0xd8), xmm14, xmm15)
		shufps(imm(0xd8), xmm4, xmm14)

		movaps(xmm12, xmm4)
		shufps(imm(0xd8), xmm14, xmm12)
		shufps(imm(0xd8), xmm4, xmm14)
		movaps(xmm13, xmm5)
		shufps(imm(0xd8), xmm15, xmm13)
		shufps(imm(0xd8), xmm5, xmm15)
		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab00  ( ab01  ( ab02  ( ab03
		 //   ab10    ab11    ab12    ab13
		 //   ab20    ab21    ab22    ab23
		 //   ab30 )  ab31 )  ab32 )  ab33 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab40  ( ab41  ( ab42  ( ab43
		 //   ab50    ab51    ab52    ab53
		 //   ab60    ab61    ab62    ab63
		 //   ab70 )  ab71 )  ab72 )  ab73 )

		 // now avoid loading C if beta == 0

		xorpd(xmm0, xmm0) // set xmm0 to zero.
		ucomisd(xmm0, xmm7) // check if beta == 0.
		je(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

			movaps(mem(rcx), xmm0) // load c00 ~ c30,
			mulps(xmm6, xmm8) // scale by alpha,
			mulps(xmm7, xmm0) // scale by beta,
			addps(xmm8, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.
			add(rdi, rcx)

			movaps(mem(rdx), xmm1) // load c40 ~ c70,
			mulps(xmm6, xmm12) // scale by alpha,
			mulps(xmm7, xmm1) // scale by beta,
			addps(xmm12, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.
			add(rdi, rdx)



			movaps(mem(rcx), xmm0) // load c01 ~ c31,
			mulps(xmm6, xmm9) // scale by alpha,
			mulps(xmm7, xmm0) // scale by beta,
			addps(xmm9, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.
			add(rdi, rcx)

			movaps(mem(rdx), xmm1) // load c41 ~ c71,
			mulps(xmm6, xmm13) // scale by alpha,
			mulps(xmm7, xmm1) // scale by beta,
			addps(xmm13, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.
			add(rdi, rdx)



			movaps(mem(rcx), xmm0) // load c02 ~ c32,
			mulps(xmm6, xmm10) // scale by alpha,
			mulps(xmm7, xmm0) // scale by beta,
			addps(xmm10, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.
			add(rdi, rcx)

			movaps(mem(rdx), xmm1) // load c42 ~ c72,
			mulps(xmm6, xmm14) // scale by alpha,
			mulps(xmm7, xmm1) // scale by beta,
			addps(xmm14, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.
			add(rdi, rdx)



			movaps(mem(rcx), xmm0) // load c03 ~ c33,
			mulps(xmm6, xmm11) // scale by alpha,
			mulps(xmm7, xmm0) // scale by beta,
			addps(xmm11, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.


			movaps(mem(rdx), xmm1) // load c43 ~ c73,
			mulps(xmm6, xmm15) // scale by alpha,
			mulps(xmm7, xmm1) // scale by beta,
			addps(xmm15, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.

			jmp(.SDONE) // jump to end.

		label(.SBETAZERO)

			 // skip loading c00 ~ c30,
			mulps(xmm6, xmm8) // scale by alpha,
			movaps(xmm8, mem(rcx)) // and store back to memory.
			add(rdi, rcx)
			 // skip loading c40 ~ c70,
			mulps(xmm6, xmm12) // scale by alpha,
			movaps(xmm12, mem(rdx)) // and store back to memory.
			add(rdi, rdx)


			 // skip loading c01 ~ c31,
			mulps(xmm6, xmm9) // scale by alpha,
			movaps(xmm9, mem(rcx)) // and store back to memory.
			add(rdi, rcx)
			 // skip loading c41 ~ c71,
			mulps(xmm6, xmm13) // scale by alpha,
			movaps(xmm13, mem(rdx)) // and store back to memory.
			add(rdi, rdx)


			 // skip loading c02 ~ c32,
			mulps(xmm6, xmm10) // scale by alpha,
			movaps(xmm10, mem(rcx)) // and store back to memory.
			add(rdi, rcx)
			 // skip loading c42 ~ c72,
			mulps(xmm6, xmm14) // scale by alpha,
			movaps(xmm14, mem(rdx)) // and store back to memory.
			add(rdi, rdx)


			 // skip loading c03 ~ c33,
			mulps(xmm6, xmm11) // scale by alpha,
			movaps(xmm11, mem(rcx)) // and store back to memory.

			 // skip loading c43 ~ c73,
			mulps(xmm6, xmm15) // scale by alpha,
			movaps(xmm15, mem(rdx)) // and store back to memory.

		label(.SDONE)


	end_asm(
		: // output operands (none)
		: // input operands
		  [k_iter] "m" (k_iter), // 0
		  [k_left] "m" (k_left), // 1
		  [a]      "m" (a),      // 2
		  [b]      "m" (b),      // 3
		  [alpha]  "m" (alpha),  // 4
		  [beta]   "m" (beta),   // 5
		  [c]      "m" (c),      // 6
		  [rs_c]   "m" (rs_c),   // 7
		  [cs_c]   "m" (cs_c),   // 8
		  [b_next] "m" (b_next)/*, // 9
		  [a_next] "m" (a_next)*/  // 10
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	)

	GEMM_UKR_FLUSH_CT( s );
}

void bli_dgemm_penryn_asm_4x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT_ALIGNED( d, 4, 4, false, 16 );

	begin_asm()


		mov(var(a), rax) // load address of a.
		mov(var(b), rbx) // load address of b.
		mov(var(b_next), r9) // load address of b_next.
		mov(var(a_next), r11) // load address of a_next.

		sub(imm(0-8*16), rax) // increment pointers to allow byte
		sub(imm(0-8*16), rbx) // offsets in the unrolled iterations.

		movaps(mem(rax, -8*16), xmm0) // initialize loop by pre-loading elements
		movaps(mem(rax, -7*16), xmm1) // of a and b.
		movaps(mem(rbx, -8*16), xmm2)

		mov(var(c), rcx) // load address of c
		mov(var(cs_c), rdi) // load cs_c
		lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(double)
		mov(rdi, r12) // make a copy of cs_c (in bytes)
		lea(mem(rcx, rdi, 2), r10) // load address of c + 2*cs_c;

		prefetch(2, mem(r9, 0*8)) // prefetch b_next

		xorpd(xmm3, xmm3)
		xorpd(xmm4, xmm4)
		xorpd(xmm5, xmm5)
		xorpd(xmm6, xmm6)

		prefetch(2, mem(rcx, 3*8)) // prefetch c + 0*cs_c
		xorpd(xmm8, xmm8)
		xorpd(xmm9, xmm9)
		prefetch(2, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*cs_c
		xorpd(xmm10, xmm10)
		xorpd(xmm11, xmm11)
		prefetch(2, mem(r10, 3*8)) // prefetch c + 2*cs_c
		xorpd(xmm12, xmm12)
		xorpd(xmm13, xmm13)
		prefetch(2, mem(r10, rdi, 1, 3*8)) // prefetch c + 3*cs_c
		xorpd(xmm14, xmm14)
		xorpd(xmm15, xmm15)



		mov(var(k_iter), rsi) // i = k_iter;
		test(rsi, rsi) // check i via logical AND.
		je(.DCONSIDKLEFT) // if i == 0, jump to code that
		 // contains the k_left loop.


		label(.DLOOPKITER) // MAIN LOOP

		prefetch(0, mem(rax, (4*35+1)*8))
		//prefetch(0, mem(rax, (8*97+4)*8))

		//prefetch(0, mem(r11, 67*4*8)) // prefetch a_next[0]

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


		prefetch(0, mem(rax, (4*37+1)*8))
		//prefetch(0, mem(rax, (8*97+12)*8))

		//prefetch(0, mem(r11, 69*4*8)) // prefetch a_next[8]
		//sub(imm(-4*4*8), r11) // a_next += 4*4 (unroll x mr)



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

		sub(imm(0-4*4*8), r9) // b_next += 4*4 (unroll x nr)

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

		prefetch(2, mem(r9, 0*8)) // prefetch b_next[0]
		prefetch(2, mem(r9, 8*8)) // prefetch b_next[8]

		dec(rsi) // i -= 1;
		jne(.DLOOPKITER) // iterate again if i != 0.



		//prefetch(2, mem(r9, -8*8)) // prefetch b_next[-8]



		label(.DCONSIDKLEFT)

		mov(var(k_left), rsi) // i = k_left;
		test(rsi, rsi) // check i via logical AND.
		je(.DPOSTACCUM) // if i == 0, we're done; jump to end.
		 // else, we prepare to enter k_left loop.


		label(.DLOOPKLEFT) // EDGE LOOP

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
		jne(.DLOOPKLEFT) // iterate again if i != 0.



		label(.DPOSTACCUM)

		addpd(xmm3, xmm11)
		addpd(xmm4, xmm15)
		addpd(xmm5, xmm10)
		addpd(xmm6, xmm14)


		mov(var(alpha), rax) // load address of alpha
		mov(var(beta), rbx) // load address of beta
		movddup(mem(rax), xmm6) // load alpha and duplicate
		movddup(mem(rbx), xmm7) // load beta and duplicate


		mov(var(rs_c), rsi) // load rs_c
		mov(rsi, r8) // make a copy of rs_c

		lea(mem(, rsi, 8), rsi) // rsi = rs_c * sizeof(double)

		lea(mem(rcx, rsi, 2), rdx) // load address of c + 2*rs_c;

		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab01  ( ab00  ( ab03  ( ab02
		 //   ab10 )  ab11 )  ab12 )  ab13 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab21  ( ab20  ( ab23  ( ab22
		 //   ab30 )  ab31 )  ab32 )  ab33 )
		movaps(xmm8, xmm0)
		movsd(xmm9, xmm8)
		movsd(xmm0, xmm9)

		movaps(xmm10, xmm0)
		movsd(xmm11, xmm10)
		movsd(xmm0, xmm11)

		movaps(xmm12, xmm0)
		movsd(xmm13, xmm12)
		movsd(xmm0, xmm13)

		movaps(xmm14, xmm0)
		movsd(xmm15, xmm14)
		movsd(xmm0, xmm15)
		 // xmm8:   xmm9:   xmm10:  xmm11:
		 // ( ab00  ( ab01  ( ab02  ( ab03
		 //   ab10 )  ab11 )  ab12 )  ab13 )
		 //
		 // xmm12:  xmm13:  xmm14:  xmm15:
		 // ( ab20  ( ab21  ( ab22  ( ab23
		 //   ab30 )  ab31 )  ab32 )  ab33 )

		 // now avoid loading C if beta == 0

		xorpd(xmm0, xmm0) // set xmm0 to zero.
		ucomisd(xmm0, xmm7) // check if beta == 0.
		je(.DBETAZERO) // if ZF = 1, jump to beta == 0 case

			movaps(mem(rcx), xmm0) // load c00 and c10,
			mulpd(xmm6, xmm8) // scale by alpha,
			mulpd(xmm7, xmm0) // scale by beta,
			addpd(xmm8, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.
			add(rdi, rcx)

			movaps(mem(rdx), xmm1) // load c20 and c30,
			mulpd(xmm6, xmm12) // scale by alpha,
			mulpd(xmm7, xmm1) // scale by beta,
			addpd(xmm12, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.
			add(rdi, rdx)



			movaps(mem(rcx), xmm0) // load c01 and c11,
			mulpd(xmm6, xmm9) // scale by alpha,
			mulpd(xmm7, xmm0) // scale by beta,
			addpd(xmm9, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.
			add(rdi, rcx)

			movaps(mem(rdx), xmm1) // load c21 and c31,
			mulpd(xmm6, xmm13) // scale by alpha,
			mulpd(xmm7, xmm1) // scale by beta,
			addpd(xmm13, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.
			add(rdi, rdx)



			movaps(mem(rcx), xmm0) // load c02 and c12,
			mulpd(xmm6, xmm10) // scale by alpha,
			mulpd(xmm7, xmm0) // scale by beta,
			addpd(xmm10, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.
			add(rdi, rcx)

			movaps(mem(rdx), xmm1) // load c22 and c32,
			mulpd(xmm6, xmm14) // scale by alpha,
			mulpd(xmm7, xmm1) // scale by beta,
			addpd(xmm14, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.
			add(rdi, rdx)



			movaps(mem(rcx), xmm0) // load c03 and c13,
			mulpd(xmm6, xmm11) // scale by alpha,
			mulpd(xmm7, xmm0) // scale by beta,
			addpd(xmm11, xmm0) // add the gemm result,
			movaps(xmm0, mem(rcx)) // and store back to memory.


			movaps(mem(rdx), xmm1) // load c23 and c33,
			mulpd(xmm6, xmm15) // scale by alpha,
			mulpd(xmm7, xmm1) // scale by beta,
			addpd(xmm15, xmm1) // add the gemm result,
			movaps(xmm1, mem(rdx)) // and store back to memory.

			jmp(.DDONE) // jump to end.

		label(.DBETAZERO)

			 // skip loading c00 and c10,
			mulpd(xmm6, xmm8) // scale by alpha,
			movaps(xmm8, mem(rcx)) // and store back to memory.
			add(rdi, rcx)
			 // skip loading c20 and c30,
			mulpd(xmm6, xmm12) // scale by alpha,
			movaps(xmm12, mem(rdx)) // and store back to memory.
			add(rdi, rdx)


			 // skip loading c01 and c11,
			mulpd(xmm6, xmm9) // scale by alpha,
			movaps(xmm9, mem(rcx)) // and store back to memory.
			add(rdi, rcx)
			 // skip loading c21 and c31,
			mulpd(xmm6, xmm13) // scale by alpha,
			movaps(xmm13, mem(rdx)) // and store back to memory.
			add(rdi, rdx)


			 // skip loading c02 and c12,
			mulpd(xmm6, xmm10) // scale by alpha,
			movaps(xmm10, mem(rcx)) // and store back to memory.
			add(rdi, rcx)
			 // skip loading c22 and c32,
			mulpd(xmm6, xmm14) // scale by alpha,
			movaps(xmm14, mem(rdx)) // and store back to memory.
			add(rdi, rdx)


			 // skip loading c03 and c13,
			mulpd(xmm6, xmm11) // scale by alpha,
			movaps(xmm11, mem(rcx)) // and store back to memory.

			 // skip loading c23 and c33,
			mulpd(xmm6, xmm15) // scale by alpha,
			movaps(xmm15, mem(rdx)) // and store back to memory.

		label(.DDONE)


	end_asm(
		: // output operands (none)
		: // input operands
		  [k_iter] "m" (k_iter), // 0
		  [k_left] "m" (k_left), // 1
		  [a]      "m" (a),      // 2
		  [b]      "m" (b),      // 3
		  [alpha]  "m" (alpha),  // 4
		  [beta]   "m" (beta),   // 5
		  [c]      "m" (c),      // 6
		  [rs_c]   "m" (rs_c),   // 7
		  [cs_c]   "m" (cs_c),   // 8
		  [b_next] "m" (b_next), // 9
		  [a_next] "m" (a_next)  // 10
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	)

	GEMM_UKR_FLUSH_CT( d );
}


