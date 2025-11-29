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

/* NOTE: The micro-kernels in this file were partially inspired by portions
   of code found in OpenBLAS 0.2.12 (http://www.openblas.net/). -FGVZ */

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

void bli_sgemm_piledriver_asm_16x3
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
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( s, 16, 3, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	mov(var(b_next), r15) // load address of b_next.
	mov(var(a_next), r14) // load address of a_next.

	prefetch(0, mem(rbx, 128)) // prefetch b
	prefetch(0, mem(rbx, 64+128)) // prefetch b
	prefetch(0, mem(rbx, 128+128)) // prefetch b

	add(imm(32*4), rax)
	add(imm(12*4), rbx)

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 4), rdi) // cs_c *= sizeof(float)
	lea(mem(rcx, rdi, 1), r10) // load address of c + 1*cs_c;
	lea(mem(rcx, rdi, 2), r11) // load address of c + 2*cs_c;

	vbroadcastss(mem(rbx, -12*4), xmm1)
	vbroadcastss(mem(rbx, -11*4), xmm2)
	vbroadcastss(mem(rbx, -10*4), xmm3)

	vxorps(xmm4, xmm4, xmm4)
	vxorps(xmm5, xmm5, xmm5)
	vxorps(xmm6, xmm6, xmm6)
	vxorps(xmm7, xmm7, xmm7)
	vxorps(xmm8, xmm8, xmm8)
	vxorps(xmm9, xmm9, xmm9)
	vxorps(xmm10, xmm10, xmm10)
	vxorps(xmm11, xmm11, xmm11)
	vxorps(xmm12, xmm12, xmm12)
	vxorps(xmm13, xmm13, xmm13)
	vxorps(xmm14, xmm14, xmm14)
	vxorps(xmm15, xmm15, xmm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.SCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.SLOOPKITER) // MAIN LOOP


	je(.SCONSIDKLEFT) // if i == 0, jump to k_left code.


	prefetch(0, mem(rbx, 16+192)) // prefetch b

	 // iteration 0
	vmovaps(mem(rax, -32*4), xmm0)
	prefetch(0, mem(rax, 384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -28*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -24*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -20*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, -9*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, -8*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)

	 // iteration 1
	vmovaps(mem(rax, -16*4), xmm0)
	vbroadcastss(mem(rbx, -7*4), xmm3)
	prefetch(0, mem(rax, 64+384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -12*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -8*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -4*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, -6*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, -5*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)

	 // iteration 2
	vmovaps(mem(rax, 0*4), xmm0)
	vbroadcastss(mem(rbx, -4*4), xmm3)
	prefetch(0, mem(rax, 128+384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 4*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 8*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 12*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, -3*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, -2*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)

	 // iteration 3
	vmovaps(mem(rax, 16*4), xmm0)
	vbroadcastss(mem(rbx, -1*4), xmm3)
	prefetch(0, mem(rax, 192+384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 20*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 24*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 28*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, 0*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, 1*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)


	add(imm(4*16*4), rax) // a += 4*16 (unroll x mr)

	 // iteration 4
	vmovaps(mem(rax, -32*4), xmm0)
	vbroadcastss(mem(rbx, 2*4), xmm3)
	prefetch(0, mem(rax, 384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -28*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -24*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -20*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, 3*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, 4*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)

	prefetch(0, mem(rbx, 80+192)) // prefetch b

	 // iteration 5
	vmovaps(mem(rax, -16*4), xmm0)
	vbroadcastss(mem(rbx, 5*4), xmm3)
	prefetch(0, mem(rax, 64+384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -12*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -8*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -4*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, 6*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, 7*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)

	 // iteration 6
	vmovaps(mem(rax, 0*4), xmm0)
	vbroadcastss(mem(rbx, 8*4), xmm3)
	prefetch(0, mem(rax, 128+384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 4*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 8*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 12*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, 9*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, 10*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)

	 // iteration 7
	vmovaps(mem(rax, 16*4), xmm0)
	vbroadcastss(mem(rbx, 11*4), xmm3)
	add(imm(8*3*4), rbx) // a += 4*3  (unroll x nr)
	prefetch(0, mem(rax, 192+384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 20*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 24*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 28*4), xmm0)
	add(imm(4*16*4), rax) // a += 4*16 (unroll x mr)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, -12*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, -11*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)
	vbroadcastss(mem(rbx, -10*4), xmm3)




	dec(rsi) // i -= 1;
	jmp(.SLOOPKITER) // jump to beginning of loop.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.SPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT) // EDGE LOOP


	je(.SPOSTACCUM) // if i == 0, we're done.


	prefetch(0, mem(rbx, 16+192)) // prefetch b

	 // iteration 0
	vmovaps(mem(rax, -32*4), xmm0)
	prefetch(0, mem(rax, 384))
	vfmadd231ps(xmm1, xmm0, xmm4)
	vfmadd231ps(xmm2, xmm0, xmm5)
	vfmadd231ps(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -28*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm7)
	vfmadd231ps(xmm2, xmm0, xmm8)
	vfmadd231ps(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -24*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm10)
	vfmadd231ps(xmm2, xmm0, xmm11)
	vfmadd231ps(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -20*4), xmm0)
	vfmadd231ps(xmm1, xmm0, xmm13)
	vbroadcastss(mem(rbx, -9*4), xmm1)
	vfmadd231ps(xmm2, xmm0, xmm14)
	vbroadcastss(mem(rbx, -8*4), xmm2)
	vfmadd231ps(xmm3, xmm0, xmm15)
	vbroadcastss(mem(rbx, -7*4), xmm3)


	add(imm(1*16*4), rax) // a += 4*16 (unroll x mr)
	add(imm(1*3*4), rbx) // a += 4*3  (unroll x nr)


	dec(rsi) // i -= 1;
	jmp(.SLOOPKLEFT) // jump to beginning of loop.



	label(.SPOSTACCUM)


	prefetchw0(mem(rcx, 0*8)) // prefetch c + 0*cs_c
	prefetchw0(mem(r10, 0*8)) // prefetch c + 1*cs_c
	prefetchw0(mem(r11, 0*8)) // prefetch c + 2*cs_c


	 // xmm4:   xmm5:   xmm6:
	 // ( ab00  ( ab01  ( ab02
	 //   ab10    ab11    ab12
	 //   ab20    ab21    ab22
	 //   ab30 )  ab31 )  ab32 )

	 // xmm7:   xmm8:   xmm9:
	 // ( ab40  ( ab41  ( ab42
	 //   ab50    ab51    ab52
	 //   ab60    ab61    ab62
	 //   ab70 )  ab71 )  ab72 )

	 // xmm10:  xmm11:  xmm12:
	 // ( ab80  ( ab01  ( ab02
	 //   ab90    ab11    ab12
	 //   abA0    abA1    abA2
	 //   abB0 )  abB1 )  abB2 )

	 // xmm13:  xmm14:  xmm15:
	 // ( abC0  ( abC1  ( abC2
	 //   abD0    abD1    abD2
	 //   abE0    abE1    abE2
	 //   abF0 )  abF1 )  abF2 )



	mov(var(alpha), rax) // load address of alpha
	mov(var(beta), rbx) // load address of beta
	vbroadcastss(mem(rax), xmm0) // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm2) // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4) // scale by alpha
	vmulps(xmm0, xmm5, xmm5)
	vmulps(xmm0, xmm6, xmm6)
	vmulps(xmm0, xmm7, xmm7)
	vmulps(xmm0, xmm8, xmm8)
	vmulps(xmm0, xmm9, xmm9)
	vmulps(xmm0, xmm10, xmm10)
	vmulps(xmm0, xmm11, xmm11)
	vmulps(xmm0, xmm12, xmm12)
	vmulps(xmm0, xmm13, xmm13)
	vmulps(xmm0, xmm14, xmm14)
	vmulps(xmm0, xmm15, xmm15)



	prefetch(0, mem(r14)) // prefetch a_next
	prefetch(0, mem(r14, 64)) // prefetch a_next




	mov(var(rs_c), rsi) // load rs_c
	lea(mem(, rsi, 4), rsi) // rsi = rs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx) // load address of c + 4*rs_c;

	lea(mem(, rsi, 2), r12) // r12 = 2*rs_c;
	lea(mem(rsi, rsi, 2), r13) // r13 = 3*rs_c;



	 // determine if
	 //    c    % 32 == 0, AND
	 //  4*cs_c % 32 == 0, AND
	 //    rs_c      == 1
	 // ie: aligned, ldim aligned, and
	 // column-stored

	cmp(imm(4), rsi) // set ZF if (4*rs_c) == 4.
	sete(bl) // bl = ( ZF == 1 ? 1 : 0 );
	test(imm(31), rcx) // set ZF if c & 32 is zero.
	setz(bh) // bh = ( ZF == 0 ? 1 : 0 );
	test(imm(31), rdi) // set ZF if (4*cs_c) & 32 is zero.
	setz(al) // al = ( ZF == 0 ? 1 : 0 );
	 // and(bl,bh) followed by
	 // and(bh,al) will reveal result

	prefetch(0, mem(r15)) // prefetch b_next
	prefetch(0, mem(r15, 64)) // prefetch b_next

	 // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0) // set xmm0 to zero.
	vucomiss(xmm0, xmm2) // set ZF if beta == 0.
	je(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

		vfmadd231ps(mem(rcx, 0*16), xmm2, xmm4)
		vfmadd231ps(mem(rcx, 1*16), xmm2, xmm7)
		vfmadd231ps(mem(rcx, 2*16), xmm2, xmm10)
		vfmadd231ps(mem(rcx, 3*16), xmm2, xmm13)

		vfmadd231ps(mem(r10, 0*16), xmm2, xmm5)
		vfmadd231ps(mem(r10, 1*16), xmm2, xmm8)
		vfmadd231ps(mem(r10, 2*16), xmm2, xmm11)
		vfmadd231ps(mem(r10, 3*16), xmm2, xmm14)

		vfmadd231ps(mem(r11, 0*16), xmm2, xmm6)
		vfmadd231ps(mem(r11, 1*16), xmm2, xmm9)
		vfmadd231ps(mem(r11, 2*16), xmm2, xmm12)
		vfmadd231ps(mem(r11, 3*16), xmm2, xmm15)

		 // fall through

	label(.SBETAZERO)

		vmovups(xmm4, mem(rcx, 0*16))
		vmovups(xmm7, mem(rcx, 1*16))
		vmovups(xmm10, mem(rcx, 2*16))
		vmovups(xmm13, mem(rcx, 3*16))

		vmovups(xmm5, mem(r10, 0*16))
		vmovups(xmm8, mem(r10, 1*16))
		vmovups(xmm11, mem(r10, 2*16))
		vmovups(xmm14, mem(r10, 3*16))

		vmovups(xmm6, mem(r11, 0*16))
		vmovups(xmm9, mem(r11, 1*16))
		vmovups(xmm12, mem(r11, 2*16))
		vmovups(xmm15, mem(r11, 3*16))

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
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( s );
}

void bli_dgemm_piledriver_asm_8x3
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
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( d, 8, 3, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	mov(var(b_next), r15) // load address of b_next.
	mov(var(a_next), r14) // load address of a_next.

	prefetch(0, mem(rbx, 128)) // prefetch b
	prefetch(0, mem(rbx, 64+128)) // prefetch b
	prefetch(0, mem(rbx, 128+128)) // prefetch b

	add(imm(16*8), rax)
	add(imm(12*8), rbx)

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(double)
	lea(mem(rcx, rdi, 1), r10) // load address of c + 1*cs_c;
	lea(mem(rcx, rdi, 2), r11) // load address of c + 2*cs_c;

	vmovddup(mem(rbx, -12*8), xmm1)
	vmovddup(mem(rbx, -11*8), xmm2)
	vmovddup(mem(rbx, -10*8), xmm3)

	vxorpd(xmm4, xmm4, xmm4)
	vxorpd(xmm5, xmm5, xmm5)
	vxorpd(xmm6, xmm6, xmm6)
	vxorpd(xmm7, xmm7, xmm7)
	vxorpd(xmm8, xmm8, xmm8)
	vxorpd(xmm9, xmm9, xmm9)
	vxorpd(xmm10, xmm10, xmm10)
	vxorpd(xmm11, xmm11, xmm11)
	vxorpd(xmm12, xmm12, xmm12)
	vxorpd(xmm13, xmm13, xmm13)
	vxorpd(xmm14, xmm14, xmm14)
	vxorpd(xmm15, xmm15, xmm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.DCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.DLOOPKITER) // MAIN LOOP


	je(.DCONSIDKLEFT) // if i == 0, jump to k_left code.


	prefetch(0, mem(rbx, -32+256)) // prefetch b
	prefetch(0, mem(rbx, 32+256)) // prefetch b

	 // iteration 0
	vmovaps(mem(rax, -8*16), xmm0)
	prefetch(0, mem(rax, 384)) // prefetch a
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -7*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -6*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -5*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, -9*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, -8*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)

	 // iteration 1
	vmovaps(mem(rax, -4*16), xmm0)
	prefetch(0, mem(rax, 64+384)) // prefetch a
	vmovddup(mem(rbx, -7*8), xmm3)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -3*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -2*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -1*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, -6*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, -5*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)

	 // iteration 2
	vmovaps(mem(rax, 0*16), xmm0)
	prefetch(0, mem(rax, 128+384)) // prefetch a
	vmovddup(mem(rbx, -4*8), xmm3)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 1*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 2*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 3*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, -3*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, -2*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)

	 // iteration 3
	vmovaps(mem(rax, 4*16), xmm0)
	prefetch(0, mem(rax, 192+384)) // prefetch a
	vmovddup(mem(rbx, -1*8), xmm3)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 5*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 6*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 7*16), xmm0)
	add(imm(4*8*8), rax) // a += 4*8 (unroll x mr)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, 0*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, 1*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)

	 // iteration 4
	vmovaps(mem(rax, -8*16), xmm0)
	prefetch(0, mem(rax, 384)) // prefetch a
	vmovddup(mem(rbx, 2*8), xmm3)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -7*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -6*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -5*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, 3*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, 4*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)

	prefetch(0, mem(rbx, 96+256)) // prefetch b

	 // iteration 5
	vmovaps(mem(rax, -4*16), xmm0)
	prefetch(0, mem(rax, 64+384)) // prefetch a
	vmovddup(mem(rbx, 5*8), xmm3)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -3*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -2*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -1*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, 6*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, 7*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)


	 // iteration 6
	vmovaps(mem(rax, 0*16), xmm0)
	prefetch(0, mem(rax, 128+384)) // prefetch a
	vmovddup(mem(rbx, 8*8), xmm3)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 1*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 2*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 3*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, 9*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, 10*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)

	 // iteration 7
	vmovaps(mem(rax, 4*16), xmm0)
	prefetch(0, mem(rax, 192+384)) // prefetch a
	vmovddup(mem(rbx, 11*8), xmm3)
	add(imm(8*3*8), rbx) // b += 8*3 (unroll x nr)
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, 5*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, 6*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, 7*16), xmm0)
	add(imm(4*8*8), rax) // a += 4*8 (unroll x mr)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, -12*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, -11*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)
	vmovddup(mem(rbx, -10*8), xmm3)



	dec(rsi) // i -= 1;
	jmp(.DLOOPKITER) // jump to beginning of loop.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.DPOSTACCUM) // if i == 0, we're done.
	 // else, we prepare to
	 // enter k_left loop.


	label(.DLOOPKLEFT) // EDGE LOOP


	je(.DPOSTACCUM) // if i == 0, we're done.

	 // iteration 0
	vmovaps(mem(rax, -8*16), xmm0)
	prefetch(0, mem(rax, 512)) // prefetch a
	vfmadd231pd(xmm1, xmm0, xmm4)
	vfmadd231pd(xmm2, xmm0, xmm5)
	vfmadd231pd(xmm3, xmm0, xmm6)
	vmovaps(mem(rax, -7*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm7)
	vfmadd231pd(xmm2, xmm0, xmm8)
	vfmadd231pd(xmm3, xmm0, xmm9)
	vmovaps(mem(rax, -6*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm10)
	vfmadd231pd(xmm2, xmm0, xmm11)
	vfmadd231pd(xmm3, xmm0, xmm12)
	vmovaps(mem(rax, -5*16), xmm0)
	vfmadd231pd(xmm1, xmm0, xmm13)
	vmovddup(mem(rbx, -9*8), xmm1)
	vfmadd231pd(xmm2, xmm0, xmm14)
	vmovddup(mem(rbx, -8*8), xmm2)
	vfmadd231pd(xmm3, xmm0, xmm15)
	vmovddup(mem(rbx, -7*8), xmm3)


	add(imm(1*8*8), rax) // a += 1*8 (1 x mr)
	add(imm(1*3*8), rbx) // b += 1*3 (1 x nr)


	dec(rsi) // i -= 1;
	jmp(.DLOOPKLEFT) // jump to beginning of loop.



	label(.DPOSTACCUM)

	prefetchw0(mem(rcx, 0*8)) // prefetch c + 0*cs_c
	prefetchw0(mem(r10, 0*8)) // prefetch c + 1*cs_c
	prefetchw0(mem(r11, 0*8)) // prefetch c + 2*cs_c


	 // xmm4:   xmm5:   xmm6:
	 // ( ab00  ( ab01  ( ab02
	 //   ab10 )  ab11 )  ab12 )
	 //
	 // xmm7:   xmm8:   xmm9:
	 // ( ab20  ( ab21  ( ab22
	 //   ab30 )  ab31 )  ab32 )
	 //
	 // xmm10:  xmm11:  xmm12:
	 // ( ab40  ( ab41  ( ab42
	 //   ab50 )  ab51 )  ab52 )
	 //
	 // xmm13:  xmm14:  xmm15:
	 // ( ab60  ( ab61  ( ab62
	 //   ab70 )  ab71 )  ab72 )




	mov(var(alpha), rax) // load address of alpha
	mov(var(beta), rbx) // load address of beta
	vmovddup(mem(rax), xmm0) // load alpha and duplicate
	vmovddup(mem(rbx), xmm2) // load beta and duplicate

	vmulpd(xmm0, xmm4, xmm4) // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(xmm0, xmm6, xmm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(xmm0, xmm8, xmm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(xmm0, xmm10, xmm10)
	vmulpd(xmm0, xmm11, xmm11)
	vmulpd(xmm0, xmm12, xmm12)
	vmulpd(xmm0, xmm13, xmm13)
	vmulpd(xmm0, xmm14, xmm14)
	vmulpd(xmm0, xmm15, xmm15)


	prefetch(0, mem(r14)) // prefetch a_next
	prefetch(0, mem(r14, 64)) // prefetch a_next

	prefetch(0, mem(r15)) // prefetch b_next
	prefetch(0, mem(r15, 64)) // prefetch b_next

	 // now avoid loading C if beta == 0

	vxorpd(xmm0, xmm0, xmm0) // set xmm0 to zero.
	vucomisd(xmm0, xmm2) // set ZF if beta == 0.
	je(.DBETAZERO) // if ZF = 1, jump to beta == 0 case

		 // xmm4:   xmm5:   xmm6:
		 // ( ab00  ( ab01  ( ab02
		 //   ab10 )  ab11 )  ab12 )
		 //
		 // xmm7:   xmm8:   xmm9:
		 // ( ab20  ( ab21  ( ab22
		 //   ab30 )  ab31 )  ab32 )
		 //
		 // xmm10:  xmm11:  xmm12:
		 // ( ab40  ( ab41  ( ab42
		 //   ab50 )  ab51 )  ab52 )
		 //
		 // xmm13:  xmm14:  xmm15:
		 // ( ab60  ( ab61  ( ab62
		 //   ab70 )  ab71 )  ab72 )

		vfmadd231pd(mem(rcx, 0*16), xmm2, xmm4)
		vfmadd231pd(mem(rcx, 1*16), xmm2, xmm7)
		vfmadd231pd(mem(rcx, 2*16), xmm2, xmm10)
		vfmadd231pd(mem(rcx, 3*16), xmm2, xmm13)

		vfmadd231pd(mem(r10, 0*16), xmm2, xmm5)
		vfmadd231pd(mem(r10, 1*16), xmm2, xmm8)
		vfmadd231pd(mem(r10, 2*16), xmm2, xmm11)
		vfmadd231pd(mem(r10, 3*16), xmm2, xmm14)

		vfmadd231pd(mem(r11, 0*16), xmm2, xmm6)
		vfmadd231pd(mem(r11, 1*16), xmm2, xmm9)
		vfmadd231pd(mem(r11, 2*16), xmm2, xmm12)
		vfmadd231pd(mem(r11, 3*16), xmm2, xmm15)

		 // fall through

	label(.DBETAZERO)

		vmovups(xmm4, mem(rcx, 0*16))
		vmovups(xmm7, mem(rcx, 1*16))
		vmovups(xmm10, mem(rcx, 2*16))
		vmovups(xmm13, mem(rcx, 3*16))

		vmovups(xmm5, mem(r10, 0*16))
		vmovups(xmm8, mem(r10, 1*16))
		vmovups(xmm11, mem(r10, 2*16))
		vmovups(xmm14, mem(r10, 3*16))

		vmovups(xmm6, mem(r11, 0*16))
		vmovups(xmm9, mem(r11, 1*16))
		vmovups(xmm12, mem(r11, 2*16))
		vmovups(xmm15, mem(r11, 3*16))

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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( d );
}

void bli_cgemm_piledriver_asm_4x2
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
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( c, 4, 2, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	mov(var(b_next), r15) // load address of b_next.
	mov(var(a_next), r14) // load address of a_next.

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(scomplex)
	lea(mem(rcx, rdi, 1), r10) // load address of c + 1*cs_c;

	add(imm(32*4), rax)
	add(imm(16*4), rbx)


	vxorps(xmm8, xmm8, xmm8)
	vxorps(xmm9, xmm9, xmm9)
	vxorps(xmm10, xmm10, xmm10)
	vxorps(xmm11, xmm11, xmm11)
	vxorps(xmm12, xmm12, xmm12)
	vxorps(xmm13, xmm13, xmm13)
	vxorps(xmm14, xmm14, xmm14)
	vxorps(xmm15, xmm15, xmm15)
	//vzeroall()



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.CCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.CLOOPKITER) // MAIN LOOP


	je(.CCONSIDKLEFT) // if i == 0, jump to k_left code.


	prefetch(0, mem(rbx, 256))
	prefetch(0, mem(rax, 512))

	 // iteration 0
	vmovaps(mem(rax, -32*4), xmm0)
	vbroadcastss(mem(rbx, -16*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -28*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, -15*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, -14*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, -13*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	 // iteration 1
	vmovaps(mem(rax, -24*4), xmm0)
	vbroadcastss(mem(rbx, -12*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -20*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, -11*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, -10*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, -9*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	prefetch(0, mem(rbx, 64+256))
	prefetch(0, mem(rax, 64+512))

	 // iteration 2
	vmovaps(mem(rax, -16*4), xmm0)
	vbroadcastss(mem(rbx, -8*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -12*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, -7*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, -6*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, -5*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	 // iteration 3
	vmovaps(mem(rax, -8*4), xmm0)
	vbroadcastss(mem(rbx, -4*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -4*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, -3*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, -2*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, -1*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	prefetch(0, mem(rbx, 128+256))
	prefetch(0, mem(rax, 128+512))

	 // iteration 4
	vmovaps(mem(rax, 0*4), xmm0)
	vbroadcastss(mem(rbx, 0*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 4*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, 1*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, 2*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, 3*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	 // iteration 5
	vmovaps(mem(rax, 8*4), xmm0)
	vbroadcastss(mem(rbx, 4*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 12*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, 5*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, 6*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, 7*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	prefetch(0, mem(rbx, 128+256))
	prefetch(0, mem(rax, 128+512))

	 // iteration 6
	vmovaps(mem(rax, 16*4), xmm0)
	vbroadcastss(mem(rbx, 8*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 20*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, 9*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, 10*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, 11*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)

	 // iteration 7
	vmovaps(mem(rax, 24*4), xmm0)
	vbroadcastss(mem(rbx, 12*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 28*4), xmm1)
	add(imm(8*4*8), rax) // a += 8*2 (unroll x mr)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, 13*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, 14*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, 15*4), xmm7)
	add(imm(8*2*8), rbx) // b += 8*2 (unroll x nr)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)



	dec(rsi) // i -= 1;
	jmp(.CLOOPKITER) // jump to beginning of loop.






	label(.CCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.CPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.CLOOPKLEFT) // EDGE LOOP


	je(.CPOSTACCUM) // if i == 0, we're done.

	prefetch(0, mem(rbx, 256))
	prefetch(0, mem(rax, 512))

	 // iteration 0
	vmovaps(mem(rax, -32*4), xmm0)
	vbroadcastss(mem(rbx, -16*4), xmm4)
	vfmadd231ps(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -28*4), xmm1)
	vfmadd231ps(xmm1, xmm4, xmm12)
	vbroadcastss(mem(rbx, -15*4), xmm5)
	vfmadd231ps(xmm0, xmm5, xmm9)
	vfmadd231ps(xmm1, xmm5, xmm13)
	vbroadcastss(mem(rbx, -14*4), xmm6)
	vfmadd231ps(xmm0, xmm6, xmm10)
	vfmadd231ps(xmm1, xmm6, xmm14)
	vbroadcastss(mem(rbx, -13*4), xmm7)
	vfmadd231ps(xmm0, xmm7, xmm11)
	vfmadd231ps(xmm1, xmm7, xmm15)


	add(imm(1*4*8), rax) // a += 1*2 (1 x mr)
	add(imm(1*2*8), rbx) // b += 1*2 (1 x nr)


	dec(rsi) // i -= 1;
	jmp(.CLOOPKLEFT) // jump to beginning of loop.



	label(.CPOSTACCUM)


	prefetchw0(mem(rcx, 0*8)) // prefetch c + 0*cs_c
	prefetchw0(mem(r10, 0*8)) // prefetch c + 1*cs_c


	vpermilps(imm(0xb1), xmm9, xmm9)
	vpermilps(imm(0xb1), xmm11, xmm11)
	vpermilps(imm(0xb1), xmm13, xmm13)
	vpermilps(imm(0xb1), xmm15, xmm15)

	vaddsubps(xmm9, xmm8, xmm8)
	vaddsubps(xmm11, xmm10, xmm10)
	vaddsubps(xmm13, xmm12, xmm12)
	vaddsubps(xmm15, xmm14, xmm14)


	 // xmm8:   xmm10:
	 // ( ab00  ( ab01
	 //   ab10    ab11
	 //   ab20    ab21
	 //   ab30 )  ab31 )

	 // xmm12:  xmm14:
	 // ( ab40  ( ab41
	 //   ab50    ab51
	 //   ab60    ab61
	 //   ab70 )  ab71 )


	prefetch(0, mem(r14)) // prefetch a_next
	prefetch(0, mem(r14, 64)) // prefetch a_next


	 // scale by alpha

	mov(var(alpha), rax) // load address of alpha
	vbroadcastss(mem(rax), xmm0) // load alpha_r and duplicate
	vbroadcastss(mem(rax, 4), xmm1) // load alpha_i and duplicate

	vpermilps(imm(0xb1), xmm8, xmm9)
	vpermilps(imm(0xb1), xmm10, xmm11)
	vpermilps(imm(0xb1), xmm12, xmm13)
	vpermilps(imm(0xb1), xmm14, xmm15)

	vmulps(xmm8, xmm0, xmm8)
	vmulps(xmm10, xmm0, xmm10)
	vmulps(xmm12, xmm0, xmm12)
	vmulps(xmm14, xmm0, xmm14)

	vmulps(xmm9, xmm1, xmm9)
	vmulps(xmm11, xmm1, xmm11)
	vmulps(xmm13, xmm1, xmm13)
	vmulps(xmm15, xmm1, xmm15)

	vaddsubps(xmm9, xmm8, xmm8)
	vaddsubps(xmm11, xmm10, xmm10)
	vaddsubps(xmm13, xmm12, xmm12)
	vaddsubps(xmm15, xmm14, xmm14)


	mov(var(beta), rbx) // load address of beta
	vbroadcastss(mem(rbx), xmm6) // load beta_r and duplicate
	vbroadcastss(mem(rbx, 4), xmm7) // load beta_i and duplicate

	prefetch(0, mem(r15)) // prefetch b_next
	prefetch(0, mem(r15, 64)) // prefetch b_next

	 // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0) // set xmm0 to zero.
	vucomiss(xmm0, xmm6) // set ZF if beta_r == 0.
	sete(r8b) // r8b = ( ZF == 1 ? 1 : 0 );
	vucomiss(xmm0, xmm7) // set ZF if beta_i == 0.
	sete(r9b) // r9b = ( ZF == 1 ? 1 : 0 );
	and(r8b, r9b) // set ZF if r8b & r9b == 1.
	jne(.CBETAZERO) // if ZF = 0, jump to beta == 0 case

		vmovups(mem(rcx), xmm0) // load c00:c10
		vmovups(mem(rcx, 16), xmm2) // load c20:c30
		vpermilps(imm(0xb1), xmm0, xmm1)
		vpermilps(imm(0xb1), xmm2, xmm3)

		vmulps(xmm6, xmm0, xmm0)
		vmulps(xmm7, xmm1, xmm1)
		vaddsubps(xmm1, xmm0, xmm0)
		vaddps(xmm8, xmm0, xmm8)

		vmulps(xmm6, xmm2, xmm2)
		vmulps(xmm7, xmm3, xmm3)
		vaddsubps(xmm3, xmm2, xmm2)
		vaddps(xmm12, xmm2, xmm12)

		vmovups(mem(r10), xmm0) // load c01:c11
		vmovups(mem(r10, 16), xmm2) // load c21:c31
		vpermilps(imm(0xb1), xmm0, xmm1)
		vpermilps(imm(0xb1), xmm2, xmm3)

		vmulps(xmm6, xmm0, xmm0)
		vmulps(xmm7, xmm1, xmm1)
		vaddsubps(xmm1, xmm0, xmm0)
		vaddps(xmm10, xmm0, xmm10)

		vmulps(xmm6, xmm2, xmm2)
		vmulps(xmm7, xmm3, xmm3)
		vaddsubps(xmm3, xmm2, xmm2)
		vaddps(xmm14, xmm2, xmm14)

		 // fall through

	label(.CBETAZERO)

		vmovups(xmm8, mem(rcx)) // store c00:c10
		vmovups(xmm12, mem(rcx, 16)) // store c20:c30

		vmovups(xmm10, mem(r10)) // store c01:c11
		vmovups(xmm14, mem(r10, 16)) // store c21:c31

	label(.CDONE)


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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( c );
}

void bli_zgemm_piledriver_asm_2x2
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
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( z, 2, 2, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	mov(var(b_next), r15) // load address of b_next.
	mov(var(a_next), r14) // load address of a_next.

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(dcomplex)
	lea(mem(, rdi, 2), rdi)
	lea(mem(rcx, rdi, 1), r10) // load address of c + 1*cs_c;

	add(imm(16*8), rax)
	add(imm(16*8), rbx)

	vxorpd(xmm8, xmm8, xmm8)
	vxorpd(xmm9, xmm9, xmm9)
	vxorpd(xmm10, xmm10, xmm10)
	vxorpd(xmm11, xmm11, xmm11)
	vxorpd(xmm12, xmm12, xmm12)
	vxorpd(xmm13, xmm13, xmm13)
	vxorpd(xmm14, xmm14, xmm14)
	vxorpd(xmm15, xmm15, xmm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.ZCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.ZLOOPKITER) // MAIN LOOP


	je(.ZCONSIDKLEFT) // if i == 0, jump to k_left code.


	prefetch(0, mem(rbx, 256))

	prefetch(0, mem(rax, 512))

	 // iteration 0
	vmovaps(mem(rax, -16*8), xmm0)
	vmovddup(mem(rbx, -16*8), xmm4)
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -14*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, -15*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, -14*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, -13*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, -12*8), xmm0)
	vmovddup(mem(rbx, -12*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	 // iteration 1
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -10*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, -11*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, -10*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, -9*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, -8*8), xmm0)
	vmovddup(mem(rbx, -8*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	prefetch(0, mem(rbx, 64+256))

	prefetch(0, mem(rax, 64+512))

	 // iteration 2
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -6*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, -7*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, -6*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, -5*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, -4*8), xmm0)
	vmovddup(mem(rbx, -4*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	 // iteration 3
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -2*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, -3*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, -2*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, -1*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, 0*8), xmm0)
	vmovddup(mem(rbx, 0*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	prefetch(0, mem(rbx, 128+256))

	prefetch(0, mem(rax, 128+512))

	 // iteration 4
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 2*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, 1*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, 2*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, 3*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, 4*8), xmm0)
	vmovddup(mem(rbx, 4*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	 // iteration 5
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 6*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, 5*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, 6*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, 7*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, 8*8), xmm0)
	vmovddup(mem(rbx, 8*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	prefetch(0, mem(rbx, 128+256))

	prefetch(0, mem(rax, 128+512))

	 // iteration 6
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 10*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, 9*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, 10*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, 11*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vmovaps(mem(rax, 12*8), xmm0)
	vmovddup(mem(rbx, 12*8), xmm4)
	vfmadd231pd(xmm1, xmm7, xmm15)

	 // iteration 7
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, 14*8), xmm1)
	add(imm(8*2*16), rax) // a += 8*2 (unroll x mr)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, 13*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, 14*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, 15*8), xmm7)
	add(imm(8*2*16), rbx) // b += 8*2 (unroll x nr)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vfmadd231pd(xmm1, xmm7, xmm15)



	dec(rsi) // i -= 1;
	jmp(.ZLOOPKITER) // jump to beginning of loop.






	label(.ZCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.ZPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.ZLOOPKLEFT) // EDGE LOOP


	je(.ZPOSTACCUM) // if i == 0, we're done.

	prefetch(0, mem(rbx, 256))

	prefetch(0, mem(rax, 512))

	 // iteration 0
	vmovaps(mem(rax, -16*8), xmm0)
	vmovddup(mem(rbx, -16*8), xmm4)
	vfmadd231pd(xmm0, xmm4, xmm8)
	vmovaps(mem(rax, -14*8), xmm1)
	vfmadd231pd(xmm1, xmm4, xmm12)
	vmovddup(mem(rbx, -15*8), xmm5)
	vfmadd231pd(xmm0, xmm5, xmm9)
	vfmadd231pd(xmm1, xmm5, xmm13)
	vmovddup(mem(rbx, -14*8), xmm6)
	vfmadd231pd(xmm0, xmm6, xmm10)
	vfmadd231pd(xmm1, xmm6, xmm14)
	vmovddup(mem(rbx, -13*8), xmm7)
	vfmadd231pd(xmm0, xmm7, xmm11)
	vfmadd231pd(xmm1, xmm7, xmm15)


	add(imm(1*2*16), rax) // a += 1*2 (1 x mr)
	add(imm(1*2*16), rbx) // b += 1*2 (1 x nr)


	dec(rsi) // i -= 1;
	jmp(.ZLOOPKLEFT) // jump to beginning of loop.



	label(.ZPOSTACCUM)


	prefetchw0(mem(rcx, 0*8)) // prefetch c + 0*cs_c
	prefetchw0(mem(r10, 0*8)) // prefetch c + 1*cs_c


	vpermilpd(imm(0x1), xmm9, xmm9)
	vpermilpd(imm(0x1), xmm11, xmm11)
	vpermilpd(imm(0x1), xmm13, xmm13)
	vpermilpd(imm(0x1), xmm15, xmm15)

	vaddsubpd(xmm9, xmm8, xmm8)
	vaddsubpd(xmm11, xmm10, xmm10)
	vaddsubpd(xmm13, xmm12, xmm12)
	vaddsubpd(xmm15, xmm14, xmm14)


	 // xmm8:   xmm10:
	 // ( ab00  ( ab01
	 //   ab10 )  ab11 )

	 // xmm12:  xmm14:
	 // ( ab20  ( ab21
	 //   ab30 )  ab31 )


	prefetch(0, mem(r14)) // prefetch a_next
	prefetch(0, mem(r14, 64)) // prefetch a_next


	 // scale by alpha

	mov(var(alpha), rax) // load address of alpha
	vmovddup(mem(rax), xmm0) // load alpha_r and duplicate
	vmovddup(mem(rax, 8), xmm1) // load alpha_i and duplicate

	vpermilpd(imm(0x1), xmm8, xmm9)
	vpermilpd(imm(0x1), xmm10, xmm11)
	vpermilpd(imm(0x1), xmm12, xmm13)
	vpermilpd(imm(0x1), xmm14, xmm15)

	vmulpd(xmm8, xmm0, xmm8)
	vmulpd(xmm10, xmm0, xmm10)
	vmulpd(xmm12, xmm0, xmm12)
	vmulpd(xmm14, xmm0, xmm14)

	vmulpd(xmm9, xmm1, xmm9)
	vmulpd(xmm11, xmm1, xmm11)
	vmulpd(xmm13, xmm1, xmm13)
	vmulpd(xmm15, xmm1, xmm15)

	vaddsubpd(xmm9, xmm8, xmm8)
	vaddsubpd(xmm11, xmm10, xmm10)
	vaddsubpd(xmm13, xmm12, xmm12)
	vaddsubpd(xmm15, xmm14, xmm14)




	mov(var(beta), rbx) // load address of beta
	vmovddup(mem(rbx), xmm6) // load beta_r and duplicate
	vmovddup(mem(rbx, 8), xmm7) // load beta_i and duplicate

	prefetch(0, mem(r15)) // prefetch b_next
	prefetch(0, mem(r15, 64)) // prefetch b_next

	 // now avoid loading C if beta == 0

	vxorpd(xmm0, xmm0, xmm0) // set xmm0 to zero.
	vucomisd(xmm0, xmm6) // set ZF if beta_r == 0.
	sete(r8b) // r8b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm7) // set ZF if beta_i == 0.
	sete(r9b) // r9b = ( ZF == 1 ? 1 : 0 );
	and(r8b, r9b) // set ZF if r8b & r9b == 1.
	jne(.ZBETAZERO) // if ZF = 0, jump to beta == 0 case

		vmovups(mem(rcx), xmm0) // load c00
		vmovups(mem(rcx, 16), xmm2) // load c10
		vpermilpd(imm(0x1), xmm0, xmm1)
		vpermilpd(imm(0x1), xmm2, xmm3)

		vmulpd(xmm6, xmm0, xmm0)
		vmulpd(xmm7, xmm1, xmm1)
		vaddsubpd(xmm1, xmm0, xmm0)
		vaddpd(xmm8, xmm0, xmm8)

		vmulpd(xmm6, xmm2, xmm2)
		vmulpd(xmm7, xmm3, xmm3)
		vaddsubpd(xmm3, xmm2, xmm2)
		vaddpd(xmm12, xmm2, xmm12)

		vmovups(mem(r10), xmm0) // load c01
		vmovups(mem(r10, 16), xmm2) // load c11
		vpermilpd(imm(0x1), xmm0, xmm1)
		vpermilpd(imm(0x1), xmm2, xmm3)

		vmulpd(xmm6, xmm0, xmm0)
		vmulpd(xmm7, xmm1, xmm1)
		vaddsubpd(xmm1, xmm0, xmm0)
		vaddpd(xmm10, xmm0, xmm10)

		vmulpd(xmm6, xmm2, xmm2)
		vmulpd(xmm7, xmm3, xmm3)
		vaddsubpd(xmm3, xmm2, xmm2)
		vaddpd(xmm14, xmm2, xmm14)

		 // fall through

	label(.ZBETAZERO)

		vmovups(xmm8, mem(rcx)) // store c00
		vmovups(xmm12, mem(rcx, 16)) // store c10

		vmovups(xmm10, mem(r10)) // store c01
		vmovups(xmm14, mem(r10, 16)) // store c11

	label(.ZDONE)


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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( z );
}


