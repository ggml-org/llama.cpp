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
void bli_strsm_u_penryn_asm_8x4
     (
     )
{
}
#endif

void bli_dtrsm_u_penryn_asm_4x4
     (
       const void*      a11,
             void*      b11,
             void*      c11, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	begin_asm()

		mov(var(b11), rbx) // load address of b11.

		movaps(mem(rbx, 0*16), xmm8) // xmm8  = ( beta00 beta01 )
		movaps(mem(rbx, 1*16), xmm12) // xmm9  = ( beta02 beta03 )
		movaps(mem(rbx, 2*16), xmm9) // xmm10 = ( beta10 beta11 )
		movaps(mem(rbx, 3*16), xmm13) // xmm11 = ( beta12 beta13 )
		movaps(mem(rbx, 4*16), xmm10) // xmm12 = ( beta20 beta21 )
		movaps(mem(rbx, 5*16), xmm14) // xmm13 = ( beta22 beta23 )
		movaps(mem(rbx, 6*16), xmm11) // xmm14 = ( beta30 beta31 )
		movaps(mem(rbx, 7*16), xmm15) // xmm15 = ( beta32 beta33 )



		mov(var(a11), rax) // load address of a11
		mov(var(c11), rcx) // load address of c11

		mov(var(rs_c), rsi) // load rs_c
		mov(var(cs_c), rdi) // load cs_c
		sal(imm(3), rsi) // rs_c *= sizeof( double )
		sal(imm(3), rdi) // cs_c *= sizeof( double )

		add(rsi, rcx) // c11 += (4-1)*rs_c
		add(rsi, rcx)
		add(rsi, rcx)
		lea(mem(rcx, rdi, 2), rdx) // c11_2 = c11 + 2*cs_c;



		 // iteration 0

		movddup(mem(3+3*4)*8(rax), xmm3) // load xmm3 = (1/alpha33)

		mulpd(xmm3, xmm11) // xmm11 *= (1/alpha33);
		mulpd(xmm3, xmm15) // xmm15 *= (1/alpha33);

		movaps(xmm11, mem(rbx, 6*16)) // store ( beta30 beta31 ) = xmm11
		movaps(xmm15, mem(rbx, 7*16)) // store ( beta32 beta33 ) = xmm15
		movlpd(xmm11, mem(rcx)) // store ( gamma30 ) = xmm11[0]
		movhpd(xmm11, mem(rcx, rdi, 1)) // store ( gamma31 ) = xmm11[1]
		movlpd(xmm15, mem(rdx)) // store ( gamma32 ) = xmm15[0]
		movhpd(xmm15, mem(rdx, rdi, 1)) // store ( gamma33 ) = xmm15[1]
		sub(rsi, rcx) // c11   -= rs_c
		sub(rsi, rdx) // c11_2 -= rs_c



		 // iteration 1

		movddup(mem(2+2*4)*8(rax), xmm2) // load xmm2 = (1/alpha22)
		movddup(mem(2+3*4)*8(rax), xmm3) // load xmm3 = alpha23

		movaps(xmm3, xmm7) // xmm7 = xmm3
		mulpd(xmm11, xmm3) // xmm3 = alpha23 * ( beta30 beta31 )
		mulpd(xmm15, xmm7) // xmm7 = alpha23 * ( beta32 beta33 )
		subpd(xmm3, xmm10) // xmm10 -= xmm3
		subpd(xmm7, xmm14) // xmm14 -= xmm7
		mulpd(xmm2, xmm10) // xmm10 *= (1/alpha22);
		mulpd(xmm2, xmm14) // xmm14 *= (1/alpha22);

		movaps(xmm10, mem(rbx, 4*16)) // store ( beta20 beta21 ) = xmm10
		movaps(xmm14, mem(rbx, 5*16)) // store ( beta22 beta23 ) = xmm14
		movlpd(xmm10, mem(rcx)) // store ( gamma20 ) = xmm10[0]
		movhpd(xmm10, mem(rcx, rdi, 1)) // store ( gamma21 ) = xmm10[1]
		movlpd(xmm14, mem(rdx)) // store ( gamma22 ) = xmm14[0]
		movhpd(xmm14, mem(rdx, rdi, 1)) // store ( gamma23 ) = xmm14[1]
		sub(rsi, rcx) // c11   -= rs_c
		sub(rsi, rdx) // c11_2 -= rs_c



		 // iteration 2

		movddup(mem(1+1*4)*8(rax), xmm1) // load xmm1 = (1/alpha11)
		movddup(mem(1+2*4)*8(rax), xmm2) // load xmm2 = alpha12
		movddup(mem(1+3*4)*8(rax), xmm3) // load xmm3 = alpha13

		movaps(xmm2, xmm6) // xmm6 = xmm2
		movaps(xmm3, xmm7) // xmm7 = xmm3
		mulpd(xmm10, xmm2) // xmm2 = alpha12 * ( beta20 beta21 )
		mulpd(xmm14, xmm6) // xmm6 = alpha12 * ( beta22 beta23 )
		mulpd(xmm11, xmm3) // xmm3 = alpha13 * ( beta30 beta31 )
		mulpd(xmm15, xmm7) // xmm7 = alpha13 * ( beta32 beta33 )
		addpd(xmm3, xmm2) // xmm2 += xmm3;
		addpd(xmm7, xmm6) // xmm6 += xmm7;
		subpd(xmm2, xmm9) // xmm9  -= xmm2
		subpd(xmm6, xmm13) // xmm13 -= xmm6
		mulpd(xmm1, xmm9) // xmm9  *= (1/alpha11);
		mulpd(xmm1, xmm13) // xmm13 *= (1/alpha11);

		movaps(xmm9, mem(rbx, 2*16)) // store ( beta10 beta11 ) = xmm9
		movaps(xmm13, mem(rbx, 3*16)) // store ( beta12 beta13 ) = xmm13
		movlpd(xmm9, mem(rcx)) // store ( gamma10 ) = xmm9[0]
		movhpd(xmm9, mem(rcx, rdi, 1)) // store ( gamma11 ) = xmm9[1]
		movlpd(xmm13, mem(rdx)) // store ( gamma12 ) = xmm13[0]
		movhpd(xmm13, mem(rdx, rdi, 1)) // store ( gamma13 ) = xmm13[1]
		sub(rsi, rcx) // c11   -= rs_c
		sub(rsi, rdx) // c11_2 -= rs_c



		 // iteration 3

		movddup(mem(0+0*4)*8(rax), xmm0) // load xmm0 = (1/alpha00)
		movddup(mem(0+1*4)*8(rax), xmm1) // load xmm1 = alpha01
		movddup(mem(0+2*4)*8(rax), xmm2) // load xmm2 = alpha02
		movddup(mem(0+3*4)*8(rax), xmm3) // load xmm3 = alpha03

		movaps(xmm1, xmm5) // xmm5 = xmm1
		movaps(xmm2, xmm6) // xmm6 = xmm2
		movaps(xmm3, xmm7) // xmm7 = xmm3
		mulpd(xmm9, xmm1) // xmm1 = alpha01 * ( beta10 beta11 )
		mulpd(xmm13, xmm5) // xmm5 = alpha01 * ( beta12 beta13 )
		mulpd(xmm10, xmm2) // xmm2 = alpha02 * ( beta20 beta21 )
		mulpd(xmm14, xmm6) // xmm6 = alpha02 * ( beta22 beta23 )
		mulpd(xmm11, xmm3) // xmm3 = alpha03 * ( beta30 beta31 )
		mulpd(xmm15, xmm7) // xmm7 = alpha03 * ( beta32 beta33 )
		addpd(xmm2, xmm1) // xmm1 += xmm2;
		addpd(xmm6, xmm5) // xmm5 += xmm6;
		addpd(xmm3, xmm1) // xmm1 += xmm3;
		addpd(xmm7, xmm5) // xmm5 += xmm7;
		subpd(xmm1, xmm8) // xmm8  -= xmm1
		subpd(xmm5, xmm12) // xmm12 -= xmm5
		mulpd(xmm0, xmm8) // xmm8  *= (1/alpha00);
		mulpd(xmm0, xmm12) // xmm12 *= (1/alpha00);

		movaps(xmm8, mem(rbx, 0*16)) // store ( beta00 beta01 ) = xmm8
		movaps(xmm12, mem(rbx, 1*16)) // store ( beta02 beta03 ) = xmm12
		movlpd(xmm8, mem(rcx)) // store ( gamma00 ) = xmm8[0]
		movhpd(xmm8, mem(rcx, rdi, 1)) // store ( gamma01 ) = xmm8[1]
		movlpd(xmm12, mem(rdx)) // store ( gamma02 ) = xmm12[0]
		movhpd(xmm12, mem(rdx, rdi, 1)) // store ( gamma03 ) = xmm12[1]




    end_asm(
		: // output operands (none)
		: // input operands
		  [a11]  "m" (a11),    // 0
		  [b11]  "m" (b11),    // 1
		  [c11]  "m" (c11),    // 2
		  [rs_c] "m" (rs_c),   // 3
		  [cs_c] "m" (cs_c)    // 4
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	)

}


