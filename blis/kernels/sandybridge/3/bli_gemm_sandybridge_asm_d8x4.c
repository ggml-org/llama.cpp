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
   of code found in OpenBLAS 0.2.8 (http://www.openblas.net/). -FGVZ */

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

void bli_sgemm_sandybridge_asm_8x8
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
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( s, 8, 8, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	//mov(var(b_next), r15) // load address of b_next.

	vmovaps(mem(rax, 0*32), ymm0) // initialize loop by pre-loading
	vmovsldup(mem(rbx, 0*32), ymm2) // elements of a and b.
	vpermilps(imm(0x4e), ymm2, ymm3)

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 4), rdi) // cs_c *= sizeof(float)
	lea(mem(rcx, rdi, 4), r10) // load address of c + 4*cs_c;

	lea(mem(rdi, rdi, 2), r14) // r14 = 3*cs_c;
	prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, r14, 1, 7*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(r10, 7*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(r10, rdi, 1, 7*8)) // prefetch c + 5*cs_c
	prefetch(0, mem(r10, rdi, 2, 7*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(r10, r14, 1, 7*8)) // prefetch c + 7*cs_c

	vxorps(ymm8, ymm8, ymm8)
	vxorps(ymm9, ymm9, ymm9)
	vxorps(ymm10, ymm10, ymm10)
	vxorps(ymm11, ymm11, ymm11)
	vxorps(ymm12, ymm12, ymm12)
	vxorps(ymm13, ymm13, ymm13)
	vxorps(ymm14, ymm14, ymm14)
	vxorps(ymm15, ymm15, ymm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.SCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.SLOOPKITER) // MAIN LOOP


	 // iteration 0
	prefetch(0, mem(rax, 16*32))
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovshdup(mem(rbx, 0*32), ymm2)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm15, ymm6, ymm15)
	vaddps(ymm13, ymm7, ymm13)

	vmovaps(mem(rax, 1*32), ymm1)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vaddps(ymm11, ymm6, ymm11)
	vaddps(ymm9, ymm7, ymm9)

	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovsldup(mem(rbx, 1*32), ymm2)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm14, ymm6, ymm14)
	vaddps(ymm12, ymm7, ymm12)

	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vaddps(ymm10, ymm6, ymm10)
	vaddps(ymm8, ymm7, ymm8)

	 // iteration 1
	vmulps(ymm1, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovshdup(mem(rbx, 1*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm15, ymm6, ymm15)
	vaddps(ymm13, ymm7, ymm13)

	vmovaps(mem(rax, 2*32), ymm0)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddps(ymm11, ymm6, ymm11)
	vaddps(ymm9, ymm7, ymm9)

	vmulps(ymm1, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovsldup(mem(rbx, 2*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm14, ymm6, ymm14)
	vaddps(ymm12, ymm7, ymm12)

	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddps(ymm10, ymm6, ymm10)
	vaddps(ymm8, ymm7, ymm8)


	 // iteration 2
	prefetch(0, mem(rax, 18*32))
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovshdup(mem(rbx, 2*32), ymm2)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm15, ymm6, ymm15)
	vaddps(ymm13, ymm7, ymm13)

	vmovaps(mem(rax, 3*32), ymm1)
	add(imm(4*8*4), rax) // a += 4*8 (unroll x mr)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vaddps(ymm11, ymm6, ymm11)
	vaddps(ymm9, ymm7, ymm9)

	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovsldup(mem(rbx, 3*32), ymm2)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm14, ymm6, ymm14)
	vaddps(ymm12, ymm7, ymm12)

	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vaddps(ymm10, ymm6, ymm10)
	vaddps(ymm8, ymm7, ymm8)


	 // iteration 3
	vmulps(ymm1, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovshdup(mem(rbx, 3*32), ymm2)
	add(imm(4*8*4), rbx) // b += 4*8 (unroll x nr)
	vmulps(ymm1, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm15, ymm6, ymm15)
	vaddps(ymm13, ymm7, ymm13)

	vmovaps(mem(rax, 0*32), ymm0)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddps(ymm11, ymm6, ymm11)
	vaddps(ymm9, ymm7, ymm9)

	vmulps(ymm1, ymm2, ymm6)
	vperm2f128(imm(0x03), ymm2, ymm2, ymm4)
	vmovsldup(mem(rbx, 0*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vperm2f128(imm(0x03), ymm3, ymm3, ymm5)
	vaddps(ymm14, ymm6, ymm14)
	vaddps(ymm12, ymm7, ymm12)

	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddps(ymm10, ymm6, ymm10)
	vaddps(ymm8, ymm7, ymm8)




	dec(rsi) // i -= 1;
	jne(.SLOOPKITER) // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.SPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT) // EDGE LOOP


	prefetch(0, mem(rax, 16*32))
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmovshdup(mem(rbx, 0*32), ymm2)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm15, ymm6, ymm15)
	vaddps(ymm13, ymm7, ymm13)

	vmovaps(mem(rax, 1*32), ymm1)
	add(imm(8*1*4), rax) // a += 8 (1 x mr)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vaddps(ymm11, ymm6, ymm11)
	vaddps(ymm9, ymm7, ymm9)

	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmovsldup(mem(rbx, 1*32), ymm2)
	add(imm(8*1*4), rbx) // b += 8 (1 x nr)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm14, ymm6, ymm14)
	vaddps(ymm12, ymm7, ymm12)

	vpermilps(imm(0x4e), ymm2, ymm3)
	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vmovaps(ymm1, ymm0)
	vaddps(ymm10, ymm6, ymm10)
	vaddps(ymm8, ymm7, ymm8)



	dec(rsi) // i -= 1;
	jne(.SLOOPKLEFT) // iterate again if i != 0.



	label(.SPOSTACCUM)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab02  ( ab04  ( ab06
	 //   ab10    ab12    ab14    ab16
	 //   ab22    ab20    ab26    ab24
	 //   ab32    ab30    ab36    ab34
	 //   ab44    ab46    ab40    ab42
	 //   ab54    ab56    ab50    ab52
	 //   ab66    ab64    ab62    ab60
	 //   ab76 )  ab74 )  ab72 )  ab70 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab01  ( ab03  ( ab05  ( ab07
	 //   ab11    ab13    ab15    ab17
	 //   ab23    ab21    ab27    ab25
	 //   ab33    ab31    ab37    ab35
	 //   ab45    ab47    ab41    ab43
	 //   ab55    ab57    ab51    ab53
	 //   ab67    ab65    ab63    ab61
	 //   ab77 )  ab75 )  ab73 )  ab71 )

	vmovaps(ymm15, ymm7)
	vshufps(imm(0xe4), ymm13, ymm15, ymm15)
	vshufps(imm(0xe4), ymm7, ymm13, ymm13)

	vmovaps(ymm11, ymm7)
	vshufps(imm(0xe4), ymm9, ymm11, ymm11)
	vshufps(imm(0xe4), ymm7, ymm9, ymm9)

	vmovaps(ymm14, ymm7)
	vshufps(imm(0xe4), ymm12, ymm14, ymm14)
	vshufps(imm(0xe4), ymm7, ymm12, ymm12)

	vmovaps(ymm10, ymm7)
	vshufps(imm(0xe4), ymm8, ymm10, ymm10)
	vshufps(imm(0xe4), ymm7, ymm8, ymm8)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab02  ( ab04  ( ab06
	 //   ab10    ab12    ab14    ab16
	 //   ab20    ab22    ab24    ab26
	 //   ab30    ab32    ab34    ab36
	 //   ab44    ab46    ab40    ab42
	 //   ab54    ab56    ab50    ab52
	 //   ab64    ab66    ab60    ab62
	 //   ab74 )  ab76 )  ab70 )  ab72 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab01  ( ab03  ( ab05  ( ab07
	 //   ab11    ab13    ab15    ab17
	 //   ab21    ab23    ab25    ab27
	 //   ab31    ab33    ab35    ab37
	 //   ab45    ab47    ab41    ab43
	 //   ab55    ab57    ab51    ab53
	 //   ab65    ab67    ab61    ab63
	 //   ab75 )  ab77 )  ab71 )  ab73 )

	vmovaps(ymm15, ymm7)
	vperm2f128(imm(0x30), ymm11, ymm15, ymm15)
	vperm2f128(imm(0x12), ymm11, ymm7, ymm11)

	vmovaps(ymm13, ymm7)
	vperm2f128(imm(0x30), ymm9, ymm13, ymm13)
	vperm2f128(imm(0x12), ymm9, ymm7, ymm9)

	vmovaps(ymm14, ymm7)
	vperm2f128(imm(0x30), ymm10, ymm14, ymm14)
	vperm2f128(imm(0x12), ymm10, ymm7, ymm10)

	vmovaps(ymm12, ymm7)
	vperm2f128(imm(0x30), ymm8, ymm12, ymm12)
	vperm2f128(imm(0x12), ymm8, ymm7, ymm8)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab02  ( ab04  ( ab06
	 //   ab10    ab12    ab14    ab16
	 //   ab20    ab22    ab24    ab26
	 //   ab30    ab32    ab34    ab36
	 //   ab40    ab42    ab44    ab46
	 //   ab50    ab52    ab54    ab56
	 //   ab60    ab62    ab64    ab66
	 //   ab70 )  ab72 )  ab74 )  ab76 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab01  ( ab03  ( ab05  ( ab07
	 //   ab11    ab13    ab15    ab17
	 //   ab21    ab23    ab25    ab27
	 //   ab31    ab33    ab35    ab37
	 //   ab41    ab43    ab45    ab47
	 //   ab51    ab53    ab55    ab57
	 //   ab61    ab63    ab65    ab67
	 //   ab71 )  ab73 )  ab75 )  ab77 )



	mov(var(alpha), rax) // load address of alpha
	mov(var(beta), rbx) // load address of beta
	vbroadcastss(mem(rax), ymm0) // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm4) // load beta and duplicate

	vmulps(ymm0, ymm8, ymm8) // scale by alpha
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm0, ymm10, ymm10)
	vmulps(ymm0, ymm11, ymm11)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm0, ymm13, ymm13)
	vmulps(ymm0, ymm14, ymm14)
	vmulps(ymm0, ymm15, ymm15)


	mov(var(rs_c), rsi) // load rs_c
	lea(mem(, rsi, 4), rsi) // rsi = rs_c * sizeof(float)

	lea(mem(rcx, rsi, 4), rdx) // load address of c + 4*rs_c;

	lea(mem(, rsi, 2), r12) // r12 = 2*rs_c;
	lea(mem(r12, rsi, 1), r13) // r13 = 3*rs_c;


	 // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomiss(xmm0, xmm4) // set ZF if beta == 0.
	je(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

		vmovups(mem(rcx), ymm0) // load c00:c70,
		vmulps(ymm4, ymm0, ymm0) // scale by beta,
		vaddps(ymm15, ymm0, ymm0) // add the gemm result,
		vmovups(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm1) // load c01:c71,
		vmulps(ymm4, ymm1, ymm1) // scale by beta,
		vaddps(ymm14, ymm1, ymm1) // add the gemm result,
		vmovups(ymm1, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm0) // load c02:c72,
		vmulps(ymm4, ymm0, ymm0) // scale by beta,
		vaddps(ymm13, ymm0, ymm0) // add the gemm result,
		vmovups(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm1) // load c03:c73,
		vmulps(ymm4, ymm1, ymm1) // scale by beta,
		vaddps(ymm12, ymm1, ymm1) // add the gemm result,
		vmovups(ymm1, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm0) // load c04:c74,
		vmulps(ymm4, ymm0, ymm0) // scale by beta,
		vaddps(ymm11, ymm0, ymm0) // add the gemm result,
		vmovups(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm1) // load c05:c75,
		vmulps(ymm4, ymm1, ymm1) // scale by beta,
		vaddps(ymm10, ymm1, ymm1) // add the gemm result,
		vmovups(ymm1, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm0) // load c06:c76,
		vmulps(ymm4, ymm0, ymm0) // scale by beta,
		vaddps(ymm9, ymm0, ymm0) // add the gemm result,
		vmovups(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(mem(rcx), ymm1) // load c07:c77,
		vmulps(ymm4, ymm1, ymm1) // scale by beta,
		vaddps(ymm8, ymm1, ymm1) // add the gemm result,
		vmovups(ymm1, mem(rcx)) // and store back to memory.

		jmp(.SDONE) // jump to end.

	label(.SBETAZERO)

		vmovups(ymm15, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm14, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm13, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm12, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm11, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm10, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm9, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm8, mem(rcx)) // and store back to memory.

	label(.SDONE)

	vzeroupper()


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
	  [cs_c]   "m" (cs_c)/*,   // 8
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)*/  // 10
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

void bli_dgemm_sandybridge_asm_8x4
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
	//const void*   a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( d, 8, 4, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	mov(var(b_next), r15) // load address of b_next.
	//mov(var(a_next), r14) // load address of a_next.
	sub(imm(4*64), r15)

	vmovapd(mem(rax, 0*32), ymm0) // initialize loop by pre-loading
	vmovapd(mem(rbx, 0*32), ymm2) // elements of a and b.
	vpermilpd(imm(0x5), ymm2, ymm3)

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(double)
	lea(mem(rcx, rdi, 2), r10) // load address of c + 2*cs_c;

	prefetch(0, mem(rcx, 3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r10, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(r10, rdi, 1, 3*8)) // prefetch c + 3*cs_c

	vxorpd(ymm8, ymm8, ymm8)
	vxorpd(ymm9, ymm9, ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.DCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.DLOOPKITER) // MAIN LOOP

	add(imm(4*4*8), r15) // b_next += 4*4 (unroll x nr)

	 // iteration 0
	vmovapd(mem(rax, 1*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm15, ymm6, ymm15)
	vaddpd(ymm13, ymm7, ymm13)

	prefetch(0, mem(rax, 16*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovapd(mem(rbx, 1*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vpermilpd(imm(0x5), ymm2, ymm3)
	vaddpd(ymm14, ymm6, ymm14)
	vaddpd(ymm12, ymm7, ymm12)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 2*32), ymm0)
	vaddpd(ymm11, ymm6, ymm11)
	vaddpd(ymm9, ymm7, ymm9)
	prefetch(0, mem(r15, 0*32)) // prefetch b_next[0*4]

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddpd(ymm10, ymm6, ymm10)
	vaddpd(ymm8, ymm7, ymm8)


	 // iteration 1
	vmovapd(mem(rax, 3*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm15, ymm6, ymm15)
	vaddpd(ymm13, ymm7, ymm13)

	prefetch(0, mem(rax, 18*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovapd(mem(rbx, 2*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vpermilpd(imm(0x5), ymm2, ymm3)
	vaddpd(ymm14, ymm6, ymm14)
	vaddpd(ymm12, ymm7, ymm12)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 4*32), ymm0)
	vaddpd(ymm11, ymm6, ymm11)
	vaddpd(ymm9, ymm7, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddpd(ymm10, ymm6, ymm10)
	vaddpd(ymm8, ymm7, ymm8)


	 // iteration 2
	vmovapd(mem(rax, 5*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm15, ymm6, ymm15)
	vaddpd(ymm13, ymm7, ymm13)

	prefetch(0, mem(rax, 20*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovapd(mem(rbx, 3*32), ymm2)
	add(imm(4*4*8), rbx) // b += 4*4 (unroll x nr)
	vmulpd(ymm1, ymm3, ymm7)
	vpermilpd(imm(0x5), ymm2, ymm3)
	vaddpd(ymm14, ymm6, ymm14)
	vaddpd(ymm12, ymm7, ymm12)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 6*32), ymm0)
	vaddpd(ymm11, ymm6, ymm11)
	vaddpd(ymm9, ymm7, ymm9)
	prefetch(0, mem(r15, 2*32)) // prefetch b_next[2*4]

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddpd(ymm10, ymm6, ymm10)
	vaddpd(ymm8, ymm7, ymm8)


	 // iteration 3
	vmovapd(mem(rax, 7*32), ymm1)
	add(imm(4*8*8), rax) // a += 4*8 (unroll x mr)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm15, ymm6, ymm15)
	vaddpd(ymm13, ymm7, ymm13)

	//prefetch(0, mem(rax, 22*32))
	prefetch(0, mem(rax, 14*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovapd(mem(rbx, 0*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vpermilpd(imm(0x5), ymm2, ymm3)
	vaddpd(ymm14, ymm6, ymm14)
	vaddpd(ymm12, ymm7, ymm12)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 0*32), ymm0)
	vaddpd(ymm11, ymm6, ymm11)
	vaddpd(ymm9, ymm7, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddpd(ymm10, ymm6, ymm10)
	vaddpd(ymm8, ymm7, ymm8)



	//add(imm(4*8*8), rax) // a      += 4*8 (unroll x mr)
	//add(imm(4*4*8), rbx) // b      += 4*4 (unroll x nr)

	dec(rsi) // i -= 1;
	jne(.DLOOPKITER) // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.DPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT) // EDGE LOOP

	vmovapd(mem(rax, 1*32), ymm1)
	add(imm(8*1*8), rax) // a += 8 (1 x mr)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm15, ymm6, ymm15)
	vaddpd(ymm13, ymm7, ymm13)

	prefetch(0, mem(rax, 14*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovapd(mem(rbx, 1*32), ymm2)
	add(imm(4*1*8), rbx) // b += 4 (1 x nr)
	vmulpd(ymm1, ymm3, ymm7)
	vpermilpd(imm(0x5), ymm2, ymm3)
	vaddpd(ymm14, ymm6, ymm14)
	vaddpd(ymm12, ymm7, ymm12)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 0*32), ymm0)
	vaddpd(ymm11, ymm6, ymm11)
	vaddpd(ymm9, ymm7, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddpd(ymm10, ymm6, ymm10)
	vaddpd(ymm8, ymm7, ymm8)


	dec(rsi) // i -= 1;
	jne(.DLOOPKLEFT) // iterate again if i != 0.



	label(.DPOSTACCUM)


	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab11    ab10    ab13    ab12
	 //   ab22    ab23    ab20    ab21
	 //   ab33 )  ab32 )  ab31 )  ab30 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab40  ( ab41  ( ab42  ( ab43
	 //   ab51    ab50    ab53    ab52
	 //   ab62    ab63    ab60    ab61
	 //   ab73 )  ab72 )  ab71 )  ab70 )

	vmovapd(ymm15, ymm7)
	vshufpd(imm(0xa), ymm15, ymm13, ymm15)
	vshufpd(imm(0xa), ymm13, ymm7, ymm13)

	vmovapd(ymm11, ymm7)
	vshufpd(imm(0xa), ymm11, ymm9, ymm11)
	vshufpd(imm(0xa), ymm9, ymm7, ymm9)

	vmovapd(ymm14, ymm7)
	vshufpd(imm(0xa), ymm14, ymm12, ymm14)
	vshufpd(imm(0xa), ymm12, ymm7, ymm12)

	vmovapd(ymm10, ymm7)
	vshufpd(imm(0xa), ymm10, ymm8, ymm10)
	vshufpd(imm(0xa), ymm8, ymm7, ymm8)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab01  ( ab00  ( ab03  ( ab02
	 //   ab11    ab10    ab13    ab12
	 //   ab23    ab22    ab21    ab20
	 //   ab33 )  ab32 )  ab31 )  ab30 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab41  ( ab40  ( ab43  ( ab42
	 //   ab51    ab50    ab53    ab52
	 //   ab63    ab62    ab61    ab60
	 //   ab73 )  ab72 )  ab71 )  ab70 )

	vmovapd(ymm15, ymm7)
	vperm2f128(imm(0x30), ymm15, ymm11, ymm15)
	vperm2f128(imm(0x12), ymm7, ymm11, ymm11)

	vmovapd(ymm13, ymm7)
	vperm2f128(imm(0x30), ymm13, ymm9, ymm13)
	vperm2f128(imm(0x12), ymm7, ymm9, ymm9)

	vmovapd(ymm14, ymm7)
	vperm2f128(imm(0x30), ymm14, ymm10, ymm14)
	vperm2f128(imm(0x12), ymm7, ymm10, ymm10)

	vmovapd(ymm12, ymm7)
	vperm2f128(imm(0x30), ymm12, ymm8, ymm12)
	vperm2f128(imm(0x12), ymm7, ymm8, ymm8)

	 // ymm9:   ymm11:  ymm13:  ymm15:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab10    ab11    ab12    ab13
	 //   ab20    ab21    ab22    ab23
	 //   ab30 )  ab31 )  ab32 )  ab33 )

	 // ymm8:   ymm10:  ymm12:  ymm14:
	 // ( ab40  ( ab41  ( ab42  ( ab43
	 //   ab50    ab51    ab52    ab53
	 //   ab60    ab61    ab62    ab63
	 //   ab70 )  ab71 )  ab72 )  ab73 )


	mov(var(alpha), rax) // load address of alpha
	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rax), ymm0) // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm2) // load beta and duplicate

	vmulpd(ymm0, ymm8, ymm8) // scale by alpha
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)


	mov(var(rs_c), rsi) // load rs_c
	lea(mem(, rsi, 8), rsi) // rsi = rs_c * sizeof(double)

	lea(mem(rcx, rsi, 4), rdx) // load address of c + 4*rs_c;

	lea(mem(, rsi, 2), r12) // r12 = 2*rs_c;
	lea(mem(r12, rsi, 1), r13) // r13 = 3*rs_c;


	 // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm2) // set ZF if beta == 0.
	je(.DBETAZERO) // if ZF = 1, jump to beta == 0 case

		 // update c00:c33

		vmovupd(mem(rcx), ymm0) // load c00:c30,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm9, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovupd(mem(rcx), ymm0) // load c01:c31,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm11, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovupd(mem(rcx), ymm0) // load c02:c32,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm13, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rcx)) // and store back to memory.
		add(rdi, rcx) // c += cs_c;

		vmovupd(mem(rcx), ymm0) // load c03:c33,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm15, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rcx)) // and store back to memory.

		 // update c40:c73

		vmovupd(mem(rdx), ymm0) // load c40:c70,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm8, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rdx)) // and store back to memory.
		add(rdi, rdx) // c += cs_c;

		vmovupd(mem(rdx), ymm0) // load c41:c71,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm10, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rdx)) // and store back to memory.
		add(rdi, rdx) // c += cs_c;

		vmovupd(mem(rdx), ymm0) // load c42:c72,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm12, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rdx)) // and store back to memory.
		add(rdi, rdx) // c += cs_c;

		vmovupd(mem(rdx), ymm0) // load c43:c73,
		vmulpd(ymm2, ymm0, ymm0) // scale by beta,
		vaddpd(ymm14, ymm0, ymm0) // add the gemm result,
		vmovupd(ymm0, mem(rdx)) // and store back to memory.

		jmp(.DDONE) // jump to end.

	label(.DBETAZERO)

		 // update c00:c33

		vmovupd(ymm9, mem(rcx)) // store c00:c30
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm11, mem(rcx)) // store c01:c31
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm13, mem(rcx)) // store c02:c32
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm15, mem(rcx)) // store c03:c33

		 // update c40:c73

		vmovupd(ymm8, mem(rdx)) // store c40:c70
		add(rdi, rdx) // c += cs_c;

		vmovupd(ymm10, mem(rdx)) // store c41:c71
		add(rdi, rdx) // c += cs_c;

		vmovupd(ymm12, mem(rdx)) // store c42:c72
		add(rdi, rdx) // c += cs_c;

		vmovupd(ymm14, mem(rdx)) // store c43:c73

	label(.DDONE)

	vzeroupper()


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

void bli_cgemm_sandybridge_asm_8x4
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
	//const void*   a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( c, 8, 4, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	mov(var(b_next), r15) // load address of b_next.
	//mov(var(a_next), r14) // load address of a_next.
	sub(imm(4*64), r15)

	vmovaps(mem(rax, 0*32), ymm0) // initialize loop by pre-loading
	vmovsldup(mem(rbx, 0*32), ymm2)
	vpermilps(imm(0x4e), ymm2, ymm3)

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(scomplex)
	lea(mem(rcx, rdi, 2), r10) // load address of c + 2*cs_c;

	prefetch(0, mem(rcx, 3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r10, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(r10, rdi, 1, 3*8)) // prefetch c + 3*cs_c

	vxorps(ymm8, ymm8, ymm8)
	vxorps(ymm9, ymm9, ymm9)
	vxorps(ymm10, ymm10, ymm10)
	vxorps(ymm11, ymm11, ymm11)
	vxorps(ymm12, ymm12, ymm12)
	vxorps(ymm13, ymm13, ymm13)
	vxorps(ymm14, ymm14, ymm14)
	vxorps(ymm15, ymm15, ymm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.CCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.CLOOPKITER) // MAIN LOOP

	add(imm(4*4*8), r15) // b_next += 4*4 (unroll x nr)

	 // iteration 0
	prefetch(0, mem(rax, 8*32))
	vmovaps(mem(rax, 1*32), ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm15, ymm15)
	vaddps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovshdup(mem(rbx, 0*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddps(ymm6, ymm14, ymm14)
	vaddps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vpermilps(imm(0xb1), ymm0, ymm0)
	vaddps(ymm6, ymm11, ymm11)
	vaddps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm10, ymm10)
	vaddps(ymm7, ymm8, ymm8)
	prefetch(0, mem(r15, 0*32)) // prefetch b_next[0*4]

	vpermilps(imm(0xb1), ymm1, ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vmulps(ymm0, ymm3, ymm7)
	vaddsubps(ymm6, ymm15, ymm15)
	vaddsubps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovsldup(mem(rbx, 1*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddsubps(ymm6, ymm14, ymm14)
	vaddsubps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vmovaps(mem(rax, 2*32), ymm0)
	vaddsubps(ymm6, ymm11, ymm11)
	vaddsubps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddsubps(ymm6, ymm10, ymm10)
	vaddsubps(ymm7, ymm8, ymm8)


	 // iteration 1
	prefetch(0, mem(rax, 10*32))
	vmovaps(mem(rax, 3*32), ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm15, ymm15)
	vaddps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovshdup(mem(rbx, 1*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddps(ymm6, ymm14, ymm14)
	vaddps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vpermilps(imm(0xb1), ymm0, ymm0)
	vaddps(ymm6, ymm11, ymm11)
	vaddps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm10, ymm10)
	vaddps(ymm7, ymm8, ymm8)

	vpermilps(imm(0xb1), ymm1, ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vmulps(ymm0, ymm3, ymm7)
	vaddsubps(ymm6, ymm15, ymm15)
	vaddsubps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovsldup(mem(rbx, 2*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddsubps(ymm6, ymm14, ymm14)
	vaddsubps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vmovaps(mem(rax, 4*32), ymm0)
	vaddsubps(ymm6, ymm11, ymm11)
	vaddsubps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddsubps(ymm6, ymm10, ymm10)
	vaddsubps(ymm7, ymm8, ymm8)


	 // iteration 2
	prefetch(0, mem(rax, 12*32))
	vmovaps(mem(rax, 5*32), ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm15, ymm15)
	vaddps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovshdup(mem(rbx, 2*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddps(ymm6, ymm14, ymm14)
	vaddps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vpermilps(imm(0xb1), ymm0, ymm0)
	vaddps(ymm6, ymm11, ymm11)
	vaddps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm10, ymm10)
	vaddps(ymm7, ymm8, ymm8)
	prefetch(0, mem(r15, 2*32)) // prefetch b_next[2*4]

	vpermilps(imm(0xb1), ymm1, ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vmulps(ymm0, ymm3, ymm7)
	vaddsubps(ymm6, ymm15, ymm15)
	vaddsubps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovsldup(mem(rbx, 3*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddsubps(ymm6, ymm14, ymm14)
	vaddsubps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vmovaps(mem(rax, 6*32), ymm0)
	vaddsubps(ymm6, ymm11, ymm11)
	vaddsubps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddsubps(ymm6, ymm10, ymm10)
	vaddsubps(ymm7, ymm8, ymm8)


	 // iteration 3
	prefetch(0, mem(rax, 14*32))
	vmovaps(mem(rax, 7*32), ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm15, ymm15)
	vaddps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovshdup(mem(rbx, 3*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddps(ymm6, ymm14, ymm14)
	vaddps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vpermilps(imm(0xb1), ymm0, ymm0)
	vaddps(ymm6, ymm11, ymm11)
	vaddps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm10, ymm10)
	vaddps(ymm7, ymm8, ymm8)

	vpermilps(imm(0xb1), ymm1, ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vmulps(ymm0, ymm3, ymm7)
	vaddsubps(ymm6, ymm15, ymm15)
	vaddsubps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovsldup(mem(rbx, 4*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddsubps(ymm6, ymm14, ymm14)
	vaddsubps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vmovaps(mem(rax, 8*32), ymm0)
	vaddsubps(ymm6, ymm11, ymm11)
	vaddsubps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddsubps(ymm6, ymm10, ymm10)
	vaddsubps(ymm7, ymm8, ymm8)


	add(imm(8*4*8), rax) // a += 8*4 (unroll x mr)
	add(imm(4*4*8), rbx) // b += 4*4 (unroll x nr)


	dec(rsi) // i -= 1;
	jne(.CLOOPKITER) // iterate again if i != 0.






	label(.CCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.CPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.CLOOPKLEFT) // EDGE LOOP

	 // iteration 0
	prefetch(0, mem(rax, 8*32))
	vmovaps(mem(rax, 1*32), ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm15, ymm15)
	vaddps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovshdup(mem(rbx, 0*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddps(ymm6, ymm14, ymm14)
	vaddps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vpermilps(imm(0xb1), ymm0, ymm0)
	vaddps(ymm6, ymm11, ymm11)
	vaddps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulps(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddps(ymm6, ymm10, ymm10)
	vaddps(ymm7, ymm8, ymm8)

	vpermilps(imm(0xb1), ymm1, ymm1)
	vmulps(ymm0, ymm2, ymm6)
	vmulps(ymm0, ymm3, ymm7)
	vaddsubps(ymm6, ymm15, ymm15)
	vaddsubps(ymm7, ymm13, ymm13)

	vmulps(ymm1, ymm2, ymm6)
	vmovsldup(mem(rbx, 1*32), ymm2)
	vmulps(ymm1, ymm3, ymm7)
	vpermilps(imm(0x4e), ymm2, ymm3)
	vaddsubps(ymm6, ymm14, ymm14)
	vaddsubps(ymm7, ymm12, ymm12)

	vmulps(ymm0, ymm4, ymm6)
	vmulps(ymm0, ymm5, ymm7)
	vmovaps(mem(rax, 2*32), ymm0)
	vaddsubps(ymm6, ymm11, ymm11)
	vaddsubps(ymm7, ymm9, ymm9)

	vmulps(ymm1, ymm4, ymm6)
	vmulps(ymm1, ymm5, ymm7)
	vaddsubps(ymm6, ymm10, ymm10)
	vaddsubps(ymm7, ymm8, ymm8)


	add(imm(8*1*8), rax) // a += 8 (1 x mr)
	add(imm(4*1*8), rbx) // b += 4 (1 x nr)


	dec(rsi) // i -= 1;
	jne(.CLOOPKLEFT) // iterate again if i != 0.



	label(.CPOSTACCUM)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab10    ab11    ab12    ab13
	 //   ab21    ab20    ab23    ab22
	 //   ab31    ab30    ab33    ab32
	 //   ab42    ab43    ab40    ab41
	 //   ab52    ab53    ab50    ab51
	 //   ab63    ab62    ab61    ab60
	 //   ab73 )  ab72 )  ab71 )  ab70 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab80  ( ab81  ( ab82  ( ab83
	 //   ab90    ab91    ab92    ab93
	 //   aba1    aba0    aba3    aba2
	 //   abb1    abb0    abb3    abb2
	 //   abc2    abc3    abc0    abc1
	 //   abd2    abd3    abd0    abd1
	 //   abe3    abe2    abe1    abe0
	 //   abf3    abf2    abf1    abf0 )

	vmovaps(ymm15, ymm7)
	vshufps(imm(0xe4), ymm13, ymm15, ymm15)
	vshufps(imm(0xe4), ymm7, ymm13, ymm13)

	vmovaps(ymm11, ymm7)
	vshufps(imm(0xe4), ymm9, ymm11, ymm11)
	vshufps(imm(0xe4), ymm7, ymm9, ymm9)

	vmovaps(ymm14, ymm7)
	vshufps(imm(0xe4), ymm12, ymm14, ymm14)
	vshufps(imm(0xe4), ymm7, ymm12, ymm12)

	vmovaps(ymm10, ymm7)
	vshufps(imm(0xe4), ymm8, ymm10, ymm10)
	vshufps(imm(0xe4), ymm7, ymm8, ymm8)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab10    ab11    ab12    ab13
	 //   ab20    ab21    ab22    ab23
	 //   ab30    ab31    ab32    ab33
	 //   ab42    ab43    ab40    ab41
	 //   ab52    ab53    ab50    ab51
	 //   ab62    ab63    ab60    ab61
	 //   ab72 )  ab73 )  ab70 )  ab71 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab80  ( ab81  ( ab82  ( ab83
	 //   ab90    ab91    ab92    ab93
	 //   aba0    aba1    aba2    aba3
	 //   abb0    abb1    abb2    abb3
	 //   abc2    abc3    abc0    abc1
	 //   abd2    abd3    abd0    abd1
	 //   abe2    abe3    abe0    abe1
	 //   abf2 )  abf3 )  abf0 )  abf1 )

	vmovaps(ymm15, ymm7)
	vperm2f128(imm(0x12), ymm15, ymm11, ymm15)
	vperm2f128(imm(0x30), ymm7, ymm11, ymm11)

	vmovaps(ymm13, ymm7)
	vperm2f128(imm(0x12), ymm13, ymm9, ymm13)
	vperm2f128(imm(0x30), ymm7, ymm9, ymm9)

	vmovaps(ymm14, ymm7)
	vperm2f128(imm(0x12), ymm14, ymm10, ymm14)
	vperm2f128(imm(0x30), ymm7, ymm10, ymm10)

	vmovaps(ymm12, ymm7)
	vperm2f128(imm(0x12), ymm12, ymm8, ymm12)
	vperm2f128(imm(0x30), ymm7, ymm8, ymm8)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab10    ab11    ab12    ab13
	 //   ab20    ab21    ab22    ab23
	 //   ab30    ab31    ab32    ab33
	 //   ab40    ab41    ab42    ab43
	 //   ab50    ab51    ab52    ab53
	 //   ab60    ab61    ab62    ab63
	 //   ab70 )  ab71 )  ab72 )  ab73 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab80  ( ab81  ( ab82  ( ab83
	 //   ab90    ab91    ab92    ab93
	 //   aba0    aba1    aba2    aba3
	 //   abb0    abb1    abb2    abb3
	 //   abc0    abc1    abc2    abc3
	 //   abd0    abd1    abd2    abd3
	 //   abe0    abe1    abe2    abe3
	 //   abf0 )  abf1 )  abf2 )  abf3 )




	 // scale by alpha

	mov(var(alpha), rax) // load address of alpha
	vbroadcastss(mem(rax), ymm7) // load alpha_r and duplicate
	vbroadcastss(mem(rax, 4), ymm6) // load alpha_i and duplicate

	vpermilps(imm(0xb1), ymm15, ymm3)
	vmulps(ymm7, ymm15, ymm15)
	vmulps(ymm6, ymm3, ymm3)
	vaddsubps(ymm3, ymm15, ymm15)

	vpermilps(imm(0xb1), ymm14, ymm2)
	vmulps(ymm7, ymm14, ymm14)
	vmulps(ymm6, ymm2, ymm2)
	vaddsubps(ymm2, ymm14, ymm14)

	vpermilps(imm(0xb1), ymm13, ymm1)
	vmulps(ymm7, ymm13, ymm13)
	vmulps(ymm6, ymm1, ymm1)
	vaddsubps(ymm1, ymm13, ymm13)

	vpermilps(imm(0xb1), ymm12, ymm0)
	vmulps(ymm7, ymm12, ymm12)
	vmulps(ymm6, ymm0, ymm0)
	vaddsubps(ymm0, ymm12, ymm12)

	vpermilps(imm(0xb1), ymm11, ymm3)
	vmulps(ymm7, ymm11, ymm11)
	vmulps(ymm6, ymm3, ymm3)
	vaddsubps(ymm3, ymm11, ymm11)

	vpermilps(imm(0xb1), ymm10, ymm2)
	vmulps(ymm7, ymm10, ymm10)
	vmulps(ymm6, ymm2, ymm2)
	vaddsubps(ymm2, ymm10, ymm10)

	vpermilps(imm(0xb1), ymm9, ymm1)
	vmulps(ymm7, ymm9, ymm9)
	vmulps(ymm6, ymm1, ymm1)
	vaddsubps(ymm1, ymm9, ymm9)

	vpermilps(imm(0xb1), ymm8, ymm0)
	vmulps(ymm7, ymm8, ymm8)
	vmulps(ymm6, ymm0, ymm0)
	vaddsubps(ymm0, ymm8, ymm8)




	mov(var(beta), rbx) // load address of beta
	vbroadcastss(mem(rbx), ymm7) // load beta_r and duplicate
	vbroadcastss(mem(rbx, 4), ymm6) // load beta_i and duplicate







	mov(var(rs_c), rsi) // load rs_c
	lea(mem(, rsi, 8), rsi) // rsi = rs_c * sizeof(scomplex)

	lea(mem(rcx, rsi, 4), rdx) // load address of c + 4*rs_c;

	lea(mem(, rsi, 2), r12) // r12 = 2*rs_c;
	lea(mem(r12, rsi, 1), r13) // r13 = 3*rs_c;


	 // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomiss(xmm0, xmm7) // set ZF if beta_r == 0.
	sete(r8b) // r8b = ( ZF == 1 ? 1 : 0 );
	vucomiss(xmm0, xmm6) // set ZF if beta_i == 0.
	sete(r9b) // r9b = ( ZF == 1 ? 1 : 0 );
	and(r8b, r9b) // set ZF if r8b & r9b == 1.
	jne(.CBETAZERO) // if ZF = 0, jump to beta == 0 case

		 // update c00:c70

		vmovups(mem(rcx), ymm0) // load c00:c70 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm15, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rcx)) // store c00:c70
		add(rdi, rcx) // c += cs_c;

		 // update c80:cf0

		vmovups(mem(rdx), ymm0) // load c80:f0 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm14, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rdx)) // store c80:cf0
		add(rdi, rdx) // c += cs_c;

		 // update c00:c70

		vmovups(mem(rcx), ymm0) // load c01:c71 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm13, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rcx)) // store c01:c71
		add(rdi, rcx) // c += cs_c;

		 // update c81:cf1

		vmovups(mem(rdx), ymm0) // load c81:f1 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm12, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rdx)) // store c81:cf1
		add(rdi, rdx) // c += cs_c;

		 // update c02:c72

		vmovups(mem(rcx), ymm0) // load c02:c72 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm11, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rcx)) // store c02:c72
		add(rdi, rcx) // c += cs_c;

		 // update c82:cf2

		vmovups(mem(rdx), ymm0) // load c82:f2 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm10, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rdx)) // store c82:cf2
		add(rdi, rdx) // c += cs_c;

		 // update c03:c73

		vmovups(mem(rcx), ymm0) // load c03:c73 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm9, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rcx)) // store c03:c73
		add(rdi, rcx) // c += cs_c;

		 // update c83:cf3

		vmovups(mem(rdx), ymm0) // load c83:f3 into ymm0
		vpermilps(imm(0xb1), ymm0, ymm2) // scale ymm0 by beta
		vmulps(ymm7, ymm0, ymm0)
		vmulps(ymm6, ymm2, ymm2)
		vaddsubps(ymm2, ymm0, ymm0)
		vaddps(ymm8, ymm0, ymm0) // add the gemm result to ymm0
		vmovups(ymm0, mem(rdx)) // store c83:cf3
		add(rdi, rdx) // c += cs_c;

		jmp(.CDONE) // jump to end.

	label(.CBETAZERO)

		vmovups(ymm15, mem(rcx)) // store c00:c70
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm14, mem(rdx)) // store c80:cf0
		add(rdi, rdx) // c += cs_c;

		vmovups(ymm13, mem(rcx)) // store c01:c71
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm12, mem(rdx)) // store c81:cf1
		add(rdi, rdx) // c += cs_c;

		vmovups(ymm11, mem(rcx)) // store c02:c72
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm10, mem(rdx)) // store c82:cf2
		add(rdi, rdx) // c += cs_c;

		vmovups(ymm9, mem(rcx)) // store c03:c73
		add(rdi, rcx) // c += cs_c;

		vmovups(ymm8, mem(rdx)) // store c83:cf3
		add(rdi, rdx) // c += cs_c;

	label(.CDONE)

	vzeroupper()


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



void bli_zgemm_sandybridge_asm_4x4
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
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( z, 4, 4, false );

	begin_asm()


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	//mov(var(b_next), r15) // load address of b_next.
	//mov(var(a_next), r14) // load address of a_next.

	vmovapd(mem(rax, 0*32), ymm0) // initialize loop by pre-loading
	vmovddup(mem(rbx, 0+0*32), ymm2)
	vmovddup(mem(rbx, 0+1*32), ymm3)

	mov(var(c), rcx) // load address of c
	mov(var(cs_c), rdi) // load cs_c
	lea(mem(, rdi, 8), rdi) // cs_c *= sizeof(dcomplex)
	lea(mem(, rdi, 2), rdi)
	lea(mem(rcx, rdi, 2), r10) // load address of c + 2*cs_c;

	prefetch(0, mem(rcx, 3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r10, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(r10, rdi, 1, 3*8)) // prefetch c + 3*cs_c

	vxorpd(ymm8, ymm8, ymm8)
	vxorpd(ymm9, ymm9, ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.ZCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.ZLOOPKITER) // MAIN LOOP


	 // iteration 0
	vmovapd(mem(rax, 1*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm15, ymm15)
	vaddpd(ymm7, ymm11, ymm11)

	prefetch(0, mem(rax, 16*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 8+0*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 8+1*32), ymm3)
	vaddpd(ymm6, ymm14, ymm14)
	vaddpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vpermilpd(imm(0x5), ymm0, ymm0)
	vaddpd(ymm6, ymm13, ymm13)
	vaddpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm12, ymm12)
	vaddpd(ymm7, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm1, ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vmulpd(ymm0, ymm3, ymm7)
	vaddsubpd(ymm6, ymm15, ymm15)
	vaddsubpd(ymm7, ymm11, ymm11)

	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 0+2*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 0+3*32), ymm3)
	vaddsubpd(ymm6, ymm14, ymm14)
	vaddsubpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 2*32), ymm0)
	vaddsubpd(ymm6, ymm13, ymm13)
	vaddsubpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddsubpd(ymm6, ymm12, ymm12)
	vaddsubpd(ymm7, ymm8, ymm8)


	 // iteration 1
	vmovapd(mem(rax, 3*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm15, ymm15)
	vaddpd(ymm7, ymm11, ymm11)

	prefetch(0, mem(rax, 18*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 8+2*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 8+3*32), ymm3)
	vaddpd(ymm6, ymm14, ymm14)
	vaddpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vpermilpd(imm(0x5), ymm0, ymm0)
	vaddpd(ymm6, ymm13, ymm13)
	vaddpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm12, ymm12)
	vaddpd(ymm7, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm1, ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vmulpd(ymm0, ymm3, ymm7)
	vaddsubpd(ymm6, ymm15, ymm15)
	vaddsubpd(ymm7, ymm11, ymm11)

	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 0+4*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 0+5*32), ymm3)
	vaddsubpd(ymm6, ymm14, ymm14)
	vaddsubpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 4*32), ymm0)
	vaddsubpd(ymm6, ymm13, ymm13)
	vaddsubpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddsubpd(ymm6, ymm12, ymm12)
	vaddsubpd(ymm7, ymm8, ymm8)


	 // iteration 2
	vmovapd(mem(rax, 5*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm15, ymm15)
	vaddpd(ymm7, ymm11, ymm11)

	prefetch(0, mem(rax, 20*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 8+4*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 8+5*32), ymm3)
	vaddpd(ymm6, ymm14, ymm14)
	vaddpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vpermilpd(imm(0x5), ymm0, ymm0)
	vaddpd(ymm6, ymm13, ymm13)
	vaddpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm12, ymm12)
	vaddpd(ymm7, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm1, ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vmulpd(ymm0, ymm3, ymm7)
	vaddsubpd(ymm6, ymm15, ymm15)
	vaddsubpd(ymm7, ymm11, ymm11)

	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 0+6*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 0+7*32), ymm3)
	vaddsubpd(ymm6, ymm14, ymm14)
	vaddsubpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 6*32), ymm0)
	vaddsubpd(ymm6, ymm13, ymm13)
	vaddsubpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddsubpd(ymm6, ymm12, ymm12)
	vaddsubpd(ymm7, ymm8, ymm8)


	 // iteration 3
	vmovapd(mem(rax, 7*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm15, ymm15)
	vaddpd(ymm7, ymm11, ymm11)

	prefetch(0, mem(rax, 22*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 8+6*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 8+7*32), ymm3)
	vaddpd(ymm6, ymm14, ymm14)
	vaddpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vpermilpd(imm(0x5), ymm0, ymm0)
	vaddpd(ymm6, ymm13, ymm13)
	vaddpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm12, ymm12)
	vaddpd(ymm7, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm1, ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vmulpd(ymm0, ymm3, ymm7)
	vaddsubpd(ymm6, ymm15, ymm15)
	vaddsubpd(ymm7, ymm11, ymm11)

	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 0+8*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 0+9*32), ymm3)
	vaddsubpd(ymm6, ymm14, ymm14)
	vaddsubpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 8*32), ymm0)
	vaddsubpd(ymm6, ymm13, ymm13)
	vaddsubpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddsubpd(ymm6, ymm12, ymm12)
	vaddsubpd(ymm7, ymm8, ymm8)


	add(imm(4*4*16), rbx) // b += 4*4 (unroll x nr)
	add(imm(4*4*16), rax) // a += 4*4 (unroll x mr)


	dec(rsi) // i -= 1;
	jne(.ZLOOPKITER) // iterate again if i != 0.






	label(.ZCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.ZPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.ZLOOPKLEFT) // EDGE LOOP

	 // iteration 0
	vmovapd(mem(rax, 1*32), ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm0, ymm3, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm15, ymm15)
	vaddpd(ymm7, ymm11, ymm11)

	prefetch(0, mem(rax, 16*32))
	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 8+0*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 8+1*32), ymm3)
	vaddpd(ymm6, ymm14, ymm14)
	vaddpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vpermilpd(imm(0x5), ymm0, ymm0)
	vaddpd(ymm6, ymm13, ymm13)
	vaddpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vperm2f128(imm(0x3), ymm2, ymm2, ymm4)
	vmulpd(ymm1, ymm5, ymm7)
	vperm2f128(imm(0x3), ymm3, ymm3, ymm5)
	vaddpd(ymm6, ymm12, ymm12)
	vaddpd(ymm7, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm1, ymm1)
	vmulpd(ymm0, ymm2, ymm6)
	vmulpd(ymm0, ymm3, ymm7)
	vaddsubpd(ymm6, ymm15, ymm15)
	vaddsubpd(ymm7, ymm11, ymm11)

	vmulpd(ymm1, ymm2, ymm6)
	vmovddup(mem(rbx, 0+2*32), ymm2)
	vmulpd(ymm1, ymm3, ymm7)
	vmovddup(mem(rbx, 0+3*32), ymm3)
	vaddsubpd(ymm6, ymm14, ymm14)
	vaddsubpd(ymm7, ymm10, ymm10)

	vmulpd(ymm0, ymm4, ymm6)
	vmulpd(ymm0, ymm5, ymm7)
	vmovapd(mem(rax, 2*32), ymm0)
	vaddsubpd(ymm6, ymm13, ymm13)
	vaddsubpd(ymm7, ymm9, ymm9)

	vmulpd(ymm1, ymm4, ymm6)
	vmulpd(ymm1, ymm5, ymm7)
	vaddsubpd(ymm6, ymm12, ymm12)
	vaddsubpd(ymm7, ymm8, ymm8)


	add(imm(4*1*16), rax) // a += 4 (1 x mr)
	add(imm(4*1*16), rbx) // b += 4 (1 x nr)


	dec(rsi) // i -= 1;
	jne(.ZLOOPKLEFT) // iterate again if i != 0.



	label(.ZPOSTACCUM)

	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab10    ab11    ab12    ab13
	 //   ab21    ab20    ab23    ab22
	 //   ab31 )  ab30 )  ab33 )  ab32 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab40  ( ab41  ( ab42  ( ab43
	 //   ab50    ab51    ab52    ab53
	 //   ab61    ab60    ab63    ab62
	 //   ab71 )  ab70 )  ab73 )  ab72 )


	vmovapd(ymm15, ymm7)
	vperm2f128(imm(0x12), ymm15, ymm13, ymm15)
	vperm2f128(imm(0x30), ymm7, ymm13, ymm13)

	vmovapd(ymm11, ymm7)
	vperm2f128(imm(0x12), ymm11, ymm9, ymm11)
	vperm2f128(imm(0x30), ymm7, ymm9, ymm9)

	vmovapd(ymm14, ymm7)
	vperm2f128(imm(0x12), ymm14, ymm12, ymm14)
	vperm2f128(imm(0x30), ymm7, ymm12, ymm12)

	vmovapd(ymm10, ymm7)
	vperm2f128(imm(0x12), ymm10, ymm8, ymm10)
	vperm2f128(imm(0x30), ymm7, ymm8, ymm8)


	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab01  ( ab02  ( ab03
	 //   ab10    ab11    ab12    ab13
	 //   ab20    ab21    ab22    ab23
	 //   ab30 )  ab31 )  ab32 )  ab33 )

	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab40  ( ab41  ( ab42  ( ab43
	 //   ab50    ab51    ab52    ab53
	 //   ab60    ab61    ab62    ab63
	 //   ab70 )  ab71 )  ab72 )  ab73 )


	 // scale by alpha

	mov(var(alpha), rax) // load address of alpha
	vbroadcastsd(mem(rax), ymm7) // load alpha_r and duplicate
	vbroadcastsd(mem(rax, 8), ymm6) // load alpha_i and duplicate

	vpermilpd(imm(0x5), ymm15, ymm3)
	vmulpd(ymm7, ymm15, ymm15)
	vmulpd(ymm6, ymm3, ymm3)
	vaddsubpd(ymm3, ymm15, ymm15)

	vpermilpd(imm(0x5), ymm14, ymm2)
	vmulpd(ymm7, ymm14, ymm14)
	vmulpd(ymm6, ymm2, ymm2)
	vaddsubpd(ymm2, ymm14, ymm14)

	vpermilpd(imm(0x5), ymm13, ymm1)
	vmulpd(ymm7, ymm13, ymm13)
	vmulpd(ymm6, ymm1, ymm1)
	vaddsubpd(ymm1, ymm13, ymm13)

	vpermilpd(imm(0x5), ymm12, ymm0)
	vmulpd(ymm7, ymm12, ymm12)
	vmulpd(ymm6, ymm0, ymm0)
	vaddsubpd(ymm0, ymm12, ymm12)

	vpermilpd(imm(0x5), ymm11, ymm3)
	vmulpd(ymm7, ymm11, ymm11)
	vmulpd(ymm6, ymm3, ymm3)
	vaddsubpd(ymm3, ymm11, ymm11)

	vpermilpd(imm(0x5), ymm10, ymm2)
	vmulpd(ymm7, ymm10, ymm10)
	vmulpd(ymm6, ymm2, ymm2)
	vaddsubpd(ymm2, ymm10, ymm10)

	vpermilpd(imm(0x5), ymm9, ymm1)
	vmulpd(ymm7, ymm9, ymm9)
	vmulpd(ymm6, ymm1, ymm1)
	vaddsubpd(ymm1, ymm9, ymm9)

	vpermilpd(imm(0x5), ymm8, ymm0)
	vmulpd(ymm7, ymm8, ymm8)
	vmulpd(ymm6, ymm0, ymm0)
	vaddsubpd(ymm0, ymm8, ymm8)




	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rbx), ymm7) // load beta_r and duplicate
	vbroadcastsd(mem(rbx, 8), ymm6) // load beta_i and duplicate







	mov(var(rs_c), rsi) // load rs_c
	lea(mem(, rsi, 8), rsi) // rsi = rs_c * sizeof(dcomplex)
	lea(mem(, rsi, 2), rsi)
	lea(mem(rcx, rsi, 2), rdx) // load address of c + 2*rs_c;


	 // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm7) // set ZF if beta_r == 0.
	sete(r8b) // r8b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm6) // set ZF if beta_i == 0.
	sete(r9b) // r9b = ( ZF == 1 ? 1 : 0 );
	and(r8b, r9b) // set ZF if r8b & r9b == 1.
	jne(.ZBETAZERO) // if ZF = 0, jump to beta == 0 case

		 // update c00:c30

		vmovupd(mem(rcx), ymm0) // load c00:c30 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm15, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rcx)) // store c00:c30
		add(rdi, rcx) // c += cs_c;

		 // update c40:c70

		vmovupd(mem(rdx), ymm0) // load c40:c70 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm14, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rdx)) // store c40:c70
		add(rdi, rdx) // c += cs_c;

		 // update c01:c31

		vmovupd(mem(rcx), ymm0) // load c01:c31 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm13, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rcx)) // store c01:c31
		add(rdi, rcx) // c += cs_c;

		 // update c41:c71

		vmovupd(mem(rdx), ymm0) // load c41:c71 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm12, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rdx)) // store c41:c71
		add(rdi, rdx) // c += cs_c;

		 // update c02:c32

		vmovupd(mem(rcx), ymm0) // load c02:c32 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm11, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rcx)) // store c02:c32
		add(rdi, rcx) // c += cs_c;

		 // update c42:c72

		vmovupd(mem(rdx), ymm0) // load c42:c72 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm10, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rdx)) // store c42:c72
		add(rdi, rdx) // c += cs_c;

		 // update c03:c33

		vmovupd(mem(rcx), ymm0) // load c03:c33 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm9, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rcx)) // store c03:c33
		add(rdi, rcx) // c += cs_c;

		 // update c43:c73

		vmovupd(mem(rdx), ymm0) // load c43:c73 into ymm0
		vpermilpd(imm(0x5), ymm0, ymm2) // scale ymm0 by beta
		vmulpd(ymm7, ymm0, ymm0)
		vmulpd(ymm6, ymm2, ymm2)
		vaddsubpd(ymm2, ymm0, ymm0)
		vaddpd(ymm8, ymm0, ymm0) // add the gemm result to ymm0
		vmovupd(ymm0, mem(rdx)) // store c43:c73

		jmp(.ZDONE) // jump to end.

	label(.ZBETAZERO)

		vmovupd(ymm15, mem(rcx)) // store c00:c30
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm14, mem(rdx)) // store c40:c70
		add(rdi, rdx) // c += cs_c;

		vmovupd(ymm13, mem(rcx)) // store c01:c31
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm12, mem(rdx)) // store c41:c71
		add(rdi, rdx) // c += cs_c;

		vmovupd(ymm11, mem(rcx)) // store c02:c32
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm10, mem(rdx)) // store c42:c72
		add(rdi, rdx) // c += cs_c;

		vmovupd(ymm9, mem(rcx)) // store c03:c33
		add(rdi, rcx) // c += cs_c;

		vmovupd(ymm8, mem(rdx)) // store c43:c73

	label(.ZDONE)

	vzeroupper()


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
	  [cs_c]   "m" (cs_c)/*,   // 8
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)*/  // 10
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


