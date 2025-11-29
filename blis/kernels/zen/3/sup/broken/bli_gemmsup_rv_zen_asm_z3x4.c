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

// assumes beta.r, beta.i have been broadcast into ymm1, ymm2.
// outputs to ymm0
#define ZGEMM_INPUT_SCALE_CS_BETA_NZ \
	vmovupd(mem(rcx), xmm0) \
	vmovupd(mem(rcx, rsi, 1), xmm3) \
	vinsertf128(imm(1), xmm3, ymm0, ymm0) \
	vpermilpd(imm(0x5), ymm0, ymm3) \
	vmulpd(ymm1, ymm0, ymm0) \
	vmulpd(ymm2, ymm3, ymm3) \
	vaddsubpd(ymm3, ymm0, ymm0)

#define ZGEMM_INPUT_SCALE_RS_BETA_NZ \
	vmovupd(mem(rcx), ymm0) \
	vpermilpd(imm(0x5), ymm0, ymm3) \
	vmulpd(ymm1, ymm0, ymm0) \
	vmulpd(ymm2, ymm3, ymm3) \
	vaddsubpd(ymm3, ymm0, ymm0)

#define ZGEMM_OUTPUT_RS \
	vmovupd(ymm0, mem(rcx)) \

#define ZGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT \
	vmovupd(mem(rcx, rsi, 8), ymm0) \
	vpermilpd(imm(0x5), ymm0, ymm3) \
	vmulpd(ymm1, ymm0, ymm0) \
	vmulpd(ymm2, ymm3, ymm3) \
	vaddsubpd(ymm3, ymm0, ymm0)

#define ZGEMM_OUTPUT_RS_NEXT \
	vmovupd(ymm0, mem(rcx, rsi, 8)) \


void bli_zgemmsup_rv_zen_asm_2x4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*            data,
       cntx_t*               cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r8, 2), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)
	lea(mem(, r9, 2), r9)              // cs_a *= sizeof(dt)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)
	lea(mem(, r10, 2), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)
	lea(mem(, rdi, 2), rdi)            // rs_c *= sizeof(dt)

	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.SLOOP3X8I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                    // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(dt)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;

	label(.SPOSTPFETCH)                // done prefetching c

	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	lea(mem(rax, r8,  4), rdx)         // use rdx for pre-fetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	label(.SLOOPKITER)                 // MAIN LOOP

	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)

	vbroadcastsd(mem(rax , 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.

	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.SLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	 // of ymm6/7, ymm10/11, ymm/14/15
	vpermilpd(imm(0x5), ymm6, ymm6)
	vpermilpd(imm(0x5), ymm7, ymm7)
	vpermilpd(imm(0x5), ymm10, ymm10)
	vpermilpd(imm(0x5), ymm11, ymm11)

	 // subtract/add even/odd elements
	vaddsubpd(ymm6, ymm4, ymm4)
	vaddsubpd(ymm7, ymm5, ymm5)

	vaddsubpd(ymm10, ymm8, ymm8)
	vaddsubpd(ymm11, ymm9, ymm9)

	/* (ar + ai) x AB */
	mov(var(alpha), rax) // load address of alpha
	vbroadcastsd(mem(rax), ymm0) // load alpha_r and duplicate
	vbroadcastsd(mem(rax, 8), ymm1) // load alpha_i and duplicate

	vpermilpd(imm(0x5), ymm4, ymm3)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm4, ymm4)

	vpermilpd(imm(0x5), ymm5, ymm3)
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm5, ymm5)

	vpermilpd(imm(0x5), ymm8, ymm3)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm9, ymm3)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm9, ymm9)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rbx), ymm1) // load beta_r and duplicate
	vbroadcastsd(mem(rbx, 8), ymm2) // load beta_i and duplicate

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 4), rsi)    // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r13b) // r13b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r15b) // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b) // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) ==16.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	ZGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm0)
	ZGEMM_OUTPUT_RS

	ZGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT
	vaddpd(ymm5, ymm0, ymm0)
	ZGEMM_OUTPUT_RS_NEXT
	add(rdi, rcx) // rcx = c + 1*rs_c

	ZGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddpd(ymm8, ymm0, ymm0)
	ZGEMM_OUTPUT_RS

	ZGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT
	vaddpd(ymm9, ymm0, ymm0)
	ZGEMM_OUTPUT_RS_NEXT

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)
	/*|--------|           |-------|
	  |        |           |       |
	  |    2x4 |           |  4x2  |
	  |--------|           |-------|
	*/

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(real dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof((real+imag) dt)

	lea(mem(rsi, rsi, 2), r13)           // r13 = 3*rs_a

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm4)

	add(rdi, rcx)
	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm8, ymm0, ymm8)
	add(rdi, rcx)

	lea(mem(r12, rsi, 2), rcx)

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm5, ymm0, ymm5)
	add(rdi, rcx)

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm9, ymm0, ymm9)
	add(rdi, rcx)

	mov(r12, rcx)                      // reset rcx to current utile of c.


	/****3x4 tile going to save into 4x2 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	/******************Transpose top tile 4x3***************************/
	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	add(rsi, rcx)

	vmovups(xmm5, mem(rcx))
	vmovups(xmm9, mem(rcx, 16))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm5, xmm5)
	vextractf128(imm(0x1), ymm9, xmm9)
	vmovups(xmm5, mem(rcx))
	vmovups(xmm9, mem(rcx, 16))

	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 8))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx))
	vmovupd(ymm9, mem(rcx, rsi, 8))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)
	/****2x4 tile going to save into 4x2 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	/******************Transpose tile 2x4***************************/
	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	add(rsi, rcx)

	vmovups(xmm5, mem(rcx))
	vmovups(xmm9, mem(rcx, 16))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm5, xmm5)
	vextractf128(imm(0x1), ymm9, xmm9)
	vmovups(xmm5, mem(rcx))
	vmovups(xmm9, mem(rcx, 16))

	label(.SDONE)

	end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
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


void bli_zgemmsup_rv_zen_asm_1x4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*            data,
       cntx_t*               cntx
     )
{

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r8, 2), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)
	lea(mem(, r9, 2), r9)              // cs_a *= sizeof(dt)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)
	lea(mem(, r10, 2), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)
	lea(mem(, rdi, 2), rdi)            // rs_c *= sizeof(dt)

	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.SLOOP3X8I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                    // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(dt)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;

	label(.SPOSTPFETCH)                // done prefetching c

	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	lea(mem(rax, r8,  4), rdx)         // use rdx for pre-fetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	label(.SLOOPKITER)                 // MAIN LOOP

	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	vbroadcastsd(mem(rax , 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)



	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.

	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.SLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	 // of ymm6/7, ymm10/11, ymm/14/15
	vpermilpd(imm(0x5), ymm6, ymm6)
	vpermilpd(imm(0x5), ymm7, ymm7)

	 // subtract/add even/odd elements
	vaddsubpd(ymm6, ymm4, ymm4)
	vaddsubpd(ymm7, ymm5, ymm5)

	/* (ar + ai) x AB */
	mov(var(alpha), rax) // load address of alpha
	vbroadcastsd(mem(rax), ymm0) // load alpha_r and duplicate
	vbroadcastsd(mem(rax, 8), ymm1) // load alpha_i and duplicate

	vpermilpd(imm(0x5), ymm4, ymm3)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm4, ymm4)

	vpermilpd(imm(0x5), ymm5, ymm3)
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm5, ymm5)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rbx), ymm1) // load beta_r and duplicate
	vbroadcastsd(mem(rbx, 8), ymm2) // load beta_i and duplicate

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 4), rsi)    // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r13b) // r13b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r15b) // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b) // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) ==16.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	ZGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm0)
	ZGEMM_OUTPUT_RS

	ZGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT
	vaddpd(ymm5, ymm0, ymm0)
	ZGEMM_OUTPUT_RS_NEXT

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)
	/*|--------|           |-------|
	  |        |           |       |
	  |    1x4 |           |  4x1  |
	  |--------|           |-------|
	*/

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(real dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof((real+imag) dt)

	lea(mem(rsi, rsi, 2), r13)           // r13 = 3*rs_a

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm4)

	lea(mem(r12, rsi, 2), rcx)

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm5, ymm0, ymm5)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	/****1x4 tile going to save into 4x1 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	vmovups(xmm4, mem(rcx))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vmovups(xmm4, mem(rcx))

	add(rsi, rcx)

	vmovups(xmm5, mem(rcx))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm5, xmm5)
	vmovups(xmm5, mem(rcx))

	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 8))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)
	/****1x4 tile going to save into 4x1 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	vmovups(xmm4, mem(rcx))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vmovups(xmm4, mem(rcx))

	add(rsi, rcx)

	vmovups(xmm5, mem(rcx))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm5, xmm5)
	vmovups(xmm5, mem(rcx))

	label(.SDONE)

	end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
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

void bli_zgemmsup_rv_zen_asm_2x2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*            data,
       cntx_t*               cntx

     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r8, 2), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)
	lea(mem(, r9, 2), r9)              // cs_a *= sizeof(dt)

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)
	lea(mem(, r10, 2), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)
	lea(mem(, rdi, 2), rdi)            // rs_c *= sizeof(dt)

	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.SLOOP3X8I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                    // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;

	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(dt)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;

	label(.SPOSTPFETCH)                // done prefetching c

	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	lea(mem(rax, r8,  4), rdx)         // use rdx for pre-fetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	label(.SLOOPKITER)                 // MAIN LOOP

	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)

	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)

	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.

	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.SLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm8)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8, 1, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	 // of ymm6/7, ymm10/11, ymm/14/15
	vpermilpd(imm(0x5), ymm6, ymm6)
	vpermilpd(imm(0x5), ymm10, ymm10)

	// subtract/add even/odd elements
	vaddsubpd(ymm6, ymm4, ymm4)

	vaddsubpd(ymm10, ymm8, ymm8)

	/* (ar + ai) x AB */
	mov(var(alpha), rax) // load address of alpha
	vbroadcastsd(mem(rax), ymm0) // load alpha_r and duplicate
	vbroadcastsd(mem(rax, 8), ymm1) // load alpha_i and duplicate

	vpermilpd(imm(0x5), ymm4, ymm3)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm4, ymm4)

	vpermilpd(imm(0x5), ymm8, ymm3)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm8, ymm8)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rbx), ymm1) // load beta_r and duplicate
	vbroadcastsd(mem(rbx, 8), ymm2) // load beta_i and duplicate

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 4), rsi)    // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	 // now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r13b) // r13b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r15b) // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b) // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	ZGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm0)
	ZGEMM_OUTPUT_RS

	add(rdi, rcx) // rcx = c + 1*rs_c

	ZGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddpd(ymm8, ymm0, ymm0)
	ZGEMM_OUTPUT_RS

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)
	/*|--------|           |-------|
	  |        |           |       |
	  |    2x2 |           |  2x2  |
	  |--------|           |-------|
	*/

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(real dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof((real+imag) dt)

	lea(mem(rsi, rsi, 2), r13)           // r13 = 3*rs_a

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm4)

	add(rdi, rcx)
	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm8, ymm0, ymm8)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	/****2x2 tile going to save into 2x2 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)

	cmp(imm(16), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)

	/****2x2 tile going to save into 2x2 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovups(xmm4, mem(rcx))
	vmovups(xmm8, mem(rcx, 16))

	label(.SDONE)

	end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
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

void bli_zgemmsup_rv_zen_asm_1x2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*            data,
       cntx_t*               cntx

     )
{

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 3;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r8, 2), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)
	lea(mem(, r9, 2), r9)              // cs_a *= sizeof(dt)

//	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)
	lea(mem(, r10, 2), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)
	lea(mem(, rdi, 2), rdi)            // rs_c *= sizeof(dt)

	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.SLOOP3X8I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                    // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;

	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(dt)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;

	label(.SPOSTPFETCH)                // done prefetching c

	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	lea(mem(rax, r8,  4), rdx)         // use rdx for pre-fetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	label(.SLOOPKITER)                 // MAIN LOOP

	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)


	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.

	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.SLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm4)

	vbroadcastsd(mem(rax, 8), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	 // of ymm6/7, ymm10/11, ymm/14/15
	vpermilpd(imm(0x5), ymm6, ymm6)

	// subtract/add even/odd elements
	vaddsubpd(ymm6, ymm4, ymm4)

	/* (ar + ai) x AB */
	mov(var(alpha), rax) // load address of alpha
	vbroadcastsd(mem(rax), ymm0) // load alpha_r and duplicate
	vbroadcastsd(mem(rax, 8), ymm1) // load alpha_i and duplicate

	vpermilpd(imm(0x5), ymm4, ymm3)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm4, ymm4)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rbx), ymm1) // load beta_r and duplicate
	vbroadcastsd(mem(rbx, 8), ymm2) // load beta_i and duplicate

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 4), rsi)    // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	 // now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r13b) // r13b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r15b) // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b) // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(16), rdi)                   // set ZF if (16*rs_c) == 16.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	ZGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm0)
	ZGEMM_OUTPUT_RS

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)
	/*|--------|           |-------|
	  |        |           |       |
	  |    1x2 |           |  2x1  |
	  |--------|           |-------|
	*/

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(real dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof((real+imag) dt)

	lea(mem(rsi, rsi, 2), r13)           // r13 = 3*rs_a

	ZGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddpd(ymm4, ymm0, ymm4)


	/****3x4 tile going to save into 4x3 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	/******************Transpose tile 1x2***************************/
	vmovups(xmm4, mem(rcx))
	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vmovups(xmm4, mem(rcx))


	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)

	cmp(imm(16), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)

	/****1x2 tile going to save into 2x1 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(, rsi, 2), rsi)    // rsi = cs_c * sizeof(dt)

	/******************Transpose top tile 4x3***************************/
	vmovups(xmm4, mem(rcx))


	add(rsi, rcx)

	vextractf128(imm(0x1), ymm4, xmm4)
	vmovups(xmm4, mem(rcx))

	label(.SDONE)

	end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
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


