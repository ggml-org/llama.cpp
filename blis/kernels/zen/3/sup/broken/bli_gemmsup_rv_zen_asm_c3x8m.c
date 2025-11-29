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
#define CGEMM_INPUT_SCALE_CS_BETA_NZ \
	vmovlpd(mem(rcx), xmm0, xmm0) \
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0) \
	vmovlpd(mem(rcx, rsi, 2), xmm3, xmm3) \
	vmovhpd(mem(rcx, r13, 1), xmm3, xmm3) \
	vinsertf128(imm(1), xmm3, ymm0, ymm0) \
	vpermilps(imm(0xb1), ymm0, ymm3) \
	vmulps(ymm1, ymm0, ymm0) \
	vmulps(ymm2, ymm3, ymm3) \
	vaddsubps(ymm3, ymm0, ymm0)

#define CGEMM_INPUT_SCALE_CS_BETA_NZ_128 \
	vmovlpd(mem(rcx), xmm0, xmm0) \
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0) \
	vpermilps(imm(0xb1), xmm0, xmm3) \
	vmulps(xmm1, xmm0, xmm0) \
	vmulps(xmm2, xmm3, xmm3) \
	vaddsubps(xmm3, xmm0, xmm0)

#define CGEMM_INPUT_SCALE_RS_BETA_NZ \
	vmovups(mem(rcx), ymm0) \
	vpermilps(imm(0xb1), ymm0, ymm3) \
	vmulps(ymm1, ymm0, ymm0) \
	vmulps(ymm2, ymm3, ymm3) \
	vaddsubps(ymm3, ymm0, ymm0)

#define CGEMM_OUTPUT_RS \
	vmovups(ymm0, mem(rcx)) \

#define CGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT \
	vmovups(mem(rcx, rsi, 8), ymm0) \
	vpermilps(imm(0xb1), ymm0, ymm3) \
	vmulps(ymm1, ymm0, ymm0) \
	vmulps(ymm2, ymm3, ymm3) \
	vaddsubps(ymm3, ymm0, ymm0)

#define CGEMM_OUTPUT_RS_NEXT \
	vmovups(ymm0, mem(rcx, rsi, 8)) \

/*
   rrr:
	 --------        ------        --------
	 --------   +=   ------ ...    --------
	 --------        ------        --------
	 --------        ------            :

   rcr:
	 --------        | | | |       --------
	 --------   +=   | | | | ...   --------
	 --------        | | | |       --------
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
	 | | | | | | | |  +=   ------ ...    --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------            :
*/
void bli_cgemmsup_rv_zen_asm_3x8m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*            data,
       cntx_t*               cntx
     )
{
	uint64_t n_left = n0 % 8;

	// First check whether this is a edge case in the n dimension. If so,
	// dispatch other 3x?m kernels, as needed.
	if (n_left )
	{
		scomplex*  cij = c;
		scomplex*  bj  = b;
		scomplex*  ai  = a;

		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_cgemmsup_rv_zen_asm_3x4m
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

			bli_cgemmsup_rv_zen_asm_3x2m
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			bli_cgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
		}

		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)

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

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;

	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(dt)
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

	vmovups(mem(rbx,  0*32), ymm0)
	vmovups(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)


	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovups(mem(rbx,  0*32), ymm0)
	vmovups(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovups(mem(rbx,  0*32), ymm0)
	vmovups(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(r9, rax)                       // a += cs_a;

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
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)

	vbroadcastss(mem(rax, 4      ), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	 // of ymm6/7, ymm10/11, ymm/14/15
	vpermilps(imm(0xb1), ymm6, ymm6)
	vpermilps(imm(0xb1), ymm7, ymm7)
	vpermilps(imm(0xb1), ymm10, ymm10)
	vpermilps(imm(0xb1), ymm11, ymm11)
	vpermilps(imm(0xb1), ymm14, ymm14)
	vpermilps(imm(0xb1), ymm15, ymm15)

	 // subtract/add even/odd elements
	vaddsubps(ymm6, ymm4, ymm4)
	vaddsubps(ymm7, ymm5, ymm5)

	vaddsubps(ymm10, ymm8, ymm8)
	vaddsubps(ymm11, ymm9, ymm9)

	vaddsubps(ymm14, ymm12, ymm12)
	vaddsubps(ymm15, ymm13, ymm13)

	/* (ar + ai) x AB */
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastss(mem(rax), ymm0)       // load alpha_r and duplicate
	vbroadcastss(mem(rax, 4), ymm1)    // load alpha_i and duplicate

	vpermilps(imm(0xb1), ymm4, ymm3)
	vmulps(ymm0, ymm4, ymm4)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm4, ymm4)

	vpermilps(imm(0xb1), ymm5, ymm3)
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm5, ymm5)

	vpermilps(imm(0xb1), ymm8, ymm3)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm8, ymm8)

	vpermilps(imm(0xb1), ymm9, ymm3)
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm9, ymm9)

	vpermilps(imm(0xb1), ymm12, ymm3)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm12, ymm12)

	vpermilps(imm(0xb1), ymm13, ymm3)
	vmulps(ymm0, ymm13, ymm13)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm13, ymm13)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rbx), ymm1)       // load beta_r and duplicate
	vbroadcastss(mem(rbx, 4), ymm2)    // load beta_i and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm1)               // set ZF if beta_r == 0.
	sete(r13b)                         // r13b = ( ZF == 1 ? 1 : 0 );
	vucomiss(xmm0, xmm2)               // set ZF if beta_i == 0.
	sete(r15b)                         // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b)                    // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO)                    // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	CGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddps(ymm4, ymm0, ymm0)
	CGEMM_OUTPUT_RS

	CGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT
	vaddps(ymm5, ymm0, ymm0)
	CGEMM_OUTPUT_RS_NEXT
	add(rdi, rcx) // rcx = c + 1*rs_c

	CGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddps(ymm8, ymm0, ymm0)
	CGEMM_OUTPUT_RS

	CGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT
	vaddps(ymm9, ymm0, ymm0)
	CGEMM_OUTPUT_RS_NEXT
	add(rdi, rcx) // rcx = c + 2*rs_c

	CGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddps(ymm12, ymm0, ymm0)
	CGEMM_OUTPUT_RS

	CGEMM_INPUT_SCALE_RS_BETA_NZ_NEXT
	vaddps(ymm13, ymm0, ymm0)
	CGEMM_OUTPUT_RS_NEXT

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)
  /*|----------------|          |-------|
	|        |       |          |       |
	|    3x4 |   3x4 |          |  4x3  |
	|        |       |          |-------|
	|----------------|          |       |
	                            |  4x3  |
	                            |-------|
   */

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(rsi, rsi, 2), r13) // r13 = 3*rs_a

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm4, ymm0, ymm4)

	add(rdi, rcx)
	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm8, ymm0, ymm8)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm12, ymm0, ymm12)

	lea(mem(r12, rsi, 4), rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm5, ymm0, ymm5)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm9, ymm0, ymm9)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm13, ymm0, ymm13)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	vunpcklpd(ymm8, ymm4, ymm0)        //a0a1b0b1 a4a4b4b5 //gamma00-10 gamma02-12
	vunpckhpd(ymm8, ymm4, ymm2)        //a2a3b2b3 a6a7b6b7 //gamma01-11 gamma03-13

	/******************Transpose top tile 4x3***************************/
	vmovups(xmm0, mem(rcx))				// store (gamma00-10)
	vmovlpd(xmm12, mem(rcx, 16))	// store (gamma20)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma01-11)
	vmovhpd(xmm12, mem(rcx, 16))	// store (gamma21)
	lea(mem(rcx, rsi, 1), rcx)

	vextractf128(imm(0x1), ymm0, xmm0)
	vextractf128(imm(0x1), ymm2, xmm2)
	vextractf128(imm(0x1), ymm12, xmm12)
	vmovups(xmm0, mem(rcx))				// store (gamma02-12)
	vmovlpd(xmm12, mem(rcx, 16))	// store (gamma22)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma03-13)
	vmovhpd(xmm12, mem(rcx, 16))	// store (gamma33)
	lea(mem(rcx, rsi, 1), rcx)

	/******************Transpose bottom tile 4x3***************************/
	vunpcklpd(ymm9, ymm5, ymm0)        //a8a9b8b9     a12a13b12b13 //gamma04-14 gamma06-16
	vunpckhpd(ymm9, ymm5, ymm2)        //a10a11b10b11 a14a15b14b15 //gamma05-15 gamma07-17

	vmovups(xmm0, mem(rcx))				// store (gamma04-14)
	vmovlpd(xmm13, mem(rcx, 16))	// store (gamma24)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma05-15)
	vmovhpd(xmm13, mem(rcx, 16))	// store (gamma25)
	lea(mem(rcx, rsi, 1), rcx)

	vextractf128(imm(0x1), ymm0, xmm0)
	vextractf128(imm(0x1), ymm2, xmm2)
	vextractf128(imm(0x1), ymm13, xmm13)
	vmovups(xmm0, mem(rcx))             // store (gamma06-16)
	vmovlpd(xmm13, mem(rcx, 16))    // store (gamma26)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))            // store (gamma07-17)
	vmovhpd(xmm13, mem(rcx, 16))    // store (gamma27)

	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovups(ymm4, mem(rcx))
	vmovups(ymm5, mem(rcx, rsi, 8))
	add(rdi, rcx)

	vmovups(ymm8, mem(rcx))
	vmovups(ymm9, mem(rcx, rsi, 8))
	add(rdi, rcx)

	vmovups(ymm12, mem(rcx))
	vmovups(ymm13, mem(rcx, rsi, 8))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)

	/****3x8 tile going to save into 8x3 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)

	vunpcklpd(ymm8, ymm4, ymm0) //a0a1b0b1 a4a4b4b5
	vunpckhpd(ymm8, ymm4, ymm2) //a2a3b2b3 a6a7b6b7

	/******************Transpose top tile 4x3***************************/
	vmovups(xmm0, mem(rcx))
	vmovlpd(xmm12, mem(rcx,16))
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))
	vmovhpd(xmm12,mem(rcx,16))
	lea(mem(rcx, rsi, 1), rcx)

	vextractf128(imm(0x1), ymm0, xmm0)
	vextractf128(imm(0x1), ymm2, xmm2)
	vextractf128(imm(0x1), ymm12, xmm12)
	vmovups(xmm0, mem(rcx))
	vmovlpd(xmm12, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))
	vmovhpd(xmm12, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)

	/******************Transpose bottom tile 4x3***************************/
	vunpcklpd(ymm9, ymm5, ymm0)  //a8a9b8b9     a12a13b12b13
	vunpckhpd(ymm9, ymm5, ymm2)  //a10a11b10b11 a14a15b14b15

	vmovups(xmm0, mem(rcx))
	vmovlpd(xmm13, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))
	vmovhpd(xmm13, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)

	vextractf128(imm(0x1), ymm0, xmm0)
	vextractf128(imm(0x1), ymm2, xmm2)
	vextractf128(imm(0x1), ymm13, xmm13)
	vmovups(xmm0, mem(rcx))
	vmovlpd(xmm13, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)

	vmovups(xmm2, mem(rcx))
	vmovhpd(xmm13, mem(rcx, 16))

	label(.SDONE)

	lea(mem(r12, rdi, 2), r12)
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)
	lea(mem(r14, r8,  1), r14)         //a_ii = r14 += 3*rs_a

	dec(r11)                           // ii -= 1;
	jne(.SLOOP3X8I)                    // iterate again if ii != 0.

	label(.SRETURN)

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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 8;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		scomplex*  cij = c + i_edge*rs_c;
		scomplex*  ai  = a + i_edge*rs_a;
		scomplex*  bj  = b;

		cgemmsup_ker_ft ker_fps[3] =
		{
		  NULL,
		  bli_cgemmsup_rv_zen_asm_1x8,
		  bli_cgemmsup_rv_zen_asm_2x8,
		};

		cgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;

	}

}

void bli_cgemmsup_rv_zen_asm_3x4m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
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

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)

	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.SLOOP3X4I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

	vzeroall()                         // zero all xmm/ymm registers.

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                    // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;

	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(dt)
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

	vmovups(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovups(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)


	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovups(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)


	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovups(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm4)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)


	add(r9, rax)                       // a += cs_a;

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
	vfmadd231ps(ymm0, ymm2, ymm4)

	vbroadcastss(mem(rax, r8, 1), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm8)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vfmadd231ps(ymm0, ymm2, ymm12)

	vbroadcastss(mem(rax, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm6)

	vbroadcastss(mem(rax, r8, 1, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm10)

	vbroadcastss(mem(rax, r8, 2, 4), ymm3)
	vfmadd231ps(ymm0, ymm3, ymm14)


	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	 // of ymm6/7, ymm10/11, ymm/14/15
	vpermilps(imm(0xb1), ymm6, ymm6)
	vpermilps(imm(0xb1), ymm10, ymm10)
	vpermilps(imm(0xb1), ymm14, ymm14)

	// subtract/add even/odd elements
	vaddsubps(ymm6, ymm4, ymm4)

	vaddsubps(ymm10, ymm8, ymm8)

	vaddsubps(ymm14, ymm12, ymm12)

	/* (ar + ai) x AB */
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastss(mem(rax), ymm0)       // load alpha_r and duplicate
	vbroadcastss(mem(rax, 4), ymm1)    // load alpha_i and duplicate

	vpermilps(imm(0xb1), ymm4, ymm3)
	vmulps(ymm0, ymm4, ymm4)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm4, ymm4)

	vpermilps(imm(0xb1), ymm8, ymm3)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm8, ymm8)

	vpermilps(imm(0xb1), ymm12, ymm3)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm12, ymm12)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rbx), ymm1)       // load beta_r and duplicate
	vbroadcastss(mem(rbx, 4), ymm2)    // load beta_i and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	 // now avoid loading C if beta == 0
	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm1)               // set ZF if beta_r == 0.
 	sete(r13b)                         // r13b = ( ZF == 1 ? 1 : 0 );
	vucomiss(xmm0, xmm2)               // set ZF if beta_i == 0.
	sete(r15b)                         // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b)                    // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO)                    // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	CGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddps(ymm4, ymm0, ymm0)
	CGEMM_OUTPUT_RS

	add(rdi, rcx)                      // rcx = c + 1*rs_c

	CGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddps(ymm8, ymm0, ymm0)
	CGEMM_OUTPUT_RS

	add(rdi, rcx)                      // rcx = c + 2*rs_c

	CGEMM_INPUT_SCALE_RS_BETA_NZ
	vaddps(ymm12, ymm0, ymm0)
	CGEMM_OUTPUT_RS

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)
  /*|--------|          |-------|
	|        |          |       |
	|   3x4  |          |  4x3  |
	|--------|          |-------|
   */
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(rsi, rsi, 2), r13) // r13 = 3*rs_a

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm4, ymm0, ymm4)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm8, ymm0, ymm8)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ
	vaddps(ymm12, ymm0, ymm12)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	vunpcklpd(ymm8, ymm4, ymm0)        //a0a1b0b1 a4a4b4b5 //gamma00-10 gamma02-12
	vunpckhpd(ymm8, ymm4, ymm2)        //a2a3b2b3 a6a7b6b7 //gamma01-11 gamma03-13

	/******************Transpose tile 4x3***************************/
	vmovups(xmm0, mem(rcx))				// store (gamma00-10)
	vmovlpd(xmm12, mem(rcx, 16))	// store (gamma20)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma01-11)
	vmovhpd(xmm12, mem(rcx, 16))	// store (gamma21)
	lea(mem(rcx, rsi, 1), rcx)

	vextractf128(imm(0x1), ymm0, xmm0)
	vextractf128(imm(0x1), ymm2, xmm2)
	vextractf128(imm(0x1), ymm12, xmm12)
	vmovups(xmm0, mem(rcx))				// store (gamma02-12)
	vmovlpd(xmm12, mem(rcx, 16))	// store (gamma22)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma03-13)
	vmovhpd(xmm12, mem(rcx, 16))	// store (gamma33)
	lea(mem(rcx, rsi, 1), rcx)

	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovups(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovups(ymm8, mem(rcx))
	add(rdi, rcx)

	vmovups(ymm12, mem(rcx))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)

	/****3x4 tile going to save into 4x3 tile in C*****/
	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)

	vunpcklpd(ymm8, ymm4, ymm0) //a0a1b0b1 a4a4b4b5
	vunpckhpd(ymm8, ymm4, ymm2) //a2a3b2b3 a6a7b6b7

	vmovups(xmm0, mem(rcx))
	vmovlpd(xmm12, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))
	vmovhpd(xmm12, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)

	vextractf128(imm(0x1), ymm0, xmm0)
	vextractf128(imm(0x1), ymm2, xmm2)
	vextractf128(imm(0x1), ymm12, xmm12)
	vmovups(xmm0, mem(rcx))
	vmovlpd(xmm12, mem(rcx, 16))
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))
	vmovhpd(xmm12, mem(rcx, 16))

	label(.SDONE)

	lea(mem(r12, rdi, 2), r12)
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)
	lea(mem(r14, r8,  1), r14)         //a_ii = r14 += 3*rs_a

	dec(r11)                           // ii -= 1;
	jne(.SLOOP3X4I)                    // iterate again if ii != 0.

	label(.SRETURN)

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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 4;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		scomplex*  cij = c + i_edge*rs_c;
		scomplex*  ai  = a + i_edge*rs_a;
		scomplex*  bj  = b;

		cgemmsup_ker_ft ker_fps[3] =
		{
		  NULL,
		  bli_cgemmsup_rv_zen_asm_1x4,
		  bli_cgemmsup_rv_zen_asm_2x4,
		};

		cgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
}

void bli_cgemmsup_rv_zen_asm_3x2m
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       scomplex*    restrict alpha,
       scomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       scomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       scomplex*    restrict beta,
       scomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
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

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(dt)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(dt)

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(dt)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(dt)

	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.SLOOP3X2I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

    vzeroall()                         // zero all xmm/ymm registers.

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                    // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;

	jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
	label(.SCOLPFETCH)                 // column-stored pre-fetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(dt)
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

	vmovups(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm4)

	vbroadcastss(mem(rax, r8, 1), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm8)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm12)

	vbroadcastss(mem(rax, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8, 1, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8, 2, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm14)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 1

	vmovups(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm4)

	vbroadcastss(mem(rax, r8, 1), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm8)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm12)

	vbroadcastss(mem(rax, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8, 1, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8, 2, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm14)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 2

	vmovups(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm4)

	vbroadcastss(mem(rax, r8, 1), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm8)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm12)

	vbroadcastss(mem(rax, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8, 1, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8, 2, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm14)

	add(r9, rax)                       // a += cs_a;

	// ---------------------------------- iteration 3
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm4)

	vbroadcastss(mem(rax, r8, 1), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm8)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm12)

	vbroadcastss(mem(rax, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8, 1, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8, 2, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm14)

	add(r9, rax)                       // a += cs_a;

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
	vfmadd231ps(xmm0, xmm2, xmm4)

	vbroadcastss(mem(rax, r8, 1), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm8)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vfmadd231ps(xmm0, xmm2, xmm12)

	vbroadcastss(mem(rax, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8, 1, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8, 2, 4), xmm3)
	vfmadd231ps(xmm0, xmm3, xmm14)

	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.

	label(.SPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.

	// permute even and odd elements
	// of xmm6/7, xmm10/11, xmm/14/15
	vpermilps(imm(0xb1), xmm6, xmm6)
	vpermilps(imm(0xb1), xmm10, xmm10)
	vpermilps(imm(0xb1), xmm14, xmm14)

	// subtract/add even/odd elements
	vaddsubps(xmm6, xmm4, xmm4)
	vaddsubps(xmm10, xmm8, xmm8)
	vaddsubps(xmm14, xmm12, xmm12)

	/* (ar + ai) x AB */
	mov(var(alpha), rax)               // load address of alpha
	vbroadcastss(mem(rax), xmm0)       // load alpha_r and duplicate
	vbroadcastss(mem(rax, 4), xmm1)    // load alpha_i and duplicate

	vpermilps(imm(0xb1), xmm4, xmm3)
	vmulps(xmm0, xmm4, xmm4)
	vmulps(xmm1, xmm3, xmm3)
	vaddsubps(xmm3, xmm4, xmm4)

	vpermilps(imm(0xb1), xmm8, xmm3)
	vmulps(xmm0, xmm8, xmm8)
	vmulps(xmm1, xmm3, xmm3)
	vaddsubps(xmm3, xmm8, xmm8)

	vpermilps(imm(0xb1), xmm12, xmm3)
	vmulps(xmm0, xmm12, xmm12)
	vmulps(xmm1, xmm3, xmm3)
	vaddsubps(xmm3, xmm12, xmm12)

	/* (�r + �i)x C + ((ar + ai) x AB) */
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rbx), xmm1)       // load beta_r and duplicate
	vbroadcastss(mem(rbx, 4), xmm2)    // load beta_i and duplicate

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(dt)

	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	 // now avoid loading C if beta == 0
	vxorps(xmm0, xmm0, xmm0) // set xmm0 to zero.
	vucomiss(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r13b) // r13b = ( ZF == 1 ? 1 : 0 );
	vucomiss(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r15b) // r15b = ( ZF == 1 ? 1 : 0 );
	and(r13b, r15b) // set ZF if r13b & r15b == 1.
	jne(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORED)                    // jump to column storage case

	label(.SROWSTORED)

	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vpermilps(imm(0xb1), xmm0, xmm3)
	vmulps(xmm1, xmm0, xmm0)
	vmulps(xmm2, xmm3, xmm3)
	vaddsubps(xmm3, xmm0, xmm0)

	vaddps(xmm4, xmm0, xmm0)

	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))

	add(rdi, rcx) // rcx = c + 1*rs_c

	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vpermilps(imm(0xb1), xmm0, xmm3)
	vmulps(xmm1, xmm0, xmm0)
	vmulps(xmm2, xmm3, xmm3)
	vaddsubps(xmm3, xmm0, xmm0)

	vaddps(xmm8, xmm0, xmm0)

	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))

	add(rdi, rcx) // rcx = c + 2*rs_c

	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vpermilps(imm(0xb1), xmm0, xmm3)
	vmulps(xmm1, xmm0, xmm0)
	vmulps(xmm2, xmm3, xmm3)
	vaddsubps(xmm3, xmm0, xmm0)

	vaddps(xmm12, xmm0, xmm0)

	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORED)

  /*|--------|          |-------|
	|        |          |       |
	|   3x2  |          |  2x3  |
	|        |          |-------|
	|--------|
   */

	mov(var(cs_c), rsi)        // load cs_c
	lea(mem(, rsi, 8), rsi)    // rsi = cs_c * sizeof(dt)
	lea(mem(rsi, rsi, 2), r13)           // r13 = 3*rs_a

	CGEMM_INPUT_SCALE_CS_BETA_NZ_128
	vaddps(xmm4, xmm0, xmm4)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ_128
	vaddps(xmm8, xmm0, xmm8)
	add(rdi, rcx)

	CGEMM_INPUT_SCALE_CS_BETA_NZ_128
	vaddps(xmm12, xmm0, xmm12)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	vunpcklpd(xmm8, xmm4, xmm0) //a0a1b0b1 a4a4b4b5 //gamma00-10 gamma02-02
	vunpckhpd(xmm8, xmm4, xmm2) //a2a3b2b3 a6a7b6b7 //gamma01-11 gamma03-13

	vmovups(xmm0, mem(rcx))				// store (gamma00-10)
	vmovlpd(xmm12, mem(rcx, 16))	// store (gamma20)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma01-11)
	vmovhpd(xmm12, mem(rcx, 16))	// store (gamma21)

	jmp(.SDONE)                        // jump to end.

	label(.SBETAZERO)
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.SCOLSTORBZ)                    // jump to column storage case

	label(.SROWSTORBZ)

	vmovups(xmm4, mem(rcx))
	add(rdi, rcx)

	vmovups(xmm8, mem(rcx))
	add(rdi, rcx)

	vmovups(xmm12, mem(rcx))

	jmp(.SDONE)                        // jump to end.

	label(.SCOLSTORBZ)
	/****3x2 tile going to save into 2x3 tile in C*****/
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)           // rsi = cs_c * sizeof(dt)

	vunpcklpd(xmm8, xmm4, xmm0) //a0a1b0b1 a4a4b4b5 //gamma00-10 gamma02-02
	vunpckhpd(xmm8, xmm4, xmm2) //a2a3b2b3 a6a7b6b7 //gamma01-11 gamma03-13

	vmovups(xmm0, mem(rcx))				// store (gamma00-10)
	vmovlpd(xmm12, mem(rcx, 16))	// store (gamma20)
	lea(mem(rcx, rsi, 1), rcx)
	vmovups(xmm2, mem(rcx))				// store (gamma01-11)
	vmovhpd(xmm12, mem(rcx, 16))	// store (gamma21)

	label(.SDONE)

	lea(mem(r12, rdi, 2), r12)
	lea(mem(r12, rdi, 1), r12)         // c_ii = r12 += 3*rs_c

	lea(mem(r14, r8,  2), r14)
	lea(mem(r14, r8,  1), r14)         //a_ii = r14 += 3*rs_a

	dec(r11)                           // ii -= 1;
	jne(.SLOOP3X2I)                    // iterate again if ii != 0.

	label(.SRETURN)

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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 2;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		scomplex*  cij = c + i_edge*rs_c;
		scomplex*  ai  = a + i_edge*rs_a;
		scomplex*  bj  = b;

		cgemmsup_ker_ft ker_fps[3] =
		{
		  NULL,
		  bli_cgemmsup_rv_zen_asm_1x2,
		  bli_cgemmsup_rv_zen_asm_2x2,
		};

		cgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
}

